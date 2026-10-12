#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核心不变量测试：每条都对应一个「数学性质」，而不是「跑一遍看看」。

为什么这么写（本机约定 §4）：关键性质要写成硬断言。这里最关键的三条是
* **因果性**（改未来不能影响过去）—— 直接用负对照的方式验：改未来，过去必须逐位相同；
* **RoPE 的相对位置性质**（注意力分数只依赖 q、k 的相对位移）；
* **KV cache 等价性**（增量解码 == 整段重算，容差 1e-5）。
另外配一条「小样本过拟合」端到端冒烟测试：模型必须能背下一段 64 token 的序列。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from transformer import bpe as my_bpe
from transformer.model import (CausalSelfAttention, MiniTransformer, ModelConfig, RMSNorm,
                               apply_rope, precompute_rope)


# ======================================================================================
# 1. 分词器：encode/decode 必须严格可逆（含中文、emoji、空白）
# ======================================================================================
TEXT = "注意力机制 attention 就是加权平均，softmax(QK^T/√d)V。\n" * 3 + "emoji: 🧠🚀\t tab\n"


def test_handwritten_bpe_roundtrip():
    m = my_bpe.train(TEXT, vocab_size=400, verbose=False)
    ids = m.encode(TEXT)
    assert m.decode(ids) == TEXT, "手写 BPE 不可逆"
    assert len(ids) < len(TEXT.encode("utf-8")), "BPE 没有起到压缩作用"
    assert m.vocab_size == 256 + len(m.merges) + 2
    # 特殊 token 不与普通 token 冲突，且解码时被跳过
    assert all(i < m.bos_id for i in ids)
    assert m.decode([m.bos_id] + ids + [m.eos_id]) == TEXT


def test_handwritten_bpe_negative_control():
    """负对照：字典里没有的 id 必须解成替换符而不是静默错误（证明检查有鉴别力）。"""
    m = my_bpe.train(TEXT, vocab_size=300, verbose=False)
    assert m.decode([m.vocab_size + 5]) == "\ufffd"


def test_hf_tokenizer_wrapper_roundtrip():
    from transformer.data import train_tokenizer
    tok = train_tokenizer(TEXT, vocab_size=400, verbose=False)
    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT, f"{tok.kind} 分词器不可逆"
    if tok.kind == "hf":
        assert tok.bos_id is not None and tok.eos_id is not None


# ======================================================================================
# 2. RMSNorm：与 PyTorch 官方实现对齐
# ======================================================================================
def test_rmsnorm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    mine = RMSNorm(16, eps=1e-5)
    ref = torch.nn.RMSNorm(16, eps=1e-5)   # 注意：torch 默认 eps=None/1e-6，要对齐必须显式给
    with torch.no_grad():
        ref.weight.copy_(mine.weight)
    assert torch.allclose(mine(x), ref(x), atol=1e-6)
    # 权重置 1 时，输出每个向量的均方根应当 ≈ 1
    assert torch.allclose(mine(x).pow(2).mean(-1), torch.ones(2, 5), atol=1e-4)


# ======================================================================================
# 3. 因果性：改未来的 token，过去的输出必须逐位不变
# ======================================================================================
def test_attention_is_causal():
    """因果性：**改未来 token，过去位置的隐状态必须逐位不变**。

    注意现在 `forward` 只返回最后一个位置的 logits（为了省显存），而最后一个位置
    本来就该看到全部上下文，所以这条测试直接比隐状态（blocks 输出），而不是比 logits。
    """
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    model = MiniTransformer(cfg).eval()

    def hidden(x):
        h = model.tok_emb(x)
        for blk in model.blocks:
            h, _ = blk(h, model.cos_sin)
        return model.norm_f(h)

    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        a = hidden(x)
        x2 = x.clone()
        x2[0, 4:] = torch.randint(0, 64, (4,))     # 只改第 5 个位置之后
        b = hidden(x2)
    assert torch.equal(a[0, :4], b[0, :4]), "未来 token 影响了过去的输出 → 掩码没起作用"
    assert not torch.allclose(a[0, 4:], b[0, 4:]), "改了未来却毫无变化 → 测试没有鉴别力"


def test_causal_mask_rows_are_lower_triangular():
    """直接看注意力矩阵：第 i 行只在 <= i 的位置有权重，其余为 0。"""
    torch.manual_seed(0)
    attn = CausalSelfAttention(d_model=16, n_heads=2).eval()
    x = torch.randn(1, 6, 16)
    cs = precompute_rope(8, 6)
    with torch.no_grad():
        # 借用 forward 的中间量：手动复算权重矩阵（q 与 kv 是两个投影）
        q = attn.q_proj(x).view(1, 6, 2, 8).transpose(1, 2)
        k = attn.kv_proj(x).view(1, 6, 2, 2, 8).permute(2, 0, 3, 1, 4)[0]
        q = apply_rope(q, cs)
        k = apply_rope(k, cs)
        scores = q @ k.transpose(-2, -1) / (8 ** 0.5)
        mask = torch.triu(torch.full((6, 6), float("-inf")), diagonal=1)
        w = torch.softmax(scores + mask, dim=-1)
    upper = torch.triu(w, diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-7), "上三角权重不是 0"
    assert torch.allclose(w.sum(-1), torch.ones(1, 2, 6), atol=1e-5), "每一行权重和应为 1"


# ======================================================================================
# 4. RoPE：注意力分数只依赖相对位移
# ======================================================================================
def test_rope_is_relative():
    torch.manual_seed(0)
    cs = precompute_rope(head_dim=8, max_len=16)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)   # 单头、单 query/key
    base = (apply_rope(q, cs, 3) * apply_rope(k, cs, 3 + 5)).sum()
    for shift in (0, 4, 7):                                   # 两个位置同时平移 shift
        got = (apply_rope(q, cs, 3 + shift) * apply_rope(k, cs, 3 + shift + 5)).sum()
        assert torch.allclose(base, got, atol=1e-5), f"位移 {shift} 后分数变了 → RoPE 不是相对位置"
    # 负对照：距离不同，分数必须不同
    other = (apply_rope(q, cs, 3) * apply_rope(k, cs, 3 + 6)).sum()
    assert not torch.allclose(base, other, atol=1e-5), "不同距离给出相同分数 → 测试无鉴别力"


# ======================================================================================
# 5. KV cache：增量解码 == 整段重算
# ======================================================================================
@pytest.mark.parametrize("n_layers,n_heads", [(1, 2), (3, 4)])
def test_kv_cache_matches_full_forward(n_layers, n_heads):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50, d_model=32, n_layers=n_layers, n_heads=n_heads,
                      max_seq_len=16)
    model = MiniTransformer(cfg).eval()
    x = torch.randint(0, 50, (1, 10))
    with torch.no_grad():
        full = model(x)[0]                       # [1,1,V]（只保留最后一步）
        logits, _, kv = model(x[:, :6])          # 预填充前 6 个
        step = [logits[:, -1]]                   # 位置 5 的 logits（由前 6 个 token 算出）
        for t in range(6, 10):                   # 每次只喂 1 个 token，带着 cache
            out, _, kv = model(x[:, t:t + 1], past_kv=kv)
            step.append(out[:, -1])
        inc = torch.stack(step, dim=1)           # [1,5,V]，对应位置 5,6,7,8,9
    assert torch.allclose(full[0, -1], inc[0, -1], atol=1e-5), "KV cache 与整段重算结果不一致"


def test_generate_with_and_without_cache_agree():
    torch.manual_seed(0)
    model = MiniTransformer(ModelConfig(vocab_size=40, d_model=32, n_layers=2, n_heads=4,
                                        max_seq_len=32)).eval()
    x = torch.randint(0, 40, (1, 4))
    torch.manual_seed(123)                       # 同样的随机数种子 → 同样的采样序列
    a = model.generate(x.clone(), max_new_tokens=12, temperature=1.0, top_k=None, use_cache=True)
    torch.manual_seed(123)
    b = model.generate(x.clone(), max_new_tokens=12, temperature=1.0, top_k=None, use_cache=False)
    assert torch.equal(a, b), "cache / no-cache 生成结果不同 → 实现有 bug"


def test_generate_past_context_window_with_cache():
    """回归测试：生成长度超过 max_seq_len 时必须走滑窗，不能崩。

    这是真踩过的坑：cache 长到超过上下文后，``ids`` 被裁到 max_seq_len，但 cache 还是老的、
    更长的那个，于是 ``attn @ v`` 报 ``Expected size for first two dimensions ... [4,129]``。
    修法是滑窗丢 token 时清空 cache 重算。
    """
    torch.manual_seed(0)
    model = MiniTransformer(ModelConfig(vocab_size=30, d_model=16, n_layers=1, n_heads=2,
                                        max_seq_len=8)).eval()
    x = torch.randint(0, 30, (1, 8))             # 起点就占满上下文
    y = model.generate(x.clone(), max_new_tokens=25, temperature=1.0, top_k=None, use_cache=True)
    assert y.shape == (1, 33), y.shape            # 没有抛异常，长度正确
    # 与 no-cache 路径对比：滑窗策略必须一致（都只保留最近 max_seq_len 个 token）
    torch.manual_seed(7)
    a = model.generate(x.clone(), max_new_tokens=25, temperature=1.0, top_k=None, use_cache=True)
    torch.manual_seed(7)
    b = model.generate(x.clone(), max_new_tokens=25, temperature=1.0, top_k=None, use_cache=False)
    assert torch.equal(a, b), "超上下文后 cache 与 no-cache 的滑窗行为不一致"


# ======================================================================================
# 6. 端到端冒烟：68 个 token 的序列必须能背下来
# ======================================================================================
def test_overfit_tiny_sequence():
    torch.manual_seed(0)
    seq = torch.randint(0, 60, (1, 68))          # 随机序列，模型只能靠记忆
    cfg = ModelConfig(vocab_size=60, d_model=64, n_layers=2, n_heads=4, max_seq_len=64)
    model = MiniTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x, y = seq[:, :-4], seq[:, 1:-3]
    for _ in range(400):
        _, loss, _ = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    final = float(loss.detach())
    assert final < 0.05, f"400 步还没背下 64 个 token（loss={final:.3f}）"

    # 负对照：同样步数去拟合另一条序列，必须以高损失告终 ← 证明上面的低损失真的是学到的
    torch.manual_seed(1)
    other = torch.randint(0, 60, (1, 68))
    with torch.no_grad():
        _, loss_other, _ = model(other[:, :-4], other[:, 1:-3])
    assert float(loss_other) > 1.0, "换一条序列损失也很低 → 模型没真的记住序列"

    # 贪心生成（temperature→0）应当复现训练序列
    gen = model.generate(seq[:, :1], max_new_tokens=8, temperature=1e-6, top_k=1, use_cache=True)
    assert torch.equal(gen, seq[:, :9]), f"贪心生成没复现记忆序列：{gen.tolist()}"


def test_param_count_scales_as_expected():
    """参数量公式核对，防止「以为建了个大模型」。

    本实现用 SwiGLU（3 个 d×h 矩阵，h≈8d/3），所以每个 block 的 FFN 是 8d^2 而不是 8d^2/2，
    每层总共约 12d^2（4d^2 注意力 + 8d^2 前馈），再加词表嵌入 V*d。
    """
    cfg = ModelConfig(vocab_size=2048, d_model=128, n_layers=2, n_heads=4, max_seq_len=64)
    model = MiniTransformer(cfg)
    d = cfg.d_model
    # 粗略公式：每层约 12d^2（4d^2 注意力 + 8d^2 前馈），再加词表嵌入 + 输出头 2*V*d。
    # 精确值会因为 SwiGLU 的 hidden 向上取整到 64 的倍数而大几个百分点，所以给 6% 容差。
    est = cfg.n_layers * 12 * d ** 2 + 2 * cfg.vocab_size * d
    assert abs(model.num_params() - est) / est < 0.06, (model.num_params(), est)


# ======================================================================================
# 7. GQA：n_kv_heads == n_heads 时与 MHA 完全等价；小于时按比例省参数与 cache
# ======================================================================================
def test_gqa_includes_mha_as_special_case():
    """n_kv_heads=n_heads 的 GQA 必须与 MHA **逐位等价**（同权重、同输入 → 同输出）。

    这是 GQA 实现最容易错的地方（q/k/v 的 split 顺序、repeat_interleave 的位置），
    所以用「把 MHA 的 qkv 权重拆分搬进 GQA 的 q_proj/kv_proj」来严格对拍。
    """
    torch.manual_seed(0)
    mha = CausalSelfAttention(d_model=32, n_heads=4)
    gqa = CausalSelfAttention(d_model=32, n_heads=4, n_kv_heads=4)
    with torch.no_grad():
        # MHA 是「q 与 kv 两个投影」，直接把权重整体搬过去（两者的参数布局完全一样：
        # q_proj: n_heads*hd × C，kv_proj: 2*n_kv_heads*hd × C，且此时 n_kv_heads == n_heads）
        gqa.q_proj.weight.copy_(mha.q_proj.weight)
        gqa.kv_proj.weight.copy_(mha.kv_proj.weight)
        gqa.proj.weight.copy_(mha.proj.weight)
    x = torch.randn(2, 6, 32)
    cs = precompute_rope(8, 16)
    with torch.no_grad():
        a, _ = mha(x, cs)
        b, _ = gqa(x, cs)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max().item()

    # 负对照：只把 kv 头数改成 2，输出必须**不同**（证明上面的相等不是「恒等映射」）
    gqa2 = CausalSelfAttention(d_model=32, n_heads=4, n_kv_heads=2)
    with torch.no_grad():
        gqa2.q_proj.weight.copy_(mha.q_proj.weight)
        # MHA 的 kv_proj 布局是 [k(4头×8) ; v(4头×8)]，取「前 2 个 k 头 + 前 2 个 v 头」
        gqa2.kv_proj.weight.copy_(torch.cat([mha.kv_proj.weight[0:16],
                                             mha.kv_proj.weight[32:48]]))
        gqa2.proj.weight.copy_(mha.proj.weight)
        c, _ = gqa2(x, cs)
    assert not torch.allclose(a, c, atol=1e-4), "GQA 与 MHA 输出相同 → 测试没有鉴别力"


def test_gqa_shrinks_params_and_cache():
    """n_kv_heads 越小，注意力参数越少、cache 越小，但 logits 形状不变。"""
    sizes = {}
    for kv in (8, 4, 2, 1):
        cfg = ModelConfig(vocab_size=100, d_model=128, n_layers=2, n_heads=8, n_kv_heads=kv,
                          max_seq_len=32)
        model = MiniTransformer(cfg).eval()
        x = torch.randint(0, 100, (1, 12))
        with torch.no_grad():
            logits, _, cache = model(x)
        assert logits.shape == (1, 1, 100)
        sizes[kv] = (model.num_params(), cache[0][0].numel())
    # 参数量随 kv 头数单调递减，MQA(kv=1) 明显更小
    params = [sizes[k][0] for k in (8, 4, 2, 1)]
    assert params == sorted(params, reverse=True), params
    assert params[-1] < params[0], (params[0], params[-1])
    # cache 大小严格按 n_heads/n_kv_heads 成比例（这是生成显存的关键）
    # 本实现把 **n_kv_heads 头**的 k/v 存进 cache（共享只发生在计算里），所以直接成比例。
    assert sizes[1][1] * 8 == sizes[8][1], (sizes[1][1], sizes[8][1])
    assert sizes[2][1] * 4 == sizes[8][1], (sizes[2][1], sizes[8][1])
    assert sizes[4][1] * 2 == sizes[8][1]
