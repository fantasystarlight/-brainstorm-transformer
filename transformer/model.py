#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小可训练的 Transformer（decoder-only，Llama 式），约 200 行有效代码。

刻意只保留「现代 LLM 真正在用的最小集合」，每个部件都能一句话说清：

    Embedding       token 查表 → [B,T,d]
    RoPE            把位置信息以「旋转」的形式注入 Q/K（相对位置，外推友好）
    Attention       每个位置看到历史，按相似度加权平均 value；支持 KV cache
    RMSNorm         只做缩放不居中，比 LayerNorm 少一半统计量
    SwiGLU 前馈       FFN 的现代版，用门控代替单个 ReLU
    Pre-Norm + 残差   x = x + Sub(x)，让 4 层以上也能训得动

数据形状约定（全文统一，看代码时对照这张表）：
    B = batch size, T = 序列长度, C = d_model, H = n_heads, hd = head_dim = C/H
    ids        [B, T]                输入 token id
    h          [B, T, C]             每个位置一个向量
    q/k/v      [B, H, T, hd]         拆头之后
    attn       [B, H, T, T]          每行 = 一个 query 对所有 key 的权重（行和为 1）
    logits     [B, T, V]             V = 词表大小
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# 1. 归一化：RMSNorm
# ======================================================================================
class RMSNorm(nn.Module):
    """把向量按「均方根」缩放到单位尺度，再乘一个可学习的缩放向量。

        x / sqrt(mean(x^2) + eps) * weight

    对比 LayerNorm：不减均值、不加偏置。少一些计算，效果在 LLM 上一样好（甚至更稳）。
    注意 ``mean`` 是在最后一维（特征维）上算的，与 batch/序列长度无关。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 初值全 1 → 初始时是恒等缩放

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()                                        # 用 fp32 算统计量更稳
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


# ======================================================================================
# 2. 位置编码：RoPE（旋转位置编码）
# ======================================================================================
def precompute_rope(head_dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    """预计算每个位置的旋转角，返回 [max_len, head_dim/2] 的 cos/sin 拼在一起。

    做法：把 head_dim 维两两配对成 head_dim/2 个二维平面，第 i 对用频率
    ``theta^(-2i/head_dim)`` 旋转，低频对转得慢（管长距离），高频对转得快（管近邻）。
    """
    assert head_dim % 2 == 0, "head_dim 必须是偶数才能两两配对"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # [hd/2]
    angles = torch.outer(torch.arange(max_len).float(), freqs)                 # [max_len, hd/2]
    return torch.stack([angles.cos(), angles.sin()], dim=0)                    # [2, max_len, hd/2]


def apply_rope(x: torch.Tensor, cos_sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """对 q 或 k 施加旋转。x: [B,H,T,hd]，cos_sin: [2, max_len, hd/2]。"""
    T = x.shape[-2]
    cos, sin = cos_sin[:, offset:offset + T]              # [2, T, hd/2]
    cos, sin = cos[None, None], sin[None, None]           # 广播到 [B,H,T,hd/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]                   # 偶数维、奇数维各一半
    # 二维旋转公式：(x1,x2) → (x1*cos - x2*sin, x1*sin + x2*cos)
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


# ======================================================================================
# 3. 因果自注意力（支持 KV cache）
# ======================================================================================
class CausalSelfAttention(nn.Module):
    """多头因果注意力，支持 **GQA**（分组查询注意力，Llama-2/3 的做法）。

    一次前向的四步：
      1) 线性投影：q 是 ``n_heads`` 份，k/v 只有 ``n_kv_heads`` 份
         （``n_kv_heads == n_heads`` 就是普通 MHA；``1`` 就是 MQA）
      2) 给 q/k 加 RoPE，让注意力天然带位置感
      3) softmax(qk^T / sqrt(hd)) @ v，并保证第 t 个位置只看 <= t 的 key
      4) 输出投影把各头信息混回来

    GQA 的意义：每 ``n_heads/n_kv_heads`` 个 query 头**共享**一组 k/v 头。
    k/v 的参数量和 KV cache 都按这个比例缩小，生成时显存和带宽直接省下来。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 n_kv_heads: int | None = None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads 必须是 n_kv_heads 的整数倍"
        assert self.n_kv_heads > 0
        self.n_rep = n_heads // self.n_kv_heads       # 每个 kv 头被几个 query 头共享
        # q 投影输出 n_heads*hd = C；k/v 只输出 n_kv_heads*hd（比 C 小，这就是省下来的部分）
        self.q_proj = nn.Linear(d_model, self.n_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)  # 输出投影：把各头信息混回来
        self.dropout = dropout

    def forward(self, x: torch.Tensor, cos_sin: torch.Tensor,
                past_kv: tuple[torch.Tensor, torch.Tensor] | None = None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # 上面一行 = split 成 k/v 再各 reshape 成 [B, n_kv_heads, T, hd]

        offset = 0 if past_kv is None else past_kv[0].shape[-2]  # 增量解码时的位置偏移
        q, k, v = apply_rope(q, cos_sin, offset), apply_rope(k, cos_sin, offset), v
        past = past_kv
        if past is not None:                                      # 把历史 k/v 接在前面
            k = torch.cat([past[0], k], dim=-2)
            v = torch.cat([past[1], v], dim=-2)
        # 返回给 cache 的 k/v 保持 **n_kv_heads 头**（这就是 GQA 省显存的地方：1/n_rep）。
        # 共享发生在计算里，而不是把 cache 撑大——这也是真实 GQA 实现的做法。
        to_cache = (k, v)
        # 真正参与打分/加权平均的 k/v 要复制成 n_heads 份（q 是 n_heads 份，头数必须对齐）
        if self.n_rep > 1:
            k, v = k.repeat_interleave(self.n_rep, dim=1), v.repeat_interleave(self.n_rep, dim=1)

        # ---- 因果掩码 ----
        Tq, Tk = q.shape[-2], k.shape[-2]
        # 增量解码（Tq == 1 且带 cache）时，那一个 query 本来就该看全部历史 → 不需要掩码。
        # 这很关键：否则要造一个 [1, Tk] 的全 -inf 掩码，白白多一次 HBM 读写。
        if Tq == 1 and past is not None:
            attn_mask = None
        else:
            # PyTorch 的布尔掩码语义：**True = 允许参与注意力**。上三角（未来）设为 False。
            allowed = torch.ones(Tq, Tk, device=x.device, dtype=torch.bool)
            allowed = torch.tril(allowed, diagonal=Tk - Tq)
            attn_mask = allowed.view(1, 1, Tq, Tk)
        # 用 F.scaled_dot_product_attention：它内部会挑最优 kernel（flash / mem-efficient），
        # 不会真的物化 [B,H,Tq,Tk] 的注意力矩阵——这是我第一次在 16 GB 卡上跑 512 上下文
        # OOM（“tried to allocate 6.12 GiB”）之后换过来的写法。
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, Tq, C)           # 各头拼回 [B,Tq,C]
        return self.proj(out), to_cache


# ======================================================================================
# 4. 前馈网络：SwiGLU
# ======================================================================================
class SwiGLU(nn.Module):
    """FFN：``down(silu(gate(x)) * up(x))``。

    为什么需要 FFN：注意力只做「位置之间的加权平均」（线性操作），
    真正在单个位置上做非线性变换、存事实知识的其实是 FFN，它占模型参数的大头。
    SwiGLU 用一条 gate 支路做门控，比两层 MLP+ReLU 效果好，所以现代 LLM 都用它。
    """

    def __init__(self, d_model: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or int(8 * d_model / 3)
        hidden = (hidden + 63) // 64 * 64          # 向上取整到 64 的倍数，对 GPU 友好
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ======================================================================================
# 5. Transformer Block：Pre-Norm + 残差
# ======================================================================================
class Block(nn.Module):
    """一个 block = 注意力 + 前馈，每路都是 ``x = x + Sub(Norm(x))``。

    残差是「能堆深」的关键：梯度可以沿着恒等路径直接回流；
    Norm 放在子层**之前**（Pre-Norm）比放在之后更容易训练，是现代默认写法。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 n_kv_heads: int | None = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, n_kv_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x, cos_sin, past_kv=None):
        a, kv = self.attn(self.norm1(x), cos_sin, past_kv)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x, kv


# ======================================================================================
# 6. 配置 + 完整模型
# ======================================================================================
@dataclass
class ModelConfig:
    vocab_size: int = 2306
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int | None = None   # None = MHA（每个 q 头一组 k/v）；设为 n_heads//4 就是 GQA
    max_seq_len: int = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_weights: bool = False       # True = lm_head 与 tok_emb 共享权重（小语料/大词表时很关键）


class MiniTransformer(nn.Module):
    """decoder-only 语言模型：预测「下一个 token」。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.n_kv_heads)
             for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        cos_sin = precompute_rope(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos_sin", cos_sin, persistent=False)  # 不写进 checkpoint

        if cfg.tie_weights:
            # 权重绑定：输出层直接复用输入嵌入矩阵。它省下 V*d 个参数（cl100k 词表下是 1.28 亿），
            # 也让「每个 token 的输出向量」和「它的输入向量」被同一份数据训练——小语料上通常更稳。
            # 代价：不能让 torch.compile 把 embedding 的梯度做特殊降维，需要
            # torch.compiler.disable（见 lm_head 前置的 wrapped 函数）。
            self.lm_head.weight = self.tok_emb.weight
            self._tied_forward = torch.compiler.disable(self._tied_lm_head)
        self.apply(self._init_weights)  # 手写初始化：训练初期不发散的关键
        self._init_weights(self.lm_head)

    def _tied_lm_head(self, x):
        """权重绑定时的输出层：等价于 F.linear(x, tok_emb.weight)。"""
        return F.linear(x, self.tok_emb.weight)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())   # 共享的权重只出现一次
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    # ------------------------------------------------------------------ 前向
    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None,
                past_kv: list | None = None, loss_chunk: int = 64):
        """ids: [B,T] int64。返回 (logits [B,1,V], loss 标量或 None, 新的 kv 列表)。

        **为什么要分块算 loss，而不是一次算完 [B,T,V]**：
        ``B*T*V`` 在 B=32、T=512、V=100279 时是 1.6G 个数 —— fp32 下 6.1 GiB，
        反向还要再存一份，16 GB 的卡会直接 OOM（我第一次跑 512 上下文就是这么炸的）。
        分块之后峰值只有 ``B*chunk*V``；而且**训练时只需要最后一个位置的 logits**（生成用），
        中间位置的 logits 根本没别处用，所以这里也只返回最后一步，省掉一个 [B,T,V] 的常驻张量。
        """
        B, T = ids.shape
        x = self.tok_emb(ids)                                 # [B,T,C]
        new_kv = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, self.cos_sin, None if past_kv is None else past_kv[i])
            new_kv.append(kv)
        x = self.norm_f(x)                                    # [B,T,C]

        loss = None
        if targets is not None:
            # 语言建模就是「每个位置都做一次 V 分类」：第 t 个位置猜第 t+1 个 token。
            # 这里按时间维分块：每块算完 CE 立刻丢掉 logits，梯度照样流回 lm_head 与主干。
            total, count = 0.0, 0
            head = (self._tied_forward if self.cfg.tie_weights else self.lm_head)
            for s in range(0, T, loss_chunk):
                e = min(s + loss_chunk, T)
                lg = head(x[:, s:e])                          # [B, chunk, V]
                n = int((targets[:, s:e] != -100).sum())
                ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), targets[:, s:e].reshape(-1),
                                     ignore_index=-100, reduction="sum")
                total, count = total + ce, count + n
            loss = total / max(count, 1)
        head = (self._tied_forward if self.cfg.tie_weights else self.lm_head)
        return head(x[:, -1:]), loss, new_kv                  # 只留最后一步的 logits

    # ------------------------------------------------------------------ 生成
    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int = 64, temperature: float = 0.9,
                 top_k: int | None = 40, use_cache: bool = True,
                 eos_id: int | None = None) -> torch.Tensor:
        """自回归采样。use_cache=True 时每步只喂 1 个新 token（推理速度差几十倍）。

        注意上下文滑窗：一旦总长超过 ``max_seq_len``，就得丢掉最老的 token。
        这时 cache 里的位置也要跟着清空（否则 cache 比可见上下文还长，张量对不上——
        我第一版就漏了这一步，生成长文本时崩在 ``attn @ v``）。
        """
        self.eval()
        cache, cache_len = None, 0                            # cache: 每层一个 (k, v)
        for _ in range(max_new_tokens):
            ctx = ids[:, -self.cfg.max_seq_len:]              # 可见上下文（滑窗）
            if use_cache and cache is not None and cache_len == ctx.shape[1] - 1:
                logits, _, cache = self(ctx[:, -1:], past_kv=cache)   # 只算最新那个 token
                cache_len += 1
            else:                                             # 第一次 / 滑窗丢弃 / 无 cache：整段重算
                logits, _, cache = self(ctx, past_kv=None)
                cache_len = ctx.shape[1]
            logits = logits[:, -1, :]

            logits = logits / max(temperature, 1e-6)
            if top_k:                                          # 只保留概率最高的 k 个
                kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)      # [B,1]
            ids = torch.cat([ids, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return ids
