#!/usr/bin/env python
"""probe_token_compression.py -- token 压缩（把 m 个连续 token 压成 1 个）在真实注意力上的可行性实测。

三个问题，全部不需要训练、不需要改动模型：

  C1  每行的注意力"有效 key 数"是多少？
      spikiness（IPR = 1/Σp²）与熵。若每行只看极少数 key，稠密池化没有意义；
      若分散，池化才有机会。**这是决定后面一切的门槛量。**

  C2  池化 vs 剪枝，在同预算下谁更接近原注意力？
      - pooling : 块内 V 取平均（K 也平均），保留块总权重 → 每个 (query, block) 一个条目
      - pruning : 丢掉块内除最大者外的所有 token → 同样一个条目
      - sample  : 每块只取第一个 token（无信息的下界）
      指标：对 (query, block) 分布做总变差距离（TVD），以及"丢掉了多少注意力质量"。

  C3  池化损失的解析分解（Jensen 间隙）：
      error_pool = Σ_b Σ_{j∈b} a_ij · ||v_j − v̄_b||²
      把它与 C2 的 TVD 对照，验证"池化误差由块内方差控制"这个解释。

结论用途：判断"免训练把 token 压成摘要向量"是否可行，以及块大小该取多大。
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL = os.path.expanduser("~/models/models/Qwen2.5-Math-1.5B-Instruct")
T = int(os.environ.get("PROBE_T", "512"))
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
BLOCKS = (2, 4, 8, 16)          # 池化块大小
EPS = 1e-12


# ---------------------------------------------------------------- 数据
def get_attention_and_values(seq_len=T):
    """返回 (A, V)：A=[L,H,T,T] 注意力（因果），V=[L,H,T,dv] 每层的 value 向量。

    用 eager 注意力 + fp32：fp16 下该模型 logits 会溢出（第 0 行 softmax 出 NaN）。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True).eval()

    filler = ("Report on internal review of queue latency. "
              "The scheduler retried the job after the lease expired. "
              "Operator noted that throughput stayed flat while backlog grew. ")
    ids = tok(filler * 400, return_tensors="pt",
              truncation=True, max_length=seq_len).input_ids
    print(f"[real] tokens = {ids.shape[1]}", flush=True)

    # 抓每层的 value 向量：直接读 q_proj/k_proj/v_proj 的权重太麻烦，
    # 这里用 forward hook 拿 o_proj 的输入（即加权后的 value 输出），
    # 但我们要的是**未经混合**的 v_j，所以改为手动算 v = x @ W_v。
    v_list = []

    def hook_v(module, args, output):
        v_list.append(output.detach().float().cpu().numpy())

    handles = []
    for layer in model.model.layers:
        handles.append(layer.self_attn.v_proj.register_forward_hook(hook_v))

    with torch.no_grad():
        out = model(ids, output_attentions=True)
    for h in handles:
        h.remove()

    A = torch.stack(out.attentions).float().cpu().numpy()[:, 0]     # [L,Hq,T,T]
    V = np.stack(v_list)                                            # [L,T, Hkv*dv]
    L, Hq, Tt, _ = A.shape
    Hkv = model.config.num_key_value_heads
    dv = V.shape[-1] // Hkv
    # ⚠ GQA：这个模型 Hq=12 但 Hkv=2，v_proj 只输出 2*dv。
    # 第一版按 Hq 切 → 形状对不上还会把 head 映射错。正确做法是把每个 KV head
    # 广播给 Hq/Hkv 个 query head（Qwen 的重复方式是每个 KV head 连续服务 6 个 Q head）。
    Vn = V.reshape(L, Tt, Hkv, dv).transpose(0, 2, 1, 3)            # [L,Hkv,T,dv]
    rep = Hq // Hkv
    Vn = np.repeat(Vn, rep, axis=1)                                  # [L,Hq,T,dv]
    assert Vn.shape[1] == Hq, (Vn.shape, Hq, Hkv)
    print(f"[real] A={A.shape} V={Vn.shape} (Hq={Hq}, Hkv={Hkv}, dv={dv})",
          flush=True)
    return A, Vn


# ---------------------------------------------------------------- C1
def spikiness(A):
    """每行的有效 key 数：IPR = 1/Σp²（越小越尖）、熵（越小越尖）。"""
    p = np.clip(A, 0, None)
    s = p.sum(-1, keepdims=True)
    p = np.divide(p, s, out=np.zeros_like(p), where=s > EPS)
    ipr = 1.0 / np.clip((p ** 2).sum(-1), EPS, None)
    ent = -(p * np.log(np.clip(p, EPS, None))).sum(-1)
    return ipr, ent


# ---------------------------------------------------------------- C2
def pool_block_matrix(X, blk):
    """把 [.., T, d] 按连续 blk 个求平均 → [.., nb, d]（尾部不足则丢弃）。"""
    Tt = X.shape[-2]
    nb = Tt // blk
    Xc = X[..., : nb * blk, :]
    newshape = Xc.shape[:-2] + (nb, blk, Xc.shape[-1])
    return Xc.reshape(newshape).mean(-2)


def pool_attention_blocks(A, blk):
    """把 [L,H,T,T] 的 key 维按块求和 → [L,H,T,nb]（块内总权重，质量守恒）。

    行侧不压缩：每个 query 仍然一行，只是它看到的 key 变成了 nb 个"块"。
    这正是"把连续 token 压成摘要 token"在注意力上的投影。
    """
    Tt = A.shape[-1]
    nb = Tt // blk
    Ac = A[..., : nb * blk]
    return Ac.reshape(Ac.shape[:-1] + (nb, blk)).sum(-1)


def prune_attention_blocks(A, blk):
    """剪枝：每块只保留块内最大权重，其余置零。条目数同为 nb。"""
    Tt = A.shape[-1]
    nb = Tt // blk
    Ac = A[..., : nb * blk]
    Ab = Ac.reshape(Ac.shape[:-1] + (nb, blk))
    keep = Ab.max(-1, keepdims=True)
    return (Ab * (Ab >= keep - EPS)).sum(-1)


def sample_attention_blocks(A, blk):
    """每块只取块内第一个 token（无信息基线）。"""
    Tt = A.shape[-1]
    nb = Tt // blk
    Ac = A[..., : nb * blk]
    return Ac.reshape(Ac.shape[:-1] + (nb, blk))[..., 0]


def tvd_to_uniform(P, Q):
    """把两组 [.., nb] 的权重各自归一化后算 TVD（行的分布对比）。"""
    Pn = np.divide(P, np.clip(P.sum(-1, keepdims=True), EPS, None))
    Qn = np.divide(Q, np.clip(Q.sum(-1, keepdims=True), EPS, None))
    return 0.5 * np.abs(Pn - Qn).sum(-1)


# ---------------------------------------------------------------- C3
def jensen_gap(A, V, blk):
    """池化损失的解析量：Σ_b Σ_{j∈b} a_ij · ||v_j − v̄_b||²（逐 query 行）。

    A:[L,H,T,T]，V:[L,H,T,dv]。返回 [L,H,T]。
    """
    L, H, Tt, _ = A.shape
    nb = Tt // blk
    Vc = V[..., : nb * blk, :]
    dv = Vc.shape[-1]
    Vb = Vc.reshape(L, H, nb, blk, dv)
    Vbar = Vb.mean(3, keepdims=True)                       # [L,H,nb,1,dv]
    dev2 = ((Vb - Vbar) ** 2).sum(-1)                      # [L,H,nb,blk]
    Ac = A[..., : nb * blk].reshape(L, H, Tt, nb, blk)     # 按包分组
    return (Ac * dev2[:, :, None, :, :]).sum(-1).sum(-1)   # [L,H,T]


# ---------------------------------------------------------------- 主流程
def analyze(A, V, tag):
    L, H, Tt, _ = A.shape
    ipr, ent = spikiness(A)
    out = {
        "tag": tag, "shape": list(A.shape),
        "c1": {
            "ipr_mean": float(ipr.mean()),
            "ipr_median": float(np.median(ipr)),
            "ipr_p10": float(np.percentile(ipr, 10)),
            "ipr_p90": float(np.percentile(ipr, 90)),
            "entropy_mean": float(ent.mean()),
            "entropy_median": float(np.median(ent)),
        },
        "c2": [], "c3": [],
    }
    print(f"\n=== {tag}  A={A.shape} ===")
    print(f"[C1] 有效 key 数 IPR：均值 {ipr.mean():.2f}  中位 {np.median(ipr):.2f}  "
          f"p10 {np.percentile(ipr,10):.2f}  p90 {np.percentile(ipr,90):.2f}")
    print(f"[C1] 注意力熵：均值 {ent.mean():.2f} bits  中位 {np.median(ent):.2f} bits")

    print(f"\n{'blk':>4} {'压缩率':>7} {'pool_TVD':>9} {'prune_TVD':>10} "
          f"{'sample_TVD':>11} {'pool_质量':>10} {'prune_质量':>11} {'Jensen间隙':>11}")
    for blk in BLOCKS:
        if Tt // blk < 2:
            continue
        nb = Tt // blk
        # 参照分布 = 真实注意力按块求和（质量守恒，这就是"理想压缩"要逼近的东西）
        A_ref = pool_attention_blocks(A, blk)
        Ap = A_ref                              # 池化方案在 (query,block) 上即为它本身
        Ar = prune_attention_blocks(A, blk)
        As = sample_attention_blocks(A, blk)
        tv_pool = float(tvd_to_uniform(A_ref, Ap).mean())      # 恒为 0，见下方说明
        tv_prune = float(tvd_to_uniform(A_ref, Ar).mean())
        tv_samp = float(tvd_to_uniform(A_ref, As).mean())
        mass_pool = float((A_ref.sum(-1) / np.clip(A.sum(-1), EPS, None)).mean())
        mass_prune = float((Ar.sum(-1) / np.clip(A.sum(-1), EPS, None)).mean())
        gap = jensen_gap(A, V, blk)
        out["c2"].append(dict(blk=blk, ratio=1.0 / blk, tvd_pool=tv_pool,
                              tvd_prune=tv_prune, tvd_sample=tv_samp,
                              mass_pool=mass_pool, mass_prune=mass_prune))
        out["c3"].append(dict(blk=blk, jensen_gap_mean=float(gap.mean()),
                              jensen_gap_median=float(np.median(gap))))
        print(f"{blk:>4} {'1/'+str(blk):>7} {tv_pool:>9.4f} {tv_prune:>10.4f} "
              f"{tv_samp:>11.4f} {mass_pool:>10.4f} {mass_prune:>11.4f} "
              f"{gap.mean():>11.4f}")

    # 对照：同一行内"块内是否相似"——池化之所以可行的前提
    sim = []
    for blk in (4, 8):
        nb = Tt // blk
        p = A[..., : nb * blk].reshape(L, H, Tt, nb, blk)
        # 块内注意力权重与该块总和的比值：越接近均匀，块内越"不可区分"
        tot = p.sum(-1, keepdims=True)
        frac = np.divide(p, np.clip(tot, EPS, None))
        sim.append(dict(blk=blk, uniform_frac_std=float(frac.std()),
                        max_share_mean=float(frac.max(-1).mean())))
        print(f"[C3] blk={blk}: 块内权重占比标准差 {frac.std():.4f}，"
              f"最大占比均值 {frac.max(-1).mean():.4f}（越接近 1/blk 越均匀）")
    out["block_uniformity"] = sim
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    A, V = get_attention_and_values(T)
    res = analyze(A, V, "real")
    path = os.path.join(OUTDIR, "token_compression_probe.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
