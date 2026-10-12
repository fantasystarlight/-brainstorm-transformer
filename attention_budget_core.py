#!/usr/bin/env python
"""attention_budget_core.py -- 注意力预算分配的核心数学（与模型/绘图解耦，可单测）。

术语
----
A            注意力矩阵，形状 [..., T, T]，最后一维沿 key 已归一化（行和为 1 或因果行和为 1）。
B            边预算：允许保留的 (query, key) 对数上界。B/T 即"每行平均可见 key 数"。
k_i          第 i 行实际分到的配额，Σ k_i ≤ B。

核心结论（本文件负责保证它是对的）
----------------------------------
**可达最优**：注意力掩码在每个 query 行上独立，所以预算 B 落到每行就是
"每行取自己的 top-k"，可达最优 = Σ_i (第 i 行最大的 k 个之和)。
见 row_topk_optimum。

**我在这个问题上连错三次，都记在这里，因为比结论本身更有用：**
  1. 声称"行自适应 water-filling == 全矩阵 top-B"。当时那个"带约束贪心"
     实现先把每行候选排序，等于把要验证的约束假设掉了 → 恒等式。
  2. 为了造一个"约束真的绑紧"的负对照，穷举里写了 `pre[i, combo[i]-1]`，
     combo[i]==0 时 Python 负索引取到该行**最后一个**元素，凭空多出贡献，
     于是"证明"了约束绑紧——假象。
  3. 把"配额前缀模型"当成正确的可达模型：要求第 i 行取了第 m 名就必须先取前 m−1 名。
     这个约束对应的是"只能注意最早的 key"，而真实注意力的大权重散布在任意位置
     （本例 sink 在 0，其余质量在距离 >256 的远端）。实测 64–97% 的行违反该前缀性
     → 这个模型系统性低估可达性，是在解另一个问题。

结论：**对"注意力作为有限资源"，正确的可达最优就是每行 top-k**，全矩阵 top-B 是它的
上界且通常只略高（差额来自跨行余额重分配）。
"""
from __future__ import annotations

import heapq

import numpy as np


# ---------------------------------------------------------------- 质量度量
def topk_mass_rows(A, k):
    """每行 top-k 捕获质量。A:[...,T,T]；k 为 int 或形状 [...]-兼容的数组。"""
    A = np.asarray(A)
    kk = np.full(A.shape[:-1], int(k)) if np.isscalar(k) else np.asarray(k)
    if np.any(kk < 0):
        raise ValueError("k 不能为负")
    if np.any(kk > A.shape[-1]):
        raise ValueError(f"k 超过行长 {A.shape[-1]}")
    part = np.partition(A, -1, axis=-1)
    order = np.argsort(part, axis=-1)[..., ::-1]
    sorted_A = np.take_along_axis(A, order, axis=-1)
    mask = np.arange(A.shape[-1]) < kk[..., None]
    return np.where(mask, sorted_A, 0.0).sum(-1)


def uniform_k(A, B):
    """把总预算 B 平均分给所有行；余数从前往后各 +1。"""
    n_rows = int(np.prod(A.shape[:-1]))
    if B < 0:
        raise ValueError("预算不能为负")
    base, rem = divmod(B, n_rows)
    if base > A.shape[-1]:
        raise ValueError("平均配额已超过行长，预算无意义")
    k = np.full(A.shape[:-1], base, dtype=np.int64)
    if rem:
        k.reshape(-1)[:rem] += 1
    return k


def global_topB(sorted_desc, B):
    """全矩阵 top-B 之和。**通常不可达**，只是上界（见 waterfill_optimum 的注释）。"""
    sorted_desc = np.asarray(sorted_desc)
    if sorted_desc.size and sorted_desc[0] < sorted_desc[-1]:
        raise ValueError("输入必须按降序排列")
    c = np.cumsum(sorted_desc.astype(np.float64))
    return float(c[min(B, c.size) - 1]) if B > 0 else 0.0


def row_topk_optimum(A, B):
    """**可达最优**（且这是真实 top-k 注意力能实现的）：每行独立取自己的 top-k。

    设每行行长 T、行数 R = L·H。边预算 B 对应每行 k ≈ B/R。
    可达最优 = Σ_i (第 i 行前 k 个最大的权重之和)。

    为什么这才是对的可达基准：注意力掩码在每个 query 行上是**独立**的，
    该行可以挑任意位置做 top-k，不需要"位置前缀"约束。任何要求
    "想取第 m 名就必须先取前 m−1 名"的模型都是把问题换成了另一个问题——
    真实注意力的大权重恰恰散布在任意 key 位置（本例中 sink 在位置 0，
    其余质量在距离 >256 的远端），前缀模型会系统性低估可达性。

    ⚠ 另一条我踩过的坑：**全矩阵 top-B（global_topB）比本函数更大**，
    因为它按值从全矩阵里挑 B 个最大的，允许"某行只贡献 3 个、但另一行贡献 40 个"，
    再把某个 query 的配额算成 40。这在**同一个头内**是可行的（行与行独立），
    所以两者在"每行 k 相同"时应相等；在有余额、可跨行重分配时才出现差额。
    实现上这里按精确的每行 top-k 计（行数不整除预算时前 rem 行多 1 个）。
    """
    A = np.asarray(A, dtype=np.float64)
    L, H, Tt, _ = A.shape
    rows = A.reshape(-1, Tt)
    n_rows = rows.shape[0]
    if B <= 0:
        return 0.0, {"k": 0, "n_rows": n_rows}
    k, rem = divmod(B, n_rows)
    if k >= Tt:
        return float(rows.sum()), {"k": Tt, "n_rows": n_rows, "capped": True}
    part = np.partition(rows, Tt - 1 - k, axis=1)[:, Tt - k:]   # 每行最大的 k 个
    total = float(part.sum())
    if rem:                       # 余额给前 rem 行各加一个名额
        extra = np.partition(rows[:rem], Tt - 2 - k, axis=1)[:, Tt - 1 - k]
        total += float(extra.sum())
    return total, {"k": k, "rem": rem, "n_rows": n_rows}


def quota_bruteforce(A, B):
    """前缀配额模型的最优（**仅用于展示这个模型与真实可达最优的差距**）。

    max Σ_i prefix_i(m_i) s.t. Σ m_i ≤ B。不是注意力能实现的量，见 row_topk_optimum。
    """
    A = np.asarray(A, dtype=np.float64)
    L, H, Tt, _ = A.shape
    rows = A.reshape(-1, Tt)
    pre = np.cumsum(rows, axis=1)
    m = min(Tt, B)
    dp = np.full(B + 1, -np.inf)
    dp[0] = 0.0
    for i in range(rows.shape[0]):
        cand = dp.copy()
        for kk in range(1, m + 1):
            np.maximum(cand[kk:], dp[: B + 1 - kk] + pre[i, kk - 1], out=cand[kk:])
        dp = cand
    return float(np.max(dp))


def gini_coef(x):
    """Gini 系数，float64。

    陷阱：因果 softmax 下每行质量恒为 1 → 真值 0，但 float32 排序噪声会给出
    -0.0026 这种负值。所以 float64 计算，并把 |g| < 1e-6 钳到 0。
    """
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    g = float((2.0 * (idx * x).sum()) / (n * x.sum()) - (n + 1.0) / n)
    return 0.0 if abs(g) < 1e-6 else g


def sink_mass(A):
    """位置 0 独占的质量占比——attention sink 的量化。"""
    A = np.asarray(A, dtype=np.float64)
    return float(A[..., 0].sum() / A.sum())


def per_head_sink_frac(A):
    """每个 [..., T, T] 头自己的 sink 占比，形状 A.shape[:-2]。"""
    A = np.asarray(A, dtype=np.float64)
    return A[..., 0].sum(-1) / np.clip(A.sum(axis=(-1, -2)), 1e-12, None)
