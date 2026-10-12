#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""attention_budget_core.py 的硬断言测试（含"错误实现必须被抓住"的负对照）。

运行：python3 test_attention_budget_core.py   （pytest 同样可跑）

这个文件存在的理由，是它已经抓出过我三个错误：
  ① "带前缀约束的贪心"实现里先给候选排序，把要验证的约束偷偷假设掉了；
  ② 穷举参照里写了 `pre[i, combo[i]-1]`，combo[i]==0 时 Python 负索引取到末尾元素，
     凭空造出"约束绑紧"的假象；
  ③ 把一个**不可达**的上界（全矩阵 top-B）当成可达最优值写进了结论。
下面每条"负对照"都对应一次真实翻车，不是装饰。
"""
from __future__ import annotations

import json
import os

import numpy as np

import attention_budget_core as core

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")


def make_A(alpha=0.3, T=64, L=2, H=3, seed=0, row_scale_sigma=0.7):
    rng = np.random.default_rng(seed)
    s = rng.lognormal(0.0, row_scale_sigma, size=(L, H, T, 1))
    p = rng.dirichlet(np.full(T, alpha), size=(L, H, T))
    A = s * p
    return (A / A.sum(-1, keepdims=True)).astype(np.float64)


def row_greedy(A, B):
    """**错误的**实现（留作负对照）：逐行贪心，每次给"下一个边际增益最大"的行。

    这是"注意力作为有限资源"最自然的第一反应，但它不是最优：
    边际增益按行交错，贪心一旦把预算投给某行，就会被该行后续更差的名次拖住。
    """
    A = np.asarray(A, dtype=np.float64)
    L, H, T, _ = A.shape
    n_rows = L * H * T
    rows = A.reshape(n_rows, T)
    ptr = np.zeros(n_rows, dtype=np.int64)
    nxt = rows[:, 0].copy()
    total = 0.0
    for _ in range(min(B, n_rows * T)):
        i = int(np.argmax(nxt))
        total += nxt[i]
        ptr[i] += 1
        nxt[i] = rows[i, ptr[i]] if ptr[i] < T else -np.inf
    return total


# ---------------------------------------------------------------- 1
def test_row_topk_matches_naive_partition():
    """可达最优必须等于"逐行 partition 取 top-k"的朴素实现。"""
    A = make_A()
    n_rows = int(np.prod(A.shape[:-1]))
    T = A.shape[-1]
    rows = A.reshape(n_rows, T)
    for B in (1, 5, 37, n_rows * 3, n_rows * 10, n_rows * T):
        got, _ = core.row_topk_optimum(A, B)
        k, rem = divmod(B, n_rows)
        k = min(k, T)
        naive = float(np.sort(rows, axis=1)[:, -k:].sum()) if k else 0.0
        if rem and k < T:
            naive += float(np.sort(rows[:rem], axis=1)[:, -(k + 1)].sum())
        rel = abs(got - naive) / max(abs(naive), 1e-12)
        # 累加顺序不同 → float64 下也会有 ~1e-15 相对差；容差留 1e-9
        assert rel < 1e-9, f"B={B}: 实现 {got} vs 朴素 {naive}（相对差 {rel:.2e}）"


# ---------------------------------------------------------------- 2
def test_global_topB_is_strict_upper_bound():
    """全矩阵 top-B 必须 ≥ 可达最优，**且通常严格大于**——这正是我写错的地方。"""
    A = make_A(alpha=0.05, seed=5)          # 尖峰散布 → 空洞常见
    n_rows = int(np.prod(A.shape[:-1]))
    T = A.shape[-1]
    flat = np.sort(A.reshape(-1))[::-1]
    strictly_greater = 0
    for r in (0.02, 0.1, 0.3, 0.6):
        B = max(1, int(round(r * n_rows * T)))
        best, _ = core.row_topk_optimum(A, B)
        upper = core.global_topB(flat, B)
        assert best <= upper + 1e-9, f"可达最优 {best} 竟超过上界 {upper}"
        if best < upper - 1e-6 * upper:
            strictly_greater += 1
    assert strictly_greater >= 2, (
        f"上界与可达最优在 {4 - strictly_greater}/4 个预算点上相等 → "
        "构造无鉴别力，无法区分'可达'与'上界'")


# ---------------------------------------------------------------- 3
def test_row_greedy_is_detectably_suboptimal():
    """负对照：逐行贪心必须明显差于 DP 最优，否则本测试无鉴别力。"""
    A = make_A(alpha=0.05, T=64, L=1, H=4, seed=7)
    n_rows = int(np.prod(A.shape[:-1]))
    B = n_rows * 12
    best, _ = core.row_topk_optimum(A, B)
    greedy = row_greedy(A, B)
    assert best > 0
    assert greedy < 0.9 * best, (
        f"逐行贪心 {greedy:.4f} 竟然接近最优 {best:.4f} → 负对照失效或 DP 算错")


# ---------------------------------------------------------------- 4
def test_uniform_k_conserves_budget():
    A = make_A(T=32)
    n_rows = int(np.prod(A.shape[:-1]))
    for B in (0, 1, n_rows, n_rows * 5 + 1, n_rows * 7):
        k = core.uniform_k(A, B)
        assert k.shape == A.shape[:-1]
        assert int(k.sum()) == B, f"B={B} 分掉了 {int(k.sum())}"
        assert int(k.min()) >= 0 and int(k.max()) <= A.shape[-1]


def test_uniform_k_rejects_impossible_budget():
    A = make_A(T=8)
    n_rows = int(np.prod(A.shape[:-1]))
    for bad in (-1, n_rows * (A.shape[-1] + 1)):
        try:
            core.uniform_k(A, bad)
        except ValueError:
            continue
        raise AssertionError(f"预算 {bad} 应该被拒绝")


# ---------------------------------------------------------------- 5
def test_gini_is_zero_on_uniform_and_no_negative_noise():
    flat = np.ones(1000, dtype=np.float32)
    assert core.gini_coef(flat) == 0.0
    assert core.gini_coef(flat) >= 0.0          # float32 噪声不该产生负值
    x = np.zeros(1000)
    x[-1] = 1.0
    assert core.gini_coef(x) > 0.99
    assert core.gini_coef(np.zeros(5)) == 0.0


# ---------------------------------------------------------------- 6
def test_row_topk_monotone_in_budget():
    A = make_A()
    n_rows = int(np.prod(A.shape[:-1]))
    T = A.shape[-1]
    prev = -1.0
    for r in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
        B = max(1, int(round(r * n_rows * T)))
        v, _ = core.row_topk_optimum(A, B)
        assert v >= prev - 1e-9, (v, prev)
        prev = v


# ---------------------------------------------------------------- 7
def test_sink_and_per_head_frac():
    A = make_A(T=16)
    A = A.copy()
    A[..., 0] += 5.0
    A /= A.sum(-1, keepdims=True)
    assert core.sink_mass(A) > 0.5
    f = core.per_head_sink_frac(A)
    assert f.shape == A.shape[:-2]
    assert np.all(f > 0.5)


# ---------------------------------------------------------------- 8
def test_prefix_quota_model_underestimates():
    """负对照：前缀配额模型必须**严格低于**可达最优。

    这是我第三个错误的锚点：配额前缀模型（想取第 m 名必须先取前 m−1 名）
    等价于"只能注意最早的 key"，会系统性低估可达性。若这条退化成相等，
    说明数据失去了鉴别力，或者又把两个模型混在一起了。
    """
    A = make_A(alpha=0.05, T=32, L=1, H=1, seed=13)   # 行数少才跑得动 DP
    n_rows = int(np.prod(A.shape[:-1]))
    T = A.shape[-1]
    under = 0
    for r in (0.15, 0.3, 0.6):
        B = max(1, int(round(r * n_rows * T)))
        reach, _ = core.row_topk_optimum(A, B)
        prefix = core.quota_bruteforce(A, B)
        assert prefix <= reach + 1e-9, (
            f"前缀模型的 {prefix} 竟超过真实可达 {reach} → 模型方向反了")
        if prefix < reach - 1e-6 * max(reach, 1e-12):
            under += 1
    assert under >= 2, f"前缀模型在 {3 - under}/3 个预算点上没被抓住低估 → 无鉴别力"


# ---------------------------------------------------------------- 9
def test_probe_json_matches_core_math():
    """文档引用的数字必须能由 core 重算出来（防止 JSON 与代码漂移）。"""
    path = os.path.join(DOCS, "attention_budget_probe.json")
    if not os.path.exists(path):
        print(f"  [skip] {path} 不存在（先跑 attention_budget_probe.py）")
        return
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    real = next((r for r in blob["results"] if r["meta"]["source"] == "real"), None)
    if real is None:
        print("  [skip] JSON 里没有 real 结果")
        return
    for row in real["curve"]:
        assert 0.0 <= row["q_uniform"] <= row["q_best"] <= 1.0, row
        assert row["q_best"] <= row["q_upper_unreachable"] + 1e-9, row
    assert real["equivalence_rel"] < 1e-9, real["equivalence_rel"]


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = []
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failures.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - len(failures)}/{len(ALL_TESTS)} passed")
    raise SystemExit(1 if failures else 0)
