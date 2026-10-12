#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""credit_oracle.py 的硬断言测试（含"错误输入必须失败"的负对照）。

运行：python3 -m pytest test_credit_oracle.py -q   （或直接 python3 test_credit_oracle.py）
"""
from __future__ import annotations

import numpy as np

import credit_oracle as co

ENV = co.Env(K=5, A=3, TOL=1)


# --------------------------------------------------------------------------------------
# 1. oracle 本身必须正确：精确 DP 与蒙特卡洛模拟一致
# --------------------------------------------------------------------------------------
def test_exact_value_matches_monte_carlo():
    rng = np.random.default_rng(0)
    n = 40000
    logits = np.zeros((ENV.K, ENV.max_deficit + 1, ENV.A))
    rolls = co.sample_group(ENV, logits, n, rng)
    mc = float(np.mean([r.reward for r in rolls]))
    V, _ = co.exact_values(ENV, logits)
    v0 = float(V[0, 0])
    se = math_sqrt(mc * (1 - mc) / n)
    assert abs(mc - v0) < 5 * se + 1e-6, f"DP={v0:.4f} vs MC={mc:.4f} (se={se:.4f})"


def math_sqrt(x):
    return float(np.sqrt(max(x, 0.0)))


def test_exact_values_are_float64():
    V, Q = co.exact_values(ENV)
    assert V.dtype == np.float64 and Q.dtype == np.float64


# --------------------------------------------------------------------------------------
# 2. 鉴别力：故意写错的 DP 必须与 MC 不符（否则测试没有区分能力）
# --------------------------------------------------------------------------------------
def test_buggy_dp_is_detected():
    rng = np.random.default_rng(1)
    n = 40000
    logits = np.zeros((ENV.K, ENV.max_deficit + 1, ENV.A))
    mc = float(np.mean([r.reward for r in co.sample_group(ENV, logits, n, rng)]))
    wrong = float(co.exact_values_buggy(ENV)[0, 0])
    V, _ = co.exact_values(ENV, logits)
    assert abs(wrong - mc) > 10 * math_sqrt(mc * (1 - mc) / n), \
        "错误 DP 竟然通过了检验 → 测试无鉴别力"
    assert abs(float(V[0, 0]) - mc) < 5 * math_sqrt(mc * (1 - mc) / n)


# --------------------------------------------------------------------------------------
# 3. oracle 估计器必须与真实信用完全一致
# --------------------------------------------------------------------------------------
def test_oracle_estimator_is_perfect():
    rng = np.random.default_rng(2)
    logits = np.zeros((ENV.K, ENV.max_deficit + 1, ENV.A))
    est = co.est_true_advantage(ENV, co.sample_group(ENV, logits, 8, rng), logits, rng)
    A = co.exact_advantage(ENV, logits)
    for (k, d, a), v in est.items():
        assert abs(v - float(A[k, d, a])) < 1e-12


# --------------------------------------------------------------------------------------
# 4. 负对照：打乱的信用必须与 oracle 零相关、且学不到东西
# --------------------------------------------------------------------------------------
def test_shuffled_credit_is_uninformative():
    rng = np.random.default_rng(3)
    m = co.evaluate_credit(co.est_shuffled, ENV, n_group=8, n_prompts=120, rng=rng)
    assert abs(m["spearman_vs_oracle"]) < 0.10, m
    rng = np.random.default_rng(4)
    tr = co.train_with_credit(co.est_shuffled, ENV, steps=250, batch=512, group_size=8,
                              lr=0.5, rng=rng)
    baseline = 0.10  # K=5,A=3,TOL=1 下均匀策略的成功率量级
    assert tr["success_rate"] < 3 * baseline, tr


# --------------------------------------------------------------------------------------
# 5. 现象守卫：TOL=0（不容错）时，广播式优势与真实信用的相关性显著低于"首个错误"启发式。
#    这是文档结论的锚点：若估计器定义被改动，这条断言必须先失败。
# --------------------------------------------------------------------------------------
def test_regime_dependence_is_preserved():
    hard = co.Env(K=6, A=3, TOL=0)
    m = {}
    for name in ("trajectory_broadcast", "first_mistake(Cliff-like)", "true_advantage(oracle)"):
        rng = np.random.default_rng(11)
        m[name] = co.evaluate_credit(co.ESTIMATORS[name], hard, n_group=8, n_prompts=200, rng=rng)
    assert m["true_advantage(oracle)"]["spearman_vs_oracle"] == 1.0, m
    assert m["first_mistake(Cliff-like)"]["sign_accuracy"] >= 0.99, m
    assert m["first_mistake(Cliff-like)"]["spearman_vs_oracle"] > \
        m["trajectory_broadcast"]["spearman_vs_oracle"] + 0.10, m
    assert m["trajectory_broadcast"]["sign_accuracy"] < 0.80, m


def test_sparse_regime_breaks_broadcast():
    """固定随机种子下的一个稀疏奖励区制复现：K=6、不容错、group=8 时，
    广播式信用几乎学不动（组内几乎永远没有奖励方差），而 oracle 与首个错误启发式能学会。"""
    hard = co.Env(K=6, A=3, TOL=0)
    out = {}
    for name in ("trajectory_broadcast", "first_mistake(Cliff-like)", "true_advantage(oracle)"):
        rng = np.random.default_rng(5)
        out[name] = co.train_with_credit(co.ESTIMATORS[name], hard, steps=300, batch=512,
                                         group_size=8, lr=0.5, rng=rng)["success_rate"]
    assert out["true_advantage(oracle)"] > 0.80, out
    assert out["first_mistake(Cliff-like)"] > 0.80, out
    assert out["trajectory_broadcast"] < 0.20, out


def test_normalization_only_changes_scale_not_direction():
    env = co.Env(K=4, A=3, TOL=1)
    rng1 = np.random.default_rng(7)
    a = co.train_with_credit(co.est_true_advantage, env, steps=50, batch=256, group_size=8,
                             lr=0.5, rng=rng1, normalize_credit=True)
    rng2 = np.random.default_rng(7)
    b = co.train_with_credit(co.est_true_advantage, env, steps=50, batch=256, group_size=8,
                             lr=0.5, rng=rng2, normalize_credit=False)
    assert a["success_rate"] > 0 and b["success_rate"] > 0


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
