#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1「信用分配尺子」的最小可运行实现（tabular，CPU，纯 numpy）。

背景：2026-08/09 的 RL 日报里，至少 12 篇论文声称改进了 token/step/turn 级信用分配
（HISPO 2609.15471、GRAFT 2609.28963、Cliff 2609.02817、GMTS 2608.30632、PACT 2609.26355 …），
但没有一篇在「真实信用已知」的环境里检验过自己的估计器。本脚本构造一个真实信用
可精确计算的小型 MDP，把几种主流信用估计器放在同一批轨迹上对拍。

环境：K 步决策，每步 A 个动作，其中恰好一个是正确动作；错误动作使 deficit += 1。
终局奖励 = 1 当且仅当 deficit <= TOL（允许犯 TOL 次错，因此早期错误可被后续完美表现弥补，
step-level 信用天然非均匀）。行为策略为均匀随机策略，故 V/Q/A 可用 DP 在 float64 下精确求解。

运行：python3 credit_oracle.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np

RNG_SEED = 20260928


# --------------------------------------------------------------------------------------
# 1. 环境与精确 oracle
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Env:
    K: int = 6        # 步数
    A: int = 3        # 每步动作数
    TOL: int = 1      # 允许的累计错误数

    @property
    def max_deficit(self) -> int:
        return self.TOL + 1  # 再多一个错误就必然失败

    def correct_action(self, k: int) -> int:
        """确定性的正确动作（与 k 有关，避免策略学到位置无关的捷径）。"""
        return k % self.A


def exact_values(env: Env, logits: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """返回 (V, Q)：V[k, d]，Q[k, d, a]，在**当前策略**下的精确值（float64）。

    策略由 logits 给出（形状 (K, D, A)），默认均匀随机策略。终端奖励只在 K 步后发放，
    故 Q_k(d,a) = V_{k+1}(d')，V_k(d) = Σ_a π(a|k,d) Q_k(d,a)。这是精确的策略评估。
    """
    K, A, TOL = env.K, env.A, env.TOL
    D = env.max_deficit + 1
    pi = np.full((K, D, A), 1.0 / A, dtype=np.float64) if logits is None else \
        np.stack([[softmax(logits[k, d]) for d in range(D)] for k in range(K)])
    V = np.zeros((K + 1, D), dtype=np.float64)
    Q = np.zeros((K, D, A), dtype=np.float64)
    for d in range(D):
        V[K, d] = 1.0 if d <= TOL else 0.0
    for k in range(K - 1, -1, -1):
        c = env.correct_action(k)
        for d in range(D):
            nd_ok = d
            nd_bad = min(d + 1, D - 1)
            q_ok = V[k + 1, nd_ok]
            q_bad = V[k + 1, nd_bad]
            Q[k, d, c] = q_ok
            for a in range(A):
                if a != c:
                    Q[k, d, a] = q_bad
            V[k, d] = float(np.dot(pi[k, d], Q[k, d]))
    return V, Q


def exact_advantage(env: Env, logits: np.ndarray | None = None) -> np.ndarray:
    """A[k, d, a] = Q - V，即本实验的 ground-truth 信用（随当前策略变化）。"""
    V, Q = exact_values(env, logits)
    return Q - V[: env.K, :, None]


def exact_values_buggy(env: Env) -> np.ndarray:
    """故意写错的 DP：忘记 deficit 转移（错误动作不改变状态）。

    仅用于测试「测试本身有鉴别力」——它必须与蒙特卡洛模拟显著不符。
    """
    K, A, TOL = env.K, env.A, env.TOL
    D = env.max_deficit + 1
    V = np.zeros((K + 1, D), dtype=np.float64)
    for d in range(D):
        V[K, d] = 1.0 if d <= TOL else 0.0
    for k in range(K - 1, -1, -1):
        for d in range(D):
            c = env.correct_action(k)
            q_ok = V[k + 1, d]
            q_bad = V[k + 1, d]          # ← 错误：应转移到 d+1
            V[k, d] = (q_ok + (A - 1) * q_bad) / A
    return V


# --------------------------------------------------------------------------------------
# 2. rollout
# --------------------------------------------------------------------------------------
@dataclass
class Rollout:
    states: list[tuple[int, int]]        # (k, deficit) 逐步
    actions: list[int]
    reward: float                        # 终局奖励（0/1）
    logits_snapshot: np.ndarray | None = None


def sample_group(env: Env, logits: np.ndarray, n: int, rng: np.random.Generator) -> list[Rollout]:
    """从 softmax(logits) 策略采样 n 条轨迹。logits 形状 (K, D, A)。"""
    out: list[Rollout] = []
    for _ in range(n):
        st, ac = [], []
        d, r = 0, 0.0
        for k in range(env.K):
            p = softmax(logits[k, d])
            a = int(rng.choice(env.A, p=p))
            st.append((k, d))
            ac.append(a)
            if a != env.correct_action(k):
                d = min(d + 1, env.max_deficit)
        r = 1.0 if d <= env.TOL else 0.0
        out.append(Rollout(st, ac, r))
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def group_normalized_returns(group: list[Rollout]) -> list[float]:
    r = np.array([g.reward for g in group], dtype=np.float64)
    if r.std() < 1e-12:
        return [0.0] * len(group)
    return list((r - r.mean()) / (r.std() + 1e-8))


# --------------------------------------------------------------------------------------
# 3. 被检验的信用估计器（每个返回 dict[(k,d,a)] -> credit）
# --------------------------------------------------------------------------------------
def est_true_advantage(env, group, logits, rng):
    A = exact_advantage(env, logits)
    return {(k, d, a): float(A[k, d, a]) for g in group for (k, d), a in zip(g.states, g.actions)}


def est_trajectory_broadcast(env, group, logits, rng):
    """GRPO / RLOO 式：轨迹级标量优势广播到每个 (state, action)。"""
    g = group_normalized_returns(group)
    return {(k, d, a): float(sc) for r, sc in zip(group, g) for (k, d), a in zip(r.states, r.actions)}


def est_mc_value_diff(env, group, logits, rng):
    """GRAFT / 轨迹图式：用采样得到的访问统计估计 V，再做 r + V(s') - V(s)。

    这里用同一策略下**另一批** rollout 的 MC 均值作 V 估计（模拟"图上 Bellman 迭代"的
    有限样本版本），以暴露估计噪声。
    """
    extra = sample_group(env, logits, max(len(group) * 8, 64), rng)
    acc: dict[tuple[int, int], list[float]] = {}
    for r in extra:
        # 从每个状态出发的折扣回报（此处无折扣，用剩余步的终局奖励）
        for (k, d) in r.states:
            acc.setdefault((k, d), []).append(r.reward)
    Vhat = {s: float(np.mean(v)) for s, v in acc.items()}
    out = {}
    for r in group:
        for i, ((k, d), a) in enumerate(zip(r.states, r.actions)):
            if k + 1 < env.K:
                nd = d if a == env.correct_action(k) else min(d + 1, env.max_deficit)
                sp = (k + 1, nd)
            else:
                sp = None
            v_next = 0.0 if sp is None else Vhat.get(sp, 0.0)
            v_cur = Vhat.get((k, d), 0.0)
            out[(k, d, a)] = v_next - v_cur
    return out


def est_first_mistake(env, group, logits, rng):
    """Cliff 2609.02817 式：首个错误之前给正、之后给负。"""
    out = {}
    for r in group:
        first = None
        for i, ((k, d), a) in enumerate(zip(r.states, r.actions)):
            if a != env.correct_action(k):
                first = i
                break
        for i, ((k, d), a) in enumerate(zip(r.states, r.actions)):
            if first is None:
                out[(k, d, a)] = 1.0
            else:
                out[(k, d, a)] = 0.5 if i < first else -1.0
    return out


def est_position_only(env, group, logits, rng):
    """只按位置（步序）给信用——对应"折扣/长度归一化"这类与状态无关的启发式。"""
    out = {}
    for r in group:
        for (k, d), a in zip(r.states, r.actions):
            out[(k, d, a)] = -float(k) / max(env.K - 1, 1)
    return out


def est_shuffled(env, group, logits, rng):
    """负对照：真实信用被随机置换 —— 必须与 oracle 零相关且学不到东西。"""
    keys = [(k, d, a) for g in group for (k, d), a in zip(g.states, g.actions)]
    vals = [est_true_advantage(env, [g], logits, rng)[(k, d, a)] for g in group
            for (k, d), a in zip(g.states, g.actions)]
    vals = list(rng.permutation(np.array(vals, dtype=np.float64)))
    return {key: float(v) for key, v in zip(keys, vals)}


ESTIMATORS = {
    "true_advantage(oracle)": est_true_advantage,
    "trajectory_broadcast": est_trajectory_broadcast,
    "mc_value_diff(GRAFT-like)": est_mc_value_diff,
    "first_mistake(Cliff-like)": est_first_mistake,
    "position_only": est_position_only,
    "shuffled(negative control)": est_shuffled,
}


# --------------------------------------------------------------------------------------
# 4. 指标
# --------------------------------------------------------------------------------------
def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry)))


def evaluate_credit(estimator, env: Env, *, n_group: int, n_prompts: int, rng) -> dict:
    """在初始策略上采集若干 prompt，比较估计信用与 oracle 信用的相关性与符号正确率。"""
    logits = np.zeros((env.K, env.max_deficit + 1, env.A), dtype=np.float64)
    true_all, est_all = [], []
    for _ in range(n_prompts):
        group = sample_group(env, logits, n_group, rng)
        est = estimator(env, group, logits, rng)
        oracle = est_true_advantage(env, group, logits, rng)
        for key, v in est.items():
            est_all.append(v)
            true_all.append(oracle[key])
    t = np.array(true_all, dtype=np.float64)
    e = np.array(est_all, dtype=np.float64)
    sign_acc = float(np.mean((np.sign(t) == np.sign(e)) | (np.abs(t) < 1e-12)))
    return {
        "spearman_vs_oracle": round(spearman(e, t), 4),
        "sign_accuracy": round(sign_acc, 4),
        "mean_abs_error": round(float(np.mean(np.abs(e - t))), 4),
        "n_pairs": int(t.size),
    }


def train_with_credit(estimator, env: Env, *, steps: int, batch: int, group_size: int,
                      lr: float, rng, normalize_credit: bool = True) -> dict:
    """把信用当优势做 REINFORCE 更新（tabular softmax 策略），看最终成功率。

    normalize_credit=True 时对每个 batch 的信用做零均值单位方差化——这样各估计器
    只在**方向**上竞争，不被自身量纲/尺度差异占便宜（即 F 桶指出的"固定有效 lr"要求）。
    """
    logits = np.zeros((env.K, env.max_deficit + 1, env.A), dtype=np.float64)
    for _ in range(steps):
        group = sample_group(env, logits, group_size, rng)
        est = estimator(env, group, logits, rng)
        if normalize_credit:
            vals = np.array(list(est.values()), dtype=np.float64)
            if vals.std() > 1e-12:
                mu, sd = vals.mean(), vals.std()
                est = {k: (v - mu) / sd for k, v in est.items()}
        grad = np.zeros_like(logits)
        for r in group:
            for (k, d), a in zip(r.states, r.actions):
                c = est[(k, d, a)]
                p = softmax(logits[k, d])
                g = -p
                g[a] += 1.0
                grad[k, d] += c * g
        logits += lr * grad / max(len(group), 1)
    succ = float(np.mean([r.reward for r in sample_group(env, logits, batch, rng)]))
    return {"success_rate": round(succ, 4)}


def _parse(argv=None):
    ap = argparse.ArgumentParser(description="信用分配估计器的 oracle 对拍（tabular）")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--A", type=int, default=3)
    ap.add_argument("--tol", type=int, default=1)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--prompts", type=int, default=200)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--tols", type=str, default="0,1,2", help="要扫描的 TOL（允许错误数）")
    ap.add_argument("--no-normalize", action="store_true", help="关闭信用归一化（暴露量纲混杂效应）")
    ap.add_argument("--train-seeds", type=int, default=3, help="训练重复次数（报告 mean±sd）")
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                            "credit_oracle_report.json"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    tols = [int(t) for t in args.tols.split(",") if t.strip() != ""]
    results: dict = {}
    for tol in tols:
        env = Env(K=args.K, A=args.A, TOL=tol)
        print(f"\nenv: K={env.K} A={env.A} TOL={env.TOL}  "
              f"(oracle credit = 当前策略下的精确 advantage, float64, normalize={not args.no_normalize})")
        print(f"{'estimator':30s} {'spearman':>9s} {'sign_acc':>9s} {'MAE':>8s} "
              f"{'train_succ':>16s}")
        res = {}
        for name, fn in ESTIMATORS.items():
            rng = np.random.default_rng(args.seed)
            m = evaluate_credit(fn, env, n_group=args.group, n_prompts=args.prompts, rng=rng)
            succs = []
            for rs in range(args.train_seeds):
                rng = np.random.default_rng(args.seed + 1 + rs)
                succs.append(train_with_credit(fn, env, steps=args.steps, batch=512,
                                               group_size=args.group, lr=args.lr, rng=rng,
                                               normalize_credit=not args.no_normalize)["success_rate"])
            arr = np.array(succs, dtype=np.float64)
            res[name] = {**m, "success_mean": round(float(arr.mean()), 4),
                         "success_sd": round(float(arr.std()), 4), "success_runs": succs}
            print(f"{name:30s} {m['spearman_vs_oracle']:9.4f} {m['sign_accuracy']:9.4f} "
                  f"{m['mean_abs_error']:8.4f} {arr.mean():9.4f}±{arr.std():.4f}")
        results[f"TOL={tol}"] = res

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"env": {"K": args.K, "A": args.A},
                   "seed": args.seed, "normalize_credit": not args.no_normalize,
                   "results": results}, fh, ensure_ascii=False, indent=2)
    print(f"\n报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
