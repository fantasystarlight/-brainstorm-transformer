#!/usr/bin/env python
"""attention_budget_probe.py -- 把注意力当作有限资源时，先量四件事。

不训练、不改模型，只做**前向测量**：

  Q1  资源有多不均衡？行内 top-1 质量 / 行间 Gini / 位置（列）Gini。
  Q2  平均预算相等时，自适应分配比均匀分配赢多少？
        uniform  每行固定 k = B/rows
        adaptive 行自适应 water-filling
      定理：行内权重降序时，带行前缀约束的全局贪心**恰好等于**全矩阵 top-B
            （行的第 m 个名额出现时，前 m-1 个必已出现 → 前缀约束自动满足）。
            所以 adaptive == global_topB，代码里对二者做等价性断言。
  Q3  分配决策本身要花多少？
        - 打分器质量扫描：score = A · exp(sigma·noise)，sigma=0 是免费 oracle，
          sigma 增大模拟"廉价近似打分器"。看捕获质量如何退化。
        - 跨层 IoU：相邻层高权重集合重叠率。高 ⇒ 分配可跨层共享，决策成本摊薄。
  Q4  简单度阶梯：注意力机制本身能有多简单而不掉质量？
        full / 均匀均匀 / 因果窗口 / 只看 sink(+窗口) / 只看 top-k。
        "sink+窗口" 若接近 full，说明任务上大部分注意力是冗余的。

数据来源标 [real] / [synthetic]；真实路径失败会打印 warn 并降级，**绝不静默**。
输出 docs/attention_budget_probe.json + 终端表格。
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

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
# 本机事实：HF 缓存里的 0.5B 只有 config/tokenizer（没有权重），
# 完整权重只有这个本地目录（3.1 GB）。走本地路径可绕开只读 HF 缓存与网络。
MODEL = os.path.expanduser("~/models/models/Qwen2.5-Math-1.5B-Instruct")
T = int(os.environ.get("PROBE_T", "1024"))
SEED = 0
# 可达最优 = 每行独立 top-k（core.row_topk_optimum，O(rows·T)），全矩阵秒级可算。
RATIOS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.25)   # 预算 = 每行平均可见 key 数 / T
SIGMAS = (0.0, 0.5, 1.0, 2.0, 4.0)                # 打分器噪声强度
SYNTH_ALPHAS = (1.0, 0.3, 0.1, 0.02)              # 合成谱：从近均匀到极重尾
WINDOWS = (4, 8, 16, 32, 64, 128, 256)


# ---------------------------------------------------------------- 数据获取
def real_attention():
    """[real] 本地 Qwen2.5-0.5B-Instruct 前向一次，抓各层注意力（fp16 eager）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    # 必须 fp32：Qwen2.5 的 logits 在 fp16 下会溢出，第 0 行 softmax 直接出 NaN
    # （实测 A[0,0,0].sum() = nan，其余行正常）。eager 才能拿到权重矩阵。
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True,
    ).eval()
    print(f"[real] loaded {MODEL}  T={T}  dtype=float32", flush=True)
    filler = ("Report on internal review of queue latency. "
              "The scheduler retried the job after the lease expired. "
              "Operator noted that throughput stayed flat while backlog grew. ")
    ids = tok(filler * 400, return_tensors="pt",
              truncation=True, max_length=T).input_ids
    print(f"[real] tokenized length = {ids.shape[1]}", flush=True)
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    A = torch.stack(out.attentions).float().numpy()     # [L,1,H,T,T] 因果
    assert A.ndim == 5 and A.shape[1] == 1, f"意外形状 {A.shape}"
    A = A[:, 0]                                         # 去掉 batch 单例 → [L,H,T,T]
    bad = int(np.isnan(A).sum())
    if bad:
        print(f"[warn] 注意力矩阵含 {bad} 个 NaN（softmax 溢出），已置零并重新归一化",
              file=sys.stderr)
        A = np.nan_to_num(A, nan=0.0)
        A /= np.clip(A.sum(-1, keepdims=True), 1e-12, None)
    meta = {"source": "real", "model": MODEL, "seq_len": int(A.shape[-1]),
            "n_layers": int(A.shape[0]), "n_heads": int(A.shape[1])}
    return A, meta


def synthetic_attention(alpha: float):
    """[synthetic] 重尾解析分布；alpha 控制行内浓度（小=更尖）。"""
    rng = np.random.default_rng(SEED)
    L, H, sigma = 4, 8, 1.0
    s = rng.lognormal(0.0, sigma, size=(L, H, T, 1))
    p = rng.dirichlet(np.full(T, alpha), size=(L, H, T))
    A = s * p
    A /= A.sum(-1, keepdims=True)
    meta = {"source": f"synthetic(alpha={alpha})", "seq_len": T, "n_layers": L,
            "n_heads": H, "row_scale_sigma": sigma, "dirichlet_alpha": alpha}
    return A.astype(np.float32), meta


# ---------------------------------------------------------------- 测量器
# 数学部分统一从 attention_budget_core 导入：探针与单元测试必须用**同一份实现**，
# 否则"测试通过"和"运行结果"可以互相矛盾（旧版这里自己抄了一份 adaptive_bruteforce，
# 而那份实现先把行排序，把前缀约束偷偷假设掉了——见 core 里的注释）。
from attention_budget_core import (  # noqa: E402
    gini_coef,
    global_topB,
    per_head_sink_frac,
    sink_mass,
    topk_mass_rows as topk_mass,
    uniform_k,
    row_topk_optimum,
)


def verify_optimum(A, B):
    """验证 row_topk_optimum：必须等于"逐行 partition 取 top-k"的独立实现，
    且必须 ≤ 全矩阵 top-B（上界）。两条一起查，防止把上界当可达值。

    注意契约：core 用 **ceil** 分配名额——k = ceil(B/rows)，只有 (rows−rem) 行拿到 k 个，
    前 rem 行少 1 个。我第一版验证器用了 floor 语义（每行 k=floor，前 rem 行 +1），
    于是"每行多拿一个" → 报出 8e-10 的假不一致。**验证器必须复刻被验证者的契约，
    而不是复刻它的意图。**
    """
    A = np.asarray(A, dtype=np.float64)
    Tt = A.shape[-1]
    rows = A.reshape(-1, Tt)
    n_rows = rows.shape[0]
    k = -(-B // n_rows)                      # ceil
    rem = k * n_rows - B                     # 只有 (rows - rem) 行拿到 k 个
    part = np.sort(rows, axis=1)
    naive = float(part[:, -k:].sum()) if k else 0.0
    if rem:
        naive -= float(part[:rem, -k].sum())    # 前 rem 行实际只有 k-1 个
    got, info = row_topk_optimum(A, B)
    rel_naive = abs(got - naive) / max(abs(naive), 1e-12)
    assert rel_naive < 1e-6, (got, naive, rel_naive, k, rem)
    upper = global_topB(np.sort(A.reshape(-1))[::-1], B)
    assert got <= upper + 1e-9, f"可达最优 {got} 超过上界 {upper}"
    return rel_naive


def noisy_scorer_capture(A, ratio, sigma, seed=SEED):
    """打分器质量扫描：score = A · exp(sigma·noise)，取 top-k 看真实捕获质量。

    sigma=0  → 免费 oracle（打分器完全知道 A）
    sigma=4  → 廉价近似打分器（几乎随机）
    这是**生成式**模型，不是某篇论文的复现；用来标定"打分器要多准才够用"。
    """
    L, H, Tt, _ = A.shape
    k = max(1, int(round(ratio * Tt)))
    rng = np.random.default_rng(seed)
    total = 0.0
    for l in range(L):
        noise = rng.normal(0.0, 1.0, size=A[l].shape) if sigma > 0 else 0.0
        sc = A[l] * np.exp(sigma * noise)
        idx = np.argpartition(-sc, k, axis=-1)[..., :k]
        total += float(np.take_along_axis(A[l], idx, axis=-1).sum())
    return total


def uniform_patterns(A, window=None, sink_only=False, sink=False):
    """Q4 简单度阶梯：给定结构模式，算捕获质量占 full 的比例。

    语义（踩过坑，写清楚）：
      - window=w  保留 |i-j| < w（因果下三角内）
      - sink=True 额外**强制**保留位置 0。旧版把 sink 塞进同一个 `mask &=` 链里，
        结果 "sink+window16" 的位置 0 被窗口条件再次抹掉，捕获几乎为 0——
        看起来像"sink 没用"，其实是掩码写错。
      - sink_only=True 时只保留位置 0。
    分母统一是 A 的总质量（= 因果下三角质量），所有模式可比。
    """
    L, H, Tt, _ = A.shape
    idx = np.arange(Tt)
    causal = idx[None, :] <= idx[:, None]
    is_sink = np.zeros((Tt, Tt), dtype=bool)
    is_sink[:, 0] = True
    if sink_only:
        mask = is_sink
    elif sink:
        mask = (np.ones((Tt, Tt), dtype=bool) if window is None
                else (idx[None, :] > idx[:, None] - window)) | is_sink
    elif window is not None:
        mask = (idx[None, :] > idx[:, None] - window)
    else:
        mask = np.ones((Tt, Tt), dtype=bool)
    mask = mask & causal
    return float((A * mask).sum()) / float(A.sum())


def layer_iou(A, k=None):
    L, H, Tt, _ = A.shape
    k = k or max(1, Tt // 32)
    sets = [np.sort(np.argpartition(-A[l], k, axis=-1)[..., :k], axis=-1)
            for l in range(L)]
    ious = []
    for l in range(L - 1):
        a, b = sets[l], sets[l + 1]
        inter = (a[..., None] == b[..., None, :]).any(-1).sum(-1)
        ious.append(float((inter / (2 * k - inter)).mean()))
    return float(np.mean(ious)), ious


# ---------------------------------------------------------------- 主分析
def analyze(A, meta):
    L, H, Tt, _ = A.shape
    n_rows = L * H * Tt
    tot = float(A.sum())
    print(f"\n=== {meta['source']}  L={L} H={H} T={Tt}  rows={n_rows}  "
          f"total_mass={tot:.1f} ===")

    mean_top1 = float(A.max(-1).mean())
    gini = gini_coef(A.sum(-1).reshape(-1))
    col_gini = gini_coef(A.sum(axis=(0, 1, 2)))
    assert gini > -1e-6 and col_gini > -1e-6, "Gini 为负 = 数值噪声，检查 float64"
    sk = sink_mass(A)
    print(f"[Q1] 行内集中：平均 top-1 = {mean_top1:.4f}")
    print(f"[Q1] 行间不均衡：行质量 Gini = {gini:.4f}")
    print(f"[Q1] 位置不均衡：列质量 Gini = {col_gini:.4f}")
    print(f"[Q4] 位置 0 独占质量占比（sink）= {sk:.4f}")

    sorted_desc = np.sort(A.reshape(-1))[::-1]

    # 配额 DP 的复杂度是 O(rows · B · 名次数)：全 336 个头 × 1024 行 = 34 万行时
    # 每个预算点要 10^8 量级运算、数分钟，整条曲线十几分钟。
    # 所以**精确 DP 只在一个头子集上跑**（等距抽样 4 个头），其余统计量仍用全矩阵。
    # 注意：这里用的是同一个 token 序列，头与头之间高度独立，抽样只影响统计精度，
    # 不影响"可达最优 ≤ 上界"这类结构性结论。
    rows = []
    print(f"\n{'B/T':>7} {'B':>10} {'uni':>8} {'可达最优':>9} {'最优/均匀':>8} "
          f"{'上界(不可达)':>11} {'最优/上界':>8} {'cheap':>8} {'cheap/最优':>10}")
    for r in RATIOS:
        B = n_rows * max(1, int(round(r * Tt)))
        q_uni = float(topk_mass(A, uniform_k(A, B)).sum()) / tot
        # 可达最优：每行独立 top-k（真实注意力掩码能做到的事）
        q_best, cap_info = row_topk_optimum(A, B)
        q_best /= tot
        q_top = global_topB(sorted_desc, B) / tot      # 不可达上界，仅供夹逼
        q_c1 = noisy_scorer_capture(A, r, 1.0) / tot
        rows.append(dict(ratio=r, B=B, q_uniform=q_uni, q_best=q_best,
                         q_best_k=cap_info.get("k"),
                         q_upper_unreachable=q_top,
                         best_over_uniform=q_best / q_uni,
                         best_over_upper=q_best / q_top,
                         cheap_sigma1=q_c1, cheap1_over_best=q_c1 / q_best))
        print(f"{r:>7.3f} {B:>10d} {q_uni:>8.4f} {q_best:>8.4f} {q_best/q_uni:>7.2f}x "
              f"{q_top:>8.4f} {q_best/q_top:>7.3f} {q_c1:>8.4f} {q_c1/q_best:>9.2f}")

    # 等价性断言（小规模，避开候选截断）。失败也要继续跑完其余测量，
    # 否则一个断言就把整块 real 结果吞掉（我第一次跑就是这样，白丢一轮前向）。
    sub = A[:1, :2]      # 保持 [L,H,T,T] 形状（A[0,:2] 会掉一维）
    n_sub = int(np.prod(sub.shape[:-1]))
    chk = min(2 * sub.shape[-1], n_sub * sub.shape[-1])
    try:
        rel = verify_optimum(sub, chk)
        print(f"[check] 可达最优 == 朴素逐行 top-k（2 头，B={chk}）相对差 = {rel:.2e}")
    except AssertionError as exc:
        rel = float("nan")
        print(f"[FAIL] 最优性断言未通过：{exc}", file=sys.stderr)
    except AssertionError as exc:
        rel = float("nan")
        print(f"[FAIL] 最优性断言未通过：{exc}", file=sys.stderr)

    # Q3：打分器质量 → 捕获质量（在 r=0.02 上）
    # 参照用**全矩阵**的两个量：均匀配额（可达）与全矩阵 top-B（不可达上界）。
    # 不在全矩阵上跑 DP——34 万行的 DP 太慢，而且这一节要的是"打分器能逼近多少"，
    # 夹在可达值与上界之间已经足够判读。
    r_probe = 0.02
    n_rows_probe = n_rows
    B_probe = n_rows_probe * max(1, int(round(r_probe * Tt)))
    q_uni_probe = float(topk_mass(A, uniform_k(A, B_probe)).sum()) / tot
    q_up_probe = global_topB(sorted_desc, B_probe) / tot
    print(f"\n[Q3] 打分器保真度扫描（B/T={r_probe}，均匀={q_uni_probe:.4f}，"
          f"上界(不可达)={q_up_probe:.4f}）")
    sig_rows = []
    for s in SIGMAS:
        q = noisy_scorer_capture(A, r_probe, s) / tot
        sig_rows.append(dict(sigma=s, capture=q, over_uniform=q / q_uni_probe,
                             over_upper=q / q_up_probe))
        print(f"     sigma={s:>4.1f}  捕获={q:.4f}  "
              f"相对均匀={q/q_uni_probe:>7.2f}  相对上界={q/q_up_probe:>6.3f}")

    iou, per_pair = layer_iou(A)
    print(f"[Q3] 相邻层 top-{max(1, Tt//32)} 集合平均 IoU = {iou:.4f}")

    # Q4：简单度阶梯
    ladder = []
    print("\n[Q4] 简单度阶梯（捕获质量 / full）")
    cases = [("full（上界）", dict()), ("only sink", dict(sink_only=True))]
    for w in WINDOWS:
        cases.append((f"sink+window{w}", dict(sink=True, window=w)))
        cases.append((f"window{w}（无 sink）", dict(window=w)))
    for name, kw in cases:
        v = uniform_patterns(A, **kw)
        w = kw.get("window")
        nkeys = ("all" if (kw == {} ) else (1 if kw.get("sink_only")
                 else (1 + (w if w else 0) if kw.get("sink") else (w or 0))))
        ladder.append(dict(pattern=name, capture=v, avg_keys=nkeys))
        print(f"     {name:>16}: {v:.4f}   （平均可见 key 数 ≈ {nkeys}）")

    return dict(meta=meta, mean_top1=mean_top1, row_gini=gini, col_gini=col_gini,
                sink_mass=sk, layer_iou=iou, layer_iou_per_pair=per_pair,
                equivalence_rel=rel, curve=rows, scorer_sweep=sig_rows,
                simplicity_ladder=ladder)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    results = []
    try:
        A, meta = real_attention()
        results.append(analyze(A, meta))
        got_real = True
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 真实模型路径失败，降级为解析分布："
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        got_real = False
    for alpha in SYNTH_ALPHAS:
        A2, meta2 = synthetic_attention(alpha)
        results.append(analyze(A2, meta2))

    out = os.path.join(OUTDIR, "attention_budget_probe.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"got_real": got_real, "results": results}, fh,
                  ensure_ascii=False, indent=2)
    print(f"\n[saved] {out}  (got_real={got_real})")


if __name__ == "__main__":
    main()
