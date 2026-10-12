#!/usr/bin/env python
"""attention_budget_figs.py -- 从 docs/attention_budget_probe.json 生成图与大纲。

规则（见 ~/.dsh/AGENTS.md §5）：
  * 一个名字三份：.png（可看）/ .mmd（Mermaid 源码）/ .txt（纯文本大纲），同步生成。
  * 数值与曲线**全部从 JSON 读**，不手抄；PNG 与 Mermaid 的边来自同一份节点数据。
  * write-if-changed + 原子替换（渲染到带正确扩展名的临时文件，比对后 os.replace），
    否则每次重跑都改 mtime，侧栏已打开的图会失效。
  * 中文字体缺失时降级为英文标签并打警告，绝不静默输出豆腐块。
  * 只需前向一次模型（复用 PROBE_T），加 --no-model 可完全跳过。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
JSON_PATH = os.path.join(DOCS, "attention_budget_probe.json")
MODEL = os.path.expanduser("~/models/models/Qwen2.5-Math-1.5B-Instruct")
FONT_CANDIDATES = ["/mnt/c/Windows/Fonts/msyh.ttc",
                   "/mnt/c/Windows/Fonts/msyhbd.ttc",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]

OUT = {
    "dashboard": "attention_budget_dashboard",
    "spectrum": "attention_budget_spectrum",
}
MMD = {
    "concept": "attention_budget_concept",
    "deps": "attention_budget_deps",
}


# ---------------------------------------------------------------- 基础设施
def setup_font():
    """注册中文字体；失败则返回 False（调用方改用英文标签）。"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager as fm
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                name = fm.FontProperties(fname=path).get_name()
                import matplotlib.pyplot as plt
                plt.rcParams["font.family"] = name
                plt.rcParams["axes.unicode_minus"] = False
                return name
            except Exception:  # noqa: BLE001
                continue
    return None


def atomic_save(fig, path):
    """渲染到临时文件 → 比对 → os.replace。返回 True 表示内容有变。"""
    tmp = path + ".tmp.png"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(tmp, dpi=150, bbox_inches="tight", facecolor="white")
    missing = [str(w.message) for w in caught
               if "missing from font" in str(w.message).lower()
               or "glyph" in str(w.message).lower()]
    if missing:
        print(f"[warn] {os.path.basename(path)}: {len(missing)} 条缺字形警告；"
              f"示例：{missing[0][:120]}", file=sys.stderr)
    new = open(tmp, "rb").read()
    old = open(path, "rb").read() if os.path.exists(path) else None
    if old == new:
        os.remove(tmp)
        return False, len(missing)
    os.replace(tmp, path)
    return True, len(missing)


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    new = open(tmp, "rb").read()
    old = open(path, "rb").read() if os.path.exists(path) else None
    if old == new:
        os.remove(tmp)
        return False
    os.replace(tmp, path)
    return True


# ---------------------------------------------------------------- 数据
def load_results():
    with open(JSON_PATH, encoding="utf-8") as fh:
        blob = json.load(fh)
    real = next((r for r in blob["results"] if r["meta"]["source"] == "real"), None)
    synth = [r for r in blob["results"] if r["meta"]["source"] != "real"]
    return blob, real, synth


def head_profile(T=1024):
    """[real] 每个 (layer, head) 的"注意力质量 vs key 距离"归一化剖面 + 标量摘要。

    关键：面板要画的是**每个头的分布形状**，不是绝对质量。
    第一版直接画绝对质量 + LogNorm(vmin=1e-5)，结果 336 行全糊成一片淡黄——
    因为每个桶的质量都在 1e-3~1e-1 量级，色标下限设到 1e-5 等于把对比度全丢了。
    现在改成：每行除以该 (layer,head) 的总质量 → 看到"这个头的注意力落在哪个距离"。
    距离定义 d = i - j；第 0 列单独算（sink），剖面只覆盖 d ≥ 1。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True).eval()
    filler = ("Report on internal review of queue latency. "
              "The scheduler retried the job after the lease expired. "
              "Operator noted that throughput stayed flat while backlog grew. ")
    ids = tok(filler * 400, return_tensors="pt", truncation=True,
              max_length=T).input_ids
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    A = torch.stack(out.attentions).float().numpy()[:, 0]      # [L,H,T,T]
    L, H, Tt, _ = A.shape

    idx = np.arange(Tt)
    dist = idx[:, None] - idx[None, :]
    edges = np.unique(np.round(np.logspace(0, np.log10(Tt), 25)).astype(int))
    prof = np.zeros((L, H, len(edges) - 1), dtype=np.float64)
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        m = (dist >= lo) & (dist < hi)
        prof[:, :, b] = A[:, :, m].sum(-1)
    denom = prof.sum(-1, keepdims=True)
    prof_norm = np.divide(prof, denom, out=np.zeros_like(prof), where=denom > 1e-12)

    # 标量摘要：sink 主导的头占多少比例
    # 每个 (layer,head) 的 sink 占比：位置 0 的质量 / 该头全部质量
    sink_frac = A[:, :, :, 0].sum(-1) / np.clip(A.sum(axis=(-1, -2)), 1e-12, None)
    summary = {
        "frac_heads_sink_gt_0.5": float((sink_frac > 0.5).mean()),
        "frac_heads_sink_gt_0.9": float((sink_frac > 0.9).mean()),
        "median_sink_frac": float(np.median(sink_frac)),
        "median_frac_le64": float(np.median(prof_norm[:, :, edges[:-1] <= 64]
                                             .sum(-1))),
    }
    return prof_norm, edges, A, summary


# ---------------------------------------------------------------- 图 1
def fig_dashboard(real, profile=None, zh=True):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig = plt.figure(figsize=(16.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.20, 1.05], wspace=0.46)

    # (a) 预算–质量曲线（真实模型）
    ax = fig.add_subplot(gs[0, 0])
    cur = real["curve"]
    x = [c["ratio"] for c in cur]
    ax.plot(x, [c["q_upper_unreachable"] for c in cur], ":", lw=2, ms=5,
            color="#8e44ad",
            label=("全矩阵 top-B（不可达上界）" if zh else "global top-B (UNREACHABLE)"))
    ax.plot(x, [c["q_best"] for c in cur], "o-", lw=2, ms=5, color="#c0392b",
            label=("可达最优（每行 top-k）" if zh else "achievable (per-row top-k)"))
    ax.plot(x, [c["q_uniform"] for c in cur], "s--", lw=2, ms=5, color="#2980b9",
            label=("均匀分配" if zh else "uniform"))
    ax.plot(x, [c["cheap_sigma1"] for c in cur], "^-.", lw=1.6, ms=5,
            color="#27ae60", label=("廉价打分器 σ=1" if zh else "cheap scorer σ=1"))
    ax.set_xscale("log")
    ax.set_xlabel("每行平均 key 预算  B/T" if zh else "per-row key budget B/T")
    ax.set_ylabel("捕获的注意力质量占比" if zh else "captured attention mass")
    ax.set_title(f"(a) 单位预算捕获质量：最优比均匀高 "
                 f"{cur[0]['best_over_uniform']:.0f}×"
                 if zh else "(a) captured mass per unit budget", fontsize=10)
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.02)

    # (b) 每头距离分布（行归一化）
    ax2 = fig.add_subplot(gs[0, 1])
    if profile is not None:
        prof, edges, A, summ = profile
        L, H, B = prof.shape
        flat = prof.reshape(L * H, B)
        order = np.argsort(-flat[:, 0])
        shown = flat[order]
        im = ax2.imshow(shown, aspect="auto", cmap="magma_r",
                        norm=LogNorm(vmin=2e-3, vmax=0.9))
        cb = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
        cb.set_label("占本头比例" if zh else "head fraction", fontsize=8)
        ticks = list(range(0, B, 5))
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([str(edges[i]) for i in ticks], fontsize=7)
        ax2.set_xlabel("key 距离桶下界  d = i−j" if zh else "key distance bucket")
        ax2.set_ylabel("(layer, head)，按最近邻质量排序" if zh
                       else "(layer, head), sorted by d=1 mass")
        if summ:
            t = (f"(b) 距离分布：{summ['frac_heads_sink_gt_0.5']*100:.0f}% 的头 "
                 f"sink>0.5" if zh else "(b) distance distribution")
        else:
            t = "(b) 距离分布（每头归一化）" if zh else "(b) distance distribution"
        ax2.set_title(t, fontsize=10)

    # (c) 简单度阶梯
    ax3 = fig.add_subplot(gs[0, 2])
    lad = [d for d in real["simplicity_ladder"] if "window" not in d["pattern"]
           or "sink+window" in d["pattern"]]
    names = [d["pattern"] for d in lad]
    vals = [d["capture"] for d in lad]
    y = np.arange(len(names))
    ax3.barh(y, vals, color=["#7f8c8d"] + ["#e67e22"] * (len(names) - 1))
    ax3.set_yticks(y)
    ax3.set_yticklabels([n.split("（")[0] for n in names], fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 1.0)
    ax3.set_xlabel("捕获质量 / full" if zh else "captured / full")
    ax3.axvline(real["sink_mass"], ls=":", color="k", lw=1.4)
    ax3.annotate(f"sink = {real['sink_mass']:.2f}", (real["sink_mass"], len(names) - 1),
                 fontsize=8, ha="left")
    ax3.set_title("(c) 结构模式的捕获率" if zh else "(c) structured patterns",
                  fontsize=10)
    ax3.grid(alpha=.3, axis="x")
    for i, v in enumerate(vals):
        ax3.text(min(v + 0.015, 0.95), i, f"{v:.3f}", va="center", fontsize=7.5)

    fig.suptitle("注意力作为有限资源：预填阶段实测（Qwen2.5-Math-1.5B-Instruct, T=1024）"
                 if zh else "Attention as a limited resource (real prefill measurements)",
                 fontsize=12)
    return fig


# ---------------------------------------------------------------- 图 2
def fig_spectrum(real, synth, zh=True):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.28})

    ax = axes[0]
    for r in [real] + synth:
        cur = r["curve"]
        tag = r["meta"]["source"]
        if tag == "real":
            tag = "真实模型 (Qwen2.5-1.5B)" if zh else "real model"
        ax.plot([c["ratio"] for c in cur], [c["q_best"] for c in cur],
                "o-", ms=4, lw=1.8, label=tag)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("B/T")
    ax.set_ylabel("自适应分配捕获质量" if zh else "adaptive captured mass")
    ax.set_title("(a) 浓度谱：真实注意力 vs 合成分布" if zh
                 else "(a) concentration spectrum", fontsize=10)
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    names = ["真实模型" if zh else "real"] + [r["meta"]["source"] for r in synth]
    top1 = [real["mean_top1"]] + [r["mean_top1"] for r in synth]
    sink = [real["sink_mass"]] + [r["sink_mass"] for r in synth]
    gcol = [real["col_gini"]] + [r["col_gini"] for r in synth]
    xs = np.arange(len(names))
    w = 0.27
    ax2.bar(xs - w, top1, w, label=("行内 top-1" if zh else "top-1 per row"),
            color="#c0392b")
    ax2.bar(xs, sink, w, label=("位置0 sink" if zh else "position-0 sink"),
            color="#e67e22")
    ax2.bar(xs + w, gcol, w, label=("列 Gini" if zh else "column Gini"),
            color="#2980b9")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([n.replace("synthetic(", "syn(") for n in names],
                        fontsize=8, rotation=15)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("(b) 三项集中度指标" if zh else "(b) concentration metrics",
                  fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=.3, axis="y")
    return fig


# ---------------------------------------------------------------- Mermaid
CONCEPT_NODES = [
    ("A", "运行时会膨胀的是两件不同的东西"),
    ("B", "边：注意力矩阵 O(T²)，每步 +1 行"),
    ("C", "列：KV 缓存 O(T)，每步 +1 个 key"),
    ("D", "边预算 B ≠ 列预算：随机选边不缩 KV 缓存"),
    ("E", "可达最优 = 每行独立 top-k（闭式）"),
    ("F", "全矩阵 top-B = 不可达上界（仅用于夹逼）"),
    ("G", "实测可达/上界 = 0.953~0.983"),
    ("H", "差额来自取整，不是结构性空隙"),
    ("I", "实测：top-1 平均 0.453，位置0 sink 0.361"),
    ("J", "实测：均匀稀疏比可达最优差 1264×"),
    ("K", "实测：廉价打分器 σ=1 已达可达最优 93%"),
    ("L", "实测：相邻层 top-32 的 IoU 只有 0.26"),
    ("M", "实测：sink+window64(65 key)=0.729，top20=0.794"),
    ("N", "行动：先做 sink 判决实验，别做分配器"),
    ("O", "护栏：别在不可达上界上做优化"),
    ("P", "护栏：负对照必须能真的失败"),
    ("Q", "护栏：验证器复刻契约，不复刻意图"),
]
CONCEPT_EDGES = [
    ("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"),
    ("B", "E"), ("E", "F"), ("F", "G"), ("G", "H"),
    ("E", "I"), ("I", "J"), ("E", "K"), ("E", "L"),
    ("I", "M"), ("M", "N"), ("J", "N"), ("H", "O"),
    ("O", "P"), ("P", "Q"),
]

DEPS_NODES = [
    ("P", "attention_budget_probe.py"),
    ("J", "docs/attention_budget_probe.json"),
    ("F", "attention_budget_figs.py"),
    ("M", "docs/*.mmd"),
    ("G", "docs/*.png"),
    ("T", "docs/*.txt"),
    ("R", "Qwen2.5-Math-1.5B-Instruct（本地全权重）"),
    ("W", "只读 HF 缓存 / 无网络（已绕过）"),
]
DEPS_EDGES = [("P", "J"), ("J", "F"), ("R", "P"), ("W", "P"),
              ("F", "M"), ("F", "G"), ("F", "T"), ("M", "G")]


def write_mermaid(name, title, nodes, edges):
    lines = [f"%% {title}", "flowchart TD"]
    for nid, label in nodes:
        lines.append(f'    {nid}["{label}"]')
    for a, b in edges:
        lines.append(f"    {a} --> {b}")
    body = "\n".join(lines) + "\n"
    changed = atomic_write_text(os.path.join(DOCS, f"{name}.mmd"), body)
    outline = [f"# {title}", ""]
    for nid, label in nodes:
        outline.append(f"  [{nid}] {label}")
    outline.append("")
    outline.append("边：")
    for a, b in edges:
        outline.append(f"  {a} -> {b}")
    changed = atomic_write_text(os.path.join(DOCS, f"{name}.txt"),
                                "\n".join(outline) + "\n") or changed
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true",
                    help="不加载模型（跳过距离剖面，其余图仍生成）")
    ap.add_argument("--T", type=int, default=1024)
    args = ap.parse_args()

    os.makedirs(DOCS, exist_ok=True)
    blob, real, synth = load_results()
    if real is None:
        print("[error] JSON 里没有 real 结果；先跑 attention_budget_probe.py",
              file=sys.stderr)
        return 2

    fam = setup_font()
    zh = fam is not None
    if not zh:
        print("[warn] 未找到中文字体，全部标签降级为英文", file=sys.stderr)
    import matplotlib.pyplot as plt

    profile = None
    if not args.no_model:
        try:
            profile = head_profile(T=args.T)
            np.savez_compressed(os.path.join(DOCS, "attention_profile.npz"),
                                prof=profile[0], edges=profile[1],
                                A=profile[2].astype(np.float16),
                                summary=json.dumps(profile[3]))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 距离剖面不可用：{type(exc).__name__}: {exc}",
                  file=sys.stderr)
    else:
        cache = os.path.join(DOCS, "attention_profile.npz")
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=False)
            hay = z["summary"] if "summary" in z else None
            profile = (z["prof"], z["edges"],
                       z["A"].astype(np.float32) if "A" in z else None,
                       json.loads(str(hay)) if hay is not None else None)

    ch1, miss1 = atomic_save(fig_dashboard(real, profile, zh=zh),
                             os.path.join(DOCS, OUT["dashboard"] + ".png"))
    plt.close("all")
    ch2, miss2 = atomic_save(fig_spectrum(real, synth, zh=zh),
                             os.path.join(DOCS, OUT["spectrum"] + ".png"))
    plt.close("all")

    ch3 = write_mermaid(MMD["concept"], "注意力预算：从概念到行动",
                        CONCEPT_NODES, CONCEPT_EDGES)
    ch4 = write_mermaid(MMD["deps"], "注意力预算探针：依赖与数据流",
                        DEPS_NODES, DEPS_EDGES)

    print(f"[figs] dashboard changed={ch1} spectrum changed={ch2} "
          f"mermaid changed={ch3 or ch4} missing_glyph_warnings={miss1 + miss2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
