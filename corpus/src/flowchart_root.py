#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成头脑风暴文档的流程图（PNG + Mermaid + 纯文本，三者同源）。

设计约束（遵循本机约定）：
* 节点/边只定义一次，PNG 与 Mermaid 从同一份数据渲染，避免两套边不一致；
* 每个节点带 `anchor`，测试断言该字符串仍存在于 `RL新策略头脑风暴.md`（图过期 → 测试失败）；
* 中文优先使用微软雅黑（WSL2 可访问 Windows 字体树），缺字体时**降级为英文并告警**，
  绝不静默输出豆腐块；
* 输出 write-if-changed + 原子替换（先渲染到带正确扩展名的临时文件，再 os.replace），
  避免每次重跑刷新 mtime 导致侧栏已打开的图失效；
* 坐标范围从实际节点位置推导，不硬编码 xlim/ylim。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig-dsh"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "RL新策略头脑风暴.md")
OUTDIR = os.path.join(HERE, "docs")

CJK_FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/msyhbd.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]

# 颜色按层区分
LAYER_COLORS = ["#e8f0fb", "#fdf1e3", "#e9f7ef", "#f6e9f7", "#f2f2f2"]


# --------------------------------------------------------------------------------------
# 图数据（唯一定义来源）
# --------------------------------------------------------------------------------------
DIAGRAMS: dict[str, dict] = {
    "flow_proposals": {
        "title": "RL 改进策略地图：6 个缺口 → 11 个提案 → 首个判决动作",
        "direction": "LR",
        "layers": ["缺口", "提案", "首要判决动作"],
        "nodes": [
            # layer 0：缺口
            {"id": "G1", "layer": 0, "label": "G1 没有尺子", "en": "G1 no yardstick",
             "anchor": "G1 没有尺子"},
            {"id": "G2", "layer": 0, "label": "G2 无梯度被当垃圾", "en": "G2 zero-variance wasted",
             "anchor": "G2 无梯度被当垃圾"},
            {"id": "G3", "layer": 0, "label": "G3 静态超参 vs 可测状态", "en": "G3 static hyperparams",
             "anchor": "G3 静态超参 vs 可测状态"},
            {"id": "G4", "layer": 0, "label": "G4 hack 没有预算", "en": "G4 no hacking budget",
             "anchor": "G4 奖励是固定代理，hack 没有预算"},
            {"id": "G5", "layer": 0, "label": "G5 裁判激励不相容", "en": "G5 incompatible judge",
             "anchor": "G5 裁判既当运动员又当裁判"},
            {"id": "G6", "layer": 0, "label": "G6 覆盖率被牺牲", "en": "G6 coverage sacrificed",
             "anchor": "G6 单目标最大化 vs 覆盖率"},
            # layer 1：提案
            # 顺序＝图中的行序；按缺口顺序排列以减少连线交叉
            {"id": "P1", "layer": 1, "label": "P1 CreditBench 尺子", "en": "P1 CreditBench",
             "anchor": "P1　给信用分配造一把尺子"},
            {"id": "P2", "layer": 1, "label": "P2 零方差=后验证据", "en": "P2 zero-var as evidence",
             "anchor": "P2　把\"全对/全错\"当作后验证据"},
            {"id": "P3", "layer": 1, "label": "P3 DriftAudit", "en": "P3 DriftAudit",
             "anchor": "P3　`DriftAudit`"},
            {"id": "P4", "layer": 1, "label": "P4 奖励投资组合", "en": "P4 reward portfolio",
             "anchor": "P4　奖励组合的在线投资组合"},
            {"id": "P5", "layer": 1, "label": "P5 激励相容裁判", "en": "P5 proper scoring judge",
             "anchor": "P5　激励相容的裁判"},
            {"id": "P6", "layer": 1, "label": "P6 档案馆/MAP-Elites", "en": "P6 MAP-Elites",
             "anchor": "P6　把档案馆搬进 LLM RL"},
            {"id": "P7", "layer": 1, "label": "P7 可核查性预算", "en": "P7 checkability budget",
             "anchor": "P7　可核查性预算"},
            {"id": "P8", "layer": 1, "label": "P8 算力兑换率", "en": "P8 budget exchange",
             "anchor": "P8　训练 ↔ 推理的算力兑换率调度器"},
            {"id": "P9", "layer": 1, "label": "P9 梯度方差设计", "en": "P9 gradient variance DOE",
             "anchor": "P9　把 rollout 当对照实验来设计"},
            {"id": "P10", "layer": 1, "label": "P10 能力分区正交", "en": "P10 partitioned orthogonal",
             "anchor": "P10　RL 的持续学习"},
            {"id": "P11", "layer": 1, "label": "P11 世界模型反事实", "en": "P11 WM counterfactual",
             "anchor": "P11　用世界模型做反事实过程奖励"},
            # layer 2：首要判决动作
            {"id": "X7", "layer": 2, "label": "2×2：真 V vs 随机 V", "en": "2x2 real-V vs random-V",
             "anchor": "在真实 V 上加稠密奖励"},
            {"id": "X1", "layer": 2, "label": "oracle 对拍 + 负对照", "en": "oracle compare + control",
             "anchor": "打乱（负对照）"},
            {"id": "X3", "layer": 2, "label": "打乱 Δ 的负对照", "en": "shuffled-drift control",
             "anchor": "把 `Δ` 打乱（shuffle）"},
            {"id": "X2", "layer": 2, "label": "丢/遮/悲观 等算力", "en": "drop/mask/pessimism",
             "anchor": "丢弃 / CGE 式遮蔽 / 悲观下界"},
            {"id": "X8", "layer": 2, "label": "RL vs best-of-n vs 混合", "en": "RL vs BoN vs mixed",
             "anchor": "等总算力下比三种配置"},
            {"id": "X45", "layer": 2, "label": "金标发散曲线", "en": "gold-divergence curve",
             "anchor": "训练代理分、金标分、累计 hack 暴露上界"},
            {"id": "X6", "layer": 2, "label": "QD-score + coverage", "en": "QD-score + coverage",
             "anchor": "随机 cell 采样（去掉新颖度项）"},
            {"id": "X9", "layer": 2, "label": "测 tr(Cov(g))", "en": "measure tr(Cov(g))",
             "anchor": "直接测梯度协方差迹"},
            {"id": "X10", "layer": 2, "label": "数学→代码 连续 RL", "en": "math to code continual",
             "anchor": "小模型两阶段连续 RL"},
            {"id": "X11", "layer": 2, "label": "核销率扫描", "en": "settlement-rate sweep",
             "anchor": "核销比例 0% / 1% / 10%"},
        ],
        "edges": [
            ("G1", "P1"), ("G1", "P9"),
            ("G2", "P2"), ("G2", "P6"),
            ("G3", "P3"), ("G3", "P8"),
            ("G4", "P4"), ("G4", "P7"),
            ("G5", "P5"), ("G5", "P4"),
            ("G6", "P6"), ("G6", "P10"), ("G6", "P11"),
            ("P1", "X1"), ("P7", "X7"), ("P3", "X3"), ("P2", "X2"),
            ("P8", "X8"), ("P4", "X45"), ("P5", "X45"),
            ("P6", "X6"), ("P9", "X9"), ("P10", "X10"), ("P11", "X11"),
        ],
        "note": "排序依据：信息量 × 本地可行性（单卡 16GB）。P1/P7/P3/P2/P8 为首批。",
    },
    "flow_experiment": {
        "title": "判决实验决策树：从提案到「继续 / 转下一个」",
        "direction": "TB",
        "layers": ["选点", "前置检验", "执行", "判决", "行动"],
        "nodes": [
            {"id": "S", "layer": 0, "label": "从首批提案中选一个", "en": "pick one proposal",
             "anchor": "优先级：先做哪一个"},
            {"id": "Q1", "layer": 1, "label": "有可运行的负对照？", "en": "runnable negative control?",
             "anchor": "每个实验带负对照"},
            {"id": "FIX", "layer": 1, "label": "先补负对照再跑", "en": "add control first",
             "anchor": "负对照必须**有鉴别力**"},
            {"id": "Q2", "layer": 2, "label": "等算力 & ≥3 种子？", "en": "iso-compute & >=3 seeds?",
             "anchor": "**等算力**：报告 GPU-小时"},
            {"id": "RUN", "layer": 2, "label": "跑实验，记录判决判据", "en": "run and record criterion",
             "anchor": "**预注册**：先写下判据再跑"},
            {"id": "Q3", "layer": 3, "label": "A ≈ B > C ？", "en": "A ~ B > C ?",
             "anchor": "A ≈ B > C"},
            {"id": "Q4", "layer": 3, "label": "A > B ≈ C ？", "en": "A > B ~ C ?",
             "anchor": "A > B ≈ C"},
            {"id": "Q5", "layer": 3, "label": "A > C > B ？", "en": "A > C > B ?",
             "anchor": "A > C > B"},
            {"id": "V1", "layer": 4, "label": "证伪：收益来自密集性\n不是可核查性", "en": "falsified: density matters",
             "anchor": "收益来自**密集性**"},
            {"id": "V2", "layer": 4, "label": "成立：改写过程奖励\n设计原则", "en": "confirmed: rewrite principle",
             "anchor": "收益来自**可核查性**"},
            {"id": "V3", "layer": 4, "label": "文献里的启发式位置\n正在制造伤害", "en": "heuristics may harm",
             "anchor": "随机稠密奖励有害"},
            {"id": "V4", "layer": 4, "label": "无显著差异 →\n加种子/加算力再判", "en": "no effect: add seeds",
             "anchor": "噪声就是这么大"},
            {"id": "NEXT", "layer": 4, "label": "记录入库，转下一个提案", "en": "log and move on",
             "anchor": "判决记录（两种结果都入库）"},
        ],
        "edges": [
            ("S", "Q1"), ("Q1", "FIX", "否"), ("Q1", "Q2", "是"),
            ("FIX", "Q2"), ("Q2", "RUN", "是"),
            ("RUN", "Q3"), ("Q3", "V1", "是"), ("Q3", "Q4", "否"),
            ("Q4", "V2", "是"), ("Q4", "Q5", "否"), ("Q5", "V3", "是"),
            ("Q5", "V4", "否"),
            ("V1", "NEXT"), ("V2", "NEXT"), ("V3", "NEXT"), ("V4", "NEXT"),
        ],
        "note": "本树以 P7 的 2×2 判决实验为例；换成 P1/P3 时只替换 Q3–Q5 的分支标签。",
    },
}


# --------------------------------------------------------------------------------------
# 字体
# --------------------------------------------------------------------------------------
def setup_font() -> tuple[FontProperties, bool]:
    """返回 (字体, 是否支持中文)。找不到中文字体时返回英文模式。"""
    for path in CJK_FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                fontManager.addfont(path)
                name = FontProperties(fname=path).get_name()
                plt.rcParams["font.family"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return FontProperties(fname=path), True
            except Exception as exc:  # pragma: no cover
                print(f"[warn] 加载字体失败 {path}: {exc}", file=sys.stderr)
    print("[warn] 未找到中文字体，降级为英文标签", file=sys.stderr)
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    return FontProperties(), False


# --------------------------------------------------------------------------------------
# 布局
# --------------------------------------------------------------------------------------
def layout(nodes: list[dict], n_layers: int | None = None) -> dict[str, tuple[float, float]]:
    n_layers = (max(n["layer"] for n in nodes) + 1) if n_layers is None else n_layers
    pos: dict[str, tuple[float, float]] = {}
    for li in range(n_layers):
        layer = [n for n in nodes if n["layer"] == li]
        k = len(layer)
        for i, n in enumerate(layer):
            y = 1.0 - (i + 0.5) / k
            x = 0.5 if n_layers == 1 else 0.07 + 0.86 * li / (n_layers - 1)
            pos[n["id"]] = (x, y)
    return pos


def node_size(n: dict) -> tuple[float, float]:
    label = n.get("label", "")
    lines = label.split("\n")
    w = 0.030 + 0.0068 * max(len(ln) for ln in lines)
    h = 0.030 + 0.024 * len(lines)
    return min(w, 0.24), h


# --------------------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------------------
def render_png(spec: dict, out_png: str, cn: bool, font: FontProperties,
               positions: dict[str, tuple[float, float]] | None = None,
               n_layers: int | None = None) -> list[str]:
    """渲染 PNG，返回缺字形告警列表。

    ``positions`` / ``n_layers`` 是可选覆盖：调用方（如 transformer/flowchart.py）想留出
    顶部标题空间时，可以自己算好布局传进来。默认行为与以前完全一致。
    """
    nodes, edges = spec["nodes"], spec["edges"]
    n_layers = n_layers or max(len(spec["layers"]), max(n["layer"] for n in nodes) + 1)
    pos = layout(nodes, n_layers) if positions is None else positions
    fig, ax = plt.subplots(figsize=(15, 12) if len(nodes) > 12 else (13, 8))
    ax.axis("off")

    # 层标题
    layer_names = list(spec["layers"]) + [f"L{i}" for i in range(len(spec["layers"]), n_layers)]
    for li, name in enumerate(layer_names):
        x = 0.5 if n_layers == 1 else 0.07 + 0.86 * li / (n_layers - 1)
        ax.text(x, 1.02, name if cn else f"layer {li}",
                ha="center", va="bottom", fontsize=13, fontproperties=font, color="#333333")

    for n in nodes:
        x, y = pos[n["id"]]
        w, h = node_size(n)
        color = LAYER_COLORS[n["layer"] % len(LAYER_COLORS)]
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.006,rounding_size=0.012",
                                    linewidth=1.0, edgecolor="#5c6b7a", facecolor=color, zorder=2))
        ax.text(x, y, n["label"] if cn else n.get("en", n["label"]),
                ha="center", va="center", fontsize=8.5, fontproperties=font, zorder=3)

    for edge in edges:
        a, b = edge[0], edge[1]
        elabel = edge[2] if len(edge) > 2 else None
        (x1, y1), (x2, y2) = pos[a], pos[b]
        wa, _ = node_size(next(n for n in nodes if n["id"] == a))
        wb, _ = node_size(next(n for n in nodes if n["id"] == b))
        dx = x2 - x1
        if abs(dx) < 1e-9:                      # 同列：从上/下边出发
            dy = y2 - y1
            s = (x1, y1 + (0.5 if dy > 0 else -0.5) * node_size(next(n for n in nodes if n["id"] == a))[1])
            e = (x2, y2 - (0.5 if dy > 0 else -0.5) * node_size(next(n for n in nodes if n["id"] == b))[1])
        else:                                    # 跨列：从左右边出发
            s = (x1 + (wa / 2 if dx > 0 else -wa / 2), y1)
            e = (x2 - (wb / 2 if dx > 0 else -wb / 2), y2)
        ax.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>", mutation_scale=11,
                                     linewidth=0.9, color="#7b8b9a", zorder=1,
                                     shrinkA=0, shrinkB=0,
                                     connectionstyle="arc3,rad=0.06"))
        if elabel:
            ax.text((s[0] + e[0]) / 2, (s[1] + e[1]) / 2 + 0.012,
                    elabel if cn else elabel, ha="center", va="bottom", fontsize=8.5,
                    fontproperties=font, color="#a0522d", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    ax.set_title(spec["title"] if cn else spec.get("en_title", spec["title"]),
                 fontsize=15, fontproperties=font, pad=26)

    # 坐标范围由实际位置推导
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    half_w = max(node_size(n)[0] for n in nodes) / 2
    half_h = max(node_size(n)[1] for n in nodes) / 2
    ax.set_xlim(min(xs) - half_w - 0.03, max(xs) + half_w + 0.03)
    ax.set_ylim(min(ys) - half_h - 0.12, max(ys) + half_h + 0.09)
    ax.text(0.5, min(ys) - half_h - 0.10, spec["note"] if cn else spec.get("en_note", spec["note"]),
            ha="center", va="top", fontsize=10, fontproperties=font, color="#555555")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [str(w.message) for w in caught if "missing from font" in str(w.message)]


def render_mermaid(spec: dict) -> str:
    lines = [f"%% {spec['title']}", f"flowchart {spec.get('direction', 'LR')}"]
    layer_names = list(spec["layers"]) + [f"L{i}" for i in range(len(spec["layers"]),
                                                                    max(n["layer"] for n in spec["nodes"]) + 1)]
    for li, name in enumerate(layer_names):
        lines.append(f"  subgraph L{li}[{name}]")
        for n in spec["nodes"]:
            if n["layer"] == li:
                label = n["label"].replace("\n", "<br/>")
                lines.append(f'    {n["id"]}["{label}"]')
        lines.append("  end")
    for edge in spec["edges"]:
        if len(edge) > 2:
            lines.append(f'  {edge[0]} -->|{edge[2]}| {edge[1]}')
        else:
            lines.append(f"  {edge[0]} --> {edge[1]}")
    lines.append(f"  %% note: {spec['note']}")
    return "\n".join(lines) + "\n"


def render_txt(spec: dict) -> str:
    out = [spec["title"], "=" * len(spec["title"]), ""]
    layer_names = list(spec["layers"]) + [f"L{i}" for i in range(len(spec["layers"]),
                                                                    max(n["layer"] for n in spec["nodes"]) + 1)]
    for li, name in enumerate(layer_names):
        out.append(f"[{name}]")
        for n in spec["nodes"]:
            if n["layer"] == li:
                out.append(f"  - {n['id']}: {n['label'].replace(chr(10), ' / ')}")
        out.append("")
    out.append("[边]")
    for edge in spec["edges"]:
        arrow = f"  {edge[0]} -> {edge[1]}"
        if len(edge) > 2:
            arrow += f"   [{edge[2]}]"
        out.append(arrow)
    out.append("")
    out.append(f"[注] {spec['note']}")
    return "\n".join(out) + "\n"


def write_if_changed(path: str, data: bytes | str) -> bool:
    """内容有变化才写；原子替换。返回是否真的写了。"""
    if isinstance(data, str):
        data = data.encode("utf-8")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            if hashlib.sha256(fh.read()).digest() == hashlib.sha256(data).digest():
                return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1]
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp" + ext)
    os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return True


def build(outdir: str = OUTDIR, force_cn: bool | None = None) -> dict:
    font, cn = setup_font()
    if force_cn is not None:
        cn = force_cn
    os.makedirs(outdir, exist_ok=True)
    report = {}
    for name, spec in DIAGRAMS.items():
        spec.setdefault("en_title", spec["title"])
        spec.setdefault("en_layers", [f"layer {i}" for i in range(len(spec["layers"]))])
        png = os.path.join(outdir, f"{name}.png")
        # PNG 需要真实渲染，无法 write-if-changed 预判内容 → 渲染到临时文件再比对
        fd, tmp_png = tempfile.mkstemp(dir=outdir, suffix=".tmp.png")
        os.close(fd)
        try:
            warns = render_png(spec, tmp_png, cn, font)
            with open(tmp_png, "rb") as fh:
                new_png = fh.read()
        finally:
            if os.path.exists(tmp_png):
                os.remove(tmp_png)
        changed_png = write_if_changed(png, new_png)
        changed_mmd = write_if_changed(os.path.join(outdir, f"{name}.mmd"), render_mermaid(spec))
        changed_txt = write_if_changed(os.path.join(outdir, f"{name}.txt"), render_txt(spec))
        report[name] = {"png": png, "bytes": len(new_png), "changed": changed_png or changed_mmd or changed_txt,
                        "missing_glyphs": warns, "nodes": len(spec["nodes"]), "edges": len(spec["edges"])}
    return report


def check_anchors(doc: str = DOC) -> list[str]:
    """断言每个节点的锚点字符串仍存在于文档里；返回缺失列表。"""
    with open(doc, encoding="utf-8") as fh:
        text = fh.read()
    missing = []
    for name, spec in DIAGRAMS.items():
        for n in spec["nodes"]:
            if n["anchor"] not in text:
                missing.append(f"{name}:{n['id']} -> {n['anchor']!r}")
    return missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--check", action="store_true", help="只校验锚点，不渲染")
    args = ap.parse_args(argv)
    missing = check_anchors()
    if missing:
        print("锚点缺失：")
        for m in missing:
            print("  ", m)
        return 2
    if args.check:
        print("锚点全部存在")
        return 0
    rep = build(args.outdir)
    for name, info in rep.items():
        flag = "written" if info["changed"] else "unchanged"
        print(f"{name}: {info['nodes']} nodes, {info['edges']} edges, {info['bytes']} bytes, {flag}")
        if info["missing_glyphs"]:
            print(f"  [ERROR] 缺字形 {len(info['missing_glyphs'])} 条：{info['missing_glyphs'][:3]}")
            return 3
    print(f"输出目录：{args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
