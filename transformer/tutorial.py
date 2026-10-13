#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""教程配图：把「一个 Transformer 怎么工作」画成 5 张示意图 + 3 张实测曲线。

与 `transformer/flowchart.py` 的分工：
* `flowchart.py`  画**工程图**（流水线、依赖、cache 分支），也负责锚点校验；
* 本文件画**教程图**（面向初学者的直觉图 + 实测曲线），
  输出到同一个 `docs/`，每个名字三份：`.png`（可看）/ `.svg`（矢量，放大不糊）/ `.mmd`。

设计约束：
* 每个图配一条 `caption`，`check_anchors()` 会断言**这句话仍出现在 README 里**——
  删掉正文里的图注会让测试失败，而不是留下一张没人引用的图；
* 英文标签优先（跨机器不会缺字形）；有中文字体时自动切换中文；
* 所有写盘都走 `flowchart.write_if_changed`（内容没变就不动 mtime，
  否则侧栏已打开的图会失效）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "docs")
sys.path.insert(0, ROOT)
import flowchart as rootfc  # noqa: E402
if not hasattr(rootfc, "setup_font"):        # 模块同名撞车时的兜底（见 transformer/flowchart.py）
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("root_flowchart2", os.path.join(ROOT, "flowchart.py"))
    rootfc = _ilu.module_from_spec(_spec)
    sys.modules["root_flowchart2"] = rootfc
    _spec.loader.exec_module(rootfc)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

try:                                                     # 正常包导入
    from .flowchart import normalize_svg as _normalize_svg
except ImportError:                                      # 按文件路径加载时（测试/脚本）的兜底
    import importlib.util as _ilu2
    _s2 = _ilu2.spec_from_file_location("pkg_flowchart", os.path.join(HERE, "flowchart.py"))
    _m2 = _ilu2.module_from_spec(_s2)
    sys.modules["pkg_flowchart"] = _m2
    _s2.loader.exec_module(_m2)
    _normalize_svg = _m2.normalize_svg

RUNS = os.path.join(ROOT, "runs")

# 图注：**只作为锚点**（必须逐字出现在 README 里），不再画进图里。
# 为什么：曾经画在图里 → 在 PDF 里与 <figcaption> 重复成两条；
# 而 README 里图片下方本来就有斜体图注。图内不留文字，只有一个来源。
CAPTIONS = {
    "tut_hero": "图 1　一张图看懂：Transformer 只做一件事——看着前文猜下一个 token",
    "tut_forward": "图 2　一次前向的形状变化：从 [B,T] 到 [B,T,V]",
    "tut_attention": "图 3　注意力到底在算什么：先打分，再按分数取平均",
    "tut_sample": "图 4　怎么从 logits 变出一个字：温度与 top-k",
    "tut_loop": "图 5　训练一个 step 的六行代码",
    "tut_loss_curves": "图 6　实测训练曲线：val 先降后升——过拟合长什么样",
    "tut_scaling": "图 7　实测缩放：同一份数据下，增大模型并不带来改善",
    "tut_bench": "图 8　实测基准：GQA 与 KV cache 在生成时值多少",
}


# ======================================================================================
# 通用小工具
# ======================================================================================
FONT = None
CN = False


def _box(ax, x, y, w, h, lines, fc_="#e8f0fb", ec="#5c6b7a", fs=10.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.014",
                                linewidth=1.1, edgecolor=ec, facecolor=fc_, zorder=2))
    for i, ln in enumerate(lines):
        ax.text(x, y + (len(lines) - 1 - i) * 0.045, ln, ha="center", va="center",
                fontsize=fs, fontproperties=FONT, zorder=3, fontweight=weight)


def _arrow(ax, p1, p2, color="#7b8b9a", style="-|>", lw=1.3, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13, linewidth=lw,
                                 color=color, zorder=1, shrinkA=2, shrinkB=2,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def _label(ax, x, y, text, fs=9.5, color="#a0522d", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color,
            fontproperties=FONT, zorder=4,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))


def _canvas(w=12.5, h=8.6, xlim=(0, 1), ylim=(0, 1)):
    """建一张空白画布。

    **不再把图注画进图里**：图注由正文提供（README 的斜体行 / PDF 的 <figcaption>）。
    曾经画在图里，结果在 PDF 里和 figcaption 重复出现一整条（要么靠裁图绕开、
    要么读者看到两遍）。根因在图这边，所以在这里去掉。
    """
    fig, ax = plt.subplots(figsize=(w, h))
    ax.axis("off")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return fig, ax


def _save(fig, name: str, outdir: str) -> list[str]:
    """存 PNG + SVG（原子替换 + write-if-changed），返回缺字形告警。"""
    warns = []
    for ext in ("png", "svg"):
        fd, tmp = tempfile.mkstemp(dir=outdir, suffix=".tmp." + ext)
        os.close(fd)
        try:
            import warnings
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fig.savefig(tmp, dpi=160, bbox_inches="tight", facecolor="white")
            warns += [str(w.message) for w in caught if "missing from font" in str(w.message)]
            with open(tmp, "rb") as fh:
                data = fh.read()
            if ext == "svg":
                data = _normalize_svg(data)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        rootfc.write_if_changed(os.path.join(outdir, f"{name}.{ext}"), data)
    plt.close(fig)
    return warns


def _demo_paragraph() -> str:
    """README 里用来演示分词/前向的那段话（改它必须同步改 test 的负对照）。"""
    return "注意力机制就是把历史信息按相似度加权搬进当前位置。"


def _text(cn: str, en: str) -> str:
    return cn if CN else en


# ======================================================================================
# 图 1：一张图看懂
# ======================================================================================
def fig_hero(outdir: str) -> list[str]:
    fig, ax = _canvas(12.5, 9.0)
    stages = [
        ("tokens", ["一句话", "→ token ids", "[B, T]"], "#e8f0fb"),
        ("embedding", ["Embedding", "id → 向量", "[B, T, C]"], "#eef4fb"),
        ("N x Block", ["Block × L", "注意力 + 前馈", "[B, T, C]"], "#fdf1e3"),
        ("lm_head", ["lm_head", "向量 → 词表打分", "[B, T, V]"], "#e9f7ef"),
        ("next token", ["softmax + 采样", "选 1 个 id", "拼回末尾"], "#f6e9f7"),
    ]
    x0, dx, w, h = 0.10, 0.195, 0.170, 0.235
    y = 0.62
    for i, (_key, lines, color) in enumerate(stages):
        x = x0 + i * dx
        _box(ax, x, y, w, h, lines, fc_=color)
        if i:
            _arrow(ax, (x - dx + w / 2, y), (x - w / 2, y))
    # 回路：猜出来的字拼回输入，再猜下一个
    _arrow(ax, (x0 + 4 * dx, y - h / 2 - 0.02), (x0, y - h / 2 - 0.02), rad=0.16, color="#c0392b")
    _label(ax, x0 + 2 * dx, y - h / 2 - 0.115,
           _text("把猜出来的字拼回末尾，再猜下一个（自回归）",
                 "append the sampled id, then predict again (autoregressive)"), color="#c0392b")

    _box(ax, 0.5, 0.20, 0.86, 0.17,
         [_text("训练时：把「下一个字」当答案，算交叉熵，反向传播更新参数（不需要人工标注）",
                "Training: the next token IS the label; cross-entropy + backprop (no human labels)")],
         fc_="#f2f2f2", fs=11)
    _box(ax, 0.5, 0.045, 0.86, 0.10,
         [_text("全部代码都在 transformer/ 目录：约 2200 行，CPU 可跑，GPU 快 15 倍",
                "All code in transformer/: ~2200 lines, runs on CPU, ~15x faster on GPU")],
         fc_="#fbfbfb", ec="#c8c8c8", fs=10)
    return _save(fig, "tut_hero", outdir)


# ======================================================================================
# 图 2：一次前向的形状
# ======================================================================================
def fig_forward(outdir: str) -> list[str]:
    fig, ax = _canvas(13.0, 9.2)
    steps = [
        ("ids [B, T]", _text("一句话变成一串整数（分词）", "text -> token ids"), "#e8f0fb"),
        ("emb [B, T, C]", _text("查表：每个 id 换成一个向量", "lookup: each id -> vector"), "#eef4fb"),
        ("Block 1..L", _text("每层 = 注意力 + 前馈，各带残差", "layer = attn + FFN, residual"), "#fdf1e3"),
        ("  attention", _text("每个位置去看历史，加权搬运信息", "attend to the past, weighted copy"), "#fdf1e3"),
        ("  FFN (SwiGLU)", _text("逐位置的非线性变换（存知识）", "per-position MLP (knowledge)"), "#fdf1e3"),
        ("norm + lm_head", _text("投影到词表：给下一个字打分", "project to vocab: score next id"), "#e9f7ef"),
        ("logits [B, T, V]", _text("V = 词表大小（本教程 2048~4096）", "V = vocab size (2048~4096)"), "#e9f7ef"),
        ("loss（训练时）", _text("和「下一个字」比，算交叉熵", "cross-entropy vs next token"), "#f6e9f7"),
    ]
    y0, dy, h, w = 0.86, 0.092, 0.075, 0.30
    for i, (name, desc, color) in enumerate(steps):
        y = y0 - i * dy
        _box(ax, 0.21, y, w, h, [name], fc_=color, fs=11)
        ax.text(0.40, y, desc, ha="left", va="center", fontsize=10, fontproperties=FONT,
                color="#444444")
        if i:
            _arrow(ax, (0.21, y + dy - h / 2), (0.21, y + h / 2), lw=1.1)
    _label(ax, 0.21, y0 + h / 2 + 0.05,
           _text("B=batch  T=序列长  C=向量宽  V=词表", "B=batch T=seq len C=width V=vocab"))

    # 底部一整条放「5 种形状」——放右侧会压住描述文字（第一版就是这样，看图才发现）
    _box(ax, 0.5, 0.075, 0.94, 0.13,
         [_text("整个模型只用 5 种形状：[B, T] · [B, T, C] · [B, H, T, hd] · [B, H, T, T] · [B, T, V]",
                "Only 5 shapes: [B,T] . [B,T,C] . [B,H,T,hd] . [B,H,T,T] . [B,T,V]")],
         fc_="#fbfbfb", ec="#c8c8c8", fs=11)
    return _save(fig, "tut_forward", outdir)


# ======================================================================================
# 图 3：注意力
# ======================================================================================
def fig_attention(outdir: str) -> list[str]:
    """四格：打分 → 掩码 →（行和为 1 的）权重 → 加权平均搬走信息。

    加第四格的原因：图注承诺了「再按分数取平均」，前三格只到权重，没有「取平均」那一步。
    """
    import numpy as np
    rng = np.random.default_rng(0)
    T, hd = 7, 6
    q, k = rng.normal(size=(T, hd)), rng.normal(size=(T, hd))
    v = rng.normal(size=(T, hd))
    scores = q @ k.T / math.sqrt(hd)
    masked = np.where(np.tril(np.ones((T, T))) > 0, scores, np.nan)
    e = np.exp(masked - np.nanmax(masked, axis=1, keepdims=True))
    w = np.nan_to_num(e / np.nansum(e, axis=1, keepdims=True))
    out = w @ v

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.6))
    panels = [(scores, _text("① 打分 q·k/√hd", "1) scores q.k"), "viridis", "both"),
              (masked, _text("② 因果掩码：未来置空", "2) causal mask"), "viridis", "both"),
              (w, _text("③ softmax：每行和为 1", "3) softmax: rows sum to 1"), "viridis", "both"),
              (out, _text("④ 权重 @ v：搬信息", "4) weights @ v: copy"), "coolwarm", "out")]
    for ax, (m, t, cmap, _kind) in zip(axes, panels):
        im = ax.imshow(m, cmap=cmap, aspect="auto")
        ax.set_title(t, fontsize=11.5, fontproperties=FONT)
        ax.set_xlabel(_text("key 位置 / 特征维", "key pos / feature"), fontsize=9, fontproperties=FONT)
        ax.set_ylabel(_text("query 位置", "query position"), fontsize=9, fontproperties=FONT)
        ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[1].text(T / 2, T * 0.72, _text("未来 = 禁止\n(白色)", "future\nforbidden"),
                 ha="center", va="center", color="#333333", fontsize=10, fontproperties=FONT)
    axes[2].text(T / 2, T * 0.72, _text("每行加起来 = 1\n(上三角恰好为 0)", "each row sums to 1"),
                 ha="center", va="center", color="white", fontsize=9.5, fontproperties=FONT)

    fig.tight_layout()
    return _save(fig, "tut_attention", outdir)


# ======================================================================================
# 图 4：采样
# ======================================================================================
def fig_sample(outdir: str) -> list[str]:
    import numpy as np
    rng = np.random.default_rng(3)
    V = 40
    logits = rng.normal(size=V) * 2.0
    logits[[7, 12, 19, 26]] += np.array([8.0, 6.0, 5.0, 4.5])

    def probs(T_, k=None):
        z = logits / max(T_, 1e-6)
        if k:
            thr = np.sort(z)[-k]
            z = np.where(z < thr, -np.inf, z)
        e = np.exp(z - np.nanmax(z))
        return e / e.sum()

    variants = [("temperature=0.5", probs(0.5), "#4c6ef5"),
                ("temperature=1.0", probs(1.0), "#2f9e44"),
                ("top-k=3, T=1.0", probs(1.0, 3), "#e8590c"),
                ("temperature=2.0", probs(2.0), "#c2255c")]
    x = np.arange(V)
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    for label, p, color in variants:
        ax.plot(x, p, marker="o", ms=3.2, lw=1.6, label=label, color=color)
    ax.set_yscale("log")
    ax.set_xlabel(_text("词表里的 token（按 logits 排序后的位置）", "token index in vocab"), fontsize=10,
                  fontproperties=FONT)
    ax.set_ylabel(_text("被抽中的概率（对数轴）", "sampling probability (log)"), fontsize=10,
                  fontproperties=FONT)

    ax.legend(fontsize=10, prop=FONT)
    ax.grid(alpha=0.3)
    ax.annotate(_text("温度低 → 几乎只选最高分\n（greedy，稳但无聊）",
                      "low T -> nearly greedy\n(stable but dull)"),
                xy=(7, probs(0.5)[7]), xytext=(13, 0.02), fontsize=9.5, fontproperties=FONT,
                arrowprops=dict(arrowstyle="->", color="#4c6ef5"))
    ax.annotate(_text("top-k 把长尾直接掐掉", "top-k removes the long tail"),
                xy=(28, probs(1.0, 3)[28]), xytext=(24, 3e-4), fontsize=9.5, fontproperties=FONT,
                arrowprops=dict(arrowstyle="->", color="#e8590c"))
    fig.tight_layout()
    return _save(fig, "tut_sample", outdir)


# ======================================================================================
# 图 5：训练循环
# ======================================================================================
def fig_loop(outdir: str) -> list[str]:
    fig, ax = _canvas(13.0, 9.2)
    steps = [
        (_text("① 采样一批", "1) sample a batch"), "x, y = stream.batch(B)",
         _text("y 就是 x 右移一位", "y is x shifted by one"), "#e8f0fb"),
        (_text("② 前向 + 交叉熵", "2) forward + loss"), "logits, loss, _ = model(x, y)",
         _text("每步都算 loss/pPL", "loss & PPL every step"), "#eef4fb"),
        (_text("③ 反向传播", "3) backward"), "loss.backward()",
         _text("自动微分算出所有梯度", "autograd fills .grad"), "#fdf1e3"),
        (_text("④ 梯度裁剪", "4) clip"), "clip_grad_norm_(params, 1.0)",
         _text("防止偶发大梯度打飞权重", "stops rare huge updates"), "#fdf1e3"),
        (_text("⑤ 更新参数", "5) step"), "optimizer.step()",
         _text("AdamW + warmup/余弦", "AdamW + warmup/cosine"), "#e9f7ef"),
        (_text("⑥ 清空梯度", "6) zero_grad"), "optimizer.zero_grad()",
         _text("不清会累加！", "PyTorch accumulates!"), "#f6e9f7"),
    ]
    for i, (name, code, note, color) in enumerate(steps):
        y = 0.88 - i * 0.135
        _box(ax, 0.17, y, 0.24, 0.10, [name], fc_=color, fs=11)
        ax.text(0.315, y, code, ha="left", va="center", fontsize=10.5,
                fontproperties=matplotlib.font_manager.FontProperties(family="monospace"))
        ax.text(0.315, y - 0.045, note, ha="left", va="center", fontsize=9,
                fontproperties=FONT, color="#777777")
        if i:
            _arrow(ax, (0.17, y + 0.135 - 0.05), (0.17, y + 0.05), lw=1.1)
    _arrow(ax, (0.045, 0.88), (0.045, 0.20), rad=0.0, color="#c0392b")
    _arrow(ax, (0.045, 0.20), (0.17, 0.20), rad=0.0, color="#c0392b")
    _label(ax, 0.052, 0.60, _text("重复几千步", "repeat a few\nthousand steps"), color="#c0392b", fs=9)
    _box(ax, 0.72, 0.60, 0.50, 0.72,
         [_text("三条健康判据", "Three health checks"), "",
          _text("train 与 val 一起降 → 正常", "train & val both fall -> good"),
          _text("train 降 val 升 → 过拟合", "train falls, val rises -> overfit"),
          _text("val 不动 → LR / 标签有问题", "val flat -> LR or labels"), "",
          _text("loss = nan → LR 太大 / 没裁剪", "loss = nan -> LR too big / no clip"), "",
          _text("上限由数据决定，不是参数量", "Data sets the ceiling, not params")],
         fc_="#fbfbfb", ec="#c8c8c8", fs=10)
    return _save(fig, "tut_loop", outdir)


# ======================================================================================
# 图 6/7/8：实测曲线（数据来自 runs/*/history.json）
# ======================================================================================
def _load_history(tag: str):
    path = os.path.join(RUNS, tag, "history.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fig_loss_curves(outdir: str) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    # 左图：**同一份语料**（新的内置语料，2026-10 重训）的 train/val —— 过拟合的标准形状。
    # 不再把两种语料画在一起：它们的 val 文本不同，绝对值没有可比性。
    series = [("base", _text("内置语料", "small corpus"), "#c0392b"),
              ("quick", _text("quick（更小的模型）", "quick (smaller model)"), "#4c6ef5")]
    for tag, label, color in series:
        h = _load_history(tag)
        if not h:
            continue
        steps = [d["step"] for d in h["history"]]
        axes[0].plot(steps, [d["train"] for d in h["history"]], "--", color=color, lw=1.4,
                     label=label + _text(" · train", " train"))
        axes[0].plot(steps, [d["val"] for d in h["history"]], "-", color=color, lw=2.0,
                     marker="o", ms=3.4, label=label + " · val")
        best = h["best_val"]
        axes[0].scatter([best["step"]], [best["val"]], s=90, facecolor="none",
                        edgecolor=color, lw=1.8, zorder=5)
        up = 0.75
        axes[0].annotate(f'best val={best["val"]:.2f}@{best["step"]}',
                         xy=(best["step"], best["val"]), xytext=(best["step"] + 90,
                         best["val"] + up), fontsize=9, color=color, fontproperties=FONT,
                         arrowprops=dict(arrowstyle="->", color=color, lw=0.9))
    axes[0].set_xlabel(_text("训练步数", "step"), fontproperties=FONT)
    axes[0].set_ylabel(_text("交叉熵（nats）", "cross-entropy (nats)"), fontproperties=FONT)
    axes[0].set_title(_text("train 一路降到 0.11；val 在 step 500 触底 4.82 后反弹 → 过拟合",
                            "train -> 0.11; val bottoms at 4.82 (step 500) then rebounds"),
                      fontsize=11.5, fontproperties=FONT)
    axes[0].legend(fontsize=8.5, prop=FONT)
    axes[0].grid(alpha=0.3)

    # 右图：用 **best val（nats）** 而不是 bpb —— nats 在同一族内直接可比，
    # 不会像 bpb 那样因为「每 token 字节数」不同而制造假象。
    import numpy as np
    tags = ["xl_tiny", "xl_small", "xl_mid", "xl_big", "xl_big2"]
    labels = np.array([1.38, 5.28, 11.02, 24.65, 46.05])
    small = np.array([_load_history(t)["best_val"]["val"] for t in tags])
    big = np.array([_load_history("big_" + t)["best_val"]["val"] for t in tags])
    steps_s = [_load_history(t)["best_val"]["step"] for t in tags]
    steps_b = [_load_history("big_" + t)["best_val"]["step"] for t in tags]

    ax_r = axes[1]
    ax_r.plot(labels, small, "-o", color="#c0392b", lw=2, ms=6,
              label=_text("内置语料 56 万 token", "small corpus 0.56M"))
    ax_r.plot(labels, big, "-s", color="#2f9e44", lw=2, ms=6,
              label=_text("大语料 251 万 token", "big corpus 2.51M"))
    ax_r.set_xscale("log")
    ax_r.set_xlabel(_text("参数量（百万，对数轴）", "params (M, log)"), fontproperties=FONT)
    ax_r.set_ylabel(_text("最好那一步的 val（nats，越低越好）", "best val (nats, lower better)"),
                    fontproperties=FONT)
    ax_r.set_title(_text("模型大 33 倍，best val 只挪动 0.1 上下\n（同族内可比，跨族不可比）",
                         "33x params moves best val by ~0.1"),
                   fontsize=11.5, fontproperties=FONT)
    ax_r.legend(fontsize=9, prop=FONT, loc="lower right")
    ax_r.grid(alpha=0.3, which="both")
    ax_r.text(0.03, 0.06,
              _text("换大语料后 val 整体降约 0.35（同族内可比）",
                    "more data lowers val by ~0.35 within each family"),
              transform=ax_r.transAxes, fontsize=8.8, fontproperties=FONT,
              color="#444444",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dddddd", alpha=0.9))

    fig.tight_layout()
    return _save(fig, "tut_loss_curves", outdir)


def fig_scaling(outdir: str) -> list[str]:
    """两族数据**分开画**：bpb 的分母不同，混在一根 y 轴上比是错的。"""
    import numpy as np
    tags = ["xl_tiny", "xl_small", "xl_mid", "xl_big", "xl_big2"]
    labels = ["1.4M\n128d/4L", "5.3M\n256d/6L", "11M\n384d/6L", "25M\n512d/8L", "46M\n640d/10L"]
    fam = [("small", "#e07a6a", "#a93226", _text("内置语料 56 万 token", "small corpus 0.56M tok"),
            lambda h: h["best_val"]["val"] / math.log(2) * (30278 / 127911)),
           ("big", "#69b578", "#1e6f36", _text("大语料 251 万 token", "big corpus 2.51M tok"),
            lambda h: h["best_val"]["val"] / math.log(2) * (123374 / 370589))]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))
    for ax, (prefix, color, edge, label, conv) in zip(axes, fam):
        vals, steps = [], []
        for tag in tags:
            h = _load_history(tag if prefix == "small" else "big_" + tag)
            vals.append(conv(h))
            steps.append(h["best_val"]["step"])
        bars = ax.bar(range(len(tags)), vals, 0.62, color=color, edgecolor=edge, lw=1.2)
        lo, hi = min(vals), max(vals)
        for bar, st, v in zip(bars, steps, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (hi - lo) * 0.06,
                    f"{v:.3f}\n最优@{st}步", ha="center", fontsize=8.6, fontproperties=FONT,
                    color="#333333")
        best_i = int(np.argmin(vals))
        bars[best_i].set_edgecolor("#111111")
        bars[best_i].set_linewidth(2.4)
        ax.set_ylim(lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.62)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("bits / byte " + _text("（本面板内越低越好）", "(lower is better)"),
                      fontproperties=FONT)
        ax.set_title(label, fontsize=12, fontproperties=FONT)
        ax.grid(alpha=0.3, axis="y")
    axes[0].annotate(_text("最优点在 5.3M", "optimum at 5.3M"), xy=(1, 1.513),
                     xytext=(2.05, 1.578), fontsize=9.5, fontproperties=FONT, ha="center",
                     arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9,
                                     connectionstyle="arc3,rad=-0.25"))
    axes[1].annotate(_text("最优点在 46M", "optimum at 46M"), xy=(4, 1.970),
                     xytext=(3.0, 2.005), fontsize=9.5, fontproperties=FONT, ha="center",
                     arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9,
                                     connectionstyle="arc3,rad=0.25"))

    fig.text(0.5, -0.02, _text(
        "两个面板的纵轴各自独立、数值不可互比（bpb 的分母由 val 文本决定）：只在本面板内看趋势",
        "Panels are independently scaled and NOT comparable to each other"), ha="center",
        fontsize=9.5, fontproperties=FONT, color="#555555")
    fig.tight_layout()
    return _save(fig, "tut_scaling", outdir)


def fig_bench(outdir: str) -> list[str]:
    path = os.path.join(RUNS, "bench_gpu_long.json")
    if not os.path.exists(path):
        path = os.path.join(RUNS, "bench_gpu.json")
    if not os.path.exists(path):
        print("[tutorial] 跳过 tut_bench（先跑 python -m transformer.bench --json-out runs/bench_gpu_long.json）")
        return []
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    names = [k for k, v in rows.items() if "tok_per_s" in v]
    speed = [rows[k]["tok_per_s"] for k in names]
    mem = [rows[k]["peak_mib"] for k in names]
    x = range(len(names))
    fig, ax1 = plt.subplots(figsize=(13.5, 5.0))
    colors = ["#4c6ef5" if "GQA" in n else ("#e8590c" if "MQA" in n else "#868e96") for n in names]
    bars = ax1.bar(x, speed, 0.55, color=colors, edgecolor="#333333", lw=0.6)
    for bar, s in zip(bars, speed):
        ax1.text(bar.get_x() + bar.get_width() / 2, s + 1.2, f"{s:.1f}", ha="center",
                 fontsize=10, fontproperties=FONT)
    ax1.set_ylabel(_text("生成速度（token/秒）", "decode speed (tok/s)"), fontproperties=FONT)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, fontsize=9.5, fontproperties=FONT)
    ax2 = ax1.twinx()
    ax2.plot(x, mem, "k--o", lw=1.6, ms=5, label=_text("峰值显存", "peak memory"))
    ax2.set_ylabel(_text("峰值显存（MiB）", "peak memory (MiB)"), fontproperties=FONT)
    ax2.legend(fontsize=9.5, prop=FONT, loc="upper right")

    ax1.grid(alpha=0.3, axis="y")
    ax1.set_ylim(0, max(speed) * 1.32)
    ax1.annotate(_text("GQA 比 MHA 快 11%、省 35% 显存；\n关掉 cache 反而最慢（prompt 1024 + 生成 128）",
                       "GQA vs MHA: +11% speed, -35% memory"),
                 xy=(1, speed[1]), xytext=(1.45, max(speed) * 1.16), fontsize=10, fontproperties=FONT,
                 arrowprops=dict(arrowstyle="->", color="#555555"))
    fig.tight_layout()
    return _save(fig, "tut_bench", outdir)


# ======================================================================================
# 汇总
# ======================================================================================
BUILDERS = [fig_hero, fig_forward, fig_attention, fig_sample, fig_loop,
            fig_loss_curves, fig_scaling, fig_bench]


def check_anchors() -> list[str]:
    """图注必须出现在 README 里：删了正文的图注 → 测试失败，而不是留下一张没人引用的图。"""
    readme = os.path.join(HERE, "README.md")
    if not os.path.exists(readme):
        return [f"找不到 {readme}"]
    with open(readme, encoding="utf-8") as fh:
        text = fh.read()
    return [f"{name}: README 里找不到图注 {cap!r}" for name, cap in CAPTIONS.items()
            if cap not in text]


def build(outdir: str = OUTDIR, force_cn: bool | None = None) -> dict:
    global FONT, CN
    FONT, CN = rootfc.setup_font()
    if force_cn is not None:
        CN = force_cn
    os.makedirs(outdir, exist_ok=True)
    report = {}
    for fn in BUILDERS:
        name = fn.__name__.replace("fig_", "tut_")
        warns = fn(outdir)
        report[name] = {"missing_glyphs": warns,
                        "bytes": os.path.getsize(os.path.join(outdir, f"{name}.png"))
                        if os.path.exists(os.path.join(outdir, f"{name}.png")) else 0}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--check", action="store_true", help="只校验图注锚点")
    args = ap.parse_args(argv)
    missing = check_anchors()
    if missing:
        print("图注校验失败：")
        for m in missing:
            print("  -", m)
        return 1
    print(f"图注校验通过（{len(CAPTIONS)} 张图）")
    if args.check:
        return 0
    for name, r in build(args.outdir).items():
        print(f"[{name}] {r['bytes'] // 1024} KB，缺字形 {len(r['missing_glyphs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
