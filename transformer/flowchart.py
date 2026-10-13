#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 mini transformer 的流程图（PNG + Mermaid + 纯文本，三者同源）。

遵循本机约定（§5）：
* 节点/边**只定义一次**，PNG 与 Mermaid/TXT 从同一份 nodes/edges 渲染；
* 每个节点带 ``anchor``，由 ``check_anchors()`` 断言它**仍存在于对应源码文件里**——
  改代码而没改图，会表现为**测试失败**，而不是留下一张指向不存在代码的图；
* 复用工作区根目录 ``flowchart.py`` 的渲染器（write-if-changed + 原子替换、
  中文字体自动降级、坐标范围由实际位置推导）；
* 依赖图（deps）**从源码 import 语句解析**，不手写，避免与代码漂移。
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "docs")
sys.path.insert(0, ROOT)
# ⚠️ 这里必须用**别名**导入：本文件自己就叫 transformer/flowchart.py，
# 直接 `import flowchart` 在有 `python -m transformer.*` 参与时可能解析到包内同名模块，
# 于是 `fc.setup_font` 之类的属性就找不到了（真踩过）。
import flowchart as _root_renderer  # noqa: E402
import importlib as _importlib
fc = _importlib.import_module("flowchart")   # 保持原有名字不变，下面全部沿用
if not hasattr(fc, "setup_font"):            # 万一解析到包内自己，按文件路径兜底加载
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("root_flowchart", os.path.join(ROOT, "flowchart.py"))
    fc = _ilu.module_from_spec(_spec)
    sys.modules["root_flowchart"] = fc
    _spec.loader.exec_module(fc)

PKG = "transformer"
SRC = {name: os.path.join(HERE, f"{name}.py")
       for name in ("bpe", "corpus", "data", "model", "train", "generate")}
SRC["_pkg"] = os.path.join(HERE, "__init__.py")


# ======================================================================================
# 图 1：数据 → 训练的总流水线
# ======================================================================================
FLOW_PIPELINE = {
    "file": SRC["corpus"],
    "title": "训练流水线：文本 → BPE → token 流 → batch",
    "en_title": "Pipeline: text -> BPE -> tokens -> batches -> training",
    "direction": "LR",
    # 层标题会被 fc.render_png 画在图的顶部（y=1.02），串太长就会撞上主标题，所以用短名。
    "layers": ["语料", "分词", "数据", "训练"],
    "en_layers": ["corpus", "BPE", "data", "train"],
    "nodes": [
        {"id": "C1", "layer": 0, "label": "① 收集本地文本\n*.py / *.md（约 19 万字符）",
         "en": "1 collect local .py/.md texts", "anchor": "显式排除：语料库自身（自指）"},
        {"id": "C2", "layer": 0, "label": "② 先切 train/val\n再训 BPE（否则打分不诚实）",
         "en": "2 split train/val BEFORE BPE", "anchor": "尽量切在换行处，避免把一行代码劈成两半影响验证集统计"},
        {"id": "B1", "layer": 1, "label": "③ 字节级 BPE\n反复合并最高频相邻对",
         "en": "3 byte-level BPE merges", "anchor": "vocab_size 是**总**词表大小（含 256 字节 + 2 特殊 token）"},
        {"id": "D1", "layer": 2, "label": "④ 编码成 token 流\n存 uint16 的 .bin",
         "en": "4 encode to uint16 token stream", "anchor": "HF ByteLevel-BPE"},
        {"id": "D2", "layer": 2, "label": "⑤ 随机采样 batch\nx = [t..t+T)  y = x 右移一位",
         "en": "5 sample batch, y = x shifted by 1", "anchor": "语言模型的标签就是输入本身，只是错开一位"},
        {"id": "T1", "layer": 3, "label": "⑥ 前向 + 交叉熵 loss",
         "en": "6 forward + cross entropy", "anchor": "ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), targets[:, s:e].reshape(-1),"},
        {"id": "T2", "layer": 3, "label": "⑦ 反向传播 → 梯度裁剪 → AdamW\nwarmup + 余弦衰减",
         "en": "7 backward, clip, AdamW, cosine", "anchor": "学习率调度用「线性 warmup + 余弦衰减」"},
        {"id": "T3", "layer": 3, "label": "⑧ 定期估 val loss\n只存最佳权重 ckpt.pt",
         "en": "8 eval val loss, keep best ckpt", "anchor": "最佳 val"},
    ],
    "edges": [("C1", "C2"), ("C2", "B1"), ("B1", "D1"), ("D1", "D2"), ("D2", "T1"),
              ("T1", "T2"), ("T2", "T1", "每个 step 重复", "every step"),
              ("T2", "T3", "每 log_every 步", "every log_every steps"),
              ("T3", "T2", "继续训练", "keep training")],
    "note": "关键顺序：BPE 只喂 train 文本 → val loss 才是诚实的「没见过」的困惑度。",
    "en_note": "Order matters: BPE sees train only, so val loss is honest.",
    "check_anchor": lambda: _anchors_in(FLOW_PIPELINE, {"C1": "corpus", "C2": "corpus",
                                                       "B1": "bpe", "D1": "data", "D2": "data",
                                                       "T1": "model", "T2": "train", "T3": "train"}),
}

# ======================================================================================
# 图 2：模型本体（token 进 → logits 出）
# ======================================================================================
FLOW_MODEL = {
    "file": SRC["model"],
    "title": "模型本体：token 序列 → 下一个 token 概率",
    "en_title": "Model: how a token sequence becomes next-token probabilities",
    "direction": "LR",
    "layers": ["输入", "注意力", "前馈", "输出"],
    "nodes": [
        {"id": "E1", "layer": 0, "label": "ids [B,T]\n词表查表 Embedding",
         "en": "ids [B,T] embedding", "anchor": "x = self.tok_emb(ids)"},
        {"id": "A1", "layer": 1, "label": "RMSNorm\n（Pre-Norm，先归一化再进子层）",
         "en": "RMSNorm pre-norm", "anchor": "self.norm1 = RMSNorm(d_model)"},
        {"id": "A2", "layer": 1, "label": "因果自注意力 + RoPE\n[B,T,C]→[B,H,T,hd]（GQA 可省 KV）",
         "en": "causal attention + RoPE (GQA)", "anchor": "self.attn = CausalSelfAttention(d_model, n_heads, dropout, n_kv_heads)"},
        {"id": "R1", "layer": 1, "label": "残差相加\nx = x + attn(...)",
         "en": "residual add", "anchor": "x = x + a"},
        {"id": "F1", "layer": 2, "label": "RMSNorm → SwiGLU 前馈\n存知识的主力，参数占大头",
         "en": "RMSNorm + SwiGLU FFN", "anchor": "self.ffn = SwiGLU(d_model)"},
        {"id": "R2", "layer": 2, "label": "残差相加\nx = x + ffn(...)",
         "en": "residual add", "anchor": "x = x + self.ffn(self.norm2(x))"},
        {"id": "O1", "layer": 3, "label": "最后 RMSNorm\n+ lm_head 线性层（分块算 loss）",
         "en": "final norm + lm_head", "anchor": "x = self.norm_f(x)"},
        {"id": "O2", "layer": 3, "label": "只返回最后一步 logits [B,1,V]\n省掉 [B,T,V] 常驻张量",
         "en": "logits [B,1,V] only", "anchor": "return head(x[:, -1:]), loss, new_kv"},
    ],
    "edges": [("E1", "A1"), ("A1", "A2"), ("A2", "R1"), ("R1", "F1"), ("F1", "R2"),
              ("R2", "O1"), ("O1", "O2"),
              ("R1", "F1", "重复 n_layers 次")],
    "note": "4 层就是把这套 block 堆 4 次；每层都有两条残差旁路，梯度才能回流到最底层。",
    "en_note": "n_layers = stack this block; residuals keep gradients flowing.",
    "check_anchor": lambda: _anchors_in(FLOW_MODEL, {n["id"]: "model" for n in FLOW_MODEL["nodes"]}),
}

# ======================================================================================
# 图 3：注意力的控制流（含因果掩码与 KV cache 分支）
# ======================================================================================
FLOW_ATTENTION = {
    "file": SRC["model"],
    "title": "因果自注意力：掩码与 KV cache",
    "en_title": "Causal attention control flow: masking and KV cache",
    "direction": "LR",
    "layers": ["投影", "位置", "打分", "输出"],
    "nodes": [
        {"id": "Q1", "layer": 0, "label": "两个投影：q 出 n_heads 份\nkv 只出 n_kv_heads 份（GQA）",
         "en": "q_proj + kv_proj (GQA)", "anchor": "self.kv_proj = nn.Linear(d_model, 2 * self.n_kv_heads * self.head_dim, bias=False)"},
        {"id": "Q2", "layer": 0, "label": "拆头 + GQA 共享\n[B,T,C] → [B,H,T,hd]",
         "en": "reshape + repeat kv heads", "anchor": "k, v = k.repeat_interleave(self.n_rep, dim=1), v.repeat_interleave(self.n_rep, dim=1)"},
        {"id": "P1", "layer": 1, "label": "施加 RoPE\n（只旋转 q/k，不碰 v）",
         "en": "apply RoPE to q,k", "anchor": "q, k, v = apply_rope(q, cos_sin, offset), apply_rope(k, cos_sin, offset), v"},
        {"id": "P2", "layer": 1, "label": "有 KV cache？\n把历史 k/v 接在前面（存 n_kv_heads 头）",
         "en": "concat past k/v (n_kv_heads)", "anchor": "k = torch.cat([past[0], k], dim=-2)"},
        {"id": "S1", "layer": 2, "label": "因果掩码\n未来位置 = False（不可见）",
         "en": "boolean causal mask", "anchor": "allowed = torch.tril(allowed, diagonal=Tk - Tq)"},
        {"id": "S2", "layer": 2, "label": "SDPA 一步算完\nsoftmax·V（不物化 T×T 矩阵）",
         "en": "scaled_dot_product_attention", "anchor": "out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,"},
        {"id": "V1", "layer": 3, "label": "权重 @ v\n历史信息被加权搬进当前位置",
         "en": "weights @ v (inside SDPA)", "anchor": "out = out.transpose(1, 2).reshape(B, Tq, C)"},
        {"id": "V2", "layer": 3, "label": "多头拼回 + 输出投影\n把 **未展开** 的 k/v 存回 cache",
         "en": "merge heads, cache n_kv_heads", "anchor": "return self.proj(out), to_cache"},
    ],
    "edges": [("Q1", "Q2"), ("Q2", "P1"), ("P1", "P2"), ("P2", "S1"), ("S1", "S2"),
              ("S2", "V1"), ("V1", "V2"),
              ("Q2", "P1", "Tq=1 且 cache 命中时复用历史")],
    "note": "训练时缓存不参与（每个位置并行算）；生成时每步只算 1 个 token，靠 cache 复用历史 k/v。",
    "en_note": "Training is parallel; generation reuses cached k/v one token at a time.",
    "check_anchor": lambda: _anchors_in(FLOW_ATTENTION, {n["id"]: "model" for n in FLOW_ATTENTION["nodes"]}),
}

# ======================================================================================
# 图 4：模块依赖（自动从 import 解析，不手写）
# ======================================================================================
def _parse_import_edges() -> list[tuple[str, str, str]]:
    """扫描 transformer/*.py 的 from . import x / from .x import y，返回模块级依赖边。"""
    edges, files = [], {os.path.splitext(os.path.basename(p))[0]: p for p in SRC.values()}
    for name, path in files.items():
        if name.startswith("_"):
            continue
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        # 支持三种写法：from .bpe import X / from . import bpe / from . import bpe as bp, corpus
        for m in re.finditer(r"^\s*from \.(\w+) import|^\s*from \. import (.+)$", text, re.M):
            mods = [g for g in m.groups() if g]
            for mod in mods:
                for one in [s.strip() for s in mod.split(",")]:
                    one = one.split(" as ")[0].strip()   # 去掉别名
                    if one in files and one != name:
                        edges.append((f"{one}", f"{name}", ""))
    return sorted(set(edges))


def _module_node(path: str, module: str, layer: int) -> dict:
    """一个模块的节点：id 用模块名，layer 由依赖深度决定，anchor 取文件里必然存在的字符串。"""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    for cand in (f"from . import {module}", f"from .{module} import", f"class {module.capitalize()}",
                 f"def {module}", f'"""'):
        if cand in text:
            return {"id": module, "layer": layer, "label": f"{module}.py", "en": f"{module}.py",
                    "anchor": cand, "file": path}
    return {"id": module, "layer": layer, "label": f"{module}.py", "en": f"{module}.py",
            "anchor": "", "file": path}


FLOW_DEPS = {
    "file": SRC["model"],
    "title": "模块依赖图（由源码 import 自动解析）",
    "en_title": "Module dependency graph (parsed from imports)",
    "direction": "LR",
    "layers": ["叶子（无项目内依赖）", "中层", "入口脚本"],
    "en_layers": ["leaves", "middle", "entry points"],
    "nodes": [],   # 在 _setup() 里填：需要先解析出边才能算依赖深度
    "edges": [],
    "note": "bpe/model 是叶子（零项目内依赖）；data 依赖 bpe+corpus；train/generate 在顶层。",
    "en_note": "bpe/model are leaves; data depends on bpe+corpus; train/generate are entry points.",
    "check_anchor": None,
}


def _dep_depth(modules: list[str], edges: list[tuple[str, str, str]]) -> dict[str, int]:
    """依赖深度 = 从任一叶子出发的最长路径（叶子=0）。用于把节点分列排布。"""
    depth = {m: 0 for m in modules}
    for _ in range(len(modules)):                       # 最多迭代 N 轮就收敛
        changed = False
        for src, dst, _ in edges:
            if depth[dst] < depth[src] + 1:
                depth[dst] = depth[src] + 1
                changed = True
        if not changed:
            break
    return depth

DIAGRAMS = {"flow_pipeline": FLOW_PIPELINE, "flow_model": FLOW_MODEL,
            "flow_attention": FLOW_ATTENTION, "dep_modules": FLOW_DEPS}


def _anchors_in(spec: dict, mapping: dict[str, str]) -> list[str]:
    """检查 spec 里各节点的 anchor 是否还在指定源码文件里。返回缺失描述。"""
    missing = []
    for node in spec["nodes"]:
        mod = mapping.get(node["id"])
        if mod is None or not node.get("anchor"):
            continue
        with open(SRC[mod], encoding="utf-8") as fh:
            if node["anchor"] not in fh.read():
                missing.append(f"{spec['title'][:12]}…:{node['id']} -> {SRC[mod]} 里找不到 "
                               f"{node['anchor']!r}")
    return missing


def _grid_layout(nodes: list[dict]):
    """按 layer 排成网格，并把整体**下移一点**，给顶部的主标题留出空位。

    为什么要这么写：通用渲染器 ``fc.render_png`` 把主标题（y≈1.06）和层标题（y=1.02）
    放在几乎同一高度，层名一多就会和主标题叠在一起——我第一次渲染出来的图就是这样
    （本机约定 §5：生成完必须自己看图，这一步不能省）。位置由我们自己算好传进去。
    """
    n_layers = max(n["layer"] for n in nodes) + 1
    pos = fc.layout(nodes, n_layers)
    return {k: (x, y * 0.86) for k, (x, y) in pos.items()}, n_layers


def _localized(spec: dict, cn: bool) -> dict:
    """按语言裁一份 spec：英文模式下**连脚注和边标签都必须是 ASCII**。"""
    out = dict(spec)
    if not cn:
        out["edges"] = [tuple(ed[:2]) + (ed[3],) if len(ed) > 3 else tuple(ed) for ed in spec["edges"]]
        out["note"] = spec.get("en_note", spec["note"])
        out["title"] = spec.get("en_title", spec["title"])
    return out


def _render_png(spec: dict, out_png: str, cn: bool, font) -> list[str]:
    """渲染 PNG：优先把自定义位置传给通用渲染器；它不认这些参数就退回默认布局。"""
    spec = _localized(spec, cn)
    pos, n_layers = _grid_layout(spec["nodes"])
    try:
        return fc.render_png(spec, out_png, cn, font, positions=pos, n_layers=n_layers)
    except TypeError:
        return fc.render_png(spec, out_png, cn, font)


def _setup() -> None:
    """把依赖图的边补上（它在 DIAGRAMS 定义之后才能解析），并统一补充字段。"""
    FLOW_DEPS["edges"] = _parse_import_edges()
    modules = [os.path.splitext(os.path.basename(p))[0] for p in sorted(SRC.values())
               if os.path.basename(p)[0] != "_"]
    depth = _dep_depth(modules, FLOW_DEPS["edges"])
    FLOW_DEPS["nodes"] = [_module_node(SRC[m], m, depth[m]) for m in modules]
    # 层数显式给够，否则最深那层会被通用渲染器当成最后一列（这里正好 1..3 层）
    FLOW_DEPS["layers"] = ["叶子（无项目内依赖）", "中层", "入口脚本"][:max(depth.values()) + 1]
    for spec in DIAGRAMS.values():
        spec.setdefault("en_title", spec["title"])
        spec.setdefault("en_layers", [f"layer {i}" for i in range(len(spec["layers"]))])
        spec.setdefault("en_note", spec["note"])
        for node in spec["nodes"]:
            node.setdefault("en", node["label"])
        # 边也统一成 4 元组 (src, dst, 中文标签, 英文标签)。
        # 为什么必须补英文：没有中文字体的机器（如 CI runner）会走「英文降级」，
        # 但**脚注和边标签此前仍是硬编码中文** → 渲染出来是豆腐块。
        # 这个 bug 是本仓库 CI 抓出来的（本地有微软雅黑，一直绿）。
        spec["edges"] = [tuple(ed) + (ed[3] if len(ed) > 3 else "",) if len(ed) > 2
                         else (ed[0], ed[1], "", "") for ed in spec["edges"]]


def check_anchors() -> list[str]:
    _setup()
    out = []
    for spec in DIAGRAMS.values():
        if spec.get("check_anchor"):
            out.extend(spec["check_anchor"]())
    return out


def normalize_svg(data: bytes) -> bytes:
    """把 SVG 规范化，让同一张图两次渲染**逐字节相同**（PNG 本来就没这个问题）。

    三处非确定性（都是渲染两次 diff 出来的，不是猜的）：
    1. ``<dc:date>`` 里写着渲染时刻的时间戳 —— 删掉；
    2. matplotlib 给图元生成**随机** id（``id="p018c86822e"``）；
    3. ``figure_1`` / ``patch_7`` 这类 id 是**进程内递增**的，同一进程渲染的第二张图会变成 ``figure_2``。

    做法：id 按出现顺序统一重编号为 ``x0, x1, …``，``url(#…)`` 等引用同步替换。
    顺序确定 → 结果确定 → `docs/` 里已打开的图不会因为重跑而失效。
    """
    import re
    text = data.decode("utf-8")
    text = re.sub(r"\s*<dc:date>[^<]*</dc:date>", "", text)
    seen: dict[str, str] = {}

    def _canon(m):
        raw = m.group(1)
        if raw not in seen:
            seen[raw] = f"x{len(seen)}"
        return m.group(0).replace(raw, seen[raw])

    text = re.sub(r'id="([^"]+)"', _canon, text)
    text = re.sub(r"url\(#([^)]+)\)", _canon, text)
    text = re.sub(r'(?:xlink:)?href="#([^"]+)"', _canon, text)
    return text.encode("utf-8")


def build(outdir: str = OUTDIR, force_cn: bool | None = None) -> dict:
    _setup()
    font, cn = fc.setup_font()
    if force_cn is not None:
        cn = force_cn
    os.makedirs(outdir, exist_ok=True)
    report = {}
    for name, spec in DIAGRAMS.items():
        png = os.path.join(outdir, f"{name}.png")
        tmp_png = os.path.join(outdir, f".{name}.tmp.png")   # 带 .png 后缀，savefig 才认格式
        warns = _render_png(spec, tmp_png, cn, font)
        with open(tmp_png, "rb") as fh:
            data = fh.read()
        os.remove(tmp_png)
        # SVG：矢量格式，与 PNG 同源（同一份 spec + 同一份 positions）
        tmp_svg = os.path.join(outdir, f".{name}.tmp.svg")
        _render_png(spec, tmp_svg, cn, font)
        with open(tmp_svg, "rb") as fh:
            svg = normalize_svg(fh.read())
        os.remove(tmp_svg)
        # 注意：三个 write_if_changed 必须都执行，所以用位运算 | 而不是 or——
        # 用 or 的话 PNG 一变 True 就会短路掉 MMD/TXT（我第一次就踩了这个坑，
        # 结果 docs/ 里只有 PNG，没有 Mermaid 源码和文本大纲）。
        changed = (fc.write_if_changed(png, data)
                   | fc.write_if_changed(os.path.join(outdir, f"{name}.svg"), svg)
                   | fc.write_if_changed(os.path.join(outdir, f"{name}.mmd"),
                                         fc.render_mermaid(spec))
                   | fc.write_if_changed(os.path.join(outdir, f"{name}.txt"),
                                         fc.render_txt(spec)))
        report[name] = {"png": png, "bytes": len(data), "changed": changed,
                        "missing_glyphs": warns, "nodes": len(spec["nodes"]),
                        "edges": len(spec["edges"])}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--check", action="store_true", help="只校验锚点")
    args = ap.parse_args(argv)
    missing = check_anchors()
    if missing:
        print("锚点校验失败：")
        for m in missing:
            print("  -", m)
        return 1
    print(f"锚点校验通过（{sum(len(s['nodes']) for s in DIAGRAMS.values())} 个节点）")
    if args.check:
        return 0
    for name, r in build(args.outdir).items():
        print(f"[{name}] {r['nodes']} 节点 / {r['edges']} 边，{r['bytes'] // 1024} KB，"
              f"{'已更新' if r['changed'] else '无变化'}，缺字形 {len(r['missing_glyphs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
