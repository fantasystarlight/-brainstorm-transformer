#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""transformer/flowchart.py 的守卫测试。

要点（本机约定 §5）：
* 渲染**只写临时目录**，绝不碰 ``docs/``（否则每跑一次测试就刷新交付物 mtime，
  侧栏里已打开的图会失效）；
* 断言每个节点的 anchor 仍存在于对应源码文件（图过期 → 测试失败）；
* 断言无缺字形：**有中文字体时断言中文图，没有时（如 CI）断言英文降级图**——两边都不能出豆腐块；
* PNG/MMD/TXT 三份产物同源：MMD 与 TXT 必须包含全部节点与边；
* 依赖图必须与源码 import 一致（自动解析，不手写）；
* 负对照：故意塞一个不存在的 anchor，检查器必须报出来。
"""
from __future__ import annotations

import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import flowchart as fc  # noqa: E402  根目录通用渲染器
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("mt_flowchart", os.path.join(HERE, "flowchart.py"))
mt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mt)


def _font_mode() -> bool:
    """CI（ubuntu-latest）没有中文字体，渲染器会**降级成英文标签**。

    测试要跟着这个事实走：有中文字体就断言中文图无缺字形，没有就断言英文图无缺字形。
    否则「本地绿、CI 红」，而真正的信息只是「这台机器没装中文字体」。
    """
    _, cn = fc.setup_font()
    return cn


def test_anchors_exist_in_src():
    assert mt.check_anchors() == []


def test_anchor_checker_has_discriminating_power():
    """负对照：塞一个源码里绝不存在的 anchor，检查器必须报出来。"""
    spec = mt.FLOW_MODEL
    spec["nodes"].append({"id": "_BOGUS", "layer": 0, "label": "x", "en": "x",
                          "anchor": "这句话绝不会出现在源码里-9f2a"})
    try:
        spec["check_anchor"] = lambda: mt._anchors_in(spec, {"_BOGUS": "model"})
        missing = mt.check_anchors()
    finally:
        spec["nodes"].pop()
        spec["check_anchor"] = lambda: mt._anchors_in(
            spec, {n["id"]: "model" for n in spec["nodes"]})
    assert any("_BOGUS" in m for m in missing), "锚点检查器没有鉴别力"


def test_render_products_are_consistent_and_nonempty():
    outdir = tempfile.mkdtemp(prefix="mt-flow-")
    report = mt.build(outdir, force_cn=_font_mode())
    for name, spec in mt.DIAGRAMS.items():
        for ext in ("png", "mmd", "txt"):
            path = os.path.join(outdir, f"{name}.{ext}")
            assert os.path.getsize(path) > 0, path
        assert report[name]["missing_glyphs"] == [], (name, report[name]["missing_glyphs"])
        with open(os.path.join(outdir, f"{name}.mmd"), encoding="utf-8") as fh:
            mmd = fh.read()
        with open(os.path.join(outdir, f"{name}.txt"), encoding="utf-8") as fh:
            txt = fh.read()
        for node in spec["nodes"]:
            assert node["id"] in mmd and node["id"] in txt, (name, node["id"])
        for edge in spec["edges"]:
            assert f"{edge[0]} -> {edge[1]}" in txt, (name, edge)
    import shutil
    shutil.rmtree(outdir)


def test_rebuild_is_idempotent():
    """同样输入重复渲染，第二次必须报告「无变化」（否则侧栏打开的图会反复失效）。"""
    outdir = tempfile.mkdtemp(prefix="mt-flow-idem-")
    mt.build(outdir, force_cn=_font_mode())
    second = mt.build(outdir, force_cn=_font_mode())
    changed = [k for k, v in second.items() if v["changed"]]
    import shutil
    shutil.rmtree(outdir)
    assert changed == [], f"重复渲染却改动了：{changed}"


def test_dep_graph_matches_imports():
    """依赖图的边必须等于从源码解析出来的 import 边（手写边会漂移）。"""
    mt._setup()
    parsed = mt._parse_import_edges()
    assert mt.FLOW_DEPS["edges"] == parsed
    pairs = {(a, b) for a, b, _ in parsed}
    assert ("bpe", "data") in pairs and ("corpus", "data") in pairs
    # 负对照：源码里不存在 bpe -> train 的 import
    assert ("bpe", "train") not in pairs


def test_mermaid_and_png_share_same_edges():
    """PNG 与 Mermaid 必须来自同一份 edges（图与文本不能各写一套）。"""
    outdir = tempfile.mkdtemp(prefix="mt-flow-src-")
    mt.build(outdir, force_cn=_font_mode())
    for name, spec in mt.DIAGRAMS.items():
        with open(os.path.join(outdir, f"{name}.mmd"), encoding="utf-8") as fh:
            mmd = fh.read()
        arrow_lines = [ln for ln in mmd.splitlines() if "-->" in ln]
        # Mermaid 里的边条数必须与 spec 的边条数一致（允许带标签写法）
        assert len(arrow_lines) == len(spec["edges"]), (name, arrow_lines)
        for src, dst, *rest in spec["edges"]:
            assert re.search(rf"\b{src}\b.*-->.*\b{dst}\b", mmd), (name, src, dst)
    import shutil
    shutil.rmtree(outdir)
