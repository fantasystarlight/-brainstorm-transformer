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

import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("root_flowchart_test", os.path.join(ROOT, "flowchart.py"))
fc = _ilu.module_from_spec(_spec)
sys.modules["root_flowchart_test"] = fc
_spec.loader.exec_module(fc)
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("mt_flowchart", os.path.join(HERE, "flowchart.py"))
mt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mt)


def _has_cjk_font() -> bool:
    """本机有没有可用的中文字体（CI runner 上没有）。"""
    _, cn = fc.setup_font()
    return cn


ASCII_RE = re.compile(r"^[\x00-\x7f]*$")


def _assert_labels_are_ascii(spec: dict) -> None:
    """英文标签必须是纯 ASCII——否则「英文降级」根本挡不住缺字形。"""
    for node in spec["nodes"]:
        label = node.get("en") or node["label"]
        assert ASCII_RE.match(label), f"{spec['title'][:14]} 的英文标签含非 ASCII：{label!r}"
    assert ASCII_RE.match(spec.get("en_title", "")), spec.get("en_title")


def test_anchors_exist_in_src():
    assert mt.check_anchors() == []


def _load_tutorial():
    """按文件路径加载 transformer/tutorial.py（它内部有相对导入的兜底）。"""
    spec = importlib.util.spec_from_file_location("tut_flowchart",
                                                  os.path.join(HERE, "tutorial.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tutorial_captions_are_in_readme():
    """教程配图的图注必须逐字出现在 README 里。

    删掉正文图注 → 这里失败，而不是留下一张没人引用的图。
    （这是 README 重写后最容易悄悄发生的事：改了正文措辞，图注锚点就漂了。）
    """
    tut = _load_tutorial()
    missing = tut.check_anchors()
    assert missing == [], missing
    assert len(tut.CAPTIONS) >= 8, f"教程图少于预期：{sorted(tut.CAPTIONS)}"


def test_tutorial_caption_checker_has_discriminating_power():
    """负对照：把一个图注改坏，检查器必须报出来。"""
    tut = _load_tutorial()
    original = tut.CAPTIONS["tut_hero"]
    tut.CAPTIONS["tut_hero"] = "这句话绝不会出现在 README 里-7c1d"
    try:
        missing = tut.check_anchors()
    finally:
        tut.CAPTIONS["tut_hero"] = original
    assert any("tut_hero" in m for m in missing), missing


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


def test_english_labels_are_ascii():
    """守住「无中文字体时降级为英文」这条路：英文标签里不能混进任何非 ASCII 字符。

    这条是**纯逻辑检查**，不依赖本机字体，所以 CI 和本地结果一致——
    它就是用来替代「靠字体检测猜 CI 行为」的。
    """
    for spec in mt.DIAGRAMS.values():
        _assert_labels_are_ascii(spec)


def test_render_products_are_consistent_and_nonempty():
    outdir = tempfile.mkdtemp(prefix="mt-flow-")
    # 用英文模式做守卫：纯 ASCII 标签 + DejaVu Sans，任何机器上都不该缺字形
    report = mt.build(outdir, force_cn=False)
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
    mt.build(outdir, force_cn=False)
    second = mt.build(outdir, force_cn=False)
    changed = [k for k, v in second.items() if v["changed"]]
    import shutil
    shutil.rmtree(outdir)
    assert changed == [], f"重复渲染却改动了：{changed}"


def test_dep_graph_matches_imports():
    """依赖图的边必须等于从源码解析出来的 import 边（手写边会漂移）。"""
    mt._setup()
    # 只比 (src, dst)：图里的边带中英双语标签，解析器只给 (src, dst, "")，比整元组没意义
    parsed_pairs = {(e[0], e[1]) for e in mt._parse_import_edges()}
    graph_pairs = {(e[0], e[1]) for e in mt.FLOW_DEPS["edges"]}
    assert graph_pairs == parsed_pairs, (graph_pairs ^ parsed_pairs)
    assert ("bpe", "data") in parsed_pairs and ("corpus", "data") in parsed_pairs
    # 负对照：源码里不存在 bpe -> train 的 import
    assert ("bpe", "train") not in parsed_pairs


def test_mermaid_and_png_share_same_edges():
    """PNG 与 Mermaid 必须来自同一份 edges（图与文本不能各写一套）。"""
    outdir = tempfile.mkdtemp(prefix="mt-flow-src-")
    mt.build(outdir, force_cn=False)
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


def test_chinese_mode_has_no_missing_glyphs_when_font_available():
    """有中文字体时，中文图也不许出豆腐块；没字体就跳过（CI 属于这种）。"""
    import pytest
    if not _has_cjk_font():
        pytest.skip("本机没有中文字体（如 CI runner），中文标签会降级为英文")
    outdir = tempfile.mkdtemp(prefix="mt-flow-cn-")
    report = mt.build(outdir, force_cn=True)
    bad = {k: v["missing_glyphs"] for k, v in report.items() if v["missing_glyphs"]}
    assert bad == {}, bad
    import shutil
    shutil.rmtree(outdir)


def test_readme_image_links_resolve():
    """README 里 `![](…)` 指向的图必须真的存在。

    这条是补的：README 重写时我把图片路径写成 `docs/tut_hero.png`，
    但 README 在 `transformer/` 里、图在 `../docs/` —— **8 张图全是死链**，
    本地编辑器不报错、GitHub 上也只显示破图，靠肉眼完全看不出来。
    """
    readme = os.path.join(HERE, "README.md")
    with open(readme, encoding="utf-8") as fh:
        text = fh.read()
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert links, "README 里一张图都没有？"
    missing = [u for u in links
               if not os.path.exists(os.path.normpath(os.path.join(HERE, u)))]
    assert missing == [], f"README 引用了不存在的图：{missing}"
    # 负对照：一个显然不存在的路径必须被这条检查抓出来
    bogus = os.path.normpath(os.path.join(HERE, "docs/绝不存在-4f2a.png"))
    assert not os.path.exists(bogus), "负对照本身失效了"
