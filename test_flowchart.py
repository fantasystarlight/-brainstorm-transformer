#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""flowchart.py 的守卫测试。

要点：
* 渲染**只写到临时目录**，绝不碰 `docs/`（避免跑测试就刷新交付物 mtime，让侧栏打开的图失效）；
* 断言每个节点的 anchor 仍存在于 `RL新策略头脑风暴.md`（图过期 → 测试失败，而不是留下一张
  指向不存在内容的图）；
* 断言无缺字形：**只有本机有中文字体时才断言**——CI runner 上只有 DejaVu Sans，
  这套图的标签是中文，硬断言必然红（而那不是代码错，是环境没字体）；
  没有字体时改为断言「渲染成功且产物非空」，并把缺字形当作已知降级记在报告里；
* 断言 PNG/MMD/TXT 三份产物同源：MMD 与 TXT 必须包含全部节点与边；
* 负对照：故意造一个不存在的 anchor，检查器必须报出来（证明检查器有鉴别力）。
"""
from __future__ import annotations

import os
import shutil
import tempfile

import flowchart as fc


def test_anchors_exist_in_doc():
    missing = fc.check_anchors()
    assert missing == [], f"文档里找不到这些锚点：{missing}"


def test_anchor_checker_has_discriminating_power():
    """负对照：塞一个不存在的 anchor，检查器必须报错。"""
    spec = fc.DIAGRAMS["flow_experiment"]
    spec["nodes"].append({"id": "_BOGUS", "layer": 0, "label": "x", "en": "x",
                          "anchor": "这句话绝不会出现在文档里-9f2a"})
    try:
        missing = fc.check_anchors()
    finally:
        spec["nodes"].pop()
    assert any("_BOGUS" in m for m in missing), "锚点检查器没有鉴别力"


def _has_cjk_font() -> bool:
    """本机有没有中文字体（CI runner 上没有；本地 WSL2 走 /mnt/c/Windows/Fonts）。"""
    return fc.setup_font()[1]


def test_render_products_are_consistent_and_nonempty():
    tmp = tempfile.mkdtemp(prefix="fc-test-")
    try:
        report = fc.build(outdir=tmp)
        have_font = _has_cjk_font()
        for name, spec in fc.DIAGRAMS.items():
            info = report[name]
            if have_font:
                assert info["missing_glyphs"] == [], f"{name} 缺字形：{info['missing_glyphs'][:3]}"
            else:
                print(f"[skip-glyph] {name}: 本机无中文字体，缺 {len(info['missing_glyphs'])} 个字形（已知降级）")
            png = os.path.join(tmp, f"{name}.png")
            mmd = os.path.join(tmp, f"{name}.mmd")
            txt = os.path.join(tmp, f"{name}.txt")
            for path in (png, mmd, txt):
                assert os.path.exists(path) and os.path.getsize(path) > 0, path
            assert os.path.getsize(png) > 20_000, f"{name} PNG 太小，可能渲染塌了"
            mmd_text = open(mmd, encoding="utf-8").read()
            txt_text = open(txt, encoding="utf-8").read()
            for n in spec["nodes"]:
                assert n["id"] in mmd_text, f"{name}: MMD 缺节点 {n['id']}"
                assert n["id"] in txt_text, f"{name}: TXT 缺节点 {n['id']}"
            for edge in spec["edges"]:
                a, b = edge[0], edge[1]
                assert f"{a} -->" in mmd_text or f"{a} -->" in mmd_text, (name, a, b)
                assert f"{a} -> {b}" in txt_text, f"{name}: TXT 缺边 {a}->{b}"
            for layer in spec["layers"]:
                assert layer in mmd_text and layer in txt_text, f"{name}: 缺层 {layer}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rerender_is_content_stable():
    """两次渲染的 MMD/TXT 必须逐字节一致（PNG 由 matplotlib 决定，单独校验非空）。"""
    t1 = tempfile.mkdtemp(prefix="fc-a-")
    t2 = tempfile.mkdtemp(prefix="fc-b-")
    try:
        fc.build(outdir=t1)
        fc.build(outdir=t2)
        for name in fc.DIAGRAMS:
            for ext in ("mmd", "txt"):
                a = open(os.path.join(t1, f"{name}.{ext}"), "rb").read()
                b = open(os.path.join(t2, f"{name}.{ext}"), "rb").read()
                assert a == b, f"{name}.{ext} 两次渲染不一致"
    finally:
        shutil.rmtree(t1, ignore_errors=True)
        shutil.rmtree(t2, ignore_errors=True)


def test_expected_shape():
    """结构守卫：6 个缺口、11 个提案；每个提案/缺口在图中都要有出边或入边。"""
    props = [n for n in fc.DIAGRAMS["flow_proposals"]["nodes"] if n["layer"] == 0]
    pans = [n for n in fc.DIAGRAMS["flow_proposals"]["nodes"] if n["layer"] == 1]
    assert len(props) == 6, len(props)
    assert len(pans) == 11, len(pans)
    edges = fc.DIAGRAMS["flow_proposals"]["edges"]
    touched = {a for a, b, *rest in edges} | {b for a, b, *rest in edges}
    for n in props + pans:
        assert n["id"] in touched, f"{n['id']} 在图中是孤立节点"


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
