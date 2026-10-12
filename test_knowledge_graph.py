#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""knowledge_graph.py 的守卫测试。

三件事必须同时为真，否则图谱就是错的：
1. **文档锚点**：每个缺口/提案/知识点/争论节点的 anchor 仍出现在 `RL新策略头脑风暴.md`（附录 C）；
2. **语料锚点**：每个方法节点标签里的 arXiv id 真实存在于 `rl_corpus.json`（引用可回查）；
3. **产物同源**：PNG 非空且无缺字形，MMD/TXT 包含全部节点与边，重复渲染内容稳定。

渲染一律写到临时目录，绝不碰 `docs/`（避免跑测试就刷新交付物 mtime）。
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import knowledge_graph as kg


def test_doc_anchors_exist():
    assert kg.check_anchors() == [], kg.check_anchors()


def test_method_ids_exist_in_corpus():
    assert kg.check_methods() == [], kg.check_methods()


def test_method_checker_has_discriminating_power():
    """负对照：塞一个不存在的 arXiv id，检查器必须报错。"""
    kg.METHODS.append(("_bogus", "K-OPT", "Nonexistent Method 9999.99999", kg.S_OPEN))
    try:
        problems = kg.check_methods()
    finally:
        kg.METHODS.pop()
    assert any("_bogus" in p for p in problems), "id 检查器没有鉴别力"


def test_anchor_checker_has_discriminating_power():
    kg.CONCEPTS.append(kg.C("_bogus_c", "K-OPT", "x", "这句话绝不会出现在文档里-77ab"))
    try:
        missing = kg.check_anchors()
    finally:
        kg.CONCEPTS.pop()
    assert any("_bogus_c" in m for m in missing), "锚点检查器没有鉴别力"


def test_graph_is_well_formed():
    # 每个知识点都必须挂在一个真实知识域下
    for c in kg.CONCEPTS:
        assert c["domain"] in kg.DOMAIN_BY_ID, c
    # 每条关系的两端都必须存在
    for a, b, kind in kg.CONCEPT_EDGES:
        assert a in kg.CONCEPT_BY_ID and b in kg.CONCEPT_BY_ID, (a, b)
        assert kind in ("depend", "oppose", "transfer"), kind
    # 每个提案至少依赖两个知识域
    for p in kg.PROPOSALS:
        assert len(p["domains"]) >= 2, p
        assert all(d in kg.DOMAIN_BY_ID for d in p["domains"]), p
    # 每个知识域至少有 4 个知识点
    for d in kg.DOMAINS:
        n = sum(1 for c in kg.CONCEPTS if c["domain"] == d["id"])
        assert n >= 4, (d["id"], n)
    # 关键规模
    assert len(kg.DOMAINS) == 8
    assert len(kg.GAPS) == 6
    assert len(kg.PROPOSALS) == 11
    assert len(kg.CONCEPTS) >= 45
    assert len(kg.CONFLICTS) == 4
    assert len(kg.METHODS) >= 40
    # 没有孤立知识点：每个节点至少出现在一条关系里（K-XFER 允许只作为迁移源/目标）
    touched = {a for a, _b, _k in kg.CONCEPT_EDGES} | {b for _a, b, _k in kg.CONCEPT_EDGES}
    isolated = [c["id"] for c in kg.CONCEPTS if c["id"] not in touched]
    assert isolated == [], f"孤立知识点：{isolated}"


def test_render_products_consistent_and_nonempty():
    tmp = tempfile.mkdtemp(prefix="kg-test-")
    try:
        report = kg.build(outdir=tmp)
        for name, info in report.items():
            if name.endswith(".json"):
                continue
            assert info["missing_glyphs"] == [], f"{name} 缺字形：{info['missing_glyphs'][:3]}"
            png = os.path.join(tmp, f"{name}.png")
            mmd = os.path.join(tmp, f"{name}.mmd")
            txt = os.path.join(tmp, f"{name}.txt")
            for path in (png, mmd, txt):
                assert os.path.exists(path) and os.path.getsize(path) > 0, path
            assert os.path.getsize(png) > 20_000, f"{name} PNG 太小，可能渲染塌了"
            assert os.path.getsize(mmd) > 200 and os.path.getsize(txt) > 200, name
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_knowledge_plates_cover_all_concepts():
    """三张知识点图谱的节点并集必须等于全部知识点（不许有漏画的域）。"""
    tmp = tempfile.mkdtemp(prefix="kg-cov-")
    try:
        kg.build(outdir=tmp)
        texts = "\n".join(open(os.path.join(tmp, f"kg_knowledge_{s}.txt"), encoding="utf-8").read()
                          for s in "abc")
        for c in kg.CONCEPTS:
            assert f"{c['id']}:" in texts, f"知识点 {c['id']} 没被任何一张知识点图覆盖"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rerender_is_content_stable():
    t1 = tempfile.mkdtemp(prefix="kg-a-")
    t2 = tempfile.mkdtemp(prefix="kg-b-")
    try:
        kg.build(outdir=t1)
        kg.build(outdir=t2)
        names = [f"kg_knowledge_{s}" for s in "abc"] + ["kg_overview", "kg_conflicts", "kg_methods"]
        for name in names:
            for ext in ("mmd", "txt"):
                a = open(os.path.join(t1, f"{name}.{ext}"), "rb").read()
                b = open(os.path.join(t2, f"{name}.{ext}"), "rb").read()
                assert a == b, f"{name}.{ext} 两次渲染不一致"
    finally:
        shutil.rmtree(t1, ignore_errors=True)
        shutil.rmtree(t2, ignore_errors=True)


def test_json_export_matches_memory():
    tmp = tempfile.mkdtemp(prefix="kg-json-")
    try:
        kg.dump_json(os.path.join(tmp, "kg.json"))
        data = json.load(open(os.path.join(tmp, "kg.json"), encoding="utf-8"))
        assert len(data["concepts"]) == len(kg.CONCEPTS)
        assert len(data["concept_edges"]) == len(kg.CONCEPT_EDGES)
        assert {c["id"] for c in data["concepts"]} == {c["id"] for c in kg.CONCEPTS}
        assert len(data["domains"]) == 8 and len(data["proposals"]) == 11
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
