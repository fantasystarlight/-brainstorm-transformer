#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把工作区根目录的 4 份「大且与搭 Transformer 无关」的内容归档到 corpus/，再重建 data/。

问题：``transformer/corpus.py`` 把工作区里的 ``*.py`` / ``*.md`` 当教程语料，而根目录有 4 份
文件占了语料的一半以上、却和「教人搭一个最小 Transformer」毫无关系：

    ==============================  =========  ==========================================
    文件                             字符数    性质
    ==============================  =========  ==========================================
    RL新策略头脑风暴.md              ~28 K      606 行 RL 调研笔记（10 个二级标题）
    knowledge_graph.py               ~36 K      知识图谱构建/渲染
    probe_compression_as_retrieval.py ~10 K      压缩即检索的探针实验
    flowchart.py（根目录那份）        ~19 K      实验流程图生成器
    ==============================  =========  ==========================================

直接删不行（用户自己还要用），简单截断更不行（内容就没了）。所以：

1. **复制**（不是移动；根目录原件一个字节都不动）到 ``corpus/``，并按**文档结构**拆分，
   不减一个字符：

   * ``RL新策略头脑风暴.md`` → ``corpus/rl_notes/NN_标题.md``：按 ``^## `` 二级标题切片，
     每片**保留自己的标题行**；首个 ``##`` 之前的前言（H1 + 五分钟版）单独成 ``00_*.md``；
   * ``knowledge_graph.py`` → ``corpus/src/knowledge_graph.py``（整份）
   * ``probe_compression_as_retrieval.py`` → ``corpus/src/probe_compression_as_retrieval.py``
   * ``flowchart.py`` → ``corpus/src/flowchart_root.py``（改名以免与 transformer/flowchart.py 混淆）

2. 写 ``corpus/STATS.json``：每个组件的字符数 / 行数 / sha256，以及「移出前 / 移出后」
   根目录语料总量的对比（两条规则都写清楚，数字可复算）。

3. 调 ``transformer.data.prepare(vocab_size=2048, force=True)`` 重建 ``data/``。

「移出语料」的确切含义：**原件仍在根目录**，但 ``transformer/corpus.py`` 把它们列进
``ARCHIVED_ROOT_FILES``，因此不再进内置教程语料；而 ``corpus/`` 目录本身也在排除清单里，
所以归档内容只是「存着备用」，不会悄悄流回语料。—— 想拿它们单独当语料时显式喂文件即可：
``python -m transformer.data --files corpus/src/knowledge_graph.py``（见 README §10）。

幂等：所有落盘都走 ``_write_if_changed``（内容相同就连 mtime 都不动），
``corpus/rl_notes/`` 里上一轮生成、这一轮不再生成的切片会被删掉（只删匹配 ``NN_*.md`` 的文件），
所以重复跑不会追加、不会重复，也不会留下过期切片。

用法::

    PYTHONPATH=. python -m transformer.build_corpus              # 归档 + 重建 data/
    PYTHONPATH=. python -m transformer.build_corpus --check      # 只打印计划，不写盘
    PYTHONPATH=. python -m transformer.build_corpus --no-data    # 归档 + 统计，不动 data/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile

from . import corpus

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS_DIR = os.path.join(ROOT, "corpus")
RL_NOTES_DIR = os.path.join(CORPUS_DIR, "rl_notes")
SRC_DIR = os.path.join(CORPUS_DIR, "src")
STATS_PATH = os.path.join(CORPUS_DIR, "STATS.json")

# 归档计划：根目录文件 → (目标子目录, 目标文件名 或 None 表示按结构切片)
ARCHIVE_PLAN = {
    "RL新策略头脑风暴.md": (RL_NOTES_DIR, None),
    "knowledge_graph.py": (SRC_DIR, "knowledge_graph.py"),
    "probe_compression_as_retrieval.py": (SRC_DIR, "probe_compression_as_retrieval.py"),
    "flowchart.py": (SRC_DIR, "flowchart_root.py"),
}
RL_NOTES_SOURCE = "RL新策略头脑风暴.md"
# 上一轮生成、这一轮可能过期的切片名（只在这个形状内清理）
RL_NOTES_NAME_RE = re.compile(r"^\d+_.*\.md$")


# ======================================================================================
# 落盘：读用普通 open，写先写临时文件再 os.replace；内容没变就不碰（write-if-changed）
# ======================================================================================
def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _write_if_changed(path: str, text: str) -> bool:
    """内容不同才落盘：临时文件 → os.replace。返回是否真的写了。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            if fh.read() == text:
                return False
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return True


def _write_json_if_changed(path: str, obj) -> bool:
    return _write_if_changed(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


# ======================================================================================
# 按文档结构切分 markdown
# ======================================================================================
H2_RE = re.compile(r"^## .*$", re.M)
H1_RE = re.compile(r"^# .*$", re.M)
_BAD_NAME_CHARS = re.compile(r'[\\/:*?"<>|\s]+')
_LEADING_NUM = re.compile(r"^\d+\s*[.、)]\s*")


def slugify(title: str, maxlen: int = 40) -> str:
    """标题 → 安全文件名片段（去掉编号前缀、非法字符、折叠空白），确定性且可读。"""
    s = _LEADING_NUM.sub("", title.strip())          # 前面已有 NN_ 序号，标题里不重复
    s = _BAD_NAME_CHARS.sub("_", s)
    s = re.sub(r"_{2,}", "_", s).strip("._-")
    if len(s) > maxlen:
        s = s[:maxlen].strip("._-")
    return s or "untitled"


def split_rl_notes(text: str) -> list[tuple[str, str]]:
    """按 ``^## `` 切成 [(标题, 该片全文)]；每片含自己的 ``## `` 标题行。

    首个二级标题之前的前言单独成一片（标题取文档 H1）。
    硬断言「切片拼起来 == 原文」——这条不变量保证归档不丢字符。
    """
    marks = list(H2_RE.finditer(text))
    pieces: list[tuple[str, str]] = []
    if marks:
        preamble = text[:marks[0].start()]
        if preamble.strip():
            h1 = H1_RE.search(preamble)
            pieces.append((h1.group(0)[2:].strip() if h1 else "前言", preamble))
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            pieces.append((m.group(0)[3:].strip(), text[m.start():end]))
    else:
        h1 = H1_RE.search(text)
        pieces.append((h1.group(0)[2:].strip() if h1 else "全文", text))
    assert pieces, "markdown 是空的？"
    assert "".join(p for _, p in pieces) == text, "切片拼接 != 原文，归档会丢字符"
    return pieces


def archive_rl_notes(path: str, check: bool, components: list[dict]) -> list[str]:
    """把 RL 笔记按二级标题切片写进 corpus/rl_notes/。返回过期文件（已删或待删）。"""
    text = _read(path)
    pieces = split_rl_notes(text)
    width = max(2, len(str(len(pieces) - 1)))
    expected: dict[str, str] = {}
    for i, (title, body) in enumerate(pieces):
        name = f"{i:0{width}d}_{slugify(title)}.md"
        expected[name] = body
        m = H2_RE.search(body)                      # 前言片没有 ## 标题，就用 H1 当标题
        heading = m.group(0)[3:].strip() if m else title
        components.append({
            "path": os.path.relpath(os.path.join(RL_NOTES_DIR, name), ROOT).replace(os.sep, "/"),
            "group": "rl_notes",
            "source": os.path.relpath(path, ROOT),
            "heading": heading,
            "chars": len(body),
            "lines": body.count("\n") + (0 if body.endswith("\n") else 1),
            "sha256": hashlib.sha256(body.encode()).hexdigest(),
        })
    stale = [n for n in sorted(os.listdir(RL_NOTES_DIR))
             if RL_NOTES_NAME_RE.match(n) and n not in expected] if os.path.isdir(RL_NOTES_DIR) else []
    if check:
        return stale
    for name, body in expected.items():
        wrote = _write_if_changed(os.path.join(RL_NOTES_DIR, name), body)
        print(f"[corpus] {'写入' if wrote else '未变'} corpus/rl_notes/{name}（{len(body)} 字符）")
    for name in stale:                       # 过期切片：只删本脚本自己命名形状的文件
        target = os.path.join(RL_NOTES_DIR, name)
        assert RL_NOTES_NAME_RE.match(os.path.basename(target)), target
        os.unlink(target)
        print(f"[corpus] 删除过期切片 corpus/rl_notes/{name}")
    return stale


def archive_whole_file(path: str, outdir: str, outname: str, check: bool,
                       components: list[dict]) -> None:
    """整份复制（内容不变），同样 write-if-changed。"""
    text = _read(path)
    dst = os.path.join(outdir, outname)
    components.append({
        "path": os.path.relpath(dst, ROOT).replace(os.sep, "/"),
        "group": "src",
        "source": os.path.relpath(path, ROOT),
        "heading": "",
        "chars": len(text),
        "lines": text.count("\n") + (0 if text.endswith("\n") else 1),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    })
    if check:
        return
    wrote = _write_if_changed(dst, text)
    print(f"[corpus] {'写入' if wrote else '未变'} {os.path.relpath(dst, ROOT)}（{len(text)} 字符）")


# ======================================================================================
# 「移出前 / 移出后」对比
# ======================================================================================
def _joined_chars(files: dict[str, str]) -> int:
    """复算 corpus.build() 的 total_chars（含 '# ==== name ====' 分隔标题）。"""
    return len("\n\n".join(f"# ==== {name} ====\n{text}" for name, text in files.items()))


def legacy_root_files(root: str = ROOT) -> dict[str, str]:
    """改造 corpus.py 之前的取法：只看根目录 *.py/*.md，排除自指三件套。"""
    out: dict[str, str] = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isfile(path) or not name.endswith(corpus.EXTS) or name in corpus.EXCLUDE_FILES:
            continue
        try:
            text = _read(path)
        except (UnicodeDecodeError, OSError):
            continue
        if text.strip():
            out[name] = text
    return out


def corpus_stats(before: dict[str, str], after: dict[str, str], components: list[dict],
                 scan_label: str) -> dict:
    groups: dict[str, dict] = {}
    for c in components:
        g = groups.setdefault(c["group"], {"n_files": 0, "chars": 0})
        g["n_files"] += 1
        g["chars"] += c["chars"]
    before_chars = sum(len(t) for t in before.values())
    after_chars = sum(len(t) for t in after.values())
    archived_chars = sum(c["chars"] for c in components)
    # 硬不变量：移出前后的纯文本差 == 归档内容的字符数（一句话证明「内容没丢、也没多移」）
    assert before_chars - after_chars == archived_chars, (
        f"账不平：移出前 {before_chars} − 移出后 {after_chars} != 归档 {archived_chars}")
    return {
        "generated_by": "transformer/build_corpus.py",
        "root": ROOT,
        "archive_policy": ("只复制不删除：根目录原件保留；归档内容按文档结构切分进 corpus/；"
                           "corpus.py 通过 ARCHIVED_ROOT_FILES 把这 4 份排除出内置教程语料，"
                           "corpus/ 目录本身也在排除清单里（不会悄悄回流）。"),
        "components": sorted(components, key=lambda c: c["path"]),
        "groups": {k: groups[k] for k in sorted(groups)},
        "corpus_total_chars": archived_chars,
        "corpus_total_files": len(components),
        "root_corpus": {
            "before": {
                "rule": "只看根目录 *.py/*.md，排除 rl_corpus.{md,json}/rl_titles.txt（含这 4 份大文件）",
                "n_files": len(before),
                "sum_file_chars": before_chars,
                "total_chars": _joined_chars(before),
                "files": {k: len(v) for k, v in sorted(before.items())},
            },
            "after": {
                "rule": f"corpus.collect_texts()：{scan_label} + 排除目录/自指/已归档 4 份",
                "n_files": len(after),
                "sum_file_chars": after_chars,
                "total_chars": _joined_chars(after),
                "files": {k: len(v) for k, v in sorted(after.items())},
            },
            "delta_total_chars": _joined_chars(after) - _joined_chars(before),
            "delta_pct": round(100.0 * (_joined_chars(after) - _joined_chars(before))
                               / max(1, _joined_chars(before)), 2),
        },
        "note": ("corpus/ 是归档区、不在内置教程语料里；想把它当语料用要显式喂文件，"
                 "例如 PYTHONPATH=. python -m transformer.data --files corpus/src/knowledge_graph.py。"),
    }


# ======================================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="归档根目录大文件到 corpus/，再重建 data/")
    ap.add_argument("--check", action="store_true", help="只打印计划与统计，不写任何文件")
    ap.add_argument("--no-data", action="store_true", help="不动 data/（只归档 + 写 STATS.json）")
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args(argv)

    # 归档清单是本脚本的硬事实；与 corpus.py 的排除清单必须一致（改名不漂移）
    assert set(ARCHIVE_PLAN) == set(corpus.ARCHIVED_ROOT_FILES), (
        f"归档清单与 corpus.ARCHIVED_ROOT_FILES 不一致：\n"
        f"  build_corpus: {sorted(ARCHIVE_PLAN)}\n  corpus.py:    {sorted(corpus.ARCHIVED_ROOT_FILES)}")
    missing = [n for n in ARCHIVE_PLAN if not os.path.isfile(os.path.join(ROOT, n))]
    assert not missing, f"根目录缺少源文件（无法归档）：{missing}"

    components: list[dict] = []
    stale = archive_rl_notes(os.path.join(ROOT, RL_NOTES_SOURCE), args.check, components)
    for src, (outdir, outname) in ARCHIVE_PLAN.items():
        if outname is None:
            continue
        archive_whole_file(os.path.join(ROOT, src), outdir, outname, args.check, components)

    before = legacy_root_files(ROOT)
    after = corpus.collect_texts(ROOT)                  # 默认即递归（含子目录）
    stats = corpus_stats(before, after, components, scan_label="递归（含子目录）")
    stats["stale_rl_notes_removed"] = stale
    b, a = stats["root_corpus"]["before"], stats["root_corpus"]["after"]
    print(f"[corpus] 归档 {stats['corpus_total_files']} 个文件，共 {stats['corpus_total_chars']} 字符"
          f"（rl_notes {stats['groups'].get('rl_notes', {}).get('n_files', 0)} / "
          f"src {stats['groups'].get('src', {}).get('n_files', 0)}）")
    print(f"[corpus] 根目录语料 total_chars：移出前 {b['total_chars']}（{b['n_files']} 文件）"
          f" → 移出后 {a['total_chars']}（{a['n_files']} 文件），"
          f"{stats['root_corpus']['delta_total_chars']:+d}（{stats['root_corpus']['delta_pct']:+.1f}%）")
    if args.check:
        print("[corpus] --check：没有写任何文件")
        return 0
    wrote = _write_json_if_changed(STATS_PATH, stats)
    print(f"[corpus] {'写入' if wrote else '未变'} corpus/STATS.json")

    if args.no_data:
        print("[corpus] --no-data：跳过 data/ 重建")
        return 0
    from . import data as data_mod                      # 懒引入：只有真要重建时才付 torch 的启动代价
    meta = data_mod.prepare(vocab_size=args.vocab_size, force=True, val_frac=args.val_frac)
    print(f"[data] meta：n_files={meta['n_files']} total_chars={meta['total_chars']} "
          f"train_tokens={meta['train_tokens']} val_tokens={meta['val_tokens']} "
          f"tokenizer={meta['tokenizer']} vocab={meta['vocab_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
