#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语料准备：把工作区里的文本文件拼成一份「小而真」的训练语料。

设计取舍：
* 用**本地文件**而不是下载数据集 —— 断网也能跑、可复现、体积可控；
* 默认**递归**扫工作区（``--root-only`` 可退回只看根目录），但有三层**显式排除**：
  ① 与「教人搭 Transformer」无关、只产生产物或自指的目录（见 ``EXCLUDE_DIRS`` /
     ``EXCLUDE_DIR_PREFIXES`` / ``_skip_dir``）；
  ② 语料库自身（``rl_corpus.*`` / ``rl_titles.txt``，否则「语料里包含语料」）；
  ③ 已被归档到 ``corpus/`` 的几份根目录大文件（见 ``ARCHIVED_ROOT_FILES``）——
     它们的**原件仍留在根目录**（用户自己还要用），只是不再进内置教程语料。
* 切分顺序很重要：**先在原始文本上切 train/val，再只用 train 训练 BPE**。
  否则分词器已经见过验证集文本，困惑度就不诚实了。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 显式排除：语料库自身（自指）
EXCLUDE_FILES = {"rl_corpus.md", "rl_corpus.json", "rl_titles.txt"}

# 已归档到 corpus/ 的根目录大文件：与「搭一个最小 Transformer」无关，
# 但用户自己要用，所以**只复制不删除**。归档动作在 ``transformer/build_corpus.py``，
# 它启动时会断言自己的归档清单与这里一致（改名不会两边漂移）。
ARCHIVED_ROOT_FILES = (
    "RL新策略头脑风暴.md",
    "knowledge_graph.py",
    "probe_compression_as_retrieval.py",
    "flowchart.py",          # 根目录那份；transformer/flowchart.py 因目录被排除而不可见
)

# 不扫的目录（相对工作区根目录的名字）
EXCLUDE_DIRS = {
    "transformer",           # 教程代码本身（怕自指；它的 README 由 EXTRA_FILES 单独收）
    "docs",                  # 流程图 / 长文档产物
    "slices",                # rl_corpus 的切片
    "runs",                  # 训练产物
    "logs",                  # 运行日志
    "__pycache__",
    ".git",
    ".github",
    ".pytest_cache",         # pytest 会重写它，进语料会让语料不确定
}

# 只在这些子目录里扫（白名单）：这样 corpus/ 下的归档语料能重新成为「训练数据的来源」，
# 而不再需要把 bytes 级快照散落在根目录。加新来源就在这里加一个目录名。
SOURCE_DIRS = {"corpus"}

# 白名单之外、但仍然要收进来的具体文件（相对工作区根目录）。
# 为什么收教程自己的 README：它是这份文档里结构最好的中文长文本，
# 而且「用本项目的文档当语料」比用无关文件更自洽。
EXTRA_FILES = ("transformer/README.md",)
# data*/ —— 生成的 token 流与语料快照，不是语料「来源」
EXCLUDE_DIR_PREFIXES = ("data",)
EXTS = (".py", ".md")


def _skip_dir(name: str) -> bool:
    """这个子目录要不要跳过。

    * 隐藏目录一律跳过（.git / .github / .pytest_cache 都是元数据或缓存）；
    * 在 ``EXCLUDE_DIRS`` 里的跳过；
    * ``data*`` 跳过（生成的 token 流是**产物**，不是语料来源）；
    """
    return (name.startswith(".") or name in EXCLUDE_DIRS
            or name.startswith(EXCLUDE_DIR_PREFIXES))


def iter_files(root: str = ROOT, include_subdirs: bool = True) -> list[str]:
    """列出要进语料的文件：相对路径（posix 风格）、已排序、可复现。

    ``include_subdirs=False`` 时只扫根目录（改造前的旧行为）。
    """
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if not include_subdirs:
            dirnames[:] = []                      # 只扫根目录
        elif os.path.abspath(dirpath) == os.path.abspath(root):
            # 根目录这一层：**只往下钻白名单目录**（注意不能写成 prefixes=SOURCE_DIRS，
            # 那样会让 corpus/ 和 corpus_big/ 这种前缀相同但不相等的目录互相干扰）
            dirnames[:] = sorted(d for d in dirnames if d in SOURCE_DIRS)
        else:
            dirnames[:] = sorted(d for d in dirnames if not _skip_dir(d))
        rel_dir = os.path.relpath(dirpath, root)
        for name in filenames:
            if not name.endswith(EXTS) or name in EXCLUDE_FILES:
                continue
            if rel_dir == "." and name in ARCHIVED_ROOT_FILES:
                continue
            rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            found.append(rel.replace(os.sep, "/"))
    # 白名单外的显式文件（例如 transformer/README.md）单独补进来
    for rel in EXTRA_FILES:
        path = os.path.join(root, rel)
        if os.path.isfile(path) and rel not in found:
            found.append(rel.replace(os.sep, "/"))
    return sorted(found)


def collect_texts(root: str = ROOT, include_subdirs: bool = True) -> dict[str, str]:
    """返回 {相对路径: 内容}，按相对路径排序，保证可复现。"""
    out: dict[str, str] = {}
    for rel in iter_files(root, include_subdirs=include_subdirs):
        path = os.path.join(root, rel)
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        if text.strip():
            out[rel] = text
    return out


def build(root: str = ROOT, val_frac: float = 0.05,
          include_subdirs: bool = True) -> tuple[str, str, dict]:
    """拼接语料并切分，返回 (train_text, val_text, meta)。"""
    files = collect_texts(root, include_subdirs=include_subdirs)
    assert files, f"{root} 下没找到任何 .py/.md 文件"
    # 每篇加一行分隔标题（让模型学到「文档边界」这类结构），再用一个罕见符号分隔
    joined = "\n\n".join(f"# ==== {name} ====\n{text}" for name, text in files.items())
    cut = int(len(joined) * (1 - val_frac))
    # 尽量切在换行处，避免把一行代码劈成两半影响验证集统计
    nl = joined.find("\n", cut)
    cut = nl if 0 <= nl < cut + 200 else cut
    train_text, val_text = joined[:cut], joined[cut:]
    meta = {
        "files": {name: len(text) for name, text in files.items()},
        "n_files": len(files),
        "total_chars": len(joined),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "val_frac": val_frac,
        "train_sha256": hashlib.sha256(train_text.encode()).hexdigest(),
        "split": "train=前段 / val=后段（在原始文本上先切分，再训 BPE）",
        "scan": "递归（含子目录）" if include_subdirs else "仅根目录",
        "excluded_dirs": sorted(EXCLUDE_DIRS) + [f"{p}*/" for p in EXCLUDE_DIR_PREFIXES],
        "excluded_files": sorted(EXCLUDE_FILES),
        "archived_excluded": list(ARCHIVED_ROOT_FILES),
    }
    return train_text, val_text, meta


def build_and_save(outdir: str, root: str = ROOT, val_frac: float = 0.05,
                   include_subdirs: bool = True) -> dict:
    train_text, val_text, meta = build(root, val_frac, include_subdirs=include_subdirs)
    os.makedirs(outdir, exist_ok=True)
    for name, text in (("train.txt", train_text), ("val.txt", val_text)):
        with open(os.path.join(outdir, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    with open(os.path.join(outdir, "corpus_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="把工作区文本拼成训练语料并切 train/val")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--include-subdirs", dest="include_subdirs", action="store_true",
                     default=True, help="递归扫描子目录（默认）")
    grp.add_argument("--root-only", dest="include_subdirs", action="store_false",
                     help="只扫工作区根目录（改造前的旧行为）")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data"))
    args = ap.parse_args()
    m = build_and_save(args.outdir, root=ROOT, val_frac=args.val_frac,
                       include_subdirs=args.include_subdirs)
    print(f"[corpus] {m['n_files']} 个文件，共 {m['total_chars']} 字符（{m['scan']}）")
    print(f"[corpus] train {m['train_chars']} / val {m['val_chars']} 字符")
