#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语料准备：把工作区根目录下的文本文件拼成一份「小而真」的训练语料。

设计取舍：
* 用**本地文件**而不是下载数据集 —— 断网也能跑、可复现、体积可控；
* 只取根目录下的 ``*.py`` / ``*.md``（不递归进子目录），**显式排除**语料库本身，
  避免「语料里包含语料」的自指；
* 切分顺序很重要：**先在原始文本上切 train/val，再只用 train 训练 BPE**。
  否则分词器已经见过验证集文本，困惑度就不诚实了。
"""
from __future__ import annotations

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 显式排除：语料库自身（自指）、以及超大文件
EXCLUDE = {"rl_corpus.md", "rl_corpus.json", "rl_titles.txt"}
EXTS = (".py", ".md")


def collect_texts(root: str = ROOT) -> dict[str, str]:
    """返回 {相对文件名: 内容}，按文件名排序，保证可复现。"""
    out: dict[str, str] = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isfile(path) or not name.endswith(EXTS) or name in EXCLUDE:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        if text.strip():
            out[name] = text
    return out


def build(root: str = ROOT, val_frac: float = 0.05) -> tuple[str, str, dict]:
    """拼接语料并切分，返回 (train_text, val_text, meta)。"""
    files = collect_texts(root)
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
    }
    return train_text, val_text, meta


def build_and_save(outdir: str, root: str = ROOT, val_frac: float = 0.05) -> dict:
    train_text, val_text, meta = build(root, val_frac)
    os.makedirs(outdir, exist_ok=True)
    for name, text in (("train.txt", train_text), ("val.txt", val_text)):
        with open(os.path.join(outdir, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    with open(os.path.join(outdir, "corpus_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


if __name__ == "__main__":
    m = build_and_save(os.path.join(ROOT, "data"))
    print(f"[corpus] {m['n_files']} 个文件，共 {m['total_chars']} 字符")
    print(f"[corpus] train {m['train_chars']} / val {m['val_chars']} 字符")
