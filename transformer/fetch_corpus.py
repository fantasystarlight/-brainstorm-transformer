#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""准备一份**更大**的中文语料（可选步骤，给 GPU 上的缩放实验用）。

来源：`chinese-poetry` 仓库里《全唐诗》的 JSON（公有领域，GitHub raw 可直连）。
写成纯文本，一行一首（标题 + 作者 + 正文），再拼上本地已有的语料文件。

用法：
    python -m transformer.fetch_corpus --out data_poetry/poetry.txt --max-files 60
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

API = ("https://api.github.com/repos/chinese-poetry/chinese-poetry/contents/"
       "%E5%85%A8%E5%94%90%E8%AF%97")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "mini-transformer-tutorial"})
    with urllib.request.urlopen(req, timeout=30) as fh:
        return fh.read()


RAW = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%85%A8%E5%94%90%E8%AF%97/"


def list_files() -> list[str]:
    """列出《全唐诗》目录下的分卷 json 文件名，按名字排序。

    注意：API 返回的 ``download_url`` 含百分号编码（非 ASCII），直接用 urllib 会炸在
    ``'ascii' codec can't encode``；所以这里只取名字，自己拼 raw 网址。
    """
    items = json.loads(_get(API))
    names = [it["name"] for it in items
             if it["name"].endswith(".json") and it["name"].isascii()
             and not it["name"].startswith("authors")]
    return sorted(names)


def fetch(out: str, max_files: int = 60, verbose: bool = True) -> dict:
    files = list_files()[:max_files]
    parts, n_poems = [], 0
    for i, name in enumerate(files):
        try:
            data = json.loads(_get(RAW + name))
        except Exception as exc:                        # 单个文件失败不影响整体
            print(f"[fetch] 跳过 {name}: {exc}")
            continue
        for poem in data:
            title = (poem.get("title") or "").strip()
            author = (poem.get("author") or "").strip()
            paras = "".join(poem.get("paragraphs") or [])
            if not paras:
                continue
            parts.append(f"{title}　{author}\n{paras}")
            n_poems += 1
        if verbose and (i + 1) % 10 == 0:
            print(f"[fetch] {i + 1}/{len(files)} 个文件，累计 {n_poems} 首")
    text = "\n\n".join(parts)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    if verbose:
        print(f"[fetch] 写出 {out}：{n_poems} 首、{len(text)} 字符")
    return {"poems": n_poems, "chars": len(text), "files": len(files), "path": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="poetry.txt")
    ap.add_argument("--max-files", type=int, default=60)
    args = ap.parse_args()
    print(json.dumps(fetch(args.out, args.max_files), ensure_ascii=False))
