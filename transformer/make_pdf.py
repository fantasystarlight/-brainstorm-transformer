#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 ``transformer/README.md`` 排成 PDF（自解析 Markdown，不依赖 pandoc / LaTeX / 联网）。

    PYTHONPATH=. python -m transformer.make_pdf                # -> transformer/README.pdf
    PYTHONPATH=. python -m transformer.make_pdf -o /tmp/a.pdf
    PYTHONPATH=. python -m transformer.make_pdf --dump-blocks   # 只看解析结果

为什么是这条路
--------------
本机（WSL2）没有 pandoc / LaTeX / 浏览器自动化库，pip 也大概率连不上网。但 **Windows 侧有
Edge**，它的 ``--headless --print-to-pdf`` 就是一台完整的 HTML→PDF 排版引擎。于是：

    README.md  --(本文件自带的小型 Markdown 解析器)-->  HTML+打印 CSS
               --(Edge headless)-->                    PDF

因为 HTML 和图片都必须放在 Windows 能访问的路径下（``/mnt/c/...``），本脚本会：

1. 在 Windows 临时目录下建一个**构建目录**（HTML / 图片 / Edge 产物都放那里）；
2. 解析 Markdown、重写图片路径、生成打印友好的 HTML + CSS；
3. 调 Edge 打印成 PDF；
4. **只把 PDF 复制回仓库**（``transformer/README.pdf``），中间产物一律不进 ``docs/``。

支持的 Markdown 子集（README 里实际用到的全部）：ATX 标题、段落、有序/无序列表（含嵌套）、
GFM 表格（含 ``\\|`` 转义）、``` 围栏代码块、行内 ``code``、``**粗体**``、``*斜体*``、
``![图](路径)``、``[链接](路径)``、``>`` 引用、``---`` 分隔线，以及单独一行的
``*图 N　……*`` 图注（识别成图注后渲染成图片下方的浅灰小字，**不再**当普通段落输出）。

``tutorial.py`` 出的 PNG 顶部本来就画了一行同样的图注（它把 caption 烧进了图里）。
为了不让每张图出现两次图注，本脚本在**构建目录的副本**上把那行裁掉（``docs/`` 不动，
判据不满足就原样保留），可用 ``--keep-baked-captions`` 关掉这个裁剪。

排版约定：正文微软雅黑、等宽用带 CJK 的 Consolas+雅黑（README 的代码注释里有中文，
纯西文等宽字体在 PDF 里会变豆腐块）；A4 / 18mm 边距 / 行高 1.6；``h2`` 前分页（第一个除外）；
表格、代码块、图片、引用都 ``break-inside: avoid``，不允许跨页断裂；封面只列 h2 目录
（``--toc-depth 3`` 可加上 h3），目录贴在封面底部。

PDF 里带 CreationDate 和一个内容 ID，所以严格逐字节的"幂等"要额外做归一化：本脚本默认
把这两处等长替换成固定值（``--no-normalize`` 关掉），这样**正文和图没变时重复运行不会产生
无意义的改写**，侧栏里已打开的 PDF 不会因为 mtime 变化而失效。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_MD = REPO_ROOT / "transformer" / "README.md"
DEFAULT_OUT = REPO_ROOT / "transformer" / "README.pdf"

#: 封面标题 / 副标题的兜底值（正常情况下从 README 的 H1 推导）
FALLBACK_TITLE = "从零手搭一个最小 Transformer"
#: 封面副标题：**不能与正文首段重复**（正文首段就是「目标读者：……」），所以换一句说法
SUBTITLE = "写给第一次亲手实现 Transformer 的人"

#: Edge（或 Chrome）可执行文件候选，按顺序取第一个存在的。
BROWSER_CANDIDATES = (
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe",
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
)

#: 构建目录名（放在 Windows 临时目录下）
BUILD_DIRNAME = "dshpdf_transformer"

PRINT_TIMEOUT = 240  # 秒；Edge 打印整本 README 通常 10~30 秒

# --------------------------------------------------------------------------------------
# 一、Markdown -> 块序列
# --------------------------------------------------------------------------------------

_RE_FENCE = re.compile(r"^\s*```+\s*([A-Za-z0-9_+#.-]*)\s*$")
_RE_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_RE_HR = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_RE_UL = re.compile(r"^(\s*)([-*+])\s+(.*)$")
_RE_OL = re.compile(r"^(\s*)(\d{1,3})[.)]\s+(.*)$")
_RE_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]*-{2,}[\s:|-]*\|?\s*$")
_RE_IMG_ONLY = re.compile(r"^!\[([^\]]*)\]\(\s*([^)\s]+?)(?:\s+\"[^\"]*\")?\s*\)$")
_RE_CAPTION = re.compile(r"^\*(图\s*\d+.*)\*$")
_RE_BLOCKQUOTE = re.compile(r"^\s*>\s?")

#: CJK 及全角标点：这些字符之间**不插空格**（中文换行不是空格）
_CJK = "\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef\u3000-\u303f\u2014\u2026"
_RE_IS_CJK = re.compile(f"[{_CJK}]")
_NO_SPACE_BEFORE = "，。、；：）」』】》！？…·%"


def _join_lines(parts: list[str]) -> str:
    """把被源码折行的同一段落拼回一行：中文直接接，西文之间补一个空格。"""
    out = ""
    for raw in parts:
        piece = raw.strip()
        if not piece:
            continue
        if out and _needs_space(out[-1], piece[0]):
            out += " "
        out += piece
    return out


def _needs_space(last: str, first: str) -> bool:
    if _RE_IS_CJK.match(last) or _RE_IS_CJK.match(first):
        return False
    if first in _NO_SPACE_BEFORE:
        return False
    if last in "（「『【《<":
        return False
    return bool(
        last.isalnum()
        or last in "`)]}>"
        or first.isalnum()
        or first in "`([*$"
    )


def _split_row(row: str) -> list[str]:
    """按未转义的 ``|`` 切表格行；``\\|`` 还原成字面量 ``|``。"""
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|") and not row.endswith("\\|"):
        row = row[:-1]
    cells, cur, esc = [], [], False
    for ch in row:
        if esc:
            cur.append(ch)
            esc = False
        elif ch == "\\":
            cur.append(ch)
            esc = True
        elif ch == "|":
            cells.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur))
    return [c.strip().replace("\\|", "|") for c in cells]


def _starts_block(lines: list[str], i: int) -> bool:
    """lines[i] 是否是一个新块的开头（用于判断段落何时结束）。"""
    line = lines[i]
    if _RE_FENCE.match(line) or _RE_HEADING.match(line) or _RE_HR.match(line):
        return True
    if line.lstrip().startswith(">"):
        return True
    if _RE_UL.match(line) or _RE_OL.match(line):
        return True
    if (
        line.lstrip().startswith("|")
        and i + 1 < len(lines)
        and "|" in lines[i + 1]
        and _RE_TABLE_SEP.match(lines[i + 1])
    ):
        return True
    return False


def _parse_list(lines: list[str], i: int) -> tuple[dict, int]:
    """从 lines[i]（已知是列表项）开始解析，返回 (块, 下一个下标)。支持嵌套。"""
    ordered = _RE_OL.match(lines[i]) is not None
    base = len((_RE_OL.match(lines[i]) or _RE_UL.match(lines[i])).group(1))
    items: list[dict] = []
    n = len(lines)

    while i < n:
        line = lines[i]
        if not line.strip():
            j = i
            while j < n and not lines[j].strip():
                j += 1
            # 空行后面仍是缩进的续行 -> 属于当前列表；否则列表结束
            if j < n and (len(lines[j]) - len(lines[j].lstrip())) > base:
                i = j
                continue
            break

        mo, mu = _RE_OL.match(line), _RE_UL.match(line)
        if mo or mu:
            ind = len((mo or mu).group(1))
            same_kind = (mo is not None) if ordered else (mu is not None)
            if ind > base:
                if not items:
                    break
                sub, i = _parse_list(lines, i)
                items[-1]["children"].append(sub)
                continue
            if ind < base or not same_kind:
                break
            items.append({"text": (mo or mu).group(3).strip(), "children": []})
            i += 1
            continue

        if items and (len(line) - len(line.lstrip())) > base:
            items[-1]["text"] = _join_lines([items[-1]["text"], line])
            i += 1
            continue
        break

    return {"t": "ol" if ordered else "ul", "items": items}, i


def split_blocks(text: str) -> list[dict]:
    """Markdown 文本 -> 块序列（dict 列表）。"""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[dict] = []
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        # ``` 围栏代码块
        fence = _RE_FENCE.match(line)
        if fence:
            lang = fence.group(1)
            i += 1
            buf: list[str] = []
            while i < n and not _RE_FENCE.match(lines[i]):
                buf.append(lines[i])
                i += 1
            i += 1  # 吃掉收尾 ```
            while buf and not buf[-1].strip():
                buf.pop()
            blocks.append({"t": "code", "lang": lang, "text": "\n".join(buf)})
            continue

        # ATX 标题
        m = _RE_HEADING.match(line)
        if m:
            blocks.append({"t": "h", "level": len(m.group(1)), "text": m.group(2).strip()})
            i += 1
            continue

        # --- 分隔线
        if _RE_HR.match(line):
            blocks.append({"t": "hr"})
            i += 1
            continue

        # > 引用（内部空行分段）
        if line.lstrip().startswith(">"):
            buf = []
            while i < n and lines[i].lstrip().startswith(">"):
                buf.append(_RE_BLOCKQUOTE.sub("", lines[i]))
                i += 1
            groups, cur = [], []
            for q in buf:
                if q.strip():
                    cur.append(q)
                elif cur:
                    groups.append(_join_lines(cur))
                    cur = []
            if cur:
                groups.append(_join_lines(cur))
            blocks.append({"t": "quote", "paras": groups})
            continue

        # 列表
        if _RE_UL.match(line) or _RE_OL.match(line):
            blk, i = _parse_list(lines, i)
            blocks.append(blk)
            continue

        # GFM 表格
        if (
            line.lstrip().startswith("|")
            and i + 1 < n
            and "|" in lines[i + 1]
            and _RE_TABLE_SEP.match(lines[i + 1])
        ):
            header = _split_row(line)
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            blocks.append({"t": "table", "header": header, "rows": rows})
            continue

        # 独占一行的图片
        m = _RE_IMG_ONLY.match(line.strip())
        if m:
            blocks.append({"t": "img", "alt": m.group(1), "src": m.group(2)})
            i += 1
            continue

        # 段落
        buf = [line.strip()]
        i += 1
        while i < n and lines[i].strip() and not _starts_block(lines, i):
            buf.append(lines[i])
            i += 1
        blocks.append({"t": "p", "text": _join_lines(buf)})

    return blocks


# --------------------------------------------------------------------------------------
# 二、行内元素
# --------------------------------------------------------------------------------------

_RE_IMG_INLINE = re.compile(r"!\[([^\]]*)\]\(\s*([^)\s]+?)(?:\s+\"[^\"]*\")?\s*\)")
_RE_CODE = re.compile(r"`([^`]+?)`")
_RE_LINK = re.compile(r"\[([^\]]+?)\]\(\s*([^)\s]+?)(?:\s+\"[^\"]*\")?\s*\)")
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_RE_ITALIC = re.compile(r"\*([^*\n]+?)\*")


def inline(text: str, image_map: dict[str, str] | None = None) -> str:
    """行内 Markdown -> HTML。先整体转义，再按 图 -> 代码 -> 链接 -> 粗体 -> 斜体 依次替换。"""
    s = html.escape(text, quote=False)
    image_map = image_map or {}

    def _img(m: re.Match) -> str:
        src = image_map.get(m.group(2), m.group(2))
        return f'<img class="inline-img" src="{html.escape(src, quote=True)}" alt="{m.group(1)}">'

    s = _RE_IMG_INLINE.sub(_img, s)
    s = _RE_CODE.sub(lambda m: f"<code>{m.group(1)}</code>", s)
    s = _RE_LINK.sub(
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>', s
    )
    s = _RE_BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", s)
    s = _RE_ITALIC.sub(lambda m: f"<em>{m.group(1)}</em>", s)
    return s


# --------------------------------------------------------------------------------------
# 三、块 -> HTML
# --------------------------------------------------------------------------------------

def _render_list(blk: dict, image_map: dict[str, str]) -> str:
    tag = blk["t"]
    out = [f"<{tag}>"]
    for it in blk["items"]:
        out.append(f"<li>{inline(it['text'], image_map)}")
        for child in it["children"]:
            out.append(_render_list(child, image_map))
        out.append("</li>")
    out.append(f"</{tag}>")
    return "".join(out)


def render_blocks(blocks: list[dict], image_map: dict[str, str], toc_depth: int = 2) -> tuple[str, dict]:
    """块序列 -> 正文 HTML。同时返回文档元信息（封面标题、目录、图数）。"""
    meta: dict = {"title": None, "tag": None, "toc": [], "n_img": 0}
    out: list[str] = []
    i, n = 0, len(blocks)
    first_h2 = True

    while i < n:
        b = blocks[i]
        kind = b["t"]

        if kind == "h":
            if b["level"] == 1:
                # H1 上封面，正文里不再重复
                m = re.match(r"^(.*?)[（(]([^）)]*)[）)]\s*$", b["text"])
                if meta["title"] is None:
                    meta["title"] = (m.group(1).strip() if m else b["text"])
                    meta["tag"] = m.group(2).strip() if m else None
                i += 1
                continue
            cls = ""
            if b["level"] <= toc_depth:
                meta["toc"].append((b["level"], b["text"]))
            if b["level"] == 2 and first_h2:
                cls = ' class="first-h2"'
                first_h2 = False
            out.append(f"<h{b['level']}{cls}>{inline(b['text'], image_map)}</h{b['level']}>")
            i += 1
            continue

        if kind == "hr":
            # 紧跟着 h2 的 --- 是多余的：h2 自己就会另起一页
            nxt = blocks[i + 1] if i + 1 < n else None
            if not (nxt and nxt["t"] == "h" and nxt["level"] == 2):
                out.append("<hr>")
            i += 1
            continue

        if kind == "code":
            out.append(f"<pre><code>{html.escape(b['text'], quote=False)}</code></pre>")
            i += 1
            continue

        if kind == "quote":
            inner = "".join(f"<p>{inline(p, image_map)}</p>" for p in b["paras"])
            out.append(f"<blockquote>{inner}</blockquote>")
            i += 1
            continue

        if kind in ("ul", "ol"):
            out.append(_render_list(b, image_map))
            i += 1
            continue

        if kind == "table":
            meta_h = "".join(f"<th>{inline(c, image_map)}</th>" for c in b["header"])
            body = "".join(
                "<tr>" + "".join(f"<td>{inline(c, image_map)}</td>" for c in row) + "</tr>"
                for row in b["rows"]
            )
            out.append(
                f"<table><thead><tr>{meta_h}</tr></thead><tbody>{body}</tbody></table>"
            )
            i += 1
            continue

        if kind == "img":
            meta["n_img"] += 1
            nxt = blocks[i + 1] if i + 1 < n else None
            cap = None
            if nxt and nxt["t"] == "p":
                cm = _RE_CAPTION.match(nxt["text"].strip())
                if cm:
                    cap = cm.group(1)
                    i += 1
            src = image_map.get(b["src"])
            if src is None:
                out.append(
                    f'<figure class="missing-fig"><div class="missing">'
                    f"[缺图] {html.escape(b['src'])}</div>"
                    f"<figcaption>{html.escape(cap or b['alt'])}</figcaption></figure>"
                )
            else:
                figcap = f"<figcaption>{inline(cap, image_map)}</figcaption>" if cap else ""
                out.append(
                    f'<figure><img src="{html.escape(src, quote=True)}" '
                    f'alt="{html.escape(b["alt"], quote=True)}">{figcap}</figure>'
                )
            i += 1
            continue

        # 段落（含单独的图注行）
        m = _RE_CAPTION.match(b["text"].strip())
        if m:
            out.append(f'<p class="caption">{inline(m.group(1), image_map)}</p>')
        else:
            out.append(f"<p>{inline(b['text'], image_map)}</p>")
        i += 1

    return "\n".join(out), meta


# --------------------------------------------------------------------------------------
# 四、HTML + 打印 CSS
# --------------------------------------------------------------------------------------

CSS = r"""
@page { size: A4; margin: 18mm; }

html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  margin: 0;
  font-family: "Microsoft YaHei", "SimSun", sans-serif;
  font-size: 10.5pt;
  line-height: 1.6;
  color: #1f2328;
  orphans: 2; widows: 2;
}

/* ---------- 封面 ---------- */
.cover {
  break-after: page; page-break-after: always;
  min-height: 238mm;
  display: flex; flex-direction: column;
}
.cover-head { margin-top: 22mm; text-align: center; }
.cover-rule {
  width: 42mm; height: 3px; margin: 0 auto 12mm;
  background: #1a5fb4;
}
.cover-title {
  font-size: 26pt; font-weight: 700; line-height: 1.35;
  margin: 0 0 6mm; color: #123a6b; letter-spacing: 1px;
}
.cover-sub { font-size: 12pt; color: #4a5563; margin: 0 0 4mm; }
.cover-meta { font-size: 9.5pt; color: #8a949e; margin: 0; }
/* 目录贴在封面底部（margin-top:auto 吃掉中间空白），只列 h1/h2 层级 */
.toc { margin-top: auto; margin-bottom: 3mm; }
.toc-title {
  font-size: 13pt; font-weight: 700; color: #123a6b;
  border-bottom: 2px solid #1a5fb4; padding-bottom: 2mm; margin-bottom: 4mm;
}
.toc-item { font-size: 10.5pt; line-height: 1.7; color: #2c3542; }
.toc-item.lvl3 { padding-left: 7mm; font-size: 9.5pt; color: #5a6673; }

/* ---------- 标题 ---------- */
h2 {
  break-before: page; page-break-before: always;
  break-after: avoid; page-break-after: avoid;
  break-inside: avoid; page-break-inside: avoid;
  font-size: 15pt; color: #123a6b; margin: 0 0 7px;
  padding-bottom: 4px; border-bottom: 2px solid #1a5fb4;
}
h2.first-h2 { break-before: auto; page-break-before: auto; }
h3 {
  break-after: avoid; break-inside: avoid;
  font-size: 12.5pt; color: #1f4e79; margin: 15px 0 6px;
}
h4 { break-after: avoid; break-inside: avoid; font-size: 11pt; margin: 12px 0 5px; }

/* ---------- 正文元素 ---------- */
p { margin: 0 0 7px; }
strong { color: #101418; }
a { color: #1a5fb4; text-decoration: none; }
ul, ol { margin: 5px 0 9px; padding-left: 21px; }
li { margin: 2px 0; }
li > ul, li > ol { margin: 2px 0; }

code {
  font-family: Consolas, "Microsoft YaHei", monospace;
  background: #f2f3f5; border-radius: 3px; padding: 0 3px; font-size: 0.92em;
}
pre {
  font-family: Consolas, "Microsoft YaHei", monospace;
  background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 4px;
  padding: 6px 9px; margin: 6px 0 10px;
  font-size: 9pt; line-height: 1.45;
  white-space: pre-wrap; overflow-wrap: anywhere;
  break-inside: avoid; page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: inherit; }

blockquote {
  margin: 6px 0 10px; padding: 5px 12px;
  border-left: 3px solid #c9d4e0; background: #f8fafc; color: #3a4653;
  break-inside: avoid;
}
blockquote p { margin: 0 0 5px; }
blockquote p:last-child { margin-bottom: 0; }

table {
  border-collapse: collapse; width: 100%; margin: 6px 0 10px;
  font-size: 9.5pt;
  break-inside: avoid; page-break-inside: avoid;
}
th, td {
  border: 1px solid #c8d0d8; padding: 3px 7px;
  text-align: left; vertical-align: top; overflow-wrap: break-word;
}
thead th { background: #eef3f8; color: #123a6b; }
tbody tr:nth-child(even) { background: #f7f9fb; }

hr { border: 0; border-top: 1px solid #dfe4ea; margin: 12px 0; }

figure {
  margin: 10px 0 12px; text-align: center;
  break-inside: avoid; page-break-inside: avoid;
}
figure img { max-width: 100%; max-height: 190mm; height: auto; }
figcaption, p.caption {
  font-size: 9pt; line-height: 1.5; color: #8a949e; font-style: italic;
}
figcaption { margin-top: 5px; }
p.caption { text-align: center; margin: -3px 0 10px; break-inside: avoid; }
img.inline-img { max-width: 100%; }
.missing { border: 1px dashed #d0555a; color: #a03030; padding: 6px; font-size: 9pt; }
"""


def _toc_html(text: str) -> str:
    """目录条目：把 Markdown 渲染成纯文字（保留行内 code 样式），去掉链接（PDF 里点不动）。"""
    s = inline(text)
    return re.sub(r"<a\s[^>]*>", "", s).replace("</a>", "")


def build_html(body_html: str, meta: dict, subtitle: str, date_str: str, stats: dict) -> str:
    toc = "\n".join(
        f'<div class="toc-item {"lvl2" if lvl == 2 else "lvl3"}">{_toc_html(t)}</div>'
        for lvl, t in meta["toc"]
    )
    title = meta["title"] or FALLBACK_TITLE
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<section class="cover">
  <div class="cover-head">
    <div class="cover-rule"></div>
    <h1 class="cover-title">{html.escape(title)}</h1>
    <p class="cover-sub">{html.escape(subtitle)}</p>
    <p class="cover-meta">由 {html.escape(stats['src_name'])} 生成 · {date_str} ·
      {stats['src_lines']} 行正文 · {meta['n_img']} 张配图</p>
  </div>
  <div class="toc">
    <div class="toc-title">目录</div>
    {toc}
  </div>
</section>
{body_html}
</body>
</html>
"""


# --------------------------------------------------------------------------------------
# 五、路径 / 图片 / 浏览器
# --------------------------------------------------------------------------------------

def wsl_to_windows(path: Path) -> str:
    """``/mnt/c/Users/x`` -> ``C:\\Users\\x``；不在 /mnt/<盘符> 下就报错。"""
    p = Path(path).resolve()
    parts = p.parts
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper()
        rest = "\\".join(parts[3:])
        return f"{drive}:\\{rest}" if rest else f"{drive}:\\"
    raise SystemExit(
        f"[make_pdf] 构建目录必须在 Windows 能访问的位置（/mnt/<盘符>/...），当前是 {p}。\n"
        f"           Edge 只能读 Windows 路径，请用 --build-dir 指定 /mnt/c/... 下的目录。"
    )


def windows_temp_root() -> Path:
    """找一个 Windows 侧的临时目录（/mnt/c/Users/*/AppData/Local/Temp）。"""
    env = os.environ.get("DSH_PDF_TEMP")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
        raise SystemExit(f"[make_pdf] DSH_PDF_TEMP 不是目录: {p}")
    users = Path("/mnt/c/Users")
    if not users.is_dir():
        raise SystemExit("[make_pdf] 找不到 /mnt/c/Users，本脚本需要 Windows 侧 Edge + 可访问的临时目录。")
    cands = [d / "AppData/Local/Temp" for d in sorted(users.iterdir()) if (d / "AppData/Local/Temp").is_dir()]
    for c in cands:
        if c.name.lower() == "administrator" or os.access(c, os.W_OK):
            return c
    if cands:
        return cands[0]
    raise SystemExit("[make_pdf] /mnt/c/Users/*/AppData/Local/Temp 都不存在。")


def find_browser() -> str:
    env = os.environ.get("DSH_PDF_BROWSER") or os.environ.get("DSH_EDGE")
    if env:
        if Path(env).exists():
            return env
        raise SystemExit(f"[make_pdf] DSH_PDF_BROWSER 指向的文件不存在: {env}")
    for c in BROWSER_CANDIDATES:
        if Path(c).exists():
            return c
    raise SystemExit(
        "[make_pdf] 没找到 Edge/Chrome。设 DSH_PDF_BROWSER 指向 msedge.exe 即可。\n"
        "           候选：" + "\n                 ".join(BROWSER_CANDIDATES)
    )


def _flat_rgb(src: Path):
    """读图并合到白底上（tutorial.py 出的 PNG 带 alpha，直接转 RGB 会把透明当黑）。"""
    from PIL import Image

    im = Image.open(src)
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA"))
    return im.convert("RGB")


def stage_images(
    blocks: list[dict],
    base_dir: Path,
    build_dir: Path,
) -> tuple[dict[str, str], list[str]]:
    """把正文引用的本地图片复制进构建目录。

    返回 ``(原始src -> HTML相对src, 缺图列表)``。

    注意：这里**不做任何裁剪**。曾经为了消掉「图注重复」而在这里裁掉图片顶部那行字，
    但根因在 `tutorial.py`（它把图注画进了 PNG）。根因已在那里修掉，
    所以这个启发式补丁也一并删除——补丁留着只会成为下一个读代码的人的陷阱。
    
    """
    img_dir = build_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    missing: list[str] = []
    used: set[str] = set()

    for b in blocks:
        if b["t"] != "img":
            continue
        src = b["src"]
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", src):
            mapping[src] = src  # 远程图：交给 Edge
            continue
        path = (base_dir / src).resolve()
        if not path.is_file():
            missing.append(src)
            continue
        name = path.name
        if name in used:  # 同名不同目录 -> 加序号
            name = f"{path.parent.name}_{name}"
            k = 2
            while name in used:
                name = f"{path.parent.name}{k}_{path.name}"
                k += 1
        used.add(name)
        shutil.copyfile(path, img_dir / name)
        mapping[src] = f"img/{name}"
    return mapping, missing


def pdf_stats(path: Path) -> dict:
    data = path.read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", data))
    counts = [int(x) for x in re.findall(rb"/Count\s+(\d+)", data)]
    return {
        "bytes": len(data),
        "pages": pages,
        "page_tree_max": max(counts) if counts else 0,
        "font_files": len(re.findall(rb"/FontFile", data)),
        "header_ok": data[:5] == b"%PDF-",
    }


_RE_PDF_DATE = re.compile(rb"D:\d{14}[+\-Z]\d{2}'\d{2}'")
_RE_PDF_ID = re.compile(rb"/ID\s*\[\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\]")
_FIXED_DATE = b"D:20200101000000+00'00'"


def normalize_pdf(data: bytes) -> tuple[bytes, bool]:
    """等长替换 CreationDate / ModDate 与 /ID，让内容不变时输出逐字节稳定。

    只做**长度不变**的替换（xref 偏移因此仍然有效），压缩流里几乎不可能出现这种字面量；
    即使误命中也不会破坏结构。
    """
    new, n_date = _RE_PDF_DATE.subn(_FIXED_DATE, data)
    new, n_id = _RE_PDF_ID.subn(
        lambda m: b"/ID [<" + b"0" * len(m.group(1)) + b"> <" + b"0" * len(m.group(2)) + b">]",
        new,
    )
    return new, bool(n_date or n_id) and len(new) == len(data)


def run_browser(browser: str, html_path: Path, pdf_path: Path) -> tuple[int, str]:
    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--no-first-run",
        "--disable-extensions",
        "--hide-scrollbars",
        "--no-pdf-header-footer",  # 否则每页顶部/底部会印日期、标题和 file:// 网址
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=10000",
        f"--print-to-pdf={wsl_to_windows(pdf_path)}",
        "file:///" + wsl_to_windows(html_path).replace("\\", "/"),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=PRINT_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"[make_pdf] Edge 打印超过 {PRINT_TIMEOUT}s 未结束，已放弃。")
    out = (proc.stdout + proc.stderr).decode("utf-8", "replace")
    return proc.returncode, out


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m transformer.make_pdf",
        description="把 transformer/README.md 排版成 PDF（自解析 Markdown + Windows 侧 Edge 打印）。",
    )
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT), help="输出 PDF（默认 transformer/README.pdf）")
    ap.add_argument("--source", default=str(SRC_MD), help="输入 Markdown（默认 transformer/README.md）")
    ap.add_argument("--build-dir", default=None, help="构建目录（默认 Windows 临时目录下的 dshpdf_transformer）")
    ap.add_argument("--dump-blocks", action="store_true", help="只打印解析出的块序列（JSON），不生成 PDF")
    ap.add_argument("--toc-depth", type=int, default=2, choices=(2, 3),
                    help="目录收录到几级标题（默认 2 = 只列 h2；3 = h2+h3）")
    ap.add_argument("--keep-build", action="store_true", help="结束时保留构建目录（默认也保留，供排查用）")
    ap.add_argument("--no-normalize", action="store_true", help="不做时间戳/ID 归一化（输出将逐字节不稳定）")
    args = ap.parse_args(argv)

    t0 = time.time()
    src = Path(args.source).resolve()
    if not src.is_file():
        raise SystemExit(f"[make_pdf] 源文件不存在: {src}")
    text = src.read_text(encoding="utf-8")
    src_lines = len(text.splitlines())

    blocks = split_blocks(text)
    if args.dump_blocks:
        print(json.dumps(blocks, ensure_ascii=False, indent=1))
        return 0

    build_dir = Path(args.build_dir).resolve() if args.build_dir else (windows_temp_root() / BUILD_DIRNAME)
    build_dir.mkdir(parents=True, exist_ok=True)

    image_map, missing = stage_images(blocks, src.parent, build_dir)
    body_html, meta = render_blocks(blocks, image_map, toc_depth=args.toc_depth)

    # 副标题：用 README H1 括号里的标签 + 一句**不与正文重复**的定位语
    subtitle = f"{meta['tag']} · {SUBTITLE}" if meta.get("tag") else SUBTITLE
    date_str = _dt.date.today().isoformat()
    stats = {"src_name": src.relative_to(REPO_ROOT).as_posix() if REPO_ROOT in src.parents else src.name,
             "src_lines": src_lines}

    html_path = build_dir / "README.html"
    html_path.write_text(
        build_html(body_html, meta, subtitle, date_str, stats), encoding="utf-8"
    )

    browser = find_browser()
    raw_pdf = build_dir / "README.raw.pdf"
    if raw_pdf.exists():
        raw_pdf.unlink()
    rc, log = run_browser(browser, html_path, raw_pdf)

    if rc != 0 or not raw_pdf.is_file():
        tail = "\n".join(log.strip().splitlines()[-12:])
        raise SystemExit(f"[make_pdf] Edge 返回码 {rc}，没有产出 PDF。\n--- Edge 输出（末尾） ---\n{tail}")

    before = pdf_stats(raw_pdf)
    data = raw_pdf.read_bytes()
    normalized = False
    if not args.no_normalize:
        data2, normalized = normalize_pdf(data)
        raw_pdf.write_bytes(data2)
        after = pdf_stats(raw_pdf)
        if not (after["header_ok"] and after["bytes"] == before["bytes"] and after["pages"] == before["pages"]):
            raw_pdf.write_bytes(data)  # 归一化没把握就退回原始字节
            normalized = False

    final = pdf_stats(raw_pdf)
    problems = []
    if not final["header_ok"]:
        problems.append("文件头不是 %PDF-")
    if final["pages"] < 2:
        problems.append(f"页数异常: {final['pages']}")
    if final["font_files"] == 0:
        problems.append("没有嵌入字体（FontFile）")
    if final["page_tree_max"] != final["pages"]:
        problems.append(f"/Count({final['page_tree_max']}) 与页对象数({final['pages']}) 不一致")
    if problems:
        raise SystemExit("[make_pdf] PDF 自检未通过：" + "；".join(problems))

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("." + out.name + ".tmp")
    shutil.copyfile(raw_pdf, tmp)
    os.replace(tmp, out)

    print(f"[make_pdf] 源文件      : {stats['src_name']}（{src_lines} 行，{len(blocks)} 个块）")
    print(f"[make_pdf] 构建目录    : {build_dir}")
    print(f"[make_pdf] 配图        : {meta['n_img'] - len(missing)}/{meta['n_img']} 张已映射"
          + (f"，缺图 {missing}" if missing else ""))
    print(f"[make_pdf] PDF         : {out}")
    print(f"[make_pdf] 自检        : {final['bytes']:,} 字节 · {final['pages']} 页 · "
          f"内嵌字体 {final['font_files']} 个 FontFile · 头部 {data[:5].decode('latin1')}")
    print(f"[make_pdf] 归一化      : " + ("已固定时间戳/ID（内容不变则逐字节一致）" if normalized
                                       else "未生效（输出含时间戳，不保证逐字节一致）"))
    print(f"[make_pdf] HTML        : {html_path}")
    if missing:
        print(f"[make_pdf] 警告        : 以下图片找不到，PDF 里以占位框标注：{missing}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
