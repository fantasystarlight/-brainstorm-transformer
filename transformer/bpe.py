#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从零手写的字节级 BPE 分词器（约 130 行，只依赖 numpy）。

为什么是「字节级」：
    GPT-2 之后的主流做法先把文本编码成 UTF-8 字节（0..255），再对字节序列做 BPE。
    好处是**词表天然完备**——任何字符串、任何 emoji、任何乱码都能编码，永远不需要
    <unk>。中文一个字通常 3 个字节，BPE 学几十上百次合并后就会把常用汉字并成 1 个 token。

核心只有两个函数：
    train(text, vocab_size)  —— 反复挑「出现次数最多的相邻字节对」并合并，直到词表达标
    encode(text) -> ids      —— 按**学到的合并顺序**依次施加合并
    decode(ids)  -> str      —— 每个 id 反查回字节，拼起来再 UTF-8 解码（所以完全可逆）

特殊 token：
    <|bos|> 占 id = 256 + len(merges)，<|eos|> 占 +1。
    生成时用 bos 起头、遇到 eos 停；训练时在每篇文档之间插一个 bos 做分隔。
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

# 预切分：先按这个正则把文本切成「块」，BPE 只在块内合并。
# 这是 GPT-2 的拆分规则（简化版），作用是别让 "hello" 和 ",world" 合并出跨词的怪 token。
# 注意：需要用 `regex` 模块（标准库 re 不支持 \p{L} 这类 Unicode 属性）。
SPLIT_PATTERN_SRC = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
try:
    import regex

    SPLIT_PATTERN = regex.compile(SPLIT_PATTERN_SRC, regex.UNICODE)
    SPLIT_BACKEND = "regex"
except ImportError:  # pragma: no cover - 只有极简环境才会走到
    import re

    # 退而求其次：用「非空白连续段」切分。够用，只是合并会稍微跨词一点。
    SPLIT_PATTERN = re.compile(r"\S+|\s+")
    SPLIT_BACKEND = "re"


@dataclass
class BPEModel:
    merges: list[tuple[int, int]]  # 第 i 项 = 第 i 次合并的 (左id, 右id)
    bos_id: int
    eos_id: int
    train_text_sha256: str = ""

    # ---------------------------------------------------------------- 词表
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + 2

    @property
    def ids(self) -> dict[tuple[int, int], int]:
        """(左, 右) -> 合并后的新 id。新 id 从 256 开始递增。"""
        return {pair: 256 + i for i, pair in enumerate(self.merges)}

    # ---------------------------------------------------------------- 编码
    def _encode_chunk(self, chunk: bytes, ids: dict[tuple[int, int], int],
                      rank: dict[tuple[int, int], int]) -> list[int]:
        """对一个块做 BPE：每轮找 rank 最小（=最早学到）的可合并对，全部合并掉。"""
        tokens = list(chunk)
        while len(tokens) >= 2:
            pairs = {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}
            best = min((rank[p] for p in pairs if p in rank), default=None)
            if best is None:
                break
            # 该轮要合并的是 rank==best 的那个对（可能有多个对同 rank，选出实际那个）
            target = next(p for p in pairs if rank.get(p) == best)
            merged = ids[target]
            out, i = [], 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == target:
                    out.append(merged)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        return tokens

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids, rank = self.ids, {p: i for i, p in enumerate(self.merges)}  # 只建一次，别放进循环
        out: list[int] = []
        if add_bos:
            out.append(self.bos_id)
        for chunk in SPLIT_PATTERN.findall(text):
            if chunk:
                out.extend(self._encode_chunk(chunk.encode("utf-8"), ids, rank))
        if add_eos:
            out.append(self.eos_id)
        return out

    # ---------------------------------------------------------------- 解码
    def decode(self, ids: list[int]) -> str:
        buf = bytearray()
        n_valid = 256 + len(self.merges) + 2          # 含 bos/eos
        for i in ids:
            i = int(i)
            if i == self.bos_id or i == self.eos_id:
                continue  # 特殊 token 不进字节流
            elif i < 256:
                buf.append(i)
            elif i < n_valid:
                a, b = self.merges[i - 256]
                buf.extend(self._expand(a))
                buf.extend(self._expand(b))
            else:                                     # 越界 id：给替换符，不要静默出错
                buf.extend("\ufffd".encode("utf-8"))
        return buf.decode("utf-8", errors="replace")

    def _expand(self, i: int) -> bytes:
        if i < 256:
            return bytes([i])
        a, b = self.merges[i - 256]
        return self._expand(a) + self._expand(b)

    # ---------------------------------------------------------------- 存取
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"merges": [[a, b] for a, b in self.merges],
                       "bos_id": self.bos_id, "eos_id": self.eos_id,
                       "train_text_sha256": self.train_text_sha256}, fh)

    @classmethod
    def load(cls, path: str) -> "BPEModel":
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        return cls(merges=[tuple(p) for p in obj["merges"]],
                   bos_id=obj["bos_id"], eos_id=obj["eos_id"],
                   train_text_sha256=obj.get("train_text_sha256", ""))


def _pair_stats(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """统计相邻对 (a,b) 的频次，返回 (唯一对 [M,2] int64, 频次 [M])。用 numpy 做，省时间。"""
    if tokens.size < 2:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    a, b = tokens[:-1], tokens[1:]
    key = a.astype(np.int64) * (1 << 20) + b  # 把 (a,b) 打成单个整数做 unique
    uniq, counts = np.unique(key, return_counts=True)
    pairs = np.stack([uniq // (1 << 20), uniq % (1 << 20)], axis=1)
    return pairs, counts


def _merge_all(tokens: np.ndarray, left: int, right: int, new_id: int) -> np.ndarray:
    """把 tokens 里所有相邻的 (left,right) 合并成 new_id（重叠的对只合并一次，从左往右）。

    不重叠扫描 + 就地构建新数组。也可以删完再赋值，但那样要小心「赋值后新出现的
    相邻对」——这里的写法没有这个坑。
    """
    if tokens.size < 2:
        return tokens
    out: list[int] = []
    i = 0
    hit = 0
    while i < tokens.size:
        if i < tokens.size - 1 and tokens[i] == left and tokens[i + 1] == right:
            out.append(new_id)
            i += 2
            hit += 1
        else:
            out.append(int(tokens[i]))
            i += 1
    return tokens if hit == 0 else np.asarray(out, dtype=np.int64)


def train(text: str, vocab_size: int = 2048, verbose: bool = True) -> BPEModel:
    """训练 BPE。vocab_size 是**总**词表大小（含 256 字节 + 2 特殊 token）。"""
    n_merges = vocab_size - 256 - 2
    assert n_merges >= 0, "vocab_size 至少要 258"
    chunks = [c.encode("utf-8") for c in SPLIT_PATTERN.findall(text) if c]
    tokens = np.frombuffer(b"".join(chunks), dtype=np.uint8).astype(np.int64)
    if verbose:
        print(f"[bpe] 字符 {len(text)} → 字节 {tokens.size}，目标 {n_merges} 次合并")

    merges: list[tuple[int, int]] = []
    for step in range(n_merges):
        pairs, counts = _pair_stats(tokens)
        if pairs.shape[0] == 0:
            break
        best = int(np.argmax(counts))
        left, right = int(pairs[best, 0]), int(pairs[best, 1])
        if counts[best] < 2:
            break  # 再合并下去只服务单个出现，纯属浪费词表
        new_id = 256 + len(merges)
        tokens = _merge_all(tokens, left, right, new_id)
        merges.append((left, right))
        if verbose and (step + 1) % 256 == 0:
            print(f"[bpe] merge {step + 1}/{n_merges}  最高频次 {int(counts[best])}  "
                  f"剩余 token {tokens.size}")

    model = BPEModel(merges=merges, bos_id=256 + len(merges), eos_id=256 + len(merges) + 1)
    if verbose:
        print(f"[bpe] 词表 {model.vocab_size}（其中合并 {len(merges)} 次）；"
              f"压缩率 {tokens.size / max(1, len(text.encode('utf-8'))):.3f} 字节/token")
    return model


if __name__ == "__main__":  # 小自测：python -m transformer.bpe
    demo = "注意力机制 attention 就是加权平均。" * 3
    m = train(demo, vocab_size=300, verbose=False)
    ids = m.encode(demo)
    assert m.decode(ids) == demo
    print(f"demo: {len(demo)} 字符 → {len(ids)} tokens，roundtrip OK")
