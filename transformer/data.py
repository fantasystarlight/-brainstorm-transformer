#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据管线：语料 → BPE 分词 → 定长 token 流 → 随机采样 batch。

关键概念（第一次搭 transformer 最容易搞错的地方）：

1. **语言模型的标签就是输入本身，只是错开一位**。
   给定 token 流 ``[t0, t1, t2, t3 ...]``，取长度 T 的窗口做输入
   ``x = [t0..t(T-1)]``，标签就是 ``y = [t1..tT]``。不需要任何人工标注，任何文本都是训练数据。
2. **验证集切在原始文本上，BPE 只用 train 训练**（见 ``corpus.py``），否则打分不诚实。
3. 一个 batch 的损失是 ``B*T`` 个位置的交叉熵平均。每个窗口的标签都能取到 T 个，
   所以 ``TokenStream`` 的窗口数是 ``N - T - 1``（多留一个 token 给最后一个标签）。
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from . import bpe as my_bpe
from . import corpus

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAX_MERGES = 16 * 1024  # 只是个理智上限；真实 BPE 用几万次合并（GPT-2 词表 50257）


# ======================================================================================
# 分词器封装：优先用 HuggingFace tokenizers（与 Qwen/GPT 同类实现），
# 不可用时回退到本仓库手写的 bpe.py（零依赖、同样可逆）。
# ======================================================================================
class Tok:
    def __init__(self, kind: str, backend, vocab_size: int, bos_id: int, eos_id: int):
        self.kind, self.backend = kind, backend
        self.vocab_size, self.bos_id, self.eos_id = vocab_size, bos_id, eos_id

    def encode(self, text: str) -> list[int]:
        if self.kind == "hf":
            return self.backend.encode(text).ids
        if self.kind == "tiktoken":
            return list(self.backend.encode(text))
        return self.backend.encode(text)

    def decode(self, ids) -> str:
        ids = [int(i) for i in ids if int(i) not in (self.bos_id, self.eos_id)]
        if self.kind == "hf":
            return self.backend.decode(ids, skip_special_tokens=True)
        if self.kind == "tiktoken":
            return self.backend.decode(ids)
        return self.backend.decode(ids)

    @property
    def load_hint(self) -> str:
        return {"hf": "tokenizer.json", "bpe": "tokenizer.json",
                "tiktoken": "<tiktoken 名字，见 meta.json 的 tokenizer_name>"}[self.kind]

    def save(self, path: str) -> None:
        if self.kind == "tiktoken":
            with open(path + ".meta.json", "w", encoding="utf-8") as fh:
                json.dump({"kind": "tiktoken", "tokenizer_name": self.backend.name,
                           "vocab_size": self.vocab_size, "bos_id": self.bos_id,
                           "eos_id": self.eos_id}, fh, ensure_ascii=False)
        elif self.kind == "hf":
            self.backend.save(path + ".json")
            with open(path + ".meta.json", "w", encoding="utf-8") as fh:
                json.dump({"kind": "hf", "vocab_size": self.vocab_size,
                           "bos_id": self.bos_id, "eos_id": self.eos_id}, fh, ensure_ascii=False)
        else:
            self.backend.save(path + ".json")

    @classmethod
    def from_tiktoken(cls, encoding_name: str = "cl100k_base") -> "Tok":
        """用 OpenAI 的 tiktoken 当分词器（可选，见 `--tokenizer tiktoken`）。

        为什么要提供这条路径：本项目手写的 / HF 训出来的 BPE 只有 2048~4096 个 token，
        中文常常被切碎（一个汉字拆到两个 token）；cl100k_base 有 100277 个 token，
        中文压缩率明显更好。它的 BPE 表是在海量多语文本上训出来的，属于「白拿的资产」。
        """
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        bos_id, eos_id = enc.n_vocab, enc.n_vocab + 1   # 留两个空闲 id 给特殊 token
        return cls("tiktoken", enc, enc.n_vocab + 2, bos_id, eos_id)

    @classmethod
    def load(cls, path: str) -> "Tok":
        if os.path.exists(path + ".meta.json"):
            with open(path + ".meta.json", encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("kind") == "tiktoken":                   # 无需落盘词表，直接重建
                import tiktoken
                enc = tiktoken.get_encoding(meta["tokenizer_name"])
                return cls("tiktoken", enc, meta["vocab_size"], meta["bos_id"], meta["eos_id"])
            from tokenizers import Tokenizer
            return cls("hf", Tokenizer.from_file(path + ".json"), meta["vocab_size"],
                       meta["bos_id"], meta["eos_id"])
        m = my_bpe.BPEModel.load(path + ".json")
        return cls("bpe", m, m.vocab_size, m.bos_id, m.eos_id)


# 结构化语料（json / jsonl）里哪些字段该被喂给模型。给 rl_corpus.json 这种
# 「论文元数据 + 中文总结」的数据用：把中英文字段挑出来拼成纯文本。
TEXT_FIELDS = ("title", "abstract", "summary", "analysis_zh", "innovations")


def load_text_file(path: str) -> str:
    """读一个语料文件：.json / .jsonl 走结构化提取，其它按纯文本读。"""
    if not path.endswith((".json", ".jsonl")):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    if path.endswith(".jsonl"):
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        records = json.loads(raw)
        if isinstance(records, dict):
            records = [records]
    parts = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        chunks = []
        for field in TEXT_FIELDS:
            val = rec.get(field)
            if isinstance(val, str) and val.strip():
                chunks.append(val.strip())
            elif isinstance(val, list):                      # innovations 这类是字符串列表
                chunks.extend(str(v).strip() for v in val if str(v).strip())
        if chunks:
            parts.append(f"# ==== 文档 {i} ====\n" + "\n".join(chunks))
    if not parts:
        raise ValueError(f"{path} 里没有找到可用的文本字段 {TEXT_FIELDS}")
    return "\n\n".join(parts)


def build_from_files(files: list[str], datadir: str, vocab_size: int = 2048,
                     val_frac: float = 0.05, verbose: bool = True,
                     tokenizer: str = "hf", tiktoken_name: str = "cl100k_base") -> dict:
    """用**任意外部文本文件**建数据集（想换更大的语料时用，见 README §10）。

    顺序仍然和内置流程一致：先拼文本、先在原始文本上切 train/val、**再**只用 train 训 BPE。
    """
    texts = {}
    for path in files:
        texts[os.path.basename(path)] = load_text_file(path)
    assert texts, "没有读到任何文件"
    # 关键细节：把每篇文档切成块后**用固定种子洗牌**再切 train/val。
    # 否则 val 永远是「位于文件末尾的文档」，而结构化语料的顺序本身有系统性差异
    # （例如 score 从高到低排列），val loss 会异常高、看起来像模型坏了。
    blocks = []
    for name, text in texts.items():
        for chunk in text.split("\n\n"):
            if chunk.strip():
                blocks.append(chunk)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(blocks))
    blocks = [blocks[i] for i in order]
    if verbose:
        print(f"[corpus] {len(blocks)} 个文档块已用 seed=0 洗牌后切分")
    joined = "\n\n".join(blocks)
    cut = int(len(joined) * (1 - val_frac))
    nl = joined.find("\n", cut)
    cut = nl if 0 <= nl < cut + 200 else cut
    train_text, val_text = joined[:cut], joined[cut:]
    os.makedirs(datadir, exist_ok=True)
    for name, text in (("train.txt", train_text), ("val.txt", val_text)):
        with open(os.path.join(datadir, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    meta = {"files": {k: len(v) for k, v in texts.items()}, "n_files": len(texts),
            "total_chars": len(joined), "train_chars": len(train_text),
            "val_chars": len(val_text), "val_frac": val_frac,
            "split": "train=前段 / val=后段（在原始文本上先切分，再训 BPE）",
            "source": "build_from_files"}
    tok = (Tok.from_tiktoken(tiktoken_name) if tokenizer == "tiktoken"
           else train_tokenizer(train_text, vocab_size, verbose=verbose))
    for name, text in (("train", train_text), ("val", val_text)):
        ids = tok.encode(text)
        # 词表小于 65536 时用 uint16 省一半磁盘（tiktoken cl100k 的 id 会超过 65535）
        dtype = np.uint16 if tok.vocab_size <= 65536 else np.uint32
        arr = np.array(ids, dtype=dtype)
        arr.tofile(os.path.join(datadir, f"{name}.bin"))
        # 落盘前做一次严格 round-trip：分词器错位是最难查的 bug（训练照样跑，只是学不到东西）
        back = tok.decode(ids)
        ratio = len(back.replace("\ufffd", "")) / max(1, len(text))
        assert ratio > 0.99, f"{name} 分词 round-trip 可疑（保真度 {ratio:.3f}）"
        meta[f"{name}_tokens"] = int(arr.size)
        if verbose:
            print(f"[data] {name}: {arr.size} tokens → {name}.bin")
    meta.update({"tokenizer": tok.kind, "vocab_size": tok.vocab_size,
                 "bos_id": tok.bos_id, "eos_id": tok.eos_id,
                 "tokenizer_name": getattr(tok.backend, "name", None),
                 "dtype": str(arr.dtype), "train_bytes": len(train_text.encode("utf-8"))})
    tok.save(os.path.join(datadir, "tokenizer"))
    with open(os.path.join(datadir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


def train_tokenizer(train_text: str, vocab_size: int = 2048, verbose: bool = True) -> Tok:
    """训练分词器，vocab_size 含特殊 token。"""
    assert 258 <= vocab_size <= 256 + MAX_MERGES + 2, \
        f"vocab_size 必须在 [258, {256 + MAX_MERGES + 2}]"
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        unk = "<|unk|>"  # 占位，保证特殊 token 编号稳定
        specials = [unk, "<|bos|>", "<|eos|>"]
        tok = Tokenizer(models.BPE(unk_token=unk))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=verbose,
        )
        tok.train_from_iterator([train_text], trainer=trainer)
        bos_id, eos_id = tok.token_to_id("<|bos|>"), tok.token_to_id("<|eos|>")
        if verbose:
            print(f"[tok] HF ByteLevel-BPE：词表 {tok.get_vocab_size()}，bos={bos_id} eos={eos_id}")
        return Tok("hf", tok, tok.get_vocab_size(), bos_id, eos_id)
    except Exception as exc:                            # 环境缺 tokenizers 时
        print(f"[tok] tokenizers 不可用（{type(exc).__name__}: {exc}），回退到手写 BPE")
        m = my_bpe.train(train_text, vocab_size=vocab_size, verbose=verbose)
        m.train_text_sha256 = hashlib.sha256(train_text.encode()).hexdigest()
        return Tok("bpe", m, m.vocab_size, m.bos_id, m.eos_id)


# ======================================================================================
# 语料 → token 流
# ======================================================================================
def prepare(vocab_size: int = 2048, datadir: str | None = None, val_frac: float = 0.05,
            force: bool = False, verbose: bool = True, files: list[str] | None = None,
            tokenizer: str = "hf", tiktoken_name: str = "cl100k_base") -> dict:
    """工程入口：``files`` 非空时用外部文本，否则用工作区根目录的 *.py / *.md。

    ``tokenizer``: ``hf``（自训 ByteLevel-BPE）/ ``bpe``（手写后备）/ ``tiktoken``（预训练词表）。
    """
    if files:
        return build_from_files(files, datadir or os.path.join(ROOT, "data"),
                                vocab_size=vocab_size, val_frac=val_frac, verbose=verbose,
                                tokenizer=tokenizer, tiktoken_name=tiktoken_name)
    datadir = datadir or os.path.join(ROOT, "data")
    meta_path = os.path.join(datadir, "meta.json")
    if os.path.exists(meta_path) and not force:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("vocab_size") == vocab_size:
            if verbose:
                print(f"[data] 复用已有 {meta_path}（vocab {vocab_size}）；要重建加 --force")
            return meta

    meta = corpus.build_and_save(datadir, root=ROOT, val_frac=val_frac)
    with open(os.path.join(datadir, "train.txt"), encoding="utf-8") as fh:
        train_text = fh.read()
    with open(os.path.join(datadir, "val.txt"), encoding="utf-8") as fh:
        val_text = fh.read()

    tok = train_tokenizer(train_text, vocab_size, verbose=verbose)
    train_ids = np.array(tok.encode(train_text), dtype=np.uint16)
    val_ids = np.array(tok.encode(val_text), dtype=np.uint16)
    for name, arr in (("train", train_ids), ("val", val_ids)):
        arr.tofile(os.path.join(datadir, f"{name}.bin"))
        if verbose:
            print(f"[data] {name}: {arr.size} tokens → {name}.bin")

    meta.update({"tokenizer": tok.kind, "vocab_size": tok.vocab_size,
                 "bos_id": tok.bos_id, "eos_id": tok.eos_id,
                 "train_tokens": int(train_ids.size), "val_tokens": int(val_ids.size),
                 "train_bytes": len(train_text.encode("utf-8"))})
    tok.save(os.path.join(datadir, "tokenizer"))
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


# ======================================================================================
# 训练时的随机采样
# ======================================================================================
class TokenStream:
    """把一大段 token 流看成「长度 T 的窗口」的集合，随机采样 batch。"""

    def __init__(self, path: str, seq_len: int, dtype: str | None = None):
        # dtype 由 meta.json 指定（uint16 或 uint32）；不指定时按文件大小猜
        if dtype is None:
            size = os.path.getsize(path)
            dtype = "uint16" if size % 2 == 0 else "uint32"
        self.data = np.fromfile(path, dtype=np.dtype(dtype)).astype(np.int64)
        self.seq_len = seq_len
        assert self.data.size > seq_len + 1, f"{path} 太短（{self.data.size} tokens）"

    def __len__(self) -> int:
        return self.data.size - self.seq_len - 1

    def batch(self, batch_size: int, generator: torch.Generator | None = None):
        """返回 (x [B,T], y [B,T])，y 是 x 右移一位。"""
        ix = torch.randint(len(self), (batch_size,), generator=generator)
        x = np.stack([self.data[i:i + self.seq_len] for i in ix])
        y = np.stack([self.data[i + 1:i + 1 + self.seq_len] for i in ix])
        return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def estimate_loss(model, streams: dict[str, TokenStream], batch_size: int, eval_batches: int = 20,
                  device: str = "cpu") -> dict[str, float]:
    """在若干（通常是 train/val）数据流上估平均损失。"""
    out = {}
    model.eval()
    for name, stream in streams.items():
        losses = []
        for _ in range(eval_batches):
            x, y = stream.batch(batch_size)
            _, loss, _ = model(x.to(device), y.to(device))
            losses.append(loss.item())
        out[name] = float(np.mean(losses))
    model.train()
    return out


def load_meta(datadir: str | None = None) -> dict:
    datadir = datadir or os.path.join(ROOT, "data")
    with open(os.path.join(datadir, "meta.json"), encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":  # 只做数据准备：python -m transformer.data
    print(json.dumps(prepare(force=True), ensure_ascii=False, indent=2))
