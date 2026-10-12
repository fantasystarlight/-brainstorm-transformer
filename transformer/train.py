#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练脚本：手写训练循环（不用 Trainer），每一步都看得见。

一个训练步骤只有 6 行：
    x, y = stream.batch(B)          # 随机取一批长度 T 的窗口
    logits, loss, _ = model(x, y)   # 前向：算交叉熵
    loss.backward()                 # 反向：自动微分算出所有参数的梯度
    clip_grad_norm_(params, 1.0)    # 梯度裁剪：防止偶发的巨大梯度把权重打飞
    optimizer.step()                # 更新参数
    optimizer.zero_grad()           # 清空梯度（PyTorch 默认会累加！）

学习率调度用「线性 warmup + 余弦衰减」：刚开始参数是随机的，梯度方向不可信，
所以先小步走把学习率升上来，再按余弦慢慢降下去。这是 LLM 预训练的标准配方。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from . import data as data_mod
from .model import MiniTransformer, ModelConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def pick_device(want: str = "auto") -> str:
    if want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # 本机沙箱里 CUDA 不可用时自动落到 CPU


def lr_at(step: int, total: int, peak: float, warmup: int, min_ratio: float = 0.1) -> float:
    """线性 warmup + 余弦衰减到 peak*min_ratio。"""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))


def build_config(args, meta) -> ModelConfig:
    return ModelConfig(vocab_size=meta["vocab_size"], d_model=args.d_model,
                       n_layers=args.n_layers, n_heads=args.n_heads,
                       n_kv_heads=args.n_kv_heads, max_seq_len=args.seq_len,
                       dropout=args.dropout, tie_weights=args.tie_weights)


def train(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    datadir = args.datadir or os.path.join(ROOT, "data")
    meta = data_mod.prepare(vocab_size=args.vocab_size, force=args.reprepare,
                            files=args.data, datadir=datadir, tokenizer=args.tokenizer,
                            tiktoken_name=args.tiktoken_name)
    dt = meta.get("dtype")
    streams = {"train": data_mod.TokenStream(os.path.join(datadir, "train.bin"), args.seq_len, dt),
               "val": data_mod.TokenStream(os.path.join(datadir, "val.bin"), args.seq_len, dt)}

    cfg = build_config(args, meta)
    model = MiniTransformer(cfg).to(device)
    kv = cfg.n_kv_heads or cfg.n_heads
    print(f"[model] d_model={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads}"
          f"{'' if kv == cfg.n_heads else f' (GQA kv_heads={kv})'} "
          f"ctx={cfg.max_seq_len} vocab={cfg.vocab_size}")
    print(f"[model] 参数 {model.num_params() / 1e6:.3f}M（不含词表 "
          f"{model.num_params(non_embedding=True) / 1e6:.3f}M）；device={device}")
    print(f"[data] train {meta['train_tokens']} tokens / val {meta['val_tokens']} tokens；"
          f"分词器 {meta['tokenizer']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.1)
    tokens_per_step = args.batch_size * args.seq_len
    print(f"[train] {args.steps} 步 × {tokens_per_step} tokens/步 ≈ "
          f"{args.steps * tokens_per_step / 1e6:.2f}M tokens"
          f"（约 {args.steps * tokens_per_step / meta['train_tokens']:.1f} epoch）")

    history, best = [], {"val": float("inf"), "step": -1}
    outdir = os.path.join(ROOT, "runs", args.tag)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = streams["train"].batch(args.batch_size)
        x, y = x.to(device), y.to(device)
        logits, loss, _ = model(x, y)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == args.steps - 1:
            hist = data_mod.estimate_loss(model, streams, args.batch_size, args.eval_batches, device)
            el = time.time() - t0
            print(f"step {step:5d}/{args.steps} | loss {loss.item():.4f} "
                  f"| train {hist['train']:.4f} | val {hist['val']:.4f} "
                  f"| ppl {math.exp(min(hist['val'], 20)):8.1f} | lr {lr:.2e} "
                  f"| |g| {float(gnorm):.2f} | {el:.0f}s")
            history.append({"step": step, "loss": round(loss.item(), 4), "lr": lr,
                            "train": round(hist["train"], 4), "val": round(hist["val"], 4),
                            "grad_norm": round(float(gnorm), 4), "elapsed_s": round(el, 1)})
            if hist["val"] < best["val"]:
                best = {"val": round(hist["val"], 4), "step": step}
                ckpt = {"config": cfg.__dict__, "model": model.state_dict(),
                        "tokenizer_meta": {k: meta[k] for k in
                                           ("tokenizer", "vocab_size", "bos_id", "eos_id")},
                        "step": step, "val_loss": hist["val"]}
                torch.save(ckpt, os.path.join(outdir, "ckpt.pt"))

    # 困惑度/bit 的口径说明：交叉熵是以「token」为单位的，而不同分词器 token 粒度不同，
    # 所以跨 tokenizer 比较要用 **bits per byte**：loss(nats) / ln2 / (每 token 多少字节)。
    val_bytes = len(open(os.path.join(datadir, "val.txt"), encoding="utf-8").read().encode("utf-8"))
    bpb = history[-1]["val"] / math.log(2) * (meta["val_tokens"] / max(1, val_bytes))
    summary = {"tag": args.tag, "config": cfg.__dict__, "steps": args.steps,
               "batch_size": args.batch_size, "tokens_seen": args.steps * tokens_per_step,
               "params": model.num_params(), "device": device,
               "best_val": best, "final_val": history[-1]["val"],
               "val_ppl": round(math.exp(min(history[-1]["val"], 20)), 2),
               "val_bits_per_byte": round(bpb, 3),
               "wall_s": round(time.time() - t0, 1), "history": history}
    with open(os.path.join(outdir, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    torch.save({"config": cfg.__dict__, "model": model.state_dict(), "step": args.steps,
                "tokenizer_meta": {k: meta[k] for k in ("tokenizer", "vocab_size",
                                                        "bos_id", "eos_id")}},
               os.path.join(outdir, "last.pt"))
    print(f"[train] 完成：{time.time() - t0:.0f}s，最佳 val {best['val']} @ step {best['step']}")
    print(f"[train] 产物：{outdir}/ckpt.pt（最佳）、last.pt（末步）、history.json")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="训练最小 transformer")
    ap.add_argument("--quick", action="store_true", help="小模型小步数，几十秒跑通")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-kv-heads", type=int, default=None,
                    help="GQA 的 kv 头数（默认 = n_heads 即 MHA；设 2~4 可省 KV cache）")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--tie-weights", action="store_true",
                    help="输出层与嵌入层共享权重（大词表 + 小语料时强烈建议开）")
    ap.add_argument("--tokenizer", default="hf", choices=["hf", "bpe", "tiktoken"],
                    help="hf=自训 ByteLevel-BPE；bpe=手写后备；tiktoken=预训练 cl100k 词表")
    ap.add_argument("--tiktoken-name", default="cl100k_base")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--reprepare", action="store_true", help="强制重建数据（重训 BPE）")
    ap.add_argument("--datadir", default=None,
                    help="数据集目录（默认 <工作区>/data）；不同语料建议各用各的目录，别互相覆盖")
    ap.add_argument("--data", nargs="*", default=None,
                    help="外部语料文件（想换更大的语料时用；不给则用工作区里的 md/py）")
    args = ap.parse_args(argv)
    if args.quick:
        args.tag = args.tag if args.tag != "base" else "quick"
        args.steps, args.d_model, args.n_layers, args.n_heads = 300, 128, 2, 4
        args.seq_len, args.batch_size, args.warmup, args.log_every = 128, 16, 30, 50
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
