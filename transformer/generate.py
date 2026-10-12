#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""加载训练好的权重，让它写点东西。

三种采样策略的差别（都在这个脚本里可调）：
* ``temperature``  把 logits 除以 T：T<1 更保守（贪心），T>1 更随机（容易胡说）
* ``top_k``        只在概率最高的 k 个 token 里采样，掐掉长尾垃圾
* ``use_cache``    增量解码：每步只前向 1 个 token，而不是把整段重算一遍。
                   数学上结果一样（有测试保证），但生成 200 个 token 快几十倍。
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch

from . import data as data_mod
from .model import MiniTransformer, ModelConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_run(tag: str, ckpt: str = "ckpt.pt"):
    outdir = os.path.join(ROOT, "runs", tag)
    path = os.path.join(outdir, ckpt)
    if not os.path.exists(path):
        cands = sorted(glob.glob(os.path.join(outdir, "*.pt")))
        assert cands, f"{outdir} 下没有 .pt；先跑 train.py"
        path = cands[-1]
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = MiniTransformer(ModelConfig(**blob["config"]))
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob, path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="用训练好的最小 transformer 生成文本")
    ap.add_argument("--tag", default="quick")
    ap.add_argument("--ckpt", default="ckpt.pt", help="ckpt.pt（最佳）/ last.pt（末步）")
    ap.add_argument("-p", "--prompt", default="", help="提示词；留空则从 bos 开始")
    ap.add_argument("-n", "--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datadir", default=None, help="数据集目录（要和训练时用的那个一致）")
    ap.add_argument("--no-cache", action="store_true", help="关掉 KV cache（对照用）")
    ap.add_argument("--time", action="store_true", help="只测速：cache vs no-cache")
    args = ap.parse_args(argv)

    datadir = args.datadir or os.path.join(ROOT, "data")
    meta = data_mod.load_meta(datadir)
    tok = data_mod.Tok.load(os.path.join(datadir, "tokenizer"))
    model, blob, path = load_run(args.tag, args.ckpt)
    print(f"[gen] 权重 {path}（step {blob.get('step')}，val {blob.get('val_loss', float('nan')):.4f}）")

    ids = ([tok.bos_id] if not args.prompt else [tok.bos_id] + tok.encode(args.prompt))
    x = torch.tensor([ids], dtype=torch.long)

    if args.time:
        for use_cache in (True, False):
            torch.manual_seed(0)
            import time
            t0 = time.time()
            y = model.generate(x.clone(), max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature, top_k=args.top_k,
                               use_cache=use_cache, eos_id=tok.eos_id)
            dt = time.time() - t0
            n = y.shape[1] - x.shape[1]
            print(f"[time] use_cache={use_cache}: {n} tokens / {dt:.2f}s = {n / dt:.1f} tok/s")
        return 0

    torch.manual_seed(args.seed)
    for i in range(args.samples):
        y = model.generate(x.clone(), max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature, top_k=args.top_k,
                           use_cache=not args.no_cache, eos_id=tok.eos_id)
        text = tok.decode(y[0].tolist())
        print(f"\n{'=' * 70}\nsample {i + 1}  (temperature={args.temperature}, top_k={args.top_k})\n"
              f"{'=' * 70}\n{text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
