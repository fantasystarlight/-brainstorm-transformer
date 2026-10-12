#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小基准：GQA / 权重绑定 / KV cache 在**生成**时的真实差别。

为什么要测生成而不是训练：
* KV cache 的显存和带宽开销只在**自回归解码**时被放大（每步都要读全部历史 k/v）；
* GQA 省的也正是这份 cache，所以收益要在生成里看。

用法：
    PYTHONPATH=. python -m transformer.bench --device cuda
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from .model import MiniTransformer, ModelConfig


def measure(cfg: ModelConfig, prompt_len: int, new_tokens: int, device: str,
            use_cache: bool = True) -> dict:
    torch.manual_seed(0)
    model = MiniTransformer(cfg).to(device).eval()
    x = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=new_tokens, temperature=1.0, top_k=None,
                           use_cache=use_cache)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else float("nan")
    n = y.shape[1] - x.shape[1]
    return {"params": model.num_params(), "sec": round(dt, 3), "tok_per_s": round(n / dt, 1),
            "peak_mib": round(peak, 1), "tokens": n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base = dict(vocab_size=args.vocab_size, d_model=args.d_model, n_layers=args.n_layers,
                n_heads=args.n_heads, max_seq_len=2048)
    rows = {}

    print(f"[bench] device={device}  d_model={args.d_model} layers={args.n_layers} "
          f"heads={args.n_heads}  prompt={args.prompt_len} new={args.new_tokens}")
    for name, extra in (("MHA (kv=n_heads)", {}),
                        ("GQA (kv=n_heads/4)", {"n_kv_heads": max(1, args.n_heads // 4)}),
                        ("MQA (kv=1)", {"n_kv_heads": 1})):
        r = measure(ModelConfig(**base, **extra), args.prompt_len, args.new_tokens, device)
        rows[name] = r
        print(f"  {name:<20} 参数 {r['params'] / 1e6:6.2f}M | {r['tok_per_s']:7.1f} tok/s | "
              f"峰值显存 {r['peak_mib']:7.1f} MiB")

    # 权重绑定（大词表下省掉整个输出矩阵）
    big_vocab = 100279
    for name, tie in (("大词表·不绑定", False), ("大词表·权重绑定", True)):
        cfg = ModelConfig(vocab_size=big_vocab, d_model=args.d_model, n_layers=args.n_layers,
                          n_heads=args.n_heads, n_kv_heads=max(1, args.n_heads // 4),
                          max_seq_len=2048, tie_weights=tie)
        model = MiniTransformer(cfg)
        r = {"params": model.num_params()}
        rows[name] = r
        print(f"  {name:<20} 参数 {r['params'] / 1e6:6.2f}M  "
              f"（省 {(100279 * args.d_model) / 1e6:.1f}M = 输出矩阵）")

    # KV cache 开关
    r = measure(ModelConfig(**base, n_kv_heads=max(1, args.n_heads // 4)),
                args.prompt_len, args.new_tokens, device, use_cache=False)
    rows["GQA 关 cache"] = r
    print(f"  {'GQA 关 cache':<20} 参数 {r['params'] / 1e6:6.2f}M | {r['tok_per_s']:7.1f} tok/s | "
          f"峰值显存 {r['peak_mib']:7.1f} MiB")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)
        print(f"[bench] 结果写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
