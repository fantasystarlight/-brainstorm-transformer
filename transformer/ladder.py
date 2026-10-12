#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缩放阶梯：同一份中文语料、同样的 token 预算，只改模型大小。

这是在 GPU 上做的一件事：**固定数据与步数，只看模型变大时 bits/byte 怎么变**。
每个配置都写一份 runs/<tag>/history.json，最后汇总成一张表（README §11 引用的就是它）。

用法：
    PYTHONPATH=. python -m transformer.ladder                 # 跑全部 4 档
    PYTHONPATH=. python -m transformer.ladder --only tiny,small
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

# name, d_model, n_layers, n_heads, n_kv_heads, steps
CONFIGS = [
    ("xl_tiny", 128, 4, 4, 4, 3000),
    ("xl_small", 256, 6, 8, 2, 3000),
    ("xl_mid", 384, 6, 12, 4, 3000),
    ("xl_big", 512, 8, 16, 4, 3000),
    ("xl_big2", 640, 10, 16, 4, 3000),
]
DATAFILES = ["rl_corpus.json"]     # 结构化语料：只取中英文正文字段（可用 --data 覆盖）
DATADIR = "data_rl3"                # vocab 4096 的 HF BPE 版（小词表 + 权重绑定）
SEQ_LEN, BATCH = 512, 32


def run_one(name: str, d_model: int, n_layers: int, n_heads: int, n_kv: int, steps: int,
            force_prep: bool, datadir: str, tag_prefix: str, data: list[str]) -> dict:
    cmd = [PY, "-u", "-m", "transformer.train", "--tag", tag_prefix + name,
           "--datadir", datadir,
           "--data", *data, "--tokenizer", "hf", "--vocab-size", "4096", "--tie-weights",
           "--d-model", str(d_model), "--n-layers", str(n_layers),
           "--n-heads", str(n_heads), "--n-kv-heads", str(n_kv),
           "--seq-len", str(SEQ_LEN), "--batch-size", str(BATCH), "--steps", str(steps),
           "--lr", "6e-4", "--warmup", "150", "--log-every", "250",
           "--eval-batches", "20", "--device", "cuda"]
    if force_prep:
        cmd.append("--reprepare")
    env = dict(os.environ, PYTHONPATH=".", OMP_NUM_THREADS="4")
    t0 = time.time()
    log = open(os.path.join(ROOT, "logs", f"ladder_{name}.log"), "w", encoding="utf-8")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"[ladder] {name} 失败（exit {proc.returncode}），见 logs/ladder_{name}.log")
        return {"name": name, "failed": True, "wall_s": round(wall, 1)}
    with open(os.path.join(ROOT, "runs", name, "history.json"), encoding="utf-8") as fh:
        h = json.load(fh)
    vals = [rec["val"] for rec in h["history"]]
    return {"name": name, "failed": False, "params": h["params"], "d_model": d_model,
            "n_layers": n_layers, "n_heads": n_heads, "n_kv_heads": n_kv,
            "steps": steps, "best_val": h["best_val"]["val"], "final_train": h["history"][-1]["train"],
            "final_val": h["final_val"], "ppl": h["val_ppl"],
            "val_flat_after": next((h["history"][i]["step"] for i in range(1, len(vals))
                                    if vals[i] > vals[i - 1] + 0.02 and i > 3), None),
            "bpb": h["val_bits_per_byte"], "wall_s": round(h["wall_s"], 1),
            "total_wall_s": round(wall, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="逗号分隔的配置名")
    ap.add_argument("--first-prep", action="store_true", help="第一档强制重建数据")
    ap.add_argument("--datadir", default=DATADIR)
    ap.add_argument("--data", nargs="*", default=DATAFILES)
    ap.add_argument("--tag-prefix", default="")
    ap.add_argument("--out", default="ladder.json", help="汇总结果写到 runs/<out>")
    args = ap.parse_args(argv)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    todo = CONFIGS if not args.only else [c for c in CONFIGS if c[0] in args.only.split(",")]
    results = []
    for i, (name, d, L, h, kv, steps) in enumerate(todo):
        print(f"[ladder] {name}: d={d} L={L} heads={h} kv={kv} steps={steps}", flush=True)
        r = run_one(name, d, L, h, kv, steps, force_prep=(args.first_prep and i == 0),
                    datadir=args.datadir, tag_prefix=args.tag_prefix, data=args.data)
        results.append(r)
        if not r.get("failed"):
            print(f"  → 参数量 {r['params'] / 1e6:.2f}M | best val {r['best_val']} "
                  f"| ppl {r['ppl']} | bpb {r['bpb']} | {r['wall_s']}s", flush=True)
        with open(os.path.join(ROOT, "runs", args.out), "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
    # 汇总表
    ok = [r for r in results if not r.get("failed")]
    if ok:
        print("\n=== 缩放阶梯（同一份数据、同样 3000 步、vocab 4096 + 权重绑定）===")
        print(f"{'配置':<10}{'参数量':>9}{'d':>5}{'L':>4}{'kv':>4}{'best val':>10}{'bpb':>8}{'秒':>8}")
        for r in ok:
            print(f"{r['name']:<10}{r['params'] / 1e6:>8.2f}M{r['d_model']:>5}{r['n_layers']:>4}"
                  f"{r['n_kv_heads']:>4}{r['best_val']:>10.3f}{r['bpb']:>8.3f}{r['wall_s']:>8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
