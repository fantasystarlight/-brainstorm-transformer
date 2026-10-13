#!/usr/bin/env python
"""probe_compression_as_retrieval.py -- 压缩后的 token 表示还能当检索索引用吗？

问题来源："token 压缩和 RAG 里的 embedding 应该是类似的"。
若两者同类，压缩表示应该能直接拿来建检索索引（省一次 embedding 前向，
且索引与上下文压缩共用同一份表示）。这里实测这个推论。

实验（无需训练、无需外部 embedding 模型）：
  1. 48 个块 = 8 个话题 × 6 个块（话题内相近、话题间不同）。
  2. 每块前向一次，逐层存两种句子表示：
       last : 最后一个 token 的隐藏状态（因果 LM 的标准做法）
       mean : 所有位置平均
     每种各有三个状态：
       raw         不压缩
       cmp         位置维每 POOL 个连续 token 平均（token 压缩在表示上的投影）
       ds          位置维每 POOL 个只取第一个（剪枝，负对照）
  3. 每层每表示建索引，用同一份表示做 top-1 同话题检索。
  4. 报每种表示的最佳层，以及压缩保真度（相对 raw 的归一化失真）。

判据：
  * cmp 的最佳层 ≈ raw → 压缩保留可检索语义，两者可共用索引。
  * cmp 明显掉 → 压缩丢了检索性，RAG embedding 与压缩 token 不同类。
  * ds 必须明显更差，否则测试无鉴别力。

⚠ 已踩的坑：直接用 mean-pooling + 原始余弦，48 个块两两相似度全在 0.9966、
同/跨话题间隔恰好 0（LLM 隐藏态的各向异性把语义压成正交微扰）→ 完全没鉴别力。
所以这里同时报 raw / 去均值 / 去均值+白化 三种归一化，取最好。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL = os.path.expanduser("~/models/models/Qwen2.5-Math-1.5B-Instruct")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
POOL = 4
LAYERS = list(range(1, 27))          # 跳过 embedding 层

TOPICS = {
    "queue_latency": [
        "The message broker accumulated a backlog after the consumer group rebalanced.",
        "Queue depth grew steadily while consumer lag stayed flat across partitions.",
        "Lease expiry caused repeated redelivery of the same batch of messages.",
        "Throughput was limited by the single-threaded deserialization step.",
        "The dead letter queue filled up with malformed payloads overnight.",
        "Backpressure was applied by pausing the producer for thirty seconds.",
    ],
    "gradient_training": [
        "The optimizer state was reset between rounds which broke the schedule.",
        "Gradient norms spiked whenever the replay buffer was refreshed.",
        "Learning rate warmup never engaged because the field was unwired.",
        "Mixed precision overflowed and the scaler skipped several steps.",
        "Checkpoint averaging gave a smoother final validation curve.",
        "The loss plateaued because the advantage estimate was stale.",
    ],
    "database_index": [
        "A composite index on two columns removed the sequential scan entirely.",
        "The query planner ignored the index because of a type mismatch.",
        "Vacuum reclaimed space but the bloat had already slowed lookups.",
        "Partition pruning failed when the predicate used a function call.",
        "An index-only scan avoided the heap fetch for the covering query.",
        "Write amplification increased after adding the fourth secondary index.",
    ],
    "cache_eviction": [
        "The eviction policy kept the attention sink and dropped distant keys.",
        "Hit rate collapsed once the working set exceeded the cache capacity.",
        "A two-queue policy protected frequently reused entries from eviction.",
        "The replacement victim was chosen by cumulative attention weight.",
        "Prefetching the next block hid most of the memory latency.",
        "Cold start dominated because no entries survived between requests.",
    ],
    "compiler_codegen": [
        "The pass pipeline failed to vectorize the inner reduction loop.",
        "Register pressure forced the allocator to spill across the hot path.",
        "Loop unrolling doubled instruction cache misses without any speedup.",
        "The backend fused the multiply and accumulate into one instruction.",
        "Aliasing analysis was too conservative and blocked reordering.",
        "Constant propagation removed the branch entirely after inlining.",
    ],
    "network_protocol": [
        "The retransmission timer fired before the acknowledgement arrived.",
        "Congestion window growth stalled once packet loss was detected.",
        "Head of line blocking stalled unrelated streams on the same connection.",
        "The handshake added a full round trip before any payload was sent.",
        "Keepalive probes prevented the middlebox from dropping the idle flow.",
        "Selective acknowledgement let the sender skip the lost segment.",
    ],
    "memory_allocator": [
        "Fragmentation grew because the pool never coalesced free blocks.",
        "The arena held onto pages after the tensor was released.",
        "Alignment padding wasted a third of the allocated slab.",
        "Reference counting delayed the free until the next collection.",
        "A bump allocator removed per-object bookkeeping overhead.",
        "The huge page hint reduced translation lookaside buffer misses.",
    ],
    "numerical_stability": [
        "The softmax overflowed in half precision and produced not a number.",
        "Catastrophic cancellation appeared when subtracting nearly equal terms.",
        "The accumulator used extended precision only inside the reduction.",
        "Denormals flushed to zero and biased the tiny gradient updates.",
        "Condition number of the matrix made the solve unreliable.",
        "Log-sum-exp stabilized the normalization across the long sequence.",
    ],
}


def build_inputs(tok, max_len=192):
    chunks, labels = [], []
    for topic, texts in TOPICS.items():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids[0]
            chunks.append(ids[:max_len])
            labels.append(topic)
    return chunks, labels


def compress_positions(S, pool=POOL, mode="mean"):
    """S:[T,D] 逐位置表示 → 压缩后 [T//pool, D]。mode='mean' 平均 / 'first' 剪枝。"""
    Tt = S.shape[0]
    nb = Tt // pool
    if nb < 1:
        return S[-1:].copy()
    B = S[: nb * pool].reshape(nb, pool, -1)
    return B.mean(1) if mode == "mean" else B[:, 0, :]


def normalize(X, mode):
    X = np.asarray(X, dtype=np.float64)
    if mode in ("center", "whiten"):
        X = X - X.mean(0, keepdims=True)
    if mode == "whiten":
        C = X.T @ X / max(X.shape[0] - 1, 1)
        w, V = np.linalg.eigh(C)
        X = X @ (V / np.sqrt(np.clip(w, 1e-10, None))) @ V.T
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def retrieval(sim, labels):
    lab = np.asarray(labels)
    n = len(lab)
    hit = 0
    for i in range(n):
        s = sim[i].copy()
        s[i] = -np.inf
        hit += int(lab[int(np.argmax(s))] == lab[i])
    same, diff = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (same if lab[i] == lab[j] else diff).append(sim[i, j])
    return hit / n, float(np.mean(same)), float(np.mean(diff))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True).eval()
    chunks, labels = build_inputs(tok)
    n_topics = len(set(labels))
    print(f"[data] {len(chunks)} chunks, {n_topics} topics, pool={POOL}, "
          f"随机基线={1 / n_topics:.4f}", flush=True)

    keys = ("last_raw", "mean_raw", "last_cmp", "mean_cmp", "last_ds", "mean_ds")
    store = {k: [] for k in keys}
    for ci, ids in enumerate(chunks):
        with torch.no_grad():
            out = model(ids[None, :], output_hidden_states=True,
                        output_attentions=False)
        hs = out.hidden_states
        rec = {k: [] for k in keys}
        for li in LAYERS:
            h = hs[li][0].float().cpu().numpy()          # [T, D]
            cm = compress_positions(h, mode="mean")
            cd = compress_positions(h, mode="first")
            rec["last_raw"].append(h[-1])
            rec["mean_raw"].append(h.mean(0))
            rec["last_cmp"].append(cm[-1])
            rec["mean_cmp"].append(cm.mean(0))
            rec["last_ds"].append(cd[-1])
            rec["mean_ds"].append(cd.mean(0))
        for k in keys:
            store[k].append(np.stack(rec[k]))
        if (ci + 1) % 16 == 0:
            print(f"  encoded {ci + 1}/{len(chunks)}", flush=True)

    store = {k: np.asarray(v, dtype=np.float64) for k, v in store.items()}
    report = {"pool": POOL, "n_chunks": len(chunks), "n_topics": n_topics,
              "chance": 1.0 / n_topics, "best": {}, "distortion": {}}

    print("\n=== 每种表示的最佳层（三种归一化里取间隔最大）===")
    print(f"{'表示':>10} {'归一化':>8} {'最佳层':>7} {'top-1':>7} {'间隔':>9}")
    for key in keys:
        X_all = store[key]
        best = None
        for nz in ("raw", "center", "whiten"):
            for li in range(X_all.shape[1]):
                X = normalize(X_all[:, li, :], nz)
                acc, same, diff = retrieval(X @ X.T, labels)
                if best is None or (same - diff) > best["margin"]:
                    best = dict(norm=nz, layer=LAYERS[li], acc=acc, same=same,
                                diff=diff, margin=same - diff)
        report["best"][key] = best
        print(f"{key:>10} {best['norm']:>8} {best['layer']:>7} "
              f"{best['acc']:>7.4f} {best['margin']:>9.4f}")

    print("\n=== 压缩保真度：相对 raw 表示的归一化失真 ===")
    for base in ("last", "mean"):
        raw = store[f"{base}_raw"]
        scale = float((raw ** 2).sum(-1).mean())
        for tag, key in (("平均压缩", f"{base}_cmp"), ("剪枝", f"{base}_ds")):
            d = float(((store[key] - raw) ** 2).sum(-1).mean()) / max(scale, 1e-12)
            report["distortion"][key] = d
            print(f"  {base:>4} + {tag:>6}: 归一化失真 = {d:.4f}")

    per_layer = []
    for li in range(store["mean_cmp"].shape[1]):
        a = store["mean_raw"][:, li, :]
        b = store["mean_cmp"][:, li, :]
        per_layer.append(float(((b - a) ** 2).sum(-1).mean()
                               / max((a ** 2).sum(-1).mean(), 1e-12)))
    report["per_layer_distortion"] = per_layer
    print("\n[逐层] 压缩失真（越小 = 该层越该压）：")
    print("  " + " ".join(f"L{LAYERS[i]}:{d:.3f}" for i, d in enumerate(per_layer)))

    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "compression_retrieval_probe.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
