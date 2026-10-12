#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 arXiv LLM 日报的 JSON 数据中抽取「强化学习算法」相关资料，汇总成便于通读的 corpus。"""
import json, glob, os, re, collections

SRC = "/mnt/d/Documents/qwen-agent/04r7kXA7pP/default/arxiv-llm-daily/reports"
OUT = os.path.dirname(os.path.abspath(__file__))

# 强信号：出现即高度相关（权重 3）
STRONG = [
    r"\bRL\b", r"reinforcement learning", r"\bRLHF\b", r"\bRLVR\b", r"\bRLAIF\b",
    r"\bGRPO\b", r"\bPPO\b", r"\bDPO\b", r"\bGSPO\b", r"\bDAPO\b", r"\bKTO\b",
    r"reward model", r"\bRM\b(?=\s*(?:model|training))", r"policy gradient",
    r"actor[- ]critic", r"\bQ-learning\b", r"reward shaping", r"reward hacking",
    r"advantage estimation", r"credit assignment", r"verifiable reward",
    r"\bRLVR\b", r"self-play", r"self play", r"process reward", r"outcome reward",
    r"\bGRM\b", r"\bPRM\b", r"preference optimization", r"online RL",
    r"reinforcement fine-?tuning", r"\bRFT\b", r"\bRLT\b", r"exploration bonus",
    r"value function", r"bellman", r"\bMCTS\b", r"monte carlo tree search",
    r"policy optimization", r"PPO", r"importance sampling rati", r"advantage",
]
# 中信号：需要 ≥2 次或与强信号共现（权重 1）
MID = [
    r"\breward\b", r"\bexploration\b", r"\brollout", r"\bpolicy\b", r"\bagentic\b",
    r"\bverifier\b", r"\bself-improv", r"\biterative\b", r"\bcurriculum\b",
    r"\badvantage\b", r"\bkl\b", r"\bdiversity\b", r"\bsearch\b",
]
STRONG_RE = [re.compile(p, re.I) for p in STRONG]
MID_RE = [re.compile(p, re.I) for p in MID]

NEG_CAT = {"多模态", "RAG 与知识增强", "领域应用"}


def score_entry(e):
    text = (e.get("title", "") + " \n " + e.get("abstract", ""))
    s = 0
    hits = []
    for r in STRONG_RE:
        m = r.findall(text)
        if m:
            s += 3 + min(len(m) - 1, 3)
            hits.append(r.pattern)
    mids = 0
    for r in MID_RE:
        m = r.findall(text)
        if m:
            mids += len(m)
    if any(h.startswith(("reward", "explor", "rollout", "policy", "agentic", "verifier")) for h in hits) or mids >= 6:
        s += min(mids, 8)
    cats = set(e.get("classification", []))
    if cats & {"微调与对齐", "推理与数学"}:
        s += 2
    if cats & {"Agent 与工具使用"}:
        s += 1
    return s, hits


def main():
    entries = {}
    for f in sorted(glob.glob(os.path.join(SRC, "*.json"))):
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        date = os.path.basename(f).split("_")[-1].replace(".json", "")
        for e in data:
            if not isinstance(e, dict) or "id" not in e:
                continue
            e.setdefault("_date", date)
            prev = entries.get(e["id"])
            if prev is None or len(json.dumps(e, ensure_ascii=False)) > len(json.dumps(prev, ensure_ascii=False)):
                entries[e["id"]] = e
    print("total unique papers:", len(entries))

    scored = []
    for e in entries.values():
        s, hits = score_entry(e)
        if s >= 5:
            e["_rl_score"] = s
            e["_rl_hits"] = hits
            scored.append(e)
    scored.sort(key=lambda x: (-x["_rl_score"], -x.get("score", 0)))
    print("RL-relevant:", len(scored))

    with open(os.path.join(OUT, "rl_corpus.md"), "w", encoding="utf-8") as fh:
        for i, e in enumerate(scored, 1):
            a = e.get("analysis_zh") or {}
            fh.write(f"\n### [{i}] {e.get('title','')}\n")
            fh.write(f"- id: {e['id']} | date: {e.get('_date')} | published: {e.get('published')} | "
                     f"cat: {e.get('primary_category')} | score: {e.get('score')} | rl_score: {e['_rl_score']}\n")
            fh.write(f"- classification: {', '.join(e.get('classification', []))}\n")
            if e.get("comment"):
                fh.write(f"- comment: {e['comment']}\n")
            fh.write(f"- ABSTRACT: {re.sub(chr(10), ' ', e.get('abstract',''))}\n")
            if a:
                for k in ("summary_zh", "innovations_zh", "significance", "limitations", "potential"):
                    if a.get(k):
                        fh.write(f"- {k}: {re.sub(chr(10),' ', str(a[k]))}\n")
    with open(os.path.join(OUT, "rl_corpus.json"), "w", encoding="utf-8") as fh:
        json.dump(scored, fh, ensure_ascii=False, indent=1)

    cnt = collections.Counter()
    for e in scored:
        cnt.update(e.get("classification", []))
    print(cnt.most_common())
    print("corpus bytes:", os.path.getsize(os.path.join(OUT, "rl_corpus.md")))


if __name__ == "__main__":
    main()
