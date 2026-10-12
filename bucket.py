#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 RL corpus 按主题分桶，产出 slices/*.md，供并行精读。"""
import json, os, re, collections

HERE = os.path.dirname(os.path.abspath(__file__))
SL = os.path.join(HERE, "slices")
os.makedirs(SL, exist_ok=True)

# 按优先级顺序匹配（先匹配到的桶为准），title 权重更高
BUCKETS = [
    ("A_advantage_credit", r"credit assignment|advantage|step-level|token-level|segment|trajectory graph|GAE|value function|critic|baseline|variance reduction|turn-level|process reward model|PRM|process reward"),
    ("B_reward_verifier", r"reward hack|reward shaping|reward model|verifier|rubric|reward design|reward aggregation|multi-reward|generative reward|reward signal|outcome reward|verification|self-verif|judge"),
    ("C_exploration_diversity", r"explor|entropy|diversit|temperature|curiosity|novelty|mode collapse|pass@k|sampling strateg|branch|tree search|MCTS|rollout budget|difficulty"),
    ("D_agentic_multiturn", r"agentic|multi-turn|multi turn|tool(-| )calling|tool use|long-horizon|long horizon|environment|interactive|GUI|web agent|SWE|computer use|multi-step|multi step|sequential decision"),
    ("E_offpolicy_efficiency", r"off-policy|off policy|distill|importance sampling|replay|sample efficien|compute|memory|KV cache|quantiz|asynchron|speed|throughput|scaling law|budget|cost|token efficien|efficien"),
    ("F_optimization_geometry", r"policy optimization|gradient|trust region|KL|clip|convergence|theor|landscape|geometry|subspace|stability|collapse|loss function|optimizer|scheduler|entropy regulariz|temperature schedul"),
    ("G_data_curriculum_selfplay", r"curriculum|self-play|self play|data selection|data curation|difficulty|synthes|self-improv|iterative|bootstrapp|warm-?up|listwise|pairwise|preference|DPO"),
    ("H_multimodal_embodied", r"multimodal|vision-language|VLM|MLLM|video|image|embodied|robot|VLA|speech|audio|diffusion|generation model"),
]
RE = [(n, re.compile(p, re.I)) for n, p in BUCKETS]

CAP = {  # 每桶最多收录多少篇（按 rl_score 排序取头部）
    "A_advantage_credit": 70, "B_reward_verifier": 70, "C_exploration_diversity": 55,
    "D_agentic_multiturn": 60, "E_offpolicy_efficiency": 50, "F_optimization_geometry": 55,
    "G_data_curriculum_selfplay": 55, "H_multimodal_embodied": 40, "Z_other": 30,
}


def bucket(e):
    t = e.get("title", "")
    a = e.get("abstract", "")
    for n, r in RE:
        if r.search(t):
            return n
    for n, r in RE:
        if len(r.findall(a)) >= 1:
            return n
    return "Z_other"


def main():
    data = json.load(open(os.path.join(HERE, "rl_corpus.json"), encoding="utf-8"))
    # Tier A：真正谈 RL 算法/训练方法的核心集
    core = [e for e in data if e["_rl_score"] >= 12]
    print("core RL papers (rl_score>=12):", len(core))
    buckets = collections.defaultdict(list)
    for e in data:
        buckets[bucket(e)].append(e)
    for k in buckets:
        buckets[k].sort(key=lambda x: (-x["_rl_score"], -x.get("score", 0)))
    idx = []
    for k in list(CAP):
        items = buckets.get(k, [])[: CAP[k]]
        with open(os.path.join(SL, k + ".md"), "w", encoding="utf-8") as fh:
            for i, e in enumerate(items, 1):
                a = e.get("analysis_zh") or {}
                fh.write(f"\n### [{i}] {e.get('title','')}  (id={e['id']}, {e.get('_date')}, rl={e['_rl_score']})\n")
                fh.write(f"- cats: {', '.join(e.get('classification', []))}\n")
                fh.write(f"- ABS: {re.sub(chr(10),' ', e.get('abstract',''))[:1400]}\n")
                if a.get("summary_zh"):
                    fh.write(f"- ZH: {a['summary_zh'][:700]}\n")
                if a.get("innovations_zh"):
                    v = a["innovations_zh"]
                    fh.write(f"- INNOV: {(v if isinstance(v,str) else '; '.join(v))[:600]}\n")
                if a.get("limitations"):
                    fh.write(f"- LIM: {str(a['limitations'])[:400]}\n")
        idx.append((k, len(buckets.get(k, [])), len(items), os.path.getsize(os.path.join(SL, k + ".md"))))
    print("bucket, total, kept, bytes")
    for row in idx:
        print(" ", row)

    # 核心集单独落一份完整摘要（含 potential）
    core.sort(key=lambda x: (-x["_rl_score"], -x.get("score", 0)))
    with open(os.path.join(SL, "CORE.md"), "w", encoding="utf-8") as fh:
        for i, e in enumerate(core[:220], 1):
            a = e.get("analysis_zh") or {}
            fh.write(f"\n### [{i}] {e.get('title','')}  (id={e['id']}, {e.get('_date')}, rl={e['_rl_score']}, score={e.get('score')})\n")
            fh.write(f"- cats: {', '.join(e.get('classification', []))}\n")
            fh.write(f"- ABS: {re.sub(chr(10),' ', e.get('abstract',''))[:1600]}\n")
            for k2 in ("summary_zh", "innovations_zh", "limitations", "potential"):
                if a.get(k2):
                    v = a[k2]
                    fh.write(f"- {k2}: {(v if isinstance(v,str) else '; '.join(v))[:600]}\n")
    print("CORE.md bytes:", os.path.getsize(os.path.join(SL, "CORE.md")), "n=", min(220, len(core)))

    # 全量标题索引（覆盖用）
    with open(os.path.join(HERE, "rl_titles.txt"), "w", encoding="utf-8") as fh:
        for e in data:
            fh.write(f"{e['_rl_score']:>3} {e.get('_date')} {e['id']} {e.get('title','')}\n")
    print("titles:", len(data))


if __name__ == "__main__":
    main()
