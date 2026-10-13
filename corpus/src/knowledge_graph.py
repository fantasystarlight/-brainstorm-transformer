#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RL 新策略知识图谱：单一数据源 → PNG / Mermaid / 纯文本 / JSON 四份同源产物。

图谱分六张：
  kg_overview      缺口 → 提案 → 知识域
  kg_knowledge_a   K-OPT / K-CREDIT / K-REWARD 三个域的知识点及其依赖
  kg_knowledge_b   K-EXPLORE / K-SYS / K-THEORY
  kg_knowledge_c   K-METHOD / K-XFER（迁移源，节点上直接标注流向哪个提案）
  kg_conflicts     四条未解决争论的双方与判决实验
  kg_methods       八个域的代表方法 + arXiv id + 证据状态

约定同 `flowchart.py`：节点自带 `anchor`，测试断言该字符串仍存在于
`RL新策略头脑风暴.md`（附录 C 就是这些知识点的文字版）；write-if-changed + 原子替换；
中文缺字体时降级英文并告警。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig-dsh"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "RL新策略头脑风暴.md")
OUTDIR = os.path.join(HERE, "docs")

# 复用流程图脚本里的字体与写盘逻辑（避免两套实现漂移）
_spec = importlib.util.spec_from_file_location("fc", os.path.join(HERE, "flowchart.py"))
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)
setup_font = fc.setup_font
write_if_changed = fc.write_if_changed


# ======================================================================================
# 1. 知识域
# ======================================================================================
DOMAINS = [
    {"id": "K-OPT", "name": "优化目标与信任域", "en": "objective & trust region", "color": "#e8f0fb"},
    {"id": "K-CREDIT", "name": "信用分配", "en": "credit assignment", "color": "#fdf1e3"},
    {"id": "K-REWARD", "name": "奖励与验证", "en": "reward & verification", "color": "#fdeaea"},
    {"id": "K-EXPLORE", "name": "探索与覆盖", "en": "exploration & coverage", "color": "#e9f7ef"},
    {"id": "K-SYS", "name": "系统与漂移", "en": "systems & drift", "color": "#eef2f5"},
    {"id": "K-THEORY", "name": "理论与诊断", "en": "theory & diagnostics", "color": "#f6e9f7"},
    {"id": "K-METHOD", "name": "方法论", "en": "methodology", "color": "#fff8e1"},
    {"id": "K-XFER", "name": "跨域迁移源", "en": "transfer sources", "color": "#eae4f7"},
]
DOMAIN_BY_ID = {d["id"]: d for d in DOMAINS}

# ======================================================================================
# 2. 知识点（51 个）——anchor 必须出现在文档附录 C 中
# ======================================================================================
C = lambda i, dom, label, anchor, en=None, extra="": {
    "id": i, "domain": dom, "label": label, "anchor": anchor, "en": en or label, "extra": extra}


CONCEPTS = [
    # K-OPT
    C("c_pg", "K-OPT", "策略梯度定理", "策略梯度定理", "policy gradient thm", "P1 P9 P10"),
    C("c_is", "K-OPT", "重要性采样比", "重要性采样比", "importance ratio", "P3 P9"),
    C("c_tr", "K-OPT", "信任域与 KL 约束", "信任域与 KL 约束", "trust region / KL", "P3 P8"),
    C("c_clip", "K-OPT", "clip 死区", "clip 死区", "clip dead zone", "P3"),
    C("c_ent", "K-OPT", "熵正则与熵预算", "熵正则与熵预算", "entropy budget", "P2 P6"),
    C("c_rkl", "K-OPT", "反向 KL 与模式覆盖", "反向 KL 与模式覆盖", "reverse KL", "P6"),
    C("c_md", "K-OPT", "镜像下降 / Bregman 散度", "镜像下降 / Bregman 散度", "mirror descent", "P1"),
    # K-CREDIT
    C("c_adv", "K-CREDIT", "优势函数 A=Q−V", "优势函数 A=Q−V", "advantage A=Q-V", "P1"),
    C("c_gae", "K-CREDIT", "广义优势估计 GAE", "广义优势估计 GAE", "GAE", "P1 P9"),
    C("c_group", "K-CREDIT", "组相对基线（RLOO/GRPO）", "组相对基线（RLOO/GRPO）", "group baseline", "P1 P2 P6"),
    C("c_axiom", "K-CREDIT", "信用分配三公理", "信用分配三公理", "credit axioms", "P1"),
    C("c_prm", "K-CREDIT", "过程奖励与不可验证性", "过程奖励与不可验证性", "process reward", "P7"),
    C("c_graph", "K-CREDIT", "轨迹图与图上 Bellman 反传", "轨迹图与图上 Bellman 反传", "trajectory graph", "P1 P9"),
    C("c_bv", "K-CREDIT", "偏差-方差权衡", "偏差-方差权衡", "bias-variance", "P1 P9"),
    # K-REWARD
    C("c_rlvr", "K-REWARD", "可验证奖励 RLVR", "可验证奖励 RLVR", "verifiable reward", "P4 P7"),
    C("c_shape", "K-REWARD", "奖励塑形与有界饱和", "奖励塑形与有界饱和", "reward shaping", "P4"),
    C("c_rubric", "K-REWARD", "Rubric-as-Reward", "Rubric-as-Reward", "rubric-as-reward", "P4 P5"),
    C("c_hack", "K-REWARD", "奖励黑客", "奖励黑客", "reward hacking", "P4 P5"),
    C("c_agg", "K-REWARD", "多奖励聚合与投影", "多奖励聚合与投影", "multi-reward aggregation", "P4"),
    C("c_grm", "K-REWARD", "生成式奖励模型 GRM", "生成式奖励模型 GRM", "generative RM", "P11"),
    C("c_settle", "K-REWARD", "校准与密封结算流", "校准与密封结算流", "sealed settlement", "P4 P5"),
    C("c_psr", "K-REWARD", "严格 proper scoring rule", "严格 proper scoring rule", "proper scoring rule", "P5"),
    # K-EXPLORE
    C("c_zero", "K-EXPLORE", "组内奖励方差与零梯度组", "组内奖励方差与零梯度组", "zero-variance group", "P2 P6 P7"),
    C("c_collapse", "K-EXPLORE", "熵坍缩", "熵坍缩", "entropy collapse", "P6"),
    C("c_passk", "K-EXPLORE", "pass@k 与覆盖率", "pass@k 与覆盖率", "pass@k & coverage", "P6 P8"),
    C("c_qd", "K-EXPLORE", "质量-多样性 QD / MAP-Elites", "质量-多样性 QD / MAP-Elites", "quality-diversity", "P6"),
    C("c_ts", "K-EXPLORE", "Thompson 采样与信息增益", "Thompson 采样与信息增益", "Thompson sampling", "P2 P8"),
    C("c_curr", "K-EXPLORE", "难度后验与课程", "难度后验与课程", "difficulty posterior", "P2"),
    # K-SYS
    C("c_tim", "K-SYS", "训练-推理失配 TIM 与 drift", "训练-推理失配 TIM 与 drift", "TIM & drift", "P3 P8"),
    C("c_off", "K-SYS", "离策略校正与重要性重加权", "离策略校正与重要性重加权", "off-policy correction", "P3"),
    C("c_stale", "K-SYS", "staleness 与异步", "staleness 与异步", "staleness", "P3"),
    C("c_kv", "K-SYS", "显存墙与 KV 压缩", "显存墙与 KV 压缩", "memory wall & KV", "P3 P8"),
    C("c_tail", "K-SYS", "rollout 吞吐与长尾调度", "rollout 吞吐与长尾调度", "rollout tail", "P8"),
    C("c_order", "K-SYS", "蒸馏-然后-RL 的顺序性", "蒸馏-然后-RL 的顺序性", "distill-then-RL", "P8"),
    # K-THEORY
    C("c_conv", "K-THEORY", "收敛率与策略改进下界", "收敛率与策略改进下界", "convergence rate", "P3"),
    C("c_land", "K-THEORY", "损失景观（自旋玻璃映射）", "损失景观（自旋玻璃映射）", "loss landscape", "P1"),
    C("c_sub", "K-THEORY", "主子空间几何与正交投影", "主子空间几何与正交投影", "subspace geometry", "P10"),
    C("c_param", "K-THEORY", "参数漂移 ≠ 功能变化", "参数漂移 ≠ 功能变化", "param drift != function", "P10"),
    C("c_sae", "K-THEORY", "机制可解释性与 SAE", "机制可解释性与 SAE", "mechanistic interp / SAE", "P4"),
    # K-METHOD
    C("c_pre", "K-METHOD", "预注册与判决判据", "预注册与判决判据", "preregistration", "全部"),
    C("c_ctrl", "K-METHOD", "负对照与鉴别力", "负对照与鉴别力", "negative control", "全部"),
    C("c_iso", "K-METHOD", "等算力对比", "等算力对比", "iso-compute", "全部"),
    C("c_seed", "K-METHOD", "多种子与配对随机数", "多种子与配对随机数", "seeds & CRN", "P9"),
    C("c_curve", "K-METHOD", "可监控性曲线", "可监控性曲线", "monitorability curves", "全部"),
    # K-XFER
    C("c_x_cql", "K-XFER", "离线 RL 悲观价值（CQL/IQL）", "离线 RL 悲观价值（CQL/IQL）", "offline RL pessimism", "→P2"),
    C("c_x_stop", "K-XFER", "最优停止与运筹调度", "最优停止与运筹调度", "optimal stopping", "→P1 P8"),
    C("c_x_tmle", "K-XFER", "因果推断双重稳健估计", "因果推断双重稳健估计", "TMLE / doubly robust", "→P1 P7"),
    C("c_x_dp", "K-XFER", "差分隐私与噪声会计", "差分隐私与噪声会计", "DP accounting", "→P4"),
    C("c_x_audit", "K-XFER", "审计抽样与 materiality", "审计抽样与 materiality", "audit sampling", "→P4"),
    C("c_x_vcg", "K-XFER", "机制设计与拍卖（VCG）", "机制设计与拍卖（VCG）", "mechanism design", "→P5"),
    C("c_x_orth", "K-XFER", "持续学习正交梯度（PCGrad/OGD）", "持续学习正交梯度（PCGrad/OGD）", "orthogonal gradients", "→P10"),
    C("c_x_queue", "K-XFER", "排队论与物化视图", "排队论与物化视图", "queueing & matview", "→P8"),
]
CONCEPT_BY_ID = {c["id"]: c for c in CONCEPTS}
assert len(CONCEPTS) == len(CONCEPT_BY_ID), "知识点 id 重复"

# 知识点之间的关系：depend（基于）/ oppose（对立）/ transfer（迁移）
CONCEPT_EDGES = [
    ("c_pg", "c_adv", "depend"), ("c_adv", "c_gae", "depend"), ("c_gae", "c_group", "depend"),
    ("c_group", "c_axiom", "depend"), ("c_axiom", "c_prm", "depend"), ("c_graph", "c_axiom", "depend"),
    ("c_bv", "c_gae", "depend"), ("c_bv", "c_prm", "oppose"),
    ("c_is", "c_tr", "depend"), ("c_tr", "c_clip", "oppose"), ("c_tr", "c_md", "depend"),
    ("c_ent", "c_rkl", "oppose"), ("c_ent", "c_collapse", "depend"),
    ("c_rlvr", "c_hack", "oppose"), ("c_shape", "c_hack", "oppose"), ("c_rubric", "c_hack", "oppose"),
    ("c_agg", "c_hack", "depend"), ("c_grm", "c_hack", "oppose"), ("c_psr", "c_hack", "oppose"),
    ("c_settle", "c_hack", "depend"), ("c_settle", "c_psr", "depend"),
    ("c_zero", "c_ts", "depend"), ("c_zero", "c_curr", "depend"), ("c_passk", "c_qd", "depend"),
    ("c_passk", "c_collapse", "oppose"), ("c_group", "c_zero", "depend"), ("c_rkl", "c_qd", "depend"),
    ("c_tim", "c_off", "depend"), ("c_tim", "c_stale", "depend"), ("c_kv", "c_off", "depend"),
    ("c_tail", "c_tim", "depend"), ("c_order", "c_rlvr", "depend"),
    ("c_land", "c_conv", "depend"), ("c_sub", "c_param", "depend"), ("c_sae", "c_param", "depend"),
    ("c_sub", "c_orth" if False else "c_x_orth", "depend"),
    ("c_pre", "c_iso", "depend"), ("c_pre", "c_ctrl", "depend"), ("c_seed", "c_iso", "depend"),
    ("c_ctrl", "c_curve", "depend"),
    ("c_x_cql", "c_zero", "transfer"), ("c_x_stop", "c_prm", "transfer"),
    ("c_x_tmle", "c_prm", "transfer"), ("c_x_dp", "c_hack", "transfer"),
    ("c_x_audit", "c_hack", "transfer"), ("c_x_vcg", "c_psr", "transfer"),
    ("c_x_orth", "c_sub", "transfer"), ("c_x_queue", "c_order", "transfer"),
]

# ======================================================================================
# 3. 缺口 / 提案 / 提案↔知识点
# ======================================================================================
GAPS = [
    {"id": "G1", "label": "G1 没有尺子", "anchor": "G1 没有尺子"},
    {"id": "G2", "label": "G2 无梯度被当垃圾", "anchor": "G2 无梯度被当垃圾"},
    {"id": "G3", "label": "G3 静态超参 vs 可测状态", "anchor": "G3 静态超参 vs 可测状态"},
    {"id": "G4", "label": "G4 hack 没有预算", "anchor": "G4 奖励是固定代理，hack 没有预算"},
    {"id": "G5", "label": "G5 裁判激励不相容", "anchor": "G5 裁判既当运动员又当裁判"},
    {"id": "G6", "label": "G6 覆盖率被牺牲", "anchor": "G6 单目标最大化 vs 覆盖率"},
]

PROPOSALS = [
    {"id": "P1", "label": "P1 CreditBench 尺子", "anchor": "P1　给信用分配造一把尺子",
     "domains": ["K-CREDIT", "K-METHOD", "K-THEORY"]},
    {"id": "P2", "label": "P2 零方差=后验证据", "anchor": "P2　把\"全对/全错\"当作后验证据",
     "domains": ["K-EXPLORE", "K-XFER"]},
    {"id": "P3", "label": "P3 DriftAudit", "anchor": "P3　`DriftAudit`",
     "domains": ["K-SYS", "K-OPT"]},
    {"id": "P4", "label": "P4 奖励投资组合", "anchor": "P4　奖励组合的在线投资组合",
     "domains": ["K-REWARD", "K-XFER"]},
    {"id": "P5", "label": "P5 激励相容裁判", "anchor": "P5　激励相容的裁判",
     "domains": ["K-REWARD", "K-XFER"]},
    {"id": "P6", "label": "P6 档案馆/MAP-Elites", "anchor": "P6　把档案馆搬进 LLM RL",
     "domains": ["K-EXPLORE", "K-METHOD"]},
    {"id": "P7", "label": "P7 可核查性预算", "anchor": "P7　可核查性预算",
     "domains": ["K-CREDIT", "K-REWARD", "K-METHOD"]},
    {"id": "P8", "label": "P8 算力兑换率", "anchor": "P8　训练 ↔ 推理的算力兑换率调度器",
     "domains": ["K-SYS", "K-EXPLORE"]},
    {"id": "P9", "label": "P9 梯度方差设计", "anchor": "P9　把 rollout 当对照实验来设计",
     "domains": ["K-METHOD", "K-CREDIT"]},
    {"id": "P10", "label": "P10 能力分区正交", "anchor": "P10　RL 的持续学习",
     "domains": ["K-THEORY", "K-XFER"]},
    {"id": "P11", "label": "P11 世界模型反事实", "anchor": "P11　用世界模型做反事实过程奖励",
     "domains": ["K-REWARD", "K-SYS"]},
]

GAP_TO_PROPOSAL = [
    ("G1", "P1", ""), ("G1", "P7", ""), ("G1", "P9", ""),
    ("G2", "P2", ""), ("G2", "P6", ""),
    ("G3", "P3", ""), ("G3", "P8", ""),
    ("G4", "P4", ""), ("G4", "P7", ""),
    ("G5", "P5", ""), ("G5", "P4", ""),
    ("G6", "P6", ""), ("G6", "P10", ""), ("G6", "P11", ""),
]

# ======================================================================================
# 4. 四条争论（双方 + 判决实验 + 状态）
# ======================================================================================
CONFLICTS = [
    {
        "id": "C1", "question": "要不要 critic？",
        "anchor": "critic 该不该有",
        "left": "BPCO：每 prompt 只采 1 次\n就追平 group 基线",
        "right": "主流默认 critic 不划算\nSP³O：失败是实现问题\n（value flattening）",
        "test": "同一硬件、同模型上比\nGPU-小时 / 达到同等 Pass@K",
        "status": "未判决",
    },
    {
        "id": "C2", "question": "细粒度信用真有用？",
        "anchor": "细粒度信用是否真有用",
        "left": "HISPO：段级优于\ntoken / 序列级",
        "right": "PACT：RLOO 粗粒度在期望\n梯度上与 token 级一致\nRLDS：2/4 基准无显著增益",
        "test": "P1：在真实信用已知的环境\n对拍所有估计器",
        "status": "有反证",
    },
    {
        "id": "C3", "question": "熵该涨还是该降？",
        "anchor": "熵该涨还是该降",
        "left": "正确轨迹熵降更快\n（ERR+：该奖励消解）",
        "right": "熵坍缩→大 k pass@k 崩\n须保高熵护覆盖",
        "test": "固定熵轨迹的干预实验\n+ 熵/长度/pass@k 三曲线",
        "status": "可调和",
    },
    {
        "id": "C4", "question": "过程奖励的收益从哪来？",
        "anchor": "过程奖励的收益从哪来",
        "left": "局部性：首个错误位置\n（Cliff）／密集性：每步都给",
        "right": "教师信息：特权解\n仅终局奖励也能 +10 点\n（三因素从未分离）",
        "test": "P7：真 V vs 随机 V\n的 2×2 判决实验",
        "status": "判决实验优先",
    },
]

# ======================================================================================
# 5. 代表方法（含 arXiv id 与证据状态）
# ======================================================================================
S_CONSENSUS, S_CONFLICT, S_OPEN = "consensus", "conflict", "open"
STATUS_COLOR = {S_CONSENSUS: "#e9f7ef", S_CONFLICT: "#fdeaea", S_OPEN: "#f4f4f4"}
STATUS_LABEL = {S_CONSENSUS: "共识 / 证据强", S_CONFLICT: "存在冲突或反例", S_OPEN: "空白 / 待验证"}

METHODS = [
    # (节点 id, 知识域, 标签（含 arXiv id）, 证据状态)
    ("m_pact",    "K-CREDIT", "PACT 2609.26355", S_CONFLICT),
    ("m_graft",   "K-CREDIT", "GRAFT 2609.28963", S_OPEN),
    ("m_hispo",   "K-CREDIT", "HISPO 2609.15471", S_CONFLICT),
    ("m_slca",    "K-CREDIT", "SLCA-GRPO 2609.29050", S_CONSENSUS),
    ("m_rlds",    "K-CREDIT", "RLDS 2609.27035", S_CONFLICT),
    ("m_basis",   "K-CREDIT", "BASIS 2605.27293", S_OPEN),
    ("m_cliff",   "K-CREDIT", "Cliff 2609.02817", S_OPEN),
    ("m_crest",   "K-CREDIT", "CrEST 2608.13179", S_OPEN),
    ("m_ano",     "K-OPT", "ANO 2605.02320", S_OPEN),
    ("m_quatro",  "K-OPT", "QUATRO 2602.04620", S_OPEN),
    ("m_infoppo", "K-OPT", "InfoPPO 2609.24380", S_OPEN),
    ("m_conspo",  "K-OPT", "ConSPO 2605.12969", S_OPEN),
    ("m_bpo",     "K-OPT", "BPO 2609.15987", S_OPEN),
    ("m_gcpo",    "K-OPT", "GCPO 2608.11674", S_CONSENSUS),
    ("m_gmts",    "K-OPT", "GMTS 2608.30632", S_CONFLICT),
    ("m_par",     "K-REWARD", "PAR 2502.18770", S_CONSENSUS),
    ("m_amrp",    "K-REWARD", "AMRP 2609.00213", S_OPEN),
    ("m_drop",    "K-REWARD", "Rubric Dropout 2608.11669", S_CONSENSUS),
    ("m_debate",  "K-REWARD", "Debate 2608.17776", S_OPEN),
    ("m_flare",   "K-REWARD", "FLARE 2609.23808", S_OPEN),
    ("m_pcc",     "K-REWARD", "Proof-Carrying 2609.09776", S_OPEN),
    ("m_bts",     "K-REWARD", "Bayesian Truth Serum 2608.25267", S_OPEN),
    ("m_paw",     "K-REWARD", "PaW 2606.02388", S_OPEN),
    ("m_cge",     "K-EXPLORE", "CGE 2609.25773", S_OPEN),
    ("m_deepo",   "K-EXPLORE", "DEEPO 2609.28570", S_OPEN),
    ("m_zppo",    "K-EXPLORE", "ZPPO 2606.18216", S_CONFLICT),
    ("m_moda",    "K-EXPLORE", "MoDA 2609.14896", S_CONSENSUS),
    ("m_es",      "K-EXPLORE", "ES vs GRPO 2608.27351", S_CONSENSUS),
    ("m_epig",    "K-EXPLORE", "EPIG-Tree 2609.20004", S_OPEN),
    ("m_cent",    "K-SYS", "Score Centering 2609.20807", S_CONSENSUS),
    ("m_shadow",  "K-SYS", "Shadow Mask 2605.06850", S_OPEN),
    ("m_brace",   "K-SYS", "BRACE 2609.09783", S_OPEN),
    ("m_opd",     "K-SYS", "OPD-then-RL 2609.04108", S_CONSENSUS),
    ("m_boptr",   "K-SYS", "BOPTR 2609.01274", S_CONSENSUS),
    ("m_rtpo",    "K-SYS", "RTPO 2608.18682", S_CONSENSUS),
    ("m_luck",    "K-METHOD", "Luck Is Not Skill 2609.24144", S_CONSENSUS),
    ("m_spin",    "K-THEORY", "Spin-glass RLVR 2609.28625", S_OPEN),
    ("m_sae",     "K-THEORY", "Fixed-SAE Track 2609.15064", S_OPEN),
    ("m_tame",    "K-THEORY", "TAME 2609.24243", S_CONSENSUS),
    ("m_geom",    "K-THEORY", "Refusal Geometry 2609.06934", S_OPEN),
    ("m_poem",    "K-METHOD", "PoEM 2609.30226", S_OPEN),
    ("m_dutch",   "K-METHOD", "Dutch Books for LMs 2609.02797", S_OPEN),
    ("m_crlhf",   "K-XFER", "Offline Constrained RLHF 2604.00200", S_OPEN),
    ("m_bandit",  "K-XFER", "Compositional Bandits 2608.21993", S_OPEN),
]

# ======================================================================================
# 6. 通用绘制工具（英寸坐标画布：1 数据单位 = 1 英寸，彻底避免宽图把箱子拉扁）
# ======================================================================================
def text_w(label: str, fs: float = 9.0) -> float:
    """粗略估算一行文本的英寸宽度：中文按 1em，拉丁按 0.52em。"""
    cjk = sum(1 for ch in label if ord(ch) > 0x2E80)
    return cjk * fs / 72.0 + (len(label) - cjk) * 0.52 * fs / 72.0


def box_size(label: str, fs: float = 9.0, pad_x: float = 0.22, pad_y: float = 0.16):
    lines = label.split("\n")
    return max(text_w(l, fs) for l in lines) + pad_x, len(lines) * fs * 1.50 / 72.0 + pad_y


def _box(ax, cx, cy, label, color, font, fs=9.0, w=None, h=None, edge="#5c6b7a", lw=1.0, z=3):
    if w is None or h is None:
        w, h = box_size(label, fs)
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=lw, edgecolor=edge, facecolor=color, zorder=z))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, fontproperties=font, zorder=z + 1)
    return w, h


def _arrow(ax, s, e, color="#7b8b9a", lw=0.9, ls="-", rad=0.06, z=1, alpha=1.0):
    ax.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>", mutation_scale=10, linewidth=lw,
                                 color=color, alpha=alpha, zorder=z, shrinkA=1, shrinkB=1,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def _canvas(W: float, H: float):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    return fig, ax


def _save(fig, ax, out_png, spec, cn, font, W, H, note_y=None):
    ax.text(W / 2, H - 0.32, spec["title"] if cn else spec.get("en", spec["title"]),
            ha="center", va="center", fontsize=15, fontproperties=font)
    ax.text(W / 2, note_y if note_y is not None else 0.16, spec["note"],
            ha="center", va="bottom", fontsize=9.5, fontproperties=font, color="#555555")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [str(w.message) for w in caught if "missing from font" in str(w.message)]


# ======================================================================================
# 7. 各图谱的渲染
# ======================================================================================
def render_overview(out_png, cn, font):
    W, H = 17.0, 10.6
    top, bot = H - 0.85, 0.75
    fig, ax = _canvas(W, H)
    col_x = [2.1, 8.0, 14.2]
    for x, name in zip(col_x, ["缺口", "提案", "知识域"]):
        ax.text(x, H - 0.62, name, ha="center", va="center", fontsize=13, fontproperties=font)

    def rows(n):
        return [top - (bot + (top - bot) * (i + 0.5) / n) + (top - bot) * 1.0 for i in range(n)]

    gy = [top - (top - bot) * (i + 0.5) / len(GAPS) for i in range(len(GAPS))]
    py = [top - (top - bot) * (i + 0.5) / len(PROPOSALS) for i in range(len(PROPOSALS))]
    dy = [top - (top - bot) * (i + 0.5) / len(DOMAINS) for i in range(len(DOMAINS))]
    pos = {}
    for g, y in zip(GAPS, gy):
        pos[g["id"]] = (col_x[0], y)
    for p, y in zip(PROPOSALS, py):
        pos[p["id"]] = (col_x[1], y)
    for d, y in zip(DOMAINS, dy):
        pos[d["id"]] = (col_x[2], y)

    edge_color = {"K-OPT": "#3b6ea5", "K-CREDIT": "#c9822b", "K-REWARD": "#c0392b",
                  "K-EXPLORE": "#2e8b57", "K-SYS": "#4a6b7c", "K-THEORY": "#8e44ad",
                  "K-METHOD": "#b8a02a", "K-XFER": "#6c5ce7"}
    # 先画连线（在箱子下面）
    for a, b, _ in GAP_TO_PROPOSAL:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        _arrow(ax, (x1 + 1.15, y1), (x2 - 1.15, y2), color="#9aa7b2", lw=0.8, rad=0.05, alpha=0.75)
    for p in PROPOSALS:
        for d in p["domains"]:
            (x1, y1), (x2, y2) = pos[p["id"]], pos[d]
            _arrow(ax, (x1 + 1.15, y1), (x2 - 1.35, y2), color=edge_color[d], lw=0.75,
                   rad=0.06, alpha=0.45)

    for g, y in zip(GAPS, gy):
        _box(ax, col_x[0], y, g["label"], "#ffffff", font, fs=9.0, w=2.3, h=0.48)
    for p, y in zip(PROPOSALS, py):
        _box(ax, col_x[1], y, p["label"], "#fdf6ec", font, fs=8.8, w=2.3, h=0.44)
    for d, y in zip(DOMAINS, dy):
        _box(ax, col_x[2], y, d["name"] + f"\n({d['id']})", d["color"], font, fs=9.2, w=2.7, h=0.72)

    spec = {"title": "知识图谱·全景：缺口 → 提案 → 知识域",
            "en": "overview: gaps -> proposals -> knowledge domains",
            "note": "灰线＝该提案针对的缺口；彩线＝该提案主要依赖的知识域（颜色＝右下角图例）。节点完整定义见文档附录 C。"}
    lx = 8.0 - 4.0
    for d in DOMAINS:
        ax.add_patch(plt.Rectangle((lx, 0.32), 0.22, 0.16, facecolor=edge_color[d["id"]], edgecolor="none"))
        ax.text(lx + 0.30, 0.40, d["id"], fontsize=9, fontproperties=font, va="center")
        lx += 1.35
    return _save(fig, ax, out_png, spec, cn, font, W, H, note_y=0.72)


def render_knowledge(dom_ids, out_png, cn, font, title, en, ncols=3, note=""):
    cell_w, cell_h, pad = 6.0, 5.2, 0.18
    nrows = (len(dom_ids) + ncols - 1) // ncols
    W, H = ncols * cell_w + 0.4, nrows * cell_h + 1.35
    fig, ax = _canvas(W, H)
    pos = {}
    for pi, did in enumerate(dom_ids):
        r, c = divmod(pi, ncols)
        x0 = 0.2 + c * cell_w
        y0 = 0.60 + (nrows - 1 - r) * cell_h
        dom = DOMAIN_BY_ID[did]
        ax.add_patch(FancyBboxPatch((x0 + 0.08, y0 + 0.08), cell_w - 0.16, cell_h - 0.16,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    linewidth=1.2, edgecolor="#9aa7b2", facecolor=dom["color"],
                                    alpha=0.55, zorder=0))
        ax.text(x0 + cell_w / 2, y0 + cell_h - 0.42, f"{dom['name']}（{did}）",
                ha="center", va="center", fontsize=12.5, fontproperties=font, zorder=1)
        members = [x for x in CONCEPTS if x["domain"] == did]
        ncol_in = 1 if len(members) <= 4 else 2
        nrow_in = (len(members) + ncol_in - 1) // ncol_in
        avail_h = cell_h - 1.05
        for i, m in enumerate(members):
            cc, rr = divmod(i, nrow_in)
            cx = x0 + cell_w / 2 + (cc - (ncol_in - 1) / 2) * (cell_w / ncol_in)
            cy = y0 + cell_h - 0.95 - (rr + 0.5) * (avail_h / nrow_in)
            pos[m["id"]] = (cx, cy)

    colors = {"depend": ("#7b8b9a", "-"), "oppose": ("#c0392b", "--"), "transfer": ("#2e8b57", ":")}
    for a, b, kind in CONCEPT_EDGES:
        if a in pos and b in pos:
            col, ls = colors[kind]
            _arrow(ax, pos[a], pos[b], color=col, lw=1.0 if kind != "oppose" else 1.4,
                   ls=ls, rad=0.12, z=1)

    for m in CONCEPTS:
        if m["id"] not in pos:
            continue
        lab = m["label"] if cn else m["en"]
        if m.get("extra", "").startswith("→"):     # 迁移源：把流向直接画在节点上
            lab = f"{lab}\n{m['extra']}"
        w, h = box_size(lab, 8.6)
        _box(ax, pos[m["id"]][0], pos[m["id"]][1], lab, "#ffffff", font, fs=8.6,
             w=min(w, cell_w / 2 - 0.45), h=min(h, 0.62), z=3)

    spec = {"title": title, "en": en,
            "note": note or "灰实线＝基于/细化；红虚线＝对立；绿点线＝跨域迁移。节点文字取自文档附录 C。"}
    return _save(fig, ax, out_png, spec, cn, font, W, H, note_y=0.20)


def render_conflicts(out_png, cn, font):
    n = len(CONFLICTS)
    W, H = 15.5, n * 2.15 + 1.5
    fig, ax = _canvas(W, H)
    col_x = [2.0, 5.6, 9.5, 13.2]
    col_w = [3.2, 3.2, 3.4, 3.3]
    for cx, h in zip(col_x, ["正方 / 支持", "争论", "反方 / 反例", "判决实验"]):
        ax.text(cx, H - 0.62, h, ha="center", va="center", fontsize=12.5, fontproperties=font)
    top = H - 0.95
    for i, cf in enumerate(CONFLICTS):
        y = top - (i + 0.5) * (n and ((top - 0.95) / n))
        _box(ax, col_x[0], y, cf["left"], "#e9f7ef", font, fs=9.0, w=col_w[0], h=1.15)
        _box(ax, col_x[1], y, cf["question"] + f"\n[{cf['status']}]", "#fff3d6", font,
             fs=10.5, w=col_w[1], h=1.05)
        _box(ax, col_x[2], y, cf["right"], "#fdeaea", font, fs=9.0, w=col_w[2], h=1.35)
        _box(ax, col_x[3], y, cf["test"], "#e8f0fb", font, fs=9.0, w=col_w[3], h=1.15)
        _arrow(ax, (col_x[0] + col_w[0] / 2, y), (col_x[1] - col_w[1] / 2, y),
               color="#2e8b57", lw=1.4, rad=0.0)
        _arrow(ax, (col_x[2] - col_w[2] / 2, y), (col_x[1] + col_w[1] / 2, y),
               color="#c0392b", lw=1.4, rad=0.0)
        _arrow(ax, (col_x[1] + col_w[1] / 2, y - 0.28), (col_x[3] - col_w[3] / 2, y - 0.10),
               color="#7b8b9a", lw=1.1, rad=-0.10)
    spec = {"title": "四条未解决争论：双方、反例与判决实验",
            "en": "four open conflicts and their deciding experiments",
            "note": "状态：[未判决] 缺对照实验；[有反证] 已有反例但未收敛；[可调和] 分歧在条件而非结论；[判决实验优先] 一次 2×2 即可判决。"}
    return _save(fig, ax, out_png, spec, cn, font, W, H, note_y=0.45)


def render_methods(out_png, cn, font):
    ncols, nrows = 4, 2
    cell_w, cell_h = 5.9, 5.6
    W, H = ncols * cell_w + 0.4, nrows * cell_h + 1.6
    fig, ax = _canvas(W, H)
    for pi, p in enumerate(DOMAINS):
        r, c = divmod(pi, ncols)
        x0 = 0.2 + c * cell_w
        y0 = 0.9 + (nrows - 1 - r) * cell_h
        ax.add_patch(FancyBboxPatch((x0 + 0.08, y0 + 0.08), cell_w - 0.16, cell_h - 0.16,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    linewidth=1.2, edgecolor="#9aa7b2", facecolor=p["color"],
                                    alpha=0.5, zorder=0))
        ax.text(x0 + cell_w / 2, y0 + cell_h - 0.45, f"{p['name']}（{p['id']}）",
                ha="center", va="center", fontsize=12.5, fontproperties=font, zorder=1)
        members = [m for m in METHODS if m[1] == p["id"]]
        if not members:
            ax.text(x0 + cell_w / 2, y0 + cell_h / 2, "（本域为迁移来源，\n见知识图谱 C）",
                    ha="center", va="center", fontsize=9.5, fontproperties=font, color="#888888")
            continue
        for i, (_id, _dom, label, status) in enumerate(members):
            cc, rr = i % 2, i // 2
            cx = x0 + cell_w * 0.27 + cc * cell_w * 0.46
            cy = y0 + cell_h - 0.95 - rr * 0.62
            _box(ax, cx, cy, label, STATUS_COLOR[status], font, fs=8.8,
                 w=cell_w * 0.42, h=0.50)
    lx = 1.0
    for st in (S_CONSENSUS, S_CONFLICT, S_OPEN):
        ax.add_patch(plt.Rectangle((lx, 0.35), 0.24, 0.20, facecolor=STATUS_COLOR[st],
                                   edgecolor="#5c6b7a", linewidth=0.9))
        ax.text(lx + 0.34, 0.45, STATUS_LABEL[st], fontsize=10, fontproperties=font, va="center")
        lx += 2.3
    spec = {"title": "方法图谱：八个知识域的代表方法（含 arXiv id 与证据状态）",
            "en": "method map: representative methods with ids and evidence status",
            "note": "状态由本次语料精读判定：绿＝多篇一致或证据较强；红＝存在冲突或反例；灰＝空白/仅摘要级证据。id 可在 arXiv 回查。"}
    return _save(fig, ax, out_png, spec, cn, font, W, H, note_y=0.72)


# 8. Mermaid / 文本 / JSON
# ======================================================================================
def mermaid_overview() -> str:
    L = ["%% 知识图谱·全景", "flowchart LR"]
    L.append("  subgraph GAPS[缺口]")
    for g in GAPS:
        L.append(f'    {g["id"]}["{g["label"]}"]')
    L.append("  end")
    L.append("  subgraph PROPS[提案]")
    for p in PROPOSALS:
        L.append(f'    {p["id"]}["{p["label"]}"]')
    L.append("  end")
    L.append("  subgraph DOMS[知识域]")
    for d in DOMAINS:
        L.append(f'    {d["id"].replace("-", "_")}["{d["name"]}<br/>({d["id"]})"]')
    L.append("  end")
    for a, b, _ in GAP_TO_PROPOSAL:
        L.append(f"  {a} --> {b}")
    for p in PROPOSALS:
        for d in p["domains"]:
            L.append(f'  {p["id"]} -.-> {d.replace("-", "_")}')
    return "\n".join(L) + "\n"


def mermaid_knowledge(dom_ids, title: str) -> str:
    L = [f"%% {title}", "flowchart LR"]
    for d in dom_ids:
        dom = DOMAIN_BY_ID[d]
        L.append(f'  subgraph {d.replace("-", "_")}["{dom["name"]}"]')
        for c in CONCEPTS:
            if c["domain"] == d:
                L.append(f'    {c["id"]}["{c["label"]}"]')
        L.append("  end")
    for a, b, kind in CONCEPT_EDGES:
        if CONCEPT_BY_ID[a]["domain"] in dom_ids and CONCEPT_BY_ID[b]["domain"] in dom_ids:
            arrow = {"depend": "-->", "oppose": "-.->|对立|", "transfer": "==>|迁移|"}[kind]
            L.append(f"  {a} {arrow} {b}")
    return "\n".join(L) + "\n"


def mermaid_conflicts() -> str:
    L = ["%% 四条争论", "flowchart LR"]
    for cf in CONFLICTS:
        L.append(f'  {cf["id"]}["{cf["question"]}<br/>[{cf["status"]}]"]')
        L.append(f'  {cf["id"]}_L["{cf["left"].replace(chr(10), "<br/>")}"]')
        L.append(f'  {cf["id"]}_R["{cf["right"].replace(chr(10), "<br/>")}"]')
        L.append(f'  {cf["id"]}_T["{cf["test"].replace(chr(10), "<br/>")}"]')
        L.append(f'  {cf["id"]}_L --> {cf["id"]}')
        L.append(f'  {cf["id"]}_R --> {cf["id"]}')
        L.append(f'  {cf["id"]} --> {cf["id"]}_T')
    return "\n".join(L) + "\n"


def mermaid_methods() -> str:
    L = ["%% 方法图谱", "flowchart LR"]
    for d in DOMAINS:
        L.append(f'  subgraph {d["id"].replace("-", "_")}["{d["name"]}"]')
        for _id, dom, label, status in METHODS:
            if dom == d["id"]:
                L.append(f'    {_id}["{label}<br/>({STATUS_LABEL[status]})"]')
        L.append("  end")
    for g in GAPS:
        pass
    return "\n".join(L) + "\n"


def txt_overview() -> str:
    out = ["知识图谱·全景：缺口 → 提案 → 知识域", ""]
    for g in GAPS:
        ps = [p for p in PROPOSALS if any(a == g["id"] and b == p["id"] for a, b, _ in GAP_TO_PROPOSAL)]
        out.append(f"[{g['label']}]")
        for p in ps:
            out.append(f"  - {p['label']}  ⇒ 知识域: {', '.join(p['domains'])}")
        out.append("")
    return "\n".join(out) + "\n"


def txt_knowledge(dom_ids, title: str) -> str:
    out = [title, ""]
    for d in dom_ids:
        dom = DOMAIN_BY_ID[d]
        out.append(f"[{dom['name']}（{d}）]")
        for c in CONCEPTS:
            if c["domain"] == d:
                tag = f"  {c['extra']}" if c.get("extra") else ""
                out.append(f"  - {c['id']}: {c['label']}{tag}")
        out.append("")
    out.append("[关系]  depend=基于/细化（实线）  oppose=对立（红虚线）  transfer=跨域迁移（绿点线）")
    for a, b, kind in CONCEPT_EDGES:
        if CONCEPT_BY_ID[a]["domain"] in dom_ids and CONCEPT_BY_ID[b]["domain"] in dom_ids:
            out.append(f"  {a} --{kind}--> {b}")
    return "\n".join(out) + "\n"


def txt_conflicts() -> str:
    out = ["四条未解决争论", ""]
    for cf in CONFLICTS:
        out += [f"[{cf['id']}] {cf['question']}  （状态：{cf['status']}）",
                f"  正方: {cf['left'].replace(chr(10), ' / ')}",
                f"  反方: {cf['right'].replace(chr(10), ' / ')}",
                f"  判决: {cf['test'].replace(chr(10), ' / ')}", ""]
    return "\n".join(out) + "\n"


def txt_methods() -> str:
    out = ["方法图谱", ""]
    for d in DOMAINS:
        out.append(f"[{d['name']}（{d['id']}）]")
        for _id, dom, label, status in METHODS:
            if dom == d["id"]:
                out.append(f"  - {label}  [{STATUS_LABEL[status]}]")
        out.append("")
    return "\n".join(out) + "\n"


def dump_json(path: str) -> bool:
    data = {
        "domains": DOMAINS,
        "concepts": [{k: v for k, v in c.items()} for c in CONCEPTS],
        "concept_edges": [{"from": a, "to": b, "kind": k} for a, b, k in CONCEPT_EDGES],
        "gaps": GAPS,
        "proposals": PROPOSALS,
        "gap_to_proposal": [{"gap": a, "proposal": b} for a, b, _ in GAP_TO_PROPOSAL],
        "conflicts": CONFLICTS,
        "methods": [{"id": i, "domain": d, "label": l, "status": s}
                    for i, d, l, s in METHODS],
    }
    return write_if_changed(path, json.dumps(data, ensure_ascii=False, indent=2))


# ======================================================================================
# 9. 构建与校验
# ======================================================================================
CORPUS = os.path.join(HERE, "rl_corpus.json")


def check_methods(corpus: str = CORPUS) -> list[str]:
    """方法节点的锚定对象是**语料本身**：标签里的 arXiv id 必须真实存在于 rl_corpus.json。

    这比"字符串出现在文档里"更强——它保证图谱里的每篇论文都能回查，而文档不必为 40 个
    方法逐一列出引文。
    """
    import re
    with open(corpus, encoding="utf-8") as fh:
        idx = {e["id"] for e in json.load(fh)}
    problems = []
    for _id, _dom, label, _st in METHODS:
        m = re.search(r"\b(\d{4}\.\d{4,5})\b", label)
        if not m:
            problems.append(f"method:{_id} -> 标签里没有 arXiv id：{label!r}")
        elif m.group(1) not in idx:
            problems.append(f"method:{_id} -> id {m.group(1)} 不在语料里")
    return problems


def check_anchors(doc: str = DOC) -> list[str]:
    with open(doc, encoding="utf-8") as fh:
        text = fh.read()
    missing = []
    for group, items in (("gap", GAPS), ("proposal", PROPOSALS),
                         ("concept", CONCEPTS),
                         ("conflict", CONFLICTS)):
        for n in items:
            if n["anchor"] not in text:
                missing.append(f"{group}:{n['id']} -> {n['anchor']!r}")
    return missing


def build(outdir: str = OUTDIR, force_cn: bool | None = None) -> dict:
    font, cn = setup_font()
    if force_cn is not None:
        cn = force_cn
    os.makedirs(outdir, exist_ok=True)
    jobs = [
        ("kg_overview", lambda p: render_overview(p, cn, font), txt_overview, mermaid_overview),
        ("kg_knowledge_a", lambda p: render_knowledge(
            ["K-OPT", "K-CREDIT", "K-REWARD"], p, cn, font,
            "知识点图谱 A：目标函数 / 信用分配 / 奖励与验证",
            "knowledge map A: objective, credit, reward", 3), lambda: txt_knowledge(
            ["K-OPT", "K-CREDIT", "K-REWARD"], "知识点图谱 A"), lambda: mermaid_knowledge(
            ["K-OPT", "K-CREDIT", "K-REWARD"], "知识点图谱 A")),
        ("kg_knowledge_b", lambda p: render_knowledge(
            ["K-EXPLORE", "K-SYS", "K-THEORY"], p, cn, font,
            "知识点图谱 B：探索覆盖 / 系统漂移 / 理论与诊断",
            "knowledge map B: exploration, systems, theory", 3), lambda: txt_knowledge(
            ["K-EXPLORE", "K-SYS", "K-THEORY"], "知识点图谱 B"), lambda: mermaid_knowledge(
            ["K-EXPLORE", "K-SYS", "K-THEORY"], "知识点图谱 B")),
        ("kg_knowledge_c", lambda p: render_knowledge(
            ["K-METHOD", "K-XFER"], p, cn, font,
            "知识点图谱 C：方法论 / 跨域迁移源（标注流向的提案）",
            "knowledge map C: methodology, transfer sources", 2), lambda: txt_knowledge(
            ["K-METHOD", "K-XFER"], "知识点图谱 C"), lambda: mermaid_knowledge(
            ["K-METHOD", "K-XFER"], "知识点图谱 C")),
        ("kg_conflicts", lambda p: render_conflicts(p, cn, font), txt_conflicts, mermaid_conflicts),
        ("kg_methods", lambda p: render_methods(p, cn, font), txt_methods, mermaid_methods),
    ]
    report = {}
    for name, png_fn, txt_fn, mmd_fn in jobs:
        fd, tmp = tempfile.mkstemp(dir=outdir, suffix=".tmp.png")
        os.close(fd)
        try:
            warns = png_fn(tmp)
            with open(tmp, "rb") as fh:
                blob = fh.read()
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        paths = {"png": os.path.join(outdir, f"{name}.png")}
        changed = write_if_changed(paths["png"], blob)
        changed |= write_if_changed(os.path.join(outdir, f"{name}.txt"), txt_fn())
        changed |= write_if_changed(os.path.join(outdir, f"{name}.mmd"), mmd_fn())
        report[name] = {"bytes": len(blob), "changed": changed, "missing_glyphs": warns}
    json_changed = dump_json(os.path.join(HERE, "knowledge_graph.json"))
    report["knowledge_graph.json"] = {"bytes": os.path.getsize(os.path.join(HERE, "knowledge_graph.json")),
                                      "changed": json_changed, "missing_glyphs": []}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    missing = check_anchors() + check_methods()
    if missing:
        print("锚点缺失（文档附录 C 与图谱数据已漂移）：")
        for m in missing:
            print("  ", m)
        return 2
    if args.check:
        print("锚点全部存在（文档锚点 + 语料 arXiv id）")
        return 0
    rep = build(args.outdir)
    bad = 0
    for name, info in rep.items():
        flag = "written" if info["changed"] else "unchanged"
        print(f"{name}: {info['bytes']} bytes, {flag}")
        if info["missing_glyphs"]:
            print(f"  [ERROR] 缺字形 {len(info['missing_glyphs'])} 条：{info['missing_glyphs'][:3]}")
            bad = 3
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
