# 头脑风暴工作区

这里是两个不相干的项目的共同落脚点，外加一批归档内容：

| 目录 / 文件 | 是什么 | 从哪读起 |
|---|---|---|
| **`transformer/`** | **手把手搭最小 Transformer 的图文教程 + 可跑代码**（CPU 可跑，GPU 快 6 倍） | [`transformer/README.md`](transformer/README.md) · [`transformer/README.pdf`](transformer/README.pdf) |
| `credit_oracle.py` | 「信用分配尺子」：在**真实信用已知**的小 MDP 上对拍几种主流信用估计器 | `python credit_oracle.py` → `credit_oracle_report.json` |
| `knowledge_graph.py` | RL 新策略知识图谱：一份数据源 → PNG / Mermaid / 文本 / JSON 四份同源产物 | `python knowledge_graph.py` → `docs/kg_*.png` |
| `attention_budget_*.py` | 注意力预算的探测与画图（实验记录见 `docs/`） | `python attention_budget_probe.py` |
| `corpus/` | 语料归档区：几份长文按文档结构切开，**同时是训练语料的来源** | [`corpus/STATS.json`](corpus/STATS.json) |
| `docs/` | 所有图（`.png` 可看 / `.svg` 矢量 / `.mmd` Mermaid 源码 / `.txt` 文本大纲） | `docs/flow_pipeline.png`、`docs/tut_hero.png` |
| `slices/` | RL 语料的主题切片 | `slices/CORE.md` |
| `logs/` `runs/` `data*/` | 运行产物（日志、模型权重、token 流）——**不进版本库** | — |

---

## 快速开始

### 1. 最小 Transformer（推荐从这里开始）

```bash
cd /home/yjrrk1989/头脑风暴

# 训一个迷你模型：CPU 约 19 秒 / GPU 约 3 秒
PYTHONPATH=. python -m transformer.train --quick            # 有 GPU 加 --device cuda

# 让它写 200 个字
PYTHONPATH=. python -m transformer.generate --tag quick -n 200

# 31 条不变量测试
PYTHONPATH=. python -m pytest transformer/ -q
```

教程本身（含 8 张配图、实测曲线、踩坑记录）在 [`transformer/README.md`](transformer/README.md)，
同名 PDF 在 [`transformer/README.pdf`](transformer/README.pdf)。

### 2. 研究脚本

```bash
python credit_oracle.py          # 信用估计器对拍
python knowledge_graph.py        # 知识图谱 + 四份同源产物
python attention_budget_probe.py # 注意力预算探测
python -m pytest -q              # 这些脚本的守卫测试
```

---

## 语料为什么被搬进 `corpus/`

工作区根目录原来有几份长文（RL 研究笔记、知识图谱源码、探测脚本、通用流程图脚本），
它们占了内置训练语料的九成，却和学习「怎么搭 Transformer」无关。
于是有个脚本把它们**按文档结构切开归档**到 `corpus/`（**只复制，根目录原件保留**），
再把语料扫描规则收紧成「递归 + 白名单」：

```bash
PYTHONPATH=. python -m transformer.build_corpus    # 归档 + 重建 data/
```

结果：内置语料从 182K 字符（混杂）变成 **208K 字符 / 10.1 万 token**（来源明确：根目录脚本 + 归档笔记 + 教程自身）。
细节见 [`transformer/README.md`](transformer/README.md) 附录 B。

---

## 版本库与 CI

* 远程：<https://github.com/fantasystarlight/-brainstorm-transformer>（分支 `main`）
* 认证：SSH key（`~/.ssh/id_ed25519_fantasystarlight`，走 `ssh.github.com:443`）——没有用 PAT
* **`.gitignore` 只排除可再生产的产物**：`runs/`、`data*/`、`logs/`、`corpus_big/`、`*.pt`、`*.bin`。
  `corpus/` 是**语料来源**（训练管线会扫它），所以入库。
* CI（[`.github/workflows/tests.yml`](.github/workflows/tests.yml)）跑两条线：
  `full`（装了 tokenizers/regex，跑全部测试 + 端到端冒烟）与
  `minimal`（**故意不装** tokenizers/regex，验证手写 BPE 后备路径在裸环境也能跑）。
  两条线都绿才算通过。

---

## 本机环境

| 项 | 值 |
|---|---|
| GPU | RTX 4080 SUPER 16 GB（WSL2，实际可用 ~14.7 GiB） |
| 主力 env | `/home/yjrrk1989/miniconda3/envs/grpo/bin/python`（torch 2.10 / transformers 4.57 / vllm 0.18） |
| 中文字体 | 走 Windows 字体树 `/mnt/c/Windows/Fonts/msyh.ttc`（微软雅黑）；**没有它时所有出图脚本会自动降级成英文标签**，不输出豆腐块 |

⚠️ 基础环境（`python` 不加 env）**没有 torch**，跑任何脚本都用上面的 env 路径或先 `conda activate grpo`。

---

## 约定：图都是脚本生成的

`docs/` 里的每张图都能由脚本重画，**不手画**：

```bash
PYTHONPATH=. python -m transformer.tutorial     # 教程的 8 张图
PYTHONPATH=. python -m transformer.flowchart    # 工程流程图
python knowledge_graph.py                       # 知识图谱
python flowchart.py                             # RL 策略图
```

每个图脚本都遵守三条：① 每个节点/图注带**锚点**，测试会断言它仍存在于对应文档里
（图过期 = 测试失败，而不是留下一张指向不存在内容的图）；② 同一个名字输出多份格式，互为同源；
③ 内容没变就不改文件（write-if-changed + 原子替换），否则侧栏里已打开的图会失效。
