# 手把手搭建一个最小 Transformer

这套代码在你的机器上**真的能训练出东西**：18 万字符的中英混合语料（工作区里的 md/py 文件）、420 万参数、CPU 上两分钟，
训练完它能写出「长得像这个工作区里的代码/文档」的文本。

本文是教程本体，按顺序边讲边做。全部产物都在 `transformer/` 目录下。
**CPU 上就能跑通**；有 GPU 就加一个 `--device cuda`，同样的代码快 15~25 倍（§11 有实测）。

---

## 0. 五分钟跑通（先看到结果，再回头理解）

```bash
cd /home/yjrrk1989/头脑风暴

# ① 300 步的迷你版：先确认整条链路是通的（CPU 45 秒 / GPU 3 秒）
PYTHONPATH=. python -m transformer.train --quick
PYTHONPATH=. python -m transformer.train --quick --tag quick_gpu --device cuda   # 有卡就加这一行

# ② 让它写点东西
PYTHONPATH=. python -m transformer.generate --tag quick -n 200

# ③ 看它快不快（KV cache 的威力）
PYTHONPATH=. python -m transformer.generate --tag quick --time -n 200

# ④ 正式一点的模型：2000 步、3.3M 参数、上下文 256
PYTHONPATH=. python -m transformer.train --steps 2000 --tag base

# ⑤ 跑测试（21 条不变量，见 §7）
PYTHONPATH=. python -m pytest transformer/ -q

# ⑥ 生成流程图（PNG + Mermaid + 文本大纲）
PYTHONPATH=. python -m transformer.flowchart
```

> 用哪个 python：`/home/yjrrk1989/miniconda3/envs/grpo/bin/python`（基础环境没有 torch）。
> 在 `grpo` 环境里直接 `python` 即可；上面写 `PYTHONPATH=.` 是因为包在当前工作区根目录。

**这套东西一共 8 个文件**，加起来不到 1500 行：

| 文件 | 干什么 | 关键概念 |
|---|---|---|
| `transformer/bpe.py` | 手写字节级 BPE 分词器 | 把文本变成整数序列，可逆 |
| `transformer/corpus.py` | 收集本地文本、切 train/val | 切分**必须**在训分词器之前 |
| `transformer/data.py` | 分词 → token 流 → 采样 batch | `y = x` 右移一位 |
| `transformer/model.py` | 模型本体 | RMSNorm / RoPE / 因果注意力 / SwiGLU / 残差 |
| `transformer/train.py` | 训练循环 | 交叉熵、梯度裁剪、warmup+余弦 |
| `transformer/generate.py` | 加载权重生成 | 温度、top-k、KV cache |
| `transformer/ladder.py` | 缩放阶梯实验 | 固定数据与步数，只改模型大小（§11） |
| `transformer/fetch_corpus.py` | 抓更大的公开语料 | 换数据是提升质量的第一手段（§10） |
| `transformer/flowchart.py` | 生成流程图 | 图必须有锚点，代码改了图就失效 |
| `transformer/test_*.py` | 18 条不变量测试 | 因果性、RoPE、cache 等价、过拟合 |

![训练流水线](docs/flow_pipeline.png)

---

## 1. 先想清楚：Transformer 到底在算什么

一句话：**它是一个「下一个 token 的概率分布」函数**。

$$
p(t_{n+1} \mid t_1, t_2, \dots, t_n)
$$

就这么一个目标。没有别的花活：训练时把「下一个词」当标签（自监督），推理时把预测出的词拼回输入再预测下一个（自回归）。
所以整套代码里最核心的三行是：

```python
logits, loss, _ = model(x, y)   # x 是输入 token，y 是「x 右移一位」
loss.backward()                 # 反向传播
optimizer.step()                # 更新参数
```

而 `loss` 就是交叉熵：`y` 里每个位置都是一个「在词表 V 里选哪个」的分类问题。

### 数据形状约定（看代码时对照这张表，全文统一）

| 符号 | 含义 | 本教程取值 |
|---|---|---|
| `B` | batch size | 16 |
| `T` | 序列长度（上下文） | 256 |
| `C` | `d_model`，每个位置的向量维度 | 256 |
| `H` | 注意力头数 | 8 |
| `hd` | `head_dim = C / H` | 32 |
| `V` | 词表大小 | 2048~2306 |

| 张量 | 形状 | 含义 |
|---|---|---|
| `ids` / `x` / `y` | `[B, T]` | token 编号 |
| `h` | `[B, T, C]` | 每个位置一个向量 |
| `q` / `k` / `v` | `[B, H, T, hd]` | 拆头之后 |
| `attn` | `[B, H, T, T]` | 每个 query 对所有 key 的权重（每行和为 1） |
| `logits` | `[B, T, V]` | 每个位置一份「下一个词」的打分 |

**只有 5 个形状**，剩下都是在这几个之间变来变去。

---

## 2. 第 1 步：把文本变成数字（`bpe.py`）

模型只吃整数。最朴素的方案是「每个字符一个编号」，但中文和代码这样词表太大、序列太长。
工业界的答案是 **BPE**（Byte Pair Encoding）：先按字节看（0~255），然后反复把「最常一起出现的相邻对」合并成一个新符号。

```
"注意力" → UTF-8 字节 e6 b3 a8 e6 b3 a8 e5 8a 9b
         → 合并 (e6,b3) 出现两次 → [X a8 X a8 e5 8a 9b]  （X 是新的编号 256）
         → 再合并 (X,a8) → ... → 最后常见汉字变成 1 个 token
```

自己实现只要 130 行，核心两步：

```python
# 训练：反复挑最高频的相邻对合并
pairs, counts = _pair_stats(tokens)
left, right = pairs[argmax(counts)]
tokens = _merge_all(tokens, left, right, new_id)   # 新 id = 256 + 第几次合并

# 编码：按「学到的合并顺序」依次施加合并（顺序很重要，不能乱）
# 解码：每个 id 展开成字节，拼起来再 UTF-8 解码 → 所以严格可逆
```

三个容易踩的坑，代码里都有注释：

1. **相邻对要按「不重叠」的方向合并**：`a a a` 里的 `(a,a)` 只能合一次，不是两次。
   我第一版用 numpy 的 fancy indexing 写这个，在某个前缀下索引越界了——现在改成显式扫描。
2. **标准库 `re` 不支持 `\p{L}`**，GPT-2 的切分正则必须用第三方 `regex` 模块（装 `tokenizers` 时会带上）。
3. **特殊 token 要占固定 id**：`<|bos|>` / `<|eos|>`，生成时用 bos 起头、遇 eos 停。

本教程的数据管线里，**优先用 HuggingFace `tokenizers` 的 ByteLevel-BPE**（和 Qwen/GPT 同款实现、更快、特殊 token 更规范），
手写那份 `bpe.py` 作为**零依赖后备**：环境里没有 `tokenizers` 也能跑通全流程。
两者都写了 round-trip 测试（§7）。

**一个容易被误判的现象**：本语料词表只有 2048，汉语覆盖有限，所以单个 token 有时会切在一个汉字中间
（例如 `"注意力机制就是把历史信息按相似度加权搬进当前位置。"` 编成 24 个 token，其中几个单独解码会是 `�`）。
**这不是 bug**：一个汉字的 3 个 UTF-8 字节被分到两个 token 里，只要**整段** encode→decode 能还原就完全正确
（`tok.decode(ids) == 原文`，round-trip 测试就是在守这条）。想减少这种切分就把词表调大、语料换成中文占主导的文本。

```bash
# 只看手写 BPE 的自测
PYTHONPATH=. python -m transformer.bpe
# 输出：demo: 69 字符 → 36 tokens，roundtrip OK
```

---

## 3. 第 2 步：切分与打包（`corpus.py` / `data.py`）

**顺序错误会毁掉你所有的评测数字**：

```
❌ 先用全部文本训 BPE，再切 train/val  → 分词器见过验证集，val loss 虚低
✅ 先在原始文本上切 train/val，再只用 train 训 BPE
```

`corpus.py` 把工作区根目录的 `*.py` / `*.md`（15 个文件、18.2 万字符）拼成一份语料，取后 5% 作验证集；
`data.py` 把两段文本编码成 `uint16` 的 `.bin`，然后随机采样训练窗口：

```python
class TokenStream:
    def batch(self, batch_size):
        ix = torch.randint(len(self), (batch_size,))            # 随机窗口起点
        x = np.stack([self.data[i:i + T] for i in ix])
        y = np.stack([self.data[i + 1:i + 1 + T] for i in ix])  # 标签 = 右移一位
        return torch.from_numpy(x), torch.from_numpy(y)
```

数据是**流式随机采样**的，没有 epoch 概念：把语料当成一条无限长的带子，随机剪窗口。这样不会过拟合到固定顺序。

量化一下你的语料：

```bash
PYTHONPATH=. python -m transformer.data     # 重建数据（会重训 BPE）
cat data/meta.json
```

---

## 4. 第 3 步：模型本体（`model.py`）

整个模型 = Embedding + N × Block + 输出层。一个 Block 只有两条支路：

```python
class Block(nn.Module):
    def forward(self, x, cos_sin, past_kv=None):
        a, kv = self.attn(self.norm1(x), cos_sin, past_kv)
        x = x + a                        # 残差 1：注意力
        x = x + self.ffn(self.norm2(x))  # 残差 2：前馈
        return x, kv
```

![模型本体](docs/flow_model.png)

下面按「为什么这么设计」讲每个部件。

### 4.1 RMSNorm：把向量缩放到单位尺度

```python
x / sqrt(mean(x^2) + eps) * weight
```

- 在**最后一维**（特征维）上求均方，与 batch/序列长度无关；
- 相比 LayerNorm：不减均值、不加偏置。少算一点，效果一样好，是 Llama 之后的默认；
- `weight` 初值全 1，所以训练开始时它是恒等变换。

测试里和 `torch.nn.RMSNorm` 逐元素对齐（注意 PyTorch 默认 `eps` 不同，要对齐得显式传参）。

### 4.2 RoPE：用「旋转」注入位置

注意力本身是**置换不变**的（打乱输入顺序，输出也跟着打乱，但内容不变），所以必须显式加位置信息。
RoPE 的做法很巧：把 `head_dim` 维两两配成 `hd/2` 个二维平面，第 `i` 对按角度 `θ_i × 位置` 旋转：

$$
\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} =
\begin{pmatrix} \cos & -\sin \\ \sin & \cos \end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
$$

于是两个位置的 q·k 内积里，**角度只以 (m − n) 的形式出现**——天然是「相对位置」。
这就是测试 `test_rope_is_relative` 验的性质：两个位置同时平移，分数不变；距离变了，分数就变。

```python
def apply_rope(x, cos_sin, offset=0):
    cos, sin = cos_sin[:, offset:offset + x.shape[-2]]   # offset 是 KV cache 的位置偏移
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
```

### 4.3 因果自注意力：本项目的心脏

```python
q, k, v = self.qkv(x).split(C, dim=-1)                  # 一次线性投影，拆 q/k/v
q, k, v = (t.view(B, T, H, hd).transpose(1, 2) for t in (q, k, v))
q, k = apply_rope(q, cos_sin, offset), apply_rope(k, cos_sin, offset)
if past_kv is not None:                                  # 增量解码：接上历史 k/v
    k, v = torch.cat([past_kv[0], k], -2), torch.cat([past_kv[1], v], -2)
mask = torch.triu(torch.full((Tq, Tk), -inf), diagonal=1 + (Tk - Tq))
attn = F.softmax(q @ k.transpose(-2, -1) / sqrt(hd) + mask, dim=-1)
out = (attn @ v).transpose(1, 2).reshape(B, Tq, C)
return self.proj(out), (k, v)                            # 把本步 k/v 交给 cache
```

![注意力控制流](docs/flow_attention.png)

四个必须理解的细节：

1. **为什么要除以 `sqrt(hd)`**：`q·k` 是 `hd` 个乘积之和，方差随 `hd` 线性增长。
   不缩放的话 softmax 会被推到饱和区（几乎 one-hot），梯度消失。
2. **因果掩码用 `-inf` 而不是 0**：softmax 之前加上 `-inf`，那些位置的概率正好是 0；
   加 0 是没用的（softmax 里 exp(0)=1，照样分到权重）。
3. **`diagonal=1+(Tk-Tq)`**：这一项让同一个 `forward` 同时支持两种模式——
   训练/预填充时 `Tq=Tk`，是标准的上三角；增量解码时 `Tq=1`，掩码整体平移，正好等于「全部可见」。
4. **`v` 不加 RoPE**：位置信息只需要影响「谁该关注谁」（q·k 打分），不需要污染被搬运的内容。

本项目是 MHA（`n_kv_heads == n_heads`）。想改成 Llama 的 **GQA** 只需两步：
`k/v` 投影输出改成 `n_kv_heads*hd`，取出后 `repeat_interleave(n_heads // n_kv_heads, dim=1)`。
KV cache 的显存就降为 `1/(n_heads/n_kv_heads)`。

### 4.4 SwiGLU 前馈：真正存知识的地方

```python
self.down(F.silu(self.gate(x)) * self.up(x))
```

注意力只做「位置之间的加权平均」（本质是线性的）；单个位置上的非线性变换、事实知识都在 FFN 里，
所以 FFN 参数占一个大头（本项目每层 `4*d*h` ≈ `8d²` 中的绝大部分）。

参数量速算（本实现，`h = ceil(8d/3)` 向上取整到 64 的倍数）：

$$
\text{params} \approx \underbrace{12 L d^2}_{\text{每层}} + \underbrace{2 V d}_{\text{嵌入 + 输出头}}
$$

`d=256, L=4, V=2306` 时约 **3.3M**。测试里对这条公式做了 6% 容差的断言，
并在 `train.py` 开头打印实际参数量——**先知道自己在建多大的模型**。

### 4.5 残差 + Pre-Norm：为什么能堆深

```python
x = x + Sub(Norm(x))
```

- 残差给出了一条「恒等高速公路」，梯度可以不过任何非线性层直接回流到底层；
- Norm 放在子层**之前**（Pre-Norm）比放在之后收敛更稳，是现代默认。

---

## 5. 第 4 步：训练循环（`train.py`）

一个 step 就 6 行，但每一行都有理由：

```python
x, y = stream.batch(B)                       # ① 采样
logits, loss, _ = model(x, y)                # ② 前向 + 交叉熵
loss.backward()                              # ③ 反向：自动微分
gnorm = clip_grad_norm_(model.parameters(), 1.0)  # ④ 梯度裁剪
optimizer.step()                             # ⑤ AdamW 更新
optimizer.zero_grad(set_to_none=True)        # ⑥ 清梯度（不清会累加！）
```

**学习率调度**：线性 warmup + 余弦衰减。刚开始参数随机、梯度方向不可信，先小步升上来；
后期按余弦降到峰值的 10%。这不是玄学，是 LLM 预训练的标准配方：

```python
def lr_at(step, total, peak, warmup, min_ratio=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / (total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + cos(pi * min(prog, 1.0))))
```

**指标怎么读**（下表的数字是本机 `--steps 2000 --tag base` 的**实测**结果，不是估的）：

| 指标 | 含义 | 实测 |
|---|---|---|
| `loss` | 当前 batch 的交叉熵（nats） | 7.68（step 0）→ 0.08（step 2000） |
| `train` / `val` | 采样估计的平均损失 | 最低 val **3.95** 出现在 step 400；末步 train 0.08 / val 4.70 |
| `ppl = exp(val)` | 困惑度：模型在多少个候选之间「犹豫」 | 最好 52；末步 110 |
| `bits per byte` | **跨分词器可比**的指标：`loss/ln2 ÷ (每 token 字节数)` | 最好约 **2.55**；末步 2.70 |
| `|g|` | 裁剪前的梯度范数 | 1~3（有裁剪兜底，不会爆） |

> 为什么要有 `bits per byte`：困惑度依赖分词粒度，词表大小不同没法直接比。
> bpb 把「每个字节的不确定度」算出来，和 GPT-2 论文里的数字是同一口径。
> 本语料上 val 4.7 → bpb 2.7，意思是「平均每个字节还有 2.7 bit 的不确定度」——一个可比的基线。

**判断训练是否健康**（三条经验）：
- `train` 和 `val` 一起降 → 正常；`train` 降、`val` 升 → **过拟合**（本教程 86K token 的语料在 step 400 之后就是这个状态）；
- `val` 完全不动 → 学习率太小或数据/标签错位（**检查 `y` 是不是真的右移了一位**）；
- loss 变成 `nan` → 学习率太大或没有裁剪/没有缩放点积。

### 过拟合：这是本教程最该记住的一课

`runs/base/history.json` 里的真实曲线（每隔 100 步取一次）：

| step | 100 | 200 | 300 | **400** | 500 | 700 | 1000 | 1500 | 2000 |
|---|---|---|---|---|---|---|---|---|---|
| train | 6.01 | 4.35 | 3.17 | **2.43** | 1.71 | 0.85 | 0.29 | 0.11 | **0.08** |
| val | 6.10 | 4.79 | 4.17 | **3.95** | 4.09 | 4.09 | 4.40 | 4.53 | **4.70** |

训练集 86K token、每步 4096 token，2000 步等于把语料来回看了 **95 遍**——
模型把它背下来了（train 0.08），但验证集一点没变好，反而越来越差。
**这不是代码 bug，是数据不够**。三件事必须记住：

1. **`val` 反升就该停**，`train.py` 已经自动把最好的一步存成 `ckpt.pt`（本例是 step 400），
   所以「多训了」不会毁掉模型——但要知道 `last.pt` 不是你能拿到的最好权重；
2. **想知道自己有没有过拟合，必须要有 val 集**，而且 val 必须对分词器保密（§3）；
3. **小语料上，加正则有点用，但加数据才是解药**。本机做了一个严格对照
   （同样的数据、同样的 600 步余弦调度，只改 dropout；`--eval-batches 16` 让 val 估计噪声更小）：

   | 配置 | 最好 val | 对应 step | bpb | 墙钟 |
   |---|---|---|---|---|
   | `--dropout 0.0`（对照） | 3.984 | 599 | 2.287 | 295 s |
   | `--dropout 0.2` | **3.821** | 550 | 2.241 | 447 s |

   dropout 带来 **4.1% 的 val 改善**（3.984 → 3.821），同时训练变慢 51%。
   结论要说清楚：**有效，但幅度不大**——因为瓶颈不是容量也不是正则，而是**语料只有 86K token**。
   想真正变好，去 §10 换更大的语料，而不是继续拧 dropout。

跑正式训练：

```bash
PYTHONPATH=. python -m transformer.train --steps 2000 --tag base --log-every 100
```

产物在 `runs/<tag>/`：`ckpt.pt`（val 最好的一步）、`last.pt`、`history.json`（每个记录点的 lr/loss/val/梯度范数）。
`history.json` 是「声明了就必须接线」的检查手段：如果 lr 是平线，说明调度没接上。
**汇报结果时请引用 `ckpt.pt` 的 val，而不是 `last.pt` 的**——这是两个不同的模型。

---

## 6. 第 5 步：生成与 KV cache（`generate.py`）

```bash
PYTHONPATH=. python -m transformer.generate --tag quick -p "# " -n 200
PYTHONPATH=. python -m transformer.generate --tag quick --time -n 200   # cache 对比
```

三个旋钮：

| 旋钮 | 作用 | 建议 |
|---|---|---|
| `temperature` | logits 除以 T。小=保守、大=胡言乱语 | 0.7~1.0 |
| `top_k` | 只在概率最高的 k 个里采样，砍掉长尾垃圾 | 40；想复现就 `top_k=1, temperature→0` |
| `use_cache` | 每步只前向 1 个 token，复用历史 k/v | 默认开 |

**KV cache 为什么快**：不缓存时，生成第 n 个 token 要把前 n 个重新算一遍，前向总计算量随长度**平方**增长；
缓存之后每步只算 1 个新 token（注意力仍要看全部历史 key，所以是线性增长）。

本机实测（CPU，`python -m transformer.generate --tag base --time -n 200`）：

| 路径 | 200 token 耗时 | 速度 |
|---|---|---|
| `use_cache=True` | 2.61 s | 76.6 tok/s |
| `use_cache=False` | 6.80 s | 29.4 tok/s |

也就是这个规模下约 **2.6 倍**；上下文越长差距越大（这是乘性项，不是常数项）。
诚实起见说明测量条件：小模型 + 短上下文时，单步的固定开销占比大，所以倍数看起来不夸张；
如果把 `max_seq_len` 提到 2048、模型提到 100M 参数，no-cache 会被平方项拖垮，差距才会拉到几十倍。

**但 cache 是最容易写错的地方**，两个坑：

1. **位置偏移**：增量步里 q/k 的 RoPE 必须用「它在整段里的绝对位置」，不能从 0 重新数。
   代码里用 `offset = past_kv[0].shape[-2]`。
2. **掩码平移**：预填充之后 `Tq=1`，不能用 `triu(diagonal=1)`，否则唯一那个 query 会被全部屏蔽成 NaN。

这两点都由 `test_kv_cache_matches_full_forward` 断言：**增量解码的 logits 必须与整段重算一致（1e-5）**。
写完任何 cache 逻辑，先跑这条测试，别靠肉眼。

**真实输出长什么样**（`--tag base`，即那个 4.2M 参数、best val 3.95 的模型，`temperature 0.7`）：

```
$ python -m transformer.generate --tag base -p "# " -n 120
#  {re.get("c_pd", "pose"), ("c_rea", "c_advantage", "depend"),
    ("c_c_c_setn", "c_curve", "depend"), ("c_axi", "c_ent", "c_hack", "depend"),
    ("c_x_axi", "depend"),
    ("c_h", "c_c_hack", "depend"), ("c_mean", "c_tsr", "c_qd", "depend"),
```

它学会了**这个语料的形状**：`#` 开头的注释、Python 的元组与字典、缩进、括号配对、标识符命名风格。
这已经是「一个能用的语言模型」的雏形了——它不知道自己在说什么，但它知道「这里该出现什么形状的东西」。
再往下（更大语料 + 更多步数）才会开始出现真正的词与句法。

---

## 7. 第 6 步：怎么证明「我搭对了」

```bash
PYTHONPATH=. python -m pytest transformer/ -q     # 21 passed
```

21 条不变量分成两类。**关键性质写成断言**，而不是「跑一遍看着像」：

| 测试 | 验证的性质 | 为什么需要它 |
|---|---|---|
| `test_handwritten_bpe_roundtrip` | 手写 BPE 严格可逆（含中文/emoji/空白） | 分词错了后面全错，而且要能被发现 |
| `test_handwritten_bpe_negative_control` | 越界 id 必须解成替换符 | 证明检查器有鉴别力 |
| `test_rmsnorm_matches_torch` | 与官方实现逐元素一致 | 别自己发明归一化 |
| `test_attention_is_causal` | **改未来 token，过去的输出逐位不变** | 因果性只有这条能证明；配了「改未来必须有变化」的反向断言 |
| `test_causal_mask_rows_are_lower_triangular` | 上三角权重恒为 0、每行和为 1 | 掩码是否真的生效 |
| `test_rope_is_relative` | 两位置同平移分数不变；距离不同分数不同 | RoPE 的定义性质 + 负对照 |
| `test_kv_cache_matches_full_forward` | 增量解码 == 整段重算 | cache 的正确性 |
| `test_generate_past_context_window_with_cache` | 生成超过 `max_seq_len` 时滑窗不崩、与 no-cache 一致 | 真踩过的坑：cache 比可见上下文长 → `attn @ v` 报尺寸错 |
| `test_generate_with_and_without_cache_agree` | 同种子下 cache/no-cache 生成完全相同 | 端到端一致 |
| `test_overfit_tiny_sequence` | 400 步背下 64 token 的随机序列 | 模型有足够容量；**换一条序列损失必须 >1**（证明真的记住了） |
| `test_param_count_scales_as_expected` | 参数量公式 | 防止「以为建了个大模型」 |
| `test_gqa_includes_mha_as_special_case` | GQA 在 n_kv_heads=n_heads 时与 MHA **逐位等价**（+ 负对照） | 分组注意力的实现极易写错，必须与 MHA 对拍 |
| `test_gqa_shrinks_params_and_cache` | kv 头数越少，参数与 cache 按比例变小 | GQA 的唯一目的就是省这个 |
| `test_anchors_exist_in_src` 等 5 条 | 流程图锚点、三产物同源、渲染幂等、依赖图与 import 一致 | 防止文档/图与代码漂移 |

**「过拟合一条小数据」是搭模型时最有用的调试手段**：如果连 64 个 token 都背不下来，
问题一定在代码里（掩码错了、标签没错位、初始化太糟、LR 不对），而不是在数据上。

---

## 8. 流程图（`flowchart.py`）

```bash
PYTHONPATH=. python -m transformer.flowchart          # 生成
PYTHONPATH=. python -m transformer.flowchart --check  # 只校验锚点
```

四张图输出到 `docs/`，一个名字三份（`.png` 可看 / `.mmd` 可 diff / `.txt` 可检索）：

| 图 | 内容 |
|---|---|
| `flow_pipeline` | 数据→分词→batch→训练的流水线，含反向边 |
| `flow_model` | 模型本体：ids → block → logits |
| `flow_attention` | 注意力的控制流：投影→RoPE→掩码→softmax→输出，含 cache 分支 |
| `dep_modules` | 模块依赖图（**由源码 import 自动解析**，不手写） |

三条纪律（都在测试里守着）：

1. **每个节点带一个锚点字符串**，测试断言它仍存在于对应源码文件里 → 改代码不改图 = 测试失败；
2. **边只定义一次**，PNG 与 Mermaid 从同一份数据渲染（否则会出现「图里连 A、文本里连 B」）；
3. **生成必须 write-if-changed + 原子替换**，否则每次重跑都刷新 mtime，侧栏里已打开的图会失效。

---

## 9. 接下来往哪走（按性价比排序）

| 想改什么 | 怎么改 | 实测/预期效果 |
|---|---|---|---|
| **想更好（第一优先级）** | 换更大语料：`--data 大文件.md --reprepare`（见 §10） | 这是唯一的解药；86K token 已经到了天花板 |
| 想更快 | `--device cuda`（本机 WSL2 可用） | 快 20~50 倍，可以跑 20M 参数 |
| 想更稳 | `--dropout 0.2` | 实测 val 3.984 → 3.821（−4.1%），但训练慢 51% |
| 想更早停 | 看 `history.json` 的 val，取 `ckpt.pt` 而不是 `last.pt` | 免费；本例 step 400 的 val 3.95 vs 末步 4.70 |
| 想省显存 | 把 MHA 改成 GQA（§4.3） | KV cache 降到 1/4 |
| 想更大 | `--d-model 512 --n-layers 8 --n-heads 8` | ~30M 参数；但**先解决数据**，否则只是背得更快 |
| 想更现代 | 加 QK-Norm、把 LM head 与 embedding 权重绑定（weight tying） | 小模型上通常更稳 |
| 想懂训练 | 打开 `runs/*/history.json`，把 lr / loss / 梯度范数画出来 | 调参靠它，不靠感觉 |

**这套代码故意没做的事**（都不影响它是个完整的 Transformer）：没有 flash-attention
（用 SDPA 就能自动用上）、没有混合精度、没有分布式、没有 RLHF。等你把这 5 个形状、
2 条残差、1 个掩码彻底吃透，那些都是可以按需加装的外挂。

---

## 10. 换更大语料（一行命令）

数据是唯一的瓶颈，所以换个语料只要指个文件就行：

```bash
# 用工作区里那个 4MB 的 RL 语料（实测 290 万字符 → 193 万 token，比内置语料大 22 倍）
# ⚠️ 一定要配 --datadir：否则会覆盖内置语料的 data/，让 quick/base 那些权重对不上分词器
PYTHONPATH=. python -m transformer.train --datadir data_rl --data ../rl_corpus.md --reprepare \
    --vocab-size 4096 --steps 3000 --tag big --dropout 0.1 --log-every 100

# 也可以用任意多个文件
PYTHONPATH=. python -m transformer.train --datadir data_mixed --data a.md b.txt c.py --reprepare --tag mixed
```

原理和内置流程完全一致（`data.build_from_files`）：拼文本 → **先在原始文本上切** train/val →
只用 train 训 BPE。所以「换成更大的语料」不会破坏评测的诚实性。

> **每个语料配一个 `--datadir`**，这是本教程踩过的坑：数据目录里躺着 `tokenizer.json` /
> `train.bin` / `meta.json`，用 A 语料训的权重配上 B 语料的分词器，生成的会是乱码，
> 而且**不报错**。生成时也要指同一个目录：
> `python -m transformer.generate --tag big --datadir data_rl -p "..."`。

想再大一点又不想下载数据集，就把工作区里所有 `.md` / `.py` 都喂进去：

```bash
PYTHONPATH=. python -m transformer.train --datadir data_all --reprepare --data $(ls ../*.md ../*.py) --tag all
```

要判断「数据变多有没有用」，只看两个数：**`ckpt.pt` 对应的 val** 和 **`history.json` 里的 bpb**。
如果 val 还在一路下降（没有反弹），说明数据还没喂饱——继续加大语料、加步数；
如果 val 早早反弹，那就是数据不够，调模型结构没用。

### 实测：换大 22 倍的语料之后（本机真跑过）

```bash
PYTHONPATH=. python -m transformer.train --datadir data_rl --data ../rl_corpus.md \
    --reprepare --vocab-size 4096 --steps 1200 --tag big --dropout 0.1 --log-every 100
# 824 s（CPU），5.31M 参数（词表变大），best val 4.199 @ step 1100
```

| step | 0 | 200 | 400 | 600 | 800 | **1000** | **1200** |
|---|---|---|---|---|---|---|---|
| train | 8.36 | 5.68 | 4.66 | 4.29 | 3.91 | 3.90 | **3.83** |
| val | 8.37 | 5.83 | 4.87 | 4.52 | 4.39 | 4.25 | **4.20** |

两个关键观察：

1. **`train − val` 的差距一直只有 0.4~0.5，而且 val 到最后还在降**——完全没有过拟合。
   对比内置语料的 base run（train 0.08 / val 4.70，差距 4.6），差别一目了然：
   **过拟合与否，取决于数据量，而不是步数**。
2. **困惑度会骗人，bits per byte 不会**：big run 的 `ppl` 是 66.8，比 base600 的 53.7 **更差**；
   但 bpb 是 **1.614**，比 base600 的 2.287 **好得多**（−29%）。
   因为 4096 词表一个 token 携带更多字节，困惑度天然变大。
   **跨分词器比较只能看 bpb**。

生成的文本也明显上了台阶（同一个 4.2M 参数的模型，只是换了数据）：

```
$ python -m transformer.generate --tag big --datadir data_rl -p "强化学习" -n 120
强化学习并优化，并在Llama-Bench上达到60.36，平均后将14.5-个百分和0.29%和15.10倍。
- innovations_zh: ['提出显式强化学习框架，通过结构化参数化并与检索反馈库，支持检索与LLM感知奖励']
- significance: 该工作填补了ScAUMLLM评估中长程语言模型可能成为模型推理模型的训练提供了可行性，
  为LLM在复杂推理任务中的鲁棒性与可扩展性。其系统，为LLM评估和模型蒸馏提供了重要启发……
```

中文通顺、术语（Llama-Bench / 检索 / 奖励 / 鲁棒性 / 蒸馏）用得基本对、列表结构对——
**这就是「数据 > 结构」在 5M 参数规模上的直观体现**。

---

## 11. 上 GPU：实测加速与踩到的两个显存巨坑

本机是 RTX 4080 SUPER（16 GB，WSL2 里可用 ~14.7 GiB）。**代码一行没改**，只加 `--device cuda`：

| 任务 | CPU | GPU | 加速 |
|---|---|---|---|
| `--quick`（300 步，0.95M 参数） | 45.2 s | **3.1 s** | 14.6× |
| 缩放阶梯最大档（24.7M 参数、512 上下文、3000 步 × 16k token） | 数小时（估） | **600 s** | — |

而且 GPU 上跑出来的 loss 与 CPU **逐位一致**（step 100/150/200 的 train/val 完全相同），
说明设备切换没有引入任何数值差异——这一点值得每次都验一下，比「看起来能跑」可靠。

### 坑 1：显存爆炸不在注意力，而在输出层

第一次在 GPU 上跑 `d_model=384 / seq_len=512 / batch=32` 直接 OOM：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.12 GiB.
```

算一下就知道 6.12 GiB 是谁：``B*T*V*4 字节 = 32*512*100279*4 ≈ 6.1 GiB``。
**是 `lm_head` 的 logits**（词表 10 万 + 512 上下文 + 32 batch），而反向还要再存一份。
我一开始以为是注意力矩阵（`B*H*T*T`）太大，其实那个只有 402 MB。

两个修复（都在 `model.py` 里）：

1. **用 `F.scaled_dot_product_attention`** 代替手写 `softmax(qk^T/√d + mask) @ v`：
   掩码改成**布尔**语义（`True = 可见`），由 SDPA 挑 flash / mem-efficient kernel，
   不再物化 `[B,H,T,T]`。顺带一个小优化：增量解码（`Tq==1` 且带 cache）时**根本不需要掩码**。
2. **分块算 loss**：按时间维每 64 步一块，`sum-CE / n` 得到整批平均（梯度完全等价），
   峰值从 `B*T*V` 降到 `B*chunk*V`，直接省 8 倍；再加一条：
   **训练时只返回最后一个位置的 logits**，因为中间位置的 logits 只有生成才用得上。

修完之后同一配置的显存占用降到 ~11 GiB（含 PyTorch 上下文），跑满 98% 利用率、220 W。

### 坑 2：大词表让小模型「白养」一个输出层

cl100k 的 `lm_head` 是 `d × 100279`，`d=512` 时**光是输出层就 5100 万参数**，
比整个主干还大；而语料只有 58 万 token，于是模型整天在学一个几乎用不到的巨型分类器。
两个对策都实现了：

* `--tie-weights`：输出层与嵌入层**共享权重**，省掉 `V*d` 个参数（`d=512` 时省 5100 万）；
* 在 2.5M 字符的语料上，用**自训的 vocab-4096 BPE** 反而比 cl100k 更划算（紧凑一倍）。

### GQA：真正省的是生成显存

`--n-kv-heads` 把 k/v 头数降下来（`n_heads` 的 1/4、1/8、甚至 1 = MQA）。
参数只省一点，**KV cache 按比例缩小**：cache 里存的就是 `n_kv_heads` 头，
所以 4096 上下文下 MQA 的 cache 是 MHA 的 1/8。测试 `test_gqa_shrinks_params_and_cache`
把这条性质钉死了（`sizes[1]*8 == sizes[8]`），`test_gqa_includes_mha_as_special_case`
则保证 `n_kv_heads == n_heads` 时与 MHA **逐位等价**。

实测（`python -m transformer.bench --device cuda`，d=512/L=8/16 头，**prompt 512 + 生成 1024**）：

| 配置 | 参数 | 生成速度 | 峰值显存 |
|---|---|---|---|
| MHA（kv=16） | 29.89M | 55.6 tok/s | 220.0 MiB |
| **GQA（kv=4）** | 26.75M | **61.8 tok/s** | 143.2 MiB |
| MQA（kv=1） | 25.96M | 55.8 tok/s | **124.4 MiB** |
| GQA 关掉 KV cache | 26.75M | 39.6 tok/s | 188.2 MiB |

三条结论：
1. **GQA 在这个尺寸下又快又省**（+11% 速度、−35% 显存），MQA 更省显存但不再更快；
2. **KV cache 的收益要看上下文长度**：prompt 512 + 生成 1024 时开 cache 是 61.8 vs 39.6 tok/s（**1.56×**）；
   而 prompt 1024 + 只生成 128 时，关掉 cache 反而更快（66.2 vs 55.0）——因为那种情况下
   「每步只算 1 个 token」的 kernel 启动开销占主导，而大 batch 的整段重算能把 GPU 喂满。
   **别把「cache 一定更快」当教条，要按上下文长度实测**；
3. 大词表下**权重绑定省的是真金白银**：`V=100279, d=512` 时输出矩阵 51.3M 参数，
   绑不绑是 125.24M vs 73.90M（−41%）。测试 `test_gqa_shrinks_params_and_cache` 之外，
   `bench` 的表里也能直接看到。

### 缩放阶梯的实测结果（也是本节最重要的结论）

```bash
PYTHONPATH=. python -m transformer.ladder --first-prep      # 5 档模型，同一份数据、同样 3000 步
```

| 配置 | 参数量 | d_model | 层 | kv 头 | best val | @step | bpb(最好) | 训练吞吐 |
|---|---|---|---|---|---|---|---|---|
| xl_tiny | 1.38M | 128 | 4 | 4 | 4.447 | 750 | **1.519** | 569k tok/s |
| xl_small | 5.28M | 256 | 6 | 2 | 4.429 | 500 | **1.513** | 211k tok/s |
| xl_mid | 11.02M | 384 | 6 | 4 | 4.550 | 250 | 1.554 | 131k tok/s |
| xl_big | 24.65M | 512 | 8 | 4 | 4.518 | 250 | 1.543 | 79k tok/s |
| xl_big2 | 46.05M | 640 | 10 | 4 | 4.438 | 250 | **1.516** | 40k tok/s |

**这是本次 GPU 实验最有价值、也最反直觉的结果：参数量从 1.4M 涨到 46M（33 倍），
最好 val 却在 4.43~4.55 之间来回晃，bpb 全都在 1.51~1.55 徘徊。**
为什么？看 train 一侧就懂了：`xl_big` 在 step 250 就把 train 压到 1.0 以下（best val 也在那一步），
**模型越大、越早开始背数据**；而 58 万 token 的语料只有 24.7M 模型的 **0.024 token/参数**
（Chinchilla 的经验值是 20 token/参数，差了三个数量级）。

结论写死了：**在数据受限的区域，加参数的边际收益是负的**（因为 val 最小值出现得更早、更浅）。
要看到正常的缩放曲线，必须先把数据加够——这就是 §10 和下面那轮实验的意义。

### 把语料加大 4.5 倍之后，缩放曲线终于出来了

```bash
# 用 transformer.fetch_corpus 抓的《全唐诗》(1.83M 字符) + 本地 RL 语料(2.56M 字符)
PYTHONPATH=. python -m transformer.ladder --datadir data_rl4 \
    --data rl_corpus.json corpus_big/quan_tang_shi.txt --tag-prefix big_ --out ladder_big.json
```

同样是 3000 步 × 16k token，但训练集从 **56 万** 涨到 **251 万** token：

| 配置 | 参数量 | best val | @step | bpb(最好) | 训练吞吐 | 结论 |
|---|---|---|---|---|---|---|
| big_xl_tiny | 1.38M | 4.201 | 2500 | 2.018 | 569k tok/s | 到 3000 步**还在降**（没吃满数据）|
| big_xl_small | 5.28M | **4.114** | 1250 | **1.976** | 211k tok/s | 这一轮的最佳 |
| big_xl_mid | 11.02M | 4.140 | 1000 | 1.988 | 131k tok/s | 开始回退 → 数据又不够了 |
| big_xl_big | 24.65M | 4.142 | 1000 | 1.989 | 79k tok/s | 持平 |
| big_xl_big2 | 46.05M | 4.101 | 750 | 1.970 | 40k tok/s | 略好一点，但 train 已背到 0.07 |

> bpb 的分母不同（data_rl4 的 val 里混了诗，tokens/byte=0.333 而不是 0.236），
> 所以**只能在同一列里横向比，不能和上一张表纵向比**。这也是 bit-per-byte 的正确用法：
> 它在同一数据集上可比，跨数据集比就等于换了考卷。

两个观察值得记住：
1. **同样的模型，数据多 4.5 倍之后 best val 从 4.43 降到 4.20**（tiny），
   而且最优点从 step 750 推到 2500——**data-limited 的模型在数据变多后，会自己找到更晚的最优点**；
2. **11M 以上就基本不动了**（1.988 / 1.989 / 1.970），而 46M 的 train loss 已经掉到 0.07——
   它把 250 万 token 的语料背下来了（val 6.13 vs best 4.10）。
   **每一档模型都有一个「数据吃不下就过拟合」的临界点**，这份语料上的临界点在 **5M 参数附近**；
   这正好复现 Chinchilla 的直觉：**参数量和数据量要一起涨**，单涨一边都没有好结果。
   要按 Chinchilla 的 20 token/参数喂饱 46M 模型，需要 **≈9 亿 token**——比现有语料大 360 倍。

生成的文本也跟着变了——同一个 5.28M 参数的模型（`big_xl_small`），
只因为语料里加了诗，就能写出下面这种东西（prompt 只有一个「春」字）：

```
秋日至瞰首座舍　范浚
一瓶不許風中春，百里梅梢月日秋。三歲不來真笑我，百年清淺作孤猿。

再次韻　其一　陳棣
小篂不言天地生，青青不動一生明。老將心事如相識，只有詩情亦不同。
```

七言律诗的格式（标题、作者、上下句、韵脚位置）都对了，字也用得是那个年代的字。
当然它不知道自己在说什么——**这是 5M 参数 + 500 万字符能得到的最好水平**，
但足以说明「模型的形状学习」和「数据决定天花板」这两件事。

> 两轮实验合起来才是完整结论：**先修数据，再谈模型**。
> 把数据从 56 万 token 提到 251 万（4.5×），1.38M 模型的 best val 从 4.447 → 4.201、
> 最优点从 step 750 → 2500；而把模型从 1.4M 提到 46M（33×），best val 只在 4.10~4.55 之间晃。
> **这一轮 GPU 时间里最值钱的信息不是「我能跑多大的模型」，而是「我的数据在哪个参数规模上成为瓶颈」。**

> 口径说明：上表 bpb 用**最好那一步**的 val 换算（`val/ln2 × tokens/byte`，tokens/byte=0.2363），
> 与 `history.json` 里 `val_bits_per_byte`（按**末步** val 算）不是同一个数。汇报时要写清用哪个。

---

## 附：本机环境与排错

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: torch` | 用了基础 python；用 `~/miniconda3/envs/grpo/bin/python` |
| `re.error: bad escape \p` | 标准库 `re` 不支持 `\p{L}`；装 `regex`（`tokenizers` 会带） |
| `torch.cuda.is_available() == False` | 本沙箱里 GPU 被禁；`train.py` 会自动落到 CPU，不需要改代码 |
| matplotlib 「missing from font」 | 中文字形缺失；`docs/` 的图会显示豆腐块，生成脚本会打警告（本机用 `/mnt/c/Windows/Fonts/msyh.ttc`） |
| 生成的是乱码 | 多半是分词器与权重不匹配；`ckpt.pt` 里存了 `tokenizer_meta`，用它对应的 `data/tokenizer.json` |
| `uint16` 装不下 token id | cl100k 的 id 到 100278，超过 65535；`data.py` 现在按词表大小自动选 `uint16`/`uint32`，并把 dtype 写进 `meta.json` |
| 排队等上一个实验的 shell 永远不退出 | `pgrep -f "[t]ransformer.ladder"` 会匹配到**自己**（waiting 脚本的完整命令行里含这个字符串）→ 自锁死循环。改用 `pgrep -f "^.*python -u -m transformer\.ladder"` 之类更严格的模式，或者直接顺序执行不排队 |
