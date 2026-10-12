#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小 transformer：可以真的训练出东西的一小套代码（CPU 也能跑）。

文件分工：
    bpe.py        手写字节级 BPE 分词器（可逆、零依赖）
    corpus.py     用工作区本地文本拼一份语料，先切 train/val
    data.py       分词 + 打包成 token 流 + 随机采样 batch
    model.py      Transformer 本体：RMSNorm / RoPE / 因果注意力 / SwiGLU
    train.py      训练循环（warmup + 余弦、梯度裁剪、最佳权重）
    generate.py   加载权重生成文本（KV cache 增量解码）

典型用法：
    cd <工作区>
    python -m transformer.train --quick       # 300 步，几十秒跑通
    python -m transformer.generate --tag quick -p "def "
    python -m transformer.train               # 2000 步的正式小模型
"""

__all__ = ["bpe", "corpus", "data", "model", "train", "generate"]
