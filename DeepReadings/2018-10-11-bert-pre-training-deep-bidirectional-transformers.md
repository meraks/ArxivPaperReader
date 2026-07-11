---
# BERT 论文精读报告

## 论文信息

- **标题**:BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- **作者**:Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova(Google AI Language)
- **发表**:NAACL 2019,arXiv:1810.04805
- **开源代码**:https://github.com/google-research/bert

---

## 第一章 概述

### 1.1 核心问题

在 BERT 之前,主流的预训练语言模型(如 GPT、ELMo)都存在一个根本性约束:它们本质上是**单向的(unidirectional)**,每个 token 只能利用一侧的上下文(left-to-right 或 right-to-left),从而无法学习到真正的**深度双向表示**。

- **ELMo**:将独立训练的 left-to-right LM 与 right-to-left LM 进行**浅层拼接**(shallow concatenation)。两个方向各自独立建模、缺乏交互,因此并非真正的深度双向。
- **GPT**:采用 left-to-right 的 Transformer,每个 token 只能 attend 到其之前的 token,严格受限于因果(causal)注意力。

这种单向性使得模型在 token 级任务(如 SQuAD 阅读理解)中,无法同时利用左右两侧的完整上下文,从而限制了表示能力。

### 1.2 核心贡献

BERT(Bidirectional Encoder Representations from Transformers)的核心创新在于提出两个新的预训练任务,使**深层双向表示**的预训练成为可能:

1. **Masked Language Model(MLM,掩码语言模型)**:随机选择 15% 的输入 token 进行 mask,随后利用**双向上下文**来预测这些被 mask 的 token。MLM 打破了标准语言模型固有的单向性约束,使每个 token 都能融合左右两侧的信息。

2. **Next Sentence Prediction(NSP,下一句预测)**:一个二分类任务,判断句子 B 是否紧跟在句子 A 之后。该任务让模型得以建模**句间关系**,对 QA、NLI 等句对任务尤为关键。

凭借上述两个任务,BERT 能够预训练出深层双向表示。在迁移到下游任务时,只需在其上**添加一个额外的输出层(one additional output layer)**进行 fine-tune 即可,无需为每个任务设计定制化的架构。

### 1.3 模型架构与注意力机制

BERT 的主干是标准 Transformer 的 **Encoder**。其核心运算为标准的 scaled dot-product attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!
     \left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 $Q$、$K$、$V$ 分别为查询(query)、键(key)、值(value)矩阵,$d_k$ 为键的维度,$\sqrt{d_k}$ 用于缩放点积以稳定 softmax 的梯度。

BERT 将多个 attention head 并行堆叠构成 multi-head attention。由于采用 Encoder(而非 Decoder)结构、并配合 MLM 预训练目标,每个 token 都能同时 attend 到序列中左右两侧的全部 token,这正是 BERT 实现"真正深度双向"的机制根源。

### 1.4 模型规模

BERT 提供两种规模的模型:

| 模型 | 层数 (L) | 隐藏维度 (H) | 注意力头数 (A) | 参数量 |
|------|:--------:|:------------:|:--------------:|:------:|
| BERT_BASE  | 12 | 768  | 12 | 110M |
| BERT_LARGE | 24 | 1024 | 16 | 340M |

其中 BERT_BASE 的规模被刻意设计为与 GPT 相同,以便进行**公平对比**(fair comparison)。

### 1.5 关键结果

BERT 在 11 项 NLP 任务上刷新了当时的 SOTA(State-of-the-Art):

| 任务 | 指标 | 结果 | 较前 SOTA 提升 |
|------|------|:----:|:--------------:|
| GLUE     | (平均)  | 80.5  | +7.7 pp(绝对提升) |
| MultiNLI | Accuracy | 86.7% | +4.6%(绝对提升) |
| SQuAD v1.1 | Test F1 | 93.2 | +1.5 |
| SQuAD v2.0 | Test F1 | 83.1 | +5.1 |
| SWAG    | Accuracy | 86.3% | — |

其中 GLUE 上 7.7 pp 的绝对提升在当时是一项非常显著的突破。

### 1.6 论文图表概览

- **Figure 1**:Pre-training / Fine-tuning 框架(预训练与微调统一流程)
- **Figure 2**:输入表示(Input representation)= Token Embedding + Segment Embedding + Position Embedding
- **Table 1**:GLUE 结果
- **Table 2**:SQuAD 1.1 结果
- **Table 3**:SQuAD 2.0 结果
- **Table 4**:SWAG 结果
- **Table 5**:消融实验——预训练任务的影响(Pre-training tasks effect)
- **Table 6**:消融实验——模型规模的影响(Model size effect)
- **Table 7**:基于特征(feature-based)方法在 NER 任务上的结果

---

## 第二章 研究背景

BERT 处于 **预训练(pre-training)+ 迁移学习(transfer learning)** 这一研究脉络中。论文将既有工作按技术路线划分为三类,并指出它们的共同局限在于**单向性**。

### 2.1 基于特征的方法(Feature-based Approaches)

这类方法预训练出表示后,**冻结**模型参数,将预训练得到的特征作为下游模型的输入,而非微调整个网络。

- **ELMo**:从 left-to-right 与 right-to-left 两个 LM 中提取上下文特征,采用**浅层拼接**(shallow concatenation)融合。这正解释了它为何"并非真正的深度双向"——两个方向各自独立、缺乏深层交互。
- **预训练词向量**:Word2Vec、GloVe —— 提供静态的、上下文无关的词表示。
- **句子级表示**:Skip-thought、FastSent —— 将预训练思想从词扩展到句子。

### 2.2 微调方法(Fine-tuning Approaches)

这类方法在预训练后,**直接微调整个模型**参数以适配下游任务。

- **OpenAI GPT**:使用 left-to-right 的 Transformer 语言模型预训练,再在下游任务上 fine-tune。其采用的 Decoder 架构与 BERT 的 Encoder 架构形成本质区别。
- **ULMFiT**:基于 LSTM 的预训练 + fine-tuning,提出了一套通用的 fine-tuning 流程(如判别式微调、逐层解冻等技巧)。

**共同局限**:这些方法的预训练目标都是**单向的**(unidirectional),与 BERT 的双向预训练目标存在本质差异。

### 2.3 从有监督数据迁移学习(Transfer Learning from Supervised Data)

除了从无监督文本预训练外,也有研究探索从大规模**有监督数据**迁移表示:

- **自然语言推理(NLI)**:Conneau 等人利用 NLI 数据训练通用句向量。
- **机器翻译(MT)**:McCann 等人利用机器翻译任务习得的表示。

这一思路可与计算机视觉(CV)领域作类比:**ImageNet 预训练**已成为 CV 的标准范式——在大规模有监督数据上预训练,再迁移到下游视觉任务。BERT 则在 NLP 中将"无监督预训练 + 微调"的范式推向了新的高度,并在很大程度上复现了 ImageNet 预训练对 CV 领域的推动作用。

---

## 第三章 Model Architecture — 模型架构

### 3.1 Architecture overview — 架构概述

BERT 基于 multi-layer bidirectional Transformer encoder 构建而成,其基本结构来自 Vaswani et al. (2017)。模型由若干关键超参数刻画:层数 $L$、hidden size $H$、self-attention 的 attention heads 数 $A$,以及 feed-forward 层的中间维度 $4H$。

论文提供了两种 model size:

| Model | $L$ | $H$ | $A$ | Parameters |
|-------|-----|-----|-----|------------|
| BERT_BASE | 12 | 768 | 12 | 110M |
| BERT_LARGE | 24 | 1024 | 16 | 340M |

其中 BERT_BASE 与 GPT 拥有相同的 model size,这一设定意在保证二者之间的公平比较。

BERT 与 GPT 在架构上最核心的差异在于 attention 的方向性。BERT 采用 **bidirectional** self-attention,即每个 token 可以同时 attend 到其左右两侧的全部 context;而 GPT 采用 constrained self-attention,仅允许 left-to-right 方向,即每个 token 只能 attend 到其左侧的 tokens。正是这一双向性使 BERT 能够获得真正融合了完整上下文的 context-aware representations。

### 3.2 Input representation — 输入表示

#### Tokenization

输入文本首先经过 WordPiece tokenization 处理,词表大小为 30,000 tokens。

#### Embedding 叠加

对于每个输入 token,其 representation 由三种 embedding 相加得到:

$$\text{Input} = \text{Token Embeddings} + \text{Segment Embeddings} + \text{Position Embeddings}$$

- **Token Embeddings**: 每个 WordPiece token 对应的向量表示
- **Segment Embeddings**: 标识每个 token 所属的 sentence
- **Position Embeddings**: 编码每个 token 在 sequence 中的位置信息

#### Special tokens

BERT 定义了两个 special tokens:

- **[CLS]**: classification 专用 token,始终置于每个 sequence 的首位。其 final hidden state 作为整个 sequence 的 aggregate representation。
- **[SEP]**: 用于分隔 sentences。

#### Sentence pair 表示

对于 sentence pair 输入,组织形式为 Sentence A + [SEP] + Sentence B。其中 sentence A 的每个 token 携带 segment embedding A,sentence B 的每个 token 携带 segment embedding B。

需要特别说明的是,论文中的两个关键术语并非严格的语言学概念:

- **"Sentence"**: 指任意 contiguous span of text,并不一定是 linguistic sentence。
- **"Sequence"**: 指 input sequence,可以是单个 sentence,也可以是打包后的 sentence pair。

### 3.3 Output for downstream tasks — 下游任务的输出表示

BERT 的 final hidden layer 提供两种输出,分别对应 sequence-level 与 token-level 任务:

- **[CLS] vector $C \in \mathbb{R}^H$**: 用于 sequence-level classification。在 fine-tuning 时新增一个权重矩阵 $W \in \mathbb{R}^{K \times H}$(其中 $K$ 为 label 数量),将 $C$ 映射到 $K$ 维输出空间进行分类。
- **Token vectors $T_i \in \mathbb{R}^H$**: 用于 token-level tasks,如 span prediction、sequence tagging 等。

---

## 第四章 Pre-training — 预训练

### 4.1 Pre-training data — 预训练数据

BERT 的预训练语料包含两个来源:

- **BooksCorpus**: 800M words
- **English Wikipedia**: 2,500M words(仅提取 text passages,忽略 lists、tables、headers 等结构化内容)

一个重要的设计原则是:语料保持 **document-level** 结构,不进行 sentence shuffling。这样做的目的是使模型能够学习到跨句、跨段的 long-range dependencies。

### 4.2 Task #1: Masked LM (MLM) — 掩码语言模型

#### 基本思想

为了使 BERT 能够利用 bidirectional context 学习 representation,论文提出了 Masked LM(MLM)任务:在每个 sequence 中随机 mask 15% 的 WordPiece tokens,然后基于被 mask token 的 bidirectional context 来预测这些 tokens。

#### Masking strategy (80/10/10)

被选中的 15% tokens 按如下比例处理:

| 比例 | 处理方式 |
|------|----------|
| 80% | 替换为 [MASK] |
| 10% | 替换为 random token |
| 10% | 保持不变 |

#### 80/10/10 的设计动机

如果对被选中的 token 100% 使用 [MASK],会引入 pre-training 与 fine-tuning 之间的 distribution mismatch:在 pre-training 阶段 input 中存在 [MASK] token,而在 fine-tuning 阶段下游任务的 input 中并不包含 [MASK]。80/10/10 的混合策略正是为了缓解这一 mismatch。

#### 与 denoising autoencoder 的区别

与 denoising autoencoder 不同,BERT 仅预测被 mask 的 words,而非对完整 input 进行 full reconstruction。

### 4.3 Task #2: Next Sentence Prediction (NSP) — 下一句预测

#### 任务定义

NSP 是一个 binary classification 任务:给定 sentence pair (A, B),判断 sentence B 是否是 sentence A 在原文中真正的下一句。

#### 数据构造

训练数据中两类样本各占 50%:

- **50% IsNext**: B 为 A 的实际下一句
- **50% NotNext**: B 为从语料中随机抽取的 sentence

#### 预测方式

[CLS] vector $C$ 被用于 NSP 的 prediction。

#### 效果

最终 NSP accuracy 达到 97–98%。NSP 任务对 QA(Question Answering)与 NLI(Natural Language Inference)等需要理解 sentence pair 关系的下游任务有益。

### 4.4 Training details — 训练细节

#### 主要超参数

| 超参数 | 取值 |
|--------|------|
| Batch size | 256 sequences |
| Tokens / batch | 128,000(对应 1M steps) |
| Training steps | 1,000,000 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| $\beta_1$ | 0.9 |
| $\beta_2$ | 0.999 |
| L2 weight decay | 0.01 |
| Learning rate warmup | 前 10,000 steps |
| Warmup 后调度 | Linear decay |
| Dropout | 0.1(all layers) |
| Activation | GELU (Gaussian Error Linear Unit) |

#### 训练损失

总训练损失为 MLM loss 与 NSP loss 之和:

$$\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$$

#### 预训练成本

整个预训练过程耗时约 4 天,使用 4 Cloud TPUs(共 16 TPU chips)。

---

## 第五章 实验

本章在多个自然语言理解任务上对 BERT 进行全面评估。

### 5.1 GLUE (General Language Understanding Evaluation)

#### Fine-tuning 设置

- **Batch size**: 32
- **Epochs**: 3
- **Learning rate**: 从 `{5e-5, 4e-5, 3e-5, 2e-5}` 中基于 Dev set 选择

对于单句分类任务,取 [CLS] token 的最终隐藏向量 $C \in \mathbb{R}^H$,通过分类层 $W \in \mathbb{R}^{K \times H}$($K$ 为类别数)计算类别概率。BERT_LARGE 在小规模数据集上偶尔出现不稳定现象,此时采用 random restarts 策略。

#### GLUE 结果

| System | MNLI-(m/mm) | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Average |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pre-OpenAI SOTA | 80.6/80.1 | 66.1 | 82.3 | 93.2 | 35.0 | 81.0 | 86.0 | 61.7 | 74.0 |
| BiLSTM+ELMo+Attn | 76.4/76.1 | 64.8 | 79.8 | 90.4 | 36.0 | 73.3 | 84.9 | 56.8 | 71.0 |
| OpenAI GPT | 82.1/81.4 | 70.3 | 87.4 | 91.3 | 45.4 | 80.0 | 82.3 | 56.0 | 75.1 |
| BERT_BASE | 84.6/83.4 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 | 79.6 |
| BERT_LARGE | 86.7/85.9 | 72.1 | 92.7 | 94.9 | 60.5 | 86.5 | 89.3 | 70.1 | 82.1 |

**关键分析:**
- BERT_BASE(与 GPT 规模相同)在全部任务上均优于 OpenAI GPT,平均高出 +4.5%。
- BERT_LARGE 相比 GPT 平均高出 +7.0%。
- GLUE leaderboard: BERT_LARGE 80.5 vs GPT 72.8。

### 5.2 SQuAD v1.1

抽取式问答,通过学习 **Start vector** $S \in \mathbb{R}^H$ 与 **End vector** $E \in \mathbb{R}^H$ 定位答案 span:

$$P_i = \frac{e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}}$$

Fine-tuning: 3 epochs, lr=5e-5, batch size=32。

| System | Dev EM | Dev F1 | Test EM | Test F1 |
|--------|:------:|:------:|:-------:|:-------:|
| BERT_BASE (Single) | 80.8 | 88.5 | - | - |
| BERT_LARGE (Single) | 84.1 | 90.9 | - | - |
| BERT_LARGE (Ensemble) | 85.8 | 91.8 | - | - |
| BERT_LARGE (Sgl.+TriviaQA) | 84.2 | 91.1 | 85.1 | 91.8 |
| BERT_LARGE (Ens.+TriviaQA) | 86.2 | 92.2 | 87.4 | 93.2 |

Human: Test EM 82.3, F1 91.2。最佳单模型已超越所有 ensemble 系统。

### 5.3 SQuAD v2.0

| System | Dev EM | Dev F1 | Test EM | Test F1 |
|--------|:------:|:------:|:-------:|:-------:|
| BERT_LARGE (Single) | 78.7 | 81.9 | 80.0 | 83.1 |

Human: Test EM 86.9, F1 89.5。

### 5.4 SWAG

| System | Accuracy |
|--------|:--------:|
| OpenAI GPT | 75.0% |
| Human | 81.7% |
| BERT_LARGE | 86.3% |

Fine-tuning: 3 epochs, lr=2e-5, batch size=16。

### 5.5 NER (CoNLL 2003)

BERT_LARGE: F1 96.4(与 SOTA 持平)。

---

## 第六章 消融研究

### 6.1 Pre-training 任务的影响

| Task | Dev | Test |
|------|:---:|:---:|
| BERT_BASE | 84.4 | 85.8 |
| No NSP | 83.9 | 84.9 |
| LTR + BiLSTM | 84.0 | 84.9 |
| LTR + BiLSTM + NSP | 84.1 | 84.9 |

**MLM 是实现真正双向表示的关键**,LTR+BiLSTM 无法弥补差距。

### 6.2 Model Size 的影响

| Layers (L) | Hidden size (H) | Parameters |
|:----------:|:---------------:|:----------:|
| 12 | 256 | 15.8M |
| 12 | 512 | 42.3M |
| 24 | 512 | 105.6M |
| 24 | 768 | 194.6M |
| 24 | 1024 | 340M |

模型规模增大显著提升准确率,尤其在小数据集任务(CoLA、RTE、MRPC)上。

### 6.3 Feature-based Approach

| 方法 | F1 |
|------|:--:|
| Fine-tuned | 96.4 |
| Feature-based (ELMo-style weighted) | 95.6 |
| Feature-based (last 4 layers concat) | 96.1 |

### 6.4 MLM Masking 策略消融

| Mask | Same | Rnd | MNLI | NER Fine-tune | NER Feature |
|:----:|:----:|:---:|:----:|:-------------:|:-----------:|
| 80% | 10% | 10% | 84.2 | 95.4 | 94.9 |
| 100% | 0% | 0% | 84.3 | 94.9 | 94.0 |
| 80% | 0% | 20% | 84.1 | 95.2 | 94.6 |
| 80% | 20% | 0% | 84.4 | 95.2 | 94.7 |
| 0% | 20% | 80% | 83.7 | 94.8 | 94.6 |
| 0% | 0% | 100% | 83.6 | 94.9 | 94.6 |

---

## 第七章 代码实现

- **开源仓库**: https://github.com/google-research/bert
- **实现框架**: TensorFlow
- **预训练模型**: BERT_BASE、BERT_LARGE、multilingual、Chinese 版本
- **Fine-tuning 效率**: 任意任务 ≤1 小时(single Cloud TPU)或数小时(GPU)
- **SQuAD 模型**: ~30 分钟达到 Dev F1 91.0%

---

## 第八章 讨论与影响

### 8.1 Significance

BERT 的提出对自然语言处理领域产生了深远影响:
- 首个基于 fine-tuning 的模型在大规模任务套件上同时取得 SOTA
- 双向预训练是一次范式转换(paradigm shift),催生了 RoBERTa、ALBERT、DistilBERT、ELECTRA、SpanBERT 等一系列工作
- MLM 目标函数成为语言模型预训练的标准做法

### 8.2 Limitations

- Pre-training / Fine-tuning 不一致(80/10/10 部分缓解)
- 计算成本高昂(4 Cloud TPUs, ~4 天)
- NSP 有效性存疑(后续研究如 RoBERTa 发现移除 NSP 后性能可持平)

### 8.3 Training Details 与 GPT 对比

| 配置项 | BERT | GPT |
|--------|------|-----|
| Training steps | 1M | 1M |
| Batch size | 256 (128K words/batch) | 32K words |
| Learning rate | 1e-4 | 5e-5(统一) |
| Task-specific lr | 从 Dev set 选择 | 无 |

BERT 与 GPT 同样训练 1M steps,但 BERT 使用了更大的 batch(4 倍)与更高的 base learning rate,以及任务自适应的 fine-tuning 策略。