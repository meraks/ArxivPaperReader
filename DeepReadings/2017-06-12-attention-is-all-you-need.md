# 《Attention Is All You Need》论文精读报告

- **论文标题**: Attention Is All You Need
- **作者**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain/Google Research)
- **发表**: NeurIPS 2017
- **arXiv**: 1706.03762
- **核心思想**: 提出 Transformer 架构，完全基于注意力机制，摒弃循环和卷积

## Ch1 论文概述与核心贡献

### 1.1 研究范式转变

Transformer 是首个完全基于注意力机制（attention mechanism）的序列转换（sequence transduction）模型，摒弃了此前占据主导地位的循环神经网络（RNN/LSTM/GRU）与卷积神经网络（CNN）。该架构彻底改变了序列建模与 NLP 领域的研究范式。

### 1.2 三大核心贡献

**贡献一：完全基于注意力的架构**

Encoder-Decoder 全程由 self-attention 与 position-wise feed-forward 层构成，不包含任何循环或卷积操作。Encoder 与 Decoder 各堆叠 $N=6$ 层。

**贡献二：Scaled Dot-Product Attention 与 Multi-Head Attention**

Scaled Dot-Product Attention 通过缩放因子 $\sqrt{d_k}$ 抑制大维度下 softmax 的梯度饱和：

$$\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Multi-Head Attention 将 $d_{model}$ 投影到 $h$ 个子空间并行计算后拼接。base model 配置为 $h=8$, $d_k=d_v=d_{model}/h=64$。

**贡献三：Positional Encoding（位置编码）**

由于 attention 本身对输入顺序不敏感，论文使用不同频率的正弦/余弦函数注入位置信息：

$$PE_{(pos,2i)}=\sin\left(pos/10000^{2i/d_{model}}\right)$$

$$PE_{(pos,2i+1)}=\cos\left(pos/10000^{2i/d_{model}}\right)$$

### 1.3 关键结果速览

| 任务 | 模型 | BLEU | 训练成本 |
|------|------|------|---------|
| WMT 2014 EN-DE | Transformer (big) | 28.4 | $2.3\times10^{19}$ FLOPs |
| WMT 2014 EN-FR | Transformer (big) | 41.0 | 3.5 days / 8×P100 |

- EN-DE 较此前最优结果（含集成模型）提升 **2+ BLEU**（达到 28.4 BLEU）
- EN-FR 建立新的单模型 SOTA（41.0 BLEU），训练成本低于此前最优模型的 1/4
- base model 在 8×P100 GPU 上仅训练 **12 小时**（100,000 steps）即超越此前所有模型与集成模型

### 1.4 论文章节结构

| 章节 | 主题 |
|------|------|
| 1 引言（Introduction） | 动机与 Transformer 总览 |
| 2 背景（Background） | 减少顺序计算的已有工作与 self-attention 应用 |
| 3 模型架构（Model Architecture） | Encoder-Decoder、Attention、FFN、Positional Encoding |
| 4 为什么 Self-Attention（Why Self-Attention） | 与 RNN/CNN 在复杂度、并行度、路径长度上的对比 |
| 5 训练（Training） | 数据、硬件、优化器、学习率调度、正则化 |
| 6 结果（Results） | 机器翻译、模型变体、英语成分句法分析 |
| 7 结论（Conclusion） | 总结与多模态展望 |

## Ch2 研究背景与动机

### 2.1 主流方法：RNN/LSTM/GRU

Transformer 提出之前，序列建模与机器翻译的主流是基于循环神经网络（RNN、LSTM、GRU）的 encoder-decoder 架构。代表性工作：

- **Sutskever et al. (2014)**：sequence-to-sequence（seq2seq）学习
- **Bahdanau et al. (2014)**：将注意力机制引入 encoder-decoder，实现软对齐
- **Cho et al. (2014)**：提出 Gated Recurrent Unit（GRU）

### 2.2 RNN 的核心局限：顺序计算阻碍并行化

RNN 沿时间步逐步计算，隐藏状态 $h_t$ 依赖 $h_{t-1}$，导致训练样本内部无法并行。论文原文指出：

> "The inherently sequential nature precludes parallelization within training examples"

这带来两个后果：序列计算形成内存瓶颈（memory constraint），训练时间过长。

### 2.3 已有的改进尝试

为缓解顺序计算瓶颈，已有两类工作：

- **Factorization tricks**（Kuchaiev & Ginsburg, 2017）：通过分解技巧降低循环计算开销
- **Conditional computation**（Shazeer et al., 2017, Mixture-of-Experts）：按需激活部分参数以提升容量

二者仍未从根本上摆脱顺序依赖。

### 2.4 注意力机制与 RNN 的绑定

注意力机制自 Bahdanau et al. (2014) 起被广泛用于 seq2seq，但几乎总是与 RNN 配合使用——注意力负责软对齐，RNN 负责序列编码。

### 2.5 减少顺序计算的尝试：CNN 路线

Extended Neural GPU、ByteNet、ConvS2S 采用卷积神经网络构建 encoder-decoder，使训练可并行化。但任意两个输入/输出位置的连接路径长度仍随距离增长：

- **ByteNet**：路径长度随距离线性增长
- **ConvS2S**：路径长度随距离对数增长

长路径使模型难以学习长距离依赖（long-range dependency）。论文指出 Transformer 将该路径长度降为常数：

> "In the Transformer this is reduced to a constant number of operations"

对应 Table 1 中 self-attention 的 $O(1)$ 最大路径长度（vs. Recurrent $O(n)$、Convolutional $O(\log_k n)$）。

### 2.6 自注意力的已有应用

Self-attention（又称 intra-attention）已在多个任务中成功应用：阅读理解、摘要生成、文本蕴含（textual entailment）。

### 2.7 本文定位

Transformer 是首个端到端、完全基于自注意力的序列转换模型，将注意力从"对齐辅助工具"提升为唯一的序列建模原语。

---


## 第 3 章 核心技术：Scaled Dot-Product & Multi-Head Attention

### 3.1 注意力函数的一般定义

论文将**注意力函数（attention function）**抽象为这样一个映射：给定一个查询（query）和一组键-值对（key-value pairs），输出一个加权和。形式化地，输出是值的加权求和，其中每个值 $v_i$ 的权重由 query $q$ 与对应 key $k_i$ 之间的**兼容性函数（compatibility function）**计算得到。

$$\text{Output} = \sum_{i} \text{compatibility}(q, k_i) \cdot v_i$$

这一抽象定义统一了论文中所有注意力变体。下文的两类注意力——Scaled Dot-Product Attention 与 Multi-Head Attention——都是这一框架下的具体实例，区别仅在于 query/key/value 的来源与维度的划分。

### 3.2 Scaled Dot-Product Attention

#### 3.2.1 公式

Scaled Dot-Product Attention 是论文最基础的注意力单元（论文 Figure 2 左半部分）。给定查询矩阵 $Q$、键矩阵 $K$、值矩阵 $V$，计算公式为（论文公式 1）：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V \tag{1}$$

其中 $d_k$ 是 query/key 的维度。其计算流程为三步：
1. $Q$ 与 $K^{\top}$ 做矩阵乘法，得到 $n \times n$ 的**点积相似度矩阵**（logits）；
2. 对 logits 除以 $\sqrt{d_k}$ 进行**缩放（scaling）**；
3. 逐行 softmax 得到注意力权重，再与 $V$ 相乘得到加权和。

#### 3.2.2 为什么需要缩放（$\sqrt{d_k}$）

缩放因子 $\sqrt{d_k}$ 的引入是为了应对维度增大时的数值问题。论文给出了如下论证：假设 query 和 key 的分量 $q_i, k_i$ 是均值为 0、方差为 1 的独立随机变量，则点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的均值为 0、**方差为 $d_k$**。当 $d_k$ 较大时（base 模型中 $d_k = 64$），点积的绝对值会变得很大，把 softmax 推入**梯度极小的饱和区**（saturation region）。

- 不加缩放时，点积分布随 $d_k$ 增大而方差膨胀，softmax 输出趋近 one-hot，反向传播的梯度几乎消失；
- 除以 $\sqrt{d_k}$ 将点积的方差重新归一为 1 量级，使 softmax 保持在梯度健康的工作区间。

这一缩放正是论文标题中 "Scaled" 的含义，也是 dot-product attention 在大维度下能与加性注意力匹敌的关键。

#### 3.2.3 与加性注意力的对比

论文将本方法与**加性注意力（additive attention）**做了对比。两种注意力函数在**理论复杂度上相似**，但点积注意力在实践中更快、更省内存，原因是它能够利用高度优化的矩阵乘法实现（highly optimized matrix multiplication code）来完成。

不过作者也指出一个反直觉的现象：对于较小的 $d_k$ 两者表现相当；**当 $d_k$ 较大时，不缩放的点积注意力反而劣于加性注意力**。这正是因为大维度下未缩放的点积把 softmax 推入饱和区、梯度过小所致。$\sqrt{d_k}$ 缩放正是为弥合这一差距而设计的。

> 论文同时提到，本文使用的缩放点积注意力与此前工作所用的乘性（multiplicative）注意力相近，区别只在于多了一个 $\frac{1}{\sqrt{d_k}}$ 因子。

### 3.3 Multi-Head Attention

#### 3.3.1 动机与公式

与其在 $d_{\text{model}}$ 全维度上执行单次注意力，论文更倾向于**把 query、key、value 分别线性投影（project）$h$ 次到较低维度**，对每个投影版本并行执行注意力，再把结果拼接（concatenate）后做一次线性映射。每个投影版本称为一个"头"（head）。

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^{O}$$

$$\text{head}_i = \text{Attention}(Q W_i^{Q},\; K W_i^{K},\; V W_i^{V})$$

其中投影矩阵 $W_i^{Q} \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W_i^{K} \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W_i^{V} \in \mathbb{R}^{d_{\text{model}} \times d_v}$、$W^{O} \in \mathbb{R}^{h d_v \times d_{\text{model}}}$ 均为可学习参数。

#### 3.3.2 维度配置（base 模型）

| 超参数 | 取值 | 说明 |
|--------|------|------|
| 头数 $h$ | **8** | base 模型；big 模型为 16 |
| 每头 query/key 维度 $d_k$ | **64** | $= d_{\text{model}} / h$ |
| 每头 value 维度 $d_v$ | **64** | $= d_{\text{model}} / h$ |
| 模型维度 $d_{\text{model}}$ | **512**（base）/ 1024（big） | 所有子层与嵌入输出维度 |

由于每个头的维度被压缩到 $d_k = d_v = d_{\text{model}}/h = 64$，**8 个头并行后的总计算成本与单头全维度（$d_k = d_{\text{model}} = 512$）的注意力几乎相同**，却获得了多头表示能力。

#### 3.3.3 多头的设计意义

多头机制让模型能够在**不同的表示子空间（representation subspaces）、不同的位置上**共同关注信息。单一注意力头平均下来会稀释这种能力；而多个头可以分别捕捉不同类型的关系。论文后续的可解释性分析（第 4.6 节 "Why Self-Attention"）也印证了"**不同的 head 执行不同的任务**"这一现象。

### 3.4 注意力的三种使用场景

Transformer 中将 Multi-Head Attention 以三种不同的连接方式使用，区别在于 $Q/K/V$ 的来源：

#### (1) Encoder-Decoder Attention（编码器-解码器注意力）
位于解码器每一层中。**Queries 来自解码器的前一子层**，而 **keys 和 values 来自编码器的输出**。这等价于经典 seq2seq 中"解码器查询编码器"的对齐机制：解码器每一步都能关注到编码器输出的全部位置，从而实现源端到目标端的信息流动。

#### (2) Encoder Self-Attention（编码器自注意力）
位于编码器每一层中。**$Q$、$K$、$V$ 全部来自编码器上一层**的输出（同一序列）。每个位置都可以关注编码器前一层的所有位置，从而在源序列内部建立任意两词之间的依赖关系。

#### (3) Decoder Masked Self-Attention（解码器带掩码自注意力）
位于解码器每一层中。同样地 $Q/K/V$ 都来自解码器自身，但必须加一个**掩码（mask）**：在计算 softmax 之前，把所有"未来"位置对应的注意力 logits 设为 $-\infty$（softmax 后权重为 0），从而**防止解码器中位置向左（向后看）的信息流动**，严格保证自回归性质（auto-regressive property）——预测第 $t$ 个词时只能看到第 $1 \dots t-1$ 个词。

> 小结：三种用法的核心区别就是 $Q/K/V$ 的来源与是否施加因果掩码。掩码的存在使解码器既能并行训练，又能在推理时保持严格的从左到右生成。

---

## 第 4 章 Encoder-Decoder 架构与 Positional Encoding

Transformer 整体沿用 encoder-decoder 范式：编码器把输入符号序列 $(x_1, \dots, x_n)$ 映射为连续表示序列 $\mathbf{z} = (z_1, \dots, z_n)$；解码器在给定 $\mathbf{z}$ 的条件下，逐个生成输出符号序列 $(y_1, \dots, y_m)$。每一步生成时，模型都是自回归（auto-regressive）的——在生成下一个符号时把此前已生成的符号当作额外输入。

### 4.1 Encoder（编码器）

- **层数**：$N = 6$ 个完全相同的层堆叠。
- **每层结构**：两个子层（sub-layer）：
  1. **Multi-Head Self-Attention**；
  2. **Position-wise Feed-Forward Network**（见 4.3）。
- **残差连接 + 层归一化**：每个子层都包裹在
$$\text{LayerNorm}\bigl(x + \text{Sublayer}(x)\bigr)$$
  的结构中。为方便残差相加，所有子层以及嵌入层的输出维度统一为 **$d_{\text{model}} = 512$**。

### 4.2 Decoder（解码器）

- **层数**：同样 $N = 6$ 个相同层堆叠。
- **每层结构**：相比编码器**多插入一个子层**，共三个：
  1. **Masked Multi-Head Self-Attention**（带掩码，保证自回归）；
  2. **Encoder-Decoder Attention**（query 来自本解码层，key/value 来自编码器输出 $\mathbf{z}$）；
  3. **Position-wise Feed-Forward Network**。
- **残差 + 层归一化**：同样地，每个子层都做 $\text{LayerNorm}(x + \text{Sublayer}(x))$，输出维度 $d_{\text{model}} = 512$。

解码器的掩码确保了位置 $i$ 的预测只依赖已知的、位置小于 $i$ 的输出，从而维持训练时并行计算与推理时自回归的一致性。

### 4.3 Position-wise Feed-Forward Network（位置级前馈网络）

编码器/解码器每层的第二个（解码器为第三个）子层是一个**逐位置应用的两层全连接网络**：

$$\text{FFN}(x) = \max(0,\; x W_1 + b_1)\, W_2 + b_2 \tag{2}$$

- **结构**：两层线性变换，中间用 **ReLU** 激活。隐藏维度 $d_{\text{ff}}$ 远大于 $d_{\text{model}}$。
- **维度**：$d_{\text{ff}} = \mathbf{2048}$（base）/ $\mathbf{4096}$（big），即输入/输出 $d_{\text{model}} = 512$（base），内层扩到 2048。
- **"Position-wise" 的含义**：同一组 $W_1, W_2$ **独立地应用于序列中的每一个位置**（不同位置参数共享，位置之间不交互）。论文指出，这**等价于 kernel size = 1 的两个卷积**（卷积核宽度为 1 的两次 1D 卷积），所以又称为 position-wise。

### 4.4 Embeddings and Softmax（嵌入与输出投影）

- **输入/输出嵌入**：使用学习到的（learned）线性嵌入将输入符号与输出符号映射到 $d_{\text{model}} = 512$ 维向量。
- **权重共享（weight tying）**：在两个翻译模型中，论文让 **输入嵌入矩阵、输出嵌入矩阵、pre-softmax 线性变换的权重三者共享**（share the same weight matrix）。在嵌入层与 pre-softmax 线性层之间，把输出投影到词表大小上的权重复用嵌入权重。
- **缩放因子**：共享后的嵌入权重在送入模型前需**乘以 $\sqrt{d_{\text{model}}}$**。这一缩放使嵌入向量的尺度与模型内部其他子层（经过残差/层归一化后量级稳定在 1 附近）相匹配。

### 4.5 Positional Encoding（位置编码）

Transformer 完全没有循环或卷积，而自注意力本身是**排列不变的**（对位置无感知）。为了让模型利用序列的顺序信息，必须在输入嵌入中注入**位置信息**。论文的做法是在编码器和解码器底部、嵌入之后，叠加一个 **Positional Encoding（PE）**，与嵌入向量维度相同（$d_{\text{model}}$），两者相加后送入第一层。

论文采用**不同频率的 sine/cosine 函数**：

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{\,2i/d_{\text{model}}}}\right)$$

$$PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{\,2i/d_{\text{model}}}}\right)$$

其中 $pos$ 是位置（$0, 1, 2, \dots$），$i$ 是维度索引。也就是说，PE 的偶数维用 $\sin$、奇数维用 $\cos$，且不同维度对应不同频率。

#### 4.5.1 频率的几何级数
PE 的正弦函数**波长**构成一个**几何级数（geometric progression）**，波长从 **$2\pi$ 到 $10000 \cdot 2\pi$**。低维对应高频（短周期、捕捉局部位置），高维对应低频（长周期、捕捉全局相对位置），从而用一组合适数量的频率覆盖了从极近到极远的相对距离。

#### 4.5.2 为什么选正余弦而非可学习位置嵌入
论文选择三角函数版本有两个理论理由：
1. **相对位置的可线性表示性**：由于 $\sin/\cos$ 的和角公式，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的**线性函数**（依赖固定偏移 $k$）。这使得模型能很容易地学会"**关注相对位置**"，因为任意固定相对偏移 $k$ 的位置编码之间都存在一个不随 $pos$ 变化的线性变换。
2. **外推能力**：正余弦函数对任意长度 $pos$ 都有定义，使模型能外推到比训练中见过的更长的序列。

#### 4.5.3 实证：学习版 vs 正弦版几乎相同
论文在 Table 3 的实验（变体 E）中比较了**可学习的位置嵌入（learned positional embedding）**与本文正弦 PE：**两者结果几乎相同（nearly identical）**。作者据此推断，正弦版本之所以被采用，并非因为它在指标上明显更优，而是因为它能天然外推到更长序列、且不增加可学习参数。

### 4.6 Why Self-Attention：与循环/卷积的三维对比（Table 1）

论文在 Section 4 用 **Table 1** 从三个维度系统比较了 self-attention、recurrent（如 RNN/LSTM）和 convolutional 三类层的设计权衡：

| Layer Type | 每层复杂度 (Complexity per Layer) | 顺序操作数 (Sequential Operations) | 最大路径长度 (Max Path Length) |
|------------|----------------------------------|----------------------------------|-------------------------------|
| **Self-Attention** | $\mathbf{O(n^2 \cdot d)}$ | $\mathbf{O(1)}$ | $\mathbf{O(1)}$ |
| **Recurrent** | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| **Convolutional** | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

其中 $n$ 为序列长度，$d$ 为表示维度，$k$ 为卷积核宽度，$r$ 为受限注意力的邻域大小。

#### 4.6.1 三个维度的解读
- **每层计算复杂度**：当序列长度 $n$ **小于**表示维度 $d$ 时（机器翻译的典型情况），自注意力每层 $O(n^2 d)$ 比循环层的 $O(n d^2)$ **更小**，因此自注意力层计算更快。受限自注意力可进一步在长序列上降到 $O(rnd)$。
- **顺序操作数（并行度）**：循环层需要 $O(n)$ 个不可并行的顺序步骤；自注意力与卷积都是 $O(1)$，可高度并行，这正是 Transformer 训练快的根本原因。
- **长距离依赖路径长度**：自注意力的任意两个位置之间路径长度为 $O(1)$（任意两词直接相连），循环层为 $O(n)$（信息须逐步传递），卷积层为 $O(\log_k n)$。路径越短，学习长距离依赖越容易。

#### 4.6.2 核心结论
当 $n < d$ 时（典型机器翻译场景），**self-attention 在三者中同时取得最短依赖路径 $O(1)$ 与最高并行度 $O(1)$，且单层计算量反而最小**——这正是 Transformer 选择纯注意力架构、彻底摒弃循环的量化依据。可学习卷积虽并行度也高，但依赖路径为 $O(\log_k n)$，不如自注意力短。

#### 4.6.3 可解释性作为额外优势
除上述三项量化指标外，论文还指出 self-attention 带**可解释性（interpretability）**红利：不同的 attention head 明显在执行**不同的任务**（例如有的头学到了语法依存、有的头关注长距离共指），可以通过可视化注意力分布来检视模型"在看哪里"。这一性质是循环/卷积网络难以提供的。

---

## 本章小结

| 组件 | 关键设计 / 公式 | 关键数值 |
|------|----------------|----------|
| Scaled Dot-Product Attention | $\text{softmax}(QK^{\top}/\sqrt{d_k})V$ | 缩放因子 $\sqrt{d_k}$；base $d_k=64$ |
| Multi-Head Attention | $\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^{O}$ | $h=8$（base）/ $16$（big）；$d_k=d_v=64$ |
| FFN | $\max(0, xW_1+b_1)W_2+b_2$ | $d_{\text{ff}}=2048$（base）/ $4096$（big） |
| Encoder | 2 子层 + 残差 + LayerNorm | $N=6$，$d_{\text{model}}=512$ |
| Decoder | 3 子层（含 masked self-attn + enc-dec attn）+ 残差 + LayerNorm | $N=6$，$d_{\text{model}}=512$ |
| Embeddings/Softmax | 三处权重共享；嵌入 $\times\sqrt{d_{\text{model}}}$ | $d_{\text{model}}=512$ |
| Positional Encoding | $\sin/\cos$，波长 $2\pi \to 10000{\cdot}2\pi$ | 与学习版结果几乎相同（Table 3 row E） |
| Why Self-Attention | 复杂度/顺序操作/路径长度三维对比 | 自注意力：$O(n^2d)$, $O(1)$, $O(1)$ |

本章给出了 Transformer 的"如何工作"（第 3 章注意力机制 + 第 4 章完整架构与位置编码）的完整技术图景：以 **Scaled Dot-Product Attention** 为原子单元，以 **Multi-Head** 扩展表示能力，再以 **残差+LayerNorm 的 6 层 Encoder-Decoder**、**Position-wise FFN**、**共享的 Embedding/Softmax** 和**正弦 Positional Encoding** 拼装成端到端模型，并通过 **Table 1** 的三维量化论证了以自注意力取代循环/卷积的合理性。下一部分将进入训练设置与实验结果。

---


## 第 5 章 实验结果与分析

Transformer 在两个大规模机器翻译任务（WMT 2014 英德、英法）和一个英语成分句法分析任务上验证了其有效性与跨任务泛化能力。本章依次给出训练设置（5.1）、机器翻译主结果（5.2，Table 2）、模型变体消融（5.3，Table 3）与句法分析结果（5.4，Table 4）。

### 5.1 训练设置（Training Setup）

#### 5.1.1 数据集

| 任务 | 训练集规模 | 分词/词表 |
|------|-----------|-----------|
| WMT 2014 English-to-German | ~4.5M 句对 | byte-pair encoding (BPE)，词表 ~37K |
| WMT 2014 English-to-French | 36M 句子 | word-piece，词表 32K |

EN-DE 采用 BPE 切分（~37K 子词），EN-FR 采用 32K 的 word-piece 词表。两类分词都将稀有词拆为子词，从而控制词表规模、缓解 OOV 问题。

#### 5.1.2 硬件与训练时长

- **硬件**：8 张 NVIDIA P100 GPU。
- **批大小**：约 25,000 source tokens + 25,000 target tokens（以 token 数而非句对数计量，便于长序列稳定）。
- **base model**：训练 100,000 steps，耗时约 **12 小时**，单步约 **0.4s**。
- **big model**：训练 300,000 steps，耗时约 **3.5 天**，单步约 **1.0s**。

由于 Transformer 无循环依赖，训练可高度并行，base 模型在 8×P100 上仅 12 小时即超越此前所有已发表模型与集成模型——这是训练效率优势的直接体现。

#### 5.1.3 优化器与学习率调度

使用 Adam 优化器，参数为 $\beta_1 = 0.9,\; \beta_2 = 0.98,\; \varepsilon = 10^{-9}$。

学习率采用**先线性预热、后按步数平方根倒数衰减**的调度（论文公式）：

$$\text{lrate} = d_{\text{model}}^{-0.5} \cdot \min\!\left(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup\_steps}^{-1.5}\right)$$

其中 $\text{warmup\_steps} = 4000$。该调度有两段行为：

- **预热阶段（step ≤ 4000）**：$\min$ 取右支 $\text{step}\cdot\text{warmup}^{-1.5}$，学习率随步数**线性增长**；
- **衰减阶段（step > 4000）**：$\min$ 取左支 $\text{step}^{-0.5}$，学习率随步数按 $1/\sqrt{\text{step}}$ **缓慢下降**。

公式中还含有 $d_{\text{model}}^{-0.5}$ 因子，使大模型（big, $d_{\text{model}}=1024$）的学习率自动小于小模型（base, $d_{\text{model}}=512$），保证不同规模模型梯度的合理尺度。

#### 5.1.4 正则化

论文使用三项正则化手段抑制过拟合：

1. **残差 Dropout**：对每个子层的输出（在 residual addition 与 LayerNorm 之前）以概率 $P=0.1$ 施加 dropout；base 与 big（EN-FR）均用 $P=0.1$，big（EN-DE）用 $P=0.3$。
2. **Attention Dropout**：在 attention 内部的权重上同样施加 dropout（$P=0.1$）。
3. **Label Smoothing（标签平滑）**：$\varepsilon_{ls} = 0.1$。即把 one-hot 标签的 1 改为 $1-\varepsilon+\varepsilon/V$，其余类各分到 $\varepsilon/V$（$V$ 为词表大小）。这使模型不必对单一类过度自信，论文指出它"损害了 log-likelihood（PPL 略升）但提升了 BLEU 与准确率"。

### 5.2 机器翻译结果（Table 2）

#### 5.2.1 结果表

Transformer 两个变体在两个翻译任务上的 BLEU（newstest2013 / newstest2014）与训练成本如下：

| 模型 | WMT'14 EN-DE BLEU | WMT'14 EN-FR BLEU | 训练成本 |
|------|------------------:|------------------:|----------|
| **Transformer (big)** | **28.4** | **41.0** | EN-DE: $2.3\times10^{19}$ FLOPs；EN-FR: 3.5 天 / 8×P100 |
| **Transformer (base)** | 27.3 | 38.1 | EN-DE: $3.3\times10^{18}$ FLOPs |

> 注：论文 Table 2 还列出了此前基线模型（如 ByteNet、ConvS2S、GNMT + RL、MoE 等）的 BLEU 分数供对比。此处仅展示 Transformer 自身结果及其定性比较结论。

#### 5.2.2 EN-DE：超越所有已发表模型与集成

- Transformer(big) 在 EN-DE 上取得 **28.4 BLEU**，**较此前所有已发表模型（含集成模型 ensembles）高出 2+ BLEU**。
- 即便是不做任何特别调优的 **base model（27.3 BLEU），也已经超越了此前所有模型与集成模型**。
- 训练成本（big）估算为 $2.3\times10^{19}$ FLOPs（基于 P100 单卡约 9.5 TFLOPS 的估算）。

#### 5.2.3 EN-FR：单模型 SOTA，训练成本不足此前最优的 1/4

- Transformer(big) 在 EN-FR 上取得 **41.0 BLEU**，建立**新的单模型 SOTA**。
- 训练耗时仅 **3.5 天（8×P100）**，论文明确指出其训练成本**低于此前最优模型的 1/4**。

#### 5.2.4 推理设置

翻译推理采用 beam search：

- **beam size = 4**；
- **length penalty $\alpha = 0.6$**（对短句施加惩罚，鼓励合理长度）；
- **checkpoint 平均**：base 模型取最后 5 个 checkpoint 的权重平均，big 模型取最后 20 个 checkpoint 平均，以提升稳定性。

> checkpoint 平均（weight averaging）是一种低成本的后处理技巧：对训练末期多个 checkpoint 做等权平均，往往比单一最优 checkpoint 略优，且不增加推理延迟。

### 5.3 模型变体消融（Table 3）

为厘清各设计选择对性能的贡献，论文在 WMT'14 EN-DE 的开发集上做了一组消融实验（以 base 配置为参照）。开发集指标为 dev BLEU（部分行含 dev PPL）。

#### 5.3.1 变体表

| 组别 | 变化 | dev BLEU（相对 base） | 关键观察 |
|------|------|:---------------------|----------|
| **base（参照）** | $h{=}8,\ d_{\text{model}}{=}512,\ d_{\text{ff}}{=}2048,\ P{=}0.1$ | **25.8** | 65M 参数 |
| (A) 头数 | $h=1$ | **24.9**（Δ −0.9） | 单头明显掉点 |
| (A) 头数 | $h=16$ | **25.8** | 头数翻倍与 base 持平 |
| (B) 缩小 $d_k$ | 减小 key/query 维度 | 质量下降 | 更小的 $d_k$ 损害质量 |
| (C) 更大 $d_{\text{model}}$ | $d_{\text{model}}=1024$ | **26.0** | 更宽 → 更好 |
| (C) 更大 $d_{\text{ff}}$ | $d_{\text{ff}}=4096$ | **26.2** | FFN 更宽 → 更好 |
| (D) 去 dropout | $P=0.0$ | **24.6**（过拟合） | dropout 至关重要 |
| (E) 学习版 PE | 可学习位置嵌入 | 与正弦版**几乎相同** | 二者近乎无差 |
| **big** | $h{=}16,\ d_{\text{model}}{=}1024,\ d_{\text{ff}}{=}4096,\ P{=}0.3,\ 300\text{K steps}$ | **26.4**（dev PPL **4.33**） | 213M 参数 |
|—|—|—|—|
| $_{\text{注}}$：Table 3 中 base 行 dev PPL 为 **4.92**；变体 (A)/(B)/(C) 还列有各自参数量，此处仅给出可在研究中确认的 dev BLEU。

#### 5.3.2 各组分析

**(A) 头数（number of heads）**

- $h=1$（单头）相对 $h=8$ 掉 **0.9 BLEU**（25.8 → 24.9），说明多头机制确实带来表示能力增益；
- 但 $h=16$ 并未进一步超过 $h=8$（同为 25.8），提示头数存在边际收益递减，过多头不会一直变好。

**(B) 缩小 $d_k$ 损害质量**

减小每头的 key/query 维度 $d_k$ 会使质量下降。这与第 3 章 "缩放点积注意力" 的论证一致：过小的 $d_k$ 会限制每个头区分不同兼容性的能力。

**(C) 更大模型更好**

- $d_{\text{model}}=1024$ → 26.0 BLEU；
- $d_{\text{ff}}=4096$ → 26.2 BLEU。

二者均高于 base 的 25.8，说明模型容量（宽度与 FFN 隐藏维度）的提升能直接转化为开发集质量——big 模型正是同时采用这两项增大并加 $h=16$。

**(D) Dropout 至关重要**

- $P=0.0$（关闭 dropout）→ **24.6 BLEU**，是各变体中**掉点最严重**的一组；
- 这表明在无 dropout 时模型发生过拟合，验证了残差/attention dropout 对 Transformer 泛化的必要性。

**(E) 学习版位置嵌入 vs 正弦位置编码**

将正弦 PE 替换为**可学习的位置嵌入**，开发集结果与正弦版**几乎相同（near identical）**。作者据此推断：正弦 PE 的采用并非因为它在 BLEU 上更优，而是因为它能**外推到比训练时更长的序列**、且不引入额外可学习参数（见第 4 章位置编码论证）。

#### 5.3.3 big 配置小结

big 模型相对 base 的提升来自三个方向同时扩容：$d_{\text{model}}\ 512\!\to\!1024$、$d_{\text{ff}}\ 2048\!\to\!4096$、$h\ 8\!\to\!16$，并将 dropout 提高到 $P=0.3$（EN-DE）、训练步数延长到 300,000，最终参数量从 **65M → 213M**，开发集 PPL 降到 **4.33**、dev BLEU 升到 **26.4**。

### 5.4 英语成分句法分析（Table 4）

为进一步检验 Transformer 的**跨任务泛化能力**，论文在英语成分句法分析（constituency parsing）任务上做了一次试探：

| 设置 | 配置 | WSJ23 F1 |
|------|------|---------:|
| 仅 WSJ（全监督） | 4 层 Transformer，$d_{\text{model}}=1024$ | **91.3** |
| 半监督（WSJ + 半监督语料） | 同上结构 | **92.7** |

要点：

- 模型为一个 **4 层的 Transformer，$d_{\text{model}}=1024$**，**没有针对句法分析任务做任何特定调参（task-specific tuning）**——直接沿用翻译任务的设计。
- WSJ-only 取得 **91.3 F1**；半监督设置进一步提升到 **92.7 F1**。
- 这一结果表明：Transformer 并非只对机器翻译过拟合，其架构本身具有**良好的任务无关泛化能力**，这是其后来能席卷多模态各领域的早期信号。

### 5.5 本章小结

| 实验 | 核心结论 | 关键数值 |
|------|----------|----------|
| Table 2 翻译 | big 在 EN-DE 超（含集成）2+ BLEU；EN-FR 单模型 SOTA，成本<1/4 | EN-DE 28.4；EN-FR 41.0 |
| Table 3 消融 | 多头有效、容量越大越好、dropout 不可缺、学习/正弦 PE 几无差 | base 25.8；big 26.4 dev BLEU |
| Table 4 句法 | 无任务特调即强，验证跨任务泛化 | WSJ-only 91.3；半监督 92.7 F1 |

实验从**质量（BLEU/F1）**与**效率（FLOPs/训练时长）**两个维度共同支撑了论文的核心主张：纯注意力架构在质量上达到/超过 SOTA 的同时，训练成本显著低于基于循环/卷积的先前模型。

---

## 第 6 章 代码实现详解

### 6.1 官方实现：tensor2tensor

论文的官方实现发布于 **tensor2tensor** 项目（TensorFlow），许可证为 **Apache License 2.0**。

- **仓库**：`https://github.com/tensorflow/tensor2tensor`
- **核心文件**：`tensor2tensor/models/transformer.py`
- **关键函数**：`transformer_encoder()`（编码器堆叠）、`transformer_decoder()`（解码器堆叠），注意力相关辅助集中在 `common_attention` 模块。

整体结构遵循论文的 encoder-decoder 范式：编码器由若干 `[Multi-Head Self-Attention, FFN]` 子层（带残差 + LayerNorm）堆叠；解码器每层额外插入 masked self-attention 与 encoder-decoder attention。

### 6.2 核心组件的参考实现

下面给出与论文公式严格对应的参考实现（NumPy 版，用于说明数学结构；生产实现见 tensor2tensor 的 TensorFlow 版本）。

#### 6.2.1 Scaled Dot-Product Attention

对应公式 (1) $\text{Attention}(Q,K,V)=\text{softmax}(QK^{\top}/\sqrt{d_k})V$。`mask` 用于解码器的因果掩码（未来位置置 $-\infty$）。

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)   # 数值稳定
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (..., seq_q, d_k)   K: (..., seq_k, d_k)   V: (..., seq_k, d_v)
    返回: (..., seq_q, d_v)
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)   # (..., seq_q, seq_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)       # 屏蔽位置 -> softmax 权重为 0
    weights = softmax(scores, axis=-1)                   # 注意力权重
    return weights @ V                                   # (..., seq_q, d_v)
```

#### 6.2.2 Multi-Head Attention

对应 $\text{MultiHead}=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^{O}$，每个 $\text{head}_i=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)$。实现要点：用一组大矩阵一次性投影出所有头的 $Q/K/V$，再 reshape 成 `(batch, h, seq, d_k)` 并行计算。

```python
def split_heads(x, h):
    """(batch, seq, d_model) -> (batch, h, seq, d_k)"""
    batch, seq, d_model = x.shape
    d_k = d_model // h
    x = x.reshape(batch, seq, h, d_k)        # 分头
    return x.transpose(0, 2, 1, 3)           # 头放到 batch 维后，便于并行

def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, h, mask=None):
    """
    W_q/W_k: (d_model, d_model)  W_v: (d_model, d_model)  W_o: (d_model, d_model)
    Q/K/V: (batch, seq, d_model)
    """
    q = split_heads(Q @ W_q, h)             # (batch, h, seq_q, d_k)
    k = split_heads(K @ W_k, h)             # (batch, h, seq_k, d_k)
    v = split_heads(V @ W_v, h)             # (batch, h, seq_k, d_v)

    attn = scaled_dot_product_attention(q, k, v, mask)   # (batch, h, seq_q, d_v)
    batch, _, seq_q, d_v = attn.shape
    concat = attn.transpose(0, 2, 1, 3).reshape(batch, seq_q, h * d_v)  # 拼接头
    return concat @ W_o                    # (batch, seq_q, d_model)
```

> 由于每头维度被压到 $d_k=d_v=d_{\text{model}}/h$，$h$ 个头并行后的总计算量与单头全维度几乎相同，却获得了多头表示能力（见第 3 章）。

#### 6.2.3 Positional Encoding

对应正弦/余弦位置编码，波长构成从 $2\pi$ 到 $10000\cdot2\pi$ 的几何级数：

```python
def positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, None]                   # (max_len, 1)
    div = np.power(10000.0, 2 * (np.arange(d_model) // 2) / d_model)  # (d_model,)
    pe[:, 0::2] = np.sin(pos / div[0::2])               # 偶数维 sin
    pe[:, 1::2] = np.cos(pos / div[1::2])               # 奇数维 cos
    return pe                                            # (max_len, d_model)，与嵌入相加
```

#### 6.2.4 FFN、残差与完整层

FFN 是逐位置的两层线性 + ReLU；每个子层包在 $\text{LayerNorm}(x+\text{Dropout}(\text{Sublayer}(x)))$ 中。

```python
def feed_forward(x, W1, b1, W2, b2):
    """位置级 FFN: max(0, xW1+b1)W2+b2，对所有位置共享参数"""
    return np.maximum(0, x @ W1 + b1) @ W2 + b2

def encoder_layer(x, attn_params, ffn_params, h, dropout_p, training):
    # 子层 1: Multi-Head Self-Attention + 残差 + LayerNorm
    a = multi_head_attention(x, x, x, *attn_params, h)
    if training: a = dropout(a, dropout_p)
    x = layer_norm(x + a)
    # 子层 2: FFN + 残差 + LayerNorm
    f = feed_forward(x, *ffn_params)
    if training: f = dropout(f, dropout_p)
    x = layer_norm(x + f)
    return x
```

解码器层结构类似，但在 self-attention 上施加因果掩码（mask 上三角为 0），并额外插入一层 encoder-decoder attention（$Q$ 来自解码器，$K/V$ 来自编码器输出）。

### 6.3 训练与推理流程要点

- **训练**：以 token 计量的批（~25K src + 25K tgt tokens），Adam（$\beta_1{=}0.9,\beta_2{=}0.98,\varepsilon{=}10^{-9}$）+ warmup=4000 的学习率调度，残差/attention dropout + label smoothing。
- **推理**：beam search（beam=4，length penalty $\alpha=0.6$），对末尾若干 checkpoint 做权重平均（base 取 5、big 取 20）。解码自回归生成，每步在 encoder-decoder attention 上复用编码器输出。

---

## 第 7 章 局限性与延伸阅读

### 7.1 论文自身的局限

尽管 Transformer 取得了里程碑式成果，原文及后续工作也指出了若干局限：

1. **自注意力的 $O(n^2\cdot d)$ 复杂度限制长序列**。每层 self-attention 须计算 $n\times n$ 的注意力矩阵，序列长度 $n$ 增大时计算与内存呈平方增长，难以直接处理长文档、长视频、高分辨率图像等长序列输入（参见第 4 章 Table 1）。

2. **"受限注意力"（restricted attention）仅提出而未实现**。论文在 Table 1 列出了 self-attention(restricted) 这一行（复杂度 $O(r\cdot n\cdot d)$、路径 $O(n/r)$），把固定邻域窗口内的注意力作为应对长序列的设想，但**论文本身并未在实验中实现或验证它**，留待后续工作。

3. **缺乏条件计算（conditional computation）**。论文未引入按需激活部分参数的机制（如 mixture-of-experts），模型容量增长必然带来等比例的计算增长。

4. **仅在两类翻译任务 + 一个句法任务上验证**。原文的实证范围限于 WMT'14 EN-DE、EN-FR 与 WSJ 成分句法分析，对图像、音频、视频等非文本模态尚未给出实验（论文结论中作为"未来工作"提及）。

### 7.2 后续发展（延伸阅读）

Transformer 的"纯注意力"思想很快被推广到几乎所有模态与范式，代表性脉络如下：

- **预训练语言模型**：**BERT**（Devlin et al., 2018）使用 Transformer encoder 做双向预训练；**GPT 系列**（Radford et al., 2018 起）使用 Transformer decoder 做自回归预训练；**T5**（Raffel et al., 2019）将一切任务统一为 encoder-decoder 的 text-to-text 范式。
- **视觉**：**ViT**（Dosovitskiy et al., 2020）把图像切成 patch 序列后送入 Transformer encoder，证明注意力架构在视觉分类上可媲美/超越 CNN。
- **语音**：**Speech Transformer** 等工作将 Transformer 引入语音识别，逐步取代 RNN/Attention 混合的声学模型。
- **高效注意力**：为缓解 $O(n^2)$ 瓶颈，涌现出大量近似/优化方案，其中 **FlashAttention**（Dao et al., 2022）通过分块与 IO 感知的精确注意力实现，在不牺牲精度的前提下大幅降低显存与加速训练。
- **位置编码的演进**：**RoPE**（旋转位置编码，Su et al., 2021）等相对位置编码方案取代了原始正弦 PE，在长序列外推上表现更优。
- **规模化与稀疏化**：**MoE（Mixture-of-Experts）** 被重新引入 Transformer（如 GShard、Switch Transformer），以条件计算的方式在固定算力下扩大模型容量——正是回应了 7.1 节中"缺乏条件计算"的局限。

---

## 第 7 章小结

| 主题 | 要点 |
|------|------|
| 复杂度瓶颈 | self-attention $O(n^2 d)$，长序列受限 |
| 未竟设想 | restricted attention 仅在 Table 1 提出，论文未实现 |
| 验证范围 | 仅两类翻译 + WSJ 句法 |
| 后续脉络 | BERT/GPT/T5（语言）、ViT（视觉）、Speech Transformer（语音）、FlashAttention（高效）、RoPE（位置编码）、MoE（稀疏扩容） |

至此，第三部分完成了对 Transformer 的"效果如何（第 5 章实验）、如何复现（第 6 章代码）、边界与影响（第 7 章局限与延伸）"的完整收束。结合前两部分（核心机制、完整架构），三部分共同构成了对《Attention Is All You Need》从动机、机制、架构、实验到生态影响的完整精读。
