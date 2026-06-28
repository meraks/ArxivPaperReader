# LoRA: Low-Rank Adaptation of Large Language Models

## 论文元数据

> 标题: LoRA: Low-Rank Adaptation of Large Language Models  
> 作者: Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft)  
> 发表: ICLR 2022   
> arXiv: 2106.09685  
> 官方代码: [LoRA](github.com/microsoft/LoRA)

---

## Ch1: 概述与核心贡献

LoRA 冻结预训练权重 $W_0$，注入可训练低秩矩阵 $BA$（$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$）：

$$
h = W_0 x + \Delta W x = W_0 x + BA x
$$

**初始化策略**：$A \sim \mathcal{N}(0, \sigma^2)$ 高斯初始化，$B = 0$（即 $\Delta W = 0$ 开始，训练起点与原模型一致）。

**推理零额外延迟**：推理时将 $BA$ 合并回原权重：

$$
W = W_0 + BA
$$

无需额外的前向计算开销。任务切换仅需 MB 级 LoRA 模块（对比全模型 350 GB）。

### 核心结果（GPT-3 175B）

- **可训练参数减少**：最多约 37,000×（175.3B→4.7M，仅 0.0027%）
- **训练 VRAM 减少**：约 3.4×（1.2TB→350GB）
- **检查点压缩**：约 10,000×（约 350GB→35MB，按论文里的常见 GPT-3 LoRA 配置估算）
- **训练加速**：约 25%（全量微调 32.5 tokens/s/V100 → LoRA 43.1 tokens/s/V100）

---

## Ch2: 背景

### 全量微调的困境

- GPT-3 175B 需 **1.2 TB** GPU 内存
- 部署 100+ 个任务模型 → **35 TB** 存储
- 成本极高，难以扩展到多任务场景

### 已有方案的局限

| 方案 | 提出者 | 机制 | 局限 |
|------|--------|------|------|
| **Adapter** | Houlsby (2019) | MLP `d→r→d` 层 | 推理需顺序通过，引入延迟 |
| **Prefix Tuning** | Li & Liang (2021) | 可学习虚拟 token | 占用可用序列长度，优化困难 |
| **BitFit** | Zaken (2021) | 仅微调 bias 参数 | 表达能力有限 |

**Adapter 延迟的定量证据**：原文 Table 1 在 GPT-2 medium 上测量了 Adapter 的推理延迟。在 batch size=1、sequence length=128 的在线推理场景下，AdapterH 引入 **+30.3%** 延迟，AdapterL 引入 **+20.7%** 延迟。即便在 batch size=32 的离线场景下，延迟仍增加 2%~3%。这是因为 Adapter 层必须**串行**处理，无法利用 GPU 并行性——在大模型分片（model parallelism）场景下，额外的深度还需要更多同步 GPU 操作（AllReduce、Broadcast），进一步加剧延迟。

**Prefix Tuning 的困难**：直接优化前缀 token 的嵌入向量**优化不稳定**，性能随可训练参数量非单调变化。更根本的问题是，为适配预留序列长度会**挤占任务可用的上下文长度**，降低模型处理长文本的能力。

### LoRA 的动机

预训练模型存在**过度参数化**现象。任务适配所需的权重更新变化实际发生在**低维子空间**中，即「**低内在秩假设**」(low intrinsic dimension hypothesis)。

这正是 LoRA 用低秩分解 $\Delta W = BA$ 近似全量更新的理论依据：既然真正的适配信号是低秩的，就不必训练整个 $W_0$。

### 问题形式化

给定预训练语言模型 $P_\Phi(y|x)$（如 GPT-3），下游任务的数据集 $\mathcal{Z}=\{(x_i,y_i)\}_{i=1}^{N}$，全量微调的目标是：

$$
\max_{\Phi} \sum_{(x,y)\in\mathcal{Z}} \sum_{t=1}^{|y|} \log\left(P_\Phi(y_t|x,y_{<t})\right)
$$

全量微调为**每个**下游任务学习不同的参数增量 $\Delta\Phi$，其维度 $|\Delta\Phi|=|\Phi_0|$。对于 GPT-3（$|\Phi_0|\approx 175$B），存储和部署多个独立的微调模型代价极高。

参数高效微调的核心思路是：用远小于 $|\Phi_0|$ 的参数集 $\Theta$（$|\Theta|\ll|\Phi_0|$）来编码 $\Delta\Phi=\Delta\Phi(\Theta)$，将优化目标变为：

$$
\max_{\Theta} \sum_{(x,y)\in\mathcal{Z}} \sum_{t=1}^{|y|} \log\left(P_{\Phi_0+\Delta\Phi(\Theta)}(y_t|x,y_{<t})\right)
$$

LoRA 的贡献在于用低秩分解来参数化 $\Delta\Phi$，在 GPT-3 上可将 $|\Theta|$ 压缩至 $|\Phi_0|$ 的 **0.01%** 以下。

---

## Ch3: LoRA 方法详解

### 3.1 核心假设：权重更新的低秩分解

LoRA（Low-Rank Adaptation）的出发点是一个经验性假设：**预训练大模型在下游任务上的适配，本质上是"低内在秩"（low intrinsic rank）的**。也就是说，任务特异性信息并不需要动用全部参数维度，而是可以被压缩到一个远小于原参数矩阵的子空间中。基于这一假设，LoRA 不再学习完整的权重增量 $\Delta W$，而是把增量约束为两个小矩阵的乘积。

对预训练权重 $W_0\in\mathbb{R}^{d\times k}$，LoRA 将更新写为低秩分解：

$$
h = W_0x + \Delta W x = W_0x + BAx, \quad B\in\mathbb{R}^{d\times r},\; A\in\mathbb{R}^{r\times k},\; r\ll\min(d,k)
$$

其中 $W_0$ 在训练过程中**始终保持冻结**，只有 $B$、$A$ 这两个低秩因子参与梯度更新。秩 $r$ 通常取得远小（如 $r=4, 8$），使可训练参数量从 $d\times k$ 降为 $(d+k)\times r$。

### 3.2 初始化与缩放

LoRA 采用一种**非对称的初始化**，保证训练伊始模型行为不发生突变：

- $A$ 用随机高斯初始化 $A\sim\mathcal{N}(0,\sigma^2)$；
- $B$ 初始化为零矩阵 $B=0$。

于是初始时 $\Delta W = BA = 0$，模型从与预训练完全一致的状态出发，逐步学习增量。训练时，$\Delta W x = BAx$ 会按缩放因子 $\alpha/r$ 进行缩放（$\alpha$ 为超参），从而把学习率与秩 $r$ 解耦——改变 $r$ 时不必重新调学习率。

**$\alpha$ 的实践策略**：论文指出，当使用 Adam 优化器时，调 $\alpha$ 约等于调学习率（在适当缩放初始化的前提下）。因此论文的实践是：**将 $\alpha$ 设为第一次尝试的 $r$ 值，之后不再调整**。这极大简化了超参搜索——改变 $r$ 时只需固定 $\alpha$，无需同步重调学习率。

### 3.3 推理：可合并、零额外延迟

LoRA 的一个关键工程优势在于：训练得到的 $BA$ 在推理时可以**直接合并回**预训练权重：

$$
W = W_0 + BA
$$

合并后，推理计算路径与原模型完全一致，**没有任何额外的前向延迟**，也不会像 Adapter 那样增加串行层数、像 Prefix Tuning 那样占用序列长度。当需要在不同任务间切换时，只需替换 LoRA 模块 $(B,A)$，而共享的 $W_0$ 始终不变——这让"一个基座 + 多个轻量任务插件"的部署模式成为可能。

### 3.4 应用到 Transformer

在 Transformer 架构中，self-attention 的投影矩阵 $W_q, W_k, W_v, W_o$ 都可以挂载 LoRA 适配器，而 MLP 层通常保持冻结。论文的实证结论更准确地说是：**只适配 $W_q$ 与 $W_v$ 往往能拿到很强的性价比**——在 GPT-3 的多项任务上，它通常能以更小预算逼近或达到适配更多矩阵的效果。这一"关键少数"策略是 LoRA 落地的常用默认值，但并不是对所有任务都唯一最优。

### 3.5 参数量分析

若对每个适配层只挂载一对因子，单个权重矩阵的 LoRA 参数量为 $(d+k)\cdot r$。若该矩阵是方阵且只适配一层，则约为 $2d_{\text{model}}r$；若按论文最常用的 self-attention 里的 $W_q$ 与 $W_v$ 两个矩阵一起适配，则每层约为 $4d_{\text{model}}r$。以 GPT-3 为例，$L=96$、$d_{\text{model}}=12288$，即便在较大秩下，LoRA 参数相比 1750 亿的全量参数仍是九牛一毛。

### 3.6 LoRA 是全量微调的泛化

从线性代数角度看，若把秩放到足够大，LoRA 形式当然可以表示任意权重更新；但论文更强调的是：在实际下游任务里，真正有用的更新往往只落在低秩子空间中。也就是说，LoRA 的重点不是“提出一个全量微调的新表述”，而是证明“用很小的秩就足够好”。

### 3.7 训练效率优势

冻结绝大部分参数带来了三方面的系统级收益：

- **训练加速约 25%**：冻结参数无需计算梯度，在 GPT-3 175B 上观察到训练吞吐从全量微调的 **32.5 tokens/s/V100** 提升至 LoRA 的 **43.1 tokens/s/V100**（约 32% 提升，论文正文保守概括为 25%）。
- **训练 VRAM 大幅下降**：仅需为 LoRA 参数维护优化器状态（如 Adam 的一阶/二阶矩），而无须为冻结参数分配 optimizer states。在 GPT-3 175B 上，训练 VRAM 从 **1.2TB 降至 350GB**（约 3.4× 压缩）。
- **检查点体积压缩约 10,000×**：从全量微调的约 **350GB** 检查点压缩到 LoRA 的约 **35MB**（这个量级对应论文里 GPT-3 的常见 q/v 适配配置，而不是 4.7M 参数那一档），便于存储、分发与版本管理。实际部署时，存储 100 个适配模型仅需 350GB（基座）+ 35MB×100 ≈ **354GB**，而全量微调需 350GB×100 ≈ **35TB**。

这三点共同决定了 LoRA 在"大模型 + 多任务"场景下的工程可行性。

---

## Ch4: 实验结果

LoRA 在从百兆级到千亿级的模型上进行了系统验证，核心结论是：**在远低于 1% 的可训练参数下，LoRA 能够匹配甚至超越全量微调（Full Fine-Tuning, FT）。**

### 4.1 RoBERTa on GLUE

Table 2 在 GLUE 基准上对比了 RoBERTa base / large 与 DeBERTa XXL 的全量微调与 LoRA：

| 方法 | 可训练参数 | Avg GLUE |
|------|-----------|----------|
| RoBERTa base FT | 125.0M | 86.4 |
| RoBERTa base LoRA | **0.3M** | **87.2** |
| RoBERTa large FT | 355.0M | 88.9 |
| RoBERTa large LoRA | **0.8M** | **89.0** |
| DeBERTa XXL FT | 1500.0M | 91.1 |
| DeBERTa XXL LoRA | **4.7M** | **91.3** |

**观察**：在三种模型规模上，LoRA 均以**不到全量 1%** 的参数量取得**相同或更高**的平均 GLUE 分数。模型越大，LoRA 的相对优势越明显（DeBERTa XXL 上 LoRA 达 91.3，超越 FT 的 91.1），这与"大模型内在秩更低、适配更易"的直觉一致。

### 4.2 GPT-2 on E2E NLG

Table 3 在端到端自然语言生成任务 E2E NLG 上，将 LoRA 与全量微调、Adapter、Prefix Tuning 进行对比：

| 方法 | 参数 | BLEU |
|------|------|------|
| GPT-2 M FT | — | 68.2 |
| GPT-2 M LoRA | 0.35M | **70.4** |
| GPT-2 L FT | — | 68.5 |
| GPT-2 L LoRA | 0.77M | **70.4** |

**观察**：GPT-2 M 的 LoRA（0.35M 参数）取得 BLEU 70.4，超过其全量微调的 68.2 和 PreLayer 的 69.7；GPT-2 L 的 LoRA（0.77M 参数）同样取得 70.4，超过全量微调的 68.5 和 PreLayer 的 70.3。在同等参数预算下，LoRA 一致优于 Adapter 与 Prefix Tuning，说明低秩约束在此任务上兼具"省参数"与"防过拟合"的双重作用。

### 4.3 GPT-3 175B

Table 4 将验证推进到 1750 亿参数的 GPT-3，覆盖自然语言理解（MNLI-m）、结构化推理（WikiSQL）与摘要生成（SAMSum）。论文测试了两种 LoRA 参数预算配置：

| 方法                 | 参数量        | WikiSQL  | MNLI-m   | SAMSum (R1/R2/RL)  |
| ------------------ | ---------- | -------- | -------- | ------------------ |
| FT                 | 175,255.8M | 73.8     | 89.5     | 52.0 / 28.0 / 44.5 |
| LoRA (r$_q=r_v=1$) | **4.7M**   | 73.4     | **91.7** | 53.8 / 29.8 / 45.9 |
| LoRA ($r_q=r_v=8$) | **37.7M**  | **74.0** | 91.6     | 53.4 / 29.2 / 45.1 |

其中 4.7M 配置对应 $r_q=r_v=1$ 或仅 $r_v=2$；37.7M 配置对应 $r_q=r_v=8$ 或 $r_q=r_k=r_v=r_o=4$。$r_q=r_k=r_v=r_o=2$ 对应的是 18.8M 这一档，不是 37.7M。

**观察**：
- **参数占比极低**：4.7M 可训练参数仅为全量微调参数（175,255.8M）的 **0.0027%**；37.7M 约 **0.022%**。
- **效果全面超越 FT**：两种 LoRA 配置在三个任务上均匹配或超越全量微调——WikiSQL 达 73.4/74.0（FT 73.8）；MNLI-m 达 91.7/91.6（FT 89.5），提升尤为显著；SAMSum 的 R1/R2/RL 全面优于 FT。
- **参数量与效果非单调**：4.7M 配置在 MNLI-m（91.7 vs 91.6）和 SAMSum（53.8/29.8/45.9 vs 53.4/29.2/45.1）上反而优于 37.7M 配置，说明更多参数并不保证更好效果，印证了"任务内在秩低、小秩已足够"的假设。

### 4.4 实验小结

三个规模层级的实验共同印证了 LoRA 的核心论断：**下游适配所必需的信息维度远小于模型参数维度，低秩分解足以捕捉。** 具体而言：

1. **参数效率**：LoRA 始终以远低于 1%（在 GPT-3 上低至约 0.002%）的可训练参数运行。
2. **效果不损反超**：从 RoBERTa、GPT-2 到 GPT-3，LoRA 均匹配或超越全量微调，并优于 Adapter、Prefix Tuning 等高效适配方法。
3. **规模收益**：模型越大，LoRA 的相对优势越突出。原文 Figure 2 展示了 GPT-3 上各方法的"参数量-性能"曲线，LoRA 在论文比较的主要基线中表现始终很强。
4. **工程友好**：可合并、零推理延迟、检查点压缩约 10,000×，使 GPT-3 级别的多任务微调在实际系统中变得可行。

### 4.5 补充实验

**LoRA 与 Prefix Tuning 的组合**（Appendix E）：LoRA 与 Prefix Tuning 是**正交**的，可以同时使用。实验表明两者组合能在部分任务上带来额外增益，说明低秩权重更新与可学习前缀捕捉了不同层面的任务信息。

**GPT-2 在 DART / WebNLG 上的补充实验**（Appendix F.1）：论文还在 DART 和 WebNLG 上复现了 GPT-2 的结果，LoRA 在相同参数预算下继续保持与前缀类方法持平或更好的表现，说明它并不是只在 E2E 这一个任务上有效。

**低数据场景**（Appendix F.3）：在训练样本极少（如 100~1000 条）的低数据场景下，LoRA 相比全量微调具有更明显的优势——低秩约束本身起到了**正则化**作用，有效防止过拟合。这在实际应用中尤为重要，因为许多下游任务的标注数据有限。

---

## Ch5: 理解低秩更新

### 5.1 哪些权重适配（Which Weight Matrices to Adapt）

在 Transformer 中分别对注意力投影矩阵 $W_q, W_k, W_v, W_o$ 做适配组合实验，结论是：

- **联合适配 $W_q$ 与 $W_v$ 在同一参数预算下取得最佳质量**，是性价比最高的默认选择。
- 仅适配单种投影（如仅 $W_q$ 或仅 $W_k$）效果显著偏弱。
- 适配所有投影会带来额外参数开销，但增益有限。

这一发现支撑了 LoRA 的工程默认配置：只需对 Query 与 Value 两个投影施加低秩更新，即可逼近全量微调的效果。

### 5.2 最优秩 $r$

实验观察到秩的"甜区"非常低：

- **在 GPT-3 的一些任务上，$r=1$ 就已具备竞争力**；总体上，$r\le 8$ 对多数任务已经足够。
- 继续增大 $r$ 无显著增益（NER 等需要边界精度的任务需要略高的 $r$）。

为理解为何小秩就够，论文比较了不同 $r$ 下学到的 $\Delta W$ 的奇异向量方向。结果显示：**$r=8$ 与 $r=64$ 学到的 top 奇异向量高度重叠**（子空间相似度极高），即增大 $r$ 并未引入明显新的更新方向。这直接说明**任务适配所驱动的权重变化集中在一个低维子空间内**，过大的 $r$ 只是冗余。
在 GPT-3 的 WikiSQL / MultiNLI 上，论文还观察到 `W_q + W_v` 的小秩配置已经非常强，而只改单个矩阵通常需要更大的秩才能接近同等效果。

### 5.3 $\Delta W$ 与 $W$ 的关系

将 $\Delta W$ 与 $W$ 各自做奇异值分解并比较其主方向，得到一个反直觉但重要的结论：

- $\Delta W$ 的 top 奇异方向 **并不** 与 $W$ 的 top 方向对齐。
- 相反，$\Delta W$ 放大的是 $W$ 中**未被强调（spectral 中较低权重）的方向**——对 $r=4$ 的设置，对应的放大倍数约为 **21.5×**。

**解释**：微调并非从零学习全新特征，而是**增强预训练权重中已隐含、但未被该下游任务充分利用的任务特定特征**。预训练已提供了丰富的方向空间，LoRA 只需沿其中若干方向放大即可完成适配，这与低秩假设一致。

### 5.4 内在秩假设的验证

综合上述证据：

1. 极小的 $r$（$1\sim 8$）即足以匹配全量微调；
2. 不同 $r$ 的更新子空间高度重叠；
3. $\Delta W$ 集中放大低维方向而非扩张到全空间。

这些证据一致支持论文的核心假设：任务适配所引起的权重变化 $\Delta W$ 具有低"内在秩"（intrinsic rank）。这也解释了为何过度参数化的全量微调在统计与计算上都是浪费——真正需要的自由度远少于权重本身。

---

## Ch6: 代码实现详解

官方实现仓库为 [github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)，核心代码集中在 `loralib/` 包中。除 `Linear` 外，官方还支持 `Embedding`、`Conv1d`、`Conv2d`、`Conv3d` 等层类型的 LoRA 适配。

### 6.1 `LoRALayer` 基类

所有 LoRA 层共享一个混入类 `LoRALayer`，管理公共超参和状态标记：

```python
class LoRALayer():
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
```

关键设计：`merged` 标记追踪权重是否已合并，`merge_weights` 控制是否在 eval 时自动合并（设为 `False` 可禁用自动合并）。

### 6.2 `lora.Linear`：核心 Linear 层

`lora.Linear` 继承自 `torch.nn.Linear` 并混入 `LoRALayer`，在冻结的原权重 $W$ 之外，额外维护一对低秩分解 $BA$：

```python
class Linear(nn.Linear, LoRALayer):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1,
                 lora_dropout=0., fan_in_fan_out=False, merge_weights=True, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False  # 冻结原始权重
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
```

**初始化差异（代码 vs 论文）**：
- **论文**：$A \sim \mathcal{N}(0, \sigma^2)$ 高斯，$B = 0$
- **代码**：`lora_A` 用 **Kaiming uniform**（$a=\sqrt{5}$），`lora_B` 用 **零初始化**

```python
def reset_parameters(self):
    nn.Linear.reset_parameters(self)
    if hasattr(self, 'lora_A'):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
```

代码注释指出："这与论文描述不同，但不影响性能"——因为 $B=0$ 保证初始 $\Delta W=0$，$A$ 的初始化分布不影响这一不变性。

**`fan_in_fan_out` 参数**：某些模型（如 GPT-2）的权重存储为 `(fan_in, fan_out)` 而非标准的 `(fan_out, fan_in)`，此参数控制是否在初始化时转置权重。

**前向传播**：

```python
def forward(self, x):
    def T(w):
        return w.transpose(0, 1) if self.fan_in_fan_out else w
    if self.r > 0 and not self.merged:
        result = F.linear(x, T(self.weight), bias=self.bias)
        result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
                   @ self.lora_B.transpose(0, 1)) * self.scaling
        return result
    else:
        return F.linear(x, T(self.weight), bias=self.bias)
```

即 $y = W_0 x + \text{dropout}(x) \cdot A^\top B^\top \cdot \frac{\alpha}{r}$。

**Train/Eval 模式自动合并**：

```python
def train(self, mode=True):
    def T(w):
        return w.transpose(0, 1) if self.fan_in_fan_out else w
    nn.Linear.train(self, mode)
    if mode:
        if self.merge_weights and self.merged:
            # Unmerge: 从 W 中减去 BA·scaling
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    else:
        if self.merge_weights and not self.merged:
            # Merge: 将 BA·scaling 合并进 W
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True
```

调用 `model.eval()` 时自动合并，调用 `model.train()` 时自动分离——训练时走双路径，推理时走单路径。

### 6.3 `MergedLinear`：选择性适配 QKV

许多模型把 Q/K/V 三个投影打包在一个 `nn.Linear(d, 3d)` 里。`MergedLinear` 允许对子投影做选择性适配：

```python
class MergedLinear(nn.Linear, LoRALayer):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1,
                 lora_dropout=0., enable_lora=[False], fan_in_fan_out=False,
                 merge_weights=True, **kwargs):
        # ...
        self.enable_lora = enable_lora  # e.g. [True, False, True] for Q/K/V
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros(
                    (out_features // len(enable_lora) * sum(enable_lora), r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            # Boolean mask: 哪些输出维度受 LoRA 影响
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
```

**`merge_AB()` 的高效实现**——用 **group conv1d** 完成子空间切片与合并：

```python
def merge_AB(self):
    def T(w):
        return w.transpose(0, 1) if self.fan_in_fan_out else w
    delta_w = F.conv1d(
        self.lora_A.unsqueeze(0),       # (1, r*sum(enable), in)
        self.lora_B.unsqueeze(-1),      # (out//n*sum(enable), r*sum(enable), 1)
        groups=sum(self.enable_lora)
    ).squeeze(0)
    return T(self.zero_pad(delta_w))   # zero_pad 将结果填回完整维度
```

`zero_pad` 用 `lora_ind` 布尔掩码，将 LoRA 输出填到正确的位置（跳过未适配的子投影）。

**使用方式**：

```python
# 只对 Q 和 V 施加 LoRA，K 保持不变
qkv_proj = lora.MergedLinear(d_model, 3*d_model, r=8, enable_lora=[True, False, True])
```

### 6.4 其他支持的层类型

官方除 `Linear` 外还支持：

- **`lora.Embedding`**：$A$ 用零初始化，$B$ 用正态初始化（与 `Linear` 相反），因为 Embedding 的前向路径是 lookup 而非 matmul。
- **`lora.Conv1d` / `Conv2d` / `Conv3d`**：通过 `ConvLoRA` 基类统一实现，将卷积权重 reshape 后做低秩分解。

### 6.5 `loralib/utils.py`：参数管理工具

```python
def mark_only_lora_as_trainable(model, bias='none'):
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
```

`bias` 参数有三种模式：`'none'`（不训练任何 bias）、`'all'`（训练所有 bias）、`'lora_only'`（只训练 LoRA 层的 bias）。

```python
def lora_state_dict(model, bias='none'):
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
```

导出时只保留 `lora_` 前缀的参数，checkpoint 通常仅几 MB。

### 6.6 典型使用流程

```python
import loralib as lora

# 1) 用 LoRA 层替换目标投影层
self.query = lora.Linear(768, 768, r=8)
# 或者对打包的 QKV 做选择性适配
self.qkv_proj = lora.MergedLinear(768, 3*768, r=8, enable_lora=[True, False, True])

# 2) 冻结所有非 LoRA 参数（可选训练 bias）
lora.mark_only_lora_as_trainable(model, bias='lora_only')

# 3) 训练循环
for batch in dataloader:
    loss = model(batch)
    loss.backward()    # 只有 lora_A, lora_B 有梯度
    optimizer.step()

# 4) 保存：只存 LoRA 参数
torch.save(lora.lora_state_dict(model, bias='lora_only'), 'lora.pt')

# 5) 推理：model.eval() 自动合并权重，零额外延迟
model.eval()
output = model(x)  # 无 LoRA 路径开销

# 6) 任务切换：分离 → 替换 LoRA → 重新合并
model.train()  # 自动分离
# ... 加载新任务的 LoRA 权重 ...
model.eval()   # 自动合并新权重
```

### 6.7 设计要点总结

| 设计选择 | 实现方式 | 与论文的关系 |
|---------|---------|------------|
| A 初始化 | Kaiming uniform ($a=\sqrt{5}$) | 论文说高斯，代码注释说"不影响性能" |
| B 初始化 | 零初始化 | 与论文一致，保证 $\Delta W=0$ |
| 缩放因子 | $\alpha / r$ | 与论文一致 |
| 合并时机 | eval 时自动合并，train 时自动分离 | 论文未详述，代码工程实现 |
| Bias 训练 | 可选 `'none'/'all'/'lora_only'` | 论文未详述，代码扩展 |
| 支持层类型 | Linear, Embedding, Conv1d/2d/3d | 论文说"适用于任何 dense layer" |
| fan_in_fan_out | 处理权重存储格式差异 | 论文未提及，代码处理 GPT-2 等兼容性 |

### 6.8 HuggingFace `peft` 生态

2023 年，LoRA 被 **HuggingFace `peft`** 库采纳为**核心方法之一**，并与 transformers 生态深度融合。使用 `peft` 库的典型流程：

```python
from peft import LoraConfig, get_peft_model, TaskType

# 配置：只适配 Q/V，r=8，alpha=16
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # 论文推荐的默认值
)

# 注入 LoRA
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()
# trainable params: 1,572,864 || all params: 6,744,647,680 || trainable%: 0.0233%

# 训练后保存（只保存 LoRA 权重）
model.save_pretrained("./my_lora_adapter")

# 加载：自动注入 LoRA
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./my_lora_adapter")
```

---

## Ch7: 局限性与延伸阅读

### 7.1 局限性

论文本身承认或暴露出若干局限：

1. **最优秩的选择缺乏理论指导**：$r$ 仍主要靠经验调参，尚无原则性方法预测给定任务/模型所需的秩。
2. **高精度任务需要更高的 $r$**：NER 等对边界/类别精度敏感的任务需要比 $r\le 8$ 更大的秩才能匹配全量微调，说明"极低秩足够"并非普适。
3. **部分实验中参数增量计算有近似**：在分析 $\Delta W$ 的放大倍数等时，存在对规模/分解的近似处理，结论应理解为量级上的趋势而非精确数值。

此外，从工程实践角度还应注意：低秩分解在极宽/极深的模型上对初始化与缩放仍较敏感；推理时合并权重虽"零开销"，但**多任务多 LoRA 在线切换**时需在显存中分别驻留 $A/B$。如果想把多个任务混在同一个 batch 里推理，且每个任务对应不同的 LoRA 权重，那么就不能简单把所有 LoRA 一次性吸收到同一份基座权重里。

### 7.2 延伸阅读

**同期参数高效方法（与 LoRA 形成对比）**：

- **Adapter** (Houlsby et al., 2019)：在层内插入小型瓶颈模块，但增加推理延迟——LoRA 通过权重合并避免了这一点。
- **Prefix Tuning** (Li & Liang, 2021)：在注意力键值前加可学前缀，优化输入而非权重。
- **BitFit** (Zaken et al., 2021)：仅微调 bias 项，是更激进的极简基线。

**生态**：

- **HuggingFace `peft`**：统一封装 LoRA/Adapter/Prefix-Tuning 等的工业级库。

**后续 LoRA 变体**：

- **QLoRA**：将基座权重量化为 4-bit 后再叠加 LoRA，实现消费级显卡微调超大模型。
- **AdaLoRA**：在训练中自适应分配秩预算，按重要性裁剪/扩张各层的 $r$。
- **DoRA**：将权重更新分解为方向（direction）与幅度（magnitude）两路分别适配，提升表达力。

---

## 总结

| 维度 | 核心结论 |
|------|---------|
| **适配对象** | 联合适配 $W_q, W_v$ 是论文里最常用且性价比很高的默认值 |
| **最优秩** | $r=1$ 已有竞争力，$r\le 8$ 足够；增大 $r$ 收益递减，且更多参数不保证更好效果 |
| **$\Delta W$ 本质** | 放大 $W$ 中未强调方向（约 $21.5\times$），非学习全新特征 |
| **内在秩** | 多项证据支持任务适配变化发生在低维子空间 |
| **理论位置** | 从线性代数上可覆盖 full-rank 更新，但论文重点是低秩子空间足够用 |
| **工程优势** | train/eval 自动合并权重 → **零推理开销**；检查点压缩 ~10,000× |
| **主要局限** | 秩选择缺理论指导；高精度任务需更高 $r$；部分分析有近似 |

LoRA 的贡献可概括为：**用一个 $BA$ 低秩分解，同时回答了"参数高效微调为何可行"（低内在秩）与"如何零代价部署"（权重合并）两个问题**，由此奠定了 2022 年后参数高效微调方法的基本范式。

### 7.3 论文原文还强调的两点

- **适用范围不止 Transformer**：结论部分明确写到，LoRA 的原则也适用于“带有 dense layers 的任意神经网络”，并不只局限于语言模型。
- **矩阵选择仍偏经验主义**：论文在结论里把“更原则地选择哪些权重矩阵施加 LoRA”列为未来工作，这一点比“只改 Q/V 就行”的表述更谨慎。
