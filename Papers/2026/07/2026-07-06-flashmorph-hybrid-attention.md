# Morphing into Hybrid Attention Models：FlashMorph 方法与实验分析

## Ch1 论文概述与核心贡献

### 1.1 背景与问题

Transformer 架构是大语言模型（LLM）的核心骨干，但其 softmax attention 存在根本性的效率瓶颈：随序列长度增加，注意力计算呈 **O(T²)** 增长，且 KV cache 的显存消耗随上下文线性增长，严重限制了长上下文推理的可扩展性。

为缓解这一问题，研究者提出了多种高效序列混合器（sequence mixer），包括**线性注意力**（linear attention，如 Lightning Attention-2、GLA、GDN）和**状态空间模型**（state-space model，如 Mamba-2、Mamba-3）。这些模型通过维护固定大小的循环状态实现 O(T) 线性时间处理，消除了 KV cache 增长问题。然而，纯线性/循环架构在长上下文召回和精确匹配任务上的效果普遍不如 Transformer，尤其是在需要「needle-in-a-haystack」式精确信息定位的场景中。

**混合注意力（Hybrid Attention）** 在两者之间寻求平衡：保留少量 full-attention 层，将其他层替换为线性注意力，实现质量与效率的更好权衡。当前业界前沿模型（如 Kimi、MiniMax-M1、Nemotron-H、Falcon-H1、Jamba）已广泛采用这种混合模式。但将预训练好的 Transformer 转换为混合模型（Transformer-to-Hybrid Conversion）时，一个关键问题浮现：**在预算约束下，应保留哪些层为 full-attention？**

### 1.2 现有方法缺陷

| 方法类型 | 代表方法 | 核心思路 | 主要缺陷 |
|---------|---------|---------|---------|
| **固定模式分配** | Uniform Interleaving | 周期性交错 full/linear attention | 忽略预训练模型的层间功能异质性；不看模型权重直接分配 |
| **搜索方法** | PostNAS | 训练超网后搜索最优子结构 | 需训练超网并评估大量子结构，选择成本 50B tokens |
| **逐层打分** | KL-LS | 逐层独立替换后计算 KL 散度排序 | 孤立评估单层重要性，忽略层间互补与冗余；成本 20B tokens |
| **逐层评估** | HALO | 逐层替换并评估性能损失后排序 | 仍需逐层操作；成本 234M tokens，仍较大 |
| **本文方法** | **FlashMorph** | 联合优化 gate 值实现全局最优选择 | 仅需 20M tokens，且考虑层间依赖 |

### 1.3 核心贡献

1. **问题形式化新视角**：首次将混合层选择形式化为**预算约束的子集优化问题**，突破固定模式分配和孤立逐层打分的局限，揭示了层间依赖关系对选择结果的根本影响
2. **FlashMorph 方法**：构建**可变形模型（morphable model）**，冻结 full-attention backbone 和 linear-attention 分支的全部权重，仅联合优化逐层 gate 值 + 线性化正则化。在大幅降低选择成本的同时，通过联合优化天然捕捉了层间的互补性与冗余性
3. **极致效率**：仅需 **20M tokens** 完成层选择（对比 HALO 的 234M、KL-LS 的 20B、PostNAS 的 50B），且选择成本随模型规模增长保持稳定
4. **全面验证**：在 Qwen3 系列（0.6B/1.7B/8B/30B-A3B）× 三种线性注意力变体（Lightning Attention-2、GLA、GDN）上验证，覆盖长上下文检索（NIAH、RULER）、常识推理（ARC、PIQA、HellaSwag、WinoGrande）和召回密集型任务（SQuAD、FDA、SWDE）
5. **开源**：完整代码已在 GitHub（github.com/LanDisen/FlashMorph）发布

### 1.4 方法对比总览

| 方法 | 选择成本 (tokens) | 非孤立选择 | 基于优化 | 大规模可扩展 |
|------|:----------------:|:---------:|:--------:|:----------:|
| Uniform | N/A | ✗ | ✗ | ✓ |
| PostNAS | 50B | △ | ✓ | ✗ |
| KL-LS | 20B | ✗ | ✓ | ✗ |
| HALO | 234M | ✗ | ✗ | △ |
| **FlashMorph（本文）** | **20M** | **✓** | **✓** | **✓** |

## Ch2 研究背景与动机

### 2.1 Full Attention 与 Linear Attention 对比

**Full (Softmax) Attention**：给定查询、键、值向量：

$$q_t = x_t W_Q, \quad k_t = x_t W_K, \quad v_t = x_t W_V$$

输出为：

$$o_t = \sum_{i=1}^{t} \alpha_{t,i} v_i W_O, \quad \alpha_{t,i} = \frac{\exp(q_t k_i^\top / \sqrt{d})}{\sum_{j=1}^{t} \exp(q_t k_j^\top / \sqrt{d})}$$

每个查询与所有先前键进行 pairwise 交互——带来了强大的精确匹配和长程依赖建模能力，但复杂度 O(T²)。

**Linear Attention**：用核函数替换 softmax，改写为 RNN 形式：

$$S_t = A_t S_{t-1} + k_t^\top v_t, \quad o_t = q_t S_t W_O$$

其中 $S_t \in \mathbb{R}^{d \times d}$ 是循环记忆状态，$A_t \in \mathbb{R}^{d \times d}$ 是状态转移/衰减矩阵。该形式实现 O(T) 线性时间处理和常数大小状态缓存（无需 KV cache）。

| 属性 | Full Attention | Linear Attention |
|------|:-------------:|:----------------:|
| 计算复杂度 | O(T²) | O(T) |
| 推理 KV cache | O(T) | O(1) |
| 精确匹配能力 | ★★★ | ★★ |
| 长程召回 | ★★★ | ★★ |
| 远程依赖建模 | ★★★ | ★★☆ |

### 2.2 Hybrid Attention 定义

对于 L 层模型，令 $I_{\text{full}} \subseteq \{1, \ldots, L\}$ 表示保留 full-attention 的层集合。第 l 层的序列混合器定义为：

$$\text{Mixer}^{(l)} = \begin{cases} \text{FullAttn}^{(l)} = A_{\text{full}}^{(l)}, & l \in I_{\text{full}} \\ \text{LinearAttn}^{(l)} = A_{\text{lin}}^{(l)}, & l \notin I_{\text{full}} \end{cases}$$

对应的混合 Transformer 块为：

$$H^{(l)} = X^{(l)} + \text{Mixer}^{(l)}(\text{LN}(X^{(l)}))$$

$$X^{(l+1)} = H^{(l)} + \text{FFN}^{(l)}(\text{LN}(H^{(l)}))$$

在预算 $K = |I_{\text{full}}|$ 约束下，核心问题转化为：**哪些层应保留 full-attention？** 保留更多 full-attention 层可保持原始 Transformer 的召回能力，但增加计算和显存成本；反之降低质量但提高效率。

### 2.3 层选择问题的形式化

理想情况下，给定 K 个 full-attention 层预算，最优选择应为：

$$I_{\text{full}}^\star = \arg\max_{\substack{I_{\text{full}} \subseteq [L] \\ |I_{\text{full}}| = K}} \text{Score}(M(I_{\text{full}}))$$

这需要从 $\binom{L}{K}$ 种组合中找出最优。例如，L=28（Qwen3-0.6B/1.7B）、K=7（3:1 混合比）时，$\binom{28}{7} \approx 1.18 \times 10^6$ 种组合；L=48（Qwen3-30B-A3B）、K=12 时，$\binom{48}{12} \approx 6.9 \times 10^{10}$ 种——穷举不可行。

### 2.4 现有方法的数学视角

**Uniform Interleaving**：$I_{\text{full}} = \{i \cdot (L/K) \mid i = 0, 1, \ldots, K-1\}$——固定间距模式，完全忽略层功能异质性。

**逐层打分方法**（KL-LS、HALO）：先计算各层单独重要性得分 $s^{(l)} = \text{Score}(M(\{l\}))$，再按得分降序选择 top-K。等价于假设 $\text{Score}(M(I_{\text{full}})) \approx \sum_{l \in I_{\text{full}}} s^{(l)}$——这是一个强假设，忽略层间的加法可分解性之外的互补与冗余效应。

FlashMorph 的核心突破在于不依赖上述假设，而是通过连续松弛直接优化全局目标。

## Ch3 FlashMorph 方法详解

FlashMorph 的核心思想是将离散子集选择问题**松弛为连续优化问题**，通过构建可变形模型（morphable model）实现联合优化。

### 3.1 可变形层构建（Morphable Layers Construction）

**第一步：蒸馏全线性学生模型**

冻结原始 Full-Attention 模型 $M_{\text{full}}$ 的全部权重，为每层训练一条线性注意力分支。训练目标为**逐层隐状态对齐损失**：

$$\mathcal{L}_{\text{hidden}} = \frac{1}{L} \sum_{l=1}^{L} \left\| H_{\text{lin}}^{(l)} - H_{\text{full}}^{(l)} \right\|_2^2$$

其中 $H_{\text{full}}^{(l)}$ 和 $H_{\text{lin}}^{(l)}$ 分别是原 full-attention 分支和线性注意力分支在第 l 层输出的隐状态。训练后得到一个全线性学生模型 $M_{\text{all-linear}}$。

该阶段配置：320M tokens，cosine LR schedule（1e-3 → 1e-5），sequence length 512。

**第二步：构建可变形模型**

将训练好的线性注意力分支（$A_{\text{lin}}^{(l)}$）与冻结的全注意力分支（$A_{\text{full}}^{(l)}$）在各层并行配置。在推理时，通过插值系数 $\alpha^{(l)}$ 混合两个分支的输出。该设计使每层可**连续变形**为 full-attention（$\alpha^{(l)} = 1$）、linear-attention（$\alpha^{(l)} = 0$）或任意中间态。

### 3.2 联合 Gate 优化（Joint Gate Optimization）

引入逐层标量门控 $\alpha^{(l)} \in [0, 1]$，在 full-attention 和 linear-attention 分支之间插值：

$$H_{\text{mix}}^{(l)} = \alpha^{(l)} H_{\text{full}}^{(l)} + (1 - \alpha^{(l)}) H_{\text{lin}}^{(l)}$$

$\alpha^{(l)}$ 越大表示越依赖 full-attention 分支，越小表示该层更安全地线性化。初始设置 $\alpha^{(l)} = 1$（对应原始全注意力模型）。

**优化过程中的关键设计**：full-attention backbone 和 linear-attention 分支**全部冻结**，仅优化 gate 值 $\alpha = \{\alpha^{(l)}\}_{l=1}^{L}$。这使可训练参数量极小（仅 L 个标量），且防止选择阶段通过调整权重来补偿架构选择的不足。

对齐损失仅在答案 token 位置计算：

$$\mathcal{L}_{\text{align}} = \frac{1}{L |\mathcal{T}(x)|} \sum_{l=1}^{L} \sum_{t \in \mathcal{T}(x)} \left\| H_{\text{mix},t}^{(l)} - H_{\text{full},t}^{(l)} \right\|_2^2$$

仅在答案 token 位置计算而非全局，使选择信号聚焦于对最终预测关键的隐状态，避免被局部依赖淹没。

### 3.3 线性化正则化（Linearization Regularization）

为鼓励模型尽可能依赖线性注意力以提高效率，引入正则项：

$$\mathcal{L}_{\text{reg}} = \sum_{l=1}^{L} \alpha^{(l)}$$

最终优化目标为：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{align}} + \lambda \mathcal{L}_{\text{reg}}, \quad \lambda = 0.1$$

- **对齐项** $\mathcal{L}_{\text{align}}$：保持原始 full-attention 教师的行为
- **正则项** $\mathcal{L}_{\text{reg}}$：惩罚对 full-attention 的依赖，推动 gate 朝向线性注意力方向
- **仅在**该变化不显著改变教师输出时生效——如果某层线性化导致隐状态大幅偏离，对齐损失的惩罚会超过正则项的收益

所有 gate 值同时优化，使 FlashMorph 能在**全局混合配置下**捕捉层间依赖、冗余和互补。

### 3.4 合成检索数据

通用语言建模目标不适合层选择——局部依赖占主导，难以识别对长上下文召回关键的层。因此，FlashMorph 在**合成长上下文检索数据**上优化 gate：

- 使用 DCLM 语料作为背景文本
- 在 1K–16K token 上下文中，在 1000 个候选深度点上随机插入 **10 个随机生成 passkey**（每 key 含 32 词）
- 模型需在序列末尾**逐一召回全部 10 个 passkey**
- 该数据**仅用于优化 gate**，不更新模型权重
- 合成数据避免了真实数据的分布偏差，提供了密集的长程检索监督

### 3.5 离散化与后续训练

优化完成后，按 gate 值降序选择 top-K 层作为 full-attention：

$$I_{\text{full}}^{\text{Hybrid}} = \text{TopK}(\{\alpha^{(l)}\}_{l=1}^{L}, K)$$

剩余层 $I_{\text{lin}}^{\text{Hybrid}} = [L] \backslash I_{\text{full}}^{\text{Hybrid}}$ 替换为线性注意力（使用已训练的 $A_{\text{lin}}^{(l)}$）。离散化后**丢弃 gate 值**——混合架构在推理时无任何额外开销。

后续训练包括两个阶段：

1. **Logits 蒸馏**（1B tokens，cosine LR 1e-4 → 1e-5，seq_len=512）：

$$L_{\text{KD}} = D_{\text{KL}}(p_T(X) \| p_H(X))$$

2. **长上下文微调**（1B tokens，constant LR 1e-5，seq_len=16K）：

$$L_{\text{FT}} = -\sum \log p_H(x_t | x_{<t})$$

### 3.6 完整算法伪代码

```
Algorithm FlashMorph Transformer-to-Hybrid Conversion
Require: Full-attention model M_full (L layers); training data D;
        synthetic retrieval data D_syn; selection steps S;
        full-attention budget K; regularization weight λ

 1: Distill all-linear model M_all-linear from M_full on D
    via hidden-state alignment (Eq. 8)
 2: Construct morphable model: pair A_full^(l) ↔ A_lin^(l)
 3: Freeze both full-attention backbone and linear-attention branches
 4: Initialize layerwise gates α^(l) ← 1 for all l = 1..L
 5: for s = 1..S do
 6:   Sample synthetic retrieval examples x ~ D_syn
 7:   Compute mixed states: H_mix^(l) = α^(l)H_full^(l) + (1-α^(l))H_lin^(l)
 8:   Compute answer-token alignment loss L_align (Eq. 10)
 9:   Compute linearization regularizer L_reg = Σ_l α^(l) (Eq. 11)
10:   Update gates α by minimizing L_total = L_align + λL_reg
11: end for
12: Select full-attention layers: I_full = TopK({α^(l)}, K)
13: Set linear-attention layers: I_lin = [L] \ I_full
14: Instantiate M_hybrid (full on I_full, linear on I_lin)
15: Apply logits distillation (Eq. 14) and long-context FT (Eq. 15)
16: return M_hybrid
```

## Ch4 实验结果与分析

### 4.1 实验设置

**骨干模型**：Qwen3-0.6B（28 层）、Qwen3-1.7B（28 层）、Qwen3-8B（36 层）、Qwen3-30B-A3B MoE（48 层）。

**线性注意力变体**：Lightning Attention-2（无需 softmax 核，基于指数衰减）、GLA（Gated Linear Attention，可学习门控）、GDN（Gated DeltaNet，delta 规则更新）。所有注意力实现基于 flash-linear-attention 库。

**训练配置**：四阶段流程——1) 隐状态对齐（320M tokens），2) 层选择（20M tokens），3) Logits 蒸馏（1B tokens），4) 长上下文微调（1B tokens）。默认混合比 3:1（linear:full=21:7），完整配置详见附录 A Table 5。

**评估任务**：
- **长上下文检索**：Needle-in-a-Haystack（NIAH-Single-1/2/3，32K–256K）、RULER
- **常识推理**：ARC-Easy、ARC-Challenge、PIQA、HellaSwag、WinoGrande
- **召回密集型**：SQuAD、FDA、SWDE

### 4.2 Needle-in-a-Haystack (NIAH) 检索结果

**0.6B 骨干 × Lightning Attention—NIAH**：

| 方法 | 选择成本 | NIAH-Single-1 (32K/64K/128K/256K) | NIAH-Single-2 (32K/64K/128K/256K) | NIAH-Single-3 (32K/64K/128K/256K) |
|------|:-------:|:--------------------------------:|:--------------------------------:|:--------------------------------:|
| Qwen3 原版 | — | 100/100/0/0 | 100/99.0/0/0 | 99.8/82.8/0/0 |
| Qwen3+YaRN | — | 30.2/29.2/20.2/31.0 | 0.6/0/0/0 | 0.8/0/0/0 |
| Uniform | N/A | 99.4/98.6/97.8/98.0 | 68.4/49.2/41.2/12.8 | 57.6/36.0/28.8/12.2 |
| KL-LS | 20B | 98.2/97.6/96.4/94.8 | 69.4/60.6/50.2/24.0 | 32.2/17.2/8.6/3.6 |
| HALO | 234M | 99.6/99.8/99.6/99.2 | 86.0/80.6/62.4/21.0 | 68.6/67.4/57.2/32.4 |
| **FlashMorph** | **20M** | **99.0/99.0/99.0/99.2** | **92.2/82.4/45.8/11.0** | **81.2/73.6/45.6/28.4** |

**1.7B 骨干 × Lightning Attention—NIAH**：

| 方法 | 选择成本 | NIAH-Single-1 (32K/64K/128K/256K) | NIAH-Single-2 (32K/64K/128K/256K) | NIAH-Single-3 (32K/64K/128K/256K) |
|------|:-------:|:--------------------------------:|:--------------------------------:|:--------------------------------:|
| Qwen3 原版 | — | 100/100/0/0 | 100/98.8/0/0 | 99.8/96.4/0/0 |
| Uniform | N/A | 99.8/99.6/99.6/100 | 71.8/86.8/28.4/19.2 | 58.6/59.4/16.4/27.8 |
| PostNAS | 50B | 99.2/99.4/99.8/99.2 | 96.8/95.6/78.0/73.8 | 56.4/51.2/57.2/57.6 |
| KL-LS | 20B | 98.6/98.6/98.0/94.4 | 62.2/68.6/47.4/34.6 | 22.8/4.0/9.4/3.8 |
| HALO | 234M | 99.8/100/100/100 | 99.6/98.6/95.0/95.2 | 86.4/90.8/67.4/52.8 |
| **FlashMorph** | **20M** | **100/100/100/100** | **99.6/100/98.2/88.2** | **96.6/95.4/94.4/73.2** |

FlashMorph 在 1.7B 上以仅 20M tokens 的选择成本（HALO 的 1/11.7、PostNAS 的 1/2500）实现了 NIAH-Single-1 **全长度完美召回**（100%），且在更难的 NIAH-Single-3 上达到 96.6%（32K）/95.4%（64K）/94.4%（128K）/73.2%（256K），大幅超越 HALO。

**8B 和 30B-A3B 骨干 NIAH**（HypeNet 设置）：

| 模型 | 方法 | NIAH-Single-1 (32K/64K/128K/256K) | NIAH-Single-2 (32K/64K/128K/256K) | NIAH-Single-3 (32K/64K/128K/256K) |
|------|------|:--------------------------------:|:--------------------------------:|:--------------------------------:|
| **8B** | Uniform | 91.8/92.4/93.2/92.6 | 91.4/60.6/31.6/17.8 | 58.2/54.2/40.6/26.2 |
| | HALO | 99.2/99.4/99.2/98.4 | 96.8/95.6/89.2/68.4 | 89.8/85.2/75.0/50.8 |
| | **FlashMorph** | **99.8/99.6/99.6/99.2** | **98.0/99.4/85.6/82.6** | **99.4/98.2/92.4/94.0** |
| **30B-A3B** | Uniform | 98.4/99.4/99.0/99.6 | 73.6/53.4/35.4/19.0 | 55.2/36.0/18.4/9.6 |
| | HALO | 97.6/98.6/99.4/99.0 | 94.8/79.4/61.2/27.6 | 47.0/17.8/12.0/0.8 |
| | **FlashMorph** | **98.6/96.2/94.0/91.4** | **70.2/53.6/21.4/9.6** | **32.6/38.2/24.6/18.2** |

在 8B 上，FlashMorph 的 NIAH-Single-3 表现远超所有基线（99.4%/98.2%/92.4%/94.0% vs HALO 的 89.8%/85.2%/75.0%/50.8%）。在 30B-A3B 上 FlashMorph 略弱于其他方法，作者归因于 MoE 架构中层间依赖更为复杂。

### 4.3 常识推理与召回任务

**0.6B 骨干 × 三种线性注意力变体—常识推理**：

| 注意力变体 | 方法 | ARC-e | ARC-c | PIQA | Hella. | Wino. | Avg. |
|:---------:|:-----:|:----:|:----:|:----:|:-----:|:----:|:----:|
| Lightning | Uniform | 62.3 | 32.9 | 66.9 | 46.1 | 55.6 | 52.8 |
| | KL-LS | 62.1 | 33.0 | 67.5 | 46.2 | 54.9 | 52.7 |
| | HALO | 63.5 | 32.4 | 67.4 | 46.4 | 56.1 | 53.2 |
| | **FlashMorph** | **62.8** | **32.0** | **67.3** | **46.4** | **55.2** | **52.7** |
| GLA | Uniform | 63.1 | 33.1 | 67.3 | 46.8 | 55.3 | 53.1 |
| | KL-LS | 61.8 | 33.8 | 67.2 | 46.9 | 54.9 | 52.9 |
| | HALO | 62.9 | 32.6 | 67.4 | 46.9 | 55.1 | 53.0 |
| | **FlashMorph** | **63.9** | **32.9** | **67.5** | **46.9** | **54.1** | **53.1** |
| GDN | Uniform | 59.6 | 32.3 | 67.1 | 47.6 | 55.3 | 52.4 |
| | KL-LS | 61.1 | 35.3 | 67.9 | 47.4 | 56.7 | 53.7 |
| | HALO | 60.1 | 34.1 | 67.7 | 47.5 | 55.3 | 53.0 |
| | **FlashMorph** | **63.1** | **33.5** | **67.8** | **47.5** | **56.4** | **53.7** |

**0.6B 骨干 × 三种线性注意力变体—召回任务**：

| 注意力变体 | 方法 | SQuAD | FDA | SWDE | Recall Avg. |
|:---------:|:-----:|:----:|:---:|:----:|:----------:|
| Lightning | Uniform | 30.3 | 51.3 | 71.8 | 51.1 |
| | KL-LS | 29.3 | 58.5 | 70.5 | 52.8 |
| | HALO | 34.7 | 60.4 | 72.6 | 55.9 |
| | **FlashMorph** | **41.7** | **62.4** | **76.2** | **60.1** |
| GLA | Uniform | 31.1 | 55.6 | 75.0 | 53.9 |
| | KL-LS | 32.7 | 61.4 | 75.2 | 56.4 |
| | HALO | 36.4 | 68.5 | 75.4 | 60.1 |
| | **FlashMorph** | **35.4** | **70.7** | **76.0** | **60.7** |
| GDN | Uniform | 30.3 | 55.5 | 72.2 | 52.7 |
| | KL-LS | 33.5 | 72.8 | 75.6 | 60.6 |
| | HALO | 26.5 | 62.0 | 71.2 | 53.2 |
| | **FlashMorph** | **38.4** | **71.3** | **76.7** | **62.1** |

FlashMorph 在常识推理上保持与最强基线**持平或略优**（差距 ≤0.6 分），但在召回密集型任务上持续领先，在 GDN 变体上 Recall Avg. 达 62.1（vs KL-LS 60.6、HALO 53.2）。

### 4.4 推理效率分析

基于 Qwen3-1.7B 单 GPU（batch size=1）的推理效率测试：

**Prefill**（输入从 4K 到 1M tokens）：

| 上下文长度 | Qwen3 原版延迟 | FlashMorph 延迟 | 加速比 | Qwen3 显存 | FlashMorph 显存 |
|:--------:|:-------------:|:--------------:|:-----:|:---------:|:-------------:|
| 4K | baseline | ≈baseline | ∼1× | baseline | ↓ |
| 32K | — | — | — | — | — |
| 64K | — | — | — | — | — |
| 128K | baseline | — | **2.24×** | baseline | ↓↓ |
| 256K | baseline | — | **2.81×** | baseline | ↓↓ |
| 512K | OOM | ✅ 可运行 | — | OOM | ✅ 可用 |

**Decode**（prefill 固定 1K，decode 从 4K 到 1M tokens）：

| 上下文长度 | 加速比 |
|:--------:|:-----:|
| 256K | **1.56×** |
| 512K | **2.07×** |
| 1M | Qwen3 OOM，FlashMorph ✅ 可运行 |

FlashMorph 的混合模型可处理 **512K prefill + 1M decode**，而原版 Qwen3-1.7B 在这些长度下显存溢出（OOM）。

### 4.5 层选择效率对比

| 方法 | 选择成本 (tokens) | FLOPs | GPU 小时 | 相比 FlashMorph 倍数 |
|------|:----------------:|:-----:|:--------:|:-----------------:|
| PostNAS | 50B | 8.0×10²⁰ | 2561.3 | **1219.7×** |
| KL-LS | 20B | 2.5×10²⁰ | 1071.8 | **510.4×** |
| HALO | 234M | 6.5×10¹⁷ | 15.4 | **7.3×** |
| **FlashMorph** | **20M** | **2.5×10¹⁷** | **2.1** | **1×** |

FLOPs 对比：FlashMorph（2.5×10¹⁷）vs HALO（6.5×10¹⁷）— 仅 **38.5%** 的 HALO 计算量，且选择质量更高。与 KL-LS 的差距达 **1000×**。

### 4.6 消融与分析

**监督信号对比**（Qwen3-0.6B，RULER 评估）：

| 方法 | GLA (RULER) | GDN (RULER) |
|------|:----------:|:----------:|
| KL-LS | baseline | baseline |
| HALO | baseline | baseline |
| FlashMorph w/ LM supervision | 57.2 | 61.6 |
| FlashMorph w/ synthetic retrieval | **59.0** | **64.7** |

即使仅使用 LM 监督（无专门检索数据），FlashMorph 的联合优化已优于 KL-LS 和 HALO。合成检索数据进一步提升 +1.8（GLA）和 +3.1（GDN）分。

**混合比鲁棒性**（Qwen3-1.7B，RULER 评估）：

| 混合比 (linear:full) | full-attention 占比 | FlashMorph 表现 |
|:------------------:|:---------------:|:--------------:|
| 6:1 | 14.3% | 领先所有基线，优势最大 |
| 3:1 | 25.0% | 最佳或持平最佳 |
| 1:1 | 50.0% | 接近 all-full 上界 |

在稀疏 full-attention 分配（6:1，仅保留 4/28 层）下，FlashMorph 的优势最为显著——说明其能在极有限预算下更精准地定位关键层。

## Ch5 局限性与未来方向

### 5.1 局限性

1. **合成数据依赖性**：层选择阶段依赖合成的 passkey 检索数据。虽然 LM 监督也有效（RULER 57.2–61.6），但合成数据效果更好（59.0–64.7）。对超出合成数据分布的检索模式（如跨文档推理、多跳推理）的适应性有待验证
2. **大规模评估有限**：在 8B 和 30B-A3B 规模上仅与 Uniform 和 HALO 对比（PostNAS/KL-LS 在此规模复制成本过高），且仅在 HypeNet 设置下评估。MoE 架构（30B-A3B）下 FlashMorph 的优势不显著，可能需针对 MoE 调整方法
3. **线性注意力变体覆盖**：当前验证了 Lightning Attention-2、GLA、GDN 三种变体，对更新的线性注意力架构（如 Mamba-3、HGRN2）和状态空间模型的兼容性未测试
4. **蒸馏阶段仍有 Token 开销**：虽然层选择仅需 20M tokens（0.02B），但后续蒸馏（1B tokens）和长上下文微调（1B tokens）需要总约 2.02B tokens 的迁移成本
5. **1.7B 以上无法完全复现基线**：在 8B/30B-A3B 上因 PostNAS/KL-LS 的计算成本过高而无法复制其所有结果，仅维持与 HALO 和 Uniform 的对比

### 5.2 未来方向

1. **端到端联合训练**：将层选择融入预训练过程，而非仅在 conversion 阶段使用——在预训练早期即可确定最优混合配置
2. **动态混合比**：基于输入长度或任务复杂度动态调整 full-attention 预算，而非固定 3:1 比例
3. **更多架构验证**：扩展到 Llama 4、DeepSeek-V4、Gemma 3 等其他系列，验证方法的跨架构迁移性
4. **多样化监督信号**：使用真实长上下文任务（如多跳 QA、文档摘要、长文本分类）替代合成检索数据，提高选择与下游任务的相关性
5. **MoE 适配**：研究 MoE 架构下 expert routing 与 attention layer selection 的耦合关系，设计针对 MoE 的专用层选择策略

### 5.3 总结

FlashMorph 将 Transformer-to-hybrid conversion 中的层选择从启发式孤立打分升级为**基于联合优化的全局最优选择**，在仅需 20M tokens（2.1 GPU hours）选择成本下，实现了 NIAH 全长度完美召回（100%）、1.56–2.81× 推理加速，且选择成本仅为 HALO 的 1/11.7、KL-LS 的 1/1000、PostNAS 的 1/2500。这项工作为将现有 Transformer 高效转化为混合注意力模型提供了实用且可扩展的方案，代码已开源于 github.com/LanDisen/FlashMorph。

## 附录A：FlashMorph 完整超参数配置

| 阶段 | 超参数 | 0.6B | 1.7B | 8B | 30B-A3B |
|------|--------|:----:|:----:|:--:|:-------:|
| **架构** | #layers | 28 | 28 | 36 | 48 |
| | hidden size | 1024 | 2048 | 4096 | 2048 |
| | FFN width | 3072 | 6144 | 12288 | 6144 |
| | #full-attention layers | 7 | 7 | 9 | 12 |
| | #linear-attention layers | 21 | 21 | 27 | 36 |
| | head dimension | 128 | 128 | 128 | 128 |
| | #attention heads | 16 | 16 | 32 | 32 |
| | #full-attention KV heads | 8 | 8 | 8 | 4 |
| | #linear-attention KV heads | 16 | 16 | 32 | 32 |
| **Stage 1** (对齐) | tokens | 320M | 320M | 320M | 320M |
| | learning rate | 1e-3→1e-5 | 1e-3→1e-5 | 1e-3→1e-5 | 1e-3→1e-5 |
| | seq_len | 512 | 512 | 512 | 512 |
| **Stage 2** (选择) | tokens | 20M | 20M | 20M | 20M |
| | learning rate | 2e-2→2e-3 | 2e-2→2e-3 | 2e-2→2e-3 | 2e-2→2e-3 |
| | seq_len | <16K | <16K | <16K | <16K |
| **Stage 3** (蒸馏) | tokens | 1B | 1B | 1B | 1B |
| | learning rate | 1e-4→1e-5 | 1e-4→1e-5 | 1e-4→1e-5 | 1e-4→1e-5 |
| | seq_len | 512 | 512 | 512 | 512 |
| **Stage 4** (微调) | tokens | 1B | 1B | 1B | 1B |
| | learning rate | 1e-5 | 1e-5 | 1e-5 | 1e-5 |
| | seq_len | 16K | 16K | 16K | 16K |

## 附录B：合成检索数据示例

```
query <|im_start|> This is a very long story book: <book> ...
background DCLM text ...
Remember this sequence of words, it is the first passkey to the vault:
[Passkey 1] xray whiskey lima charlie papa hotel mike india quebec ...
more DCLM text with remaining passkeys inserted at different depths ...
</book>.
response Based on the content of the book, what is the first passkey to the vault?
[Answer 1] Passkey: xray whiskey lima charlie papa hotel mike india quebec ...
```

## 附录C：相关工作对比

FlashMorph 与现有方法的根本区别在于其**联合优化**框架。现有方法（KL-LS、HALO）的逐层打分本质上假设各层重要性可加性分解，但实际中层的功能存在高度耦合——例如，前几层（token 嵌入到语义抽象）和后几层（输出生成）的功能不同，中间层可能有冗余。FlashMorph 通过联合优化 gate 值，自动发现这种结构化模式：实验发现 FlashMorph 选择的 top 层往往集中在中间层（如 Qwen3-0.6B + Lightning 下，层 1、16、21 为 top-3），这与「中间层对上下文信息整合最关键的」直觉一致。
