# Scaling Laws for Neural Language Models

## 论文元数据
- 标题：Scaling Laws for Neural Language Models
- 作者：Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
- 机构：Jared Kaplan（Johns Hopkins University & OpenAI）；其余作者均来自 OpenAI
- arXiv ID：2001.08361
- 提交日期：2020-01-23（v1）
- 官方代码：无官方实现；社区复现：github.com/shehper/scaling_laws

---

## Ch1: 论文概述与核心发现

### 1.1 背景

2020年，OpenAI 发表了一篇题为"Scaling Laws for Neural Language Models"的论文，系统性地研究了语言模型性能与模型规模、数据量和计算量之间的定量关系。这是首次在多个数量级的范围内，通过大量受控实验验证了语言模型损失与这三个维度之间的幂律关系（Cmin 跨越 8 个数量级，N 跨越 6 个数量级，D 跨越逾 2 个数量级）。

### 1.2 六大核心发现

| # | 发现 | 一句话表述 |
|---|------|-----------|
| 1 | 规模决定性能，形状几乎无关 | 总非嵌入参数固定时，深度/宽度/头数大范围变化对损失影响极小（< 3%）|
| 2 | 幂律缩放关系 | 损失与 N、D、C 均呈幂律关系，各维度跨越多个数量级 |
| 3 | 过拟合可预测 | 过拟合惩罚由 N^0.74/D 比值决定，可提前估算数据需求 |
| 4 | 大模型样本效率更高 | 达到相同损失，大模型需要的训练步数和数据量均远少于小模型 |
| 5 | 最优计算分配为"大模型+适量数据" | 给定计算预算，绝大部分增长应投向模型规模而非数据量 |
| 6 | 跨数据集泛化 | 在训练分布之外测试时，损失与训练集验证损失强相关，仅有固定偏移量，且与训练时长无关 |

### 1.3 历史重要性

这篇论文是**现代LLM扩展理论的基础**。GPT-3 (175B) 的规模和训练配置直接基于其预测。后续的 Chinchilla (2022) 虽然修正了数据-参数的最优比例（将 D ∝ C^0.27 修正为 D ∝ C^0.5，即等比例增长），但不降低 Kaplan et al. 作为首篇系统性 scaling law 论文的开创性地位。该论文建立了"performance is determined by scale, not shape"的共识，引导了后续几乎所有大规模语言模型的扩展方向。

---

## Ch2: 实验设定与测量方法

### 2.1 模型架构

所有模型使用标准自回归 Transformer（decoder-only，GPT-2 风格）：

- **架构配置**：Pre-LayerNorm + 残差连接，GELU 激活函数
- **位置编码**：可学习位置编码（非 Sinusoidal）
- **词汇表**：~50,257 BPE tokens（GPT-2 tokenizer，nvocab = 50257）
- **优化器**：Adam（小模型）/ Adafactor（1B 参数以上模型，因显存限制）
- **学习率**：3000 步线性 warmup + cosine decay；学习率按模型大小调整：LR(N) ≈ 0.003239 − 0.0001395 log(N)

### 2.2 模型规模范围

| 维度 | 范围 | 备注 |
|------|------|------|
| 非嵌入参数 N | 768 → 1.5B | n_params ≈ 12 × n_layers × d_model² |
| 层数 n_layers | 2 → 207 | 最深配置 (207, 768) |
| 宽度 d_model | 128 → 4288 | 最宽配置 (6, 4288) |
| 注意力头数 n_heads | 随 d_model 同步变化 | |
| FFN 中间维度 d_ff | 与 d_model 成比例（~4×） | |

**关键设计选择**：论文将嵌入参数（word_embed + pos_embed）从 N 中排除，因为这能产生更干净的 scaling 趋势（见 Figure 6）。

### 2.3 数据集

使用 **WebText2**（WebText 数据集的扩展版本），来源为 Reddit 外链（2017年12月前及2018年1月至10月，至少 3 karma）：

- 总文档数：20.3M
- 总大小：96 GB 纯文本，约 1.62×10^10 词（wc 统计）
- 总 token 数：**2.29×10^10 tokens**（经 BPE tokenizer 处理后）
- 测试集：保留 6.6×10^8 tokens；另在 Books Corpus、Common Crawl、英文 Wikipedia、Internet Books 上测试泛化
- 数据子采样范围：22M → 22.9B tokens（对应完整数据集）

### 2.4 训练细节

| 参数 | 配置 |
|------|------|
| 上下文长度 | 1024 tokens（特殊实验除外） |
| 批大小（标准） | 512 sequences × 1024 tokens = **2^19 ≈ 524,288 tokens** |
| 训练步数 | 固定 2.5×10^5 步（默认设置） |
| 学习率调度 | 3000步线性 warmup + cosine decay to ~0 |
| 权重正则化 | 10% dropout（过拟合实验中使用）|
| 计算量估算 | C ≈ 6N（FLOPs/token），即每训练 token 约 6N 次浮点运算 |

### 2.5 测量方法论

论文的方法论创新在于：

1. **记录完整训练曲线**而非仅最终损失——从所有模型的训练动态中提取统一规律
2. **跨规模对比**：对每个数据量 D 训练多个不同大小的模型 N，在二维平面中定位幂律方程
3. **计算量估算**：C ≈ 6NBS（非嵌入部分 FLOPs），1 PF-day = 10^15 × 24 × 3600 ≈ 8.64×10^19 FLOPs
4. **引入 Cmin**：对训练批大小进行修正，将实际使用批大小换算到"临界批大小以下"的等效计算量，得到更干净的 scaling 趋势

### 2.6 核心符号对照

| 符号 | 含义 | 量纲/范围 |
|------|------|-----------|
| N | 非嵌入参数量 | 768 → 1.5×10^9 |
| D | 训练 token 数 | 2.2×10^7 → 2.29×10^10 |
| C | 总计算量（FLOPs，固定批大小训练）| ~ 10^18 → 10^23 |
| Cmin | 最优批大小下的等效计算量 | 跨越 8 个数量级 |
| S | 优化步数 | |
| Smin | 等效到 B << Bcrit 时的最小步数 | |
| B | 批大小（tokens） | B ≈ D / S |
| L | 交叉熵损失（nats） | |
| αN | 参数幂律指数 | 0.076 |
| αD | 数据幂律指数 | 0.095（单独拟合 L(D)）|
| αS | 步数幂律指数 | 0.76 |
| αCmin | 最优计算幂律指数 | 0.050 |

---

## Ch3: 三大 Scaling Law 方程

### 3.1 参数量 Scaling Law（无限数据假设）

当训练数据充分多使模型接近收敛时，损失仅取决于参数量：

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076, \; N_c \approx 8.8 \times 10^{13}
$$

- $\alpha_N = 0.076$：每10倍参数增长，损失下降约 $1 - 10^{-0.076} \approx$ 0.077 nats（等价地，参数翻倍使损失减少因子 $2^{-0.076} \approx 0.95$）
- $N_c = 8.8 \times 10^{13}$：达到 1 nats 损失所需的参数量（外推值）
- 实验范围：768 → 1.5B 参数（约6个数量级）

### 3.2 数据量 Scaling Law（有限数据、早停）

当模型大小固定、数据量不足时（通过早停控制过拟合）：

$$
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095, \; D_c \approx 5.4 \times 10^{13}
$$

- $\alpha_D > \alpha_N$：数据受限的损失上升比参数受限更快
- 跨越范围：22M → 22.9B tokens（逾2个数量级）

### 3.3 计算量 Scaling Law（最优分配）

当在给定计算预算下最优地分配 N 和 D 时：

$$
L(C_{\min}) = \left(\frac{C_{\min}^c}{C_{\min}}\right)^{\alpha_C^{\min}}, \quad \alpha_C^{\min} \approx 0.050, \; C_{\min}^c \approx 3.1 \times 10^8 \text{ PF-days}
$$

- $\alpha_C^{\min} = 0.050$：三个指数中最小的——计算量的收益递减最快
- 跨越范围：Cmin 跨越约 8 个数量级

### 3.4 二维联合方程

**参数+数据联合**（L(N,D) 方程，Equation 1.5）：

$$
L(N, D) = \left[ \left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D} \right]^{\alpha_D}
$$

注意：此处的 $\alpha_N$、$\alpha_D$、$N_c$、$D_c$ 来自对 L(N,D) 的**联合拟合**（Table 2），拟合值为：

| 参数 | 联合拟合值（Table 2）|
|------|----------------------|
| αN | 0.076 |
| αD | **0.103**（与单独拟合 L(D) 的 0.095 略有不同）|
| Nc | 6.4×10^13 |
| Dc | **1.8×10^13**（与单独拟合 L(D) 的 5.4×10^13 不同）|

物理含义：参数和数据"耦合"——增加参数不独立于增加数据；当 D→∞ 时退化为 L(N)，当 N→∞ 时退化为 L(D)。

**参数+步数联合**（无限数据下的训练动态，Equation 1.6）：

$$
L(N, S) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{\min}}\right)^{\alpha_S}, \quad \alpha_S \approx 0.76, \; S_c \approx 2.1 \times 10^3
$$

- $S_{\min}$ 是训练时若使用 B << Bcrit 所对应的等效最小步数
- $\alpha_S = 0.76$：训练步数的收益递减比参数快得多

### 3.5 临界批大小

$$
B_{\text{crit}}(L) = \frac{B_*}{L^{1/\alpha_B}}, \quad B_* \approx 2.1 \times 10^8 \text{ tokens}, \; \alpha_B \approx 0.21
$$

- 损失越低（模型越好），临界批大小越大
- 最大模型在收敛时的临界批大小约为 1–2M tokens
- 临界批大小仅取决于损失值，**不直接取决于模型大小**

---

## Ch4: 架构形状无关性与过拟合

### 4.1 形状无关性

论文最反直觉的发现之一：**固定非嵌入参数 N 时，改变架构形状几乎不影响性能**。

**实验方法**：固定 N，系统性地单独改变 nlayer、dmodel、nheads、dff

**关键数据（Figure 5）**：
- **Aspect ratio**（dmodel/nlayer）变化 **40 倍**：(nlayer, dmodel) = (6, 4288) 与 (48, 1600) 相比，损失差异 < 3%
- **Feed-forward ratio**（dff/dmodel）和 **Attention head dimension**（dmodel/nhead）大范围变化，损失变化也均在几个百分点以内
- 仅当 nlayer < 2 或极端深度/宽度比时才显著偏离

**意义**：性能由**参数总量**决定，而非具体网络结构。

### 4.2 LSTM vs Transformer（与其他架构的对比）

- **LSTM**：在上下文早期 token（短程依赖）与 Transformer 表现相近，但随 token 位置增加，Transformer 持续改善而 LSTM 趋于饱和（Figure 7）
- **Universal Transformer（Recurrent Transformer）**：通过参数复用，每非嵌入参数略好于标准 Transformer；但由于每次前向传播重复使用参数，每 FLOP 的效率略低（Figure 17）
- **核心结论**：幂律趋势在不同架构间一致，但 Transformer 在利用长上下文方面具有明显优势

### 4.3 过拟合行为

过拟合惩罚量由以下比值决定（Equation 4.3）：

$$
\delta L \propto \left(\frac{N^{\alpha_N/\alpha_D}}{D}\right)^{\alpha_D} \approx \left(\frac{N^{0.74}}{D}\right)^{0.103}
$$

**关键洞察**：
- 指数 0.74 < 1.0：参数增长比数据增长**更慢地**触发过拟合
- 这意味着：**参数翻8倍，数据只需翻约5倍**（$8^{0.74} \approx 4.9$）

要保持过拟合损失惩罚 < 0.02 nats 的条件：

$$
D \gtrsim (5 \times 10^3) \cdot N^{0.74}
$$

| 模型大小 N | 所需最小数据量 D（Kaplan 原文计算）|
|-----------|-----------------------------------|
| 1B | ~5×10^9 tokens |
| 10B | ~2.8×10^10 tokens |
| 100B | ~1.6×10^11 tokens |

（注：Chinchilla 2022 将最优数据-参数比修正为等比例增长 D ∝ N，意味着实际数据需求比 Kaplan 预测更大。）

---

## Ch5: 训练曲线与样本效率

### 5.1 统一训练动态

所有模型的训练曲线遵循统一的幂律形式（Equation 1.6）：

$$
L(N, S_{\min}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{\min}}\right)^{\alpha_S}
$$

- 第一项为该模型在无限数据、无限训练步数时的收敛极限损失 $L(N,\infty)$
- 第二项描述随训练步数增加的收益递减，指数 $\alpha_S \approx 0.76$ 与模型大小无关
- 不同 N 的训练曲线参数独立于模型大小——训练动态是**scale-invariant**的

### 5.2 大模型的样本效率

大模型不仅最终性能更好，而且训练效率更高（Figure 19）：

- 达到相同的损失值，大模型需要的**步数**（Smin）和**数据量**（Emin）均远少于小模型
- Figure 19 显示：从最小模型到较大模型，sample efficiency 可改善近 **100 倍**
- 这种差异在训练初期就显现，随训练进行持续维持

### 5.3 最优训练并未收敛（Early Stopping 作为最优策略）

由 Appendix B.2 的理论推导可得（Equation B.5）：

$$
L(N_{\text{eff}}, C) = \left(1 + \frac{\alpha_N}{\alpha_S}\right) \cdot L(N_{\text{eff}}, \infty) \approx 1.10 \times L(N_{\text{eff}}, \infty)
$$

即：**compute-efficient 训练应停在高于收敛损失约 $\alpha_N/\alpha_S \approx 10\%$ 处**。早停并非妥协，而是理论最优。

相比之下，若将模型训练到接近收敛（偏差 f' ≈ 2%），则 compute-efficient 策略：
- 使用 2.7× 更多参数
- 减少 7.7× 训练步数
- **节省 65% 的总计算量**

---

## Ch6: 最优计算分配

### 6.1 核心问题

给定总计算预算 $C_{\min}$，如何分配以最小化最终损失？即选择最优的 $N$（模型大小）和 $D$（数据量），使得 $C_{\min} = 6NB_{\text{crit}}S_{\min}$ 约束下 $L(N, S_{\min})$ 最小。

### 6.2 三个最优关系

$$
N_{\text{opt}} \propto C_{\min}^{0.73}, \quad B_{\text{opt}} \propto C_{\min}^{0.24}, \quad S_{\min} \propto C_{\min}^{0.03}
$$

| 维度 | 指数 | 计算翻10亿倍时 | 解读 |
|------|:---:|:---:|------|
| 模型大小 N | 0.73 | ×10^(0.73×9) ≈ 3×10^6 | 绝大多数预算投向模型规模 |
| 批大小 B | 0.24 | ×10^(0.24×9) ≈ 3×10^2 | 适度增长 |
| 串行步数 Smin | 0.03 | ×10^(0.03×9) ≈ 2 | 几乎不增长 |

**对实际训练的指导**：计算预算翻倍时，绝大部分增量应投向更大的模型，而非更长的训练或更多数据。这直接指导了 GPT-3 175B 的规模选择。

### 6.3 Scaling Law 的极限：矛盾与推断

原文 Section 6.3 指出，compute-efficient 训练中数据需求增长较慢（$D \propto C_{\min}^{0.27}$），而避免过拟合需要数据增长更快（$D \propto C_{\min}^{0.54}$）。两者在以下临界点相交：

$$
C^* \sim 10^4 \text{ PF-days}, \quad N^* \sim 10^{12} \text{ params}, \quad D^* \sim 10^{12} \text{ tokens}, \quad L^* \sim 1.7 \text{ nats/token}
$$

作者推测：这可能是 Transformer 语言模型性能的理论上限——在此规模附近，模型可能已从自然语言数据中提取了几乎全部可靠信息，L* 可近似视为自然语言的熵下界估计。作者同时指出，该临界点的数值对幂律拟合参数高度敏感，不确定性达一个数量级。

### 6.4 与 Chinchilla 的核心分歧

| 维度 | Kaplan (2020) | Chinchilla (2022) |
|------|:---:|:---:|
| $N_{\text{opt}} \propto C$ 指数 | 0.73 | 0.50 |
| $D_{\text{opt}} \propto C$ 指数 | 0.27 | 0.50 |
| 对GPT-3的评估 | 模型偏小（数据偏多） | 数据严重不足（需~10×更多）|
| 预测的175B模型最优数据量 | ~0.3T tokens | ~3.3T tokens |

分歧的3个技术原因（Porian et al. 2024）：
1. **最后一层计算成本**：Kaplan 未将 embedding/unembedding 矩阵计算成本计入 C，低估了模型实际计算开销
2. **warmup 时长**：不同规模模型使用相同 warmup 步数，导致小模型相对训练不足
3. **scale-dependent 优化器调参**：学习率和权重衰减未跨规模独立优化

---

## Ch7: 最优批大小

### 7.1 临界批大小的定义

源于 McCandlish et al. (2018) [arXiv:1812.06162]：

- $B < B_{\text{crit}}$：计算高效但时间低效（需要更多串行更新步数）
- $B > B_{\text{crit}}$：时间高效但计算低效（每个额外 token 的信息增益递减）
- 梯度噪声比（gradient noise scale）决定临界批大小，可直接测量

### 7.2 临界批大小与损失的关系（Figure 10）

$$
B_{\text{crit}}(L) = \frac{B_*}{L^{1/\alpha_B}}, \quad B_* \approx 2.1 \times 10^8 \text{ tokens}, \; \alpha_B \approx 0.21
$$

- 损失越低（模型越好），临界批大小越大
- $B_{\text{crit}}$ **仅取决于损失值，不直接依赖模型大小**
- 最大模型在收敛时 $B_{\text{crit}} \approx 1\text{–}2\text{M}$ tokens

### 7.3 实际训练建议

在 $B \approx B_{\text{crit}}$ 处训练，时间效率与计算效率的折中最优（约需 $2S_{\min}$ 步，处理 $2E_{\min}$ 个样本）。实际操作中，$B_{\text{crit}}$ 随训练进行而增大（因为损失在下降），属于移动目标。

---

## Ch8: 上下文位置的 Scaling（补充发现）

论文在 Appendix D.5 中报告了上下文长度与损失的关系（Figure 20、21）：

- 对于上下文中位置为 T 的 token，其损失随 T 近似呈**幂律下降**（图 20），形式约为 $L(T) \approx a + b \cdot T^{-\beta}$（各模型的 $\beta$ 在 0.47–0.62 之间）
- 更大的模型在**早期 token**上改善更快，在**后期 token**上达到更低的损失（Figure 21）
- 短上下文（nctx = 8）训练的模型在早期 token 上可优于大 nctx 模型，说明 context 分配与模型大小存在交互
- 这些趋势可能与自然语言中的长程幂律相关性有关

---

## Ch9: 历史影响与延伸阅读

### 9.1 直接影响

- **GPT-3 (175B)**：规模直接基于 Kaplan scaling law 的 $N_{\text{opt}} \propto C^{0.73}$ 预测
- **后续工作**：几乎所有大规模语言模型的扩展策略都从此论文出发

### 9.2 后续关键修正

1. **Chinchilla (Hoffmann et al., 2022)**：修正数据-参数最优比例为等比例增长（各 ∝ C^0.5），揭示大语言模型普遍训练数据不足
2. **Scaling Data-Constrained LMs (2023)**：数据受限情境（如低资源语言）下的 scaling law 修正
3. **Porian et al. (2024)**：识别出 Kaplan 与 Chinchilla 分歧的3个技术原因，统一了两篇论文的框架

### 9.3 局限性

- 基于自回归 Transformer，不完全适用于 encoder-only / encoder-decoder 架构
- 最优数据-参数比已被 Chinchilla 修正；GPT-3 事后被证明数据不足
- 实验范围（N ≤ 1.5B）之外的行为（如涌现能力、相变点）未能捕获
- 所有实验在单一数据集（WebText2）上进行，跨数据集的定量泛化性未验证
- 训练数据质量对 scaling 斜率的影响未被考虑
- 部分超参数（如初始化 scale、momentum）调优不够系统，可能影响结论的精确性

### 9.4 推荐延伸阅读

1. **"Training Compute-Optimal Large Language Models"** — Hoffmann et al. (Chinchilla, 2022) [arXiv:2203.15556]
2. **"Resolving Discrepancies in Compute-Optimal Scaling of Language Models"** — Porian et al. (NeurIPS 2024)
3. Jared Kaplan 学术讲座 — [Neural Scaling Laws and GPT-3](https://www.youtube.com/watch?v=sNfkZFVm_xs)

---

*报告基于 arXiv:2001.08361 "Scaling Laws for Neural Language Models" by Kaplan et al. (OpenAI, 2020)*