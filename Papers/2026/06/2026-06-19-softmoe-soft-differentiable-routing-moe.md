# SoftMoE: Soft Differentiable Routing for Mixture-of-Experts in LLMs

## 论文元数据
- 标题：SoftMoE: Soft Differentiable Routing for Mixture-of-Experts in LLMs
- 作者：Mikołaj Zasada, Łukasz Struski, Jacek Tabor, Marcin Kurdziel
- 机构：AGH University of Krakow, Jagiellonian University, Warsaw University of Technology
- arXiv ID：2606.17952
- 发表/提交日期：2026-06-16
- 会议：ICML 2026
- 官方代码：https://github.com/dlcuda/SoftMoE
- 代码发现方式：PDF扫描
- CodeGraph分析：跳过（仓库基于Megatron-LM补丁）

---

## Ch1: 论文概述与核心贡献

Mixture-of-Experts (MoE) 架构通过条件计算（conditional computation）突破密集模型（dense transformer）的扩展瓶颈：在固定推理成本下，仅激活小部分专家网络处理每个输入，从而大幅扩展模型总参数量。标准稀疏 MoE 采用 hard top-k 路由机制，对每个 token 选择得分最高的 k 个专家，其余专家完全不参与计算。这种离散选择算子虽然保持了因果性、兼容自回归语言模型，但其根本缺陷在于不可微性。

Hard top-k 路由的不可微性导致三个深层问题：(1) 梯度无法流经专家选择过程，路由器只能通过选择之前的 softmax surrogate gradient 训练；(2) 每层激活的专家数量必须预先固定，无法根据输入难度自适应调整计算分配；(3) 全局计算容量静态僵化，难以在层间动态重分配资源。这使得 MoE 在面对复杂输入时仍需强制激活固定数量专家，造成计算资源浪费。

SoftMoE 通过可微软路由机制解决上述问题。其核心创新在于将离散 top-k 算子替换为基于 LapSum relaxation 的 truncated soft top-k，允许梯度流经路由决策，支持端到端优化专家选择。LapSum 算子在 $O(n)$ 时间内计算 differentiable threshold，生成软选择权重 $\tilde{p}_i \in [0,1]$，且满足 $\sum_i \tilde{p}_i = k$。经截断（truncation）后，低权重专家被剔除，每个 token 激活的专家数量变为输入依赖的变量（input-dependent），而非常量。

SoftMoE 的第二项贡献是可学习的全局约束专家预算。传统 MoE 在每层固定 $k$ 个专家，而 SoftMoE 参数化每层平均活跃专家数 $k_l$，施加全局预算约束 $\sum_l k_l = K$（$k_l \geq 1$）。通过 softmax reparameterization，$k_l$ 在训练中与模型参数联合优化，层间形成竞争关系：增加某层专家分配需在其他层补偿。实验显示，模型学习到高度非均匀的分配策略，后 3 层吸收约 50% 总预算，与前层形成鲜明对比。

实验在 GPT-2 架构（10 层、32 专家/层、每专家 5M 参数 → 总计 1.63B）上进行，使用 OWT (~9B tokens) 和 C4 (~206B tokens) 数据集。SoftMoE 在语言建模和下游任务（PIQA, HellaSwag, ARC-E）上匹配或超越 Sparse MoE 性能，同时激活更少专家。在 OWT 数据集上，SoftMoE* ($k=1.5$, $\alpha=2$) 达到 Loss 2.78，超越 Sparse MoE ($k=2$) 的 2.79；在 $\leq 2$ 专家/层预算下，训练阶段平均活跃专家数减少 17%（C4）至 24%（OWT）。在 OWT 上，SoftMoE (含 learned allocation, $k=2$, $\alpha=2$) 进一步将 loss 降至 2.74，并在 ARC-E 上达到最强 38.27%。非均匀层间分配在训练前 25k 步内快速收敛，展现出高效的计算资源利用。

*Table: SoftMoE vs Sparse MoE 核心区别*

| **特性** | **Sparse MoE** | **SoftMoE** |
|---------|--------------|-------------|
| **路由机制** | Hard top-k（离散选择） | Soft top-k + truncation（可微） |
| **每层激活专家数** | 固定 k | 输入依赖的变量（均值 k_l） |
| **层间分配** | 均匀（每层 k） | 可学习（非均匀，受全局预算约束） |
| **梯度流** | 仅通过 softmax surrogate | 完整流经路由决策 |
| **计算效率** | 强制激活固定数量专家 | 自适应激活，平均更少专家 |
| **因果性** | 保持 | 保持 |

---

## Ch2: 研究背景与动机

Dense Transformer 扩展受推理成本线性增长制约。Kaplan et al. (2020) 和 Hoffmann et al. (2022) 的 scaling laws 表明，模型性能随参数量、数据量和计算量呈现幂律关系，但密集模型推理时需激活全部参数，成本随规模线性增加，部署难度大。Mixture-of-Experts 通过条件计算突破此限制：将网络划分为多个专家模块，每个 token 仅路由至少量专家，推理时激活参数量远小于总参数量，实现"大参数、小计算"的扩展路径。

MoE 思想可追溯至 Jacobs et al. (1991) 和 Jordan & Jacobs (1994) 的混合专家网络，其核心是门控网络（gating network）动态选择专家处理输入。Shazeer et al. (2017) 将 MoE 引入 Transformer，在自然语言处理任务中首次展示条件计算的规模效益。Fedus et al. (2022) 提出 Switch Transformer，简化负载均衡策略，将专家数扩展至数千，在 1.6T 参数下达到优于密集模型的性能。Lepikhin et al. (2021) 的 GShard 进一步沿此方向推进，但这些工作均沿用 hard top-k 路由，未触及不可微性的根本限制。

Hard top-k 的不可微性在训练-推理间制造梯度不匹配。训练时，路由器权重 $W_g$ 通过 softmax 的 surrogate gradient 更新，但实际专家选择由 TopK 算子执行，梯度在离散边界截断。这种不匹配使路由器无法精确学习"哪些专家应被激活"，只能依赖间接信号。推理时，固定 $k$ 导致对所有输入（无论难易）强制激活相同数量专家，计算资源无法动态分配。此外，均匀的层间分配（每层 $k$ 个专家）忽略了 Transformer 层的功能差异：前层处理局部句法，后层处理全局语义，可能需要不同专家容量。

关键问题在于：能否设计既保持因果性、又可微的路由机制，使模型能端到端优化专家选择？Struski et al. (2025) 提出的 LapSum 算子为此提供基础。LapSum 定义为 $\sum_i F_{\text{Lap}}(r_i - x) = k$，其中 $F_{\text{Lap}}$ 为 Laplace 累积分布函数，在 $O(n)$ 时间内求解 differentiable threshold $\tilde{x}$，输出软权重 $\tilde{p}_i = F_{\text{Lap}}(r_i - \tilde{x})$。与 softmax 不同，LapSum 输出和为 $k$（非 1），且每个元素在 $[0,1]$ 区间，天然适合 top-k relaxation。LapSum 的 closed-form 解避免迭代或排序，计算成本低廉，适合大规模 MoE 部署。

SoftMoE 在此基础上构建完整框架：用 SoftTopK(LapSum(·, k)) 替换 TopK(·)，添加截断恢复稀疏性，参数化层间预算并施加全局约束。这实现真正的可微 MoE 路由，保持自回归兼容，同时学习非均匀的层间专家分配，在语言建模任务中展现出更优的计算效率。

---

### 第3章 从离散到软化的专家选择

## 3.1 Standard Sparse MoE Routing 回顾

在 Mixture-of-Experts 架构中，核心挑战是如何高效地为每个输入 token 选择合适的专家子集。标准稀疏 MoE 采用硬路由机制：gate 网络先对得分做 softmax，再执行 TopK（与 SoftMoE 直接对原始 gate 得分做 SoftTopK 不同，见 3.3.1 节）。

具体而言，给定输入特征 $\mathbf{x}$，gate 网络首先计算路由得分：

$$r = xW_g$$

其中 $W_g$ 是 gate 权重矩阵，$r \in \mathbb{R}^n$（$n$ 为专家总数）。然后通过 softmax 将得分归一化为概率分布：

$$p = \text{softmax}(r)$$

接下来执行硬 TopK 选择（与当前整理口径一致）：

$$G(x) = \text{TopK}(p, k)$$

其中 $G(x) \in \{0,1\}^n$ 是一个二值化 mask，仅保留概率最高的 $k$ 个专家（对应位置为 1，其余为 0）。最终输出为选中专家的加权组合：

$$y = \sum_{i=1}^{n} G_i(x) \cdot p_i \cdot E_i(x)$$

其中E_i(x)是第i个专家网络的输出。

**不可微性问题**：TopK 操作引入的离散性导致梯度无法流经选择过程。尽管 gate 网络 $W_g$ 通过 TopK 之前的 softmax 接收 surrogate 梯度，但真正的选择决策（即哪些专家被激活）无法通过端到端反向传播优化。这导致两个限制：

1. **固定专家数量**：每个token必须激活恰好k个专家，无法根据输入难度或重要性自适应调整
2. **次优路由**：gate网络只能间接优化选择，无法直接基于下游任务损失优化专家分配

从cardinality约束角度理解，TopK等价于求解：

$$\sum_{i=1}^{n} m_i = k, \quad m_i \in \{0,1\}$$

这是一个组合优化问题，其离散性阻碍了基于梯度的优化。

## 3.2 LapSum: Differentiable Top-k Relaxation

为解决上述不可微性问题，SoftMoE采用LapSum松弛（Struski et al., 2025），将离散TopK转化为连续可微的soft选择。

### 3.2.1 基本思想

TopK 问题可重新表述为 order statistics 问题：寻找阈值 $\tilde{x}$，使得恰好有 $k$ 个元素超过该阈值。给定排序后的得分 $r_{(1)} \geq r_{(2)} \geq \cdots \geq r_{(n)}$，TopK 等价于找到 $\tilde{x}$ 满足：

$$r_{(k)} \geq \tilde{x} > r_{(k+1)}$$

LapSum通过引入Laplace分布的累积分布函数（CDF）F_Lap将此连续化。LapSum的核心方程定义为：

$$\text{LapSum}(x) = \sum_{i=1}^{n} F_{\text{Lap}}(r_i - x) = k$$

其中F_Lap是Laplace CDF：

$$F_{\text{Lap}}(z) = \begin{cases}
\frac{1}{2} \exp(z) & \text{if } z \leq 0 \\
1 - \frac{1}{2} \exp(-z) & \text{if } z > 0
\end{cases}$$

该方程有唯一解 $\tilde{x}$，可在 $O(n)$ 时间内求解（无需显式排序，也不需要遍历所有候选阈值）。

### 3.2.2 Soft选择权重

一旦获得 $\tilde{x}$，计算 soft selection weights：

$$\tilde{p}_i = F_{\text{Lap}}(r_i - \tilde{x})$$

关键性质：

1. **有界性**：每个权重 $\tilde{p}_i \in [0,1]$，因为 $F_{\text{Lap}}$ 输出为概率
2. **和约束**：$\sum_i \tilde{p}_i = k$，直接满足 cardinality 要求
3. **可微性**：$\tilde{x}$ 是 $r$ 的可微函数，因此整个操作支持端到端梯度流

### 3.2.3 与softmax的对比

| 特性 | Softmax | LapSum (SoftTopK) |
|------|---------|-------------------|
| 输出维度 | 与输入相同 | 与输入相同 |
| 输出范围 | [0,1]，但无明确上界 | [0,1]，每个元素有界 |
| 输出和 | 恒为1（概率分布） | 恒为k（cardinality） |
| 梯度特性 | 输入可微，k不可微 | 输入和k都可微 |
| 解释 | 归一化为概率分布 | soft mask满足基数约束 |

LapSum可视为softmax的推广：当k=1时，LapSum退化为类似softmax的软分配；但当k>1时，LapSum明确控制"激活强度"总量而非概率归一化。

### 3.2.4 计算复杂度

LapSum的求解需要：
- **时间复杂度**：O(n)，因为涉及n次CDF评估与阈值求解
- **空间复杂度**：O(n)，存储中间得分和CDF值

对于典型的n=32-64个专家，此开销相比后续专家网络的计算可忽略不计。

## 3.3 SoftMoE Layer 架构

基于LapSum的可微性，SoftMoE层设计如下：

### 3.3.1 完整前向传播

与 Sparse MoE 不同，SoftMoE **不对 gate 得分做 softmax**，而是直接对原始线性输出执行 SoftTopK（按当前笔记整理，见相关实现描述）。

给定输入 $\mathbf{x}$，首先计算 gate 得分：

$$r = xW_g$$

然后执行 soft top-k 选择：

$$\tilde{p} = \text{SoftTopK}(r, k)$$

其中 SoftTopK 内部执行 LapSum 求解 $\tilde{x}$，然后计算 $\tilde{p}_i = F_{\text{Lap}}(r_i - \tilde{x})$。

接下来应用截断（truncation）以恢复稀疏性（当前笔记记号 $\mathcal{T}$）：

$$T(z) = z \cdot \mathbb{I}[z > \tau]$$

其中 $\mathbb{I}[\cdot]$ 是指示函数，$\tau$ 为截断超参数。按当前笔记整理，$\tau$ 不是一个完全固定的常数，而是会结合初始 batch 的激活统计做标定，使训练初期每层每 token 的平均激活量保持在目标附近；若要写成严格论文表述，建议回看正文或附录原式。截断后的权重为：

$$p_{\text{final}} = T(\tilde{p})$$

此外，每层活跃专家数上界可写成与 $\alpha$ 相关的形式；这里把它记作 $\lceil k_i \cdot \alpha \rceil$，但如果后续要严格对齐原文，最好再回查一次符号定义和约束条件。

最终输出为专家的加权组合：

$$y = \sum_{i=1}^{n} p_{\text{final},i} \cdot E_i(x)$$

### 3.3.2 截断的必要性

理论上可以直接使用所有 soft weights（即 $\tilde{p}$），但会导致密集计算——所有专家都接收非零权重（尽管很小），抵消 MoE 稀疏性的优势。截断通过去除低于 $\tau$ 的权重，恢复 per-token 的稀疏激活。

### 3.3.3 截断的不连贯性

截断引入轻微的不连贯性。论文在 $z=\tau$ 处采用次梯度约定 $\mathcal{T}'(0)=0$：

$$\frac{\partial T(z)}{\partial z} = \begin{cases}
1 & \text{if } z > \tau \\
0 & \text{if } z < \tau \\
0 & \text{if } z = \tau \quad \text{(subgradient)}
\end{cases}$$

论文指出该截断算子等价于 shifted ReLU：活跃专家接收梯度，非活跃专家梯度为零。在决策边界处专家选择不可微，类似于 hard top-$k$ 路由，但模型通过 LapSum inverse 对选择参数 $k$ 保持可微性。实验中硬截断表现稳定。

### 3.3.4 Input-Dependent 稀疏性

关键创新在于：截断基于 threshold $\tau$ 而非 rank $k$。这意味着不同 token 可能激活不同数量的专家：

- 对于"简单" token，gate 得分集中在前几个专家，大部分 $\tilde{p}_i < \tau$ → 仅激活 1–2 个专家
- 对于"困难" token，gate 得分分布更平坦，更多 $\tilde{p}_i > \tau$ → 可能激活 3–5 个专家

相比之下，Sparse MoE的TopK强制每个token激活恰好k个专家，无论输入难度。

### 3.3.5 架构示意图（Figure 2描述）

SoftMoE层的流程可概括为：

```
Input x → Gate Network (xW_g) → SoftTopK(r, k) → Truncation T(·) →
Sparse Expert Activation → Weighted Sum → Output y
```

其中：
- **Gate Network**：线性层计算原始路由得分
- **SoftTopK**：LapSum求解器产生soft weights
- **Truncation**：去除低权重，恢复稀疏性
- **Sparse Expert Activation**：仅计算 $p_{\text{final},i} > \tau$ 的专家
- **Weighted Sum**：按截断后权重组合专家输出

### 3.3.6 梯度流

相比 Sparse MoE，SoftMoE 的梯度可流经整个选择过程：

$$\frac{\partial \mathcal{L}}{\partial W_g} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial \tilde{p}} \cdot \frac{\partial \tilde{p}}{\partial r} \cdot x^\top$$

其中 $\partial y / \partial \tilde{p}$ 和 $\partial \tilde{p} / \partial r$ 都可微（$\tilde{p}$ 通过 LapSum 可微）。这允许 gate 网络直接基于任务损失优化路由决策；对于层间预算参数和路由打分，梯度都可以经 LapSum 反向传播，这正是固定整数 top-k 所不具备的。

---

### 第4章 学习层间专家分配

## 4.1 Hard Allocation vs Soft Allocation

标准Sparse MoE在每个MoE层固定激活k个专家（例如k=2），这意味着整个模型的计算容量均匀分布在所有层。然而，不同层处理的语言表征复杂度可能不同——某些层可能需要更多专家容量来处理多样化任务，而其他层可能用较少专家即可。

SoftMoE 通过引入可学习的 per-layer budget 参数 $k_l$，允许模型自适应地分配计算资源。

### 4.1.1 层间异质性

直觉上，浅层可能处理局部句法模式（可由少量专家覆盖），而深层需要整合全局语义和任务特定知识（需更多专家）。实验数据支持这一假设（见第5.3节，最后3层吸收约50%预算）。

### 4.1.2 学习目标

目标是在满足全局计算约束的前提下，最小化训练损失：

$$\min_{\theta, \{k_l\}_{l=1}^L} \mathcal{L}(\theta, \{k_l\}) \quad \text{s.t.} \quad \sum_{l=1}^{L} k_l = K, \quad k_l \geq 1$$

其中：
- θ：模型参数（包括gate网络和专家网络）
- $k_l$：第 $l$ 层的平均活跃专家数（可学习参数）
- $K$：全局预算（总活跃专家数）
- 约束 $k_l \geq 1$：确保每层至少激活 1 个专家（避免层完全退化）

## 4.2 全局预算约束

为实现可微的层间分配，笔记里将 SoftMoE 记作引入了 reparametrization 技巧；如果要严格对应原文，最好回查其符号定义，而不要只依赖这里的整理。

### 4.2.1 参数化方案

当前笔记把无约束的 $\eta_l$ 先经 softmax 转换为比例 $\pi_l$：

$$\pi_l = \frac{\exp(\eta_l)}{\sum_{j=1}^{L} \exp(\eta_j)}$$

再通过仿射变换映射到 $k_l$：

$$k_l = \pi_l \cdot (K - L) + 1$$

按这套笔记整理，可以得到：
1. **下界**：当 $\pi_l \to 0$ 时，$k_l \to 1$（满足 $k_l \geq 1$）
2. **和约束**：$\sum_l \pi_l = 1 \Rightarrow \sum_l k_l = (K-L) + L = K$

### 4.2.2 端到端可微性

$\eta_l$ 是可学习参数，因此在这套整理写法下：

$$\frac{\partial \mathcal{L}}{\partial \eta_l} = \frac{\partial \mathcal{L}}{\partial k_l} \cdot \frac{\partial k_l}{\partial \pi_l} \cdot \frac{\partial \pi_l}{\partial \eta_l}$$

所有项都可微，支持联合优化。训练过程中，各层通过梯度竞争预算；更准确地说，这是一个“预算重分配”的直觉描述，未必等同于论文正文里的完整推导。

### 4.2.3 与Sparse MoE的等价性

当 $K = 2L$ 时（即全局预算写成“平均每层 2 个专家”），这套整理方式下可与 Sparse MoE 的 $k=2$ per layer 作粗略对照。但“完全等价”不建议写死，因为 SoftMoE 的重点恰恰是允许非均匀分配。

### 4.2.4 架构示意图（Figure 3描述）

```
Global Budget K → Softmax Reparam(η) → Per-Layer k_l →
Layer 1: SoftTopK(r, k_1)
Layer 2: SoftTopK(r, k_2)
...
Layer L: SoftTopK(r, k_L)
```

## 4.3 计算开销分析

引入可学习budget会带来额外的计算和内存开销。

### 4.3.1 LapSum开销

对于每层每token：

- **时间**：O(n)求解LapSum（n=专家数，通常32-64）
- **空间**：O(n)存储gate得分和CDF值

相比标准MoE的gate计算（$xW_g$，复杂度 $O(n \cdot d)$，$d$ 为输入维度），LapSum的开销可忽略。论文指出，对于典型的 $n=32$–$64$ 个专家，LapSum 的 $O(n)$ 开销远小于 gate 矩阵乘法的 $O(n \cdot d)$，且不抵消减少活跃专家数带来的收益。

### 4.3.2 Budget参数开销

可学习参数 $\eta_l$（共 $L$ 个）增加 $O(L)$ 参数量（$L$ 通常 $<20$），相比数十亿模型参数可忽略。

### 4.3.3 整体效率

尽管LapSum引入per-token开销，但通过减少活跃专家总数（例如SoftMoE*以更少专家达到同等性能），整体计算仍可能更低。关键收益在于：

1. **更高效容量利用**：预算分配到真正需要的层
2. **稀疏性保持**：截断确保实际激活专家数接近 $k_l$
3. **端到端优化**：路由和分配联合学习，避免人工调参

---

**章节总结**：
- Ch3介绍了SoftMoE的核心技术——基于LapSum的可微top-k路由，替代传统MoE的硬选择
- Ch4展示了如何通过全局预算约束学习层间分配，实现自适应计算

### 第5章 实验结果与分析

## 5.1 实验设置

### 5.1.1 模型架构

实验基于GPT-2 decoder-only架构，具体配置如下：

- **Transformer blocks**: 10层
- **Experts per block**: 每层32个专家网络
- **Expert parameters**: 每个专家500万参数 (5M params/expert)
- **Total parameters**: 1.63B (10 blocks × 32 experts × 5M)

所有专家网络均为MLP结构，采用标准MoE block设计。

### 5.1.2 对比基线

实验对比以下三类模型：

1. **Sparse MoE (Switch Transformer 变体)**: $k \in \{1,2,3,4\}$
   - 标准硬路由 top-k 选择
   - 每层固定激活 $k$ 个专家

2. **SoftMoE*** (不含 learned allocation): $k \in \{1, 1.5, 2\}$
   - 使用 soft routing 但层间均匀分配
   - 每层激活相同数量的专家

3. **SoftMoE** (含 learned allocation): $k \in \{1, 1.5, 2\}$, $\alpha \in \{2, 4\}$
   - 学习非均匀层间分配
   - 全局预算约束；每层活跃专家上界 $\lceil k_l \cdot \alpha \rceil$

### 5.1.3 数据集

两个大规模英语语料库用于预训练：

- **OWT (OpenWebText)**: ~9B tokens
- **C4 (Colossal Clean Crawled Corpus)**: ~206B tokens

文本处理采用BPE分词，词汇表大小32k (BPE tokenizer with 32k vocabulary)。

### 5.1.4 训练配置

如果按当前实现笔记整理，训练大体是基于 Megatron-LM 框架：

- **C4 训练**: 164k steps
- **OWT 训练**: 18k steps
- **辅助损失**: balancing loss (Fedus et al.)
- **Gating noise**: 笔记里记为 SoftMoE 在 gate 预激活值上加小噪声；Sparse MoE 基线关闭 jitter，但这类实现细节建议回看附录确认
- **优化器**: Adam，学习率和 warm-up 策略按当前笔记整理；若要写成论文事实，建议补原文出处
- **LapSum 温度**: cosine 退火从 5 到 1 的说法先保留为笔记结论
- **精度**: bfloat16 + float32 混合精度，若正文未明示就不要写得太死

### 5.1.5 评估任务

Zero-shot下游任务评估：

- **PIQA (Physical Interaction Question Answering)**: 常识推理
- **HellaSwag**: 常识句子补全
- **ARC-E (Abstraction and Reasoning Corpus, Easy)**: 抽象推理

## 5.2 SoftMoE* vs Sparse MoE: 效率与性能权衡

本节分析不含learned allocation的SoftMoE*与标准Sparse MoE的对比，核心观察是SoftMoE*能够以更少活跃专家实现更低的语言建模loss。

### 5.2.1 OWT数据集结果

在OWT (~9B tokens)上，SoftMoE*在低预算配置下显著优于Sparse MoE：

| 模型配置 | Loss | Train-AE | Infer-AE | Active-Params |
|---------|------|----------|----------|---------------|
| SoftMoE* (k=1.5, α=2) | 2.78 | 1.53 | 1.73 | 143.66M |
| Sparse MoE (k=2) | 2.79 | 2 | 2 | 156.94M |

核心发现：
- SoftMoE* (k=1.5) loss (2.78) 低于 Sparse MoE (k=2) loss (2.79)
- SoftMoE*训练时平均激活1.53个专家，相比Sparse MoE的2个专家减少24%
- 推理时平均激活1.73个专家，相比Sparse MoE减少13%
- 活跃参数量更少 (143.66M vs 156.94M)

### 5.2.2 高预算配置（OWT）

当允许更高专家预算时，SoftMoE* 的 gap 更明显：

| 模型配置 | Loss | Train-AE | Infer-AE | Active-Params |
|---------|------|----------|----------|---------------|
| SoftMoE* (k=1, α=4) | 2.70 | 3.64 | 3.73 | 242.06M |
| Sparse MoE (k=4) | 2.75 | 4 | 4 | 255.34M |

关键观察：
- SoftMoE* loss (2.70) 显著低于 Sparse MoE (k=4) loss (2.75)
- 训练时平均激活3.64个专家，相比Sparse MoE的4个专家减少9%
- 性能提升来源：soft routing允许模型根据token难度自适应调整活跃专家数量，而非强制固定k

### 5.2.3 自适应激活模式

Figure 4展示了SoftMoE*训练过程中活跃专家数的演化：

- **训练早期**: 活跃专家数波动较大，模型探索不同分配策略
- **训练中期**: 活跃专家数收敛到稳定分布，平均接近k
- **训练后期**: 某些token激活1-2个专家，某些token激活3-4个专家，呈现输入依赖的自适应模式

相比之下，Sparse MoE的活跃专家数始终恒定为k（每个token激活恰好k个专家）。

### 5.2.4 效率来源分析

SoftMoE*的效率增益源自differentiability带来的两个优势：

1. **Input-dependent sparsity**: 截断机制允许"简单"token激活更少专家，"困难"token激活更多专家
2. **Gradient-based routing**: Gate网络直接基于任务损失优化，避免surrogate梯度

## 5.3 学习层间分配的实验结果

本节分析含learned allocation的SoftMoE (记为SoftMoE (alloc.)) 与SoftMoE*的对比。

### 5.3.1 非均匀分配现象

Figure 1（底部）与 Figure 6 展示了 learned allocation 收敛后的层间分配模式（以 $K=2L$ 为例，$L=10$ 层）：

- **浅层 (Layer 1-3)**: 每层仅激活1-2个专家
- **中层 (Layer 4-7)**: 每层激活约2个专家
- **深层 (Layer 8-10)**: 最后3层激活3-4个专家，吸收约50%总预算

这一非均匀分配与BERT probing研究结论一致 (Tenney et al.):
- 浅层处理句法特征，需要较少专家容量
- 深层处理语义和任务特定信息，需要更多专家容量

### 5.3.2 学习动态

Figure 6 展示了层间分配参数 $k_l$ 随训练步数的演化（Figure 5 则对比含/不含 learned allocation 的训练 loss 与每层活跃专家数）：

- **0-5k steps**: 初始均匀分配（每层约2个专家）
- **5-25k steps**: 预算迅速向深层重新分配，最后3层从2个专家增加到3-4个
- **25k+ steps**: 分配基本稳定，微小波动

### 5.3.3 性能与计算trade-off

C4数据集上的完整对比 (Table 1):

| 模型配置 | Loss | Train-AE | PIQA | HellaSwag | ARC-E |
|---------|------|----------|------|----------|-------|
| Sparse MoE (k=1) | 2.27 | 1 | 69.75 | 43.14 | 42.50 |
| Sparse MoE (k=2) | 2.24 | 2 | 71.98 | 45.41 | 40.74 |
| SoftMoE* (k=1.5, α=2) | 2.23 | 1.65 | 71.06 | 45.49 | 39.15 |
| Sparse MoE (k=3) | 2.22 | 3 | 71.60 | 46.36 | 42.50 |
| SoftMoE (k=1.5, α=2, alloc.) | 2.22 | 1.96 | 71.38 | 46.79 | 43.39 |
| Sparse MoE (k=4) | 2.20 | 4 | 71.60 | 47.23 | 43.56 |
| SoftMoE* (k=1, α=4) | 2.19 | 3.60 | 72.91 | 48.48 | 43.21 |
| SoftMoE* (k=2, α=2) | 2.21 | 2.91 | 71.60 | 48.05 | 42.86 |
| SoftMoE (k=2, α=2, alloc.) | 2.20 | 3.35 | 72.03 | 47.48 | 42.15 |

观察：
- C4 上最优 loss 配置为 SoftMoE* ($k=1$, $\alpha=4$)，loss 2.19，同时在 PIQA (72.91%) 和 HellaSwag (48.48%) 上也达到最优
- Learned allocation 的主要 trade-off：SoftMoE ($k=2$, $\alpha=2$) 的 ARC-E (42.15%) 略低于 Sparse MoE ($k=4$) 的 43.56%，但 SoftMoE ($k=1.5$, $\alpha=2$, alloc.) 的 ARC-E (43.39%) 接近 Sparse MoE 最佳
- 在 $\leq 2$ 专家/层预算下，SoftMoE* 相比 Sparse MoE 减少约 17% 活跃专家数（Train-AE 1.65 vs 2）

### 5.3.4 层间异质性的理论解释

从信息论角度理解：
- **浅层**: 编码局部句法模式，信息熵较低，少量专家即可覆盖
- **深层**: 整合全局上下文和任务知识，信息熵较高，需要更多专家容量

SoftMoE 通过学习 $k_l$ 自动发现这一结构，无需人工设计。

## 5.4 Downstream任务性能

### 5.4.1 HellaSwag

HellaSwag评估常识推理和句子补全能力。Table 1显示，在HellaSwag上soft routing在所有配置中一致优于Sparse MoE：

**OWT 预训练：**

| SoftMoE配置 | HellaSwag (%) | Sparse MoE配置 | HellaSwag (%) |
|------------|---------------|---------------|---------------|
| SoftMoE* (k=1, α=4) | 33.68 | Sparse MoE (k=4) | 32.36 |
| SoftMoE* (k=2, α=2) | 32.50 | Sparse MoE (k=2) | 31.50 |

**C4 预训练：**

| SoftMoE配置 | HellaSwag (%) | Sparse MoE配置 | HellaSwag (%) |
|------------|---------------|---------------|---------------|
| SoftMoE* (k=1, α=4) | 48.48 | Sparse MoE (k=4) | 47.23 |
| SoftMoE (k=2, α=2) | 47.48 | Sparse MoE (k=2) | 45.41 |

关键发现：在HellaSwag上，soft routing在所有配置中一致优于Sparse MoE，支持"soft routing提升表征质量"的假设。

### 5.4.2 PIQA

PIQA评估物理交互常识推理：

**OWT 预训练：**

| SoftMoE配置 | PIQA (%) | Sparse MoE配置 | PIQA (%) |
|------------|---------|---------------|---------|
| SoftMoE* (k=1, α=4) | 63.93 | Sparse MoE (k=4) | 63.87 |
| SoftMoE* (k=2, α=2) | 63.82 | Sparse MoE (k=2) | 62.40 |

**C4 预训练：**

| SoftMoE配置 | PIQA (%) | Sparse MoE配置 | PIQA (%) |
|------------|---------|---------------|---------|
| SoftMoE* (k=1, α=4) | 72.91 | Sparse MoE (k=4) | 71.60 |
| SoftMoE (k=2, α=2) | 72.03 | Sparse MoE (k=2) | 71.98 |

在C4上，SoftMoE* ($k=1$, $\alpha=4$) 达到最优 72.91%，相比 Sparse MoE ($k=4$) 的 71.60% 提升 1.31 个百分点。OWT 上 soft routing 也 consistently 优于或匹配 Sparse MoE。

### 5.4.3 ARC-E

ARC-E评估抽象推理能力：

**OWT 预训练：**

| SoftMoE配置 | ARC-E (%) | Sparse MoE配置 | ARC-E (%) |
|------------|----------|---------------|----------|
| SoftMoE (k=2, α=2, alloc.) | 38.27 | Sparse MoE (k=4) | 36.68 |
| SoftMoE* (k=2, α=2) | 37.21 | Sparse MoE (k=2) | 34.74 |

**C4 预训练：**

| SoftMoE配置 | ARC-E (%) | Sparse MoE配置 | ARC-E (%) |
|------------|----------|---------------|----------|
| SoftMoE (k=1.5, α=2, alloc.) | 43.39 | Sparse MoE (k=4) | 43.56 |
| SoftMoE* (k=1, α=4) | 43.21 | Sparse MoE (k=2) | 40.74 |

OWT 上 SoftMoE（含 learned allocation，$k=2$, $\alpha=2$）达到最强 38.27%，相比 Sparse MoE ($k=4$) 的 36.68% 提升 1.59 个百分点。C4 上 Sparse MoE ($k=4$) 以 43.56% 略优于 SoftMoE 的最佳配置 43.39%，但 gap 很小 ($<0.2\%$)。ARC-E 因评估集较小，方差较高。

### 5.4.4 预训练loss与下游性能关系

综合Table 1数据，观察：

- **最佳预训练 loss 配置通常对应较强下游表现**：OWT 上 SoftMoE* ($k=1$, $\alpha=4$) loss 最低 (2.70)；同数据集上 ARC-E 最高 (38.27%) 来自 SoftMoE ($k=2$, $\alpha=2$，含 learned allocation)
- **Soft routing的优势在下游任务更明显**: 尽管loss提升可能仅为0.01-0.02，下游任务提升可达1-2个百分点

这表明soft routing不仅优化perplexity，更提升表征的迁移能力。

---

### 第6章 代码实现笔记（需谨慎）

> 说明：这一章主要是根据公开代码仓库和本地整理笔记归纳出来的实现侧信息，不是论文正文逐项明确写出的结论。为了避免把推断写成事实，下面统一采用保守表述；如果你希望“只保留论文正文可直接证实的内容”，这一章可以进一步压缩甚至删除。

## 6.1 仓库结构

如果官方仓库与当前笔记一致，那么 SoftMoE 很可能是基于 Megatron-LM 做 patch 式集成，而不是完全重写训练栈：

```
external/Megatron-LM/    # Megatron-LM git submodule
patches/                  # SoftMoE routing集成补丁
scripts/                  # 训练/评估/数据预处理脚本
train_configs/            # 论文实验配置文件
```

### 6.1.1 Megatron-LM集成方式

从仓库组织方式推测，SoftMoE routing 可能通过补丁插入 Megatron-LM 的 MoE 层，而不是直接改写全部训练代码。这个判断更像实现侧观察，论文正文未必逐项展开。

## 6.2 核心实现（推断）

### 6.2.1 SoftTopK替换TopK

如果按仓库补丁理解，MoE MLP 层的核心修改大概率集中在 gate routing 部分：

```python
# Standard Sparse MoE (Switch Transformer)
gates = torch.softmax(input @ W_gate, dim=-1)
selected_indices = torch.topk(gates, k=k, dim=-1).indices  # 硬TopK

# SoftMoE
gates = input @ W_gate
soft_weights = soft_topk_lapsum(gates, k=k)  # SoftTopK via LapSum
truncated_weights = truncate(soft_weights, threshold=tau)  # 截断
```

LapSum 求解器实现侧可概括为：
- 输入：原始 gate 得分 $r \in \mathbb{R}^n$，目标基数 $k$
- 输出：soft weights $\tilde{p} \in [0,1]^n$，$\sum_i \tilde{p}_i = k$
- 算法：基于 Laplace CDF 的阈值求解；训练时是否采用 temperature schedule，需要以仓库实现为准，不能仅凭论文摘要断言

### 6.2.2 截断实现

截断是恢复稀疏性的关键。阈值是否按初始 batch 统计动态设定、以及有效阈值是否写成 $\tau \cdot k / E$，建议以正文/附录原式为准；这里先按当前笔记保留：

```python
def truncate(soft_weights, threshold):
    # threshold 的具体标定方式建议回看原文或仓库实现
    mask = (soft_weights > threshold).float()
    return soft_weights * mask
```

截断后，仅高于阈值的专家会被保留，其余权重置零。

### 6.2.3 可学习Budget参数（推断）

Learned allocation 通过 per-layer 参数 $\eta_l$ 实现：

```python
class LayerBudget(nn.Module):
    def __init__(self, num_layers, global_budget_K):
        self.eta = nn.Parameter(torch.zeros(num_layers))  # 无约束参数
        self.K = global_budget_K
        self.L = num_layers

    def forward(self):
        pi = F.softmax(self.eta, dim=0)  # 转换为比例
        k_per_layer = pi * (self.K - self.L) + 1  # 映射到[1, K-L+1]
        return k_per_layer
```

训练时每个 MoE 层如何读取当前层的 $k_l$，更像是实现约定，建议结合仓库代码核对。

## 6.3 训练配置与环境

这一部分如果要严谨复核，最好直接看仓库配置文件或论文附录，而不要只凭笔记补全。当前文档里像硬件型号、CUDA/Python 版本、脚本名、内存估算这类细节，都应视为实现笔记而不是论文结论。

## 6.4 使用流程

如果后续你希望保留“如何跑起来”的内容，建议改成一段简短的流程说明，并明确标注“来自代码仓库/本地整理，不是论文正文”。

---

### 第7章 局限性与总结

## 7.1 局限性

### 7.1.1 评估范围限制

论文仅在英语文本数据集上评估：
- **预训练数据**: OWT (~9B tokens), C4 (~206B tokens) — 均为英语
- **下游任务**: PIQA, HellaSwag, ARC-E — 英语常识推理任务

未探索多语言或多模态场景，SoftMoE在非英语文本或视觉-语言模型的泛化性未知。

### 7.1.2 模型规模

实验模型总参数1.63B，相比当前前沿LLM（数百亿至万亿参数）较小。SoftMoE在更大规模（如10B+）上的缩放行为需进一步验证：
- Soft routing的计算开销在大模型中是否仍可忽略？
- Learnable allocation的非均匀模式在大模型中是否保持？
- 与训练稳定性相关的超参数（如 $\tau$、$\alpha$）如何缩放？

### 7.1.3 未探索方向

论文未探索以下方向：
- **多模态扩展**: SoftMoE在视觉-语言模型或纯视觉模型中的适用性
- **动态预算**: 当前 $k_l$ 在训练后固定，推理时无法根据输入难度调整全局预算
- **更细粒度控制**: 基于token重要性或任务类型的conditional budget allocation

## 7.2 未来研究方向

### 7.2.1 更大规模验证

在10B-100B参数模型上验证SoftMoE的缩放行为，关键问题：
- SoftTopK的计算复杂度O(n)在大规模专家网络（n=128-256）中是否仍高效？
- LapSum求解的数值稳定性在更大n下是否需特殊处理？

### 7.2.2 多模态与多语言扩展

将SoftMoE应用于：
- 多语言预训练（如mC4、XGLM）
- 视觉-语言模型（如Flamingo、BLIP）
- 纯视觉MoE (如Vision Transformer with MoE)

### 7.2.3 自适应推理

扩展SoftMoE支持推理时动态预算：
- 根据输入复杂度调整全局预算K (简单输入用更少专家)
- 结合early exit机制，在浅层就满足置信度时提前退出

### 7.2.4 理论分析

从理论角度分析：
- Soft routing的收敛性质：相比hard routing是否更快收敛？
- 层间非均匀分配的泛化界：什么条件下深层需要更多容量？

## 7.3 总结

SoftMoE提出基于LapSum的differentiable top-k routing，替代标准Sparse MoE的hard selection。核心贡献包括：

1. **可微路由**: 通过LapSum relaxation和truncation实现端到端可微的专家选择，支持gradient-based optimization
2. **学习分配**: 参数化per-layer budget并施加全局约束，使模型能自适应重分配层间计算容量
3. **效率提升**: 在语言建模（C4, OWT）和下游任务（PIQA, HellaSwag, ARC-E）上，SoftMoE以更少活跃专家匹配或超越Sparse MoE

实验发现：
- SoftMoE* 相比 Sparse MoE，在 OWT（$k=1.5$, $\alpha=2$ vs $k=2$）上以更低 loss（2.78 vs 2.79）达到更优效率；训练阶段 Train-AE 减少 24%，推理 Infer-AE 减少 13%
- Learnable allocation呈现高度非均匀分布：最后3层吸收约50%总预算
- Soft routing在HellaSwag上一致优于hard routing，PIQA上也多数配置更优；ARC-E因评估集较小方差较高，但SoftMoE在OWT预训练上达到最强ARC-E

这些发现为MoE计算利用提供了新见解：**differentiable routing + learnable allocation是viable alternative to hard expert selection，尤其适合需要自适应计算的场景**。

---

- 总计约360行，涵盖论文实验设置、结果分析、工程实现和局限性讨论
