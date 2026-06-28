# LLaDA: Large Language Diffusion Models 论文精读报告

> arXiv:2502.09992 · Shen Nie, Fengqi Zhu et al. · NeurIPS 2025

## 论文元数据

| 项目 | 内容 |
|------|------|
| 标题 | Large Language Diffusion Models (LLaDA) |
| 作者 | Shen Nie, Fengqi Zhu, Zebin You et al. (Renmin University / Ant Group) |
| arXiv ID | 2502.09992 |
| 发表 | NeurIPS 2025 |
| 提交日期 | 2025-02-13 |
| 官方代码 | https://github.com/ML-GSAI/LLaDA (MIT License) |
| 基础架构 | Transformer (bidirectional, no causal mask) |

---

## 第1章 概述

LLaDA(Large Language Diffusion with masking)是首个在 8B(约 80 亿参数)规模上完成预训练的 masked diffusion language model。该工作见于论文 *Large Language Diffusion Models*(Nie、Zhu 等人,2025;arXiv:2502.09992;NeurIPS 2025),由中国人民大学与蚂蚁集团的研究团队共同完成。它对当前以 autoregressive(AR)建模为绝对主流的大语言范式提出挑战,并用 8B 规模的实证结果支撑了如下核心论断:

> 大语言模型的能力本质上源于 generative modeling 的 maximum likelihood estimation(MLE)原则(对应论文 Eq.1),而非 autoregressive 这一特定的概率分解形式(对应论文 Eq.2)。

换言之,"next-token prediction from left to right" 只是生成式建模的一种实现路径。masked diffusion 作为另一种同样遵循 MLE 目标的生成范式,只要训练目标设计正确,就能赋予模型可比拟甚至局部超越 AR 模型的语言能力。

### 1.1 基准性能

LLaDA 8B Base 在多个标准 benchmark 上达到了与同规模 AR 模型可比的水平:

| Benchmark | LLaDA 8B Base |
|-----------|---------------|
| MMLU | 65.9 |
| GSM8K | 70.3 |
| HumanEval | 35.4 |
| CMMLU | 69.9 |

可以看到,无论是知识综合(MMLU / CMMLU)、数学推理(GSM8K),还是代码生成(HumanEval),diffusion 范式的 LLaDA 都已进入实用区间。这一现象本身就是其核心论断的有力注脚:AR 形式并非 LLM 能力的必要条件。

### 1.2 反向推理能力

LLaDA 在需要"逆向"理解的任务上展现出超越主流 AR 模型的潜力。在 reversal poem completion(反向诗歌补全)任务上——该任务使用 496 对中文名句,要求模型给定一句诗后生成下一句(forward)或上一句(reversal),zero-shot 评估——LLaDA 8B 取得 45.6% 的准确率,显著高于 GPT-4o 的 34.3%。这一对比直接体现了双向 diffusion 建模对 reversal curse 的天然缓解(详见第2章)。

### 1.3 开源

项目代码已开源,仓库地址为 github.com/ML-GSAI/LLaDA,采用 MIT License。

---

## 第2章 背景、动机与方法

### 2.1 Autoregressive Models 的形式

当前主流大语言模型普遍采用 autoregressive modeling(ARM)。ARM 将序列 $x=(x_1,x_2,\dots,x_L)$ 的联合概率分解为从左到右逐步递进的条件概率连乘:

$$p(x)=\prod_{i=1}^{L} p(x_i \mid x_{<i})$$

其中 $x_{<i}=(x_1,\dots,x_{i-1})$ 表示位置 $i$ 之前的全部 token。这种分解决定了生成本质上是 sequential、left-to-right 的,并要求模型使用 causal(因果)attention:位置 $i$ 只能看到它左侧的上下文。

### 2.2 Reversal Curse

单向的信息流带来了一个广为人知的缺陷,即 reversal curse(反转诅咒)。当任务要求模型沿与训练相反的方向进行推理时(例如模型学会了"A → B",却无法据此回答"B → A"),ARM 往往会失败。其根源正是 causal attention 的方向性约束:token 无法建立真正的双向依赖,因此也就难以把顺序学习到的知识"反过来"使用。

### 2.3 Masked Diffusion 的双向优势

masked diffusion 放弃了 causal mask,改用 bidirectional attention(双向注意力)。在生成过程中,每个待恢复的位置都能同时关注序列中的全部上下文,信息流动不再有固定方向。这从根本上消除了 ARM 的方向性瓶颈,为缓解 reversal curse 提供了结构性条件。

### 2.4 前向与逆向过程

LLaDA 把离散文本建模为一个 masked diffusion 过程。给定原始序列 $x_0$,forward process(前向过程)采样一个统一的时间步 $t \sim U[0,1]$,并据此对序列中的每个 token 独立地执行 masking。reverse process(逆向过程)则由一个 mask predictor $p_\theta(\cdot \mid x_t)$ 承担:给定部分被 mask 的序列 $x_t$,模型一次性预测所有被 mask 位置上的原始 token。

这里的 mask 比例处理方式有别于既有工作:BERT 采用固定的 15% mask ratio,MaskGIT 则依赖启发式的 cosine masking schedule。LLaDA 通过对 $t$ 做均匀采样,使训练过程能够覆盖从极低到极高的整个 mask 比例区间。

### 2.5 训练目标

LLaDA 的训练损失为:

$$\mathcal{L}(\theta)=-\mathbb{E}_{t, x_0, x_t}\left[\frac{1}{t}\sum_{i=1}^{L}\mathbf{1}\bigl[x_i^t=M\bigr]\,\log p_\theta(x_i^0 \mid x_t)\right]$$

其中 $\mathbf{1}[x_i^t=M]$ 指示位置 $i$ 在 $x_t$ 中是否被 mask。其中最关键的设计是 $\frac{1}{t}$ 加权:它并非任意选取的经验项,而是 negative log-likelihood 的一个上界(upper bound),从理论上保证了以 masked diffusion 的方式执行 MLE 的合法性。这一权重正是 LLaDA 核心论断的数学落点——能力来自 MLE 原则(Eq.1),而非 AR 形式(Eq.2)。

### 2.6 模型架构

LLaDA 的 backbone 是一个 standard Transformer,但移除了 causal mask,改用 bidirectional attention。模型提供 1B 与 8B 两种规模。8B 版本配置包含 32 层、hidden dimension 4096、32 个 attention heads,FFN dimension 为 12288,激活函数采用 SwiGLU,位置编码采用 RoPE。1B 版本则为 22 层、hidden dimension 2048、FFN dimension 5634、Key/Value heads 仅 4 个(总计 1.49B 参数)。两者架构并不相同。

---
## 第3章 训练

LLaDA 的训练分为两个阶段:大规模预训练 (pre-training) 与监督微调 (supervised fine-tuning, SFT)。本节依次介绍模型架构、预训练数据与超参数,以及 SFT 的设计。

### 3.1 模型架构

LLaDA 采用 Transformer 架构,最具辨识度的改动是**移除因果掩码 (causal mask)**,从而获得双向注意力 (bidirectional attention)。这一设计使模型能够同时利用序列上下文左右两侧的信息,与传统自回归模型 (autoregressive model, ARM) 的单向注意力形成本质区别。

LLaDA-8B 的具体配置如下:

- 参数量:8B
- 层数 (layers):32
- 隐藏维度 (hidden dimension):4096
- 注意力头数 (attention heads):32
- 激活函数:SwiGLU
- 位置编码:RoPE (Rotary Position Embedding)

### 3.2 预训练数据

预训练共使用 2.3T tokens,语种与模态分布如下:

- English:61%
- Chinese:11%
- Code:28%

训练序列长度固定为 4096,其中 1% 的数据采用变长训练 (variable-length training),长度在 $[1, 4096]$ 区间内采样。变长数据的引入有助于模型适应不同长度的输入分布。

### 3.3 预训练目标与超参数

LLaDA 的预训练采用掩码预测 (masked prediction) 目标。给定输入序列,随机掩码部分 token,模型在双向注意力下预测被掩码的 token:

$$\mathcal{L}_{\text{PT}} = -\mathbb{E}_{t,x_0,x_t}\left[\frac{1}{t}\sum_{i=1}^{L}\mathbf{1}[x_t^i=M]\log p_\theta(x_0^i \mid x_t)\right]$$

其中 $\mathbf{1}[x_t^i=M]$ 指示位置 $i$ 在 $x_t$ 中是否被掩码,$\frac{1}{t}$ 加权是保证 MLE 合法性的关键(详见 2.5 节)。

预训练的计算开销与超参数如下:

- 计算量:0.13M H800 GPU hours
- 优化器:AdamW
- 权重衰减 (weight decay):0.1
- 批大小 (batch size):1280
- 学习率调度:Warmup-Stable-Decay

### 3.4 监督微调

SFT 数据集共 4.5M 条样本,其来源构成为:

- 1M 条人工标注 (human-annotated)
- 3.5M 条合成数据 (synthetic)

SFT 的掩码策略与预训练不同:**prompt 部分保持未掩码 (unmasked),仅对 response 部分的 token 进行掩码**。这使得模型在微调时学习如何在给定完整 prompt 的条件下生成回答:

$$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{t, p_0, r_0, r_t}\left[\frac{1}{t}\sum_{i=1}^{L'}\mathbf{1}[r_t^i=M]\log p_\theta(r_0^i \mid p_0,\, r_t)\right]$$

SFT 超参数:

- 训练轮数 (epochs):3
- 批大小:256

---

## 第4章 推理

LLaDA 的推理是一个**离散化的逆过程 (discretized reverse process)**:从完全掩码的响应出发,逐步去掩码 (unmask) 直至生成完整序列。

### 4.1 逆向生成过程

推理的初始化为**完全掩码的响应 (fully masked response)**,即响应部分的所有 token 均设为 [MASK]。随后在均匀时间步 (uniform timesteps) 上进行离散化的逆向生成:

$$x_K \sim \text{fully masked}, \quad x_0 = x, \quad p_\theta(x_{k-1} \mid x_k),\ \ k = 1, \dots, K$$

其中 $K$ 为推理步数,$x_K$ 为全掩码起点,$x_0$ 为最终生成的序列。在每个时间步,模型在双向注意力下并行预测所有掩码位置,并依据置信度决定哪些 token 被确定 (commit),哪些被重新掩码进入下一步。

### 4.2 低置信度重掩码策略

LLaDA 采用**低置信度重掩码 (low-confidence remasking)** 策略:在每个时间步并行预测后,仅保留高置信度 (即预测概率最高) 的若干 token 予以确定,将概率最低的 token 重新掩码,留待后续时间步再次预测。

设当前时间步需确定 $n_k$ 个 token,则模型对全部掩码位置并行预测后,选取概率最高的 $n_k$ 个 token 锁定,其余掩码位置维持掩码状态进入下一时间步。该策略使模型能够对不确定的 token 反复斟酌,从而提升整体生成质量。

### 4.3 灵活的推理模式

LLaDA 的一个重要优势是**无需重新训练 (no retraining needed)** 即可支持多种推理模式:

- **掩码扩散 (masked diffusion)**:默认模式,从全掩码出发逐步去掩码
- **自回归 (autoregressive)**:逐 token 生成,行为类似传统 ARM
- **块扩散 (block diffusion)**:以块 (block) 为单位生成
- **半自回归的块扩散 LLaDA (block diffusion LLaDA, semi-AR)**:块内并行、块间自回归的混合模式

这种灵活性使 LLaDA 能够在生成质量与推理效率之间灵活权衡。

### 4.4 似然评估

对于对数似然 (log-likelihood) 的评估,LLaDA 采用蒙特卡洛 (Monte Carlo, MC) 估计 (Eq.6),通过对 $l$ 个 token 进行均匀掩码 (uniform masking) 采样来近似:

$$-\log p_\theta(x_0) \approx \frac{L}{l}\sum_{i=1}^{L}\mathbf{1}[x_l^i=M]\log p_\theta(x_0^i \mid x_l), \quad l \sim \text{Uniform}\{1,2,\dots,L\}$$

其中 $l$ 从 $\{1,2,\dots,L\}$ 中均匀采样,$x_l$ 通过对 $x_0$ 均匀随机掩码 $l$ 个 token 得到,$L$ 为序列长度。$\frac{L}{l}$ 因子用于补偿不同掩码比例下的期望偏差。

---

## 第5章 实验

本章从基础模型 (base model)、指令模型 (instruct model)、反转推理 (reversal reasoning)、可扩展性 (scalability) 以及案例分析 (case study) 五个维度评估 LLaDA。

### 5.1 基础模型对比

在基础模型设定下,LLaDA-8B 与 LLaMA3-8B、LLaMA2-7B 在五项任务上的对比如下:

| Task | LLaDA 8B | LLaMA3 8B | LLaMA2 7B |
|------|----------|-----------|-----------|
| MMLU (5-shot) | 65.9 | 65.4 | 45.9 |
| GSM8K (4-shot) | 70.3 | 48.7 | 13.1 |
| Math (4-shot) | 31.4 | 16.0 | 4.3 |
| HumanEval (0-shot) | 35.4 | 34.8 | 12.8 |
| CMMLU (5-shot) | 69.9 | 50.7 | 32.5 |

关键发现:

- **总体匹配 LLaMA3-8B**:在 MMLU、HumanEval 上与 LLaMA3-8B 基本持平 (65.9 vs 65.4;35.4 vs 34.8),并在多项任务上反超。
- **数学推理显著领先**:GSM8K 上 70.3 vs 48.7,Math 上 31.4 vs 16.0,LLaDA 接近 LLaMA3 的两倍。这一优势可能源于双向注意力对解题过程的整体规划能力。
- **中文能力突出**:CMMLU 上 69.9 vs 50.7,领先近 20 分。
- **全面超越 LLaMA2-7B**:在全部任务上大幅领先。

### 5.2 指令模型对比

指令模型的对比均在**仅使用 SFT、不含 RL** 的设定下进行,以保持对比条件可控:

| Task | LLaDA 8B | LLaMA3 8B | LLaMA2 7B |
|------|----------|-----------|-----------|
| MMLU (5-shot) | 65.5 | 68.4 | 44.1 |
| ARC-C (0-shot) | 88.5 | 82.4 | 57.3 |
| GSM8K (4-shot) | 69.4 | 78.3 | 29.0 |
| HumanEval (0-shot) | 49.4 | 59.8 | 16.5 |

关键发现:

- **ARC-C 领先**:88.5 vs 82.4,LLaDA 超过 LLaMA3。
- **部分任务落后**:MMLU (65.5 vs 68.4)、GSM8K (69.4 vs 78.3)、HumanEval (49.4 vs 59.8) 上 LLaDA 落后于 LLaMA3。这表明在纯 SFT 设定下,LLaDA 在部分任务上的正向 (forward) 能力仍有提升空间 (详见 6.3)。

### 5.3 反转推理

反转推理 (reversal reasoning) 评估模型在"逆向"任务上的能力,是 LLaDA 双向架构的核心优势所在:

| Model | Forward | Reversal |
|-------|---------|----------|
| GPT-4o | 82.7 | 34.3 |
| Qwen2.5-7B Instruct | 75.9 | 38.0 |
| LLaDA-8B Instruct | 51.8 | 45.6 |

关键发现:

- **ARM 在反向任务上急剧退化**:GPT-4o 正向 82.7 但反向仅 34.3,跌幅巨大;Qwen2.5-7B Instruct 同样存在严重的正向-反向差距 (75.9 → 38.0)。
- **LLaDA 正反向更均衡**:LLaDA-8B Instruct 的反向成绩 45.6 为三者最高,且正反向差距最小 (51.8 vs 45.6)。这验证了双向注意力在反转推理上的结构优势。
- **代价**:LLaDA 的正向成绩 (51.8) 低于 GPT-4o 和 Qwen2.5,体现了 forward accuracy 与 reversal balance 之间的权衡。

### 5.4 可扩展性

- LLaDA 可扩展至 $10^{23}$ FLOPs 的计算规模,在 6 项任务上匹配自回归基线 (ARM baselines)。
- 在 1B 规模上,LLaDA 与 ARM 采用**完全相同的架构**,便于直接比较;8B 规模因资源限制未构建相同架构的 ARM 基线。
- 随着模型规模增大,LLaDA 与 ARM 的性能差距逐渐缩小 (gap narrows with scale),表明 masked diffusion 路线具有良好的 scaling 特性。

### 5.5 案例分析

LLaDA 在以下实际场景中进行了案例分析:

- **多轮对话 (multi-turn dialogue)**
- **翻译 (translation)**
- **受限生成 (constrained generation)**

其中,受限生成场景尤为能体现掩码扩散模型的优势:由于模型可以并行地、迭代地确定 token 并对低置信位置反复修订,在需要同时满足多处约束的生成任务上,LLaDA 的灵活性优于严格从左到右生成的 ARM。

---

## 第6章 局限性

尽管 LLaDA 展示了 masked diffusion 的潜力,仍存在以下局限。

### 6.1 推理步数开销

扩散式生成需要**多次推理步 (multiple inference steps)** 才能完成一次生成,而 ARM 每生成一个 token 仅需单次前向传播 (single forward pass)。在相同生成量下,LLaDA 的推理开销更高,实时性不及 ARM。

### 6.2 缺乏 RL 对齐

本文**尚未探索 RL 对齐 (RL alignment)**,包括 RLHF 与 DPO 等方法。当前指令模型仅依赖 SFT,这可能是部分任务正向成绩落后于强 ARM 的原因之一。未来引入 RLHF/DPO 有望进一步提升对齐质量与指令遵循能力。

### 6.3 部分任务正向精度不足

在某些任务上,LLaDA 的正向精度 (forward accuracy) **低于强 ARM** (见 5.2 指令模型对比中的 MMLU、GSM8K、HumanEval)。这表明 masked diffusion 在正向生成能力上仍有改进空间。

### 6.4 块扩散的复杂度权衡

块扩散 (block diffusion) 模式能够在数学与代码任务上带来质量提升,但**同时增加了实现的复杂度 (complexity)**。使用者需在质量收益与工程复杂度之间进行权衡。

### 6.5 训练时显存开销

双向注意力 (bidirectional attention) 在**训练阶段具有更高的显存开销 (memory cost)**:由于无法像 ARM 那样借助因果掩码进行高效的 KV cache 与梯度重计算优化,双向模型的训练对大规模工程实现提出了更高要求。

