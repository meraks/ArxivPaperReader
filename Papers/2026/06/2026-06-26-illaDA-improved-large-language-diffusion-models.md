# iLLaDA: Improved Large Language Diffusion Models — 深度研读报告

## 论文元数据

- 标题：Improved Large Language Diffusion Models (iLLaDA)
- 作者：Shen Nie, Qiyang Min, Shaoxuan Xu et al. (Renmin University of China, ByteDance Seed)
- arXiv ID：2606.25331
- 提交日期：2026-06-24
- 官方代码：https://github.com/ML-GSAI/LLaDA
- 代码发现方式：来自论文原文

---

## Ch1: 论文概述与核心贡献

### 1.1 论文定位

iLLaDA 是一个 8B 参数的 masked diffusion language model（掩码扩散语言模型），从零开始训练，全程使用 fully bidirectional attention（全双向注意力）。与主流自回归 LLM（采用 causal attention 与 left-to-right 因子分解）不同，iLLaDA 在预训练与监督微调（SFT）全流程保持 masked diffusion objective。

关键规模数据：

- 预训练：12T tokens（较 LLaDA 的 2.3T 扩大约 5 倍）
- SFT：25B tokens 的指令语料，训练 12 epochs
- 参数量：总参数 7.62B，非嵌入参数 6.98B

### 1.2 核心改进清单

iLLaDA 相对 LLaDA 的改进集中在架构、训练规模与推理策略三方面：

| 改进类别 | 具体内容 |
|---------|---------|
| 架构 | GQA（8 KV heads）、tied embedding & LM-head、扩大 FFN、扩大 vocab、延长上下文 |
| 训练规模 | 预训练 2.3T → 12T tokens |
| 推理策略 | confidence-based scoring（多项选择评估）、variable-length generation（开放生成） |
| SFT | 多 epoch 训练（12 epochs），验证数据重用效应 |

### 1.3 Base 模型性能

iLLaDA-Base 在 8 项基准上平均 63.9，超越 LLaDA-Base（51.1，+12.8）与 Dream 7B（61.4），并持平 Qwen2.5 7B（63.3）。这是非自回归模型首次在平均分上与同规模强自回归 baseline 持平。

单项亮点：

- MMLU：74.8（LLaDA 65.9，Qwen2.5 71.9）
- BBH：71.3（+21.6 vs LLaDA）
- ARC-C：60.8（+14.9 vs LLaDA）
- GSM8K：81.9（Qwen2.5 78.9）

### 1.4 Instruct 模型性能

iLLaDA-Instruct 平均 67.1，超越 LLaDA-Instruct（54.5）与 Dream 7B Instruct（60.2），但落后于 Qwen2.5 7B Instruct（77.1）。MMLU-Redux 上 iLLaDA 取得所有对比模型中最高分 76.4。Instruct 与 Qwen2.5 的差距主要归因于 iLLaDA 缺少 RL 对齐阶段。单项亮点：MATH 56.7（+14.5 vs LLaDA）、HumanEval 65.9（+16.5 vs LLaDA）。

### 1.5 贡献小结

论文核心信息：fully bidirectional diffusion training from scratch 在适当 scaling 与架构优化后，可在多个基准上与强自回归 baseline 竞争，证明扩散语言模型是一条可行的替代路径，而非仅具理论意义的探索。

---

## Ch2: 研究背景与动机

### 2.1 自回归 LLM 的主导地位

现代 LLM 主要基于自回归因子化（autoregressive factorization）与 causal attention：模型按从左到右的顺序逐 token 预测，每个 token 仅条件于其左侧上下文。GPT 系列、LLaMA、Qwen 等均采用 next-token prediction 范式。该范式在 scaling、in-context learning 与 instruction following 上已被充分验证，是当前 LLM 的主流技术路线。

### 2.2 扩散语言模型作为替代路径

扩散模型在图像生成领域取得巨大成功后，研究者尝试将其迁移至离散文本领域，形成两条主要路线：

- **离散扩散（discrete diffusion）**：将文本生成建模为从噪声到数据的逐步去噪过程，噪声与数据均处于离散词表空间。
- **Masked diffusion**：以 `[MASK]` 作为噪声符号，模型学习从部分掩码序列恢复完整序列。生成时从全掩码出发，逐步去掩码得到文本。

masked diffusion 的关键特征是 **fully bidirectional attention**：每个位置的预测都可同时利用左右两侧上下文，这与自回归的 causal mask 形成本质区别。

### 2.3 LLaDA 的里程碑意义

LLaDA 是首个从零训练的 8B masked diffusion LM，其贡献包括：

- 证明 bidirectional diffusion 可以习得核心 LLM 能力（in-context learning、instruction following）
- 揭示扩散 LM 相对自回归模型的潜在优势：reversal/bidirectional reasoning（反向推理）、long-horizon planning（长程规划）、multimodal 兼容性

这些优势源于双向注意力：模型在生成每个 token 时都能利用全局信息，而非仅依赖已生成的前缀。

### 2.4 LLaDA 的不足

LLaDA 虽具开创性，但存在明显短板：

- **性能落后**：在多数基准上落后于同规模自回归模型（Qwen2/2.5）
- **训练 token 量少**：仅 2.3T，远低于当代强 LLM 的训练规模
- **架构未优化**：采用 MHA 而非 GQA（32 KV heads）、untied embedding/LM-head，参数效率与推理效率均有改进空间

### 2.5 iLLaDA 的目标

iLLaDA 的目标是通过 scaling 与架构改进弥合扩散 LM 与自回归 LM 的性能差距，同时保留扩散模型的独特优势（bidirectional reasoning 等）。具体策略：

1. 扩大训练规模（2.3T → 12T tokens），验证扩散模型的 scaling 特性
2. 采用更高效的架构（GQA、tied embedding），在不缩水模型容量的前提下提升参数与推理效率
3. 改进推理策略（confidence scoring、variable-length generation），适配扩散模型的生成范式
4. 延长 SFT 训练（12 epochs），探索数据重用对扩散 LM 的影响

---

## Ch3: iLLaDA 模型架构与预训练

### 3.1 架构总览

iLLaDA 采用 dense Transformer，组件包括 RMSNorm、SwiGLU、RoPE（旋转位置编码）、无 bias。下表对比 iLLaDA 8B 与 LLaDA 8B 的架构差异（Table 1）：

| Feature | iLLaDA 8B | LLaDA 8B |
|---------|-----------|----------|
| Layers | 32 | 32 |
| Model dim | 4096 | 4096 |
| Attention heads | 32 | 32 |
| Key/Value heads | 8 | 32 |
| FFN dim | 14,336 | 12,288 |
| Vocab size | 155,136 | 126,464 |
| Max seq len | 8192 | 4096 |
| Embedding/LM-head | Tied | Untied |
| Total params | 7.62B | 8.02B |
| Non-embedding params | 6.98B | 6.98B |

两者层数、模型维度、注意力头数与非嵌入参数完全相同（6.98B），差异集中在 KV head 数、FFN 维度、词表、上下文长度与 embedding 绑定方式。

### 3.2 架构改进解读

**GQA（Grouped-Query Attention，8 KV heads）**：iLLaDA 将 KV heads 从 32 减至 8（query heads 32 不变），即每 4 个 query head 共享 1 组 KV。效果：KV cache 规模缩小约 4×，降低推理时显存占用与带宽压力。这对扩散模型尤为重要——扩散生成需多轮迭代去噪，每次前向都要重新计算或复用 KV，KV cache 的缩减直接降低推理成本。

**Tied embedding & LM-head**：输入 embedding 矩阵与输出 LM-head 共享权重。由于词表扩张（126,464 → 155,136，+22.7%），若仍 untied 则 embedding 参数会进一步膨胀。tied 设计将总参数量从 8.02B 降至 7.62B（−0.40B），同时保持非嵌入参数 6.98B 不变。这意味着 transformer 主体（决定模型容量的核心）未缩水，缩减的仅是 embedding 矩阵的冗余。

**扩大 FFN（12,288 → 14,336）**：FFN 维度提升约 16.7%，增加每层的非线性表达容量。FFN 是 Transformer 中存储知识与特征变换的主要部位，扩大它直接提升模型容量。

**扩大 vocab（126,464 → 155,136）**：更大的词表覆盖更多 token，提升多语言与代码场景下的 tokenization 效率，使序列更紧凑，同等 token 预算下承载更多信息。

**延长上下文（4096 → 8192）**：最大序列长度翻倍，支持更长上下文输入与更长 SFT 响应输出。

### 3.3 预训练目标

iLLaDA 预训练采用 masked diffusion objective。给定完整序列 $x_0$，按掩码比例 $t$ 随机将部分 token 替换为 `[MASK]` 得到 $x_t$，模型学习从 $x_t$ 恢复被掩码的原始 token：

$$
\mathcal{L}(\theta) = -\mathbb{E}_{t,x_0,x_t} \left[ \frac{1}{t} \sum_{i=1}^L \mathbf{1}[x_t^i=M] \log p_\theta(x_0^i \mid x_t) \right]
$$

其中：

- $t \in (0,1]$ 为掩码比例（扩散时间步），$t$ 越大被掩码的 token 越多
- $\mathbf{1}[x_t^i=M]$ 为指示函数，位置 $i$ 被掩码时取 1
- 损失仅在被掩码位置计算，归一化系数 $\frac{1}{t}$ 补偿被掩码 token 的期望数量
- $p_\theta(x_0^i \mid x_t)$ 由双向 Transformer 在整条序列上联合预测

与自回归 loss 的根本区别：自回归 loss 中位置 $i$ 的预测仅条件于左侧 $\{x_0^1,\dots,x_0^{i-1}\}$；masked diffusion loss 中位置 $i$ 的预测条件于全序列 $x_t$（含左右两侧的非掩码 token）。这使得每个 token 的学习都能利用双向上下文，是扩散 LM 与自回归 LM 在建模层面的核心分野。

### 3.4 预训练设置

- **LR schedule**：linear warmup 至 $2\times10^{-4}$，进入恒定阶段，再 cosine decay 至 $5\times10^{-6}$
- **优化器**：AdamW，weight decay $0.1$
- **数据打包**：variable-length batching，配合 FlashAttention 加速
- **序列长度分配**：30% 的样本采用 random-length split（区别于常规的固定长度截断打包）

30% random-length split 的作用：在常规固定长度打包（所有样本截断/填充到 max length）之外，引入长度多样性。这避免了模型对固定截断边界的过拟合，提升对变长真实序列的适应能力——这与后续的 variable-length generation 推理形成训练-推理对齐。

FlashAttention 配合 variable-length batching，让不同长度的样本高效拼批，减少 padding 浪费，在双向注意力场景下保持计算效率。

---

## Ch4: 监督微调与推理策略

### 4.1 SFT 数据处理

iLLaDA 的 SFT 沿用与预训练完全相同的 masking 方案：将 prompt + response + EOS 视为连续语料，采样 8192-token 序列，对整条序列（而非仅 response 部分）施加随机掩码。模型仍以 masked diffusion objective 学习恢复被掩码 token。

这一设计让 SFT 在扩散框架内自然进行，无需切换为自回归目标。与自回归 SFT 的区别在于：自回归 SFT 中 prompt 部分（teacher-forcing）与 response 部分（loss 计算）严格分离；masked diffusion SFT 中整条序列统一掩码、统一计算 loss，prompt 与 response 的边界由掩码模式隐式处理。

### 4.2 SFT 训练设置

- **数据规模**：约 25B tokens 的指令语料
- **训练轮数**：12 epochs
- **LR schedule**：warmup 至 $5\times10^{-6}$，恒定阶段，再 linear decay 至 $5\times10^{-7}$

12 epochs 的多轮训练验证了"数据重用效应"：同一指令语料多次复用仍能持续提升性能（见 Ch5 Fig 1 分析）。这暗示扩散 LM 对指令数据的利用方式与自回归模型有所不同——多轮曝光非但未导致严重过拟合，反而持续受益。

### 4.3 Confidence-based Scoring（多项选择评估）

扩散模型在多项选择题上的标准做法是 likelihood-based scoring：对每个候选答案，计算其在模型下的对数似然，选最高者。iLLaDA 提出基于置信度的替代方案。

**问题动机**：扩散模型评估一个候选答案时，需要从掩码状态逐步去噪揭示。likelihood scoring 按固定顺序（如从左到右）揭示 token 并累加似然；但这种顺序未必匹配模型自身的置信度分布，可能在高不确定位置强行赋值，引入噪声。

**置信度揭示顺序**：在第 $k$ 步，从当前仍被掩码的位置集 $\mathcal{M}_{k-1}$ 中，选择模型预测置信度最高的位置优先揭示：

$$
i_k = \arg\max_{i \in \mathcal{M}_{k-1}} p_\theta(y^i \mid p, \tilde{y}_{k-1}), \quad S_{\mathrm{conf}}(y \mid p) = \sum_{k=1}^L \log p_\theta(y^{i_k} \mid p, \tilde{y}_{k-1})
$$

其中 $p$ 为 prompt，$\tilde{y}_{k-1}$ 为截至第 $k-1$ 步已揭示的部分，$y^{i_k}$ 为候选答案在位置 $i_k$ 的 token。每步揭示当前最自信的位置后，更新上下文再选下一个最自信位置，最终将所有位置的对数概率累加作为该候选的总分 $S_{\mathrm{conf}}$。

置信度打分确定性地优先处理模型最有把握的 token，逐步积累证据，让后续低置信度位置的预测建立在已确认的高置信上下文之上。

消融（Table 4）显示 confidence scoring 在 PIQA（78.5 vs 77.2）、ARC-C（60.8 vs 60.2）、Hellaswag（76.6 vs 74.3）上一致优于 likelihood scoring，差距分别为 +1.3、+0.6、+2.3。

### 4.4 Variable-length Block Generation（开放文本生成）

**问题**：扩散模型生成需预先设定输出长度（掩码 token 数量），无法像自回归模型那样在生成 EOS 时自然停止。固定长度会导致输出要么被截断、要么填充无意义 token。

**iLLaDA 的分块迭代方案**：

1. 在当前序列后追加一个 mask block（一批 `[MASK]` token）
2. 对掩码块运行扩散采样器
3. 采用 low-confidence remasking（MaskGIT 风格）：每轮仅揭示当前置信度最高的 token，低置信度 token 重新保持掩码，进入下一轮细化
4. 检查 EOS：若采样出 EOS token，则在该位置停止生成
5. 否则追加新的 mask block，回到步骤 2 继续生成

```python
# ⚠️ 非官方概念实现，未经验证（论文 Sec 2.3 流程示意）
def variable_length_generate(model, prompt, block_size, max_blocks):
    sequence = prompt
    for _ in range(max_blocks):
        sequence = sequence + [MASK] * block_size       # 步骤1: 追加掩码块
        sequence = diffusion_sample(model, sequence)     # 步骤2-3: 采样 + low-confidence remasking
        if contains_eos(sequence):                       # 步骤4: 检查EOS
            return truncate_at_eos(sequence)
    return sequence                                      # 达到上限仍未出EOS
```

该机制让扩散模型能像自回归模型一样生成不定长文本：通过 block 迭代扩展长度，通过 EOS 检测自然停止，通过置信度自适应决定每轮揭示节奏。low-confidence remasking 保证生成质量——低置信度位置反复重掩码、反复细化，直至模型收敛到稳定输出。

---

## Ch5: 实验结果与分析

### 5.1 Base 模型对比（Table 2）

四个 7-8B 模型在 8 项基准上的对比：

| Task | iLLaDA 8B | LLaDA 8B | Dream 7B | Qwen2.5 7B |
|------|-----------|----------|----------|-------------|
| MMLU | 74.8 | 65.9 | 69.5 | 71.9 |
| BBH | 71.3 | 49.7 | 57.9 | 63.9 |
| ARC-C | 60.8 | 45.9 | 59.8 | 51.5 |
| Hellaswag | 76.6 | 70.5 | 73.3 | 79.0 |
| GSM8K | 81.9 | 70.3 | 77.2 | 78.9 |
| MATH | 38.4 | 31.4 | 39.6 | 41.1 |
| HumanEval | 50.0 | 35.4 | 57.9 | 56.7 |
| MBPP | 57.8 | 40.0 | 56.2 | 63.6 |
| Average | 63.9 | 51.1 | 61.4 | 63.3 |

**关键发现**：

- iLLaDA-Base 在 MMLU（74.8）、BBH（71.3）、ARC-C（60.8）、GSM8K（81.9）四项上超越 Qwen2.5 7B
- 平均分 63.9，略高于 Qwen2.5 7B（63.3，+0.6）——非自回归模型首次在平均分上达到强自回归 baseline 水平
- 相对 LLaDA-Base 全面领先（平均 +12.8），尤其在 BBH（+21.6）与 ARC-C（+14.9）上提升巨大
- 相对 Dream 7B 平均领先 +2.5
- 在代码任务上 iLLaDA 仍有差距：HumanEval 50.0 弱于 Dream 7B（57.9）与 Qwen2.5（56.7），MBPP 57.8 弱于 Qwen2.5（63.6）
- MATH（38.4）是少数 iLLaDA 未领先 Qwen2.5（41.1）的推理任务

### 5.2 Instruct 模型对比（Table 3）

| Task | iLLaDA 8B | LLaDA 8B | Dream 7B | Qwen2.5 7B |
|------|-----------|----------|----------|-------------|
| MMLU | 71.6 | 65.5 | 67.0 | 76.6 |
| MMLU-Pro | 52.3 | 37.0 | 43.3 | 56.3 |
| MMLU-Redux | 76.4 | 68.9 | 76.3 | 75.7 |
| GSM8K | 89.0 | 77.5 | 81.0 | 91.6 |
| MATH | 56.7 | 42.2 | 39.2 | 75.5 |
| HumanEval | 65.9 | 49.4 | 55.5 | 84.8 |
| MBPP | 58.0 | 41.0 | 58.8 | 79.2 |
| Avg. | 67.1 | 54.5 | 60.2 | 77.1 |

**关键发现**：

- iLLaDA-Instruct 平均 67.1，超越 LLaDA-Instruct（54.5，+12.6）与 Dream 7B Instruct（60.2，+6.9），在扩散模型中最强
- MMLU-Redux 上 iLLaDA 取得 76.4，为所有对比模型（含 Qwen2.5 75.7）中最高
- 数学任务显著进步：GSM8K 89.0、MATH 56.7（+14.5 vs LLaDA）
- 但总体仍落后 Qwen2.5 7B Instruct（77.1，−10.0）
- 代码任务差距最大：HumanEval 65.9 vs Qwen2.5 84.8（−18.9）、MBPP 58.0 vs 79.2（−21.2）

### 5.3 Instruct Gap 归因

iLLaDA-Instruct 与 Qwen2.5 的差距（67.1 vs 77.1，−10.0）主要由对齐方法差异导致：

- Qwen2.5 经历完整的 SFT + RL 对齐（如 RLHF/RLAIF），RL 阶段显著提升指令遵循与推理质量
- iLLaDA 仅进行 SFT，未引入任何 RL 阶段
- 论文将这一 gap 明确归因于 RL 对齐的缺失，而非扩散范式本身的局限

证据：iLLaDA-Base 已能与 Qwen2.5 持平（63.9 vs 63.3），说明扩散范式在基础能力上不逊色；gap 主要出现在经对齐的 Instruct 阶段。这指向"补齐 RL 对齐"作为最直接的改进方向。

### 5.4 消融研究：Scoring Rules（Table 4）

| Scoring | PIQA | ARC-C | Hellaswag |
|---------|------|-------|-----------|
| Likelihood | 77.2 | 60.2 | 74.3 |
| Confidence | 78.5 | 60.8 | 76.6 |

confidence scoring 在全部三项上一致优于 likelihood scoring，差距 +1.3 / +0.6 / +2.3。其中 Hellaswag 提升最大（+2.3），该任务对全局语义一致性敏感，置信度揭示顺序能更好捕捉整体证据。该改进无需额外训练，仅改变推理时的打分策略。

### 5.5 SFT Epoch 消融（Fig 1）

随 SFT epochs 增加，GSM8K、MATH、MMLU-Pro 性能持续提升至 12 epochs，曲线未见明显饱和。这表明：

- 指令数据在扩散 LM 上的多轮复用有效，支持将 SFT 训练延长至 12 epochs
- 扩散 LM 对数据重用的响应不同于自回归模型（后者往往在少数 epoch 后出现过拟合）

|Fig 1 显示 GSM8K 从~86%提升至~90%、MATH 从~48%提升至~57%、MMLU-Pro 从~50%提升至~52%（趋势近似，基于曲线目视）。该持续提升曲线印证了扩散 LM 在 SFT 阶段的数据重用效应。

### 5.6 评估细节

- 多项选择任务（MMLU、BBH、ARC-C、Hellaswag、PIQA 等）采用 confidence-based scoring
- 生成任务（GSM8K、MATH、HumanEval、MBPP）采用 variable-length block generation
- 不同任务设置不同的 block 长度：BBH/GSM8K/MATH/MBPP 采用 block length 32（max length 1024），HumanEval 采用 block length 512（max length 512）。Instruct 模型设置有差异：GSM8K/HumanEval block=32 (max 2048)、MMLU-Pro/MATH block=32 (max 4096)、MBPP block=16 (max 2048)、MMLU-Redux block=3-4 (max 3-4)
- **推理 loop 缓解**：扩散模型在长链推理时可能出现重复推理（repetitive reasoning）现象，论文通过递增 `</think>` token 的出现概率来缓解，鼓励模型尽早终止推理 trace 并输出最终答案

---

## Ch6: 代码实现详解

> 本章内容主要来自仓库 README 与工程文档（ML-GSAI/LLaDA），用于说明代码实现与工程实践，不代表论文的科学声明。

### 6.1 仓库概述

iLLaDA 的官方代码与 LLaDA 共用同一仓库 ML-GSAI/LLaDA（https://github.com/ML-GSAI/LLaDA）。注意：**训练框架与训练数据未开源**，仓库主要提供推理代码与工程指南。

关键文件：

- `generate.py`：基于 variable-length blocks 的条件生成
- `get_log_likelihood.py`：似然估计（多项选择评估的基础）
- `chat.py`：多轮对话接口
- `app.py`：Gradio 演示界面
- `GUIDELINES.md`：训练工程指南（替代未开源的训练代码）

### 6.2 模型加载

官方 README 提供的加载方式（来源：仓库 README API 说明）：

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    'GSAI-ML/LLaDA-8B-Base',
    trust_remote_code=True,        # 需执行仓库自带的自定义 modeling 代码
)
# 依赖：transformers==4.38.2
```

`trust_remote_code=True` 是必需的，因为 LLaDA/iLLaDA 的 masked diffusion 建模逻辑不在标准 transformers 中，需加载仓库提供的自定义模型类。

### 6.3 关键推理函数

仓库提供三个核心推理接口（来源：仓库 README API 说明）：

- **`get_log_likelihood()`**：给定 prompt 与候选文本，返回该候选在模型下的对数似然，用于多项选择评估（likelihood scoring 的基础）
- **`generate()`**：variable-length 条件生成，支持 mask block 迭代与 low-confidence remasking
- **`chat`（chat.py）**：多轮对话封装，维护对话历史并调用生成接口

variable-length 生成的实现思路（概念性示意，非源码）：

```python
# ⚠️ 非官方概念实现，未经验证
# generate.py 核心循环思路
def generate(model, prompt, block_size, max_iters):
    output = prompt + [MASK] * block_size
    while iters < max_iters:
        logits = model(output)                            # 预测所有掩码位置
        output = reveal_top_k_confident(logits, output)   # 高置信度揭示，低置信度重掩码
        if has_eos(output):
            return truncate_at_eos(output)
        output = output + [MASK] * block_size             # 追加新块
    return output
```

confidence scoring 的实现思路（概念性示意，非源码）：

```python
# ⚠️ 非官方概念实现，未经验证
def confidence_score(model, prompt, candidate):
    score = 0.0
    masked = list(candidate)
    while has_masked(masked):
        probs = model(prompt + masked)                    # 预测所有掩码位置分布
        best_pos = argmax_over_masked(probs)              # 选当前最自信位置
        score += log(max(probs[best_pos]))                # 累加对数概率
        masked[best_pos] = argmax_token(probs[best_pos])  # 揭示该位置
    return score
```

### 6.4 训练代码参考

训练框架未开源，但同组的 SMDM 项目（https://github.com/ML-GSAI/SMDM）提供了相似的 masked diffusion 训练 pipeline，可作为复现参考。SMDM 聚焦于 masked diffusion 的 scaling，其开源训练代码与 iLLaDA 的训练流程同源，是理解 iLLaDA 预训练实现的最佳公开资源。仓库的 `GUIDELINES.md` 也给出了训练工程的实践指南。

### 6.5 训练稳定性

据仓库工程记录，预训练全程仅出现一次 crash（发生在约 1.2T tokens 处），通过将学习率从 $4\times10^{-4}$ 降至 $1\times10^{-4}$ 后解决，训练恢复后稳定完成 12T tokens 规模。

这一记录表明 masked diffusion 在大规模训练下具有良好的稳定性——全程 12T tokens 仅一次中断，且通过常规的 LR 调整即可恢复，未出现需要架构级干预的训练崩溃。注意：此处 $4\times10^{-4}$ → $1\times10^{-4}$ 是 crash 后的工程调整值（来自仓库工程记录），与论文 Sec 2.1 报告的主训练 LR（warmup 至 $2\times10^{-4}$）属不同来源。

---

## Ch7: 局限性与延伸阅读

### 7.1 局限性

1. **缺少 RL 对齐**：iLLaDA 仅用 SFT，未引入 RL 阶段。这是 Instruct 模型落后 Qwen2.5（67.1 vs 77.1，−10.0）的主要原因。论文明确将这一 gap 归因于 RL 缺失，而非扩散范式局限——iLLaDA-Base 已能与 Qwen2.5 持平（63.9 vs 63.3）即为佐证。

2. **规模限制**：仅在 8B 规模验证，缺乏与自回归模型在完全匹配条件下的对照（matched comparison）。现有对比（vs Qwen2.5 7B、Dream 7B）虽规模相近，但训练数据、词表、训练 token 量等条件并非完全对齐，难以精确归因性能差异的来源。

3. **推理 loop 问题**：扩散模型在长链推理时可能出现重复推理（repetitive reasoning），现有缓解手段（递增 `<think>` token 概率）仍属启发式，未从根本上解决。这影响了 Instruct 模型在复杂多步推理任务上的表现。

### 7.2 未来方向

- **RL 对齐**：直接可用的扩散模型 RL 方法包括 VRPO（LLaDA 1.5 已采用）、diffu-GRPO、MDPO、ESPO。引入 RL 预计可显著缩小 Instruct gap，是当前最明确的改进路径。
- **更大规模实验**：在 8B 之上继续 scaling，系统验证扩散 LM 的 scaling law，并与同规模自回归模型做 matched comparison。
- **KV cache 优化**：扩散推理的多轮迭代对 KV cache 敏感。dkv-cache、EntropyCache 等针对扩散模型设计的专用缓存优化，有望进一步降低推理成本，释放 GQA 之外的效率潜力。

### 7.3 延伸阅读

- **LLaDA（原始论文）**：Nie et al., NeurIPS 2026 —— 首个从零训练的 8B 扩散 LM，iLLaDA 的直接前作，奠定了 masked diffusion LM 的技术基础。
- **Dream 7B**：Jiacheng Ye et al., 2025 —— 另一 7B 级扩散 LM，本文实验的重要对比对象，代表了扩散 LM 的另一技术路线。
- **SMDM**：同组的 masked diffusion scaling 工作（github.com/ML-GSAI/SMDM），提供开源训练代码，是理解扩散 LM 大规模训练实现的参考。
- **LLaDA-V**：多模态扩散 LM，展示扩散范式向视觉扩展的潜力，呼应扩散 LM 在 multimodal 兼容性上的潜在优势。
