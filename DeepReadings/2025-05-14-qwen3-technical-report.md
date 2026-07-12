# Qwen3 Technical Report 技术报告解读

> **论文**：Qwen3 Technical Report · **作者**：Qwen Team, Alibaba (An Yang et al.) · **arXiv**：2505.09388 · **日期**：2025-05-14 · **许可**：Apache 2.0

---

## 第 1 章 概述

### 1.1 模型系列定位

Qwen3 是 Qwen 系列的最新一代大语言模型，一次性发布了 **8 个模型**，覆盖从端侧到旗舰的全尺度：

- **6 个 Dense 模型**：Qwen3-0.6B、Qwen3-1.7B、Qwen3-4B、Qwen3-8B、Qwen3-14B、Qwen3-32B（参数规模 0.6B ~ 32B）；
- **2 个 MoE 模型**：Qwen3-30B-A3B、Qwen3-235B-A22B。

旗舰模型为 **Qwen3-235B-A22B**：总参数 **235B**，每 token 激活参数仅 **22B**，在保持大模型容量的同时大幅降低推理计算成本。全部模型均以 **Apache 2.0** 协议开源。

### 1.2 核心创新：统一的 Thinking 模式

Qwen3 最关键的贡献是将 **thinking mode（思考模式）** 与 **non-thinking mode（非思考模式）** 统一进同一个模型：

- **thinking mode**：模型先展开逐步推理链（chain-of-thought）再作答，擅长数学、编程等复杂推理任务；
- **non-thinking mode**：模型直接快速作答，适合日常对话与高吞吐场景；
- 通过 **thinking budget（思考预算）** 可控制推理 token 的开销，在响应质量与延迟之间灵活权衡。

这种统一设计避免了分别维护"推理模型"与"通用模型"两套权重的开销，单一模型即可兼顾两种场景。

### 1.3 预训练规模

- 预训练数据量：**36T tokens**；
- 支持语言：**119 种**（Qwen2.5 仅 29 种），多语言覆盖显著扩展。

### 1.4 关键结果

旗舰模型 Qwen3-235B-A22B（thinking mode）的代表性基准成绩：

| 基准 | 得分 |
|------|------|
| AIME'24 | 85.7 |
| AIME'25 | 81.5 |
| LiveCodeBench v5 | 70.7 |
| CodeForces | 2,056 |
| BFCL v3 | 70.8 |

值得强调的是，在后训练的 reasoning RL 阶段，AIME'24 成绩仅用 **170 个 RL steps** 即从 **70.1 提升到 85.1**，体现了 RL 对推理能力的高效增益。

---

## 第 2 章 背景

### 2.1 开源大模型格局

2024–2025 年开源大模型快速迭代，代表性工作包括 **Llama-4、DeepSeek-V3、Qwen2.5** 等。模型架构逐渐趋同——GQA + RoPE + SwiGLU + RMSNorm 已成为事实标准，竞争重心转向数据质量、训练规模与后训练方法。

### 2.2 从 Qwen2.5 到 Qwen3

Qwen3 在前代 Qwen2.5 基础上的关键演进：

- **数据规模翻倍**（data scale ×2）：预训练 token 量大幅扩展至 36T；
- **语言覆盖扩大**：从 29 种扩展到 **119 种**语言；
- **引入 thinking mode**：原生支持推理 / 非推理双模式与 thinking budget。

### 2.3 Thinking 模型的兴起

OpenAI **o1**、**o3-mini** 与 **DeepSeek-R1** 等"思考型"模型证明了：通过 RL 与长链推理（long chain-of-thought），模型在数学与编程上可获得显著提升。但这些早期 thinking 模型通常与通用对话模型**分离部署**，部署与维护成本较高。

### 2.4 Qwen3 的差异化定位

Qwen3 将 thinking 与 non-thinking 统一进单一模型，并通过 thinking budget 让用户按需调节推理深度——既保持了通用对话的快速响应，又能在复杂任务上自动展开深度推理，解决了以往两类模型割裂的问题。

---

## 第 3 章 模型架构

### 3.1 模型族概览

Qwen3 延续 Dense + MoE 双路线，共 8 个模型。骨干沿用主流 Transformer 设计，并在归一化与注意力细节上做了改进。

#### Dense 模型配置

| Models | Layers | Heads (Q/KV) | Tie Embedding | Context |
|--------|--------|-------------|:-------------:|:-------:|
| Qwen3-0.6B | 28 | 16/8 | Yes | 32K |
| Qwen3-1.7B | 28 | 16/8 | Yes | 32K |
| Qwen3-4B | 36 | 32/8 | Yes | 128K |
| Qwen3-8B | 36 | 32/8 | No | 128K |
| Qwen3-14B | 40 | 40/8 | No | 128K |
| Qwen3-32B | 64 | 64/8 | No | 128K |

#### MoE 模型配置

| Models | Layers | Heads (Q/KV) | Experts (Total/Activated) | Context |
|--------|--------|-------------|:-------------------------:|:-------:|
| Qwen3-30B-A3B | 48 | 32/4 | 128/8 | 128K |
| Qwen3-235B-A22B | 94 | 64/4 | 128/8 | 128K |

两个 MoE 模型均为 **128 个总专家、每 token 激活 8 个**，激活比例约 6.25%，以大容量换取低单次计算成本。

### 3.2 核心架构组件

- **GQA (Grouped Query Attention)**：K/V head 数远少于 Q head，降低 KV cache 与注意力计算开销（Dense 多为 8 个 KV head，MoE 为 4 个）；
- **SwiGLU**：门控激活的前馈网络，

$$\text{SwiGLU}(x)=\big(\text{SiLU}(xW_1)\odot xW_2\big)W_3$$

- **RoPE (Rotary Position Embedding)**：旋转位置编码，天然支持长度外推；
- **RMSNorm (pre-norm)**：预归一化结构，

$$\text{RMSNorm}(x)=\frac{x}{\sqrt{\dfrac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}\cdot\gamma$$

### 3.3 相对 Qwen2 的架构改动

两处关键改动：

1. **新增 QK-Norm**：对注意力中的 Query 与 Key 分别施加 RMSNorm 后再做点积，稳定大规模训练时的注意力 logits 数值，

$$Q'=\text{RMSNorm}(Q),\quad K'=\text{RMSNorm}(K),\quad A=\text{softmax}\!\left(\frac{Q'K'^{\top}}{\sqrt{d}}\right)$$

2. **移除 QKV-bias**：去掉 Q/K/V 投影上的偏置项，简化结构。

### 3.4 MoE 设计

Qwen3-MoE 相比 Qwen2.5-MoE 的一个显著区别：**不再使用 shared experts（共享专家）**，所有专家均为路由专家。

- **路由**：每个 token 经门控网络选择 **8/128** 个专家；
- **Global-batch load balancing loss**：在全局 batch 维度上计算专家负载均衡损失，避免专家塌缩（少数专家被过度激活），促进专家专业化与均衡利用。

### 3.5 Tokenizer 与上下文长度

- **Tokenizer**：BBPE (Byte-level BPE)，词表大小 **151,669**；
- **上下文长度**：
  - 小模型 Qwen3-0.6B / 1.7B：**32K**；
  - 4B 及以上模型：**128K**。

---

## 第 4 章 预训练

Qwen3 的预训练围绕一个核心目标：用**更大规模、更高质量、更多语言**的数据，配合**分阶段课程**，同时建立通用能力与推理/长上下文能力。整个预训练消耗 **36T tokens**，支持 **119 种语言**（从 Qwen2.5 的 29 种大幅扩展）。

### 4.1 数据

Qwen3 的数据构建体现了"用上一代模型反哺下一代数据"的范式，这是其相比前代最显著的工程进步之一。

**规模与语言覆盖**

- 总计 **36T tokens**，覆盖 **119 种语言**。
- 语言覆盖从 Qwen2.5 的 29 种扩展到 119 种，意味着对低资源语言（low-resource languages）的支持显著增强。

**数据扩充方法（Data Expansion）**

Qwen3 系统性地利用 Qwen2.5 系列模型来"制造"高质量数据，而非仅依赖网络抓取：

| 方法 | 作用 | 产出 |
|------|------|------|
| **Qwen2.5-VL 从 PDF 提取文本** + Qwen2.5 精炼 | 把大量扫描文档/PDF/学术资料转化为干净文本 | 额外**数万亿（trillions）tokens** |
| **Qwen2.5-Math 合成数学数据** | 生成带解题过程的数学语料 | 高质量 STEM/Math 数据 |
| **Qwen2.5-Coder 合成代码数据** | 生成代码及代码相关语料 | 高质量 coding 数据 |

这一思路的关键在于：模型能力越强，合成数据质量越高，从而形成正向飞轮（data flywheel）。

**多维标注与配比优化**

- **30T+ tokens** 被进行多维度标注（annotation），维度包括：
  - **教育价值（educational value）**
  - **领域（domain）**
  - **安全（safety）**
- **Instance-level 数据配比优化**：传统做法是按数据源（source-level）调整配比，Qwen3 进一步下沉到**单条样本（instance）粒度**进行配比控制，使数据组合更精细、更可控。这是相比 source-level mixing 的一项方法学升级。

### 4.2 三阶段预训练（Three-Stage Pretraining）

Qwen3 将预训练拆为三个阶段（S1 → S2 → S3），形成"广度优先 → 深度推理 → 长上下文"的课程式（curriculum）训练。

| 阶段 | 名称 | Token 规模 | Context | 核心目标 |
|------|------|-----------|---------|----------|
| **S1** | General（通用） | **30T+ tokens** | 4K | 建立语言能力与通用知识，覆盖 119 种语言 |
| **S2** | Reasoning（推理） | **~5T tokens** | 4K | 提高 STEM / coding / reasoning 数据比例，**加速 LR decay** |
| **S3** | Long Context（长上下文） | **数百B tokens** | 32K | 扩展上下文长度，RoPE ABF + YARN + DCA |

**S1 — General Stage**
- 规模最大（30T+ tokens），4K context。
- 在 119 种语言上建立基础语言能力与广泛的世界知识。
- 这一层决定了模型的"底盘"质量。

**S2 — Reasoning Stage**
- 约 5T tokens，仍在 4K context 下训练。
- **显著提高 STEM、coding、reasoning 数据的比例**，相当于在通用底盘之上做"推理增厚"。
- 采用**加速的 learning rate decay**（accelerated LR decay），控制这一阶段的收敛节奏，使其能在不破坏 S1 已学知识的前提下强化推理。

**S3 — Long-Context Stage**
- 规模最小（数百B tokens），但 **context 扩展到 32K**。
- 关键技术组合：
  - **RoPE ABF（Attention Base Frequency）**：将 base frequency 从 **10K 提升到 1M**，以支持更长的位置编码。
  - **YARN**（Yet another RoPE extensioN）+ **DCA（Dual Chunk Attention）**：进一步将推理长度扩展约 **4 倍**。
- 这使得模型在训练时用 32K，但推理时可以处理显著更长的序列。

> **设计直觉**：三阶段是典型的"先宽后深再长"课程。S1 打基础，S2 针对性增强最难的能力（推理），S3 专门解决长上下文外推——把外推难题隔离到最后一个小阶段单独攻克，避免拖累主训练。

### 4.3 Scaling Laws 用于超参数预测

Qwen3 利用 **scaling laws** 来**外推预测**大规模模型的超参数（如 learning rate、batch size 等），而非在大模型上反复试错。这是现代大模型训练的标准工程实践：

1. 在小/中等规模模型上做受控实验，拟合 loss 与规模的关系。
2. 用拟合出的 scaling curve **预测**目标大模型的最优超参数。
3. 直接将预测值应用于最终大模型训练，大幅降低试错成本。

这种做法对 235B 级别的旗舰模型尤其关键——在该规模上做网格搜索（grid search）几乎不可承受。

---

## 第 5 章 后训练

如果说预训练决定模型的"能力上限"，后训练则决定模型能否把这些能力**可靠、可控地释放**出来。Qwen3 后训练的核心创新是：用一条 **4-stage pipeline** 把"深度推理（thinking）"和"通用对话（non-thinking）"统一到同一个模型中，并提供 **thinking budget** 进行成本控制；同时用 **strong-to-weak distillation** 高效地把大模型能力迁移到小模型。

### 5.1 四阶段后训练管线

![Figure 1: Qwen3 后训练四阶段管线图](Figures/2025-05-14-qwen3-technical-report-fig1.png)

*图1：Qwen3 的后训练流程分为四个阶段：Long-CoT Cold Start (Stage 1) → Reasoning RL with GRPO (Stage 2) → Thinking Mode Fusion (Stage 3) → General RL (Stage 4)。同时展示了 Strong-to-Weak Distillation 路径，从教师模型直接蒸馏 logits 到小模型，仅需 1/10 GPU 小时。*

#### Stage 1 — Long-CoT Cold Start（长思维链冷启动）

**目的**：为模型注入"会做长链推理"的初始能力，作为后续 RL 的起点。

- **筛选复杂推理问题**：从大规模题库中挑选难度高、适合长链推理（long Chain-of-Thought）的题目。
- **用 QwQ-32B 生成候选推理过程**：QwQ-32B（Qwen 的推理模型）作为"教师"，为这些问题生成长 CoT 解答。
- **6 轮过滤（6 rounds of filtering）**：对生成结果进行多轮严格筛选，剔除错误、低质量、不规范的推理轨迹，只保留高质量的长 CoT 数据。
- 用这些高质量数据对 base model 做 SFT，得到具备初步推理能力的 cold-start checkpoint。

#### Stage 2 — Reasoning RL（推理强化学习）

**目的**：通过强化学习把推理能力从"能做"推向"做强"。

- 算法：**GRPO**（Group Relative Policy Optimization）。
- **3,995 个 query-verifier pairs**：约 **4,000** 道带可验证答案（verifier）的题目——答案可被程序/规则客观判定（如数学题的数值答案、代码题的测试用例），从而提供可靠的 reward 信号。
- 训练配置亮点：
  - **Large batch + high rollouts**：大批量、每个 query 多次采样（rollout），提升梯度估计的稳定性与策略探索的充分性。
  - **Off-policy training**：使用与当前策略不完全同步的样本进行训练，提高样本利用率、降低算力开销。
- 效果示例：论文报告 AIME'24 在 **170 个 RL steps** 内从 **70.1 → 85.1**，体现 RL 阶段的高效增益。

#### Stage 3 — Thinking Mode Fusion（思维模式融合）

**目的**：把"会推理（thinking）"和"会正常对话（non-thinking）"两种行为统一进同一个模型，并允许用户/系统动态切换。

- **SFT 融合**：将 thinking 数据与 non-thinking 数据混合进行监督微调，使模型同时掌握两种模式，且互相不破坏。
- **Chat template 控制**：通过对话模板中的 **`/think`** 与 **`/no_think`** 标记来动态指示当前是否需要长链推理。
- 这是 Qwen3 的标志性设计——**一个模型，两种模式，按需切换**，避免了"推理模型"与"对话模型"分离部署的复杂度。

#### Stage 4 — General RL（通用领域强化学习）

**目的**：在保持推理能力的同时，全面提升模型在各类通用任务上的表现。

- 对**多个通用 domain** 施加 RL，覆盖：
  - **Coding**（编程）
  - **Math**（数学）
  - **Instruction-following**（指令遵循）
  - **Multilingual**（多语言）
  - **Creative writing**（创意写作）
  - **QA**（问答）
  - **Role-playing**（角色扮演）
- 这一阶段确保模型不仅是"理科强"，而是全面的、可日常使用的助手。

> **管线整体逻辑**：Stage 1 给模型"装上"推理引擎 → Stage 2 把推理引擎"调到最强" → Stage 3 把推理与对话"融合可控" → Stage 4 把整体能力"补全打磨"。四个阶段层层递进，最终得到一个既能深度思考、又能轻量对话的统一模型。

### 5.2 Thinking Budget（思维预算）

这是 Qwen3 在**推理成本可控性**上的关键创新。

**核心观察**：经过 Stage 1–2 的训练后，模型**自然学会了处理"不完整的思维过程"**——即推理链条即使中途被打断，模型也能基于已有思考给出合理回答，而不是崩溃或输出无意义内容。

**机制设计**：

- 用户/系统可以为一个请求设定 **thinking token 预算**（上限）。
- 当模型的 thinking 过程达到该预算时，系统**自动插入一条 stop-thinking 指令**，引导模型停止继续思考，直接转入作答。
- 因为模型本身能处理被打断的思维链，所以这种"强制截断"不会显著损害答案质量。

**实际意义**：

| 场景 | 设置 |
|------|------|
| 简单问题 / 低延迟需求 | 给很小的 thinking budget（甚至 `/no_think`） |
| 复杂推理 / 高质量需求 | 给充足的 thinking budget |

这使得 Qwen3 能在**质量与成本/延迟之间连续调节**，而不必在"推理模型"和"快模型"之间二选一。

### 5.3 Strong-to-Weak Distillation（强到弱蒸馏）

**目的**：把大模型（如 235B 旗舰）的强大能力，高效地迁移到小模型上，使整个模型系列都受益。

**方法：直接蒸馏 logits（Direct Logits Distillation）**

- 与 5.1 的 4-stage pipeline 不同，这里**不走完整四阶段**，而是让小模型直接去**模仿大模型输出的 logits 分布**（即未归一化的概率分布，比硬标签携带更丰富的"软知识"）。
- Logits 蒸馏相比仅模仿最终答案或文本，能传递教师模型在每一步上的**不确定性信息与偏好排序**，信息量更大。

**效果**：

- **更高的 Pass@1**：小模型在单次采样下的正确率更高。
- **更好的探索能力（better exploration）**：蒸馏得到的小模型在多次采样/搜索场景下表现更优，说明其学到的不仅是"标准答案"，还有更优的解题分布。

**效率优势**：

- 仅需 **4-stage 训练约 1/10 的 GPU hours** 即可达到（甚至超越）4-stage 的效果。
- 这意味着用极低的成本就能把旗舰模型的能力"灌"给系列中的每个小模型，使整个 Qwen3 系列（6 dense + 2 MoE）都能达到高水准。

> **总结性观察**：Qwen3 后训练的三项创新——**统一双模式（Stage 3）**、**Thinking Budget**、**Strong-to-Weak Distillation**——分别解决了"能力整合"、"成本可控"和"能力下沉"三个工程难题，共同构成了 Qwen3 在工程落地上的核心竞争力。

---

## 第 6 章 实验评估

Qwen3 的实验评估围绕一个核心命题展开：**用更少的参数，取得更强的性能**。评估体系覆盖 5 大类别共 15 个 benchmark，兼顾通用知识、数学与 STEM、代码生成、多语言能力，并对 base model（预训练模型）与 post-trained model（后训练模型）分别给出结果。

### 6.1 评估基准体系（5 类别，15 个 benchmark）

| 类别 | Benchmark | 评测重点 |
|:---|:---|:---|
| **General（通用）** | MMLU、MMLU-Pro、MMLU-Redux、SuperGPQA、BBH | 综合知识与推理 |
| **Math & STEM** | GPQA、GSM8K、MATH | 数学推理与科学问题 |
| **Coding（代码）** | EvalPlus（HumanEval+MBPP+HumanEval++MBPP+ 平均）、MultiPL-E（8 种语言）、MBPP、CRUX-O | 代码生成、多语言编程、代码执行推理 |
| **Multilingual（多语言）** | MGSM、MMMLU、INCLUDE | 跨语言泛化 |

这一设计的关键考量是：单一 benchmark 容易因数据污染（data contamination）或偶然因素失真，多类别交叉验证能更稳健地反映真实能力。

### 6.2 Base Model 评估结果

#### 6.2.1 Flagship: Qwen3-235B-A22B-Base（Table 3）

Qwen3-235B-A22B 是 MoE 架构旗舰模型，总参数 235B、激活参数 22B。下表对比其与 4 个同量级/更大量级模型的表现：

| Benchmark/Model | Qwen2.5-72B (Dense 72B) | Qwen2.5-Plus (MoE 271B/37B) | Llama-4-Maverick (MoE 402B/17B) | DeepSeek-V3 (MoE 671B/37B) | **Qwen3-235B-A22B (MoE 235B/22B)** |
|:---|:---:|:---:|:---:|:---:|:---:|
| MMLU | 86.06 | 85.02 | 85.16 | 87.19 | **87.81** |
| MMLU-Redux | 83.91 | 82.69 | 84.05 | 86.14 | **87.40** |
| MMLU-Pro | 58.07 | 63.52 | 63.91 | 59.84 | **68.18** |
| SuperGPQA | 36.20 | 37.18 | 40.85 | 41.53 | **44.06** |
| BBH | 86.30 | 85.60 | 83.62 | 86.22 | **88.87** |
| GPQA | 45.88 | 41.92 | 43.94 | 41.92 | **47.47** |
| GSM8K | 91.50 | 91.89 | 87.72 | 87.57 | **94.39** |
| MATH | 62.12 | 62.78 | 63.32 | 62.62 | **71.84** |
| EvalPlus | 65.93 | 61.43 | 68.38 | 63.75 | **77.60** |
| MultiPL-E | 58.70 | 62.16 | 57.28 | 62.26 | **65.94** |
| MBPP | 76.00 | 74.60 | 75.40 | 74.20 | **81.40** |
| CRUX-O | 66.20 | 68.50 | 77.00 | 76.60 | **79.00** |
| MGSM | 82.40 | 82.21 | 79.69 | 82.68 | **83.53** |
| MMMLU | 84.40 | 83.49 | 83.09 | 85.88 | **86.70** |
| INCLUDE | 69.05 | 66.97 | 73.47 | 75.17 | 73.46 |

**核心结论**：

- **效率优势突出**：Qwen3-235B-A22B 仅用 DeepSeek-V3（671B/37B）约 **1/3 的总参数**（235B vs 671B）和约 **2/3 的激活参数**（22B vs 37B），却在 15 项 benchmark 中的 **14 项超越 DeepSeek-V3 Base**。这一"以小博大"的结果是 Qwen3 架构与训练策略有效性的最强证据。
- **数学与代码提升最为显著**：MATH 提升 **+9.22**（71.84 vs 62.62）、MMLU-Pro 提升 **+8.34**（68.18 vs 59.84）、EvalPlus 提升 **+13.85**（77.60 vs 63.75）。这类需要深度推理的 benchmark 受益于更大的 token 规模与更高质量的数据配比。
- **唯一的短板：INCLUDE**：这是唯一一项 Qwen3-235B 落后于 DeepSeek-V3 的 benchmark（73.46 vs 75.17）。INCLUDE 是多语言覆盖广度评测，提示 Qwen3 在部分低资源语言上仍有提升空间，这也呼应了第 8 章关于多语言能力持续优化的讨论。

#### 6.2.2 Qwen3-32B-Base（Table 4）

Qwen3-32B 为 dense（稠密）架构的旗舰，下表对比其与同规模及更大规模模型：

| Benchmark/Model | Qwen2.5-32B (32B) | Qwen2.5-72B (72B) | Gemma-3-27B (27B) | Llama-4-Scout (MoE 109B/17B) | **Qwen3-32B (32B)** |
|:---|:---:|:---:|:---:|:---:|:---:|
| MMLU | 83.32 | 86.06 | 78.69 | 78.27 | 83.61 |
| MMLU-Pro | 55.10 | 58.07 | 52.88 | 56.13 | **65.54** |
| SuperGPQA | 33.55 | 36.20 | 29.87 | 26.51 | **39.78** |
| BBH | 84.48 | 86.30 | 79.95 | 82.40 | **87.38** |
| EvalPlus | 66.25 | 65.93 | 55.78 | 59.90 | **72.05** |

**核心结论**：

- **越级挑战**：在 MMLU-Pro（65.54 vs 58.07）、SuperGPQA（39.78 vs 36.20）、BBH（87.38 vs 86.30）、EvalPlus（72.05 vs 65.93）四项推理密集型 benchmark 上，**32B 的 Qwen3 全面超越自家上一代 72B 模型**。这说明 Qwen3 在同等规模下的"单位参数效率"有质的飞跃。
- **通用知识仍有差距**：MMLU 上 Qwen3-32B（83.61）仍略低于 Qwen2.5-72B（86.06）。记忆密集型任务对参数规模仍有依赖，但差距远小于推理型任务上的领先幅度——这正符合"用算力换推理能力"的设计取向。
- 即使面对参数规模大得多的 Llama-4-Scout（MoE 109B/17B），Qwen3-32B 在全部 5 项上均胜出，进一步印证其训练方法论的普适性。

### 6.3 Post-trained Model 评估结果

后训练（post-training）阶段通过 SFT + 强化学习大幅释放模型潜力，尤其在竞赛级数学与代码任务上：

| Benchmark | 成绩 | 说明 |
|:---|:---:|:---|
| **AIME'24** | 85.7 | 竞赛级数学 |
| **AIME'25** | 81.5 | 最新年度竞赛题 |
| **LiveCodeBench v5** | 70.7 | 实时代码评测 |
| **CodeForces** | 2056 | 竞赛编程 Elo 评分 |
| **BFCL v3** | 70.8 | Function calling / 工具调用 |

**训练动态观察**：AIME'24 成绩在约 **170 个 GRPO steps** 内从 **70.1 → 85.1**，呈现近乎线性的陡峭上升。这一组训练曲线有两个重要含义：

1. **GRPO（Group Relative Policy Optimization）对数学推理的高效性**：无需海量训练步即可激活模型的推理链能力，验证了"思维链 + 强化学习"范式的有效性。
2. **base model 已"蓄能"充分**：后训练的快速跃升本质上来自预训练阶段对数学能力的深厚储备——若 base model 缺乏相应基础，少量 RL steps 无法带来如此大幅度的提升。

### 6.4 Thinking Budget 消融效果

Qwen3 的一个标志性设计是将 **thinking（思考）模式** 与 **non-thinking（非思考）模式** 统一于单一模型：

- **Thinking budget 越大，性能越强**：增加推理 token 预算带来一致（consistent）的性能提升，不存在明显的饱和点。用户可以根据任务难度动态分配算力。
- **Non-thinking 模式保障响应速度**：简单任务可跳过思考链，实现快速响应。
- **用户级控制**：通过 `/think` 与 `/no_think` 标志（flag），用户可在单次请求层面切换模式——**一个模型即覆盖"深度推理"与"快速响应"两种使用场景**，无需在不同模型间切换。

这一设计的工程价值在于：将"是否思考"从架构选择降级为运行时参数，大幅简化了部署与调用链路。

---

## 第 7 章 开源代码与部署实现

Qwen3 采取彻底的开源策略，这是其能迅速获得社区采用的关键。

### 7.1 开源生态

- **GitHub 仓库**：`github.com/QwenLM/Qwen3`，截至报告参考时间已积累 **26.7k stars**——这一热度反映了社区对国产开源大模型的强烈需求。
- **License**：**Apache 2.0**，是业界最宽松、最商业友好的开源协议之一，允许商用、修改、分发，无 copyleft 约束。这是相比部分竞争模型（如限制商用的协议）的核心竞争优势。
- **Hugging Face 分发**：所有规模模型统一发布于 `Qwen/Qwen3-{size}`，便于通过标准工具链加载。

### 7.2 推理与调用

Qwen3 完全兼容主流的 `transformers` 库，使用标准 API 即可调用：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-{size}")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-{size}")
```

这种"零额外适配"的兼容性设计意味着：现有基于 `transformers` 的训练、推理、部署工具链（如 vLLM、SGLang、DeepSpeed 等）可以无缝接入 Qwen3，极大降低了集成成本。

### 7.3 Thinking Budget 的程序化控制

除命令行 flag 外，thinking budget 还可通过 **chat template（聊天模板）** 在代码层面精细控制。这使得：

- 应用开发者可在 prompt 构造阶段决定是否启用思考模式；
- 可根据任务类型（如简单问答 vs 复杂推理）动态切换；
- 支持在同一服务中混合处理不同复杂度的请求。

---

## 第 8 章 讨论与局限性

### 8.1 核心贡献与设计哲学

Qwen3 的贡献可归结为四个维度：

1. **统一思考模式 → 消除模型切换**
   将 thinking 与 non-thinking 整合进单一模型，从根本上解决了以往"推理模型"与"对话模型"需分别部署的痛点。这是面向工程实践的务实创新。

2. **Strong-to-weak distillation（强模型到弱模型蒸馏）**
   通过蒸馏方式训练小模型，仅需约 **1/10 的 GPU 小时**即可获得接近大模型的能力。这一效率突破对资源受限的研究者与企业意义深远，显著降低了高质量小模型的获取门槛。

3. **Apache 2.0 许可 → 推动广泛采用**
   宽松的开源协议是社区信任与商业落地的基础。相比一些附带使用限制的模型，Qwen3 的纯 Apache 2.0 策略消除了法律合规顾虑，是其能在 26.7k stars 量级获得社区认可的关键。

4. **支持 119 种语言 → 全球可访问性**
   多语言覆盖不仅是技术指标，更是 AI 平权（AI accessibility）的体现——使非英语母语用户也能获得高质量的本地化 AI 服务。

### 8.2 局限性与开放挑战

Qwen3 团队也坦诚指出若干局限：

- **训练成本高昂**：训练 235B 规模的 MoE 模型需要海量 GPU 资源与复杂的分布式训练工程，这对绝大多数机构是不可承受的门槛。这也意味着模型能力的进一步突破高度依赖少数头部团队的资源投入。

- **部署硬件门槛高**：即使有 MoE 稀疏激活（22B 激活参数），**完整运行 Qwen3-235B-A22B 仍需可观的显存**（约数百 GB 级别），普通开发者和中小企业难以本地部署。这正是 strong-to-weak distillation 与小模型（如 Qwen3-32B 及更小）存在的价值——它们让普通用户也能获得"Qwen3 级别"的能力。

- **多语言覆盖仍有空间**：如 6.2.1 节 INCLUDE 评测所示，Qwen3-235B 在部分多语言 benchmark 上仍落后于 DeepSeek-V3，低资源语言能力是持续优化的方向。

### 8.3 综合评价

Qwen3 的整体定位清晰：**用 MoE 稀疏化换取效率，用统一思考模式换取易用性，用彻底开源换取生态**。从实验数据看，"1/3 参数、2/3 激活、14/15 benchmark 胜出"是对其架构设计最有力的背书；而 post-trained 模型在 AIME、CodeForces 等竞赛级任务上的表现，则证明了其在前沿推理能力上的竞争力。

其局限性（训练成本、部署门槛）本质上是当前大模型范式的共性问题，Qwen3 通过蒸馏降本、小模型矩阵布局给出了务实的缓解路径。作为 Apache 2.0 开源模型，Qwen3 代表了推动大模型民主化（democratization）的重要一步。

---

