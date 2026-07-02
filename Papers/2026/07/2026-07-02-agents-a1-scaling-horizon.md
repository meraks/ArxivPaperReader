# Scaling the Horizon, Not the Parameters: Reaching Trillion-Parameter Performance with a 35B Agent

> **论文信息**
> - **标题**: Scaling the Horizon, Not the Parameters: Reaching Trillion-Parameter Performance with a 35B Agent
> - **作者**: Agents-A1 Team, Shanghai Artificial Intelligence Laboratory (Project Co-lead: Bo Zhang, Lei Bai)
> - **发表形式**: arXiv:2606.30616, 2026年6月
> - **核心贡献**: 提出 Agents-A1，一个 35B MoE agentic model，通过 scaling agent horizon 而非参数规模，达到万亿参数模型的性能水平
> - **开源状态**: 论文提供 Code 与 Model 链接（具体仓库路径/许可未在正文给出，请以原文链接为准）

## Ch1: 论文概述与核心贡献

### 1.1 一句话总结

Agents-A1 是一个 35B Mixture-of-Experts agentic model，通过**三段式训练**（全领域 SFT → 领域教师模型 → 多教师 on-policy 蒸馏）在多种长程智能体基准测试中超越或匹配 Kimi-K2.6、DeepSeek-V4-Pro 等万亿参数模型，证明了 scaling agent horizon 可以作为一种 scaling parameters 的有效替代路径。

### 1.2 三大核心创新

论文正式列举的三项贡献：

1. **Agents-A1 模型本身**：一个 35B MoE agentic model，在科学与研究的交互式长程能力上可匹敌甚至超越 1T 参数模型。

2. **长程知识-动作基础设施（Knowledge-Action Infrastructure）**：构建知识-动作图（KAG），将异构语料分解为原子能力并组织成可验证、可自扩展的训练监督信号。Agentic trajectory 平均长度达 45K tokens，包含证据、动作、观察和验证信号。

3. **领域路由在线蒸馏（Domain-Routed On-Policy Distillation + SVA）**：提出 Salient Vocabulary Alignment，在教师选择的 top-k token 子集上对齐学生分布，并通过 domain-normalized 聚合解决跨领域推理模式冲突。

> 上述三者通过**三段式训练流水线**组织（论文 §4 / Figure 2，属方法结构而非单独贡献项）：全领域 SFT 打基础 → 6 个领域教师模型（搜索/科研/工程/指令跟随/工具调用/通用 agent）各司其职 → 多教师 OPD 合并为统一可部署模型。

### 1.3 关键结果速览

| 基准测试 | Agents-A1 (35B) | Qwen3.6-35B-A3B | Kimi-K2.6 | DeepSeek-V4-Pro | GPT-5.5 |
|---------|:--------------:|:--------------:|:---------:|:--------------:|:------:|
| **SEAL-0** | **56.4** | 38.7 | 50.5 | 55.0 | 42.3 |
| **IFBench** | **80.6** | 64.4 | 71.8 | 73.5 | 75.9 |
| **GAIA** | 96.0 | 78.6 | 80.6 | 98.1 | 87.4 |
| **HiPhO** | **46.4** | 37.7 | 41.1 | 38.7 | 43.3 |
| **FS-Olympiad** | **79.0** | 60.3 | 73.0 | 76.0 | 78.0 |
| **FS-Research** | **40.0** | 2.9 | 17.9 | 13.3 | 26.7 |
| **BrowseComp** | 75.5 | 67.9 | 83.2 | 83.4 | 84.4 |
| **MolBench-Bind** | 56.8 | 48.7 | 21.6 | 37.8 | 62.2 |
| **SciCode** | 44.3 | 35.8 | 53.5 | 50.0 | 56.1 |
| **HLE w/ tools** | 47.6 | 36.2 | 54.0 | 48.2 | 52.2 |
| **MLE-Bench-Lite** | 43.9% | 34.9% | 62.1% | 63.6% | 72.7% |

> **粗体** = 全体最佳（所有对比模型中最高）。Agents-A1 在所列全部基准上均为 35B 级最佳；其中 **SEAL-0、IFBench、HiPhO、FS-Olympiad、FS-Research** 五项为全体最佳（超过所有对比的万亿参数模型）。注：GAIA (96.0) 略低于 DeepSeek-V4-Pro (98.1)，MolBench-Bind (56.8) 低于 GPT-5.5 (62.2)，故二者未加粗。

### 1.4 论文总览图

论文包含 **9 个 Table** 和 **5 个 Figure**，覆盖：
- Figure 1: Agents-A1 与各大模型的基准性能对比图
- Figure 2: 三段式训练流水线概览
- Figure 3: 知识-动作基础设施（KAG）架构
- Figure 4: MLE 12小时优化轨迹（右须鲸检测任务 AUC 从 0.58→0.9935）
- Figure 5: 地球科学分析案例（热带气旋 Nargis 路径重建）

### 1.5 开源与代码可用性

- **代码 / 模型**：原文首页提供 `Code` 与 `Model` 链接。常见落地页为 HuggingFace `InternScience/Agents-A1` 与项目页 `internscience.github.io/Agents-A1/`（**注：这些路径/许可未在论文正文中给出，请以原文 Code/Model 链接落地页为准核实**）
- **基座模型**：Qwen3.5-35B-A3B（Qwen 团队，原文 §4.1 明确）

## Ch2: 研究背景与动机

### 2.1 长程智能体的双路径困境

当前大语言模型正从静态问答系统转向能够规划、使用工具、与环境交互并自我修正的 autonomous agents。在软件工程、科学研究、复杂决策等真实场景中，智能体需要执行长程任务（long-horizon tasks）：获取信息、分解任务、调用工具、验证中间结果、持续调整策略。这类场景的核心挑战在于早期错误会累积放大。

现有提升长程智能体能力的方法大致分为两条路径：

1. **Scale parameters**：通过扩大模型参数量（如 Kimi-K2.6、DeepSeek-V4）来内化推理模式、工具使用行为和领域知识。该方法有效但资源需求极高，难以复现。

2. **Scale horizon**：不扩大参数量，而是通过显式化中间决策过程（知识获取、动作执行、观察解释、验证）来提供可训练的监督信号。这是 Agents-A1 采用的方向。

### 2.2 两个关键瓶颈

**瓶颈 1：知识基础设施缺失**。长程轨迹训练需要一个统一的环境来连接外部知识、智能体动作、观察和验证信号，使模型能从 grounded feedback 中学习而非孤立的文本监督。没有这样的知识-工具交互基础设施，智能体难以学会在长程轨迹中规划、适当调用工具、验证证据、从失败中恢复。

**瓶颈 2：异构能力整合困难**。长程智能体需要跨领域整合多样化的能力（多步信息检索、工具使用、可执行迭代、约束追踪、结果反思）。这些能力在不同领域中以不同的模式涌现，且相互之间以复杂方式交互。将高度不同的专门化能力合并为一个统一的 agentic model 面临严重的领域冲突问题。

### 2.3 Agents-A1 的解决思路

Agents-A1 的核心思路是**构建一个可扩展的知识-动作基础设施** + **三段式训练流水线**来同时解决这两个瓶颈：

1. **KAG（Knowledge-Action Graph）**：将异构语料分解为原子能力（信息获取、工具调用、可执行迭代、证据验证、约束追踪），构建可验证、可自扩展的监督信号。

2. **领域专门化**：训练 6 个独立领域教师模型，每个专注于一种核心能力。

3. **多教师蒸馏**：通过 domain-routed on-policy distillation 将多个教师的专长合并为一个统一模型，同时解决推理模式冲突问题。

## Ch3: 知识-动作基础设施（KAG）

### 3.1 核心思想

知识-动作图（Knowledge-Action Graph, KAG）是 Agents-A1 训练数据的基础设施。与仅存储实体-关系事实的传统知识图谱不同，KAG 保留了**答案的获取、测试、修正和验证的完整过程**。

KAG 定义为一个带类型的四元组：

$$G_d = (C_d, A_d, O_d, V_d)$$

其中：
- $C_d$：领域语料库（证据块、实体、事实、约束）
- $A_d$：动作空间（工具调用、检索查询、代码编辑和执行、推理步骤）
- $O_d$：观察空间（工具返回值、检索证据、执行状态、中间产物）
- $V_d$：验证器集（正确性检查、证据支持、约束满足、目标完成）

图的边表示支持、依赖、产生、验证和动作转换关系。

### 3.2 原子能力分解

Agents-A1 将长程智能体能力分解为五种**原子能力**：

| 原子能力 | 定义 | 对应 KAG 组件 |
|---------|------|-------------|
| 信息获取（Information Acquisition） | 从外部源检索和收集相关证据 | $C_d$ 检索路径 |
| 工具调用（Tool Calling） | 基于 schema 执行工具操作 | $A_d$ 工具序列 |
| 可执行迭代（Executable Iteration） | 代码编写、执行、调试、重试 | $A_d$ 编辑链 |
| 证据验证（Evidence Verification） | 判断输出是否正确、证据是否充分 | $V_d$ 验证信号 |
| 约束追踪（Constraint Tracking） | 跟踪用户指令中的多重要求 | $V_d$ 约束检查 |

每种能力在 KAG 中以链接的 t-步动作记录 $(s_t, a_t, o_t, v_t)$ 表示，其中 $s_t \subseteq C_d \cup O_{<t}$ 是当前状态，$a_t \in A_d$ 是动作，$o_t \in O_d$ 是观察，$v_t \in V_d$ 是验证结果。

### 3.3 自对抗图搜索与扩展

为了优化每个领域 $G_d$ 的质量，Agents-A1 设计了一个 **proposer-solver-verifier 三体对抗游戏**：

$$
\pi_P \xrightarrow{\text{采样}} \text{task} \xrightarrow{\pi_S \text{求解}} \text{trajectory} \xrightarrow{\pi_V \text{验证}} \text{accept/reject}
$$

- **Proposer $\pi_P$**：采样图区域，提出受约束的 task
- **Solver $\pi_S$**：通过检索和工具求解 task
- **Verifier $\pi_V$**：验证答案、证据、执行结果、轨迹和 shortcut 风险

生成的 task 可表示为：

$$x = (q, d, \tau, y^*, E_q, V_q), \quad E_q \subseteq C_d, \quad V_q \subseteq V_d$$

其中 $q$ 是指令，$d$ 是领域标签，$y^*$ 是目标答案，$\tau$ 是轨迹：

$$\tau = [(s_1, a_1, o_1, v_1), \dots, (s_T, a_T, o_T, v_T), y], \quad a_t \in A_d,\ o_t \in O_d,\ v_t \in V_q \cup \{\bot\}$$

其中 $v_t = \bot$ 表示该步无 step-level 验证器触发（$V_q \subseteq V_d$ 为任务级验证器子集）。

Verifier $\pi_V$ 接受 task 的条件是：（i）可验证（存在 $v \in V_q$ 能验证）；（ii）有效（$\tau$ 达到 $v$ 接受的答案）；（iii）过程有信息量（有意义中间决策）；（iv）证据覆盖（$E_q$ 在 $\tau$ 中被实际引用）；（v）无歧义，无 shortcut。

被接受的轨迹写回 KAG，形成闭环自我进化。

### 3.4 六个领域的数据管道

Agents-A1 的 SFT 数据覆盖多个异构领域。原文 Table 2 仅给出各领域的**平均 token 长度**（未给出每领域 trajectory 计数）：

| 领域 | 平均 token 长度 | 关键数据源 / 规模说明 |
|------|:--------------:|-----------|
| 长程搜索（Deep Research） | 44K | Wiki corpus + web search（最大上下文 256K） |
| 编码与工程（Coding/Engineering） | 48K | 已结束 Kaggle 竞赛 + MLE-Dojo |
| 科学推理与研究（Scientific Reasoning） | 37K | ~15K 增强科学问题库（Sec 3.3） |
| 指令跟随（Instruction Following） | 3K | 13K Nemotron-RL + 10K 自建长上下文 QA（Sec 3.4） |
| 通用智能体任务（General Agentic） | 39K | 多轮对话/规划/决策 |
| 工具调用（Tool Calling） | — | 科学/web/仓库/数据库工具（Sec 3.5；原文 Table 2 未单列此项） |
| **总体（Overall）** | **45K** | — |

> 注：除科学（~15K 问题库）与指令跟随（13K+10K）的语料规模在正文中给出外，其余领域原文未公布 trajectory 计数。工具调用作为独立数据领域在 §3.5 描述，但未计入 Table 2 的 token 长度表。

**SFT 总数据量**：约 100K trajectories，平均 45K tokens（Sec 4.1）。

### 3.5 关键设计细节

**搜索领域**：在 Wiki 数据库上构建关系链（relation chain），通过受控随机游走生成可验证的搜索问题。Trajectory 在真实互联网环境中收集，最大上下文窗口 256K tokens。

**MLE 领域**：基于 MLE-Dojo 和已经结束的 Kaggle 竞赛，构建可执行代码树。Agent 可以使用 write_full_code / patch_code / execute_code / execute_bash 等工具，通过 tree search 探索优化方案。

**科学推理领域**：构建约 15K 增强科学问题库，包含无工具纯推理轨迹和 tool-augmented 推理轨迹（search / visit / code / scholar 四个工具）。

**指令跟随领域**：13K 多约束指令数据（来自 Nemotron-RL）+ 10K 长上下文 QA（自建），包含注入的 in-context rules 和干扰项。

**工具调用领域**：从科学、Web、代码库、数据库等场景提取工具接口，通过约束图搜索生成可验证的调用任务。

## Ch4: 三段式训练流水线

### 4.1 第一阶段：全领域 SFT

**基座模型**：Qwen3.5-35B-A3B（MoE 架构，35B 总参数量，3B 激活参数）

**目标**：使基座模型对齐广泛的长程智能体行为，提升指令跟随能力。

**数据组成**：约 100K trajectories（六领域混合），总平均长度 45K tokens。

**训练配置**：

| 超参数 | 值 |
|--------|---|
| 学习率 | $1 \times 10^{-5}$ |
| 学习率调度 | Cosine with warmup |
| Warmup ratio | 0.05 |
| Batch size | 16 |
| Epochs | 1 |
| 最大序列长度 | 131,072 |
| 优化器 | AdamW |
| Weight decay | 0.1 |

为了提升训练吞吐量，采用 sample packing 策略将多个短样本拼接为单个训练序列，通过 attention mask 防止跨样本污染。

### 4.2 第二阶段：领域教师模型训练

#### 4.2.1 搜索教师（SFT + RL）

两阶段流水线：**Stage 1 SFT**（在 §3.1 的搜索轨迹上微调，习得基础工具使用与多轮搜索模式）→ **Stage 2 RL**（GRPO，强化多跳搜索推理）。

**算法**：GRPO（Group Relative Policy Optimization）

**工具**：Web Search、Read Page、Code

**训练数据**：约 2,000 精心选择的多跳推理问题（用 SFT 模型每题尝试 5 次，保留既有正确又有错误轨迹的题目），过滤掉模型"总是正确"或"总是错误"的问题。

**Rollout**：每个 prompt 生成 8 个 rollouts，最多 300 次工具调用

**奖励设计**：
- **正确性奖励**：LLM judge 评估最终答案是否正确
- **效率惩罚**：前 $K$ 轮免费，后续线性增加惩罚
- **重复惩罚**：近期重复的 search/read_page 调用施加小惩罚
- **格式校准奖励**：奖励符合输出格式的答案

**配置**：恒定学习率 $1 \times 10^{-6}$；每步采样 32 个问题 × 8 个 rollout（global batch = 256）；GRPO clip range $[0.2, 0.28]$；KL 惩罚系数 0.001；熵正则化 0.0001；最大响应长度 4,096；最大序列长度 131,072；rollout 温度 1.0。

#### 4.2.2 科学教师（两阶段 SFT）

**Stage 1：Reasoning-Enhanced SFT** — 纯推理轨迹训练，强化内在推理深度。
**Stage 2：Tool-Augmented SFT** — 工具增强推理训练（search/visit/code/scholar 四工具）。

#### 4.2.3 指令跟随教师（两阶段 RL）

**Stage 1：Instruction-Following RL** — 基于 Nemotron 指令跟随数据，使用可验证规则奖励（格式/长度/关键词/语言等）。
**Stage 2：Long-Context Learning RL** — 基于自建长上下文 QA，使用基于规则的答案匹配奖励。

**优化**：Dynamic sampling — 只保留组内奖励非均匀的 prompt，过滤全部相同奖励的 prompt。

#### 4.2.4 工具调用教师（SFT + RL）

**SFT 阶段**：使用 Section 3.5 收集的工具调用数据训练基础能力。

**RL 阶段**：奖励包含二值 outcome 奖励 $r_i^{\text{out}} \in \{0,1\}$ 与基于 LLM rubric 的过程评分：

$$r_i^{\text{out}} \in \{0,1\}, \quad r_i^{\text{proc}} = \frac{1}{|\mathcal{R}|} \sum_{j=1}^{|\mathcal{R}|} \mathbb{1}[\mathcal{R}_j \text{ is satisfied}]$$

随后使用 PAPO-style advantage（不对称设计）：成功轨迹已满足全部 rubric，故仅对**失败轨迹**加入过程奖励，避免重复计数：

$$A_i = A_i^{\text{out}} + \lambda_{\text{neg}} \cdot \mathbb{1}[r_i^{\text{out}} = 0] \cdot A_i^{\text{proc}}, \quad \lambda_{\text{neg}} = 0.5$$

其中 $A^{\text{out}}$ 在组内有效样本上归一化，$A^{\text{proc}}$ 仅在有效**负样本**上归一化。

通过 data reuse 反复利用同一批高质量数据，每任务期望生成轨迹数约为 $N_{\text{reuse}} \approx \dfrac{R \cdot B \cdot K}{|\mathcal{D}|}$（$R$=rollout 轮数，$B$=batch，$K$=每 prompt 采样数）。仅 64 个 hard tasks 通过此机制实现高效的 RL 提升。

### 4.3 第三阶段：多教师在线蒸馏（OPD）

#### 4.3.1 动机

第一阶段全领域 SFT 在合并多个领域的推理模式时出现**性能下降**（如长思考推理 vs 多轮工具调用之间存在冲突），导致指令跟随和 HLE 等任务出现倒退。

#### 4.3.2 Salient Vocabulary Alignment（SVA）

核心思想：不在全词表上做对齐，而是在教师选择的 **top-k token 子集**上对齐学生和教师分布。

在位置 $t$，令 $p_{s'}(u)$ 为学生分布，$p_{t,i}(u)$ 为路由教师分布。设 $S^{(k)}_{i,t}$ 为教师分布下 top-k 有效 token 集合，重归一化后：

$$\bar{p}_{s'}(u) = \frac{p_{s'}(u)}{\sum_{v \in S^{(k)}_{i,t}} p_{s'}(v)}, \quad
\bar{p}_{t,i}(u) = \frac{p_{t,i}(u)}{\sum_{v \in S^{(k)}_{i,t}} p_{t,i}(v)}, \quad u \in S^{(k)}_{i,t}$$

每个样本的 SVA 目标是在这个 salient support 上的 reverse KL 散度：

$$\ell^{(i)}_{SVA}(\theta'_s; \theta_{t,i}) = \frac{1}{|R_i|} \sum_{t \in R_i} \sum_{u \in S^{(k)}_{i,t}} \bar{p}_{s'}(u) \log \frac{\bar{p}_{s'}(u)}{\bar{p}_{t,i}(u)}$$

同时监控 student-side coverage：

$$\rho(i, t) = \sum_{u \in S^{(k)}_{i,t}} p_{s'}(u)$$

#### 4.3.3 Domain-Routed Normalized Objective

**硬路由**：每个样本只由其领域标签对应的教师监督，避免混合不兼容的教师信号。

**领域归一化聚合**：

$$\mathcal{L}_{\text{MT-SVA}}(\theta'_s) = \frac{1}{|D_B|} \sum_{d \in D_B} \frac{1}{|B_d|} \sum_{i \in B_d} \ell^{(i)}_{SVA}(\theta'_s; \theta_{t,i})$$

其中 $B_d$ 是 mini-batch 中来自领域 $d$ 的样本，$D_B$ 是活跃领域集合。这种聚合方式确保每个活跃领域有可比的影响力，防止高频领域主导更新。

#### 4.3.4 On-Policy Rollout 生成

学生模型在自己的 rollouts 上被优化（on-policy），教师提供 token 级别的引导。Rollout 约束：

- 轮次预算 $T_{\text{max}}$
- 响应长度预算 $L^{\text{resp}}_{\text{max}}$
- 上下文长度预算 $L^{\text{ctx}}_{\text{max}}$

被截断的 rollout 标记为 TRUNCATED 并保留为有效前缀；被系统中断的标记为 ABORTED 并重试。

学生从全领域 SFT checkpoint 初始化，教师池包含 6 个独立领域教师。OPD 训练数据从先前任务家族重组，去重并按领域平衡 unique prompts 数量。

## Ch5: 实验结果与分析

### 5.1 评估设置

| 任务类别 | 基准测试 | 评估指标 | 工具 |
|---------|---------|---------|------|
| 长程搜索 | GAIA, BrowseComp, XBench, SEAL-0 | pass@1 | Search + Visit + Code, 300 turns |
| 工程任务 | SciCode, MLE-Bench-Lite | pass@1 / Medal Rate | 12-hour wall-clock, H200 GPU |
| 科学研究 | HLE w/ tools, HiPhO, FS-O, FS-R | Accuracy | Search/Visit/Code/Scholar |
| 指令跟随 | IFBench, IFEval, LongBench V2 | Strict accuracy | 规则验证器 |
| 通用智能体 | τ2-Bench, VitaBench | pass@1 avg | DeepSeek-V3.2 作为用户模拟器 |
| 科学智能体 | MatTools, MolBench-Bind | Completion rate / Score | 自主代码探索 |

### 5.2 全领域 SFT 结果

| 基准测试 | Qwen3.5-35B-A3B | Agents-A1-SFT | Agents-A1 (OPD) |
|---------|:--------------:|:------------:|:--------------:|
| **BrowseComp** | 61.0 | 74.6 | 75.5 |
| **XBench-DS-2510** | 77.0 | 88.0 | 86.0 |
| **SEAL-0** | 41.4 | 52.3 | 56.4 |
| **GAIA** | 59.8 | 95.2 | 96.0 |
| **SciCode** | 37.1 | 42.3 | 44.3 |
| **MLE-Bench-Lite** | 24.2% | 39.4% | 43.9% |
| **HLE w/ tools** | 47.4 | 41.6 | 47.6 |
| **HiPhO** | 37.0 | 42.9 | 46.4 |
| **FS-O** | 64.5 | 75.0 | 79.0 |
| **FS-R** | 2.5 | 31.7 | 40.0 |
| **IFBench** | 70.2 | 68.7 | 80.6 |
| **LongBench V2** | 59.0 | 58.3 | 60.2 |
| **τ2-Bench** | 81.2 (32.5)† | 76.7 | 79.8 |
| **VitaBench** | 26.0 | 37.3 | 38.8 |
| **MatTools** | 21.0 | 37.0 | 47.1 |
| **MolBench-Bind** | 46.0 | 46.0 | 56.8 |

† τ2-Bench 的括号数值为复现结果（原始官方结果 81.2 存在环境差异）。

**关键发现**：
- SFT 在长程搜索、工程、科学研究和智能体任务上提升显著（如 GAIA: 59.8→95.2）
- 但在指令跟随、通用智能体任务和 HLE 上出现倒退（如 IFBench: 70.2→68.7）
- 这说明全领域 SFT 无法简单解决**不同推理模式的领域冲突**，需要 OPD 进一步优化

### 5.3 各领域教师模型结果

#### 搜索教师（SFT + RL）

| 模型 | GAIA | SEAL-0 | HLE w/ tools | XBench-DS-2510 |
|------|:---:|:------:|:----------:|:-------------:|
| Qwen3.5-35B-A3B | 59.8 | 41.4 | 47.4 | 77.0 |
| Search-enhanced Teacher | **95.1** | **54.1** | **50.3** | **86.0** |

GAIA 提升最大（59.8 → 95.1，+35.3），说明 RL 强化搜索工具使用对开放域问答效果显著。（注：原文正文此处写作 59.8→85.4 (+25.6)，与其 Table 5 的 95.1 不一致；本表采用 Table 5 数值。）

#### 科学教师（两阶段 SFT）

| 模型 | HLE w/ tools | HiPhO | FS-O | FS-R |
|------|:----------:|:----:|:---:|:---:|
| Qwen3.5-35B-A3B | 47.4 | 37.0 | 64.5 | 2.5 |
| Science-enhanced Teacher | 47.8 | **46.9** | **82.0** | **54.3** |

FS-R（FrontierScience-Research）从 2.5 大幅提升至 54.3，表明两阶段 SFT 能同时强化内在推理和工具交互能力。

#### 指令跟随教师（两阶段 RL）

| 模型 | LongBench V2 | IFBench | IFEval |
|------|:----------:|:------:|:-----:|
| Qwen3.5-35B-A3B | 59.0 | 70.2 | 91.9 |
| RL-enhanced Teacher | **62.4** | **82.0** | **93.4** |

LongBench V2 +3.4，IFBench +11.8，说明 RL 在强化约束满足和长上下文适应方面效果显著。

#### 工具调用教师（SFT + RL）

| 模型 | τ2-Bench Airline | Retail | Telecom | Avg | VitaBench Avg |
|------|:--------------:|:-----:|:------:|:---:|:------------:|
| Qwen3.5-35B-A3B | 16.00 | 30.70 | 50.90 | 32.53 | 26.00 |
| Tool-enhanced Teacher | **72.00** | **82.50** | **93.00** | **82.50** | **44.16** |

τ2-Bench 平均分从 32.53 提升至 82.50，证明工具特定的 SFT+RL 对结构化环境交互的强大效果。

### 5.4 多教师 OPD 结果（主要对比表）

| 基准测试 | Agents-A1 (35B) | Qwen3.6-35B-A3B | Nex-N2-mini | Kimi-K2.6 | DSV4-Pro (Max) | GPT-5.5 |
|---------|:--------------:|:--------------:|:----------:|:---------:|:-------------:|:------:|
| **BrowseComp** | 75.5 | 67.9 | 74.1 | 83.2 | 83.4 | 84.4 |
| **XBench-DS-2510** | 86.0 | 71.0 | 82.0 | 90.0 | 90.0 | 84.0 |
| **SEAL-0** | **56.4** | 38.7 | 49.6 | 50.5 | 55.0 | 42.3 |
| **GAIA** | 96.0 | 78.6 | 82.5 | 80.6 | 98.1 | 87.4 |
| **SciCode** | 44.3 | 35.8 | 29.9 | 53.5 | 50.0 | 56.1 |
| **MLE-Bench-Lite** | 43.9% | 34.9% | 34.9% | 62.1% | 63.6% | 72.7% |
| **HLE w/ tools** | 47.6 | 36.2 | 32.0 | 54.0 | 48.2 | 52.2 |
| **HiPhO** | **46.4** | 37.7 | 38.5 | 41.1 | 38.7 | 43.3 |
| **FS-O** | **79.0** | 60.3 | 52.0 | 73.0 | 76.0 | 78.0 |
| **FS-R** | **40.0** | 2.9 | 5.0 | 17.9 | 13.3 | 26.7 |
| **IFBench** | **80.6** | 64.4 | 54.1 | 71.8 | 73.5 | 75.9 |
| **LongBench V2** | 60.2 | 57.7 | 59.6 | 62.0 | 64.3 | - |
| **τ2-Bench** | 79.8 | 79.0 | 74.5 | 81.9 | 82.2 | 81.6 |
| **VitaBench** | 38.8 | 35.6 | 23.0 | 35.6 | 49.0 | 45.0 |
| **MatTools** | 47.1 | 15.9 | 34.1 | 63.8 | 47.1 | 68.8 |
| **MolBench-Bind** | 56.8 | 48.7 | 51.4 | 21.6 | 37.8 | 62.2 |

**下划线** = 35B 最佳，**粗体** = 全体最佳。

### 5.5 核心发现解读

**1. Agents-A1 超越万亿参数模型的领域**：
- **SEAL-0 (56.4)**：超过 Kimi-K2.6 (50.5) 和 DSV4-Pro (55.0)，接近搜索领域的前沿
- **IFBench (80.6)**：在所有对比模型中最高，包括 GPT-5.5 (75.9)
- **HiPhO (46.4)**：超过所有万亿参数模型，显示其在物理推理方面的优势
- **FS-O (79.0)**：接近 GPT-5.5 (78.0) 并超过 DSV4-Pro (76.0)
- **FS-Research (40.0)**：远超所有对比模型（第二名 GPT-5.5 仅 26.7）
- **MolBench-Bind (56.8)**：超越 Kimi-K2.6 (21.6) 与 DeepSeek-V4-Pro (37.8)，但低于 GPT-5.5 (62.2)

**2. Agents-A1 弱于万亿参数模型的领域**：
- **BrowseComp (75.5)**：低于 GPT-5.5 (84.4)、DSV4-Pro (83.4)、Kimi-K2.6 (83.2)
- **MLE-Bench-Lite (43.9%)**：显著低于 GPT-5.5 (72.7%)，说明工程类端到端任务对规模需求更高
- **SciCode (44.3)**：低于 GPT-5.5 (56.1) 和 Kimi-K2.6 (53.5)

**3. OPD 解决了 SFT 的领域冲突问题**：
- 对比 Agents-A1-SFT，OPD 后在 IFBench 上从 68.7→80.6（+11.9）
- HLE w/ tools 从 41.6→47.6（+6.0），恢复了 SFT 阶段的倒退
- HiPhO 42.9→46.4，FS-O 75.0→79.0，FS-R 31.7→40.0 均有提升

**4. 跨领域能力协同效应**：
论文指出，多步搜索、科学研究和长指令跟随能力之间存在协同效应——改进搜索工具使用也有助于科学研究任务选择正确的工具。

### 5.6 长程任务案例分析

**12小时 MLE 优化案例**（右须鲸检测）：
- 初始 CNN baseline AUC: 0.58
- 经过时间序列分析、音频增强、局部时间训练、Mel-spectrogram CNN 集成、大规模增强等多轮改进
- 最终验证 AUC: 0.9935（金牌级别）
- 展示了 Agents-A1 在长程工程任务中的自主优化能力

**地球科学分析案例**（热带气旋 Nargis 2008）：
- 自动识别 IBTrACS 数据源，完成数据提取、清洗、衍生指标计算、可视化和结果综合
- 成功重建路径：孟加拉湾中部形成 → 西北移动 → 转向东北东 → 缅甸南部登陆
- 保持 WMO/IMD 和 JTWC/USA 两种强度标准并存，避免混淆不同公约

## Ch6: 代码实现详解

### 6.1 工具接口设计

Agents-A1 为 MLE 领域设计了紧凑的工具接口，支持代码编写、执行、搜索树管理和委托分析：

| 工具 | 功能 |
|------|------|
| `write_full_code` | 从头编写完整训练脚本（打开新的根节点） |
| `patch_code` | 对已有节点代码进行局部编辑（生成子节点） |
| `execute_code` | 运行节点代码，捕获 stdout 和异常，提取验证指标 |
| `execute_bash` | 运行受保护的 shell 命令（环境设置、GPU 检查） |
| `list_nodes` | 查看解决方案树：当前选择、最近记录、无效历史 |
| `select_node` | 完整查看一个节点（代码、计划、输出、指标、父链） |
| `invalidate_node` | 将不可信节点从排序和提交中排除 |
| `update_answer` | 提交节点为当前候选答案 |
| `get_current_answer` | 查看当前已提交为答案的节点 |
| `write_notes` / `read_notes` | 持久化笔记（跨压缩保留决策和假设） |
| `analyze` | 生成独立的分析子 agent 探索数据和结果 |

### 6.2 核心概念实现

```python
# ⚠️ 非官方概念实现 — 基于论文 Section 2.3 / Section 4.3 描述编写
# 目的：帮助理解 SVA 训练目标的核心计算流程

import torch
import torch.nn.functional as F

def salient_vocabulary_alignment(
    student_logits: torch.Tensor,     # (N, vocab_size) 学生模型 logits，每行一个待训练 token 位置
    teacher_logits: torch.Tensor,     # (N, vocab_size) 路由教师模型 logits
    sample_ids: torch.Tensor,         # (N,) 每行所属样本 id，用于按样本做 1/|R_i| 归一（Eq 4）
    top_k: int = 128,                 # top-k 有效 token
) -> torch.Tensor:
    """
    Salient Vocabulary Alignment (SVA) 损失函数

    论文 Section 2.3.1 / Eq (4):
        ℓ_SVA^(i) = (1/|R_i|) Σ_{t∈R_i} Σ_{u∈S^(k)} p̄_s'(u)·log[p̄_s'(u)/p̄_t,i(u)]
    在教师选择的 top-k token 子集上，对每个样本 i 先在其可训练位置 R_i 上取均值，
    再计算学生与教师分布之间的 reverse KL（student||teacher）。
    注意：必须按 sample_ids 做逐样本的 1/|R_i| 归一，否则长轨迹会因 token 数多而主导梯度
    （这正是 Eq 6 域归一化所要避免的长度偏置）。
    """
    # 1. 计算学生/教师分布
    teacher_probs = F.softmax(teacher_logits, dim=-1)   # (N, V)
    student_probs = F.softmax(student_logits, dim=-1)    # (N, V)

    # 2. 取教师分布下的 top-k token 索引（支持集由教师决定）
    topk_indices = teacher_probs.topk(k=top_k, dim=-1).indices  # (N, k)

    # 3. 在支持集上 gather 两个分布（用 gather 避免 torch.arange 的 device 隐患）
    teacher_support = teacher_probs.gather(1, topk_indices)  # (N, k)
    student_support = student_probs.gather(1, topk_indices)  # (N, k)

    # 4. 在支持集上重归一化（Eq 4 的 p̄）
    teacher_norm = teacher_support / teacher_support.sum(dim=-1, keepdim=True)
    student_norm = student_support / student_support.sum(dim=-1, keepdim=True)

    # 5. Reverse KL: Σ_u p̄_s'(u)·log[p̄_s'(u)/p̄_t,i(u)]
    kl_per_pos = student_norm * (
        torch.log(student_norm.clamp_min(1e-12)) - torch.log(teacher_norm.clamp_min(1e-12))
    )
    pos_loss = kl_per_pos.sum(dim=-1)            # (N,) 每个 (样本,位置) 的支持集求和

    # 6. 按样本做 1/|R_i| 平均（Eq 4），再跨样本平均
    num_samples = int(sample_ids.max().item()) + 1
    sample_loss = pos_loss.new_zeros(num_samples)
    sample_count = pos_loss.new_zeros(num_samples)
    sample_loss.index_add_(0, sample_ids, pos_loss)
    sample_count.index_add_(0, sample_ids, torch.ones_like(pos_loss))
    per_sample = sample_loss / sample_count.clamp_min(1.0)   # (num_samples,) = ℓ_SVA^(i)
    return per_sample.mean()


def domain_routed_normalized_loss(
    losses_per_sample: torch.Tensor,  # (M,) 每个样本的 SVA loss ℓ^(i)（已含 1/|R_i|）
    sample_domain: torch.Tensor,      # (M,) 每个样本的领域标签 d
) -> torch.Tensor:
    """
    Domain-Routed Normalized Objective（论文 Section 2.3.2 / Eq 6）
        L = (1/|D_B|)·Σ_{d∈D_B} (1/|B_d|)·Σ_{i∈B_d} ℓ^(i)
    其中 D_B = {d : |B_d| > 0}，即 mini-batch 中出现的领域集合
    （仅按“是否出现”判定，无 < num_domains 上界过滤，否则 1-indexed 标签会静默丢弃合法域）。
    """
    if losses_per_sample.numel() == 0:                 # 空集保护，避免除零
        return losses_per_sample.new_zeros(())

    active_domains = sample_domain.unique()            # = D_B
    total = losses_per_sample.new_zeros(())
    for d in active_domains:
        mask = sample_domain == d
        total = total + losses_per_sample[mask].mean()  # (1/|B_d|)·Σ_{i∈B_d}
    return total / active_domains.numel()               # (1/|D_B|)
```

### 6.3 模型架构概览

- **基座模型**: Qwen3.5-35B-A3B (MoE)
  - 35B 总参数量，3B 激活参数量
  - 最大序列长度 131,072 (128K tokens)
- **训练框架**: GRPO + PAPO-style advantage
- **Distillation**: Multi-teacher domain-routed OPD + SVA

## Ch7: 局限性与延伸阅读

### 7.1 局限性

1. **MLE 能力仍然有限**：在 MLE-Bench-Lite 上 43.9%，远低于 GPT-5.5 的 72.7%。端到端工程优化任务需要稳定的目标保持、历史决策记忆和避免重复试验，这对 35B 模型仍是挑战。

2. **搜索领域不如 1T 模型**：BrowseComp 上 75.5 vs GPT-5.5 84.4/Kimi-K2.6 83.2，说明大规模 web 搜索场景仍受益于参数量优势。

3. **OPD 教师 vs 学生的天花板**：经过 OPD 训练的模型不一定在所有任务上超过对应的领域教师，因为学生在统一策略中需要做 trade-off。

4. **基础原子能力不足**：论文指出，规划前先推理、行动前先反思、长上下文中总结关键信息、识别重要历史信息等原子能力对长程任务至关重要，但目前这些能力主要来自 Qwen3.5 的初始能力而非专门训练。

5. **评估环境不一致**：τ2-Bench 的 Qwen3.5-35B-A3B 官方结果 (81.2) 与复现结果差距巨大（Table 4 单元格与 Table 8 脚注为 32.5/32.53，Table 4 脚注另记 33.0——原文自身不一致），表明评估框架版本依赖性强。

### 7.2 关键启示

- **Horizon Scaling 是可行的 Scaling 替代路径**：通过高质量数据基础设施和多教师蒸馏，35B 模型可以在多数长程智能体任务上匹配万亿参数模型
- **领域冲突是核心难题**：短思考（工具调用）与长思考（推理）之间存在根本性模式冲突，需要专门方法解决
- **基础设施比模型更大**：45K token 平均长度的轨迹数据 + KAG 自扩展 = 性能提升的关键

### 7.3 延伸阅读

- **Kimi K2.6**（arXiv 2026）：万亿参数 coding agent，Agents-A1 的主要对比对象
- **DeepSeek-V4**（arXiv 2026）：高效百万 token 上下文 intelligence
- **On-Policy Distillation**（Thinking Machines Lab, 2025）：OPD 的原始工作
- **GRPO**（DeepSeekMath, arXiv 2402.03300）：Group Relative Policy Optimization
- **PAPO**（arXiv 2603.26535）：稳定 rubric 集成训练的 decoupled advantage normalization
- **MLE-Dojo**（arXiv 2505.07782）：ML 工程 agent 训练环境
- **Agents-K1**（arXiv 2606.13669）：KAG 的前期铺垫工作
- **Qwen3.5-35B-A3B**（Qwen AI Blog, 2026）：基座模型
- **项目页面**: internscience.github.io/Agents-A1/
- **模型权重**: huggingface.co/InternScience/Agents-A1
