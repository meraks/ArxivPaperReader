# Procedural Memory Distillation: Online Reflection for Self-Improving Language Models

**论文信息**
> **标题**: Procedural Memory Distillation: Online Reflection for Self-Improving Language Models
> **作者**: Ye Liu, Srijan Bansal, Bo Pang, Yang Li, Zeyu Leo Liu, Yifei Ming, Zixuan Ke, Shafiq Joty, Semih Yavuz
> **机构**: Salesforce AI Research
> **发表**: arXiv:2607.01480, 2026年7月
> **代码**: 未公开

## Ch1 论文概述

### 核心问题

 在 RLVR（Reinforcement Learning with Verifiable Rewards）和偏好优化训练中，当前的主流方法（PPO、DPO、GRPO、SDPO 等）均以**单 episode 为单位**进行学习：模型生成一个 rollout → 获得奖励/反馈 → 更新策略 → 丢弃经验。这种设计忽略了训练过程中的一个重要事实：模型会在多个 epoch 中反复遇到相同或相似的问题，而跨 episode 的累积信号（哪些策略持续有效、哪些失败模式反复出现）被白白浪费。

### 核心贡献

本文提出 **Procedural Memory Distillation (PMD)**，将训练过程中反复尝试产生的跨 episode 经验转化为可复用的**程序性记忆**，并通过自蒸馏内化到策略权重中，使得最终模型在推理时无需外部记忆依赖。核心设计理念是**共演化**：策略生成 rollout 更新记忆，记忆引导的教师信号反过来训练策略。

### 关键结果

| 模型 | 方法 | SciKnowEval AVG | LiveCodeBench v6 |
|------|------|:---:|:---:|
| Qwen3-8B | Base | 47.9 | 27.1 |
| Qwen3-8B | GRPO | 69.4 | 41.2 |
| Qwen3-8B | SDPO | 74.4 | 47.9 |
| Qwen3-8B | **PMD** | **77.2** | **51.7** |
| OLMo3-7B | Base | 27.7 | 27.7 |
| OLMo3-7B | GRPO | 63.9 | 36.1 |
| OLMo3-7B | SDPO | 69.5 | 45.0 |
| OLMo3-7B | **PMD** | **73.3** | **51.1** |

PMD 相比 SDPO 在 SciKnowEval 上提升 **3.8–5.5%**，在 LiveCodeBench 上提升 **7.9–13.6%**。

### 图表速览

- **Figure 1**: PMD 整体框架图（学生尝试 → 在线记忆形成 → 教师指导 → 蒸馏入学生）
- **Figure 2**: 跨尺度记忆迁移实验（小模型+深度检索可超越大模型）
- **Figure 3**: 测试时扩展对比（PMD vs SDPO 的 verifier headroom）
- **Figure 4**: 交叠问题集 Venn 图
- **Table 1**: 各学习方法/记忆范式的对比矩阵
- **Table 2**: 主结果（含两个模型 x 两个基准）
- **Table 3**: 分解 PMD 增益（反射/持久化/共演化三机制）

---

## Ch2 研究背景与动机

### 2.1 RLVR 与偏好优化的局限

RLVR（Reinforcement Learning with Verifiable Rewards）是当前大语言模型后训练的主流范式，包括 GRPO、PPO、DPO 等方法。这些方法的核心逻辑是：

1. 从当前策略采样一组 rollout
2. 通过可验证奖励函数（如单元测试通过率、答案正确性）评估每个 rollout
3. 根据奖励信号更新策略
4. **丢弃**本次采样的 rollout 数据

这种**单 episode 局部更新**的设计忽略了一个关键机会：当模型在训练中反复遇到相似问题时，重复的尝试中蕴含了丰富的跨 episode 信息——哪些推理模式稳定有效、哪些错误反复出现、哪些策略随着训练推进而演化。

### 2.2 自蒸馏与 on-policy 蒸馏

SDPO（Self-Distillation Policy Optimization）是近期的重要进展。SDPO 让策略同时扮演学生（生成 rollout）和教师（基于训练时上下文重新评估同组 rollout）两个角色，使用 **on-policy 蒸馏**来提供比稀疏奖励更密集的学习信号。

然而，SDPO 的教师上下文仍然是**同 batch 局部**的：它只基于当前 rollout 组中的反馈或成功副本来构建教师信号，并不跨步骤积累知识。当模型在 epoch 2 遇到与 epoch 1 类似的问题时，SDPO 无法利用之前发现的有效策略。

### 2.3 现有记忆方法的不足

现有记忆相关工作可以分为两类：

- **推理时记忆系统**（RAG、MemGPT、Memory-Bank、A-MEM）：在推理时依赖外部检索来补充上下文。这类方法引入了推理时的延迟和存储依赖，且外部记忆的检索质量直接影响模型输出。
- **技能蒸馏方法**（SkillRL、Skill-SD、Mem2Evolve）：将经验蒸馏为技能并在推理时复用，但记忆往往是离线构建的、静态的，无法随策略演化而自适应更新。

PMD 的关键洞察是：**记忆应该是一个训练脚手架，而不是推理依赖**。在训练过程中，记忆帮助教师提供更强的监督信号；训练结束后，记忆被完全内化到模型权重中，推理时无需任何外部检索。

---

## Ch3 方法：Procedural Memory Distillation

### 3.1 总体框架

PMD 构建在 SDPO 的 on-policy 蒸馏框架之上，核心是四阶段循环：

1. **学生尝试**：当前策略 $\pi_\theta$ 对问题 $x_i$ 生成 $J$ 个 rollout，获得奖励和反馈
2. **在线记忆形成**：模型对每次尝试进行自我反思，提取策略和教训，形成三层记忆层次
3. **教师指导**：记忆条件化的教师利用累积经验生成更强的监督分布
4. **蒸馏入学生**：学生通过反向 KL 散度匹配教师分布，更新策略参数

形式化地，PMD 的训练目标为：

$$ \mathcal{L}_{\text{PMD}}(\theta; \theta_t, M_t) = \mathbb{E}_{x_i, y_i} \left[ \sum_{s=1}^{|y_i|} \text{KL}\left( \pi_\theta(\cdot | x_i, y_{i,<s}) \;\|\; \text{stopgrad}\left( q_{\theta_t}(\cdot | x_i, m_{i,t}, g_{i,t}, y_{i,<s}) \right) \right) \right] $$

其中 $m_{i,t}$ 是从三层记忆中为问题 $x_i$ 组装得到的记忆上下文，$g_{i,t}$ 是当前 batch 的训练时上下文（反馈/成功答案）。

### 3.2 三层记忆层次

PMD 将程序性记忆组织为三个层次，对应从具体到抽象的递进：

#### Level-0: 经验记忆（Experience Memory）

记录每个问题的原始 rollout 数据，包括：
- 成功/失败的完整推理链
- 对应的奖励分数和环境反馈
- 多轮尝试的累积记录

经验记忆具有最高的**保真度**——它保留了最完整的信息，但也最**局部化**——难以直接迁移到其他问题。

#### Level-1: 洞察记忆（Insight Memory）

对每个问题的经验进行抽象总结，提取：
- **策略**（Strategy）：系统性产生正确答案的推理模式
- **教训**（Lesson）：反复出现的错误和失败原因

当同时有成功和失败尝试时，PMD 使用**对比式反思**：让模型对比两组答案，识别正确推理和错误推理的关键区别。洞察记忆还利用成功比例作为置信度信号，使其成为可跨 epoch 复用的指导信息。

#### Level-2: 行为记忆（Behavior Memory）

跨问题的最高层次抽象。PMD 使用嵌入模型对所有训练问题进行语义聚类，然后在每个簇内聚合多个问题的洞察记忆，用 LLM 抽象出可复用的**行为规则**。例如，从一个关于 SMILES 摩尔质量计算的问题簇中，可能提取出 `behavior_verify_ring_atom_counts`：显式检查环状结构以避免原子计数错误。

三层记忆反映了**保真度-可迁移性权衡**：
- 经验记忆最忠实但最局部
- 行为记忆可迁移性最强但最粗略
- 洞察记忆在两者间取得平衡：保留问题级教训的同时保持紧凑可复用

### 3.3 在线记忆更新

PMD 的记忆是**在线构建**的，随策略演化而动态更新：

- **经验记忆**：每 batch 完成后即时更新当前问题的经验
- **洞察记忆**：每个问题的 rollout 组完成后再反思更新
- **行为记忆**：以更慢的时间尺度更新——每 K 步训练后，重新聚类并提炼全局行为库

记忆访问也有不同的作用域：经验和洞察是**问题特定**的（直接读取），行为是**全局**的（通过稠密检索获取 top-K 相关内容）。

### 3.4 共演化设计原则

PMD 最核心的设计是**策略与记忆的共演化**：

```
epoch t:  策略 π_t 生成 rollout → 更新记忆 M_t → 记忆条件化的教师 → 训练新策略 π_{t+1}
epoch t+1: 新策略 π_{t+1} 生成新 rollout → 再次刷新记忆 → ...
```

这种在线耦合区别于静态或离线记忆库：当策略在训练中不断改善时，记忆也随之演化，确保教师的监督信号始终与策略当前的能力和不足保持对齐。

---

## Ch4 实验与结果分析

### 4.1 实验设置

**基准**：
- **SciKnowEval**：科学多选题推理基准（生物/化学/物理/材料科学），报告 avg@16
- **LiveCodeBench v6**：代码生成基准，使用执行级单元测试反馈，报告 score@4

**模型**：
- Qwen3-8B（指令微调版）
- OLMo3-Instruct-7B

**训练配置**：8×H200-143GB GPU，FSDP 分片，SGLang v0.4 提供 rollout 服务，AdamW 优化器，每 prompt 采样 8 个 on-policy rollout，训练 20 epoch。行为记忆的提炼使用 GPT-5.4，聚类和检索使用 Qwen3-Embedding-0.6B。

### 4.2 主结果

PMD 在两个基准和两个模型上均一致优于 GRPO 和 SDPO（见 Ch1 表）。关键发现：

- **Qwen3-8B**：SciKnowEval AVG 从 SDPO 的 74.4 提升至 **77.2**（+2.8pp），LiveCodeBench 从 47.9 提升至 **51.7**（+3.8pp）
- **OLMo3-7B**：SciKnowEval AVG 从 69.5 提升至 **73.3**（+3.8pp），LiveCodeBench 从 45.0 提升至 **51.1**（+6.1pp）

### 4.3 消融实验：增益来源分解

为了理解 PMD 的提升来自哪些机制，设计了三组控制实验：

| 变体 | 描述 | SciKnowEval AVG | LiveCodeBench |
|------|------|:---:|:---:|
| SDPO | 基线 | 74.4 | 47.9 |
| PMD-Transient | 反射但不跨步持久化 | 75.7 | 48.1 |
| Evolving Memory + Frozen Policy | 更新记忆但不更新策略 | 54.0 | 35.9 |
| Frozen Memory + Evolving Policy | 固定记忆库 + SDPO 训练 | 65.0 | 47.5 |
| **PMD** | 共演化（完整） | **77.2** | **51.7** |

三个关键发现：

1. **反射本身已有帮助**：PMD-Transient 即使丢弃跨步记忆，仍胜过 SDPO（+1.3pp SciKnowEval, +0.2pp LiveCodeBench），说明单 batch 的结构化洞察已比原始上下文提供更强信号。

2. **持久化驱动大部分增益**：完整 PMD 相比 PMD-Transient 额外带来 SciKnowEval +1.5pp 和 LiveCodeBench +3.6pp。在代码生成中（成功模式更罕见），持久化的贡献远超一次性反射。

3. **共演化是核心驱动因素**：固定策略仅更新记忆几乎无提升（54.0）；固定记忆训练策略有部分提升（65.0）但仍远不及 PMD（77.2），因为策略演化后固定记忆的教师信号变得过时。这表明**共演化是 PMD 增益的主要来源**。

### 4.4 跨尺度记忆迁移

PMD 从 Qwen3-8B 提炼的记忆可以迁移到不同规模的模型（1.7B–32B）。关键发现：

- 所有规模上，记忆增强推理均超过无记忆基线
- **共演化记忆始终优于固定策略记忆**，说明联合优化策略和记忆提升了迁移质量
- 更深检索（top-5 > top-3 > top-1）带来单调一致的改善
- **记忆增益足以抵消模型规模差异**：4B 模型 + top-5 检索可超越 8B 无记忆模型

### 4.5 测试时扩展特性

PMD 的一个重要发现是其对测试时计算的扩展性优于 SDPO：

- PMD 在 k=1 时已领先 SDPO 2–5pp，差距在 k=16 时扩大至 7–10pp
- PMD 随着采样数 k 增加持续改善，而 SDPO 早期即饱和
- PMD 的 **verifier headroom**（maj@k 到 best@k 之间的间隙）是 SDPO 的 2–4 倍
- 在 Material 子域上，SDPO 的 verifier headroom 完全消失（best@16 = maj@16），这是模式坍缩的直接特征

交叠问题集分析显示，PMD 解决的问题集合比 SDPO 大 9–14%，而 SDPO 独有解决的问题仅 2–4%，确认 PMD 扩大了有用的答案空间覆盖，而非仅仅转移已解决实例。

### 4.6 额外分析

**记忆层次消融**：完整的三层记忆（经验 + 洞察 + 行为）蒸馏入策略后取得最佳效果。仅使用经验记忆的效果最差，加入洞察记忆有显著提升，再加入行为记忆达到最优。

**记忆内化验证**：即使去掉所有记忆提示，仅解码学生模型生成的内容，也能观察到 PMD 训练后的模型在推理中使用更多过程性术语（"策略"、"教训"、"行为"），说明程序性知识已被内化到模型参数中。

---

## Ch5 代码实现与工程技术

### 5.1 算法流程

PMD 的训练循环如 Algorithm 1 所述：

1. 对每个 batch 中的问题，从当前策略采样 J 个 rollout
2. 获取奖励和环境反馈
3. 对每个问题更新经验记忆（记录成功/失败的推理链）
4. 对每个问题进行自我反思，更新洞察记忆（提取策略和教训）
5. 对训练问题进行语义聚类，更新全局行为记忆
6. 为每个问题组装教师记忆上下文（经验+洞察+已检索行为）
7. 计算 PMD 损失并更新策略

### 5.2 工程实现要点

- **基础设施**：8×H200-143GB GPU，FSDP 跨 8 GPU 分片策略，SGLang v0.4 提供 rollout 服务
- **记忆提取器**：在 GPU 7 上使用 bf16 模式（mem-fraction-static=0.2）部署与策略同骨干的学生-对比式记忆提取器
- **行为记忆演化**：使用 GPT-5.4 进行行为提炼（warm-up 后每步最多调用一次）
- **嵌入与检索**：Qwen3-Embedding-0.6B 用于行为聚类和稠密检索

### 5.3 超参数

| 超参数 | SciKnowEval | LiveCodeBench |
|--------|:---:|:---:|
| Learning Rate | 1×10⁻⁵ | 1×10⁻⁶ |
| Train Batch Size | 32 | 32 |
| Rollouts per Prompt | 8 | 8 |
| Max Prompt Length | 2,048 | 2,048 |
| Max Response Length | 8,192 | 8,192 |
| Total Epochs | 20 | 20 |
| Validation Interval | 每 5 步 | 每 5 步 |
| SDPO α | 0.5 | 1.0 |
| Behavior Retrieval Top-K | 3 | 3 |
| Sampling | T=1.0, top-p=1.0 | T=1.0, top-p=1.0 |

---

## Ch6 局限性与讨论

### 6.1 局限性

1. **计算开销**：PMD 比 SDPO 引入了额外的记忆构建和检索开销——需要嵌入模型进行行为聚类、LLM 进行行为提炼、以及更大的教师 prompt 上下文。文中使用的 GPT-5.4 做行为提炼在 warm-up 后每训练步调用一次，对于大规模训练可能成本较高。

2. **任务域限制**：实验仅在两个 domain（科学多选题和代码生成）上验证。对于开放式生成任务（如创意写作）、需要真实人类偏好的任务（RLHF 场景），PMD 的有效性尚未验证。

3. **记忆管理复杂度**：三层记忆层次需要多个超参数（聚类频率、检索深度 K、记忆刷新策略等），调优成本可能较高。行为记忆的质量高度依赖于聚类算法的效果和提炼 LLM 的能力。

4. **行为记忆的 LLM 依赖**：行为提炼依赖于 GPT-5.4，引入了外部 API 依赖。如果行为提炼的质量下降或 API 不可用，整体训练可能受影响。

### 6.2 未来方向

1. **扩展到更大模型**：文中展示了跨尺度记忆迁移（1.7B–32B），但在更大模型（70B+）上的表现值得进一步研究。

2. **多任务迁移**：PMD 的行为记忆能否在不同类型的任务（如数学推理 + 代码生成）之间迁移，是一个有趣的问题。

3. **非可验证 reward 场景**：将对齐扩展到需要人类偏好判断的场景，研究 PMD 在 RLHF 中的适用性。

4. **更高效的记忆管理**：探索自适应记忆压缩和选择性遗忘策略，减少 memory bank 规模膨胀。

### 6.3 延伸阅读

- **SDPO**（Self-Distillation Policy Optimization）：PMD 的基础框架，使用 on-policy 蒸馏实现密集监督
- **GRPO**（Group Relative Policy Optimization）：使用组内相对奖励进行策略优化的 RLVR 方法
- **MemGPT**：推理时的虚拟上下文管理系统
- **SkillRL / Skill-SD**：将经验蒸馏为技能并在推理时复用的相关工作
- **OPCD**（On-Policy Context Distillation）：将训练时上下文蒸馏入模型参数的类似方法

---

*本报告基于 arXiv:2607.01480 "Procedural Memory Distillation: Online Reflection for Self-Improving Language Models" 撰写。所有数据均来自论文原文 Table 2、Table 3 及正文。*