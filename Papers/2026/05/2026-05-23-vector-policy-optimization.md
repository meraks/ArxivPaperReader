# Vector Policy Optimization: Training for Diversity Improves Test-Time Search



---

> **作者**: Ryan Bahlous-Boldi, Isha Puri, Idan Shenfeld, Akarsh Kumar, Mehul Damani, Sebastian Risi, Omar Khattab, Zhang-Wei Hong, Pulkit Agrawal
> **机构**: MIT, Improbable AI Lab, MIT-IBM Computing Research Lab, Sakana AI
> **发布日期**: 2026-05-21
> **arXiv链接**: https://arxiv.org/abs/2605.22817
> **PDF链接**: https://arxiv.org/pdf/2605.22817
> **代码链接**: 暂无公开代码仓库（论文发布日期为2026年5月21日，代码可能后续发布）
> **关键词**: Reinforcement Learning, LLM Post-Training, Vector Rewards, Test-Time Search, GRPO, Diversity, Policy Optimization

---

## 一、研究背景与动机

### 1.1 核心矛盾：训练收敛性 vs. 推理多样性

当前大语言模型（LLM）的后训练范式存在一个根本性的矛盾：

- **训练阶段**：标准的强化学习后训练方法（如 PPO、DPO、GRPO）优化一个**预先指定的标量奖励**（scalar reward），驱动模型向一个狭窄的高概率响应分布收敛。这导致模型输出的**低熵分布**——重复采样产生近乎相同的答案。
- **推理阶段**：现代推理管线（如 pass@k、best-of-n、AlphaEvolve 等进化搜索方法）恰恰需要**多样化的候选方案**，以便通过下游奖励函数进行筛选。

> "The mismatch is stark: we train for convergence, then ask for exploration at test time."
> （矛盾显而易见：我们为收敛而训练，却在测试时要求探索。）

### 1.2 为什么标量奖励是"有损压缩"？

大多数"标量奖励"实际上是对**向量奖励**的聚合。例如：

| 领域 | 标量奖励 | 向量奖励组成 |
|------|---------|------------|
| 代码生成 | 单一通过率 | 每个测试用例的通过/失败 (r ∈ {0,1}^d) |
| 多跳问答 | 加权求和 | 4个跳引用 + 答案F1 (r ∈ ℝ⁵) |
| 工具调用 | 平均F1 | 格式 + 工具名 + 参数键 + 参数值 F1 (r ∈ ℝ⁴) |
| 聊天质量 | 加权和 | 有用性 + 安全性 + 简洁性 |

标量聚合会**丢失关键的权衡信息**。例如，方案 A = [0.9, 0.4, 0.7]（高正确性，低效率）和方案 B = [0.7, 0.9, 0.5]（中等正确性，高效率）在标量下可能被排序，但向量表示揭示了它们实际上解决了不同的问题。

### 1.3 现有多样性方法的局限

- **DPP 采样、温度调节**：将多样性视为**事后推理技巧**（post-hoc inference trick），而非训练目标
- **Multi-RLVR**：多答案训练但梯度仍推向同一个标量最优点，多样性在训练中**坍塌**
- **高采样数 GRPO**：即便给 3 倍计算资源，仍因模式坍塌而无法匹配 VPO

---

## 二、核心贡献

本文提出 **Vector Policy Optimization (VPO)**，一个简洁而深刻的算法，主要贡献包括：

1. **重新定义后训练目标**：当存在测试时搜索（test-time search）时，RL 后训练的目标不应是收敛到单一最佳响应，而应是**最大化一组有竞争力解的多样性**。这是一个范式的根本转变。

2. **多答案链（Multi-Answer Chains）**：在**单次自回归展开**中生成多个候选方案（m=3），后续候选可以关注先前生成的候选，使多样性成为**显式的上下文内机制**（in-context mechanism）。

3. **随机奖励标量化（Stochastic Reward Scalarization）**：用 **Dirichlet 分布**上的权重采样替代固定权重，直接优化**期望 best-of-N** 目标，这是 GRPO 优势估计器的**即插即用替换**。

4. **全面实验验证**：在4个任务域（Maze、MuSiQue、EUREQA、ToolRL）和1个大规模扩展研究（LiveCodeBench）上，VPO 在搜索预算增长时**持续拉开与所有基线的差距**。

5. **深刻洞察**：证明了单独使用多答案或单独使用随机标量化都**不够**——两者的结合才是关键。

---

## 三、方法详解

### 3.1 整体架构

VPO 的核心思想是：**将"标量奖励"还原为"向量奖励"，并在训练时通过随机标量化来鼓励模型学习覆盖不同权衡空间的解。**

```
输入提示 (Prompt)
    ↓
从策略中采样 N 组解（每组含 m=3 个候选）
    ↓
计算每个候选的向量奖励: [r₁, r₂, ..., rₖ]
    ↓
从 Dirichlet 分布采样 K 组权重: w ~ Dir(α)
    ↓
对每组权重，选择该组中加权得分最高的解
    ↓
计算集合级别奖励:
    R̂(S) = (1/K) Σₖ max_{s∈S} w⁽ᵏ⁾ᵀr(x, s)
    ↓
替换 GRPO 优势估计器，执行策略梯度更新
```

### 3.2 关键技术一：多答案链（Multi-Answer Chains）

#### 核心机制
- 模型在**一次自回归前向传播**中生成 m=3 个候选方案
- 候选之间用分隔符标记（delimiter tokens）分隔
- **关键优势**：第 2、3 个候选可以**看到前面已生成的候选**，因此能识别已覆盖的解空间区域并主动转向不同区域

#### 为什么单独不够？
Multi-answer 链提供了多样性的**能力**（capacity），但缺少多样性的**激励**（incentive）。因为梯度仍然推向同一个固定的标量最优点，所有位置的输出最终坍塌到相同模式。

### 3.3 关键技术二：随机奖励标量化（Stochastic Reward Scalarization）

#### 数学形式

**集合级别奖励目标**：

```
R(S) = E_{w~Dir(α)} [ max_{y∈S} wᵀr(x, y) ]
```

其中：
- S 是一个候选方案集合
- w 是从 Dirichlet 分布 Dir(α) 采样的权重向量
- r(x, y) 是输入 x 对应输出 y 的向量奖励
- α=1 表示权重在单纯形上均匀分布

**Monte-Carlo 估计**：

```
R̂(S⁽ᵍ⁾) = (1/K) Σₖ₌₁ᴷ max_{s∈S⁽ᵍ⁾} w⁽ᵏ⁾ᵀr(x, s)
```

- K 组标量化权重在 G 组 rollout 之间**共享**，确保可比较的评估
- 这直接优化的是**期望 best-of-N** 性能

#### 为什么有效？

- 使用固定 w* 训练的策略会**激进地**收敛到单一策略，早期就抑制了替代方案
- VPO 对**任何**采样 w 下得分良好的候选给予正梯度，使多种推理策略存活更久
- 本质上是在训练时就模拟了测试时搜索的"多样性需求"

### 3.4 向量奖励结构

论文在5个任务域中定义了具体的向量奖励：

| 任务 | 模型 | 奖励维度 | GRPO 的标量聚合方式 |
|------|------|---------|-------------------|
| **Maze**（9×9导航） | Qwen3-4B | r ∈ ℝ⁴：完成度 + 金币 + 钻石 + 岩浆回避 | 均匀平均 |
| **MuSiQue**（多跳QA） | Qwen3-1.7B | r ∈ ℝ⁵：4个跳引用 + 答案F1 | (跳数之和 + 3×答案F1) / 7 |
| **EUREQA**（链式推理） | Qwen3-8B | r ∈ {0,1}⁵：每实体精确匹配 | 均匀平均 |
| **ToolRL**（函数调用） | Qwen3-1.7B | r ∈ ℝ⁴：格式 + 工具名 + 参数键 + 参数值 F1 | 均匀平均 |
| **LiveCodeBench**（编码） | Qwen2.5-Coder-7B | r ∈ {0,1}^d：每测试用例通过率 | 均匀平均 |

### 3.5 VPO 与 GRPO 的关系

VPO 本质上是 **GRPO 的优势估计器的替换**：

- **GRPO**：advantage = r_scalar - mean(r_scalar)
- **VPO**：advantage = R̂(S) - baseline，其中 R̂(S) 基于集合级别随机标量化

代码层面的修改非常小，这使得 VPO 可以**轻松集成到现有的 GRPO 训练框架中**。

---

## 四、实验结果

### 4.1 实验设置

- **基线方法**：GRPO, Multi-RLVR, Random-Weighting GRPO, Max-at-K Training, MaxRL, Goal-Conditioned GRPO, GDPO
- **评估指标**：best@k（在 k 个候选中选最好的）和 diversity（多样性）
- **评估任务**：4个核心任务 + 1个大规模扩展研究
- **训练框架**：基于 OpenRLGRPO 框架实现

### 4.2 主要结果

#### MuSiQue（多跳问答，Qwen3-1.7B）

| 方法 | best@3 | best@5 | best@10 | best@30 | 多样性 |
|------|--------|--------|---------|---------|--------|
| GRPO | 0.711 | 0.716 | 0.721 | 0.728 | 0.054 |
| Max-at-K | 0.757 | 0.768 | 0.783 | 0.802 | 0.175 |
| Multi-RLVR | 0.599 | 0.616 | 0.627 | 0.633 | 0.814 |
| **VPO** | **0.742** | **0.780** | **0.809** | **0.832** | **0.587** |

**关键观察**：
- GRPO 的 best@30 仅比 best@3 高 2.4%（0.711→0.728），表明采样更多几乎无用
- VPO 的 best@30 比 best@3 高 12.2%（0.742→0.832），搜索预算得到充分利用
- VPO 在 best@10 和 best@30 上都超越了专门优化 best@k 的 Max-at-K

#### Maze（9×9导航，Qwen3-4B）

| 方法 | best@3 | best@5 | best@10 | best@30 | 多样性 |
|------|--------|--------|---------|---------|--------|
| GRPO | 0.432 | 0.432 | 0.432 | 0.432 | 0.003 |
| **VPO** | 0.512 | 0.564 | **0.591** | **0.593** | **1.006** |

**令人震惊**：GRPO 在 best@3 到 best@30 完全不变（0.432），多样性仅 0.003，说明模型输出几乎完全相同。VPO 的多样性达到 1.006，是 GRPO 的 335 倍。

### 4.3 消融实验

#### 关键发现一：两个组件缺一不可

| 配置 | 多答案？ | 随机标量化？ | 结果 |
|------|:------:|:----------:|------|
| GRPO | ✗ | ✗ | 模式坍塌 |
| Multi-RLVR | ✓ | ✗ | 多样性在训练中坍塌 |
| Random-Weighting GRPO | ✗ | ✓ | 不稳定 |
| **VPO** | **✓** | **✓** | **稳定且多样** |

- Multi-RLVR 的奖励空间多样性在 Maze、MuSiQue、ToolRL 上**训练过程中坍塌**
- 多答案提示提供的是多样性的*能力*而非*激励*
- 随机标量化给不同位置提供了专门化的激励

#### 关键发现二：不是更多评估信号或标准化的问题

| 方法 | best@3 | E_w[best@3] |
|------|--------|-------------|
| GRPO (n=8) | 0.741 | 0.830 |
| GRPO (n=24, 3×计算量) | 0.763 | 0.841 |
| **VPO (n=8)** | **0.779** | **0.856** |

- GDPO（每维度标准化）与 GRPO 轨迹几乎一致 → 问题不在标准化
- **3倍计算量的 GRPO 仍然低于 1倍计算量的 VPO**

#### 关键发现三：目标条件化（Goal Conditioning）不工作

| 方法 | best@3 | best@6 |
|------|--------|--------|
| Goal-Cond. w=w* | 0.205 | 0.205 |
| Goal-Cond. w~Dir(1) | 0.205 | 0.205 |
| **VPO** | **0.512** | **0.576** |

尽管模型可以显式访问权重向量 w，但仍**无法可靠地将文本编码的偏好转化为有效行为**。这是一个重要发现：直接告诉模型"这次关注正确性，下次关注效率"并不如 VPO 的隐式激励有效。

#### LiveCodeBench 扩展研究

使用 Qwen2.5-Coder-7B-Instruct 在 24,269 个训练问题和 279 个评估问题（严格时间切割）上：
- VPO 在 LiveCodeBench 上展现出与核心任务一致的改进趋势
- 证明了 VPO 在**大规模、真实世界编码任务**上的可扩展性

---

## 五、关键发现与洞察

### 5.1 范式转变：后训练的目标不是找到最优解

> "In this setting, the role of RL post-training should not be to converge on a single best response, but to maximize the diversity of a set of competent solutions."

这一观点可能重塑我们对 RL 后训练的理解。当推理时搜索成为标配（如 AlphaEvolve、o-series 模型），训练的目标应当从"找到最优"转向"培养多样化的能力组合"。

### 5.2 标量奖励是信息瓶颈

论文揭示了一个被忽视的事实：大多数"标量"奖励实际上是对向量奖励的有损压缩。这种压缩在训练时丢弃了**权衡结构**（trade-off structure），导致模型无法学习覆盖不同权衡空间的解。

### 5.3 多样性需要显式的训练激励

仅靠采样策略（高温、DPP）或架构支持（多答案链）无法保证多样性。VPO 证明了多样性需要**在训练目标中显式编码**——随机标量化提供了这种激励。

### 5.4 计算效率：1倍 VPO > 3倍 GRPO

这是论文最实用的发现之一。在相同计算预算下，VPO 显著优于 GRPO；即使 GRPO 获得 3 倍计算资源，仍然不如 VPO。这意味着 VPO 不仅效果好，而且**计算更高效**。

### 5.5 文本条件化的局限性

目标条件化（Goal Conditioning）的失败揭示了一个重要问题：当前 LLM 无法可靠地将"请关注维度 A"这样的文本指令转化为实际的行为偏好。VPO 通过训练时的隐式激励避开了这个问题。

### 5.6 与 GRPO 的兼容性

VPO 仅替换 GRPO 的优势估计器，对训练框架的改动极小。这使得它可以**零成本集成**到现有的 GRPO 训练流程中。

---

## 六、个人评述

### 6.1 优势

1. **问题定义精准**：抓住了"训练收敛 vs. 推理多样性"这个被社区广泛关注但未系统解决的核心矛盾
2. **方法简洁优雅**：仅用两个组件（多答案链 + 随机标量化）就解决了问题，且是 GRPO 的即插即用替换
3. **消融实验彻底**：设计了7个基线方法，逐一排除了替代解释（更多信号？更好的标准化？显式条件化？）
4. **理论动机清晰**：基于向量优化和 Pareto 前沿的理论框架，不是纯工程 trick
5. **实用价值高**：在计算预算更少的情况下超越 GRPO，对工业界有直接吸引力

### 6.2 局限性

1. **奖励维度依赖**：需要任务天然具有可分解的向量奖励结构。对于确实只有单一标量奖励的任务（如某些人类偏好评估），可能需要人工构造奖励维度
2. **多答案链的效率**：m=3 个候选在一次前向传播中生成，虽然共享 prefix 但仍增加了推理成本
3. **超参数敏感度**：论文未充分讨论 Dirichlet 分布参数 α 和标量化采样数 K 的敏感性
4. **大规模验证不足**：LiveCodeBench 的结果描述相对简略，缺少完整的数据表格
5. **代码未公开**：截至精读日期（2026-05-23），论文代码尚未公开发布，限制了可复现性

### 6.3 未来方向

- **与 inference-time scaling 的结合**：VPO 训练的模型与 beam search、MCTS 等搜索算法的协同效果
- **多轮对话中的应用**：向量奖励在多轮对话中的定义（有用性、安全性、信息量、一致性等）
- **自动奖励分解**：从标量奖励自动发现有意义的向量分解
- **与 DPO 的结合**：将随机标量化思想引入偏好学习

### 6.4 总体评价

VPO 是一篇**高质量、高影响力**的工作。它不是在现有范式上做增量改进，而是提出了一个**重新思考 RL 后训练目标**的新视角。方法的简洁性（GRPO 的即插即用替换）和实验的全面性（4 个任务 + 消融 + 大规模扩展）都达到了顶会标准。核心洞察——"当搜索可用时，训练应为多样性而非收敛"——可能会对整个 RL 后训练领域产生深远影响。

---

## 七、参考文献

1. **VPO 论文**: Bahlous-Boldi, R., et al. "Vector Policy Optimization: Training for Diversity Improves Test-Time Search." arXiv:2605.22817, 2026.
2. **GRPO**: Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300, 2024.
3. **AlphaEvolve**: Novak, A., et al. "AlphaEvolve: A coding agent for scientific and algorithmic discovery." Google DeepMind, 2025.
4. **PPO**: Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.
5. **DPO**: Rafailov, R., et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023.
6. **Multi-RLVR**: 相关多答案训练方法
7. **GDPO**: 带每维度标准化的 GRPO 变体

---

*本精读报告生成于 2026-05-23，基于 arXiv:2605.22817v1 版本。*
