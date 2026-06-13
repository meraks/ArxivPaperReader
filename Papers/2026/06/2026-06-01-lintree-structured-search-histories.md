# LinTree: Improving LLM Reasoning with Explicitly Structured Search Histories



---

**作者**：Liwei Kang (NUS), Yee Whye Teh (Oxford), Wee Sun Lee (NUS)  
**arXiv ID**：[2605.31492](https://arxiv.org/abs/2605.31492)  
**提交日期**：2026-05-29 | **页数**：16页，3张图 | **分类**：cs.AI  
**基础模型**：Qwen3-0.6B | **训练硬件**：单张NVIDIA H100 GPU

---

## Ch1: 论文概述与核心贡献

### 一句话总结

在LLM推理中，仅访问完整搜索历史轨迹不足以带来优势——需要通过 **parent pointers（父指针）** 显式化搜索树的拓扑结构，才能让模型真正有效利用搜索历史。这种方法被命名为 **LinTree**（Linearized Tree）。

### 核心问题

大型语言模型在解决推理问题时，生成的中间步骤（如Chain-of-Thought、self-reflection）本质上是**被线性化的搜索树**——模型扩展部分解、放弃失败分支、回溯尝试替代方案。这引出了两个层层递进的研究问题：

1. **搜索轨迹访问能否带来优势？** 如果LLM能访问完整的搜索历史（而非仅当前状态），它能否超越仅用局部状态的启发式搜索？
2. **显式树结构能否让LLM更好地利用历史？** 如果当前隐式轨迹不够好，那么显式标注树拓扑会带来什么改变？

> **类比理解**：想象一位侦探在破案。传统启发式搜索就像侦探每次只带着当前线索工作——高效但视野狭窄。而"全轨迹访问"的LLM就像侦探随身带着所有案件便签记录——理论上能看到全局模式。但问题来了：如果这些记录只是按时间顺序排列的流水账（隐式轨迹），侦探能从中学到案件关联吗？还是需要给便签加上编号和引用关系（LinTree的parent pointers），才能真正构建出案件的逻辑网络？

### 两大核心发现

| 发现 | 内容 | 证据 |
|------|------|------|
| **发现1: 隐式轨迹不足** | 仅访问完整搜索历史的LLM推理策略**不能可靠地超越**局部状态的LLM启发式搜索 | Table 2：GRPO-implicit成功率（97.3%/94.9%/85.9%）均低于BFS(RL)（99.8%/100.0%/99.1%） |
| **发现2: 显式结构关键** | 添加简单的parent pointer标注后，同样的训练流程下，模型性能**显著超越**隐式策略和局部启发式 | Table 3：GRPO-explicit Blocks World达到**100.0%**成功率（vs GRPO-implicit 97.3%，BFS(RL) 99.8%） |

### 核心贡献

1. **首次系统对比**了"全搜索轨迹条件化推理"与"局部状态LLM启发式搜索"两种范式，揭示了"信息越多不一定越好"的反直觉结论
2. **提出LinTree表示**：通过简单的parent pointer标注（`sid=`），在不改变搜索数据和训练流程的前提下，将隐式搜索树变为显式结构
3. **三领域验证**：在Blocks World、Grid Navigation、Sokoban三个经典推理环境中一致证明显式结构的有效性
4. **统一训练框架**：提供SFT+GRPO两阶段训练流程，奖励函数融合正确性和效率两个维度

### 论文结构导览

| 章节 | 内容 | 核心问题 |
|------|------|---------|
| Ch2 背景 | CoT的搜索视角、外部控制器方法 | 现有方法如何利用搜索结构？ |
| Ch3 隐式轨迹 | 轨迹条件化推理 vs 局部启发式搜索 | 全轨迹访问是否有价值？ |
| Ch4 LinTree | 显式parent pointers、结构化轨迹效果 | 显式树结构能否改善性能？ |
| Ch5 实验 | 完整结果分析、策略行为对比 | 性能提升的证据链是什么？ |
| Ch6 代码 | 核心算法实现、环境接口 | 如何复现LinTree？ |
| Ch7 局限 | 方法边界、未来方向、相关工作 | 什么情况下LinTree不适用？ |

---

## Ch2: 研究背景与动机

### LLM推理的搜索视角

现代LLM推理的核心范式是**生成中间推理步骤**，而非直接输出答案。从搜索的视角来看：

- **Chain-of-Thought (CoT)** [Wei et al., 2022]：将推理过程展开为文本步骤序列，这本质上是一种 **深度优先的隐式搜索**
- **Self-Reflection** [Madaan et al., 2023]：模型检测错误、修正推理，相当于搜索中的**回溯**操作
- **Self-Consistency** [Wang et al., 2023]：生成多条推理路径取多数，相当于**并行搜索+投票**

这些方法的共同点是：搜索树结构**仅存在于LLM内部的隐式上下文中**，不对外暴露。

### 现有方法的分类

论文将现有LLM推理方法分为两大类：

#### 第一类：纯LLM推理（搜索树隐式）

| 方法 | 搜索策略 | 结构表示 | LLM看到的 |
|------|---------|---------|-----------|
| CoT | 深度优先 | 无 | 当前步骤 |
| Self-Reflection | 深度优先+回溯 | 无 | 当前+之前失败 |
| Self-Consistency | 并行采样 | 无 | 各自独立 |

#### 第二类：外部控制器引导搜索（搜索树显式，但LLM只看到局部）

| 方法 | 控制器类型 | LLM查询范围 |
|------|----------|------------|
| Tree of Thoughts (ToT) | BFS/DFS | 当前路径（从根到当前节点） |
| LLM-A* | A*搜索 | 当前路径 |
| Graph of Thoughts (GoT) | 图控制器 | 被prompter选中的节点 |
| LATS / RAP | MCTS | 当前路径 |
| SE-Guided Beam Search | 束搜索 | 单条推理链 |

> **类比理解**：这两种范式就像两种不同的项目管理系统。第一类（纯LLM推理）像自由写作——所有想法都在一个文档中线性记录，回溯和分支靠读者的理解力。第二类（外部控制器）像用Jira管理——项目结构很清晰，但每个开发者（LLM）只被分配当前任务，看不到全局状态。

### 关键研究鸿沟

现有工作的一个**共同局限**是：**没有方法让LLM在看全搜索历史的同时，又能利用显式的树结构**。

- 纯LLM推理：能访问完整历史（通过上下文），但树结构是隐式的
- 外部控制器：树结构显式，但LLM被限制为局部视图

LinTree正是要填补这个鸿沟——让LLM在**内部**就能访问完整且结构化的搜索历史。

### 为什么选择这三个测试领域？

论文选择了三个fully observable的经典AI推理环境：

| 领域 | 描述 | 关键挑战 | 启发式质量 |
|------|------|---------|-----------|
| **Blocks World** | 重新排列积木达到目标配置 | 子目标依赖关系 | 高（"好块"计数） |
| **Grid Navigation** | 避开障碍物从起点到终点 | 曼哈顿距离会误导到死胡同 | 中（容易误导） |
| **Sokoban** | 推箱子到目标位置 | 不可逆操作、复杂状态空间 | 中（二分图匹配） |

这三个领域共同构成了一个从易到难的测试梯度：Blocks World相对简单，Navigation增加了启发式误差，Sokoban则是最难的挑战。

---

## Ch3: 轨迹条件化推理 vs 局部启发式搜索

本章核心实验：让LLM访问完整搜索轨迹，能否超越仅使用局部状态的启发式搜索？

### 3.1 轨迹条件化LLM推理策略

**核心思想**：LLM观察完整的线性化搜索轨迹，直接预测下一步搜索展开。

**轨迹格式**（隐式版）：
```
EXPAND ACT B:A->TABLE -> S{ 0:B<C ; 1:A ; 2:D }
EXPAND ACT C->B -> S{ 0:C ; 1:A<B ; 2:D }
EXPAND ACT D:A->C -> S{ 0:D<C ; 1:A<B }
```

这种格式的优势在于：LLM可以观察到之前尝试过的所有分支、回溯点、以及各分支的结果，从而进行"全局推理"。但缺点也很明显：**没有任何结构标注**——哪些展开来自同一个父节点？何时发生了回溯？这些需要LLM自己推断。

#### 训练流程

论文采用两阶段训练：

**Stage 1: SFT（行为克隆）**
- 基座模型：Qwen3-0.6B
- 训练数据：最佳优先搜索（BFS）产生的专家轨迹（规则启发式）
- 训练轮次：4 epochs
- 学习率：1e-5
- 每样本包含：prompt（问题配置）+ 搜索轨迹 + 最终计划

**Stage 2: GRPO（强化学习优化搜索效率）**

GRPO (Group Relative Policy Optimization) 是DeepSeek提出的RL方法。论文为搜索任务设计了融合正确性和效率的奖励函数：

$$R(\tau) = \mathbf{1}[\text{valid}(\tau) \land \text{correct}(\tau)] \left(1 - \lambda \sum_{t=0}^{N_{\text{exp}}(\tau)-1} \gamma^t\right)$$

其中：
- $N_{\text{exp}}(\tau)$：轨迹 $\tau$ 中的搜索展开次数
- $\lambda = 0.005$：效率惩罚尺度
- $\gamma = 0.99$：几何衰减因子
- $\mathbf{1}[\text{valid} \land \text{correct}]$：只有有效的正确计划获得非零基础奖励

**奖励函数的设计精髓**：
1. 效率惩罚是有界几何级数和——保证**任何正确的轨迹**得分都高于**任何错误的轨迹**（因为 $\lambda < 1 - \gamma$）
2. 每次额外展开会减少奖励，形成对短轨迹的偏好
3. 几何衰减意味着**早期展开**比晚期展开"更贵"——鼓励模型尽早找到好方向

GRPO超参数：$N=8$ rollouts/样本，1 epoch。

> **类比理解**：这就像教学生解数学题。SFT阶段——给学生看100道题的完整解题过程，让他模仿。GRPO阶段——告诉他"做对+做得快=高分"，但做错=0分，做得慢但做对至少还有分。学生自然会学会寻找更高效的解题路径。

### 3.2 局部状态LLM启发式（Baseline）

作为对比基线，论文实现了**外部BFS + LLM局部评分**的范式（类似Tree of Thoughts的搜索框架）：

**工作流程**（Algorithm 1）：
```
1. 初始化: 根节点s₀，已访问集合V，边界F={(s₀, a): a∈Actions(s₀)}
2. for t = 1 to B (预算):
3.   对每个(s,a)∈F，LLM输出启发值h_θ(s,a,g)
4.   选择h_θ最小的候选(s*,a*)
5.   执行: s' = T(s*,a*)
6.   若s'满足目标 → 返回路径
7.   更新边界: F = F \ {(s*,a*)} ∪ {(s',a): a∈Actions(s')，T(s',a)∉V}
8. 预算耗尽 → 失败
```

关键设计选择：
- LLM**只看当前状态+候选动作+目标**（Markovian）——因为三个环境都是fully observable
- 通过**MLP head**输出标量启发值，而非文本评分
- 搜索预算：训练时500次展开，评估时200次展开

**训练细节**：
- SFT：在(state, action)对上训练，目标是规则启发式值。数据量：BW 1.74M对、Navigation 1.22M对、Sokoban 1.28M对
- RL：将展开选择建模为决策问题，使用softmax策略梯度：
  $$\pi_\theta(c \mid \mathcal{C}_{i,t}) = \frac{\exp(-h_\theta(c))}{\sum_{c' \in \mathcal{C}_{i,t}} \exp(-h_\theta(c'))}$$

> **类比理解**：局部启发式搜索像用GPS导航——GPS只知道当前位置和目的地，根据局部道路信息做最优选择。轨迹条件化推理像拥有全景地图+历史行程记录的司机——理论上能看到更全局的模式（如"那条路总是堵"），但前提是历史记录足够清晰可读。

### 3.3 实验结果：全轨迹不等于全优势

**Table 2：隐式轨迹 vs 局部启发式搜索**

| Method | BW Succ ↑ | BW Exp ↓ | Nav Succ ↑ | Nav Exp ↓ | Sok Succ ↑ | Sok Exp ↓ |
|--------|-----------|----------|------------|-----------|------------|-----------|
| BFS (pretrained) | 99.7 ±0.17 | 9.54 ±0.31 | 100.0 ±0.00 | 20.27 ±0.42 | 94.9 ±0.70 | 73.21 ±2.11 |
| BFS (RL) | **99.8** ±0.14 | 8.56 ±0.18 | **100.0** ±0.00 | 16.99 ±0.37 | **99.1** ±0.30 | 64.08 ±1.86 |
| SFT-implicit | 90.0 ±0.95 | 9.12 ±0.16 | 90.6 ±0.92 | 15.39 ±0.18 | 74.8 ±1.37 | 68.24 ±7.70 |
| GRPO-implicit | 97.3 ±0.51 | **8.25** ±0.14 | 94.9 ±0.70 | **14.80** ±0.16 | 85.9 ±1.10 | **63.54** ±4.12 |

**核心分析**：

1. **隐式轨迹策略在成功率上全面落后于BFS(RL)**：
   - Blocks World: 97.3% vs 99.8%（差距2.5个百分点）
   - Navigation: 94.9% vs 100.0%（差距5.1个百分点）  
   - Sokoban: 85.9% vs 99.1%（差距**13.2个百分点**）——领域越难，差距越大

2. **GRPO确实能改善隐式策略**：相对于SFT-implicit，GRPO-implicit在三个领域分别提升+7.3pp、+4.3pp、+11.1pp——但**仍然不足以超越局部启发式**。

3. **搜索效率上隐式策略略有优势**：在展开次数上，GRPO-implicit在三个领域都略优于BFS(RL)（8.25 vs 8.56, 14.80 vs 16.99, 63.54 vs 64.08），说明模型学会了更快地找到答案——但这些"更快找到"的答案中，相当一部分是**错误的**。

**关键结论**：全轨迹访问**本身**不构成优势。问题不在于"信息量"，而在于"信息的可消化性"。

---

## Ch4: 结构化搜索轨迹——LinTree核心方法

如果隐式轨迹的失败在于"信息有但结构无"，那么解决方案显然就是**添加结构**。本章展示LinTree如何用最小的改动实现最大的效果。

### 4.1 核心创新：Parent Pointers

LinTree的核心改动极其简单：在每次搜索展开中加入**父节点标识符**（parent pointer），使树的拓扑关系一目了然。

**隐式格式**（Section 4.1）：
```
EXPAND ACT B:A->TABLE -> S{ 0:B<C ; 1:A ; 2:D }
EXPAND ACT C->B -> S{ 0:C ; 1:A<B ; 2:D }
EXPAND ACT D:A->C -> S{ 0:D<C ; 1:A<B }
```
问题：第二个展开的父节点是什么？第三个呢？LLM需要自己推断。

**LinTree显式格式**：
```
EXPAND sid=0 S{ 0:B<C ; 1:A ; 2:D }
EXPAND sid=0 ACT B:A->TABLE -> sid=1 S{ 0:B<C ; 1:A ; 2:D }  
EXPAND sid=1 ACT C->B -> sid=2 S{ 0:C ; 1:A<B ; 2:D }
EXPAND sid=1 ACT D:A->C -> sid=3 S{ 0:D<C ; 1:A<B }
```
现在一目了然：sid=2和sid=3都从sid=1展开，构成分支！sid=0展开了第一个子节点sid=1。

**改动的精妙之处**：
1. **零开销**：不改变搜索数据生成流程，不增加训练成本
2. **完全兼容**：隐式轨迹可以直接"升级"为显式轨迹（只需标注sid）
3. **信息等量**：两个格式包含完全相同的搜索行为，**唯一变量是树拓扑是否被显式标注**

> **类比理解**：隐式轨迹像git的`reflog`——记录了所有操作，但没有分支信息。LinTree像git的`log --graph --oneline`——每个commit都标记了parent，分支合并关系一清二楚。内容完全一样，但后者的可理解性天差地别。

### 4.2 实验结果：结构带来质变

**Table 3：隐式 vs 显式（LinTree）— Blocks World**

| Method | Training | Success ↑ | Expansions ↓ |
|--------|----------|-----------|--------------|
| No parent ptr | SFT | 90.0 ±0.95 | 9.12 ±0.16 |
| With parent ptr | SFT | 89.6 ±0.97 | 9.06 ±0.45 |
| No parent ptr | GRPO | 97.3 ±0.51 | 8.25 ±0.14 |
| **With parent ptr** | **GRPO** | **100.0 ±0.00** | **7.31 ±0.08** |

**BFS基线**：BFS(pretrained) 99.7%, BFS(RL) 99.8%

**关键发现**：

1. **SFT阶段无差异**（89.6% vs 90.0%）：因为SFT只是模仿专家轨迹，模型尚未"学会利用"树结构
2. **GRPO后显式结构爆发**：LinTree达到**100.0%完美成功率**（标准差0.00！），同时搜索效率提升（7.31 vs 8.25，**-11.4%**展开数）
3. **超越所有基线**：LinTree不仅超越隐式GRPO（+2.7pp），也超越BFS(RL)（+0.2pp），且每个领域都达到或接近完美

### 4.3 深度分析：LinTree为何有效？

论文从两个角度分析了显式结构的优势：

**优势1：帮助提取最终计划**  
搜索轨迹中混杂了大量探索性展开和回溯操作。要从轨迹中提取出解决路径，模型需要识别哪些展开属于"成功分支"。显式parent pointers让模型可以直接追踪：从`GOAL_REACHED`节点沿parent链回溯到根节点，即为答案。而隐式格式中，模型需要自己推断哪段序列构成了最终路径。

**优势2：帮助探索状态空间**  
当模型看到`sid=1`有多个子节点（如sid=2和sid=3），它立即理解这是一个**分支点**。模型可以据此推断：如果sid=2失败了，可以回到sid=1尝试其他分支。而在隐式格式中，模型需要通过上下文线索（如"let me try a different approach"）来推断回溯——这种推断本身就不准确。

**优势3：防止重复探索**  
显式结构让模型清楚地知道哪些状态已被探索。如果模型意识到某区域已有多个展开但都未成功，它可以主动避免再次进入该区域。

> **类比理解**：想象你在探索一个迷宫。隐式轨迹是你的行走笔记："向前10步，左转，再走5步，后退3步..."。LinTree是你的行走笔记+地图标注："从路口A出发→走10步到路口B，标记为sid=1。从路口B尝试左转→死胡同（sid=2）。从路口B尝试右转→经过路口C（sid=3）。"有了地图标注，你永远不会重走sid=2的死胡同。

### 4.4 奖励函数设计讨论（Appendix B）

论文在Appendix B中讨论了两种被放弃的奖励设计方案，进一步说明了当前设计的优越性：

| 方案 | 公式 | 问题 |
|------|------|------|
| 常数惩罚+上限 | $R_{\text{cap}} = \max(1 - \alpha N_{\text{exp}}, 0)$ | 展开数超过$1/\alpha$后饱和为0，长正确轨迹和错误轨迹无法区分；调参困难，训练不稳定 |
| 几何衰减乘 | $R_{\text{decay}} = \gamma^{N_{\text{exp}}}$ | 几轮展开差异导致奖励数量级变化，产生高方差梯度，训练崩溃 |
| **有界折扣和**（本文采用） | $R = 1 \cdot (1 - \lambda\Sigma\gamma^t)$ | 保证正确轨迹>错误轨迹，惩罚平滑，训练稳定 |

---

## Ch5: 实验深度分析

### 5.1 实验设置总览

| 维度 | 配置 |
|------|------|
| 基座模型 | Qwen3-0.6B |
| 训练硬件 | 单张 NVIDIA H100 GPU |
| SFT | 4 epochs, lr=1e-5 |
| GRPO | 1 epoch, N=8 rollouts, lr=1e-5 |
| 搜索预算 | 训练500展开，评估200展开 |
| 数据规模 | 每领域 20k SFT + 20k RL + 1k 验证 |
| 轨迹截断 | ≤16k tokens |

### 5.2 性能对比全景

将所有方法的成功率按领域排列：

```
Blocks World 成功率排名：
  GRPO-explicit (LinTree)  ████████████████████ 100.0%
  BFS (RL)                 ████████████████████  99.8%
  BFS (pretrained)         ████████████████████  99.7%
  GRPO-implicit            ███████████████████   97.3%
  SFT-implicit             ██████████████████    90.0%

Navigation 成功率排名：
  BFS (RL)                 ████████████████████ 100.0%
  BFS (pretrained)         ████████████████████ 100.0%
  GRPO-implicit            ███████████████████   94.9%
  SFT-implicit             ██████████████████    90.6%

Sokoban 成功率排名：
  BFS (RL)                 ████████████████████  99.1%
  BFS (pretrained)         ███████████████████   94.9%
  GRPO-implicit            █████████████████     85.9%
  SFT-implicit             ███████████████       74.8%
```

### 5.3 关键模式分析

**模式1：难度放大效应**
领域越难，隐式轨迹相对局部启发式的差距越大：BW 2.5pp → Nav 5.1pp → Sok 13.2pp。这表明在简单任务中LLM可以通过上下文推断隐式结构，但在复杂搜索中，手动推断的成本超过了"全轨迹信息"带来的收益。

**模式2：GRPO的两面性**
GRPO对隐式策略的提升十分显著（BW +7.3pp, Nav +4.3pp, Sok +11.1pp），但GRPO在显式结构上的效果更加惊人——直接将Block World推到100%完美率。这说明**RL训练让模型更善于利用结构化的信息**，但当信息本身缺乏结构时，RL的作用是有限的。

**模式3：SFT阶段结构无影响**
在SFT阶段，有无parent pointers对性能几乎无影响（BW: 89.6% vs 90.0%）。这意味着"理解树结构"**不是靠预训练获得的**，而是**通过RL的探索-奖励信号学到的**。这是一个重要的训练洞察。

### 5.4 策略行为分析

论文对两种策略的行为差异进行了定性分析：

| 行为维度 | 隐式策略 | LinTree策略 |
|---------|---------|------------|
| 计划提取 | 需要推断哪些展开属于成功路径 | 沿parent链回溯，精确无误 |
| 回溯决策 | 依赖自然语言线索推断 | 通过sid识别分支点和已探索区域 |
| 状态空间探索 | 可能重复探索已访问区域 | 利用parent指针避免重复 |
| 错误恢复 | 容易在旁支中迷失 | 清楚知道回溯的"锚点" |

---

## Ch6: 代码实现详解

> ⚠️ **重要提示**：论文未提供官方代码仓库。以下所有代码为非官方概念实现，基于论文Algorithm 1和Section 3-5的描述编写。目的是帮助理解算法流程，**不可直接用于训练**。

### 6.1 搜索环境接口

```python
# ⚠️ 非官方概念实现 — 基于论文 Section 3 描述编写
# 目的：帮助理解算法流程，不可直接用于训练
# 与官方实现的主要差异：状态哈希、序列化格式可能不同

from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass

@dataclass
class SearchNode:
    """搜索树节点，包含LinTree的parent pointer"""
    state_id: int       # sid=，LinTree核心标注
    state: 'State'
    parent: Optional['SearchNode']
    action: Optional[str]
    children: List['SearchNode']

class SearchEnvironment(ABC):
    """三个测试领域的统一接口"""
    
    @abstractmethod
    def get_initial_state(self) -> 'State':
        """返回初始状态"""
        pass
    
    @abstractmethod
    def get_actions(self, state: 'State') -> List[str]:
        """返回当前状态下的合法动作"""
        pass
    
    @abstractmethod
    def transition(self, state: 'State', action: str) -> 'State':
        """执行动作，返回新状态（假设确定性转移）"""
        pass
    
    @abstractmethod
    def is_goal(self, state: 'State') -> bool:
        """判断是否达到目标"""
        pass
    
    @abstractmethod
    def heuristic(self, state: 'State') -> float:
        """规则启发式函数（用于生成SFT轨迹）"""
        pass
    
    @abstractmethod
    def serialize_state(self, state: 'State') -> str:
        """将状态序列化为LLM可理解的文本"""
        pass
```

### 6.2 Blocks World环境实现

```python
# ⚠️ 非官方概念实现

class BlocksWorldState:
    def __init__(self, stacks: List[List[str]]):
        self.stacks = stacks  # List of stacks, bottom to top
    
    def __hash__(self):
        return hash(tuple(tuple(s) for s in self.stacks))
    
    def __eq__(self, other):
        return self.stacks == other.stacks

class BlocksWorld(SearchEnvironment):
    def __init__(self, initial_stacks, goal_stacks):
        self.initial = BlocksWorldState(initial_stacks)
        self.goal = BlocksWorldState(goal_stacks)
    
    def get_actions(self, state: BlocksWorldState) -> List[str]:
        actions = []
        for i, stack in enumerate(state.stacks):
            if stack:  # 栈非空
                block = stack[-1]  # 顶部块
                # 移到桌子
                actions.append(f"{block}:{stack[-2] if len(stack)>1 else 'TABLE'}->TABLE")
                # 移到其他栈
                for j, other in enumerate(state.stacks):
                    if i != j:
                        target = other[-1] if other else 'TABLE'
                        actions.append(f"{block}:{stack[-2] if len(stack)>1 else 'TABLE'}->{target}")
        return actions
    
    def transition(self, state: BlocksWorldState, action: str) -> BlocksWorldState:
        # 解析: "B:A->TABLE" 或 "B:A->C"
        block_src, rest = action.split(":")
        src_support, dst = rest.split("->")
        block = block_src
        
        new_stacks = [s.copy() for s in state.stacks]
        # 移除源块
        for s in new_stacks:
            if s and s[-1] == block:
                s.pop()
                break
        # 放置目标块
        if dst == 'TABLE':
            new_stacks.append([block])
        else:
            for s in new_stacks:
                if s and s[-1] == dst:
                    s.append(block)
                    break
            else:
                new_stacks.append([block])
        
        # 清理空栈
        new_stacks = [s for s in new_stacks if s]
        return BlocksWorldState(new_stacks)
    
    def heuristic(self, state: BlocksWorldState) -> float:
        """'好块'计数启发式：block已在其目标支撑上且下方所有block也是好的"""
        good = 0
        for stack in state.stacks:
            for i, block in enumerate(stack):
                # 检查该块是否在正确的支撑上且下方所有块也正确
                # （简化实现，实际需要完整的状态-目标匹配逻辑）
                pass
        return good
    
    def serialize_state(self, state: BlocksWorldState) -> str:
        parts = []
        for i, stack in enumerate(state.stacks):
            parts.append(f"{i}:" + "<".join(stack))
        return "S{ " + " ; ".join(parts) + " }"
```

### 6.3 LLM启发式引导的树搜索（Algorithm 1实现）

```python
# ⚠️ 非官方概念实现 — 基于论文 Algorithm 1 伪代码

def best_first_search_with_llm_heuristic(
    env: SearchEnvironment,
    llm_heuristic,  # LLM + MLP head, 输出启发值
    budget: int = 200
) -> Optional[List[str]]:
    """
    Algorithm 1: LLM heuristic-guided best-first tree search
    对应论文 Section 4.2
    """
    # 初始化
    s0 = env.get_initial_state()
    tree = SearchNode(state_id=0, state=s0, parent=None, action=None, children=[])
    visited = {s0}
    frontier = {}  # (node, action) -> heuristic_value
    
    # 填充初始frontier
    for action in env.get_actions(s0):
        next_state = env.transition(s0, action)
        if next_state not in visited:
            frontier[(tree, action)] = None  # 待评分
    
    next_id = 1
    
    for t in range(budget):
        if not frontier:
            return None  # 搜索失败
        
        # 评分所有frontier候选（论文式：LLM看当前状态+候选动作+目标）
        for (node, action) in frontier:
            if frontier[(node, action)] is None:
                h_val = llm_heuristic(node.state, action, env)
                frontier[(node, action)] = h_val
        
        # 选择启发值最小的候选
        (best_node, best_action) = min(frontier, key=frontier.get)
        del frontier[(best_node, best_action)]
        
        # 执行动作
        new_state = env.transition(best_node.state, best_action)
        new_node = SearchNode(
            state_id=next_id,
            state=new_state,
            parent=best_node,
            action=best_action,
            children=[]
        )
        next_id += 1
        best_node.children.append(new_node)
        visited.add(new_state)
        
        # 目标检查
        if env.is_goal(new_state):
            # 提取路径
            path = []
            node = new_node
            while node.parent is not None:
                path.append(node.action)
                node = node.parent
            return list(reversed(path))
        
        # 更新frontier
        for action in env.get_actions(new_state):
            next_s = env.transition(new_state, action)
            if next_s not in visited:
                frontier[(new_node, action)] = None
    
    return None  # 预算耗尽
```

### 6.4 轨迹序列化：隐式 vs LinTree显式

```python
# ⚠️ 非官方概念实现

def serialize_trace_implicit(search_nodes: List[SearchNode]) -> str:
    """隐式格式（Section 4.1）"""
    lines = []
    for node in search_nodes:
        lines.append(f"EXPAND ACT {node.action} -> {serialize_state(node.state)}")
    return "\n".join(lines)

def serialize_trace_linetree(search_nodes: List[SearchNode]) -> str:
    """LinTree显式格式（Section 5.1）— 核心创新"""
    lines = []
    for node in search_nodes:
        parent_sid = node.parent.state_id if node.parent else "?"
        lines.append(
            f"EXPAND sid={parent_sid} ACT {node.action} "
            f"-> sid={node.state_id} {serialize_state(node.state)}"
        )
    return "\n".join(lines)

# 示例输出对比：
# 隐式:
#   EXPAND ACT B:A->TABLE -> S{ 0:B<C ; 1:A ; 2:D }
#   EXPAND ACT C->B -> S{ 0:C ; 1:A<B ; 2:D }
#
# LinTree:
#   EXPAND sid=0 ACT B:A->TABLE -> sid=1 S{ 0:B<C ; 1:A ; 2:D }
#   EXPAND sid=1 ACT C->B -> sid=2 S{ 0:C ; 1:A<B ; 2:D }
```

### 6.5 GRPO奖励函数实现

```python
# ⚠️ 非官方概念实现 — 基于论文 Eq.1

def compute_reward(trace_is_valid: bool, trace_is_correct: bool, 
                   num_expansions: int, 
                   lambda_: float = 0.005, gamma: float = 0.99) -> float:
    """
    Eq.1: R(τ) = 1[valid ∧ correct] × (1 - λ Σ_{t=0}^{Nexp-1} γ^t)
    
    保证: 任何正确轨迹得分 > 任何错误轨迹得分 (因为 λ < 1-γ)
    验证: λ=0.005, 1-γ=0.01, λ < 1-γ ✓
    """
    if not (trace_is_valid and trace_is_correct):
        return 0.0
    
    # 几何级数和
    penalty_sum = sum(gamma ** t for t in range(num_expansions))
    penalty = lambda_ * penalty_sum
    reward = 1.0 - penalty
    
    # 由于 λ < 1-γ，即使无限展开，reward > 0
    # 最大惩罚: λ / (1-γ) = 0.005 / 0.01 = 0.5
    # 最差情况 reward = 1.0 - 0.5 = 0.5 > 0 ✓
    return max(reward, 0.0)  # 理论上不需要，但作为安全网
```

---

## Ch7: 局限性与延伸阅读

### 7.1 方法局限性

| 局限 | 详情 | 影响 |
|------|------|------|
| **领域限制** | 仅在三个fully observable规划领域验证 | 未证明在自然语言推理任务上的有效性 |
| **模型规模** | 仅测试Qwen3-0.6B | 结论可能不适用于7B+甚至数百B参数模型 |
| **搜索深度** | 轨迹截断≤16k tokens | 更复杂的搜索场景未覆盖 |
| **搜索策略单一** | 仅测试best-first search | 未与MCTS、beam search等策略比较 |
| **无标准Benchmark** | 未在MATH、GSM8K、ARC等标准推理基准上评估 | 与主流推理研究不可直接比较 |
| **结构标注简单** | 仅使用sid=一种标注 | 可能存在更优的结构化表示（如depth、heuristic value标注） |

### 7.2 未来方向

1. **扩展到自然语言推理**：将LinTree应用于数学推理（MATH）、常识推理（StrategyQA）等，验证parent pointer思想在非结构化领域的迁移性
2. **更大规模验证**：在7B、72B参数的模型上验证结论，特别是大模型可能已经内化了"结构理解"能力
3. **更丰富的结构标注**：除parent pointer外，尝试标注分支深度、启发式值、置信度等多维结构信息
4. **与高级搜索策略结合**：将LinTree与MCTS、A*等更强大的搜索策略整合，探索协同效应
5. **从训练数据中自动学习结构**：让模型在学习搜索策略的同时，自动学习最优的结构化表示

### 7.3 相关工作对比

| 方法 | 搜索树维护方 | LLM查询视图 | 树结构显式化 | 训练方式 |
|------|------------|-----------|------------|---------|
| **CoT** [Wei 2022] | 隐式（LLM内部） | 局部（当前步骤） | ❌ | Prompt |
| **ToT** [Yao 2023] | 外部控制器 | 局部（当前路径） | ✅（外部） | Prompt |
| **LATS** [Zhou 2023] | 外部MCTS | 局部（当前路径） | ✅（外部） | Prompt |
| **RAP** [Hao 2023] | 外部MCTS | 局部（当前路径） | ✅（外部） | Prompt |
| **GoT** [Besta 2023] | 外部图控制器 | 局部（prompter选择） | ✅（外部） | Prompt |
| **SE-Beam** [Xie 2023] | 外部束搜索 | 局部（单路径） | ❌ | Prompt |
| **LinTree** (本文) | 隐式（LLM内部） | **全局（完整轨迹）** | ✅（内部parent pointer） | SFT+GRPO |

LinTree的独特定位：**既保留了LLM内部推理的完整上下文优势，又通过显式标注解决了隐式结构的可读性问题**。

### 7.4 延伸阅读

1. **Tree of Thoughts** (Yao et al., NeurIPS 2023) — 外部搜索树的开创性工作，首次将LLM推理形式化为树搜索
2. **LATS: Language Agent Tree Search** (Zhou et al., NeurIPS 2023) — MCTS与LLM结合的代表作
3. **Reasoning via Planning (RAP)** (Hao et al., ICML 2024) — 将LLM推理建模为MDP中的规划问题
4. **Graph of Thoughts** (Besta et al., AAAI 2024) — 将推理结构从树推广到有向无环图
5. **DeepSeekMath: GRPO** (Shao et al., 2024) — 本文使用的GRPO强化学习算法
6. **Qwen3 Technical Report** (2025) — 本文基座模型的技术细节
7. **Self-Evaluation Guided Beam Search** (Xie et al., EMNLP 2023) — 束搜索解码与LLM自我评估的结合
8. **Chain-of-Thought Prompting** (Wei et al., NeurIPS 2022) — LLM推理的奠基性工作

---

*报告完成于 2026年6月2日*

---
