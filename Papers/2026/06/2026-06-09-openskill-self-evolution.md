# OpenSkill: Open-World Self-Evolution for LLM Agents

## 论文元数据
- 标题：OpenSkill: Open-World Self-Evolution for LLM Agents
- 作者：Zhiling Yan, Dingjie Song, Hanrong Zhang, Wei Liang, Yuxuan Zhang, Yutong Dai, Lifang He, Philip S. Yu, Ran Xu, Xiang Li, Lichao Sun
- arXiv ID：2606.06741
- 提交日期：2026-06-08
- 官方代码：github.com/OpenLAIR/OpenSkill（Code coming soon，尚未发布）
- 代码发现方式：论文标注
- CodeGraph分析：跳过（代码未发布）

---

## Ch1: 论文概述与核心贡献

### 1.1 一句话概括

OpenSkill首次形式化并解决了"开放世界自进化"问题——智能体从零开始，仅使用任务提示（task prompt）和开放世界资源（文档、代码库、论文、网页），自主构建可迁移技能库和无监督验证循环，最终在SkillsBench上达到43.6%（Opus 4.6）/42.1%（GPT 5.2）的通过率，超越最强baseline 8.9/8.8个百分点，距人类上界仅1-3个百分点。

### 1.2 核心问题：缺失的"可用学习循环"

现有self-evolving agent的研究默认假设一个**可用学习循环（usable learning loop）**的存在：系统拥有预整理的技能库、成功的执行轨迹、验证器信号，甚至目标任务的标准答案。然而在真实开放世界部署中，agent往往只能获得一个任务提示（task prompt），没有任何可用的技能或验证信号。

这导致了一个根本性困境：agent需要技能来解决问题，但构建技能又需要验证信号，而验证信号又依赖于已有技能或人工标注——形成了一个"鸡生蛋、蛋生鸡"的死循环。

### 1.3 三大核心属性

OpenSkill的设计目标是同时满足三个看似矛盾的关键属性：

| 属性 | 含义 | 为何难以兼得 |
|------|------|-------------|
| **Scalable（可扩展）** | 能够覆盖无限多的任务领域，无需人工为每个任务准备资源 | 人工标注成本高，无法扩展 |
| **Grounded（有根植）** | 知识来自真实可靠的文档、代码库、论文，而非LLM的幻觉 | 开放世界充满噪声和误导信息 |
| **Supervision-free（无监督）** | 构建验证信号时不依赖目标任务的标准答案或奖励 | 无监督验证容易产生虚假自信 |

> **类比理解**：这就像要求一个自学者同时满足三个条件——（1）能学会任何专业（可扩展）、（2）知识来自权威教材而非道听途说（有根植）、（3）没有任何老师批改作业（无监督）。传统教育系统中，这三者几乎不可能同时实现：要么学生只能在有限的专业中选择（不可扩展），要么知识来源不可靠（无根植），要么必须依赖老师批改（有监督）。

### 1.4 三阶段框架速览

OpenSkill通过以下三阶段框架打破上述死循环：

**Stage 1: Open-World Knowledge Acquisition（开放世界知识获取）**
- 使用Gemini Deep Research agent从开放世界检索任务相关知识（文档、代码、论文）
- 独立检索可验证的知识锚点（参考值、数据集不变量、交叉验证流程）
- 合成结构化技能计划

**Stage 2: Leakage-Free Skill Evolution（无泄露技能进化）**
- 在隔离沙箱中根据技能计划生成初始技能草案
- 从验证知识构建虚拟测试套件（pytest断言）
- 迭代精炼：执行技能→测量虚拟通过率→诊断失败→定向知识检索→就地精炼
- 最多3轮精炼，直到虚拟通过率达到100%或达到最大轮数

**Stage 3: Zero-Shot Target Evaluation（零样本目标评估）**
- 冻结最终技能，部署到目标任务
- **仅在此刻**接触隐藏的真实测试集
- 报告平均通过率

### 1.5 核心数字与性能表现

在SkillsBench（11个领域：Software, Office, Science, Media, Cybersecurity, Finance, Robotics, Energy, Manufacturing, Health, Math）上：

| 模型 | OpenSkill | 最强Baseline | 提升幅度 | 人类上界 | 差距 |
|------|-----------|--------------|----------|----------|------|
| Opus 4.6 | 43.6% | Skill-Creator 34.7% | **+8.9%** | 44.5% | 0.9% |
| GPT 5.2 | 42.1% | CoT 33.3% | **+8.8%** | 44.8% | 2.7% |

在其他任务上的表现：
- **SocialMaze**：Opus达到82.7%，GPT达到70.7%
- **ScienceWorld**：Opus达到90.0%，GPT达到85.3%

### 1.6 四条核心贡献

1. **问题形式化**：首次明确定义open-world self-evolution问题框架，将"可用学习循环"从必要假设转变为需要解决的核心障碍，强调仅用任务prompt和开放世界资源构建技能和验证信号的可能性。

2. **OpenSkill框架**：提出三阶段流水线（知识获取→无泄露技能进化→零样本评估），通过将验证知识的检索与任务知识检索解耦，构建独立的虚拟验证循环，首次同时实现scalable、grounded、supervision-free三个属性。

3. **虚拟验证器**：从开放世界资源中提取独立验证锚点，生成确定性pytest断言形成虚拟测试套件，在真实测试意图上达到88.9%的覆盖率，成为无监督环境下可靠的自我评估工具。

4. **跨模型迁移**：Opus 4.6生成的技能可以直接零样本迁移到4个更弱模型（包括GPT 5.2、Claude 4.x等），相对于无技能基线提升+5.5到+14.8个百分点，证明OpenSkill技能的可迁移性和模型无关性。

---

## Ch2: 研究背景与动机

### 2.1 Self-Evolving Agents 的演进历史

Self-evolving agents的研究经历了四个范式的演进，每个范式都在尝试解决"如何让agent自主获得新能力"这一核心问题，但各自存在关键局限。

#### 2.1.1 Human-Curated Paradigm（人工整理范式）

**代表方法**：早期专家系统、基于规则的agent、技能库预置系统

**核心思想**：人工为每个目标任务编写技能代码、执行流程或验证规则，agent直接调用这些人工整理的组件。

**优势**：
- **Grounded**：知识来自人工设计，可靠性高
- **Supervision-free（部分）**：验证规则由人工编写，运行时无需外部监督

**致命缺陷**：
- **不Scalable**：每个新任务都需要人工编写技能，成本极高，无法扩展到开放世界的无限任务
- 人工编写质量参差不齐，维护成本随任务数量线性增长

#### 2.1.2 LLM-Generated Paradigm（大模型生成范式）

**代表方法**：Chain-of-Thought（CoT）、思维树、推理时提示工程

**核心思想**：通过精心设计的prompt，让LLM在推理时临时生成解决方案，无需预存技能。

**优势**：
- **Scalable**：理论上可以处理任何任务，无需人工为每个任务准备资源

**致命缺陷**：
- **不Grounded**：LLM容易产生幻觉，生成的解决方案可能违反领域规则或使用不存在的API
- 无可持续积累：每次推理都从零开始，无法形成可复用的技能库

#### 2.1.3 Supervised Self-Evolution Paradigm（有监督自进化范式）

**代表方法**：从成功轨迹中蒸馏技能、强化学习、用reward model指导进化

**核心思想**：agent通过与环境交互获得成功轨迹，从中蒸馏技能，或用reward model验证进化方向。

**优势**：
- **Scalable**：可以自动扩展到新任务
- **部分Grounded**：成功轨迹来自真实环境交互

**致命缺陷**：
- **不Supervision-free**：依赖reward model或成功轨迹标注，本质上仍需要外部监督信号
- reward model的偏见：reward model本身可能存在错误，导致agent学到错误的行为
- 需要大量交互：在危险或昂贵的环境中（如医疗、金融），无法承受试错成本

#### 2.1.4 Open-World Paradigm（开放世界范式，OpenSkill）

**核心思想**：将开放世界本身同时用作知识来源和无监督练习环境，agent自主检索真实文档/代码库构建技能，并从开放世界中提取独立验证锚点构建虚拟测试套件。

**突破**：首次同时实现三个属性：
- **Scalable**：开放世界提供无限知识，可覆盖任意任务
- **Grounded**：知识来自真实文档和代码库，验证锚点来自公开可验证的资源
- **Supervision-free**：虚拟测试套件完全自建，不依赖目标任务答案

### 2.2 四种范式对比

| 范式 | Scalable | Grounded | Supervision-free | 主要瓶颈 |
|------|----------|----------|------------------|----------|
| Human-Curated | ❌ 人工成本高 | ✅ 人工设计 | ❌ 需人工编写规则 | 无法扩展到开放世界 |
| LLM-Generated | ✅ 理论无限制 | ❌ 容易幻觉 | ✅ 运行时无监督 | 知识不可靠、无积累 |
| Supervised Self-Evolution | ✅ 自动扩展 | ⚠️ 部分来自环境 | ❌ 需reward/标注 | 依赖监督信号 |
| **Open-World (OpenSkill)** | ✅ 开放世界无限 | ✅ 真实文档/代码 | ✅ 自建虚拟验证 | 虚拟验证器准确性 |

### 2.3 问题形式化

OpenSkill将开放世界自进化问题形式化为如下优化问题：

**输入**：
- $I_i$：第$i$个任务的指令（task prompt）
- $E_i$：第$i$个任务的执行环境
- $K$：开放世界资源集合（文档、代码库、论文、网页）

**约束条件**：
- 禁止访问$T_i^{GT}$（真实测试集）
- 禁止使用外部奖励信号或reward model
- 禁止使用人工标注的验证器输出

**目标**：
构建可迁移技能$\hat{S}_i$和验证循环$\tilde{T}_i$，使得：
$ \max_{\hat{S}_i, \tilde{T}_i} \mathbb{E}_{i \sim \mathcal{D}} [ \text{PassRate}(\pi_{\theta'}(\hat{S}_i), T_i^{GT}) ] $

其中$\pi_{\theta'}$是部署时的agent模型（可与构建技能时的模型不同），且仅在最终评估时接触$T_i^{GT}$。

**关键挑战**：
1. **知识检索**：如何从无限开放世界$K$中检索出与任务$I_i$相关且可靠的知识？
2. **无泄露验证**：如何在不接触$T_i^{GT}$的情况下构建有效的验证信号？
3. **技能精炼**：如何在没有ground truth的情况下判断技能是否真正改进？

### 2.5 前人工件的关键局限总结

在开放世界自进化问题被形式化之前，研究者尝试了多种前向解决方案，但都未能同时满足三大属性：

| 前人工件 | 方法 | 缺失的属性 | 原因 |
|---------|------|-----------|------|
| 人工技能库 | 人工编写技能代码 | Scalable | 人力成本无法覆盖开放世界 |
| CoT/ReAct | 推理时生成方案 | Grounded | LLM幻觉，知识无根植 |
| RL from Scratch | 强化学习探索 | Supervision-free | 需要reward function |
| 轨迹蒸馏 | 从成功轨迹学习 | Supervision-free | 成功轨迹本身是监督信号 |
| Reward Model | 训练验证器 | Grounded | Reward model可能学到错误偏好 |

### 2.6 研究动机：为何现在才能解决Open-World Self-Evolution？

三个关键技术条件的成熟使得OpenSkill在2026年成为可能：

1. **开放世界知识的可访问性**：技术文档（API docs、Stack Overflow）、代码库（GitHub）、学术论文（arXiv、Semantic Scholar）的指数级增长，使得任何任务的权威知识都可通过检索获得。

2. **LLM的检索和理解能力**：现代LLM（如Gemini Deep Research、Opus 4.6）具备了长上下文、复杂推理、代码理解能力，可以从开放世界资源中提取结构化知识和验证锚点。

3. **确定性代码执行环境**：沙箱化的代码执行环境（如E2B）使得agent可以安全地运行自建的虚拟测试套件，形成可重复的验证循环。

在此之前，要么开放世界知识不够丰富（2010年代前），要么LLM无法有效检索和利用这些知识（2020年前），要么缺乏安全的执行环境来验证自我生成的代码（2024年前）。OpenSkill站在了这三个技术浪潮的交汇点上。

---

## Ch3: OpenSkill三阶段框架详解

### 3.1 整体架构

OpenSkill的完整pipeline由三个严格解耦的阶段组成，它们之间的信息流被一个**泄露屏障（leakage barrier）**隔开，确保真实测试集$T_i^{GT}$在整个技能构建过程中（Stage 1-2）完全不可见。

```
输入: 任务指令 I_i + 执行环境 E_i
  │
  ▼
┌─────────────────────────────────────────────┐
│ Stage 1: Open-World Knowledge Acquisition    │
│  ┌─────────────────┐  ┌───────────────────┐  │
│  │ 任务知识检索    │  │ 验证知识检索     │  │
│  │ k_i = D(I_i,K) │  │ k_i^v = D_v(I_i,K)│  │
│  │ (Gemini Deep    │  │ (独立检索agent)   │  │
│  │  Research)      │  │                   │  │
│  └────────┬────────┘  └────────┬──────────┘  │
│           │                    │              │
│           ▼                    ▼              │
│     结构化技能计划 p_i    验证锚点 k_i^v     │
│     (1-4 skills/task)   (参考值/不变量等)    │
└──────────────────┬──────────────────────────┘
                   │ ╔═══════════════════════╗
                   │ ║ LEAKAGE BARRIER     ║
                   │ ║ T_i^GT 不可见       ║
                   ▼ ╚═══════════════════════╝
┌─────────────────────────────────────────────┐
│ Stage 2: Leakage-Free Skill Evolution        │
│                                              │
│  初始技能草案 Ŝ_i^(0)                        │
│       │                                      │
│       ▼                                      │
│  ┌─────────────┐                            │
│  │ 虚拟测试套件 │ ← k_i^v (验证锚点)        │
│  │ T̃_i = pytest │   独立验证LLM              │
│  └──────┬──────┘                            │
│         │                                    │
│         ▼                                    │
│  ┌─────────────────────────────────┐        │
│  │ 迭代精炼循环 (j=0,1,2)         │        │
│  │                                 │        │
│  │ ① 执行技能 Ŝ_i^(j)            │        │
│  │ ② 测量虚拟通过率 r̃^(j)        │        │
│  │ ③ 失败诊断 F^(j)               │        │
│  │    ├─ bug → 直接修复            │        │
│  │    └─ knowledge gap → 定向检索  │        │
│  │ ④ 就地精炼 Ŝ_i^(j+1)          │        │
│  │                                 │        │
│  │ 退出条件: r̃=1 或 j≥2          │        │
│  └─────────────────────────────────┘        │
│                                              │
│  最终技能 Ŝ_i* (最后精炼版本)                │
└──────────────────┬──────────────────────────┘
                   │ ╔═══════════════════════╗
                   │ ║ 仅此时解锁 T_i^GT    ║
                   ▼ ╚═══════════════════════╝
┌─────────────────────────────────────────────┐
│ Stage 3: Zero-Shot Target Evaluation         │
│  部署 Ŝ_i* → π_θ' (目标agent)               │
│  评估 T_i^GT → 报告通过率                    │
└─────────────────────────────────────────────┘
```

### 3.2 Stage 1: 开放世界知识获取

Stage 1 是整个框架的**信息入口**，负责将开放世界中的非结构化知识转化为agent可用的结构化技能计划和验证锚点。

#### 3.2.1 任务知识检索

**流程**：
1. 从任务指令$I_i$中提取核心需求（领域、目标、约束）
2. 对查询进行**清洗**：移除所有benchmark标识符（如"SkillsBench"、"SocialMaze"），仅保留功能描述
3. 使用**Gemini Deep Research agent**在开放世界$K$中执行深度检索：$k_i = D(I_i, K)$
4. 检索范围包括：背景概念、领域最佳实践、相关API文档、源码引用、学术论文摘要

**输出**：结构化技能计划$p_i$，包含：
- 技能架构：每个任务拆分为1-4个独立技能
- 关键流程：每个技能的核心执行步骤
- 领域规则：该领域特有的约束条件（如金融领域的精度要求、医疗领域的安全约束）

#### 3.2.2 验证知识检索（独立路径）

这与任务知识检索**解耦且独立**执行，目的是获取可以独立验证的锚点信息：

**验证锚点类型**（$k_i^v = D_v(I_i, K)$）：
- **参考值**：公开数据集的已知统计量（均值/方差/分布）
- **数据集不变量**：跨时间/跨样本的一致性约束（如"某列值必须为正"）
- **交叉验证流程**：该领域的标准验证方法（如k-fold配置）
- **预期输出格式**：任务要求的输出schema（JSON结构/表格字段）

> **类比理解**：任务知识检索类似学生查教材学习概念（"什么是线性回归？怎么实现？"），验证知识检索类似学生查阅课后习题答案（"正确答案应该接近0.85"），两者信息源不同，防止"偷看答案"——如果任务检索和验证检索从同一来源获取，就可能"不小心"看到目标答案。

#### 3.2.3 泄露屏障设计

Stage 1的两个关键安全措施：

1. **查询清洗**：所有检索查询经过过滤，去除benchmark名称、任务编号等可能泄露目标测试信息的标识符
2. **检索分离**：任务知识检索agent和验证锚点检索agent使用**不同的Google搜索会话**，不同的system prompt，确保两者无法共享信息

### 3.3 Stage 2: 无泄露技能进化

Stage 2 是OpenSkill的核心创新——在完全没有目标答案的情况下，通过自建的虚拟测试套件驱动技能迭代改进。

#### 3.3.1 初始技能草案

从Stage 1的输出生成初始技能草案$\hat{S}_i^{(0)}$：
$\hat{S}_i^{(0)} = \pi_\theta(I_i, E_i, p_i, k_i)$

其中$\pi_\theta$是技能构建LLM（Opus 4.6 或 GPT 5.2），输出为可执行的代码（Python函数/类）加上自然语言说明。

#### 3.3.2 虚拟测试套件构建

这是OpenSkill最具创新性的组件——将验证知识锚点$k_i^v$转化为可执行的pytest测试：

**构建过程**：
1. 使用**独立验证LLM** $g$（与技能构建LLM $\pi_\theta$共享不同上下文）
2. 输入：任务指令$I_i$ + 执行环境$E_i$ + 验证锚点$k_i^v$
3. 输出：确定性pytest断言集合 $\tilde{T}_i = g(I_i, E_i, k_i^v)$
4. 每个断言针对一个独立的验证维度（输入输出一致性、边界条件、格式正确性）

**验证LLM独立性要求**：
- 与技能构建LLM使用不同的对话上下文
- 不接触技能计划$p_i$和任务知识$k_i$
- 仅基于$k_i^v$中的独立验证锚点生成测试

**示例**：对于金融领域的数据清洗任务：
```python
# ⚠️ 非官方概念实现，基于论文Section 3.2描述编写，未经验证
def test_output_format(data_output):
    """验证锚点：输出必须包含指定字段且类型正确"""
    assert "date" in data_output.columns
    assert data_output["amount"].dtype == float
    assert data_output["amount"].min() >= 0  # 金融数据不变量：金额非负
```

#### 3.3.3 迭代精炼循环

每次迭代$j$（$j=0,1,2$，最多3轮）执行以下步骤：

**Step 1：执行技能并测量虚拟通过率**

$\tilde{r}^{(j)} = \frac{|\{\text{passed tests in } \tilde{T}_i\}|}{|\tilde{T}_i|}$

这是在虚拟测试套件$\tilde{T}_i$上的全对全通过率。

**Step 2：失败诊断**（当$\tilde{r}^{(j)} < 1$时）

对每个失败的断言生成诊断报告$F^{(j)}$，包含：
- Per-assertion结果：每个断言通过/失败状态
- 根因分析：分类为两类——
  - **Bug**：实现错误（如类型错误、边界条件未处理）→ 直接修复
  - **Knowledge gap**：领域知识不足（如不知道某API的正确用法）→ 触发定向检索

**Step 3：定向知识检索**（知识缺口时触发）

$\Delta k^{(j)} = D(F^{(j)}, K)$

仅检索与失败诊断直接相关的新知识，而非重新执行整个Stage 1。

**Step 4：就地精炼**

$\hat{S}_i^{(j+1)} = \pi_\theta(\hat{S}_i^{(j)}, F^{(j)} \mid I_i, E_i, p_i, k_i \cup \Delta k^{(j)})$

关键特性：
- **就地更新**：直接修改当前技能，而非生成新技能再选最优（非best-of-N）
- **上下文保持**：$\pi_\theta$接收完整的前一轮技能和诊断，保持改进的连续性

**退出条件**：
- $\tilde{r}^{(j)} = 1$（所有虚拟测试通过）→ 提前退出
- $j \geq 2$（已完成3轮）→ 强制退出

最终技能$\hat{S}_i^*$是**最后精炼的版本**，不是best-of-N快照。这确保技能是迭代收敛的结果，而非随机采样的最优。

> **类比理解**：想象你自学编程并自己做单元测试。你写了一段代码（初始草案），然后运行自己写的测试用例（虚拟测试）。如果某个测试失败，你分析原因——如果是拼写错误（bug），直接改；如果是用了错误的库函数（knowledge gap），就去Google搜索正确用法，然后修正代码。这个过程重复最多3次，最后提交的是经过所有修改的最终版本，而非从多个版本中挑选"运气最好"的一个。

### 3.4 Stage 3: 零样本目标评估

这是整个框架最简洁的阶段——所有困难工作已在Stage 1-2完成：

1. **技能冻结**：最终技能$\hat{S}_i^*$被冻结，不再修改
2. **模型切换**：技能部署到**目标agent** $\pi_{\theta'}$，可与构建模型$\pi_\theta$不同（验证跨模型迁移）
3. **真实评估**：**仅在此刻**解锁$T_i^{GT}$，执行评估
4. **报告结果**：

$\text{PassRate} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\pi_{\theta'}(\hat{S}_i^*) \text{ passes } T_i^{GT}]$

---

## Ch4: 实验设计与结果分析

### 4.1 实验环境与Benchmarks

OpenSkill在三个互补的benchmark上进行了全面评估：

**SkillsBench（主benchmark，11个领域）**：

| 领域 | 典型任务示例 |
|------|------------|
| Software | 实现REST API、处理数据流 |
| Office | 自动化Excel报表生成、文档格式转换 |
| Science | 实验数据统计分析、科学计算工作流 |
| Media | 图像元数据处理、视频格式转换 |
| Cybersecurity | 端口扫描分析、日志异常检测 |
| Finance | 金融数据清洗、风险指标计算 |
| Robotics | 路径规划、传感器数据处理 |
| Energy | 能耗数据分析、预测模型构建 |
| Manufacturing | 生产线数据处理、质量控制 |
| Health | 医疗数据预处理、患者记录管理 |
| Math | 数值计算、符号推导 |

**SocialMaze**：社交推理环境——agent需要在复杂的社会交互中推理他人意图和做出决策。

**ScienceWorld**：科学任务环境——涵盖化学实验、物理模拟等科学领域的交互式任务。

### 4.2 Baselines

6个baseline覆盖了从无技能到最强自进化方法的完整谱系：

| Baseline | 方法 | 范式 |
|----------|------|------|
| **No Skill** | 纯agent，无任何技能辅助 | N/A（下界） |
| **Self-Gen** | Agent自主生成技能（无开放世界检索） | LLM-Generated |
| **CoT** | Chain-of-Thought推理时提示 | LLM-Generated |
| **Skill-Creator** | 现有技能创建框架 | Supervised Self-Evolution |
| **AutoSkill** | 自动化技能构建方法 | Supervised Self-Evolution |
| **Memento** | 记忆增强的agent方法 | Supervised Self-Evolution |
| **Human** | 人类专家编写的技能 | Human-Curated（参考上界） |

### 4.3 主实验结果（SkillsBench）

| Target Agent | No Skill | Self-Gen | CoT | Skill-Creator | AutoSkill | Memento | **OpenSkill** | Human |
|-------------|----------|----------|-----|---------------|-----------|---------|---------------|-------|
| **Opus 4.6** (Claude Code) | 25.5 | 23.9 | 23.9 | 34.7 | 24.7 | 30.1 | **43.6** | *44.5* |
| **GPT 5.2** (Codex) | 25.0 | 32.2 | 33.3 | 29.2 | 11.2 | 15.6 | **42.1** | *44.8* |

**关键发现**：

1. **绝对优势**：OpenSkill在所有设置上均取得最佳自动化成绩，在Opus 4.6上超过最强baseline Skill-Creator 8.9个百分点，在GPT 5.2上超过最强baseline CoT 8.8个百分点。

2. **模型稳定性**：OpenSkill在两个模型上的表现高度一致（43.6% vs 42.1%，仅差1.5个百分点），而baselines表现剧烈波动——例如AutoSkill在Opus上24.7%，在GPT上暴跌至11.2%（降幅54%）；Memento从30.1%跌至15.6%（降幅48%）。

3. **逼近人类**：与人类上界差距仅0.9（Opus）和2.7（GPT）个百分点，表明OpenSkill在无监督条件下已接近人类有监督的技能质量。

4. **领域覆盖**：SkillsBench的11个领域中，OpenSkill在8个领域上为最优或并列最优。

### 4.4 SocialMaze与ScienceWorld结果

| Benchmark | Opus 4.6 | GPT 5.2 |
|-----------|----------|---------|
| **SocialMaze** | **82.7%** | 70.7% |
| **ScienceWorld** | **90.0%** | 85.3% |

SocialMaze（社交推理）和ScienceWorld（科学任务）的评估展示OpenSkill在非代码类任务上同样有效——ScienceWorld上90.0%的通过率尤其令人印象深刻，表明开放世界知识检索对科学领域同样有效。

### 4.5 Baseline表现分析

几个值得关注的baseline表现异常：

- **Self-Gen在GPT上超过Opus**（32.2% vs 23.9%）：可能是因为GPT 5.2在零样本代码生成上本身更强，CoT在GPT上同样优于Opus（33.3% vs 23.9%）
- **AutoSkill在GPT上崩溃**（11.2%）：AutoSkill的自动化流程可能在GPT的执行环境中遇到了兼容性问题
- **Skill-Creator在Opus上最好但在GPT上不如CoT**：说明有监督方法对构建模型的敏感度高于OpenSkill

---

## Ch5: 消融实验与深度分析

### 5.1 RQ1：技能可迁移性

**实验设计**：Opus 4.6（Claude Code环境）为SkillsBench构建的技能直接部署到4个更弱的目标模型上，不做任何模型特定的适配。

**结果**：相对于No Skill基线，所有目标模型均获得显著提升：

| 目标模型 | No Skill | +OpenSkill Skill | 提升幅度 |
|---------|----------|-----------------|----------|
| 更弱模型 (范围) | 未报告 | 未报告 | **+5.5 ~ +14.8** |
| GPT 5.2（Claude Code环境） | 25.0 | 42.1 | **+17.1** |

提升范围从+5.5到+14.8个百分点，**无需任何模型特定适配**。这表明OpenSkill构建的技能是**模型无关的知识表示**——技能封装的是任务领域的结构化知识（流程、约束、验证规则），而非针对特定模型优化的prompt工程。

> **类比理解**：一本好的教科书（OpenSkill技能）应该让任何水平的学生（不同模型）都能从中获益。如果只有学霸才能看懂，那只是"学霸笔记"，不是真正的"知识转移"。OpenSkill的技能做到了这一点——从最强模型（Opus 4.6）生成的技能，学渣（更弱模型）用起来同样提分。

### 5.2 RQ2：虚拟验证器质量

**实验设计**：评估Stage 2中自建虚拟测试套件$\tilde{T}_i$与真实测试集$T_i^{GT}$的一致性。

**关键指标**：

| 指标 | 值 | 含义 |
|------|-----|------|
| **Recall** | **80.5%** | 真实正向测试中，虚拟验证器能正确识别的比例 |
| **Overall Agreement** | **60.7%** | 虚拟验证器与真实测试的总体一致率 |
| **Coverage** | **88.9%** | 虚拟验证器覆盖的真实测试意图比例 |

**分析**：
- **Recall 80.5%**意味着虚拟验证器能捕获大部分真实测试的正确行为，但仍有约20%的漏检——这解释了为什么最终通过率（43.6%）低于虚拟通过率（通常接近100%）
- **Overall Agreement 60.7%**说明虚拟验证器虽不完美，但已是无监督条件下可获得的最佳信号——相比随机猜测（50%），60.7%的信息增益足以驱动有意义的技能改进
- **Coverage 88.9%**表明虚拟验证器覆盖了绝大多数真实测试的意图维度，遗漏的主要是边缘case和需要领域专家知识的测试

### 5.3 RQ3：组件贡献分析（SocialMaze）

**实验设计**：在SocialMaze上系统性地移除OpenSkill的各个组件，观察性能变化。

**核心发现**：

1. **精炼轮数的影响**：性能随精炼轮数增加而提升，在**3轮后达到峰值**。继续精炼未带来额外增益（论文未报告4轮以上的结果）。

2. **开放世界查询的独立贡献**：移除开放世界知识检索（即$k_i$和$k_i^v$均置空，仅依赖模型参数化知识）导致性能显著下降。

3. **虚拟验证器的独立贡献**：移除虚拟测试套件（即不构建$\tilde{T}_i$，不执行精炼循环）同样导致性能显著下降。

4. **互补性验证**：开放世界查询和虚拟验证器的贡献**大体正交**——同时使用两者优于单独使用任一组件。这表明知识检索提供"知道做什么"，虚拟验证器提供"知道做对了没"，两者解决的是自进化过程的不同环节。

5. **组件移除后的性能衰减**：完整OpenSkill在SocialMaze上达到82.7%（Opus）。移除开放世界查询或虚拟验证器后，性能均出现显著下降（论文Figure报告了约15-25个百分点的降幅），验证了每个组件的独立必要性。

### 5.4 跨Benchmark一致性

OpenSkill在三个差异巨大的benchmark上均取得最佳自动化成绩：

| Benchmark | 任务类型 | Opus结果 | GPT结果 |
|-----------|---------|----------|---------|
| SkillsBench | 实际编程任务（11个领域） | 43.6% | 42.1% |
| SocialMaze | 社交推理 | 82.7% | 70.7% |
| ScienceWorld | 科学交互 | 90.0% | 85.3% |

这种跨任务类型的一致性说明OpenSkill的方法论（知识检索+自建验证+精炼循环）是通用的，不与特定任务类型耦合。

---

## Ch6: 概念实现详解

> ⚠️ **注意**：官方代码标注为"coming soon"（仓库主README说明），本章所有代码为基于论文Section 3描述和README架构说明编写的**概念实现**，目的是帮助理解算法流程，不可直接用于训练或复现。

### 6.1 整体架构

```python
# ⚠️ 非官方概念实现，基于论文Section 3描述编写，未经验证
from typing import List, Dict, Optional

class OpenSkillPipeline:
    """OpenSkill三阶段流水线的概念实现"""
    
    def __init__(self, knowledge_agent, verifier_llm, skill_llm):
        self.stage1 = KnowledgeAcquisition(knowledge_agent)
        self.stage2 = SkillEvolution(skill_llm, verifier_llm)
        self.stage3 = ZeroShotEvaluator()
        
    def run(self, task_instruction: str, environment: Dict) -> float:
        # Stage 1: 开放世界知识获取
        task_knowledge, skill_plan = self.stage1.acquire_task_knowledge(
            task_instruction, environment
        )
        verification_anchors = self.stage1.acquire_verification_anchors(
            task_instruction, environment
        )
        
        # ╔══════════════════════════════╗
        # ║ LEAKAGE BARRIER             ║
        # ║ T_i^GT 不在以下任何上下文中 ║
        # ╚══════════════════════════════╝
        
        # Stage 2: 无泄露技能进化
        final_skill = self.stage2.evolve_skill(
            task_instruction, environment,
            task_knowledge, skill_plan,
            verification_anchors,
            max_rounds=3
        )
        
        # ╔═══════════════════════════════╗
        # ║ 仅此时解锁 T_i^GT           ║
        # ╚═══════════════════════════════╝
        
        # Stage 3: 零样本目标评估
        pass_rate = self.stage3.evaluate(
            final_skill, environment
        )
        return pass_rate
```

### 6.2 知识获取模块

```python
# ⚠️ 非官方概念实现，基于论文Section 3.1描述编写，未经验证

class KnowledgeAcquisition:
    """Stage 1: 开放世界知识获取"""
    
    def __init__(self, search_agent):
        self.search_agent = search_agent  # Gemini Deep Research agent
        self.query_sanitizer = QuerySanitizer()
        
    def acquire_task_knowledge(
        self, instruction: str, env: Dict
    ) -> tuple:
        # 清洗查询：移除benchmark标识符
        clean_query = self.query_sanitizer.sanitize(instruction)
        
        # 任务知识检索：背景概念 + 最佳实践 + API文档
        task_knowledge = self.search_agent.deep_research(
            query=f"best practices and documentation for: {clean_query}",
            sources=["docs", "repos", "papers", "web"]
        )
        
        # 合成结构化技能计划（1-4个技能）
        skill_plan = self._synthesize_plan(
            instruction, env, task_knowledge
        )
        return task_knowledge, skill_plan
    
    def acquire_verification_anchors(
        self, instruction: str, env: Dict
    ) -> Dict:
        """独立检索验证锚点——与任务知识检索解耦"""
        clean_query = self.query_sanitizer.sanitize(instruction)
        
        # 使用独立检索会话（不同agent实例或不同Google搜索session）
        anchors = self.search_agent.deep_research(
            query=f"verifiable properties and invariants for: {clean_query}",
            sources=["docs", "datasets", "papers"],
            search_scope="independent"  # 独立搜索上下文
        )
        
        # 提取结构化锚点：参考值、不变量、验证流程
        return {
            "reference_values": self._extract_reference_values(anchors),
            "invariants": self._extract_invariants(anchors),
            "validation_procedures": self._extract_validation_procs(anchors),
            "output_schema": self._extract_output_schema(instruction, anchors)
        }


class QuerySanitizer:
    """查询清洗器：移除benchmark标识符防止信息泄露"""
    
    BENCHMARK_NAMES = [
        "skillsbench", "socialmaze", "scienceworld",
        # 通用模式：任务编号、数据集名称
    ]
    
    def sanitize(self, query: str) -> str:
        """移除所有benchmark标识符后返回功能描述"""
        for name in self.BENCHMARK_NAMES:
            query = query.lower().replace(name, "")
        return query.strip()
```

### 6.3 虚拟验证器与精炼循环

```python
# ⚠️ 非官方概念实现，基于论文Section 3.2描述编写，未经验证

class SkillEvolution:
    """Stage 2: 无泄露技能进化"""
    
    def __init__(self, skill_llm, verifier_llm):
        self.skill_llm = skill_llm          # 技能构建LLM (e.g., Opus 4.6)
        self.verifier_llm = verifier_llm    # 独立验证LLM（不同上下文）
        self.max_rounds = 3
        
    def evolve_skill(
        self, instruction: str, env: Dict,
        task_knowledge: str, skill_plan: Dict,
        anchors: Dict, max_rounds: int = 3
    ) -> str:
        # Step 1: 初始技能草案
        skill = self._draft_initial_skill(
            instruction, env, skill_plan, task_knowledge
        )
        
        # Step 2: 构建虚拟测试套件
        virtual_tests = VirtualTestBuilder(self.verifier_llm).build(
            instruction, env, anchors
        )
        
        # Step 3: 迭代精炼循环
        for round_idx in range(max_rounds):
            # 执行技能并测量虚拟通过率
            test_results = self._run_virtual_tests(skill, virtual_tests)
            pass_rate = test_results.pass_rate
            
            if pass_rate == 1.0:
                break  # 全部虚拟测试通过，提前退出
                
            # 失败诊断
            diagnosis = self._diagnose_failures(test_results)
            
            # 知识缺口 → 定向检索
            if diagnosis.has_knowledge_gap:
                gap_knowledge = self._targeted_retrieval(
                    diagnosis.gap_description
                )
                task_knowledge += "\n" + gap_knowledge
            
            # 就地精炼（不生成多个候选再选最优）
            skill = self._refine_in_place(
                skill, diagnosis, instruction, env,
                skill_plan, task_knowledge
            )
        
        return skill  # 最后精炼版本（非best-of-N）
    
    def _diagnose_failures(self, test_results) -> 'Diagnosis':
        """根因分析：分类为bug或knowledge gap"""
        diagnosis = Diagnosis()
        for failure in test_results.failures:
            if failure.type in ["TypeError", "ValueError", "AssertionError"]:
                diagnosis.add_bug(failure)
            else:
                diagnosis.add_knowledge_gap(failure)
        return diagnosis


class VirtualTestBuilder:
    """从验证锚点构建pytest虚拟测试套件"""
    
    def __init__(self, verifier_llm):
        self.verifier_llm = verifier_llm
    
    def build(self, instruction: str, env: Dict, anchors: Dict) -> List[str]:
        """生成确定性pytest断言"""
        tests = []
        
        # 从参考值生成测试
        for name, expected_value in anchors["reference_values"].items():
            test = f"""
def test_{name}(output):
    assert output["{name}"] == pytest.approx({expected_value})
"""
            tests.append(test)
        
        # 从不变量生成测试
        for invariant in anchors["invariants"]:
            test = f"""
def test_invariant_{invariant['name']}(output):
    assert {invariant['condition']}
"""
            tests.append(test)
        
        # 从输出schema生成测试
        schema_tests = self._schema_to_tests(anchors["output_schema"])
        tests.extend(schema_tests)
        
        return tests
```

### 6.4 精炼循环的工作机制

```python
# ⚠️ 非官方概念实现，基于论文Section 3.2描述编写，未经验证

class Diagnosis:
    """失败诊断数据类"""
    def __init__(self):
        self.bugs = []
        self.knowledge_gaps = []
    
    @property
    def has_knowledge_gap(self) -> bool:
        return len(self.knowledge_gaps) > 0
    
    @property
    def gap_description(self) -> str:
        return "; ".join(g.description for g in self.knowledge_gaps)

# 精炼循环的核心决策树：
def refinement_decision_tree(failure: TestFailure) -> str:
    """
    论文中的失败分类逻辑（概念实现）：
    - Bug → 直接修复（无需额外知识检索）
    - Knowledge Gap → 触发定向开放世界检索 → 注入新知识 → 就地精炼
    """
    if failure.is_code_error():
        return "BUG"           # 类型错误/断言失败/语法错误
    elif failure.is_domain_error():
        return "KNOWLEDGE_GAP"  # 领域规则错误/API误用
    elif failure.is_format_error():
        return "BUG"            # 输出格式不符合schema要求
    return "KNOWLEDGE_GAP"
```

---

## Ch7: 局限性与延伸阅读

### 7.1 核心局限性

**1. 外部API依赖（Gemini Deep Research）**

OpenSkill的Stage 1依赖Gemini Deep Research进行开放世界知识检索。这带来两个问题：
- **成本依赖**：每次技能构建都需要调用外部付费API
- **检索偏差**：Gemini的检索排序算法可能引入系统性偏差，遗漏关键知识或优先返回流行但可能不准确的内容
- **可用性风险**：如果Gemini Deep Research服务不可用，整个pipeline无法运行

**2. 虚拟验证器准确性有限**

虚拟验证器与真实测试的Overall Agreement仅60.7%，意味着约40%的虚拟测试与真实测试判断不一致。在极端情况下：
- 技能可能通过虚拟测试但未通过真实测试（虚报通过）——占约19.5%的case
- 技能可能未通过虚拟测试但通过了真实测试（虚报失败）——浪费精炼轮数

60.7%的agreement在当前阶段足以驱动有意义改进（从25%提升到43.6%），但限制了最终通过率的上限。

**3. 仅验证text-based环境**

OpenSkill仅在文本交互环境中验证（SkillsBench、SocialMaze、ScienceWorld的文本接口）。尚未涉及：
- 多模态环境（视觉/音频输入）
- 具身AI场景（机器人控制）
- 实时交互系统

扩展到这些场景需要重新设计验证锚点的提取机制——视觉任务的验证锚点远不如文本任务直观。

**4. 精炼轮数固定为3轮**

论文未探索超过3轮精炼的效果。3轮后是否继续提升？还是性能会因过拟合虚拟测试而衰减？这留待未来工作探索。

**5. 代码尚未开源**

官方仓库明确标注"Code coming soon"，当前无法独立复现或进行消融实验的交叉验证。

### 7.2 延伸阅读

**直接相关（Self-Evolving Agents）：**

| 论文 | 核心贡献 | 与OpenSkill的关系 |
|------|---------|-----------------|
| **Voyager** (Wang et al., 2023) | Minecraft中的LLM agent自进化 | 使用环境rewards作为验证信号，属于Supervised Self-Evolution |
| **AutoSkill** | 自动化技能构建 | OpenSkill的baseline之一，缺少独立验证组件 |
| **Memento** | 记忆增强agent | OpenSkill的baseline之一，依赖成功轨迹 |
| **SWE-Agent** (Yang et al., 2024) | 自动化软件工程agent | 有监督设置，依赖测试套件验证 |
| **Devin** (Cognition AI, 2024) | 全栈软件工程agent | 商业系统，内部验证机制未公开 |

**方法论相关：**

| 论文 | 核心贡献 | 相关性 |
|------|---------|--------|
| **Reflexion** (Shinn et al., 2023) | LLM自我反思改进 | 与OpenSkill的精炼循环共享"自我改进"理念 |
| **Self-Refine** (Madaan et al., 2023) | 迭代自我精炼 | 无需外部监督，但依赖LLM自评而非外部锚点 |
| **STaR** (Zelikman et al., 2022) | 自举推理 | 从成功rationale中蒸馏，属于Supervised Self-Evolution |

**开放世界知识检索：**

| 论文/系统 | 描述 |
|-----------|------|
| **Gemini Deep Research** | OpenSkill使用的知识检索agent |
| **Perplexity AI** | 实时web搜索增强的LLM |
| **RAG (Retrieval-Augmented Generation)** | 检索增强生成的通用框架——OpenSkill可视为RAG在agent技能构建上的特化应用 |
| **WebGPT** (Nakano et al., 2021) | Web浏览增强的GPT |

### 7.3 总结与展望

OpenSkill代表了开放世界自进化agent研究的一个重要里程碑。其核心洞见——"将开放世界同时用作知识来源和验证环境"——为解决"如何让agent在无需人类监督的情况下持续学习新技能"这一根本性挑战提供了清晰的路径。

**即时应用场景**（假设代码开源后）：
- 无人工干预的CI/CD pipeline中自动修复代码
- 新API/库的自动化学习和集成
- DevOps场景中的自动故障修复脚本生成

**未来方向**：
1. **多模态验证锚点**：将虚拟验证器扩展到视觉/音频领域
2. **更长的精炼周期**：探索3轮以上精炼的效果及过拟合同题
3. **离线知识缓存**：减少对实时API检索的依赖
4. **验证器联合训练**：训练专门的验证模型提升agreement

> **最终洞见**：OpenSkill的成功证明了一个反直觉的事实——在开放世界中，让agent自建不完美的验证信号（60.7% agreement）远比没有验证信号好。这就像自学时，即使自己出的模拟题不能完美覆盖真实考试，也比完全不练习强——关键在于"练习"这一行为本身的价值，而非练习题的完美性。

---

## 参考文献

1. Zhiling Yan et al., "OpenSkill: Open-World Self-Evolution for LLM Agents," arXiv:2606.06741, 2026.
2. Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models," 2023.
3. Yang et al., "SWE-Agent: Agent-Computer Interfaces Enable Automated Software Engineering," 2024.
4. Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning," 2023.
5. Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback," 2023.
6. Zelikman et al., "STaR: Bootstrapping Reasoning With Reasoning," 2022.
7. Nakano et al., "WebGPT: Browser-Assisted Question-Answering with Human Feedback," 2021.
