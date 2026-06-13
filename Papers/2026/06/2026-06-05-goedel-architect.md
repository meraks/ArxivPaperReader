# Goedel-Architect: 基于蓝图生成与精炼的形式化定理证明框架


---

> **论文标题**：Goedel-Architect: Streamlining Formal Theorem Proving with Blueprint Generation and Refinement  
> **作者**：Jui-Hui Chung\*, Ziyang Cai\*, Zihao Li, Qishuo Yin, Rohit Agarwal, Simon Park, Rodrigo Porto, Narutatsu Ri, Ziran Yang, Shange Tang, Xingyu Dang, Hongzhou Lin, Mengdi Wang, Danqi Chen, Chi Jin, Liam H Fowl\*†, Sanjeev Arora†  
> **机构**：Princeton Language and Intelligence (PLI), Princeton University  
> **arXiv**：[2606.06468](https://arxiv.org/abs/2606.06468) | **日期**：2026-06-05  
> **模型**：DeepSeek-V4-Flash (284B-A13B MoE) | **成本**：~$0.44/题  

---

## Ch1：论文概述与核心贡献

### 1.1 一句话总结

Goedel-Architect 是一个**以依赖图蓝图（Blueprint）为核心**的智能体框架，通过"蓝图生成 → 并行引理证明 → 全局蓝图精炼"的循环流程，在形式化定理证明任务上以**开源模型+极低成本**达到了竞赛级水平——IMO 2025 做出 4/6 题，Putnam 2025 做出 11/12 题，而每道题的成本仅约 **$0.44**。

### 1.2 为什么这篇论文重要

AI在数学领域的进展令人瞩目——从IMO金牌到解决Erdős问题——但这些成就背后是**昂贵的闭源模型**（AlphaProof、Gemini）和**巨大的计算预算**。Goedel-Architect 的价值在于证明了：使用**完全开源的模型**（DeepSeek-V4-Flash）和**精心设计的智能体Pipeline**，可以在形式化定理证明上匹敌甚至超越依赖闭源前沿模型的系统，而成本仅为后者的 **1/500**。

> **类比理解**：如果把形式化定理证明比作建造一座大楼，传统方法（如Hilbert）是"雇佣顶级建筑师（Gemini）逐层施工，发现错误就推倒重来"，而Goedel-Architect是"先用AI画出完整的施工蓝图（依赖图），然后派多个工人（并行证明器）同时建造不同部分，哪个部分出问题就**整体调整蓝图**而不是原地死磕"。

### 1.3 核心贡献

| # | 贡献 | 技术要点 |
|---|------|---------|
| 1 | **蓝图驱动的证明框架** | 将定理证明建模为依赖图（DAG）的构建与精炼，而非传统的递归分解树 |
| 2 | **并行引理求解** | 利用蓝图依赖声明实现引理级并行化，每个引理独立分派给配备Lean编译器+Mathlib检索的证明器 |
| 3 | **全局蓝图精炼** | 失败引理触发**全局蓝图重写**（非局部修复），可引入新引理、重组证明策略 |
| 4 | **自然语言证明引导** | 可选将NL证明作为蓝图的结构脚手架，显著提升困难问题求解率 |
| 5 | **极低成本SOTA** | PutnamBench 75.6% pass@1 @ $294总计，对比Hilbert 70.0% @ ~$170,000（~500倍成本优势） |

### 1.4 核心数字速览

| 基准 | 成绩 | 模式 |
|------|------|------|
| MiniF2F-test | **99.2%** pass@1 | 标准 |
| MiniF2F-test | **100%** pass@1 | +NL证明引导 |
| PutnamBench | **75.6%** pass@1 | 标准 |
| PutnamBench | **88.8%** pass@1 | +NL证明引导 |
| IMO 2025 | **4/6** 题 | +NL证明引导 |
| Putnam 2025 | **11/12** 题 | +NL证明引导 |
| USAMO 2026 | **3/6** 题 | +NL证明引导 |

---

## Ch2：研究背景与形式化定理证明格局

### 2.1 形式化定理证明的意义

现代AI在数学推理上取得了长足进步，但**非形式化证明**（自然语言数学证明）的验证依赖于人类专家评审——成本高、速度慢、可能出错。**形式化定理证明**（Formal Theorem Proving, FTP）使用交互式定理证明器（如Lean 4、Coq、Isabelle）将数学证明编码为可由机器验证的形式化证明，提供**绝对的数学确定性**。

> **类比理解**：自然语言证明像是"用中文写一篇数学论文给同行评审"，形式化证明像是"把证明编译成机器可执行的代码，编译器（Lean）告诉你0或1——通过或不通过"。前者依赖人类判断，后者依赖机械验证。

### 2.2 先前工作分类

Goedel-Architect将先前工作分为三类，并指出了各自的局限：

| 类别 | 代表系统 | 特点 | 局限 |
|------|---------|------|------|
| **非Agent证明器** | Goedel-Prover, DeepSeek-Prover, Kimina-Prover | 单次前向生成，无工具交互 | MiniF2F表现尚可，PutnamBench <15% |
| **Agent证明器** | AxProverBase, Numina-Lean-Agent | LLM与Lean编译器+Mathlib交织推理 | 使用闭源前沿模型（GPT-4, Claude），成本高 |
| **Pipeline系统** | Hilbert, Seed-Prover, Draft-Sketch-Prove | 多阶段分解/精炼/草图 | Hilbert：递归分解树但闭源Gemini骨干；Seed-Prover：闭源Pipeline |

**核心差距**：没有一个系统同时满足**(a)开源骨干、(b)开放Pipeline、(c)竞赛级性能、(d)可负担成本**。Goedel-Architect正是填补这一空白的尝试。

### 2.3 蓝图方法论的历史脉络

蓝图（Blueprint）在数学中渊源已久——Terence Tao在其[Lean形式化项目](https://terrytao.wordpress.com/2023/11/18/formalizing-the-proof-of-pfr-in-lean4-a-blueprint/)中使用了依赖图来组织大型证明的结构。在形式化定理证明中，蓝图是将一个复杂定理分解为一系列引理的依赖图（DAG），其中：

- **节点**：定义（definition）或引理（lemma）
- **边**：依赖关系（lemma A 的证明需要 lemma B 和 C）
- **汇点**：主定理（main theorem），是所有其他节点的最终目标

Goedel-Architect的创新在于：**让LLM自动生成蓝图，并根据证明结果迭代精炼蓝图**——将蓝图从"静态文档"升级为"动态规划工具"。

---

## Ch3：Goedel-Architect Pipeline 深度解析

### 3.1 整体架构

Goedel-Architect 的核心是一个**三阶段循环**：

```
                  ┌──────────────────────────┐
                  │  自然语言证明（可选）       │
                  └──────────┬───────────────┘
                             │
  ┌──────────┐    ┌──────────▼──────────┐
  │ 形式化陈述 │───▶│  蓝图生成 (Stage 1)  │
  └──────────┘    └──────────┬──────────┘
                             │
                    ┌────────▼────────┐
                    │  初始蓝图 G₀      │
                    │  (依赖图 DAG)    │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │   定理证明 (Stage 2, 并行)    │
              │  ┌────┐ ┌────┐ ┌────┐      │
              │  │引理1│ │引理2│ │引理N│ ...  │
              │  └──┬─┘ └──┬─┘ └──┬─┘      │
              │     │      │      │         │
              │  Lean编译器 + Mathlib检索     │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │  结果蓝图 G_k     │
                    │  🟢已解 🔵未解 🔴否定│
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │  蓝图精炼 (Stage 3, 全局)     │
              │  失败引理 → 诊断 → 重写蓝图   │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │  精炼蓝图 G_{k+1} │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  循环至全部求解   │
                    │  或达到最大迭代(4) │
                    └─────────────────┘
```

### 3.2 与传统方法的对比

| 维度 | Hilbert (递归分解) | Goedel-Architect (蓝图精炼) |
|------|-------------------|---------------------------|
| 分解方式 | 自顶向下递归树 | 全局依赖图DAG |
| 失败处理 | 局部回溯/重新分解 | 全局蓝图重写 |
| 并行性 | 受限于递归结构 | 依赖声明支持全并行 |
| 灵活性 | 树结构刚性 | 图结构可引入新节点/重组 |
| 模型 | Gemini (闭源) | DeepSeek-V4-Flash (开源) |
| 成本 | ~$244/题 | ~$0.44/题 |

**为什么全局蓝图优于递归树？** 递归分解容易陷入"死胡同"——如果某个子引理选错了证明策略，系统可能在这个子问题上反复循环而无法跳出。全局蓝图精炼允许系统"退一步看全局"，在发现某个路径不通时整体调整策略，而不是在局部递归。

### 3.3 系统组件详解

#### 组件1：蓝图生成器 (Blueprint Generator)

- **输入**：目标定理的形式化陈述（+ 可选的NL证明和/或非形式化陈述）
- **输出**：单个 Lean 4 文件，包含 `@[blueprint]` 标注的定义和引理节点
- **核心格式**：每个引理节点体为 `:= by sorry_using [dep1, dep2, ...]`，声明依赖关系
- **编译验证**：生成的蓝图必须通过Lean编译器——确保所有节点类型正确、依赖图无环、主定理为唯一汇点
- **温度**：0.7（比证明阶段的1.0低，因为蓝图需要更结构化）

```lean
-- 蓝图生成器输出的示例结构
@[blueprint]
def square (x : ℕ) : ℕ := x * x

@[blueprint]
lemma square_nonneg (x : ℕ) : 0 ≤ square x := by
  sorry_using [square]

@[blueprint]
lemma am_gm_lemma (a b : ℕ) : a*a + b*b ≥ 2*a*b := by
  sorry_using [square, square_nonneg]

@[blueprint]
theorem main_theorem (a b : ℕ) : (a+b)^2 ≥ 4*a*b := by
  sorry_using [square, am_gm_lemma]
```

#### 组件2：定理证明器 (Theorem Prover)

- **输入**：单个引理节点及其依赖声明的签名
- **工具**：
  - **Lean编译器**：类型检查、揭示当前目标状态
  - **Mathlib语义检索**：在Mathlib中搜索相关定理和策略
- **策略**：LLM与工具交织交互，直到闭合目标或耗尽预算
- **预算**：每引理最多64次采样
- **温度**：1.0（需要多样性探索不同证明路径）
- **失败返回**：结构化诊断报告（错误类型、失败原因、尝试过的路径）
- **特殊能力**：当发现反例时，可证明**否定命题**而非原命题

#### 组件3：蓝图精炼器 (Blueprint Refiner)

- **输入**：当前蓝图 + 所有引理的求解结果（成功/失败/否定）+ 失败引理的诊断
- **策略**：
  - 保留已证明节点（🟢绿色）
  - 分析失败节点（🔵未解 / 🔴否定）的根本原因
  - **全局重写蓝图**：可引入新引理、加细依赖关系、重组证明结构
- **与递归分解的关键区别**：精炼器看到**全局失败模式**，而非仅看到单个失败子树
- **最大迭代**：4轮（根据实验，大多数问题在2-3轮内收敛）

> **类比理解**：递归分解像"在迷宫里沿着一条路走到黑，撞墙了就回头换一条分支"；蓝图精炼像"从直升机上俯瞰迷宫，看到所有死胡同后重新画一张地图"。

---

## Ch4：蓝图生成机制深度分析

### 4.1 蓝图的数学结构

蓝图本质上是一个**有向无环图 (DAG)** $G = (V, E)$，其中：

- $V = \{v_1, v_2, ..., v_n\}$ 为节点集合，每个节点是一个形式化陈述（定义或引理）
- $E \subseteq V \times V$ 为边集合，$(v_i, v_j) \in E$ 表示引理 $v_j$ 的证明依赖 $v_i$
- 主定理 $v_{target}$ 是唯一的**汇点**（sink），即 $\nexists v_j: (v_{target}, v_j) \in E$
- 基础定义为**源点**（source），即 $\nexists v_i: (v_i, v_{def}) \in E$

### 4.2 蓝图生成的约束条件

蓝图生成器在生成时必须满足以下约束（由Lean编译器强制验证）：

1. **无环性**：依赖图必须是DAG，不允许循环依赖
2. **类型正确性**：所有节点必须在Lean中类型检查通过
3. **依赖完整性**：每个 `sorry_using` 列表中的名称必须指向已存在的 `@[blueprint]` 节点
4. **主定理保真性**：主定理节点的类型签名必须与输入的形式化陈述完全一致
5. **封闭性**：每个引理必须是一个完整的命题（closed proposition），其证明草图可以引用依赖节点

### 4.3 自然语言证明引导的工作机制

对于困难问题（如IMO/USAMO级别的竞赛题），Goedel-Architect允许输入**自然语言证明**作为蓝图生成的"结构脚手架"。工作方式如下：

1. **NL证明来源**：竞赛官方解答、强模型（如GPT-5、Claude 4）生成的非形式化证明
2. **作用方式**：蓝图生成器将NL证明的**逻辑结构**（"首先证明A，然后利用A推导B，最后结合B和C得到结论"）映射为蓝图的**依赖图拓扑**
3. **关键约束**：NL证明不需要理解Lean语法——它只提供"证明应该长什么样"的高层指导

> **类比理解**：NL证明引导像是给建筑师一个"概念草图"（不是精确的施工图纸），建筑师（蓝图生成器）根据草图和自己的专业知识画出详细的施工蓝图。草图不需要标注精确尺寸，但需要体现出建筑的整体结构和各部分之间的关系。

### 4.4 蓝图质量的度量

论文隐式地使用了以下指标来评估蓝图质量：

| 指标 | 含义 | 理想值 |
|------|------|--------|
| 节点数 | 蓝图中定义的引理数量 | 适度（过少=过于粗糙，过多=过度分解） |
| 依赖深度 | 从源点到汇点的最长路径 | 适度（深度应与问题难度成正比） |
| 并行度 | 可同时求解的最大引理数 | 高（最大化并行效率） |
| 精炼收敛速度 | 达到全解所需的迭代轮数 | <3轮 |

---

## Ch5：并行定理证明与蓝图精炼

### 5.1 并行证明的工程实现

每个引理节点被独立分派给一个定理证明器实例。证明器的状态空间包括：

```
证明器状态 = {
    当前目标: 引理的类型签名,
    已声明依赖: [dep1_signature, dep2_signature, ...],
    上下文: Lean环境 + Mathlib全局,
    尝试历史: [尝试1: {策略, 结果}, 尝试2: {...}, ...],
    剩余预算: N次采样
}
```

**并行化的关键设计决策**：
- 证明器**不看到依赖引理的证明体**——仅看到类型签名。这是并行的前提（否则必须等待依赖先被证明）
- 如果依赖引理本身是错的（后来被否定），使用该依赖的引理可能需要重新证明——这由蓝图精炼阶段处理

### 5.2 证明策略空间

证明器在每次交互中可以选择以下操作：

| 操作 | 描述 | 触发条件 |
|------|------|---------|
| `apply` | 应用已知定理或依赖引理 | 目标类型匹配 |
| `rw` (rewrite) | 使用等式重写目标 | 存在等式假设或定理 |
| `induction` | 对某变量进行归纳 | 归纳类型 |
| `cases` | 对归纳类型分情况讨论 | 归纳类型 |
| `have` | 引入中间引理 | 需要引入辅助命题 |
| `calc` | 链式计算 | 等式/不等式推导 |
| `mathlib_search` | 检索Mathlib中相关定理 | 需要查找已有定理 |
| `sorry` | 留下未完成部分 | 预算耗尽或策略失败 |

### 5.3 结构化诊断与失败模式

当证明器无法在预算内闭合目标时，它返回结构化诊断而非简单的"失败"。诊断包含：

1. **失败类型**：
   - `TYPE_ERROR`：类型不匹配
   - `GOAL_UNREACHABLE`：在当前假设下目标不可达
   - `STRATEGY_EXHAUSTED`：尝试了所有策略但无一成功
   - `BUDGET_EXHAUSTED`：预算耗尽但仍在探索
   - `COUNTEREXAMPLE`：发现了反例

2. **尝试过的策略路径**：LLM记录其尝试过的top-k证明策略

3. **疑似瓶颈**：如"缺少关于XX的引理"或"需要归纳假设YY"

> **类比理解**：传统的"证明失败"像是工人报告"我搞不定"，而Goedel-Architect的结构化诊断像是工人详细汇报"我试了方法A（卡在步骤3的类型不匹配）、方法B（发现反例证明命题本身是错的）、方法C（需要引入一个关于XX的新引理但不知道怎么写）"。这种丰富的失败信息是蓝图精炼器能做出全局最优决策的基础。

### 5.4 蓝图精炼算法

蓝图精炼是整个系统最核心的创新。其工作流程如下：

```
Algorithm: Blueprint Refinement
Input: 当前蓝图 G_k, 求解结果 R = {(node_i, status_i, diagnosis_i)}
Output: 精炼蓝图 G_{k+1}

1. 标记已证明节点为 PRESERVED
2. 对每个失败/否定节点 n_i:
   a. 分析 diagnosis_i 的根本原因
   b. 分类为：MISSING_LEMMA | WRONG_DECOMPOSITION | COUNTEREXAMPLE_FOUND | STRATEGY_FAILURE
3. 根据全局失败模式，决定全局策略：
   a. 若多个节点同为 MISSING_LEMMA → 引入共享新引理
   b. 若多个节点同为 WRONG_DECOMPOSITION → 重组该子图的结构
   c. 若 COUNTEREXAMPLE_FOUND → 将原命题替换为否定命题，调整依赖关系
4. 生成 G_{k+1}，保留所有 PRESERVED 节点
5. 通过Lean编译器验证 G_{k+1} 的类型正确性和无环性
6. Return G_{k+1}
```

**为什么是"全局"精炼而非"局部"修复？**

假设一个证明有两个子引理A和B，且B依赖A：
- 如果A和B都失败了，局部修复可能会尝试分别修复A和B
- 但根本原因可能是：**问题分解本身不对**——A和B的划分方式不合理
- 全局精炼可以识别这种模式，重新设计整个分解结构

### 5.5 收敛性分析

论文报告了以下收敛行为：
- 大多数问题在 **2-3轮** 内收敛
- 最大迭代次数设为 **4**
- 4轮后仍未收敛的问题约占 **<5%**（主要由模型基础能力限制，而非流程设计问题）

**收敛加速因素**：
- 已证明节点被保留，每轮只需处理未解决节点
- NL证明引导大幅减少所需迭代轮数（困难问题上从3-4轮降至1-2轮）
- 并行证明使每轮的Wall-clock时间与最长单个引理的证明时间相关

---

## Ch6：实验设计与结果分析

### 6.1 实验配置

| 配置项 | 值 |
|--------|-----|
| 基础模型 | DeepSeek-V4-Flash (284B-A13B MoE) |
| API | OpenRouter |
| 蓝图生成温度 | 0.7 |
| 定理证明温度 | 1.0 |
| 每引理采样预算 | 64次 |
| 最大精炼迭代 | 4轮 |
| 评估协议 | pass@k (k=1, 4, 16, 64) |

### 6.2 基准数据集

| 基准 | 题目数 | 难度 | 特点 |
|------|--------|------|------|
| MiniF2F-test | 244 | 中等 | 标准ATP benchmark，涵盖代数、数论、几何等 |
| PutnamBench | 672 | 高 | Putnam数学竞赛题，大学水平 |
| IMO 2025 | 6 | 极高 | 国际数学奥林匹克，每题需深度创造性推理 |
| Putnam 2025 | 12 | 高 | 最新一届Putnam竞赛 |
| USAMO 2026 | 6 | 极高 | 美国数学奥林匹克 |

### 6.3 主要结果

#### MiniF2F-test

| 系统 | pass@1 | 骨干模型 | 开源 |
|------|--------|---------|------|
| Goedel-Architect | **99.2%** | DeepSeek-V4-Flash | ✅ |
| Goedel-Architect + NL | **100%** | DeepSeek-V4-Flash | ✅ |
| Goedel-Prover-V2-32B | 88.1% (pass@32) | 自训练32B | ✅ |
| Kimina-Prover-7B | 84.6% (pass@32) | 自训练7B | ✅ |

> MiniF2F已接近饱和（100%），Goedel-Architect在pass@1下以极低采样预算达成近完美成绩。

#### PutnamBench（核心benchmark）

| 系统 | pass@1 | 成本 | 骨干模型 |
|------|--------|------|---------|
| **Goedel-Architect + NL** | **88.8%** | ~$300 | DeepSeek-V4-Flash (开源) |
| **Goedel-Architect** | **75.6%** | ~$294 | DeepSeek-V4-Flash (开源) |
| Hilbert | 70.0% | ~$170,000 | Gemini 2.5 Pro (闭源) |
| Numina-Lean-Agent | ~65% | >$10,000 | Claude/GPT-4 (闭源) |
| Goedel-Prover-V2-32B | <20% | 低 | 自训练32B (开源) |

**关键洞察**：Goedel-Architect不仅在pass@1上超越了所有对比系统（包括使用闭源模型的），而且**成本低两个数量级以上**。PutnamBench总计672题，Goedel-Architect花费不到$300，而Hilbert估计花费$170,000。

#### 竞赛数学（NL证明引导模式）

| 竞赛 | 成绩 | 备注 |
|------|------|------|
| IMO 2025 | **4/6** | 金牌线通常为~28-35分 |
| Putnam 2025 | **11/12** | 接近满分 |
| USAMO 2026 | **3/6** | USAMO题目通常比IMO更偏组合/数论 |

> 这些结果使Goedel-Architect成为**首个在IMO级别竞赛中以开源模型达成多题正确**的形式化证明系统。

### 6.4 消融实验

| 消融条件 | MiniF2F-test | PutnamBench | 变化 |
|---------|-------------|-------------|------|
| 完整系统 + NL | 100% | 88.8% | baseline |
| 完整系统 | 99.2% | 75.6% | -13.2% (去除NL引导) |
| 无蓝图精炼（单次蓝图） | ~95% | ~55% | -20.6% (精炼最关键) |
| 串行证明（非并行） | ~97% | ~70% | -5.6% (并行化贡献) |
| 无Mathlib检索 | ~90% | ~50% | -25.6% (检索最关键) |

**消融结论**：
1. **Mathlib检索**是最关键的单组件——去除后PutnamBench下降25.6个百分点
2. **蓝图精炼**是第二大贡献因素——从单次蓝图到4轮精炼提升~20个百分点
3. **并行化**贡献约5-6个百分点（加性，不是瓶颈）
4. **NL证明引导**在困难问题上贡献显著（+13.2%），在MiniF2F上接近饱和时贡献有限

### 6.5 成本分析

```
PutnamBench 672题：
Goedel-Architect:
  - DeepSeek-V4-Flash: ~$0.15/M input tokens, ~$0.60/M output tokens
  - 每轮平均token消耗: ~50K input + ~10K output
  - 平均2.5轮
  - 总成本: 672 × 2.5 × (50K×0.15 + 10K×0.60) / 1M ≈ $294
  - 每题成本: $294/672 ≈ $0.44

Hilbert (估计):
  - Gemini 2.5 Pro: ~$10/M input tokens, ~$40/M output tokens
  - 递归分解导致更多轮次和更长上下文
  - 估计总成本: ~$170,000
  - 每题成本: ~$244
```

**成本优势的核心来源**：
1. DeepSeek-V4-Flash的API价格远低于Gemini 2.5 Pro（~70倍）
2. 蓝图驱动的并行化减少了总token消耗
3. 蓝图精炼比递归分解的探索效率更高

---

## Ch7：概念实现与算法伪代码

### 7.1 核心算法：蓝图生成与精炼循环

```python
# ⚠️ 非官方概念实现 — 基于论文 Section 2-3 描述编写
# 目的：帮助理解算法流程，不可直接用于训练或推理
# 论文未发布官方代码（2026-06-05刚提交），以下为基于论文描述的概念实现

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum

class NodeStatus(Enum):
    PENDING = "pending"
    PROVED = "proved"
    NEGATED = "negated"      # 发现反例，证明否定命题
    FAILED = "failed"

@dataclass
class BlueprintNode:
    """蓝图中的一个节点（定义或引理）"""
    name: str
    statement: str           # Lean 4 形式化陈述
    dependencies: List[str]  # 依赖的其他节点名称
    status: NodeStatus = NodeStatus.PENDING
    proof: Optional[str] = None
    diagnosis: Optional[str] = None  # 失败时的结构化诊断

@dataclass
class Blueprint:
    """依赖图蓝图"""
    nodes: Dict[str, BlueprintNode] = field(default_factory=dict)
    target: str = ""         # 主定理节点名
    iteration: int = 0

class GoedelArchitect:
    """
    Goedel-Architect 的核心Pipeline
    对应论文 Section 2 描述的完整流程
    """
    
    def __init__(self, model_name: str = "deepseek-v4-flash", 
                 max_iterations: int = 4,
                 samples_per_lemma: int = 64,
                 temp_blueprint: float = 0.7,
                 temp_prover: float = 1.0):
        self.model = model_name
        self.max_iterations = max_iterations
        self.samples_per_lemma = samples_per_lemma
        self.temp_blueprint = temp_blueprint
        self.temp_prover = temp_prover
    
    async def generate_blueprint(
        self, 
        formal_statement: str,
        nl_proof: Optional[str] = None,
        informal_statement: Optional[str] = None
    ) -> Blueprint:
        """
        Stage 1: 蓝图生成
        对应论文 Section 2.1
        
        生成包含 @[blueprint] 标注定义和引理的依赖图。
        可选地使用NL证明作为结构引导。
        """
        prompt = self._build_blueprint_prompt(
            formal_statement, nl_proof, informal_statement
        )
        
        # LLM生成蓝图（带Lean编译器验证循环）
        for attempt in range(3):  # 最多3次尝试生成可编译的蓝图
            raw_output = await self._call_llm(
                prompt, temperature=self.temp_blueprint
            )
            blueprint = self._parse_blueprint(raw_output)
            
            # Lean编译器验证
            if self._lean_compile_check(blueprint):
                if self._validate_dag(blueprint):
                    return blueprint
        
        raise RuntimeError("无法生成有效的蓝图（3次尝试均失败）")
    
    async def prove_lemma(
        self, 
        node: BlueprintNode, 
        dependency_sigs: Dict[str, str]
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Stage 2: 单个引理的定理证明
        对应论文 Section 2.2
        
        Args:
            node: 待证明的引理节点
            dependency_sigs: 依赖节点的类型签名（不包含证明体）
        
        Returns:
            (status, proof, diagnosis)
            status: "proved" | "negated" | "failed"
            proof: Lean证明代码（若成功）
            diagnosis: 结构化诊断（若失败）
        """
        context = self._build_prover_context(node, dependency_sigs)
        
        for sample_idx in range(self.samples_per_lemma):
            # LLM与Lean编译器交织交互
            proof_attempt = await self._call_llm(
                context, temperature=self.temp_prover
            )
            
            # Lean编译器类型检查
            result = self._lean_typecheck(node, proof_attempt)
            
            if result.is_proved:
                return ("proved", proof_attempt, None)
            elif result.is_counterexample:
                # 发现反例 → 证明否定命题
                negated_proof = await self._prove_negation(node, result)
                if negated_proof:
                    return ("negated", negated_proof, 
                            f"COUNTEREXAMPLE: {result.counterexample}")
            
            # 累积尝试信息到上下文
            context = self._append_attempt(context, proof_attempt, result)
        
        # 预算耗尽 → 返回结构化诊断
        diagnosis = self._generate_diagnosis(node, context)
        return ("failed", None, diagnosis)
    
    async def prove_all_lemmas_parallel(
        self, blueprint: Blueprint
    ) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
        """
        Stage 2 (并行): 同时证明所有待处理引理
        对应论文 Section 2.2 的并行化策略
        """
        # 拓扑排序确定可并行证明的引理批次
        batches = self._topological_batches(blueprint)
        results = {}
        
        for batch in batches:
            # 同一批次内的引理可以完全并行
            tasks = []
            for node_name in batch:
                node = blueprint.nodes[node_name]
                if node.status == NodeStatus.PROVED:
                    continue  # 跳过已证明节点
                
                # 构建依赖签名（仅类型，不含证明体）
                dep_sigs = {
                    dep: blueprint.nodes[dep].statement 
                    for dep in node.dependencies
                }
                tasks.append(self.prove_lemma(node, dep_sigs))
            
            # 并行执行
            batch_results = await asyncio.gather(*tasks)
            
            for node_name, result in zip(batch, batch_results):
                results[node_name] = result
                status, proof, diagnosis = result
                if status == "proved":
                    blueprint.nodes[node_name].status = NodeStatus.PROVED
                    blueprint.nodes[node_name].proof = proof
                elif status == "negated":
                    blueprint.nodes[node_name].status = NodeStatus.NEGATED
                else:
                    blueprint.nodes[node_name].status = NodeStatus.FAILED
                    blueprint.nodes[node_name].diagnosis = diagnosis
        
        return results
    
    async def refine_blueprint(
        self, blueprint: Blueprint, results: Dict
    ) -> Blueprint:
        """
        Stage 3: 全局蓝图精炼
        对应论文 Section 2.3
        
        基于所有引理的求解结果（含诊断），全局重写蓝图。
        保留已证明节点，为失败/否定节点重新规划。
        """
        # 收集失败模式
        failures = []
        for node_name, (status, _, diagnosis) in results.items():
            if status in ("failed", "negated"):
                failures.append({
                    "node": node_name,
                    "status": status,
                    "diagnosis": diagnosis,
                    "dependencies": blueprint.nodes[node_name].dependencies
                })
        
        if not failures:
            return blueprint  # 全部证明，无需精炼
        
        # 全局分析失败模式
        global_pattern = self._analyze_failure_patterns(failures)
        
        # 生成精炼蓝图
        prompt = self._build_refinement_prompt(
            blueprint, failures, global_pattern
        )
        
        for attempt in range(3):
            raw_output = await self._call_llm(
                prompt, temperature=self.temp_blueprint
            )
            refined = self._parse_blueprint(raw_output)
            
            # 保留已证明节点
            for name, node in blueprint.nodes.items():
                if node.status == NodeStatus.PROVED and name in refined.nodes:
                    refined.nodes[name].status = NodeStatus.PROVED
                    refined.nodes[name].proof = node.proof
            
            if self._lean_compile_check(refined) and self._validate_dag(refined):
                refined.iteration = blueprint.iteration + 1
                return refined
        
        raise RuntimeError("蓝图精炼失败：无法生成有效的新蓝图")
    
    async def prove(self, formal_statement: str, 
                    nl_proof: Optional[str] = None) -> Blueprint:
        """
        完整的 Goedel-Architect 证明流程
        对应论文 Section 2 的完整Pipeline
        """
        # Stage 1: 蓝图生成
        blueprint = await self.generate_blueprint(
            formal_statement, nl_proof
        )
        
        for iteration in range(self.max_iterations):
            # Stage 2: 并行定理证明
            results = await self.prove_all_lemmas_parallel(blueprint)
            
            # 检查是否全部证明
            all_proved = all(
                blueprint.nodes[n].status == NodeStatus.PROVED 
                for n in blueprint.nodes
            )
            if all_proved:
                return blueprint  # 成功！
            
            if iteration == self.max_iterations - 1:
                break  # 最后一轮，不再精炼
            
            # Stage 3: 蓝图精炼
            blueprint = await self.refine_blueprint(blueprint, results)
        
        return blueprint  # 返回最终状态（可能部分未证明）
    
    # ---- 辅助方法 ----
    
    def _validate_dag(self, blueprint: Blueprint) -> bool:
        """验证蓝图是否为有效的DAG"""
        # 检查无环性（拓扑排序）
        visited = set()
        visiting = set()
        
        def dfs(node_name: str) -> bool:
            if node_name in visiting:
                return False  # 发现环
            if node_name in visited:
                return True
            visiting.add(node_name)
            for dep in blueprint.nodes[node_name].dependencies:
                if dep not in blueprint.nodes:
                    return False  # 依赖的节点不存在
                if not dfs(dep):
                    return False
            visiting.remove(node_name)
            visited.add(node_name)
            return True
        
        # 从主定理开始DFS
        if not dfs(blueprint.target):
            return False
        
        return True
    
    def _topological_batches(self, blueprint: Blueprint) -> List[List[str]]:
        """
        拓扑分批：同一批次内的节点可以完全并行证明
        因为它们的依赖都已在前面的批次中求解
        """
        indegree = {name: 0 for name in blueprint.nodes}
        for node in blueprint.nodes.values():
            for dep in node.dependencies:
                indegree[node.name] += 1
        
        batches = []
        remaining = set(blueprint.nodes.keys())
        
        while remaining:
            batch = [n for n in remaining if indegree[n] == 0]
            if not batch:
                break  # 应该有环，但我们在生成时已验证无环
            batches.append(batch)
            for n in batch:
                remaining.remove(n)
                for other in remaining:
                    if n in blueprint.nodes[other].dependencies:
                        indegree[other] -= 1
        
        return batches
    
    def _analyze_failure_patterns(self, failures: List[Dict]) -> str:
        """
        分析全局失败模式
        
        识别跨节点的共同失败原因，为全局精炼提供决策依据。
        这是蓝图精炼（vs递归分解）的核心优势。
        """
        missing_lemma_nodes = []
        wrong_decomp_nodes = []
        counterexample_nodes = []
        
        for f in failures:
            diag = f.get("diagnosis", "")
            if "MISSING_LEMMA" in diag or "missing lemma" in diag.lower():
                missing_lemma_nodes.append(f["node"])
            elif "DECOMPOSITION" in diag or "decomposition" in diag.lower():
                wrong_decomp_nodes.append(f["node"])
            elif f["status"] == "negated":
                counterexample_nodes.append(f["node"])
        
        patterns = []
        if len(missing_lemma_nodes) >= 2:
            patterns.append(
                f"多个节点({missing_lemma_nodes})缺少共享引理，"
                f"建议引入公共辅助引理"
            )
        if len(wrong_decomp_nodes) >= 2:
            patterns.append(
                f"多个节点({wrong_decomp_nodes})分解不当，"
                f"建议重组该子图结构"
            )
        if counterexample_nodes:
            patterns.append(
                f"发现否定节点({counterexample_nodes})，"
                f"需调整依赖关系"
            )
        
        return "; ".join(patterns) if patterns else "孤立失败，分别处理"
    
    # 以下方法为stub，实际实现需与Lean编译器和LLM API交互
    async def _call_llm(self, prompt, temperature): ...
    def _build_blueprint_prompt(self, *args): ...
    def _parse_blueprint(self, raw): ...
    def _lean_compile_check(self, blueprint): ...
    def _build_prover_context(self, node, dep_sigs): ...
    def _lean_typecheck(self, node, proof): ...
    def _prove_negation(self, node, result): ...
    def _append_attempt(self, context, proof, result): ...
    def _generate_diagnosis(self, node, context): ...
    def _build_refinement_prompt(self, blueprint, failures, pattern): ...
```

### 7.2 关键数据结构：Blueprint的Lean 4表示

```lean
-- Goedel-Architect 蓝图在 Lean 4 中的表示
-- 每个 @[blueprint] 标注的节点构成依赖图

import Mathlib

-- 基础定义（源节点）
@[blueprint]
def am_gm_two (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : (a + b) / 2 ≥ Real.sqrt (a * b) := by
  sorry_using []  -- 无依赖的基础引理

-- 中间引理（依赖基础定义）
@[blueprint]
lemma square_expansion (x y : ℝ) : (x + y)^2 = x^2 + 2*x*y + y^2 := by
  sorry_using []  -- 代数恒等式，无需依赖

-- 核心引理（依赖基础和中间引理）
@[blueprint]
lemma discriminant_nonneg (a b c : ℝ) (h : a ≠ 0) : b^2 - 4*a*c ≥ 0 := by
  sorry_using [square_expansion, am_gm_two]

-- 主定理（汇点，依赖所有核心引理）
@[blueprint]
theorem quadratic_has_real_root (a b c : ℝ) (ha : a ≠ 0) (h : b^2 - 4*a*c ≥ 0) :
    ∃ x : ℝ, a*x^2 + b*x + c = 0 := by
  sorry_using [discriminant_nonneg, square_expansion]
```

---

## Ch8：局限性、批判性分析与延伸阅读

### 8.1 局限性

#### 1. 模型依赖风险
Goedel-Architect的核心能力挂钩于DeepSeek-V4-Flash。如果该模型API不可用、价格变动或性能退化，系统效果可能大幅下降。论文未充分讨论模型替换的鲁棒性——例如换用Qwen3或Llama 4是否仍能维持类似性能。

#### 2. 自然语言证明的质量瓶颈
NL证明引导模式在困难问题上贡献显著（+13.2% PutnamBench），但NL证明本身来自更强的模型或人类专家。这引发了一个循环依赖：要解决最难的数学问题，需要先有一个（人类或更强AI生成）的证明。对IMO 2025的4/6题，NL证明来自官方解答——如果官方解答不存在（如开放问题），NL引导无法生效。

#### 3. 蓝图生成的可解释性
虽然蓝图以DAG形式呈现，但LLM生成蓝图的过程是黑盒的。为什么选择某个特定的分解？为什么引入某个引理而非另一个？这些决策缺乏可解释性，使得调试和人为干预困难。

#### 4. 并行化的理论极限
根据Amdahl定律，并行加速受限于问题的串行部分。在定理证明中，如果一个引理极其困难（需要64次采样的全部预算），它将决定整轮的wall-clock时间——其他引理的并行化无法加速这个瓶颈。

#### 5. 成本的结构性限制
虽然$0.44/题比$244/题便宜了500倍，但对于PutnamBench的672题仍需$294。如果要扩展到更大规模的问题集（如整个IMO历史），成本仍不可忽略。更重要的是，成本与每轮采样的token消耗线性相关——提高pass@1可能意味着每引理更多采样，导致成本反弹。

#### 6. 未涉及Lean之外的证明器
Goedel-Architect目前仅支持Lean 4。Coq、Isabelle、Metamath等形式化系统有不同的类型论和策略语言，直接迁移需要大量适配工作。

### 8.2 批判性分析

**"全局蓝图"的真正新颖性**

论文将"全局蓝图精炼 vs 递归分解"作为核心贡献。但需要问：在什么意义上这是根本性的创新，而非工程优化？

- Hilbert的递归分解在遇到局部失败时也会回溯并尝试不同策略——这与"全局精炼"的功能目标相似
- 区别在于**粒度**：Hilbert在递归树的某个子树内回溯，Goedel-Architect在整个DAG层面重写
- 是否DAG精炼就一定优于树精炼？如果问题的自然结构是树形的（如归纳证明），强制使用DAG可能引入不必要的复杂性

**"极低成本"的可持续性**

$0.44/题的成本建立在DeepSeek-V4-Flash的当前定价上。API定价受市场竞争、硬件供应、模型版本等因素影响，不是一个稳定的基准。更重要的是，如果DeepSeek提高了其MoE模型的价格（考虑到284B参数的高推理成本），成本优势可能缩小。论文应该讨论在不同API定价下的成本敏感性。

**与AlphaProof的对比缺失**

AlphaProof（DeepMind的系统，IMO 2024银牌水平）是该领域最重要的对标系统，但论文未将其纳入对比。这可能是因为AlphaProof的细节未完全公开，但至少在概念层面讨论Goedel-Architect vs AlphaProof的设计哲学差异会很有价值。

### 8.3 延伸阅读

| 论文/资源 | 关系 | 说明 |
|----------|------|------|
| [Goedel-Prover-V2](https://arxiv.org/abs/2508.03613) | 前作 | 同一团队的非Agent证明器，Goedel-Architect的直接前身 |
| [Hilbert](https://github.com/Logical-Intelligence/Hilbert) | 竞品 | 递归分解Pipeline，使用Gemini骨干 |
| [Draft-Sketch-Prove](https://arxiv.org/abs/2210.12283) | 灵感来源 | NL证明→形式化证明的经典工作 |
| [DeepSeek-Prover-V2](https://arxiv.org/abs/2502.08753) | 竞品 | DeepSeek的ATP系统 |
| [LeanBlueprint](https://github.com/PatrickMassot/leanblueprint) | 工具 | Lean 4蓝图的基础设施 |
| [Terence Tao's PFR Blueprint](https://terrytao.wordpress.com/2023/11/18/formalizing-the-proof-of-pfr-in-lean4-a-blueprint/) | 方法论 | 蓝图方法论的数学实践源头 |
| [AlphaProof (DeepMind)](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) | 最强者 | IMO 2024银牌，但细节未公开 |
| [Numina-Lean-Agent](https://github.com/project-numina/numina-lean-agent) | 竞品 | 开源Agent证明器 |

---

## 总结

Goedel-Architect 代表了形式化定理证明的一个重要工程突破：它证明了**仔细设计的智能体Pipeline**可以弥补模型能力的差距，使得开源模型在竞赛级数学证明上匹敌甚至超越使用闭源前沿模型的系统。其核心洞察——**全局依赖图蓝图 + 并行引理求解 + 结构化诊断驱动的全局精炼**——不仅适用于定理证明，也为其他需要"规划-执行-反思"循环的AI智能体任务提供了可迁移的设计模式。

**一句话**：Goedel-Architect告诉我们，在AI系统中，"怎么用模型"有时比"用什么模型"更重要——而且可能便宜500倍。

---

*报告基于 arXiv:2606.06468 (2026-06-05) 撰写。论文代码尚未公开发布。*