# Advancing Mathematics Research with AI-Driven Formal Proof Search


---

> **作者**: George Tsoukalas, Anton Kovsharov, Sergey Shirobokov et al.
> **机构**: Google DeepMind
> **发布日期**: 2026-05-21
> **arXiv链接**: https://arxiv.org/abs/2605.22763
> **代码链接**: https://github.com/google-deepmind/alphaproof-nexus-results
> **关键词**: Formal Proof Search, Lean, AlphaProof Nexus, Erdős Problems, Evolutionary Search, LLM Agent, Theorem Proving

---

## 一、研究背景与动机

### 1.1 LLM 在数学中的困境：能力与可靠性的矛盾

大语言模型在数学推理方面取得了显著进展，但存在一个根本性矛盾：

- **能力强**：LLM 能够理解复杂的数学问题、提出猜想、甚至给出直觉上合理的证明思路
- **不可靠**：LLM 经常在证明的中间步骤产生"幻觉"（hallucination）——看似正确但逻辑上存在微妙错误的推导

这种不可靠性严重限制了 LLM 在**真正的数学研究**中的应用价值。一个错误的证明步骤可能导致整个结论失效，而人工检查每一步的成本极高。

### 1.2 形式化证明：机器验证的可靠性保障

Lean 等形式化证明语言提供了一种解决方案：

- 每一步推理都必须通过**编译器验证**
- 不存在"几乎正确"——要么通过编译，要么不通过
- 这为 LLM 生成的证明提供了自动化的质量保证

> "A mitigation is using LLMs to generate formal proofs in languages like Lean."

### 1.3 为什么这个问题很重要？

Erdős 问题列表包含 353 个由传奇数学家 Paul Erdős 提出的开放问题，部分问题已悬而未决超过 50 年。传统上，这些问题需要顶尖数学家投入数月甚至数年的精力。而 AlphaProof Nexus 以**每个问题约 60 美元**的成本，自主解决了其中 9 个。

---

## 二、核心贡献

1. **提出 AlphaProof Nexus 框架**：一个基于 LLM 的形式化证明搜索系统，结合进化算法与 Lean 编译器反馈，自主生成经过形式验证的数学证明
2. **解决了 9 个开放 Erdős 问题**：包括两个自 1970 年以来悬而未决 56 年的问题（#12(i) 和 #12(ii)），以及 Ben Green 的 100 个问题列表中的问题
3. **证明了 44/492 个 OEIS 猜想**：从 2649 个开放猜想中筛选、形式化并证明
4. **跨学科应用**：在优化理论、代数几何、图论、量子光学等多个数学分支中取得研究成果
5. **消融实验揭示关键设计原则**：令人惊讶地发现，即使是基础 agent 也能解决所有 9 个 Erdős 问题，但高级组件在难题上提供 2-5 倍的成本效率提升

---

## 三、方法详解

### 3.1 整体架构：AlphaProof Nexus 框架

AlphaProof Nexus 的输入输出非常简洁：

- **输入**：一个包含 `sorry` 占位符的 Lean 文件（即"证明草图"），加上 `EVOLVE-BLOCK`（可编辑代码区域）和 `EVOLVE-VALUE`（可调参数）标记
- **输出**：一个不含 `sorry` 的、通过 Lean 编译器验证的完整形式化证明

```
-- EVOLVE-BLOCK-START --
-- 你可以在这里放置你的定义和引理
-- EVOLVE-BLOCK-END

theorem target_theorem : answer(
  -- EVOLVE-VALUE-START 默认值 -- EVOLVE-VALUE-END
) ↔ 0 < (...).lowerDensity := by
  -- EVOLVE-BLOCK-START sorry -- EVOLVE-BLOCK-END
```

### 3.2 四级 Agent 架构

论文设计了从简单到复杂的四个 agent 配置，系统地研究每个组件的贡献：

#### Agent (A)：基础 Agent

- **核心机制**：独立运行的"Ralph 循环"子 agent
- **工作流程**：LLM（Gemini 3.1 Pro）生成代码修改 → Lean 编译器反馈错误 → LLM 根据错误修正代码 → 重复
- **特点**：没有共享状态，每个子 agent 独立搜索

#### Agent (B)：基础 + AlphaProof

- 在 (A) 的基础上增加了 AlphaProof 工具
- AlphaProof 是一个基于强化学习的奥林匹克级定理证明系统
- 当 LLM 遇到困难的子目标时，可以调用 AlphaProof 辅助解决
- AlphaProof 会返回：证明 / 反证 / 失败

#### Agent (C)：基础 + 进化

- 在 (A) 的基础上增加了**共享种群数据库**
- 使用 Gemini 3.0 Flash 对候选证明草图进行 Elo 评分
- 通过 P-UCB（Popularity Upper Confidence Bound）采样策略选择有潜力的搜索方向

#### Agent (D)：完整 Agent（用于所有大规模评估）

- 结合 (B) 和 (C) 的所有能力
- 这是论文的主要成果来源

### 3.3 进化搜索机制

#### Elo 评分系统

- 每次"比赛"包含 P=7 个证明草图
- 由 Gemini 3.0 Flash 评估质量并排名
- 使用 Plackett-Luce 模型 + 层次化 Gamma 先验
- 通过 Gibbs 采样估计参数（I=1000 样本，B=200 burn-in）
- Elo 分数公式：`Elo_s = 1200 + 400 · log₁₀(λ_mean)`

#### P-UCB 采样策略

```
score = q + c · √(ΣVᵢ / (v+1))
```

其中：
- `q` 是候选的平均 Elo 分数
- `c = 0.2` 是探索系数
- `ΣVᵢ` 是访问次数之和
- `v` 是该候选被选中的次数
- 保留 top-64 候选

**直觉解释**：这就像一个"有经验的导师"——既倾向于推荐看起来有潜力的方向（利用），又不时探索新的可能（探索），避免陷入局部最优。

### 3.4 安全机制：SafeVerify

AlphaProof Nexus 包含一个关键的安全层 **SafeVerify**：

- 防止 agent 利用环境漏洞（如注入 `axiom` 声明来绕过证明）
- 每个生成的证明都经过 Lean 编译器的完整验证
- 确保最终输出是真正的数学证明，而非"伪证"

### 3.5 关键技术细节

| 参数 | 数值 |
|------|------|
| 每回合 AlphaProof 查询上限 | 5 次 |
| 每回合 search-replace 编辑上限 | 90 次 |
| 评分模型 | Gemini 3.0 Flash |
| 证明生成模型 | Gemini 3.1 Pro |
| 全局目标缓存 | 基于 Lean 状态的深度哈希 |
| 单个 Erdős 问题的成本 | ~$60 USD（约 27.5 TPU 小时）|

---

## 四、实验结果

### 4.1 实验设置

- **主评测基准**：353 个开放 Erdős 问题（由 Paul Erdős 提出的经典组合数学猜想）
- **辅助评测**：OEIS（在线整数序列百科全书）中的开放猜想
- **跨学科验证**：优化、代数几何、图论、量子光学中的开放问题
- **Agent 对比**：(A)(B)(C)(D) 四种配置在相同问题集上的表现

### 4.2 主要结果

#### Erdős 问题（9/353 解决）

| 编号 | 年份 | 猜想简述 | 证明方法 |
|------|------|---------|---------|
| **12(i)** | 1970 | 存在具有特定整除约束和正密度的无穷集合 | CRT + 无 3-AP 集合的分块构造 |
| **12(ii)** | 1970 | 12(i) 的更强密度版本 | Behrend 式密集 3-AP-free 块 + CRT |
| **125** | 1996 | 基 3 + 基 4 限制数字集的零下密度 | 丢番图逼近的归纳稀疏化 |
| **138*** | 1981 | van der Waerden 数满足 W(k+1)-W(k)→∞ | 贪心着色扩展 + 单色交集引理 |
| **152** | 1994 | 大 Sidon 集在 A+A 中有大量孤立点 | 内点基本界 + 四元组转移 |
| **741(i)** | 1994 | 可分解集合使两部分和集均有正上密度 | 交替块划分 + 和集界 |
| **741(ii)** | 1994 | 2 阶基中存在无法使两部分和集都有界间隙的划分 | 快速增长序列的显式构造 |
| **846** | 1992 | 无穷集合具有许多非共线三点子集但不是有限个非共线集的并 | K∞ 边标记 → ℝ² 映射 + Ramsey 定理 |
| **26*** | 1995 | 存在倒数和发散但 A+k 密度上界 < 3/4 的集合 | CRT 分块 + 素数模约束构造 |

**亮点**：
- Erdős #12(i) 和 #12(ii) 自 1970 年以来已开放 **56 年**
- Erdős #138* 是 Ben Green 的"百大问题"列表中的问题
- 这些问题涉及加性组合数学、图论、数论等多个核心数学领域

#### OEIS 猜想（44/492 证明）

- 从 2649 个开放猜想 → Gemini 筛选 500 个 → 492 个成功形式化（8 个因 Lean 版本升级被排除）
- **安全机制**：agent 先证明"测试引理"验证序列前几项的正确性，再尝试完整猜想
- 示例：A051293（整数平均子集的渐近展开）和 A228143（Hankel 行列式生成函数根的整数系数）

#### 跨学科成果

| 领域 | 成果 |
|------|------|
| **优化理论** | 证明了 Anchored GDA 的精确 O(1/t) 收敛速率 |
| **代数几何** | 解决了 ~15 年开放问题：纯 O-序列（余维 3，类型 2）的对数凹性 |
| **图论** | 证明了二部重构猜想变体；解决了 Graffiti "墙上留言"猜想 2（1996 年提出） |
| **加性组合数学** | 帮助解决了 Green 问题 #57（找到了复值版本的**反例**） |
| **量子光学** | 解决了多个单色量子图猜想（N=d∈{4,6,10}） |

### 4.3 Agent 架构消融实验

这是论文最令人惊讶的发现之一：

#### 基础 Agent 也能解决所有问题

> "Remarkably, the basic agent (A) solved all 9 problems, though at a higher cost on the harder problems."

在 9 个 Erdős 问题的 post-hoc 分析中：
- **Agent (A) 和 (B)** 在 4/6 个分析问题上表现相近
- **Agent (B)** 在 #12(ii) 和 #125 上更高效
- **Agent (D)** 在最难题目（#125, #138）上便宜 **2-5 倍**
- 但 Agent (D) 在简单题目上反而效率减半（额外开销）

#### 小模型和 AlphaProof 独立运行的失败

- **更小的 LLM**（Gemini 3.0 Flash, Flash-Lite）：**无法解决任何问题**
- **AlphaProof 独立运行**（树搜索模式，64 TPU 小时/题）：**无法解决任何问题**
- 这表明 LLM 的"数学创造力" + 形式验证的"可靠性保证"必须结合才能成功

---

## 五、关键发现与洞察

### 5.1 "简单 Agent 的惊人力量"

最出乎意料的发现是：**即使是基础的 agent (A)**——仅由 LLM 生成 + Lean 反馈组成，没有任何进化搜索或 AlphaProof 辅助——也能解决全部 9 个 Erdős 问题。

这意味着：
- 当 LLM 的推理能力足够强时（Gemini 3.1 Pro），简单的"生成-反馈"循环已经具有强大的证明搜索能力
- 复杂的进化机制主要提供**成本效率**而非**能力上的质变**
- 随着 LLM 能力的进一步提升，更简单的 agent 架构可能就足够了

### 5.2 人机协作的新模式

AlphaProof Nexus 展示了一种全新的数学研究范式：

1. **人类数学家**：提出问题、设计证明草图、形式化猜想
2. **AI Agent**：填充证明细节、搜索大量候选策略、发现非直觉的构造方法
3. **Lean 编译器**：确保每一步推理的正确性

三者各司其职，相互补充。AI 不是替代数学家，而是**极大地扩展了数学家的搜索能力**。

### 5.3 成本革命

每个 Erdős 问题的解决成本约 **$60 USD**。这与人类数学家投入数月全职工作形成鲜明对比。虽然 AI 的解决范围仍然有限（9/353），但这种成本效率暗示了未来数学研究的根本性变革。

### 5.4 "反例"同样有价值

在 Green 问题 #57 的案例中，AI 找到了一个**反例**而非证明——这说明 AI 不仅限于构造性证明，还能帮助数学家**否定错误的猜想**，从而重新校准研究方向。

### 5.5 局限性

1. **依赖形式化前置工作**：每个问题需要先人工形式化为 Lean，这是非平凡的工程任务
2. **组合数学偏重**：9 个解决的 Erdős 问题全部属于组合数学领域，在分析学、代数拓扑等领域的表现尚未验证
3. **需要前沿 LLM**：小模型（Flash, Flash-Lite）完全失败，说明当前方法依赖最强大的 LLM
4. **AlphaProof 独立使用无效**：单独使用 AlphaProof 的树搜索无法解决任何问题，说明 RL 训练的定理证明器在研究级数学上仍有局限
5. **"misformalization" 风险**：论文发现多篇已发表文献的形式化版本存在错误，这既是贡献（发现错误）也是挑战（形式化本身的准确性）

---

## 六、个人评述

### 6.1 里程碑式的成果

这篇论文标志着 AI 辅助数学研究从"玩具问题"走向"真实数学"的转折点。解决 9 个 Erdős 问题不是在受控基准上的刷分，而是对**开放数学问题**的实质性贡献。这些证明已被 Lean 编译器验证，等同于人类审稿人的逐行审核。

### 6.2 架构洞察的深远影响

"基础 agent 就够用"这个发现对整个 AI for Math 领域有深远影响：
- 它暗示 LLM 的**推理能力**是瓶颈，而非搜索算法的复杂度
- 随着 LLM 的下一代发布（更强的推理、更长的上下文），更简单的框架可能解决更难的问题
- 进化搜索和 RL 组件更像是一种"效率工具"而非"能力开关"

### 6.3 形式化作为 AI 可靠性的"地面真相"

Lean 编译器提供的不仅是验证——它是 AI 系统的**自动评判标准**。在数学之外，这种"可执行的正确性标准"理念可以推广到：
- 软件验证（AI 生成的代码必须通过测试）
- 法律推理（AI 的论证必须符合法规形式化）
- 科学假说验证（AI 的预测必须通过实验检查）

### 6.4 推荐阅读顺序

1. **摘要 + 第 1 节**：理解整体框架和核心成果
2. **第 2 节（Erdős 问题）**：最令人兴奋的部分——9 个具体问题及证明方法
3. **第 3 节（Agent 架构）**：技术细节，理解系统如何工作
4. **第 5 节（消融实验）**：关键发现——基础 agent 也能成功
5. **附录**：OEIS 证明案例、技术参数、相关工作对比

### 6.5 开放问题

- 如何降低形式化的前置成本？能否自动化将自然语言数学转化为 Lean？
- 在非组合数学领域（如微分方程、代数拓扑），这种方法的有效性如何？
- 随着模型能力提升，是否能解决 Millennium Prize 级别的难题？
- 如何将这种框架与人类数学家的直觉和创意更好地结合？

---

## 七、参考文献

1. Tsoukalas, G., Kovsharov, A., Shirobokov, S., et al. "Advancing Mathematics Research with AI-Driven Formal Proof Search." arXiv:2605.22763, 2026. [Google DeepMind]
2. Novikov, A., et al. "AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms." 2025.
3. Hubert, T., et al. "AlphaProof: System for olympiad-level Lean theorem-proving based on reinforcement learning." 2025.
4. de Moura, L., Ullrich, S. "The Lean 4 Theorem Prover and Programming Language." CADE-28, 2021.
5. Erdős, P. "On the combinatorial problems which I would most like to see solved." Combinatorica, 1981.
