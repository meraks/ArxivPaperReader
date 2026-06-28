# World Models in Pieces: Structural Certification for General Agents

## 论文元数据
- 标题：World Models in Pieces: Structural Certification for General Agents
- 作者：Yikai Lu, Yifei Wu, Xinyu Lu, Tongxin Li
- arXiv ID：2606.24842
- 发表/提交日期：2026-06-23
- 会议：ICML 2026
- 官方代码：无官方实现
- 代码发现方式：web_search — 无匹配仓库

---

## Ch1：论文概述与核心贡献

### 研究背景：大世界中的必然专业化

在"大世界"（big-world）假设下，智能体面临的状态-动作空间远超其有限计算与样本资源。此时智能体无法维持通用能力，其能力必然在世界模型的不同片段上专业化分布。这一事实使得现有标准统一保证（uniform guarantees）失效 — 它们无法区分智能体对关键瓶颈的理解与对无关错误的处理。

传统强化学习的 worst-case 分析假设智能体在所有场景下表现一致，但这在大世界中既不现实也无意义。当智能体必然在某些子领域表现优异、在其他领域失败时，统一的性能指标掩盖了其专业化的知识结构。

### 核心问题：世界模型的局部认证

给定一个在特定目标集上表现良好的一般智能体（general agent），如何保证其内部世界模型在相关迁移上的准确性？这里"迁移"（transition）指状态-动作-下一状态三元组 $(s,a,s')$，世界模型的核心是迁移概率 $P_{ss'}(a)$。

关键挑战在于：智能体的策略性能仅提供间接约束，从行为反推世界模型是欠定的。需要找到方法，将智能体在组合目标上的表现映射到特定迁移的概率估计。

### 四个核心贡献

**贡献1：不可能性定理（Proposition 3.1）**

证明通用智能体（universal agent）不存在 — 对于任何非最优确定性马氏策略 $π$，存在至少一个目标 $ψ_{fail}$ 使得智能体的成功概率严格低于最优策略的 (1-δ) 倍。形式化陈述：

$$P(\tau \models \psi_{fail} | \pi, s_0) < (1-\delta) \max_\pi P(\tau \models \psi_{fail} | \pi, s_0)$$

这意味着 worst-case 保证下的"通用性能"无法定义，因为对于任意 δ ∈ (0,1)，都存在某些目标使得智能体无法达到 (1-δ)-近优。

**贡献2：结构认证框架（Theorem 3.2）**

提出过渡局部化（transition-local）认证方法。核心思想：通过精心设计的复合目标 $\psi_n(s,a,s') \in \Psi_n$，将智能体的性能保证迁移到特定 $(s,a,s')$ 的概率估计。对每个认证迁移，策略 $\pi$ 唯一确定估计概率 $\hat{P}_{ss'}(a)$，误差界为：

$$|\hat{P}_{ss'}(a) - P_{ss'}(a)| \leq \frac{1}{2(n+1)}\cdot\frac{1}{1-2\delta} + \frac{\delta}{1-2\delta}\cdot P_{ss'}(a)(1-P_{ss'}(a))$$

当 δ ≪ 1 时，该界缩放为 O(1/n) + O(δ)。这是首个将目标条件性能映射到世界模型 entry-wise 保证的理论框架。

**贡献3：构造性算法（Algorithm 1-3）**

提供具体算法实现结构认证。核心机制：通过二分搜索式过程，利用智能体在探针目标（probe goals）上的动作选择来揭示迁移概率。算法构造特定形式的复合目标 $\psi_{a,b}(r,n)$，使得成功概率仅依赖于单条迁移 $P_{ss'}(a)$ 是否超过阈值 $r$。智能体的选择（动作 $a$ 或 $b$）映射到关于 $P_{ss'}(a)$ 的区间约束，取中点得到估计 $\hat{P} = (k+0.5)/(n+1)$。

**贡献4：基本下界（Theorem 3.3）**

证明不可理性化（irrationalizability）结果：如果策略在真实模型下对某目标集不是 (γ,n)-限界的（γ > δ），那么任何使策略看似 (δ,n)-限界的替代世界模型必须至少在一条迁移上与真实模型相差超过：

$$\frac{(\gamma-\delta) \cdot \max_\pi P(\tau \models \psi | \pi, s_0)}{(2-\delta)n}$$

这意味着不能通过任意构造世界模型来"合理化"失败 — 未认证区域的动力学偏差存在基本下界。

### 与现有方法的区别

**与PAC-RL的区别：**

PAC-MDP 理论关注样本复杂度与策略收敛性，假设智能体最终学习到精确模型。结构认证不要求学习收敛，而是认证当前智能体在有限样本下已掌握的模型片段。PAC-RL 提供关于"最终学习"的保证，结构认证提供关于"当前知识"的保证。

**与World Model Verification的区别：**

现有世界模型验证工作侧重于测试时检测模型漂移或分布偏移，通常需要额外查询或监控数据。结构认证是框架内验证（in-framework certification）— 仅使用智能体在目标上的行为，无需外部oracle。此外，结构认证提供定量误差界，而非二元通过/失败结果。

### 关键结果速览

主要理论结果 Theorem 3.2 的误差界在 δ ≪ 1 时为 O(1/n) + O(δ)：

- O(1/n) 项来自目标深度 n — 更深的目标提供更精细的概率分辨能力
- O(δ) 项来自性能保证的松弛 — 允许 δ-近优行为引入线性误差
- 当 δ → 0，界趋近于 1/[2(n+1)]，达到信息论极限（均匀分布最大方差）

该界在小 δ 区域被证明是紧的（tight）— 存在场景使得实际误差达到上界。实验验证了误差随 n 增加递减的 1/n 趋势。

---

## Ch2：研究背景与问题形式化

### 智能体验证的演进

强化学习的可信性问题长期关注策略性能验证。经典方法包括：

- **离线评估（Off-line Evaluation）**：在固定数据集上评估策略，面临分布偏移（distribution shift）问题
- **在线监控（Online Monitoring）**：部署后检测性能下降，需要额外监控成本
- **形式化验证（Formal Verification）**：对简化模型提供数学保证，难以扩展到大规模系统

这些方法的共同假设是：性能保证可以直接映射到知识质量。但当世界模型本身作为内部表示时，这一假设需要更细致的审视。

**World Model Verification 的兴起：**

模型驱动强化学习（World Models [Ha & Schmidhuber, 2018]、Dreamer [Hafner et al., 2020]）将环境建模与策略优化分离。这引入新问题：如何验证世界模型的准确性？现有工作分为：

1. **任务无关验证**：测试预测误差（如 MSE），但高误差区域未必影响任务性能
2. **任务相关验证**：评估下游任务表现，但难以定位具体错误迁移

结构认证属于第三类：通过任务定义的"迁移特定目标集"来认证相关迁移，提供定位（localization）与定量（quantification）保证。

### 标准统一保证的局限性

Worst-case 保证假设：对所有目标 $\psi \in \Psi_n$，策略 $\pi$ 满足 $P(\tau \models \psi | \pi, s_0) \geq (1-\delta) \max_\pi P(\tau \models \psi | \pi, s_0)$。这要求智能体是"通用"的。

在大世界中，这一假设过于严格：
- 状态空间规模 |S| 与样本量呈指数关系，有限样本无法覆盖所有迁移
- 智能体必然在某些子领域表现优异，在其他领域失败
- 统一的 δ 无法区分"理解瓶颈但忽略细节"与"完全不理解"

更根本的是，Proposition 3.1 将证明这种"通用智能体"不可能存在 — 任何非最优策略都存在至少一个目标使其失败率超过 δ。

因此，需要迁移到"迁移特定"的认证框架：对智能体实际使用的迁移子集提供保证，而非全部可能迁移。

### 形式化定义

**环境：受控马尔可夫过程（Controlled Markov Process, cMP）**

cMP 定义为三元组 $\mathcal{M} = \{\boldsymbol{S, A}, P_{ss'}(a)\}$：

- $\boldsymbol{S}$：有限状态集
- $\boldsymbol{A}$：有限动作集（$|\boldsymbol{A}| \geq 2$）
- $P_{ss'}(a)$：转移概率函数 $P_{ss'}(a) = P(s' | s, a)$

注：论文使用 cMP 而非 MDP，因不假设即时奖励（reward），目标由时序逻辑公式指定。

**目标表达：线性时序逻辑（LTL）**

使用 LTL 的子集表达目标，包含算子：

- ⊤（Now）：当前命题成立
- ◯（Next）：下一时刻命题成立
- ◇（Eventually）：未来某时刻命题成立

基本公式示例：◇⊤ 表示"最终到达目标状态"（reachability）。

**复合目标与深度**

复合目标定义为序列 $ψ = ⟨φ₁, φ₂, ..., φ_n⟩$，其中每个 $φ_i$ 是基本公式。深度 n 定义为序列长度。

迁移特定目标集 $\Psi_n(s,a,s')$ 包含所有深度为 $n$、其成功条件依赖迁移 $(s,a,s')$ 的复合目标。核心性质：在这些目标上，智能体的性能揭示了其对特定迁移的掌握程度。

**通用智能体 vs 一般智能体**

- **通用智能体（Universal Agent）**：对所有 $\psi \in \Psi_n$ 满足 $(1-\delta)$-近优性能
- **一般智能体（General Agent）**：通过迁移特定目标集 $\Psi_n(s,a,s')$ 定义，仅在其认证的迁移上提供保证

Proposition 3.1 将证明第一种不存在，因此所有实际智能体都是第二种。

**(δ,n)-限界性能**

策略 $\pi$ 在目标集 $\Psi_n$ 上是 $(\delta,n)$-限界的，如果对每个 $\psi \in \Psi_n$：

$$P(\tau \models \psi | \pi, s_0) \geq (1-\delta) \max_{\pi} P(\tau \models \psi | \pi, s_0)$$

这是核心假设：智能体在其专业领域内接近最优，但松弛程度 δ 允许次优行为。

**迁移特定目标集**

对迁移 $(s,a,s')$，定义 $\Psi_n(s,a,s')$ 为所有深度 $n$ 的目标，其满足条件需在轨迹中经过该迁移。结构认证的关键：通过构造探针目标 $\psi^* \in \Psi_n(s,a,s')$，使得成功概率仅依赖于 $P_{ss'}(a)$ 是否超过某阈值，从而反推迁移概率。

---

## Ch3：核心理论结果

### Proposition 3.1：不可能性定理

**陈述：**

任何非最优确定性马氏策略 $\pi$，对任意 $\delta \in (0,1)$，存在目标 $\psi_{\text{fail}}$ 使得：

$$P(\tau \models \psi_{fail} | \pi, s_0) < (1-\delta) \max_\pi P(\tau \models \psi_{fail} | \pi, s_0)$$

**证明思路：**

1. 因 $\pi$ 非最优，存在可达目标 $\psi$ 使得 $\pi$ 在其上有非零失败率 $\gamma$
2. 当 $\delta \leq \gamma$ 时，直接取 $\psi_{\text{fail}} = \psi$ 即可
3. 当 $\delta > \gamma$ 时，构造深度 $N$ 的序列目标 $\psi_{\text{fail}}$：要求重复完成 $\psi$ 共 $N$ 次，每次间返回初始状态
4. 在确定性策略下，每次重复贡献乘法失败概率，失败率比值缩小到 $(1-\gamma)^N$，选择足够大的 $N$ 使得 $(1-\gamma)^N < 1-\delta$

**含义：**

- 通用智能体不存在 — 无法在所有目标上维持均匀失败率 δ
- Worst-case 保证无意义 — 对于任意非平凡智能体，都存在某些目标使其表现显著劣于最优
- 必须迁移到"迁移特定"框架：认证智能体实际使用的迁移子集，而非全部可能迁移

这一不可能性结果是结构认证框架的动机：既然无法保证通用性，应转向局部认证。

### Theorem 3.2：结构认证

**陈述：**

给定策略 $\pi$ 在 $\Psi_n(s,a,s')$ 上是 $(\delta,n)$-限界的，对每个认证迁移 $(s,a,s')$，存在唯一估计 $\hat{P}_{ss'}(a)$ 满足误差界：

$$|\hat{P}_{ss'}(a) - P_{ss'}(a)| \leq \frac{1}{2(n+1)}\cdot\frac{1}{1-2\delta} + \frac{\delta}{1-2\delta}\cdot P_{ss'}(a)(1-P_{ss'}(a))$$

**构造机制：**

1. **探针目标设计**：构造 $\psi_{a,b}(r,n) \in \Psi_n(s,a,s')$，其成功概率关于 $P_{ss'}(a)$ 单调
2. **二分搜索**：通过调整阈值 $r$，找到 $\pi$ 的动作切换点 $k$（即 $P_{ss'}(a) \in [k/(n+1), (k+1)/(n+1)]$）
3. **估计取值**：$\hat{P}_{ss'}(a) = (k+0.5)/(n+1)$，误差最大为区间半宽

**误差分析：**

第一项 $1/[2(n+1)]·1/(1-2δ)$：
- 来自离散化误差（均匀分布最大方差）
- 随深度 n 增加而递减 O(1/n)
- δ 接近 0.5 时发散（要求 δ < 0.5 以保证有意义）

第二项 $\delta/(1-2\delta) \cdot P_{ss'}(a)(1-P_{ss'}(a))$：
- 来自允许 $\delta$-近优行为
- 概率接近 0 或 1 时项减小（利用边界约束）
- 中间概率 $P=0.5$ 时达到最大 $\delta/[4(1-2\delta)]$

**缩放行为：**

当 δ ≪ 1：

- 界缩放为 O(1/n) + O(δ)
- 深度增加提供 1/n 精度提升
- 性能松弛 δ 引入线性误差项

**紧致性（Tightness）：**

论文证明该界在小 $\delta$ 区域是紧的 — 存在策略 $\pi$ 与真实概率 $P_{ss'}(a)$ 使得实际误差达到上界。具体构造：

- 边界情况：$P_{ss'}(a)$ 接近 0 或 1/2 时达到界
- 策略在探针目标上恰好在切换点改变动作
- 估计误差恰为区间半宽

紧致性确保理论界不可改进（在 δ ≪ 1 区域）。

### Theorem 3.3：下界与不可理性化

**陈述：**

如果策略 $\pi$ 在真实模型下对目标集 $\Psi$ 不是 $(\gamma,n)$-限界的（$\gamma > \delta$），那么任何使 $\pi$ 看似 $(\delta,n)$-限界的替代世界模型 $\widehat{\mathcal{M}}$ 必须满足：

$$\left|P_{\widehat{\mathcal{M}}ss'}(a) - P_{ss'}(a)\right| > \frac{(\gamma-\delta) \cdot \max_\pi P(\tau \models \psi | \pi, s_0)}{(2-\delta)n}$$

**证明思路：**

1. 假设所有迁移误差 $\leq \varepsilon$，反推策略性能上界
2. 如果 $\pi$ 在真实模型下失败率 $> \gamma$，但在 $\widehat{\mathcal{M}}$ 下失败率 $\leq \delta$，则矛盾
3. 导出 $\varepsilon$ 的下界 — 至少一条迁移误差必须超过该值

**含义：**

1. **不可理性化**：不能通过任意构造世界模型来"解释"失败
   - 如果智能体在某目标上系统性失败（$\gamma > \delta$）
   - 任何使其看似合格的世界模型必须在动力学上有非平凡偏差
   - 偏差下界与 $(\gamma-\delta)$ 成正比（性能差距越大，所需偏差越大）

2. **认证与未认证区域的分界**：
   - $\Psi_n(s,a,s')$ 内：通过 Theorem 3.2 认证，误差 $\mathcal{O}(1/n) + \mathcal{O}(\delta)$
   - $\Psi_n(s,a,s')$ 外：动力学偏差至少有 $(\gamma-\delta)$ 量级的基本下界
   - 这界定了"已知迁移"与"未知迁移"的明确边界

3. **与 Theorem 3.2 的互补**：
   - Theorem 3.2：已知迁移的精度上界
   - Theorem 3.3：未知迁移的误差下界
   - 合起来完整刻画世界模型的知识-无知边界

### 三个定理的关系

1. **Proposition 3.1**：否定"通用智能体"存在 → 迁移到局部认证
2. **Theorem 3.2**：提供局部认证框架 → 给出已知迁移的精度保证
3. **Theorem 3.3**：界定未知迁移的误差下界 → 防止任意"合理化"

三者构成完整理论链：不可能性 → 局部认证 → 边界界定。这是首个在大世界假设下，将智能体目标条件性能映射到世界模型 entry-wise 保证的严格理论。

### 实验验证（预告）

论文第4章实验在 20 状态网格世界中验证 Theorem 3.2 的预测：

- 目标深度 $n$ 从 10 到 100 变化
- 结构认证的 MAE 随 $n$ 增加递减，符合 $1/n$ 趋势
- 对比基线：Richens et al. (2025) 界（$O(1/\sqrt n)$）、实证平均误差（无结构性知识）
- δ 设置测试：不同 $δ$ 下误差变化符合 $O(δ)$ 预测

（详细实验分析将在后续任务中展开）

---
## Ch4: 核心算法

## Algorithm 1: 确定迁移的认证

### 输入与输出

**输入：**
- 目标深度 $n$
- 待认证迁移三元组 $(s, a, s')$
- 动作 $a, b$（用于构造探针目标）
- 切换点参数 $r$

**输出：**
- 迁移概率估计值 $\hat{P}_{ss'}(a)$
- 切换点位置 k

### 核心数学原理

Algorithm 1 解决的问题是：如何通过智能体在复合目标上的行为选择，推断出特定迁移 $(s, a, s')$ 的转移概率 $P_{ss'}(a)$。

算法的关键在于构造**复合目标族 $\psi_{a,b}(r, n)$**，其满足以下性质：
- 当智能体选择动作 $a$ 时，目标成功概率与 $P_{ss'}(a)$ 单调相关
- 当智能体选择动作 $b$ 时，目标成功概率与 $(1 - P_{ss'}(a))$ 单调相关
- 通过调整参数 $r$，可以控制智能体在 $a$ 与 $b$ 之间的切换点

### 算法步骤

**步骤 1：构造目标族**
对给定的迁移 $(s, a, s')$ 和参考动作 $b$，构造参数化目标族：
$$\psi_{a,b}(r, n) = \langle \underbrace{\phi_a, \phi_a, \ldots, \phi_a}_{r \text{ 次}}, \underbrace{\phi_b, \phi_b, \ldots, \phi_b}_{n-r \text{ 次}} \rangle$$

其中 $\phi_a$ 表示"在状态 $s$ 执行动作 $a$"，$\phi_b$ 表示"在状态 $s$ 执行动作 $b$"。

**步骤 2：线性扫描寻找切换点**
在 $r = 0$ 到 $n-1$ 范围内依次查询：

```
for r = 0 to n-1:
    # 向智能体查询复合目标 ψ_{a,b}(r, n) ∨ ψ_{a,b}(r+1, n)
    a_r ← argmax_{x∈{a,b}} π(x | s; ψ_a(r,n) ∨ ψ_b(r+1,n))
    # 记录 a_r 的值，更新 p_min 或 p_max
```

**步骤 3：计算概率估计**
切换点 k 揭示了以下信息：
- 当 r ≤ k 时，智能体选择动作 a
- 当 r > k 时，智能体选择动作 b

根据期望效用最大化原理，这意味：
$$P_{ss'}(a) \in \left[\frac{k}{n+1}, \frac{k+1}{n+1}\right]$$

取中点作为保守估计：
$$\hat{P}_{ss'}(a) = \frac{k + 0.5}{n + 1}$$

### 误差分析

该估计的最大误差为：
$$\left|\hat{P}_{ss'}(a) - P_{ss'}(a)\right| \leq \frac{1}{2(n+1)} \cdot \frac{1}{1-2\delta} + \frac{\delta}{1-2\delta} \cdot P_{ss'}(a)(1-P_{ss'}(a))$$

这正是 Theorem 3.2 中 O(1/n) 项的来源。

---

## Algorithm 2 (附录F): 非平凡概率迁移的认证

### 适用场景

Algorithm 2 处理迁移概率满足 $P_{ss'}(a) \in (0, 1)$ 的**非平凡情况**。此时智能体在两个动作之间存在真实的偏好权衡。

### 目标构造方法

构造两个互补目标：
1. **主目标** ψ_a(r, n)：在前 r 步强调动作 a
2. **对照目标** ψ_b(r+1, n)：在前 r+1 步强调动作 b

### 认证逻辑

通过比较智能体在 $\psi_a(r, n)$ 和 $\psi_b(r+1, n)$ 上的选择，可以缩小 $P_{ss'}(a)$ 的置信区间：

- 若智能体在 $\psi_a(r, n)$ 上选择 $a$ 且在 $\psi_b(r+1, n)$ 上选择 $b$，说明 $P_{ss'}(a)$ 位于某个特定区间
- 通过调整 r 参数，可以逐步收紧区间边界

该算法可以处理更复杂的迁移场景，特别是当智能体的策略对多个迁移概率敏感时。

---

## Algorithm 3 (附录F): 平凡概率迁移的认证

### 适用场景

Algorithm 3 处理**边界情况**：
- $P_{ss'}(a) = 0$（迁移不可能发生）
- $P_{ss'}(a) = 1$（迁移确定发生）

### 目标构造策略

对于平凡概率迁移，算法构造特殊的单目标探针：
- 当 $P_{ss'}(a) = 0$ 时，构造验证"智能体避免动作 $a$"的目标
- 当 $P_{ss'}(a) = 1$ 时，构造验证"智能体偏好动作 $a$"的目标

### 认证优势

对于这些极端情况，算法可以在较小的目标深度 n 下完成认证，因为智能体的行为选择在确定性迁移下更加稳定。

---

## 核心思想总结：隔离 → 探测 → 估计

三个算法共同遵循一个三层方法论：

### 1. 隔离 (Isolate)

通过**深度组合目标**将特定迁移 (s, a, s') 从复杂的动力学中隔离出来：
- 构造目标使得成功概率仅依赖于单条迁移的概率
- 其他迁移通过目标结构被控制或消除影响

数学表达：设计 $\psi$ 使得
$$P(\tau \models \psi | \pi) = f(P_{ss'}(a)) + \mathcal{O}(\delta)$$
其中 $f$ 是单调函数，$\delta$ 是智能体失败率上界。

### 2. 探测 (Probe)

利用智能体的**动作选择**作为探针：
- 智能体在 $\psi_{a,b}(r, n)$ 上的选择反映了其内部对 $P_{ss'}(a)$ 的估计
- 不同的 r 值探测不同的概率阈值

关键性质：如果智能体在 $\psi_{a,b}(r, n)$ 上选择 $a$ 而非 $b$，则
$$P_{ss'}(a) > \frac{r}{n+1}$$

### 3. 估计 (Estimate)

通过**二分搜索**找到切换点并计算估计值：
- 切换点 $k$ 给出了 $P_{ss'}(a)$ 的置信区间 $[k/(n+1), (k+1)/(n+1)]$
- 取中点 (k+0.5)/(n+1) 作为最大最小（minimax）最优估计

### 精度控制机制

目标深度 n 直接控制估计精度：
- n 越大，区间宽度 1/(n+1) 越小，估计越精确
- 计算复杂度为 $\mathcal{O}(n)$：执行 $n$ 次迭代，每次迭代 $\mathcal{O}(1)$ 工作量

这种精度-成本的显式权衡是结构认证框架的实用优势。

---

## Ch5: 实验验证与分析

## 实验设置

### 环境配置

**网格世界参数：**
- 状态空间大小：20 个状态
- 动作空间大小：5 个动作（{Up, Down, Left, Right, Stay}）
- 拓扑结构：随机生成的网格连接图

**环境类型：** 受控马尔可夫过程 (Controlled Markov Process, cMP)

### 智能体训练

**训练方法：**
- 算法：有限样本随机游走 (Finite-sample random walk)
- 样本量：N_samples = 4000 条轨迹
- 策略类型：非最优确定性马尔可夫策略
- 探索机制：均匀随机动作选择

**不可访问迁移处理：**
- 初始化方式：均匀先验 (Uniform prior)
- 先验概率：1 / (可达状态数)

### 评估方案

**对比基线：**
1. **结构认证** (Structural Certification, O(1/n))：本文提出的 Theorem 3.2 误差界
2. **Richens et al. (2025) 基线界** (Baseline Bound, O(1/√n))：文献中通用智能体的原始误差界
3. **实证平均误差** (Empirical Mean Error)：对认证迁移实际计算 $|\hat{P} - P|$ 的平均值

**评估指标：**
- 主要指标：平均绝对误差 (Mean Absolute Error, MAE)
- 计算方式：$\text{MAE} = (1/|T|) \sum_{(s,a,s') \in T} |\hat{P}_{ss'}(a) - P^*_{ss'}(a)|$
- 其中 $T$ 为测试迁移集，$P^*$ 为真实迁移概率（由环境生成器确定）

---

## 主要实验结果

### 估计精度随目标深度的变化

实验验证了理论预测的核心关系：**估计误差随目标深度 n 增加而递减，遵循 O(1/n) 缩放**。

不同目标深度下的 MAE 表现：

| 目标深度 n | 结构认证 MAE | Richens 基线 MAE | 实证平均 MAE |
|-----------|-------------|-------------|-------------|
| n = 5     | 基准水平    | 较高水平    | 最高水平    |
| n = 10    | 显著降低    | 中等降低    | 无变化      |
| n = 20    | 进一步降低   | 轻微降低    | 无变化      |
| n = 50    | 接近理论下界 | 轻微降低    | 无变化      |

> **注：** 论文未提供具体数值，以上表格基于定性描述。精确数值需参考论文原始图表。

### 核心发现

**1. 结构认证的样本效率优势**
- 在相同样本量 (N_samples = 4000) 下，结构认证比 Richens 基线界达到更低误差
- 优势来源：结构认证利用智能体策略的结构知识，而非单纯依赖样本统计

**2. O(1/n) 理论预测的验证**
- 误差下降曲线与 1/n 理论曲线高度拟合
- 验证了 Theorem 3.2 中的误差界：O(1/n) + O(δ)

**3. 小样本区域的性能**
- 当 n 较小时（n < 10），结构认证的相对优势更明显
- 说明该方法在数据稀缺场景下尤其有价值

**4. 实证平均误差的局限性**\n- 实证平均 MAE 不随 n 改善（无结构性知识）
- 在复杂环境中，无结构性假设与真实动力学偏差显著

---

## 实验讨论

### 方法论意义

该实验首次在**部分可观测场景**下验证了结构认证的实用性：
- 智能体仅通过 4000 样本训练，未完全掌握环境动力学
- 结构认证仍能从智能体的策略中提取可靠的世界模型片段

### 与理论的一致性

实验结果与理论预测在三个关键方面一致：

1. **误差缩放律：** 观察 MAE ∝ 1/n，与 Theorem 3.2 的 O(1/n) 项吻合
2. **δ 参数影响：** 当智能体失败率 δ 降低时（更优策略），估计误差相应减小
3. **认证范围：** 算法成功认证了训练过程中高频访问的迁移，对不可访问迁移回退到先验

### 实际部署的启示

**优势：**
- 对训练数据量的要求低于纯统计方法
- 可迁移性：同一框架适用于不同环境的智能体认证

**限制：**
- 需要构造具体的探针目标（实验中手工设计）
- 状态空间规模受限（20 状态），可扩展性需进一步验证

---

## Ch6:概念代码实现

**⚠️ 免责声明：以下代码为非官方概念实现，基于论文 Algorithm 1-3 的描述编写，未经验证，不可直接用于生产环境。**

---

## 核心算法实现

```python
# ⚠️ 非官方概念实现 — 基于论文 Algorithm 1-3 的描述编写
# 目的：帮助理解算法流程，不可直接用于生产

import numpy as np
from typing import Tuple, Callable, Dict, Optional
from dataclasses import dataclass


@dataclass
class Transition:
    """迁移三元组"""
    s: int      # 源状态
    a: int      # 动作
    s_prime: int  # 目标状态


@dataclass
class Goal:
    """复合目标 ψ = ⟨φ₁, φ₂, ...⟩"""
    depth: int
    action_sequence: Dict[int, int]  # timestep -> action


class WorldModelCertifier:
    """
    结构认证的概念实现
    基于论文 Algorithm 1（确定迁移）和 Algorithm 2/3（非确定/平凡迁移）
    """
    
    def __init__(self, num_states: int, num_actions: int):
        self.num_states = num_states
        self.num_actions = num_actions
        self.certified_transitions: Dict[Tuple[int, int, int], float] = {}
    
    def certify_transition(
        self,
        agent_policy: Callable[[Goal], int],
        transition: Transition,
        goal_depth: int,
        reference_action: Optional[int] = None
    ) -> Tuple[float, int]:
        """
        Algorithm 1: 认证一条迁移 (s, a, s') 的转移概率
        
        参数：
            agent_policy: 智能体策略函数 π(ψ)，返回动作选择
            transition: 待认证的迁移 (s, a, s')
            goal_depth: 目标深度 n
            reference_action: 参考动作 b（若为 None，自动选择）
        
        返回：
            ($\hat{P}_{ss'}(a)$, k): 估计概率和切换点
        """
        s, a, s_prime = transition.s, transition.a, transition.s_prime
        
        # 自动选择参考动作（选择与 a 不同的动作）
        if reference_action is None:
            reference_action = self._select_reference_action(a)
        
        b = reference_action
        
        # Algorithm 1: 线性扫描寻找切换点
        p_min, p_max = 0.0, 1.0
        r_star = None
        
        for r in range(goal_depth):
            # 构造复合目标 ψ_a(r, n) ∨ ψ_b(r+1, n)
            goal_a = self._construct_compositional_goal(
                transition, r, goal_depth, a, b
            )
            goal_b = self._construct_compositional_goal(
                transition, r + 1, goal_depth, b, a
            )
            # 测试智能体在该目标下的动作选择
            action = agent_policy(goal_a)  # 简化：实际应查询 ψ_a(r,n) ∨ ψ_b(r+1,n)
            
            if action == a:  # 智能体选择动作 a
                p_max = min(p_max, (r + 1) / (goal_depth + 1 - 0.1 * (goal_depth - r)))
                if r_star is None:
                    r_star = r
            else:  # 智能体选择动作 b
                p_min = max(p_min, (r + 1) * (1 - 0.1) / (goal_depth + 1 - (r + 1) * 0.1))
        
        if r_star is not None:
            p_hat = (r_star + 0.5) / (goal_depth + 1)
        else:
            p_hat = 0.5
        
        # 缓存结果
        self.certified_transitions[(s, a, s_prime)] = p_hat
        
        return p_hat, k
    
    def _select_reference_action(self, primary_action: int) -> int:
        """选择与主动作不同的参考动作"""
        available = [a for a in range(self.num_actions) if a != primary_action]
        return available[0] if available else primary_action
    
    def _construct_compositional_goal(
        self,
        transition: Transition,
        r: int,
        n: int,
        action_a: int,
        action_b: int
    ) -> Goal:
        """
        构造复合目标 ψ_{a,b}(r, n)
        
        目标结构：前 r 步要求动作 a，后 n-r 步要求动作 b
        """
        action_sequence = {}
        for t in range(r):
            action_sequence[t] = action_a
        for t in range(r, n):
            action_sequence[t] = action_b
        
        return Goal(depth=n, action_sequence=action_sequence)
    
    def certify_nontrivial_transition(
        self,
        agent_policy: Callable[[Goal], int],
        transition: Transition,
        goal_depth: int,
        action_a: int,
        action_b: int
    ) -> Tuple[float, Tuple[int, int]]:
        """
        Algorithm 2 (附录F): 处理非平凡概率迁移 $P_{ss'}(a) \in (0,1)$
        
        构造 ψ_a(r, n) 和 ψ_b(r+1, n) 两个目标进行交叉验证
        """
        # 构造主目标
        goal_a = self._construct_compositional_goal(
            transition, goal_depth // 2, goal_depth, action_a, action_b
        )
        
        # 构造对照目标
        goal_b = self._construct_compositional_goal(
            transition, goal_depth // 2 + 1, goal_depth, action_b, action_a
        )
        
        # 交叉验证
        action_on_a = agent_policy(goal_a)
        action_on_b = agent_policy(goal_b)
        
        # 基于交叉结果估计概率区间
        if action_on_a == action_a and action_on_b == action_b:
            # 一致性结果：使用标准 Algorithm 1
            return self.certify_transition(
                agent_policy, transition, goal_depth, action_b
            )
        else:
            # 不一致情况：回退到保守估计
            return 0.5, (0, goal_depth)
    
    def certify_trivial_transition(
        self,
        agent_policy: Callable[[Goal], int],
        transition: Transition,
        goal_depth: int
    ) -> Tuple[float, str]:
        """
        Algorithm 3 (附录F): 处理平凡概率迁移 $P_{ss'}(a) \in \{0, 1\}$
        
        使用单目标探针验证确定性迁移
        """
        s, a, s_prime = transition.s, transition.a, transition.s_prime
        
        # 构造验证目标：要求在状态 s 持续执行动作 a
        goal = Goal(
            depth=goal_depth,
            action_sequence={t: a for t in range(goal_depth)}
        )
        
        # 查询智能体
        action = agent_policy(goal)
        
        if action == a:
            # 智能体持续选择 a → P_{ss'}(a) 可能接近 1
            return 1.0, "deterministic_positive"
        else:
            # 智能体避免 a → P_{ss'}(a) 可能接近 0
            return 0.0, "deterministic_negative"


# 基于 Algorithm 1 的估计器核心逻辑
def structural_estimate(
    policy_transition_test: Callable[[int], bool],
    n: int
) -> Tuple[float, int]:
    """
    通过线性扫描找到切换点 k，返回 (p_hat k)
    
    参数：
        policy_transition_test(r): 在目标 ψ_{a,b}(r,n) 下是否选择动作 a
        n: 目标深度
    
    返回：
        (p_hat, k): 估计概率和切换点
    """
    r_star = None
    for r in range(n):
        if policy_transition_test(r):
            r_star = r
            break
    
    if r_star is not None:
        p_hat = (r_star + 0.5) / (n + 1)
    else:
        p_hat = 0.5
    return p_hat, k
```

---

## 代码结构说明

### 类与函数对应关系

| 代码组件 | 论文对应 | 位置 |
|---------|---------|------|
| `WorldModelCertifier.certify_transition` | Algorithm 1: Structural Certification | 正文 |
| `certify_nontrivial_transition` | Algorithm 2: Non-trivial Filter and Recover | 附录F |
| `certify_trivial_transition` | Algorithm 3: Trivial Filter and Recover | 附录F |
| `structural_estimate` | Algorithm 1 核心逻辑 | 正文 |

### 关键设计决策

**1. 目标表示方式**
- 使用 `Goal` 类表示复合目标 ψ
- `action_sequence` 字典编码每个时间步要求的动作
- 简化了论文中 LTL 目标的完整语法

**2. 智能体接口抽象**
- `agent_policy` 函数：`Goal → int`
- 隐藏了智能体内部实现（可能是神经网络、符号模型等）
- 与论文的**黑盒认证**理念一致

**3. 误差处理策略**
- Algorithm 2 的不一致情况回退到保守估计 (0.5)
- 实际部署中需要更复杂的冲突解决机制

---

## 理论与实践的差距

### 概念实现未覆盖的方面

**1. 目标构造的完备性**
- 论文要求构造满足特定性质的 $\psi$（成功概率仅依赖 $P_{ss'}(a)$）
- 概念代码中的简化构造可能不满足理论要求

**2. 失败率 δ 的估计**
- Theorem 3.2 的误差界包含 O(δ) 项
- 概念代码未实现 δ 的估计机制

**3. 大规模状态空间的效率**
- 代码假设状态/动作空间较小
- 实际应用需要索引结构和缓存优化

### 扩展方向

**1. 自动化目标生成**
- 当前参考动作选择策略过于简单
- 可扩展为基于历史数据的自适应选择

**2. 批量认证优化**
- 可并行化多个迁移的认证过程
- 利用迁移之间的相关性加速收敛

**3. 与实际智能体的集成**
- 需要适配具体的智能体架构（RL agents, LLM agents 等）
- 目标查询接口需要标准化

---

## Ch7: 局限性与延伸阅读

## 理论局限性

### 1. 有限性与离散性假设

**限制：** 框架假设状态空间 S 和动作空间 A 有限。

**影响：** 对连续状态空间（如机器人控制）或连续动作空间（如精细操作）需要离散化近似，可能引入量化误差。

**缓解方向：**
- 结合函数逼近技术
- 研究连续版本的认证算法

### 2. 探针目标构造的实用性

**限制：** 论文未充分讨论如何自动构造满足理论要求的探针目标。

**挑战：**
- 当前依赖领域知识手工设计目标
- 复杂环境中目标构造可能计算不可行

**研究空白：** "Practical probe goal construction" 是论文明确提出的开放问题。

### 3. 计算复杂度

**限制：** 估计每条迁移需要 O(log n) 次目标查询，总复杂度为 O(|T| log n)，其中 T 为迁移集规模。

**影响：** 对大规模环境（|S| > 10^4），全量认证可能不现实。

**折衷方案：**
- 仅认证关键迁移（基于任务重要性排序）
- 分层认证（先粗粒度，后精细化）

### 4. 智能体策略的马尔可夫性假设

**限制：** 框架要求智能体策略至少是马尔可夫的（依赖历史有限）。

**现实挑战：** 深度强化学习智能体往往具有长程依赖和非马尔可夫特征。

---

## 实际部署挑战

### 1. 真实环境的状态空间规模

**问题：** 实际应用（如自动驾驶、游戏 AI）的状态空间远超论文实验的 20 状态规模。

**潜在方案：**
- 状态抽象与聚类
- 分层世界模型构建
- 稀疏认证（仅关键子空间）

### 2. 黑盒智能体的可观测性

**问题：** 框架假设可以向智能体查询任意目标 ψ 的偏好，但实际系统可能：
- 没有暴露目标查询接口
- 查询成本高昂（需要大规模模拟）

### 3. 非平稳动力学

**问题：** 论文假设环境动力学固定，但真实世界可能：
- 环境随时间变化（如机器人磨损）
- 其他智能体存在（多智能体博弈）

---

## 延伸阅读方向

### 1. PAC 强化学习

**相关领域：** Probably Approximately Correct - Reinforcement Learning

**核心思想：** 在有限样本下提供概率性能保证。

**经典工作：**
- Kearns & Singh (1999): Near-optimal reinforcement learning in polynomial time
- Strehl et al. (2009): PAC-MDP 的理论框架

**与本文联系：** 结构认证可视为 PAC-RL 在世界模型层面的应用。

### 2. MDP 可识别性

**相关领域：** MDP Identifiability / Imitation Learning

**核心问题：** 从观察行为中恢复环境动力学在什么条件下是可能的？

**经典工作：**
- Abbeel & Ng (2004): Apprenticeship learning via inverse RL
- Jinnai et al. (2021): Identification of MDPs with noisy observations

**与本文联系：** Theorem 3.3 提供了可识别性的充分条件。

### 3. AI 可解释性中的形式化验证

**相关领域：** AI Safety / Interpretability / Formal Verification

**核心思想：** 用数学证明确保 AI 系统的行为符合预期。

**相关工作：**
- DeepMind 的世界模型可解释性研究 (World Models 论文系列)
- 结构化解释 的形式化基础

**与本文联系：** 结构认证是一种**可验证的局部解释**（verified local explanation）。

### 4. 分布式智能体的协同认证

**延伸方向：** 如何认证多智能体系统的集体世界模型？

**挑战：**
- 智能体之间的信息不对称
- 协同行为导致的非马尔可夫性

**相关工作：**
- Multi-agent PAC-MDP（PAC-MDP 的多智能体扩展）
- Decentralized structural certification（未探索）

---

## 未来展望

### 结构认证作为 AI 安全工具

**潜在应用：**
1. **关键系统的验证：** 医疗、金融、自动驾驶中的智能体认证
2. **监管合规：** 为 AI 系统提供可审计的性能证明
3. **人机协作：** 让人类理解智能体的知识边界

### 与大语言模型的结合

**新前沿：** 将结构认证应用于基于 LLM 的智能体：
- 状态空间：对话上下文、工具调用历史
- 动作空间：自然语言指令、API 调用
- 挑战：高维、非平稳、语义漂移

### 开放问题

1. **连续系统的认证：** 如何扩展到连续状态/动作空间？
2. **自适应探针设计：** 如何自动化目标构造？
3. **认证迁移集的选择：** 如何确定哪些迁移对任务最关键？
4. **在线认证：** 如何在智能体运行时动态更新认证？

---

**总结：** 结构认证框架为"大世界"场景下的智能体理解提供了一种理论严谨、局部精确的方法。尽管存在实用化挑战，但其在 AI 安全和可解释性方向具有重要价值。