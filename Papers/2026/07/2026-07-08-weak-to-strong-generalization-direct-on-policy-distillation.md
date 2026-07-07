# Weak-to-Strong Generalization via Direct On-Policy Distillation — 深度精读

> **作者**: Shiyuan Feng, Huan-ang Gao, Haohan Chi, Hanlin Wu, Zhilong Zhang, Zheng Jiang, Bingxiang He, Wei-Ying Ma, Ya-Qin Zhang, Hao Zhou  
> **机构**: SIA-Lab of Tsinghua AIR & ByteDance Seed; Tsinghua University; Peking University  
> **arXiv**: [2607.05394](https://arxiv.org/abs/2607.05394) | **代码**: [BytedTsinghua-SIA/Direct-OPD](https://github.com/BytedTsinghua-SIA/Direct-OPD)  
> **项目页**: [bytedtsinghua-sia.github.io/Direct-OPD](https://bytedtsinghua-sia.github.io/Direct-OPD/)

---

## Ch1：论文概述与核心贡献

### 1.1 一句话总结

**Direct-OPD 提出将小模型 RL 训练的「策略偏移」（policy shift）作为稠密隐式奖励传递给更强目标模型，而非蒸馏弱教师的最终策略——在仅需 8×A100 GPU 约 4 小时的转移成本下，将 Qwen3-1.7B 在 AIME 2024 上的准确率从 48.3% 提升到 62.4%，且超越同等 RL 步数下的直接大模型 RL 方案。**

### 1.2 研究动机：RLVR 后训练的成本瓶颈

Reinforcement Learning with Verifiable Rewards (RLVR) 已成为提升 LLM 推理能力的主流后训练范式。然而，其成本与模型规模强相关：每次更新都需要当前策略生成 rollouts、接收验证结果、并在这些轨迹上更新。更大的目标模型意味着：

1. **Rollout 计算成本高**：7B 模型的每步 RL 成本是 1.5B 模型的约 5 倍
2. **RL 训练不稳定**：大模型的探索空间更大，reward 信号更稀疏
3. **重复浪费**：每个新的大模型都需要从头执行 RL 探索

作者的核心洞察是：RL 所发现的有用「改进方向」本身是可以跨模型规模复用的，不需要在每个目标模型上重新发现。

### 1.3 核心问题

> 能否利用小模型上已经完成的 RL 探索，将其中包含的「改进方向」传递给更强的目标模型，使得目标模型无需从零开始 RL 探索？

这个问题的挑战在于：小模型的最终策略（post-RL policy）混合了有用的 RL 增益和小模型自身的能力上限。直接蒸馏（vanilla OPD）该策略会限制学生的上限——当学生已经强于教师时，模仿反而会拖累学生。

### 1.4 五大核心贡献

1. **弱到强后训练新范式**：将小模型 RL 重新定位为「隐性奖励的廉价生成器」，通过检查点对（checkpoint pair）而非最终策略来传递 RL 增益
2. **跨教师-学生对始终一致改善**：在 2 个教师对（JustRL-1.5B、QuestA-Nemotron-1.5B）和 3 个学生（R1-Distill-7B、Qwen3-1.7B、Qwen3-4B）上，Direct-OPD 改善每个学生的 AIME 推理性能，包括初始已高于教师的学生
3. **显著计算效率优势**：在匹配的 RL 步数下，小模型 RL + Direct-OPD 转移优于直接在大模型上运行 RL，且总成本降低约 50%
4. **顺序组合能力**：不同 RL 训练过程学习到的不同能力，可以通过 Direct-OPD 顺序组合到同一个学生模型中
5. **理论分析与条件刻画**：阐明了 Direct-OPD 何时有效——低 token 重叠即可传递、短视界训练泛化到长 rollout、KL 系数对转移信号可靠性的影响

### 1.5 图表总览

| 图表 | 类型 | 内容 |
|------|------|------|
| Figure 1 | 对比图 | Vanilla OPD vs Direct-OPD 的直观对比 |
| Figure 2 | 柱状图 | 两个教师对的 Direct-OPD 转移准确率 |
| Figure 3 | 折线图 | 弱到强 vs 直接 RL 的匹配步数对比 |
| Figure 4 | 折线图 | 顺序组合（JustRL → QuestA）的轨迹 |
| Figure 5 | 图表 | 师生 top-k 重叠分析（§4.1） |
| Figure 6 | 图表 | 熵诊断 |
| Figure 7 | 图表 | 响应长度扫描 |
| Figure 8 | 图表 | 短视界训练泛化（§4.2） |
| Figure 9 | 图表 | KL 系数扫描（§4.3） |
| Table 1 | 结果表 | 不同师生对的分数汇总（Figure 2 对应） |
| 顺序组合表 | 结果表 | 顺序组合分数汇总（论文 Figure 4） |
| Table 2 | 设置表 | 评估协议 |
| Table 3 | 超参表 | Direct-OPD 训练超参数 |
| Table 4 | 超参表 | RL 训练超参数 |

### 1.6 开源/代码/数据状态

- **代码**：[GitHub](https://github.com/BytedTsinghua-SIA/Direct-OPD) — 基于 verl 框架实现
- **模型**：计划上传至 HuggingFace
- **数据**：使用 Skywork-OR1-RL-Data（Math 子集）和 DAPO-Math-17K

---

## Ch2：方法详解

### 2.1 问题形式化

考虑自回归语言模型策略。设 $x \sim \mathcal{D}$ 为 prompt，$y = (y_1, \ldots, y_T)$ 为响应。策略 $\pi$ 分解为：

$$
\log \pi(y|x) = \sum_{t=1}^{T} \log \pi(y_t | s_t), \quad s_t = (x, y_{<t})
$$

我们区分四个策略角色：
- $\pi_\theta$：正在训练的学生策略
- $\pi_S$：学生初始化检查点
- $\pi_{T_{\text{ref}}}$：教师的 pre-RL 参考策略
- $\pi_T$：教师 post-RL 策略

核心设定是：**$\pi_T$ 可能弱于或小于学生 $\pi_\theta$，但从 $\pi_{T_{\text{ref}}}$ 到 $\pi_T$ 的转变仍然编码了值得传递的信息。**

### 2.2 Policy Shift 作为隐式奖励

**KL 正则化 RL 的闭式解。** 给定奖励 $r$、参考策略 $\pi_{\text{ref}}$ 和惩罚系数 $\beta > 0$，KL 正则化 RL 目标为：

$$
\max_\pi \mathbb{E}_{y \sim \pi} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]
$$

该目标的闭式最优解为：

$$
\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)
$$

取对数得到：

$$
\log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} = \frac{1}{\beta} r(x, y) - \log Z(x)
$$

其中 $Z(x)$ 是配分函数。$\log Z(x)$ 对同一 prompt 的所有响应为常数，因此策略-参考 log-ratio 恢复出奖励 $r$（仅差一个正常数缩放和一个 per-prompt 常数）。

**从后验策略读出隐式奖励。** Direct-OPD 利用这一恒等式的反向方向：给定已训练好的 post-RL 教师 $\pi_T$ 及其 pre-RL 参考 $\pi_{T_{\text{ref}}}$，定义教师的策略偏移（policy shift）：

$$
\Delta_T(y|x) := \log \pi_T(y|x) - \log \pi_{T_{\text{ref}}}(y|x) = \frac{1}{\beta_T} r_T(x, y) - \log Z_T(x)
$$

**$\Delta_T$ 即是教师隐式奖励，不带任何额外训练即可获得到学生。** 这就是 Direct-OPD 要传递的核心对象。

### 2.3 Direct-OPD 目标函数

**序列级目标。** 学生从 $\pi_S$ 初始化，优化：

$$
J_{\text{Direct-OPD}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} [\Delta_T(y|x)] - \alpha D_{\text{KL}}(\pi_\theta(\cdot|x) \| \pi_S(\cdot|x)) \right]
$$

其中 $\alpha > 0$ 控制 KL 惩罚强度。此目标本身即为学生上的 KL 正则化 RL 目标，其最优解为：

$$
\pi^*(y|x) \propto \pi_S(y|x) \exp\left(\frac{1}{\alpha} \Delta_T(y|x)\right) = \pi_S(y|x) \left( \frac{\pi_T(y|x)}{\pi_{T_{\text{ref}}}(y|x)} \right)^{1/\alpha}
$$

代入隐式奖励形式得 $\pi^* \propto \pi_S \exp(r_T / \alpha\beta)$。**学生被优化为仿佛在用小教师的隐式奖励运行 KL 正则化 RL，使用自己的初始化 $\pi_S$ 作为参考——无需查询可验证奖励或在目标上运行稀疏奖励 RL。**

**Token 级分解。** 由于两个教师策略在相同前缀上自回归分解，序列级偏移精确分解为 token 级偏移：

$$
\Delta_T(y|x) = \sum_t r_t(y_t | s_t), \quad r_t(v) = \log \pi_T(v|s_t) - \log \pi_{T_{\text{ref}}}(v|s_t)
$$

每个 $r_t(v)$ 是一个稠密的 per-token 奖励：$r_t(v) > 0$ 表示教师在目标 token $v$ 上因 RL 而提升了概率，$r_t(v) < 0$ 表示教师降低了概率。

### 2.4 Top-K 行动空间限制与 Rao-Blackwellized 策略梯度

**Top-K 限制。** 为将信号保持在学生实际考虑的 action 上，在每个访问前缀，限制到学生的 top-k 支持集：

$$
S_t = \text{TopK}_v \pi_\theta(v|s_t), \quad \bar{p}_t(v) = \frac{\pi_\theta(v|s_t)}{\sum_{u \in S_t} \pi_\theta(u|s_t)},\ v \in S_t
$$

**Rao-Blackwellized 梯度。** 单 token 蒙特卡洛估计方差高。由于在每个访问前缀上已有完整的 top-k 奖励和概率，用期望替代单 token 估计：

$$
\nabla_\theta J_{\text{analytical}} = \mathbb{E}_{x, y \sim \pi_\theta} \left[ \sum_t \sum_{v \in S_t} \underbrace{\bar{p}_t(v)}_{\text{权重}} \underbrace{r_t(v)}_{\text{奖励}} \nabla_\theta \underbrace{\log \pi_\theta(v|s_t)}_{\text{对数似然}} \right]
$$

权重中 $\bar{p}_t(v)$ 依赖于 $\theta$，因此使用 stop-gradient 将其分离：

$$
A_t^w(v) = \texttt{stop\_gradient}(\bar{p}_t(v) \cdot r_t(v))
$$

最终局部 top-k 替代目标为：

$$
\nabla_\theta J_{\text{Direct-OPD}} \approx \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ \sum_t \sum_{v \in S_t} A_t^w(v) \nabla_\theta \log \pi_\theta(v|s_t) \right] - \alpha \nabla_\theta D_{\text{KL}}(\pi_\theta(\cdot|x) \| \pi_S(\cdot|x))
$$

### 2.5 自适应 KL 控制

$\Delta_T$ 的幅度由教师 KL 预算 $\beta_T$ 决定（$1/\beta_T \cdot r_T$ 量级），但 $\beta_T$ 对不可从检查点对中恢复。一个固定的 KL 系数 $\alpha$ 无法在所有教师-学生对上通用。

**解决方案**：基于批平均奖励符号的自适应控制器：

$$
\alpha_{m+1} = \text{clip}\left(\alpha_m (1 + \epsilon\ \text{sgn}(\bar{r}_m)), \alpha_{\min}, \alpha_{\max}\right)
$$

其中 $\bar{r}_m$ 是当前 batch 中学生加权偏移 $\bar{p}_t(v) r_t(v)$ 的均值。默认 $\epsilon=0.01,\ [\alpha_{\min}, \alpha_{\max}] = [0.5, 2.5]$。

当平均偏移为正时提高 $\alpha$（防止过度放大局部信号），为负时降低 $\alpha$（使学生能移出教师 RL 降低的 token 区域）。

### 2.6 与标准 OPD 的对比

为了理解 Direct-OPD 的定位，有必要与标准 On-Policy Distillation (OPD) 做系统对比：

| 维度 | 标准 OPD | Direct-OPD |
|------|---------|------------|
| 监督信号 | $\pi_T(y|x)$ — 教师最终策略的 token 分布 | $\Delta_T(y|x) = \log \pi_T - \log \pi_{T_{\text{ref}}}$ — 教师策略偏移 |
| 优化目标 | $\min D_{\text{KL}}(\pi_\theta \| \pi_T)$ | $\max \mathbb{E}[\Delta_T] - \alpha D_{\text{KL}}(\pi_\theta \| \pi_S)$ |
| 学生上限 | 受限于教师 $\pi_T$ | 可超越教师（上限由 $r_T$ 和 $\pi_S$ 共同决定） |
| 所需教师 | 1 个 (post-RL) | 2 个 (pre-RL + post-RL) |
| 跨模式转移 | 需要师生 token 重叠 | 低重叠仍有效 |
| 参考锚点 | 无独立参考 | 学生初始化 $\pi_S$ |
| 等价视图 | 学生匹配教师分布 | 学生执行教师隐式奖励上的 KL 正则化 RL |

**关键区别**：标准 OPD 鼓励学生的分布 $\pi_\theta$ 去匹配 $\pi_T$，这在 $\pi_\theta$ 已经强于 $\pi_T$ 时会产生「回拉」效应。Direct-OPD 则鼓励 $\pi_\theta$ 在 $\pi_S$ 的基础上向 $\pi_T$ 从 $\pi_{T_{\text{ref}}}$ 学到的改进方向移动——当 $\pi_\theta$ 已经处于正确方向时，不会强制其靠近 $\pi_T$ 的绝对分布。

### 2.7 实现细节与调参要点

Direct-OPD 基于 verl 框架实现，实际部署时有以下关键调参要点：

**教师检查点配对。** 需要确保 $\pi_{T_{\text{ref}}}$ 和 $\pi_T$ 使用完全相同的 tokenizer 和 base 架构。使用不同的 tokenizer 会导致 log-ratio 计算不一致。论文中两种教师对均满足此条件。

**Top-K 的选择。** Top-K 值决定了信号携带的 token 数量。$K=16$ 是默认值，但理论上更大的 K 带来更完整的信号但计算开销更高，更小的 K 则可能遗漏重要 token。对于推理任务，学生 top-16 通常覆盖了正确推理路径的关键 token。

**数据格式。** 使用 DAPO 风格的数学 prompt 模板（非 boxed-answer 格式），该格式在实验中优于教师 RL 时使用的格式。这一发现暗示：转移阶段的数据格式可以独立于教师训练阶段进行优化。

**训练长度 vs 推理长度。** Direct-OPD 使用 2k tokens 的响应长度训练，但推理时生成长度可达 31k tokens。短训练长度是刻意设计：教师偏移信号在短前缀上更可靠，且更短的 rollout 降低转移成本。

---

## Ch3：实验验证

### 3.1 主实验：小教师改善强学生

**实验设置。** 使用两个教师对：
- **教师对 1**：R1-Distill-1.5B (pre-RL) → JustRL-1.5B (post-RL)
- **教师对 2**：Nemotron-1.5B (pre-RL) → QuestA-Nemotron-1.5B (post-RL)

学生包括 R1-Distill-7B、Qwen3-1.7B、Qwen3-4B（来自不同模型家族和规模）。

**JustRL 策略转移结果：**

| 模型 | AIME 2024 | 提升 | AIME 2025 | 提升 |
|------|-----------|------|-----------|------|
| Teacher ref (R1-Distill-1.5B) | 28.5% | — | 24.0% | — |
| Teacher RL (JustRL-1.5B) | 51.3% | +22.8 | 37.5% | +13.5 |
| **Qwen3-1.7B** | 48.3% | — | 36.8% | — |
| + Direct-OPD | **62.4%** | **+14.1** | **46.3%** | **+9.5** |
| **Qwen3-4B** | 72.5% | — | 65.6% | — |
| + Direct-OPD | **77.6%** | **+5.1** | **68.8%** | **+3.2** |
| **R1-Distill-7B** | 56.7% | — | 40.5% | — |
| + Direct-OPD | **63.1%** | **+6.4** | **48.8%** | **+8.3** |

**QuestA 策略转移结果：**

| 模型 | AIME 2024 | 提升 | AIME 2025 | 提升 |
|------|-----------|------|-----------|------|
| Teacher ref (Nemotron-1.5B) | 61.77% | — | 49.50% | — |
| Teacher RL (QuestA-Nemotron-1.5B) | 72.50% | +10.73 | 62.29% | +12.79 |
| **Qwen3-1.7B** | 49.1% | — | 36.8% | — |
| + Direct-OPD | **59.0%** | **+9.9** | **43.1%** | **+6.3** |
| **R1-Distill-7B** | 56.3% | — | 39.5% | — |
| + Direct-OPD | **61.2%** | **+4.9** | **44.0%** | **+4.5** |

**关键发现：**
1. **跨上限改善**：Qwen3-4B 初始 72.5% 已高于 JustRL 教师 51.3%，但 Direct-OPD 仍将其提升至 77.6%（+5.1）
2. **跨教师家族泛化**：QuestA 来自不同训练 pipeline 和数据源，效果同样显著
3. **Qwen3-1.7B 增益最大**：从 48.3% 提升到 62.4%，+14.1 个百分点

### 3.2 计算效率对比：弱到强优于直接大模型 RL

**匹配步数对比。** 在相同 RL 步数下比较两条路线：
- **直接 RL 路线**：在 R1-Distill-7B 上直接运行 RL
- **弱到强路线**：先在 R1-Distill-1.5B 上运行 RL，再用 Direct-OPD 转移到 7B

**结果**：对于相同 RL 步数，小模型 RL + Direct-OPD 转移在所有检查点（300/600/900/1200/1500 步）上均优于直接在 7B 上运行 RL。

**计算成本对比：**

| 路线 | RL 成本 | 转移成本 | 总成本 |
|------|---------|---------|--------|
| 直接 7B RL (1500 步) | ~320 h × 32 A100 | — | ~320 h × 32 A100 |
| 1.5B RL (1500 步) + Direct-OPD | ~160 h × 32 A100 | ~4 h × 8 A100 | ~160 h × 32 A100 + 少量 |
| **相对节省** | | | **~50%+** |

**非思考模型验证。** 在 Qwen3-1.7B-nonthinking 上运行 100 步 RL，转移给 Qwen3-4B-nonthinking 后达到与直接 4B RL 相同的 AIME 2024 水平（68.0%），验证了该方法在非思考模型上的泛化性。

### 3.3 顺序组合：多教师策略偏移的累积

**实验**：Qwen3-1.7B 先接受 JustRL 信号训练，然后继续接受 QuestA 信号训练。

| 阶段 | AIME 2024 | AIME 2025 |
|------|-----------|-----------|
| 初始化 | 48.3% | 36.8% |
| 第一段（JustRL） | 62.4% (+14.1) | 46.3% (+9.5) |
| 第二段（QuestA） | **63.8% (+15.5)** | **46.8% (+10.0)** |

**结论**：不同 RL 训练运行可以学到不同能力，Direct-OPD 可将这些能力顺序组合到同一个学生模型中。最终 AIME 2024 63.8% 高于任一单独转移的终点。

### 3.4 实验结果核心讨论

**为什么 Direct-OPD 比标准 OPD 更适合弱到强场景？** 关键在于信号结构。标准 OPD 要求学生 $\pi_\theta$ 的分布逼近教师 $\pi_T$ 的分布——这是一个「绝对值匹配」问题，当学生强于教师时不仅无益反而有害。Direct-OPD 要求学生向 $\pi_T$ 相对于 $\pi_{T_{\text{ref}}}$ 的改进方向移动——这是一个「方向追随」问题，即使学生已经强于教师，方向仍然有意义。

**为什么会弱到强泛化？** 归因于 RL 探索的不对称性。小模型 RL 发现的改进方向（如"多验算一步"、"从已知信息出发"等元策略）往往不是模型规模特定的。这些方向在大模型上同样有效，只是大模型自身需要更多 RL 步数才能发现它们。Direct-OPD 恰好捕获并直接提供了这些方向。

**顺序组合为什么有效？** 因为不同 RL 训练过程（JustRL vs QuestA）可能在不同子能力上发现不同的改进方向。JustRL 可能擅长提升解题步骤的完整性，QuestA 可能擅长减少算术错误。两者在策略空间中表现为不同的方向向量，可以顺序叠加。

---

## Ch4：分析与训练动力学

### 4.1 跨模式转移无需 token 重叠模仿

标准 OPD 的成功与师生在学生访问状态上的 top-k token 重叠密切相关。Direct-OPD 传递的是策略偏移而非最终分布，需要检验是否仍然依赖重叠。

**度量**：师生在状态 $s_t$ 上的 top-k 重叠率：

$$
\text{Overlap}_k(S, C; s_t) = \frac{|T_k^S(s_t) \cap T_k^C(s_t)|}{k}
$$

**发现**：
- 在模式对齐的转移中（师生推理模式相同），学生开始进入与 post-RL 教师的高重叠区间
- 在跨模式转移中（师生推理模式不同），与 post-RL 教师的重叠保持低位，与教师参考也无明显上升
- **Direct-OPD 的增益不能通过逐步模仿任一教师检查点来解释**，而是通过利用 RL 诱导方向在学生自己的访问状态上传递信号

**熵诊断**：验证了 actor 熵没有坍塌——信号来自有意义的策略调整而非退化。

### 4.2 短视界训练泛化到长行为

Direct-OPD 训练使用短响应长度（2k tokens），但推理时生成长度可达 31k tokens。

**问题**：2k 训练是否只改变受监督的短前缀？

**诊断**：在固定长 rollout 上，计算学生行为向 JustRL 方向相对 R1-Distill-1.5B 偏移的累积量：

$$
G_T = \sum_{t=1}^{T} \sum_{a \in K_t} \pi_{\text{actor}}(a|x_{\leq t}) \left[ \log \pi_{\text{JustRL}}(a|x_{\leq t}) - \log \pi_{\text{R1}}(a|x_{\leq t}) \right]
$$

**发现**：40 步的 2k 训练后，actor 在远超 2k 的长 rollout 上仍向教师偏移方向移动。6k 训练移动更大但验证反而更差（45.6 vs 48.8），说明长前缀上的教师偏移可能不可靠。

### 4.3 KL 控制教师偏移奖励的可靠性

教师/参考 log-ratio 作为稠密奖励仅在学生访问的状态上教师偏移有意义的条件下可靠。

**固定 KL 扫描**：不同教师-学生对偏好不同的最优 KL 值。大的正平均奖励可能与更差的验证相关——稠密奖励不应被独立最大化。

**自适应 KL 的效果**：经过初始校正阶段后，自适应 KL 将平均教师偏移奖励拉向零附近。这表示学生停留在教师/参考比较有信息量的区域，而非漂移到两者支持集之外的噪声区域。

### 4.4 训练的稳定性与收敛性分析

Direct-OPD 的收敛行为值得深入分析：

**收敛速度。** 实验显示，仅需 300 训练步（每一步 64 个样本，rollout_n=4）即可达到稳定改善。这与直接 RL（需 1500+ 步在一个数量级更大的模型上）形成鲜明对比。原因在于 Direct-OPD 的监督信号是稠密的 per-token 奖励，而非稀疏的最终答案奖励——每个 token 都携带梯度信息。

**缩放行为。** 更长的弱模型 RL 训练（更多的教师检查点步数）产生更大的 Direct-OPD 增益，但边际递减。T900（900 步小模型 RL）以后的增益增长放缓。这提示最佳策略是让小模型 RL 运行到性能增长的拐点，而非完全收敛。

**KL 预算的非对称性。** 实验中的另一个有趣观察是：$\alpha$ 过小（KL 惩罚弱）和 $\alpha$ 过大（KL 惩罚强）对性能的影响是非对称的。过小的 $\alpha$ 导致学生在不可靠信号上漂移，损失远大于过大的 $\alpha$（只是保守地不跟随教师方向）。这与标准 RL 中的 KL 惩罚行为一致——安全边界的非对称性。

---

## Ch5：相关工作

### 5.1 推理模型的 On-Policy Distillation

OPD 已成为推理模型后训练的主流技术，近期工作涵盖 GKD（ICLR 2024）、Rethinking OPD（Tsinghua THUNLP）、G-OPD（RUC）等。这些工作都在模仿（或推过）教师的最终策略。**Direct-OPD 的区别在于仅蒸馏教师的 RL 诱导策略偏移** $\log \pi_T - \log \pi_{T_{\text{ref}}}$，丢弃教师的绝对策略。

### 5.2 弱到强泛化

从 OpenAI 的弱到强泛化（Burns et al., 2023）出发，近期工作将其扩展到推理和对齐。大多数方法仍使用弱模型的标签或分布来监督学生，导致学生上限受限于弱教师。**Direct-OPD 的监督者是 RL 检查点对**，传递它们的差值所隐含的奖励，并证明这能改善已经超过 post-RL 教师的学生。

### 5.3 隐式奖励与模型复用

DPO 及其变体利用策略-奖励恒等式从偏好中拟合策略。**Direct-OPD 反向使用这一恒等式**——从 post-RL 检查点及其参考中读出稠密隐式奖励——连接了过程奖励和稠密信用分配方法。与其他 weight-space 复用方法（task arithmetic、model merging）不同，Direct-OPD 传递的是行为 log-ratio，跨模型族和规模可转移。

### 5.4 局限与未来方向

论文自身指出的局限性包括：

1. **信号条件性**：当教师/参考改进在学生访问状态上无意义时，Direct-OPD 可能失败
2. **最佳响应长度和 KL 强度仍是教师-学生依赖的**：虽然自适应 KL 缓解了部分问题，但响应长度的选择仍需人工调整
3. **当前实验限于数学推理（AIME）**：需验证在代码生成、科学推理等更广泛任务上的效果

此外，我们还可以补充以下观察：

4. **双教师的要求**：需要保留 pre-RL 检查点，这在已有生产流程中可能不可行。一个实践问题是：如果在 RL 训练过程中没有保留初始检查点，就无法使用 Direct-OPD
5. **教师偏移的质量依赖**：如果小模型的 RL 训练本身没有发现有用的改进方向（如 reward 信号噪声过大），则 Direct-OPD 也无能为力——它最多传递发现的改进，不能创造改进
6. **Scalability 边界**：当前最大的学生是 7B，验证在 70B+ 规模上是否仍然保持效率优势是重要的下一步

值得注意的是，上述局限不改变 Direct-OPD 的核心贡献——它首次展示了「策略偏移」而非「最终策略」才是弱到强转移的正确对象。这一认知可能对 LLM 后训练的工程实践产生深远影响。

---

## Ch6：总结与思考

### 6.1 核心贡献总结

Direct-OPD 从根本上改变了小模型 RL 的语义——从「最终产品」变为「可复用的改进方向」。具体来说：

| 维度 | 传统方案 | Direct-OPD |
|------|---------|------------|
| 传递对象 | 教师最终策略 | 教师策略偏移（隐式奖励） |
| 学生上限 | 受限于教师能力 | 可超越教师 |
| 跨模式转移 | 需要 token 重叠 | 低重叠仍有效 |
| 训练成本 | 每个目标从头 RL | 小模型 RL + 廉价转移 |
| 组合性 | 无法组合 | 多教师偏移可顺序组合 |

### 6.2 方法论启示

**RL 的复用不应该停留在模型级别，而应该提升到「学习信号」级别。** 在 Direct-OPD 中，小模型 RL 运行的有价值产物不是其最终策略的参数，而是策略如何从 `ref` 检查点变化到 `post-RL` 检查点——这个变化向量浓缩了整个 RL 训练过程中所发现的改进方向。

这一思路暗示了一个更广泛的范式：**将昂贵的训练过程拆解为「探索阶段」（小模型、低成本）和「应用阶段」（大模型、快速转移）**，而非在每个规模上重复完整的探索-利用循环。

### 6.3 未来方向

1. 在代码生成、科学推理等更广泛的领域验证 Direct-OPD
2. 研究多个教师偏移的加权组合而非简单顺序叠加
3. 探索将 Direct-OPD 与模型合并（model merging）相结合
4. 分析在何种条件下教师偏移信号会完全失效

### 6.4 对 OPD 社区的影响

Direct-OPD 揭示了 OPD 中一个重要但被忽视的自由度：**传递的信号不必是教师的完整输出分布**。检查点对中的差分信息已经包含了 RL 的核心学习成果，而丢弃教师的基线偏好（pre-RL 分布）恰恰移除了弱教师的能力瓶颈。这意味着 OPD 研究社区的关注点可能从「如何更好地匹配教师」转向「如何从教师轨迹中提取更有信息量的子信号」。
