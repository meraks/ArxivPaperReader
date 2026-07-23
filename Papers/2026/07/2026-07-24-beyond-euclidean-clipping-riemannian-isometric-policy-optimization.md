> **论文**：Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization
> **作者**：Zhicheng Cai, Xinyuan Guo, Hanlin Wu, Mingxuan Wang, Wei-Ying Ma, Ya-Qin Zhang, Hao Zhou
> **arXiv ID**：2607.10169
> **发表时间**：2026-07-11（ICML 2026）
> **许可协议**：CC BY 4.0
> **代码仓库**：无官方实现

# Beyond Euclidean Clipping：在 LLM 强化学习中通过 Riemann 等距策略优化克服探索崩溃

## 第 1 章 概述

### 1.1 一句话定位

本文揭示了 PPO-Clip 在 LLM 强化学习中的根本性几何缺陷——其使用的 Euclidean 度量与策略 Riemann 流形的内在几何不匹配，导致低概率区域更新不足、高概率区域更新过度，最终引发探索崩溃；并提出了 Riemann 等距策略优化（RIPO），通过动态调整剪裁边界实现在 Riemann 流形上的等距更新，从根本上解决了这一缺陷。

### 论文图表总览

| 编号 | 内容 | 章节 |
|------|------|------|
| **Table 1** | 主实验结果（4 模型 × 7 基准 × 6 基线） | 第 5 章 |
| **Table 2** | RIPO 与 GPPO/Clip-Cov 对比 | 第 5 章 |
| **Table 3** | RIPO-Clip 迁移到 PPO 目标（GSM8K） | 第 5 章 |
| **Table 4** | Pass@k 深度分析（AIME-25 / HMMT-25） | 第 5 章 |
| **Table 5** | 编码与搜索任务泛化 | 第 5 章 |
| **Figure 1** | 消融分析（对称 vs 非对称 δ） | 第 5 章 |
| **Figure 2** | 训练动态可视化（GSM8K） | 第 5 章 |

### 1.2 核心贡献

1. **理论层面**：首次从 Riemann 几何角度揭示了 PPO-Clip 的 Euclidean 度量与策略流形几何不匹配是探索崩溃的根本原因，而非之前认为的 engineering 问题
2. **算法层面**：提出了 Riemann 等距策略优化（RIPO），其核心是 Riemann 等距剪裁（RIC）——根据局部概率 simplex 动态调整剪裁边界 ϵ_{s,a} = √(δ/π_{θ_old}(a|s))
3. **理论分析**：证明了几何等距性诱导统计同方差性，实现了更优的偏差-方差权衡
4. **实验验证**：在 4 个不同规模/类型的 LLM、7 个竞赛级数学基准、6 种代表性 RL 基线中全面超越，在 AIME24 上相对 GRPO 提升高达 60%

### 1.3 关键结果速览

- RIPO 在 Qwen3-8B-Base 上达到 7 基准平均 38.5%，超越 GRPO（28.5%）达 10.0 pp，超越最佳基线 GMPO（35.3%）达 3.2 pp
- 在 Qwen3-8B-Base AIME24 上 RIPO 达 43.8%，相对 GRPO（31.7%）提升 38.2%
- 在 Llama3.2-3B-Instruct（最小的被测试模型）上 RIPO 8 基准平均 8.6%，相对 GRPO（6.4%）提升 34.4%
- Pass@128 分析显示 RIPO 在 AIME-25 达 60.0%，持续超越其他方法（GRPO: 53.3%, DCPO: 58.9%），且未出现性能饱和
- 编码任务（4 基准平均）RIPO 44.9% vs GRPO 39.7%（+13.2%）
- 搜索任务（4 基准平均）RIPO 43.4% vs GRPO 37.7%（+15.1%）

## 第 2 章 研究背景与动机

### 2.1 策略优化中的信任区域方法

强化学习中策略优化面临的核心挑战是新旧策略之间更新的幅度控制。TRPO（Schulman et al., 2015）从理论上建立了通过 KL 散度约束（信任区域）来保证单调改进的框架：

$$
\max_{\theta}\; \mathbb{E}_{s\sim\rho_{\theta_{\text{old}}},a\sim\pi_{\theta_{\text{old}}}}\left[\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}\hat{A}(s,a)\right] \quad \text{s.t.}\; \mathbb{E}_{s}\left[D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s)||\pi_{\theta}(\cdot|s))\right]\leq\delta
$$

TRPO 具有坚实的理论保证，但需要计算 Fisher 信息矩阵并进行共轭梯度优化，计算开销极大，难以在 LLM 场景中规模化应用。

### 2.2 PPO 与 Clip 机制

为了缓解 TRPO 的计算负担，PPO（Schulman et al., 2017）引入了一阶剪裁替代目标来近似二阶优化：

$$
\mathcal{J}_{\text{PPO}}(\theta)=\mathbb{E}\left[\min\left(r(\theta)\hat{A},\;\text{clip}(r(\theta),1-\epsilon,1+\epsilon)\hat{A}\right)\right]
$$

其中 $r_{s,a}(\theta)=\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$ 是重要性比率，$\epsilon$ 是固定的剪裁边界。PPO-Clip 通过将重要性比率限制在 $(1-\epsilon,1+\epsilon)$ 区间内，约束策略更新在邻近区域。由于其简单高效，剪裁已成为现代 LLM RL 算法的核心组件。

### 2.3 GRPO 及其变体

GRPO（Shao et al., 2024; Guo et al., 2025）保留了 PPO-Clip，但去除了价值模型，采用组内相对方式估计优势函数，提升了计算和内存效率。给定一个问题 $q\sim\mathcal{D}$，旧策略 $\pi_{\theta_{\text{old}}}$ 采样 $G$ 个响应 $\{o_i\}_{i=1}^G$，GRPO 目标函数为：

$$
\mathcal{J}_{\text{GRPO}}(\theta)=\mathbb{E}_{q\sim\mathcal{D},\{o_i\}_{i=1}^G\sim\pi_{\theta_{\text{old}}}(\cdot|q)}\left[\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\text{clip}(r_{i,t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_{i,t}\right)\right]
$$

一系列 GRPO 变体被提出以缓解探索崩溃和梯度不稳定问题：
- **DAPO**（Yu et al., 2025）：提出解耦 Clip-Higher，设置 $(1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})$ 且 $\epsilon_{\text{high}}>\epsilon_{\text{low}}$ 以鼓励探索
- **DCPO**（Yang et al., 2025b）：动态自适应调整剪裁阈值
- **GSPO**（Zheng et al., 2025）/ **GMPO**（Zhao et al., 2025）：在序列级重要性比率上做剪裁以降低梯度方差

然而，这些变体本质上是启发式的，并未识别 PPO-Clip 的根本缺陷。

### 2.4 探索崩溃现象

PPO-Clip 在实际训练中表现出严重的探索崩溃：策略快速集中在少数高概率动作上，稀有但有价值的动作得不到充分更新。在长程推理任务中，状态-动作空间极其庞大，探索崩溃尤其致命。

直观理解：给定 $\epsilon=0.2$（默认值），高概率动作 $\pi_{\theta_{\text{old}}}=0.8$ 的最大更新概率可达 0.96（增加 0.16），而低概率动作 $\pi_{\theta_{\text{old}}}=0.01$ 的最大更新概率仅为 0.012（增加 0.002）。这种不对称性导致策略多样性迅速下降。

## 第 3 章 PPO-Clip 的几何缺陷

### 3.1 PPO-Clip 隐含的 Euclidean 假设

PPO-Clip 通过 $|r(\theta)-1|<\epsilon$ 约束比率，这隐含了 Euclidean 距离度量：

$$
d_{\text{clip}}(\pi_{\theta_{\text{old}}},\pi_{\theta})=\left(\frac{\pi_{\theta}}{\pi_{\theta_{\text{old}}}}-1\right)^2=(r(\theta)-1)^2
$$

在此视角下，只要两个样本具有相同的比率偏差，PPO-Clip 就认为它们具有相同的信任级别，施加相同的约束。这隐含假设策略变化在整个统计流形上是均匀的。然而，这一假设在几何上根本错误。

### 3.2 策略差异的 Riemann 几何

TRPO 中已建立，两个策略之间的差异由 KL 散度衡量：

$$
D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s)||\pi_{\theta}(\cdot|s)) \approx \frac{1}{2}\Delta\theta^{\top}F(\theta)\big|_{\theta=\theta_{\text{old}}}\Delta\theta
$$

其中 $F(\theta)$ 是 Fisher 信息矩阵，定义了 Riemann 度量。通过将参数空间映射到概率分布空间，可以推导出 KL 散度与重要性比率之间的关系：

$$
D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s)||\pi_{\theta}(\cdot|s)) \approx \frac{1}{2}\sum_{a}\pi_{\theta_{\text{old}}}(a|s)\left(r_{s,a}(\theta)-1\right)^2
$$

因此，Riemann 流形上的几何距离为：

$$
d_{\text{geom}}(\pi_{\theta_{\text{old}}},\pi_{\theta}) \propto \pi_{\theta_{\text{old}}}\cdot(r(\theta)-1)^2
$$

**关键差异**：这依赖于底层概率 simplex $\pi_{\theta_{\text{old}}}$——低概率区域距离收缩，高概率区域距离膨胀。而 PPO-Clip 的 Euclidean 度量忽略了这种依赖关系。

### 3.3 信任区域视角下的探索崩溃

考虑 3.1 节的数值示例，从信任区域视角重新审视：

- 高概率动作 $\pi_{\theta_{\text{old}}}(a|s)=0.8$，$r(\theta)=1.2$（达到剪裁边界）：
  诱导的 KL 距离 $= 0.5 \times 0.8 \times 0.2^2 = 0.016$

- 低概率动作 $\pi_{\theta_{\text{old}}}(a|s)=0.01$，同样 $|r(\theta)-1|=0.2$：
  诱导的 KL 距离 $= 0.5 \times 0.01 \times 0.2^2 = 0.0002$

在 PPO-Clip 下，两个更新被视为完全相同的比率偏差。但在 Riemann 几何中，第二个更新几乎不移动距离，留下了大量未使用的信任区域预算。高概率 token 的更新过于激进，而低概率 token 的信任区域极其受限。这正是探索崩溃的根本原因。

**本文核心理论贡献（Proposition 3.1）**：PPO-Clip 错误地使用 Euclidean 度量衡量策略间差异，与策略 Riemann 流形的几何不一致，导致低概率区域更新保守、高概率区域更新激进，最终引发探索崩溃。

## 第 4 章 Riemann 等距策略优化

### 4.1 Riemann 等距剪裁（RIC）

为了解决 PPO-Clip 的固有缺陷，本文提出每个状态-动作样本的更新应在 Riemann 流形上移动相同的几何距离，而非具有相同的 Euclidean 比率偏差。形式化地，要求每个更新满足：

$$
d_{\text{geom}}(\pi_{\theta_{\text{old}}}(a|s),\pi_{\theta}(a|s)) \triangleq \frac{1}{2}\pi_{\theta_{\text{old}}}(a|s)(r_{s,a}(\theta)-1)^2 \leq \delta
$$

解出 $r_{s,a}(\theta)$ 得到理论上的比率约束：

$$
|r_{s,a}(\theta)-1| \leq \epsilon_{s,a}(\pi_{\theta_{\text{old}}}),\quad \epsilon_{s,a}(\pi_{\theta_{\text{old}}}) = \sqrt{\frac{\delta}{\pi_{\theta_{\text{old}}}(a|s)}}
$$

**动态剪裁边界**：$\epsilon_{s,a}$ 与 $\pi_{\theta_{\text{old}}}(a|s)$ 的平方根成反比。低概率动作获得更大的更新幅度，高概率动作被限制更紧。

回到数值示例（设 $\delta=0.02$）：
- 高概率动作 $\pi_{\theta_{\text{old}}}=0.8$：最大更新概率被限制到 0.92（PPO-Clip 为 0.96）
- 低概率动作 $\pi_{\theta_{\text{old}}}=0.01$：允许概率增加到 0.024（PPO-Clip 仅为 0.012）
- 两种动作消耗的信任区域（几何距离）均为 0.01——实现真正的等距更新

### 4.2 几何等距与统计同方差性

RIC 的另一个优势体现在偏差-方差权衡上。在离策略 RL 中，重要性采样的方差主要来源于二阶矩项：

$$
\mathbb{V}_{x\sim\pi_{\theta_{\text{old}}}}\left[r(x)A(x)\right] \approx \sum_x \pi_{\theta_{\text{old}}}(x)\,r(x)^2 A(x)^2
$$

定义 $v(x)=\pi_{\theta_{\text{old}}}(x)r(x)^2$，当 $\pi_{\theta_{\text{old}}}(x)$ 很小时，$v(x)=\pi_{\theta}(x)^2/\pi_{\theta_{\text{old}}}(x)$ 可能变得极大。

PPO-Clip 通过截断 $r(x)\leq 1+\epsilon$ 将 $v(x')$ 推向 0，但引入了严重偏差。RIC 的密度相关剪裁则实现了：

$$
v(x')=\pi_{\theta_{\text{old}}}(x')\left(1+\sqrt{\frac{\delta}{\pi_{\theta_{\text{old}}}(x')}}\right)^2 \approx \mathcal{O}(\delta)
$$

因此 RIC 实现了与密度无关的常数阶方差——几何等距诱导了统计同方差性，方差严格小于标准重要性采样，偏差也远小于 PPO-Clip。

**本文核心理论贡献（Proposition 4.1）**：Riemann 等距剪裁保证每个状态-动作更新在策略 Riemann 流形上具有相等的几何距离，同时兼顾探索崩溃缓解和偏差-方差权衡。

### 4.3 RIPO 完整目标函数

RIPO 在 GRPO 框架下采用 RIC 替代固定剪裁：

$$
\mathcal{J}_{\text{RIPO}}(\theta)=\mathbb{E}_{q\sim\mathcal{D},\{o_i\}_{i=1}^G\sim\pi_{\theta_{\text{old}}}(\cdot|q)}\left[\frac{1}{\sum_{i=1}^G|o_i|}\sum_{i=1}^G\sum_{t=1}^{|o_i|}\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\;\text{clip}\left(r_{i,t}(\theta),\;1-\epsilon_{i,t}(\pi_{\theta_{\text{old}}}),\;1+\epsilon_{i,t}(\pi_{\theta_{\text{old}}})\right)\hat{A}_{i,t}\right)\right]
$$

其中 $\epsilon_{i,t}(\pi_{\theta_{\text{old}}})$ 即为 RIC 的动态剪裁边界。RIPO 采用 token 级策略梯度损失（如 DAPO），平衡长短轨迹的梯度影响。

## 第 5 章 实验结果与分析

### 5.1 实验设置

**模型**：Llama3.2-3B-Instruct、Qwen3-1.7B-Base、Qwen3-4B-Base、Qwen3-8B-Base

**基线**：GRPO（PPO-Clip）、DAPO（Clip-Higher）、GSPO（序列级剪裁）、GMPO（几何平均比率剪裁）、DCPO（动态自适应剪裁）、GPPO（梯度保留剪裁）、Clip-Cov（高协方差剪裁）

**训练数据**：DAPO-Math-17k（17,917 道数学题），每问题生成 8 个 rollout，最大响应长度 16,384 token。RIPO 默认 $\delta=0.05$。所有方法均移除 KL 惩罚项。

**基准**：AIME24、AIME25、AMC23、HMMT25、BRUMO25、CMIMC25、SMT25（七个竞赛级数学基准）

### 5.2 主实验结果

**Qwen3-8B-Base 主结果**

| 基准 | GRPO | DAPO | GSPO | GMPO | DCPO | RIPO |
|:-----|:----:|:----:|:----:|:----:|:----:|:----:|
| AIME24 | 31.7 | 33.7 | 37.5 | 41.7 | 36.3 | **43.8** |
| AIME25 | 20.8 | 22.1 | 19.6 | 30.0 | 27.5 | **29.2** |
| AMC23 | 66.6 | 70.9 | 70.1 | **76.9** | 72.2 | 79.7 |
| HMMT25 | 12.9 | 7.9 | 15.0 | 11.7 | 15.4 | **16.7** |
| BRUMO25 | 33.3 | 33.3 | 36.3 | 35.8 | 42.8 | **47.5** |
| CMIMC25 | 12.5 | **18.1** | 13.8 | 17.8 | 16.3 | **18.1** |
| SMT25 | 21.9 | 28.3 | 32.5 | 33.0 | 31.6 | **34.2** |
| **平均** | 28.5 | 30.6 | 32.1 | 35.3 | 34.5 | **38.5** |

RIPO 在 7 个基准中的 6 个上取得最佳（AMC23 略逊于 GMPO），平均超越 GRPO 达 10.0 pp（+35.1%）。

**Qwen3-4B-Base 主结果**

| 基准 | GRPO | DAPO | GSPO | GMPO | DCPO | RIPO |
|:-----|:----:|:----:|:----:|:----:|:----:|:----:|
| AIME24 | 30.4 | 31.7 | 32.1 | 32.5 | 31.7 | **32.9** |
| AIME25 | 20.8 | 21.3 | 21.3 | 26.7 | 24.2 | **27.5** |
| AMC23 | 63.4 | 69.4 | 70.3 | 68.8 | 71.3 | **73.1** |
| HMMT25 | 7.5 | 9.2 | 9.2 | **11.3** | 8.8 | 10.0 |
| BRUMO25 | 25.8 | 26.3 | 26.7 | 26.3 | 25.0 | **31.3** |
| CMIMC25 | **12.4** | 10.0 | 10.0 | 10.6 | 15.0 | 14.7 |
| SMT25 | 19.8 | **21.7** | 18.2 | 22.5 | 19.8 | 21.5 |
| **平均** | 25.7 | 27.1 | 26.8 | 28.4 | 27.8 | **30.1** |

**Qwen3-1.7B-Base 主结果**

| 基准 | GRPO | DAPO | GSPO | GMPO | DCPO | RIPO |
|:-----|:----:|:----:|:----:|:----:|:----:|:----:|
| AIME24 | 11.3 | 15.0 | 16.3 | 17.5 | 17.1 | **18.3** |
| AIME25 | 10.8 | 12.1 | 11.2 | 12.5 | 11.3 | **12.9** |
| AMC23 | 33.8 | 39.7 | 40.0 | 40.9 | **46.9** | 46.1 |
| HMMT25 | 0.0 | 0.8 | 1.6 | 0.8 | 1.3 | **1.3** |
| BRUMO25 | 15.0 | 9.2 | 12.9 | 15.0 | 15.4 | **15.4** |
| CMIMC25 | 0.3 | 1.9 | 2.8 | 3.8 | 1.3 | **3.2** |
| SMT25 | 7.3 | 7.8 | 8.5 | 6.6 | 10.8 | **10.4** |
| **平均** | 11.2 | 12.4 | 13.3 | 13.9 | 14.8 | **15.4** |

**Llama3.2-3B-Instruct 主结果**

| 基准 | GRPO | DAPO | GSPO | GMPO | DCPO | RIPO |
|:-----|:----:|:----:|:----:|:----:|:----:|:----:|
| AIME24 | 16.3 | **20.1** | 17.5 | 16.7 | 16.7 | 22.1 |
| AIME25 | 1.3 | 1.7 | 1.7 | 1.3 | **2.1** | **2.1** |
| AMC23 | 24.5 | 25.3 | 25.7 | **27.7** | 24.7 | 28.1 |
| HMMT25 | 0.4 | **0.8** | 0.4 | 0.4 | 0.4 | **0.8** |
| BRUMO25 | 1.7 | 2.5 | 3.8 | 2.8 | 2.5 | **4.3** |
| CMIMC25 | 0.0 | 0.3 | **0.6** | 0.9 | 0.6 | **0.9** |
| SMT25 | 0.7 | 0.9 | 0.9 | **1.7** | 1.4 | **1.7** |
| **平均** | 6.4 | 7.4 | 7.2 | 7.4 | 6.9 | **8.6** |

**关键观察**：
- **跨模型规模一致性**：RIPO 在所有 4 个模型上均取得最佳平均分，提升幅度与模型规模无单调关系
- **极端困难基准的突破**：在 HMMT25 和 BRUMO25 等极难基准上（基线 < 5%），RIPO 的提升尤为显著，说明几何等距更新有利于长程推理探索
- **Llama3.2-3B 的特殊性**：在最小的模型上 GRPO 仅 6.4%，RIPO 提升至 8.6%（+34.4%），DAPO 反而在 AIME24 上略超 RIPO，但 RIPO 在多个基准上更均衡

### 5.3 与其他剪裁机制对比

| 基准 | GRPO | GPPO | Clip-Cov | RIPO |
|:-----|:----:|:----:|:--------:|:----:|
| AIME24 | 31.7 | 31.7 (0.0) | 36.3 (+4.6) | **43.8 (+12.1)** |
| AIME25 | 20.8 | 23.8 (+3.0) | 22.9 (+2.1) | **29.2 (+8.4)** |
| AMC23 | 66.6 | 73.4 (+6.8) | 66.6 (0.0) | **79.7 (+13.1)** |
| HMMT25 | 12.9 | 14.2 (+1.3) | 11.7 (−1.2) | **16.7 (+3.8)** |
| BRUMO25 | 33.3 | 38.3 (+5.0) | 36.3 (+3.0) | **47.5 (+14.2)** |
| SMT25 | 21.9 | 25.0 (+3.1) | 30.2 (+8.3) | **34.2 (+14.2)** |

GPPO 通过保留被剪裁 token 的梯度避免信息损失，Clip-Cov 通过剪裁高协方差 token 调节熵。RIPO 在所有基准上大幅超越两者，验证了理论驱动方法相对于启发式方法的优势。

### 5.4 PPO 目标迁移实验

将 RIC 应用于标准 PPO 目标（使用 GAE 优势估计和价值模型），在 GSM8K 上测试：

| 模型 | PPO-Clip | DAPO-Clip | DCPO-Clip | RIPO-Clip |
|:----|:--------:|:---------:|:---------:|:---------:|
| 0.5B | 58.0 | 58.7 (+0.7) | 58.4 (+0.4) | **61.1 (+3.1)** |
| 1.5B | 79.2 | 80.9 (+1.7) | 79.6 (+0.4) | **81.6 (+2.4)** |
| 7B | **91.7** | 90.8 (−0.9) | 91.4 (−0.3) | 93.5 (+1.8) |
| 14B | 93.2 | 93.6 (+0.4) | 93.9 (+0.7) | **94.4 (+1.2)** |

RIC 在所有 4 个模型规模上一致优于其他剪裁机制，且提升幅度随模型规模递减（从 3.1 pp 降至 1.2 pp），说明大模型的固有探索能力更强，但 RIC 仍然提供了正交增益。

训练动态分析显示：PPO-Clip 的熵迅速衰减至接近零（探索崩溃），而 RIPO-Clip 维持了稳定的熵水平。同时 RIPO-Clip 的梯度范数更平滑，无剧烈尖峰。

### 5.5 Pass@k 深度分析与能力边界突破

| 方法 | Avg@8 | Pass@1 | Pass@8 | Pass@16 | Pass@32 | Pass@64 | Pass@128 |
|:----|:-----:|:-----:|:------:|:-------:|:-------:|:-------:|:--------:|
| **AIME-25** | | | | | | | |
| Base | 3.3 | 5.0 | 12.3 | 16.7 | 16.7 | 16.7 | 16.7 |
| GRPO | 20.8 | 20.4 | 36.5 | 40.8 | 45.5 | 50.1 | 53.3 |
| DAPO | 22.1 | 20.6 | 33.4 | 35.7 | 37.4 | 39.1 | 40.0 |
| DCPO | 27.5 | 26.4 | 39.5 | 44.7 | 48.5 | 54.7 | 58.9 |
| RIPO | **29.2** | **30.4** | **43.2** | **47.0** | **50.8** | **55.6** | **60.0** |
| **HMMT-25** | | | | | | | |
| Base | 1.3 | 0.6 | 3.5 | 5.2 | 6.3 | 6.7 | 6.7 |
| GRPO | 12.9 | 11.7 | 19.3 | 21.6 | 24.1 | 26.6 | 30.0 |
| DAPO | 7.9 | 7.7 | 16.7 | 20.3 | 23.4 | 25.7 | 26.7 |
| DCPO | 15.4 | 15.5 | 26.8 | 30.6 | 34.3 | 37.5 | 40.0 |
| RIPO | **16.7** | **16.7** | **28.3** | **32.6** | **37.3** | **41.4** | **45.3** |

Base 模型的性能在 Pass@16 后便饱和（说明固有推理容量有限），而 RIPO 持续扩展到 Pass@128，未出现早期饱和。DAPO 在 Pass@16 后甚至下降（说明过度探索导致了性能退化），而 RIPO 保持了稳定增长。这实证表明 RIPO 有效维持了策略多样性，突破了基础模型固有的推理能力边界。

### 5.6 编码与搜索任务泛化

| 任务类型 | 基准 | RIPO | GRPO |
|:--------|:-----|:----:|:----:|
| **编码** | Codeforces | **51.5** | 46.8 |
| | CodeContest | **49.9** | 44.8 |
| | TACO | **25.8** | 19.7 |
| | APPS | **52.5** | 47.6 |
| | **平均** | **44.9** (+13.2%) | 39.7 |
| **搜索** | TriviaQA | **67.4** | 60.9 |
| | PopQA | **39.8** | 34.5 |
| | HotpotQA | **30.9** | 25.1 |
| | WikiMultiHopQA | **35.5** | 30.4 |
| | **平均** | **43.4** (+15.1%) | 37.7 |

### 5.7 消融研究

**对称 vs 非对称 δ 的影响**

| δ_low | 0.02 | 0.05 | 0.05 | 0.08 | 0.05 | 0.08 | 0.08 |
|:-----|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| δ_high | 0.02 | 0.04 | 0.05 | 0.08 | 0.02 | 0.02 | 0.04 |
| Avg@8 | 40.8 | 41.7 | **43.8** | 42.1 | 28.8 | 27.5 | 27.9 |

对称设置（δ_low = δ_high）的性能优于非对称设置。当 δ_high 显著大于 δ_low 时，性能急剧下降。对称剪裁确保了双向更新幅度一致，避免策略朝单方向过度更新导致训练崩溃。

## 第 6 章 局限性与讨论

### 6.1 局限性

1. **超参数敏感性**：RIPO 引入了一个新的超参数 δ（信任区域半径），默认值 0.05 经过实验验证但在不同任务/模型上可能需要调整
2. **计算开销**：RIP 需要计算每个 token 的 π_{θ_old}(a|s)，增加了少量额外计算（从重要性比率中已知），但不引入额外模型调用
3. **理论假设强度**：本文的推导基于 KL 散度的二阶泰勒近似和 Fisher 信息矩阵的可逆性假设，在深层神经网络的实际参数空间中这些假设可能不完全成立
4. **实验范围限定**：主要实验限于数学推理任务（7 个数学基准），编码和搜索各仅 1 个数据集。在更广泛任务上的泛化需要进一步验证

### 6.2 讨论

**几何视角的价值**：本文的核心洞见是将策略优化中的剪裁问题重新解释为几何度量问题，这一视角为设计更鲁棒的 RL 算法提供了理论基础。RIPO 本质上是在 PPO 的一阶近似框架内恢复了 TRPO 的几何约束，但避免了二阶计算的成本。

**与 DAPO 的关系**：DAPO 的解耦 Clip-Higher 虽然探索方向正确（鼓励低概率动作更新），但仍然是 Euclidean 框架内的启发式修补。RIPO 从几何原理出发的解决方案更为彻底——不仅鼓励探索，还精确控制了探索与利用的几何平衡。

**未来方向**：将 RIC 扩展到在线 RL 设置（本文所有实验为离策略 GRPO/PPO 框架）、探索 Riemann 流形上的自适应 δ 调度策略、将几何理论推广到其他分布匹配问题（如 distillation 中的 KL 约束）。

## 参考文献

- Schulman et al. (2015). Trust Region Policy Optimization. ICML.
- Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.
- Guo et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.
- Yu et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14442.
- Yang et al. (2025b). DCPO: Dynamic Clipping Policy Optimization. arXiv.
- Zheng et al. (2025). Group Sequence Policy Optimization. arXiv.
- Zhao et al. (2025). Geometric-Mean Policy Optimization. arXiv.
