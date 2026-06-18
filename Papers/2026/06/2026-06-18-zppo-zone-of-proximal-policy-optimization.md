# Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients

## 论文元数据
- **标题：** Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients
- **作者：** Byung-Kwan Lee (†Project Lead), Ximing Lu, Shizhe Diao, Minki Kang, Saurav Muralidharan, Karan Sapra, Andrew Tao, Pavlo Molchanov, Yejin Choi, Yu-Chiang Frank Wang, Ryo Hachiuma — NVIDIA
- **arXiv ID：** 2606.18216
- **提交日期：** 2026-06-17
- **官方代码：** 无公开仓库（NVIDIA Technical Patent Filed, 2026）
- **项目页面：** https://byungkwanlee.github.io/ZPPO-page/
- **数据集：** ZPPO-77K（77K image-question pairs 多模态 RL 数据集）

---

## Ch1：论文概述与核心贡献

### 核心问题

大型语言模型（LLM）和视觉-语言模型（VLM）的性能——尤其在 RL 后训练之后——集中在数十亿参数的模型上。将前沿能力压缩到可在移动设备、AR/VR 眼镜、机器人上部署的小模型（0.8B–9B）是部署的核心瓶颈。现有两大方案各有根本缺陷：

| 方案 | 核心机制 | 根本缺陷 |
|------|---------|---------|
| **知识蒸馏** | 学生模仿教师 logit/logits/hidden states | Mode-seeking：小容量学生集中到教师最尖锐的少数模式，泛化脆弱 |
| **强化学习（GRPO）** | 学生自身 rollout 的 group-relative advantage | 所有 rollout 全失败时产生零优势 → hard question 被静默丢弃 |

ZPPO 提出第三个路径：**教师仅出现在 prompt 中，从不进入 policy gradient**。

### ZPPO 核心思想

ZPPO 受 Vygotsky 的"最近发展区（Zone of Proximal Development）"理论启发，通过三种机制将教师知识限制在 prompt 空间：

1. **BCQ（Binary Candidate-included Question）**：一条教师正确响应 + 一条学生错误响应匿名配对放入 prompt，学生甄别哪个正确
2. **NCQ（Negative Candidate-included Question）**：聚合学生当前 group 的所有错误 rollout，暴露共享失败模式
3. **Prompt Replay Buffer**：FIFO 队列（容量 10,000），准入条件 $\bar{r}_x < 0.5$，毕业条件 $\bar{r}_x \ge 0.5$

### 核心结果

| 学生规模 | 16 VLM（pp） | 10 LLM（pp） | 5 Video（pp） |
|---------|:-----------:|:-----------:|:------------:|
| **0.8B** | **+9.3** | **+7.9** | **+4.5** |
| 2B | +5.2 | +5.1 | +2.6 |
| 4B | +4.0 | +3.9 | +0.3 |
| 9B | +2.8 | +3.9 | +0.4 |

关键发现：**蒸馏在训练语料外损害泛化（LLM+Video avg -2.5pp at 0.8B），ZPPO 改进泛化（+6.8pp at 0.8B）**。

---

## Ch2：研究背景与两个已知失败模式

### 2.1 小模型蒸馏的固有问题

Off-policy 蒸馏（学生通过 KL 散度模仿教师 logit 分布）、On-policy 蒸馏（学生实时采样后教师评分再模仿）、self-distillation（学生自蒸馏），三者共享根本约束：训练信号是学生容量不足以表示的 logit 分布。对于 0.8B 或 2B 学生，有限的模型容量导致**模式寻求偏差（mode-seeking bias）**——学生集中在教师最尖锐的少数峰上，在训练语料外的 benchmark 上泛化脆弱（Tab. 2 验证：0.8B 蒸馏在 LLM+Video 上退化为 -2.5pp）。

### 2.2 GRPO 的零优势陷阱

GRPO 计算 group-relative advantage：

$$A^{(g)} = \frac{r(x, y_S^{(g)}) - \bar{r}_x}{\text{std}_x + \epsilon} \tag{1}$$

当 rollout group 全部错误（$\bar{r}_x = 0$）或全部正确（$\bar{r}_x = 1$）时，group 内每个 advantage 恰好为零，该 question 完全不产生梯度信号。对于小模型，全部错误的情况正是**需要教师引导的那组问题**。

在 $\{0,1\}$ 二值奖励下，std 在 $\bar{r}_x = 0.5$ 处达到最大值——携带最强学习信号的区域恰恰是 GRPO 的盲区。

### 2.3 Hybrid 方案的局限

一个直觉修复——将教师正确响应注入学生策略梯度——违反 on-policy 假设。教师响应来自 $\pi_T$（而非 $\pi_S$），会产生策略漂移（policy drift），因为教师响应远在学生 rollout 分布之外。

---

## Ch3：ZPPO 核心方法

### 3.1 符号定义

- $x$：问题（图像 + 文本）
- $y \sim \pi_\theta(\cdot|x)$：学生从当前策略采样的响应
- $G_S$：学生 rollout 分组的 group size
- $r(x, y) \in \{0, 1\}$：outcome reward（最终答案是否正确）
- $\bar{r}_x$：group mean reward，$\text{std}_x$：group standard deviation
- **Hard question 定义**：$\bar{r}_x < 0.5$
- **Zone of Proximal Development**：$\bar{r}_x < 0.5$ 的问题集合——学生尚未掌握但通过引导可学会

### 3.2 BCQ（Binary Candidate-included Question）

对每个 hard question $x$，BCQ 执行以下步骤：

1. **候选采样**：从当前 group 的学生 rollout 中均匀采样一条错误响应 $y_S^{(-)}$；从教师 rollout（$G_T$ 次采样，$\pi_T$ 冻结）中均匀采样一条正确响应 $y_T^{(+)}$
2. **候选压缩**：冻结教师将两条轨迹重写为短推理过程（512 token 上限，共享压缩 prompt）
3. **匿名化**：两条轨迹放入 `<candidate>` 标签，随机打乱顺序，附加指令：
   > "Here are two candidate responses in <candidate> </candidate> tags to the question above. One is correct and another is wrong."
4. **学生新采样**：学生从 BCQ prompt $x_{BCQ}$ 采样全新 rollout group $\{y_S^{(g)}\}_{g=1}^{G_S} \sim \pi_\theta(\cdot|x_{BCQ})$

**On-policy 保证**：所有梯度计数的 token 均由当前学生采样，教师文本仅作为 prompt 输入，从不进入 policy gradient 作为目标。

### 3.3 NCQ（Negative Candidate-included Question）

对每个 hard question $x$，NCQ：

1. 收集当前 group 所有错误学生 rollout，解析每个 rollout 的最终答案
2. 在 prompt 中显式列出所有这些错误答案：
   > "The following answers are all WRONG: $\langle$parsed answer$\rangle$. Below are the incorrect reasoning processes in <candidate> </candidate> tags."
3. 每个教师压缩的错误轨迹作为一个 `<candidate>` 块附加
4. 学生采样全新 rollout

**NCQ 的结构性差异**：在独立 rollout group 中，每个错误 rollout 独立贡献梯度，学生无法看到跨 rollout 的失败模式。NCQ 是训练循环中**首次将独立失败转化为集体可见信号**的地方。

### 3.4 Prompt Replay Buffer

Buffer $\mathcal{B}$ 仅存储问题 $x$（图像+文本），不存储响应。

| 属性 | 值 |
|------|----|
| **准入条件** | $\bar{r}_x < 0.5$（hard question） |
| **毕业条件** | 后续某 step 上 $\bar{r}_x \ge 0.5$ |
| **淘汰机制** | FIFO，容量 $|\mathcal{B}|_{\max} = 10{,}000$ |
| **重放比例** | $\rho_{\text{replay}} = 0.25$（重放问题数 / 新问题数） |
| **增强上限** | $\rho_{\text{aug}} = 0.25$（BCQ+NCQ 总数 / 新问题数） |

每次重放时，BCQ/NCQ 候选由当前学生和教师的新采样生成——每次访问候选各不相同。

### 3.5 RL 主干配方

ZPPO 基于 GRPO [36] + DAPO [8] 的三个成分：

1. **Clip-higher**：$\epsilon_{\text{low}} = 0.20, \epsilon_{\text{high}} = 0.28$
2. **Token-level policy gradient loss**：逐 token 计算替代损失而非整序列平均
3. **移除 KL penalty**：不需要与 reference policy 的 KL 散度惩罚

外加两个对训练动力学敏感的 recipe 选择：

4. **Iterations per step $I = 4$**（vs GRPO 默认 $I=16$；$I=16$ 导致小模型漂移）
5. **Batch-level advantage normalization（排除零优势组）**：零优势组（all-wrong/all-correct）的 batch mean/std 会扭曲 batch 级别统计，必须在计算 batch 归一化时排除

---

## Ch4：实验结果与分析

### 4.1 实验设置

| 参数 | 值 |
|------|----|
| 学生模型 | Qwen3.5: 0.8B / 2B / 4B / 9B |
| 教师模型 | Qwen3.5-27B-FP8（冻结） |
| 训练数据 | ZPPO-77K（77K image-question pairs） |
| 评估套件 | 31 benchmarks: 16 VLM + 10 LLM + 5 Video |
| 硬件 | NVIDIA GPU（论文未明确具体型号/数量） |

**基线方法**：Off-policy distillation、On-policy distillation、GRPO（ZPPO pipeline 不含 BCQ/NCQ/buffer）、GRPO†（含 prompt replay buffer 但不含 BCQ/NCQ）。

### 4.2 16 VLM 基准完整结果（Tab. 1）

| Benchmark | Base | +GRPO† | +ZPPO | Δ |
|-----------|:----:|:------:|:-----:|:-:|
| AI2D | 65.6 | 71.2 | 76.5 | +5.3 |
| BabyV | 6.7 | 9.8 | 13.9 | +4.1 |
| CharXiv | 54.3 | 59.9 | 63.9 | +4.0 |
| DynaM | 17.8 | 23.6 | 31.1 | +7.5 |
| EmbSp | 67.9 | 69.4 | 71.5 | +2.1 |
| InfoVQA | 68.6 | 72.4 | 75.3 | +2.9 |
| MVerse | 43.5 | 51.1 | 59.3 | +8.2 |
| MVision | 16.4 | 20.9 | 29.2 | +8.3 |
| MVista | 60.7 | 68.3 | 73.2 | +4.9 |
| MMMUPro | 26.8 | 30.5 | 37.6 | +7.1 |
| MM-Vet | 53.2 | 57.5 | 59.9 | +2.4 |
| OCREN | 40.0 | 41.3 | 42.5 | +1.2 |
| OCRZH | 17.0 | 17.5 | 18.7 | +1.2 |
| VisP | 20.5 | 27.8 | 35.0 | +7.2 |
| VBlind | 42.8 | 43.6 | 44.7 | +1.1 |
| WeMath | 54.4 | 62.5 | 71.7 | +9.2 |
| **Avg** | **41.0** | **45.4** | **50.3** | **+4.9** |

**2B VLM**：Base 56.8 → GRPO† 59.2 → **ZPPO 62.0**（+2.8pp）。最大增益：BabyV +4.2pp、DynaM +6.8pp、MVision +7.1pp。

**规模趋势**：ZPPO 增益随规模递减——+9.3pp (0.8B) → +5.2pp (2B) → +4.0pp (4B) → +2.8pp (9B)。蒸馏增益始终有限（最佳 <+1pp）。

### 4.3 LLM 与 Video 基准（泛化分析）（Tab. 2）

**蒸馏损害泛化，ZPPO 改进泛化。**

| Method | 0.8B LLM | 0.8B Video | 2B LLM | 2B Video |
|--------|:--------:|:----------:|:------:|:--------:|
| Base | 25.2 | 48.3 | 45.3 | 60.6 |
| +Off/On-Distill† | 22.5–23.2 | 45.0–45.8 | 43.1–43.7 | 58.6–59.2 |
| +GRPO† | 28.7 | 50.5 | 47.3 | 61.9 |
| **+ZPPO** | **33.1** | **52.8** | **50.4** | **63.2** |

**蒸馏退化幅度**（LLM+Video 平均）：-2.5pp (0.8B), -1.8pp (2B), -0.9pp (4B), -0.3pp (9B)。

**ZPPO 改进幅度**：+6.8pp (0.8B), +4.3pp (2B), +2.7pp (4B), +2.7pp (9B)。

**单项亮点**：
- 0.8B GPQA-D: **+16.9pp**（ZPPO 42.4 vs GRPO† 25.5）
- 2B IMO-AB: **+10.2pp**（ZPPO 29.5 vs GRPO† 19.3）
- 0.8B MultiCh: **+7.8pp**（ZPPO 28.6 vs GRPO† 20.8）

### 4.4 组件消融（Tab. 3）

| 方法 | 0.8B VLM Avg | 2B VLM Avg |
|------|:-----------:|:----------:|
| GRPO | 43.8 | 58.7 |
| +GRPO† (buffer only) | 45.4 | 59.2 |
| +GRPO+Both (BCQ+NCQ, no buffer) | 45.2 | 58.9 |
| +GRPO†+BCQ | 48.6 | 60.8 |
| +GRPO†+NCQ | 46.2 | 60.1 |
| **+ZPPO (GRPO†+BCQ+NCQ)** | **50.3** | **62.0** |

**关键观察**：
- **Replay × Reformulation 是超可加的**：单独 replay 仅 +1.6pp（0.8B），单独 BCQ+NCQ 仅 +1.4pp，但三者组合 +6.5pp
- **BCQ 在小规模主导，NCQ 在大规模主导**：0.8B 时 BCQ 贡献（+3.2pp = 48.6−45.4）大于 NCQ（+0.8pp）；2B 时 NCQ 贡献增长

### 4.5 毕业动力学分析（Fig. 4）

| 准入时 rollout 准确率 | ZPPO 毕业率 | GRPO† 毕业率 |
|---------------------|:----------:|:------------:|
| **0%**（最困难问题） | **28%** (432/1568) | **4%** (73/2035) |
| 12.5% | **54%** | 14% |
| 25% | 76% | 49% |
| 37.5% | 93% | 82% |

在普通 RL 完全提供零梯度的最困难问题（0% accuracy）上，ZPPO 通过 BCQ/NCQ 的 prompt reformulation 恢复了 28% 的毕业率，而 GRPO† 仅有 4%。

### 4.6 RL 配方消融（Fig. 6）

**$I=4$ 是最优点**：$I=16$（GRPO 默认）产生 4 倍更新但增益有限；$I=1$ 消除漂移但训练不足。论文发现 $I=4$ 在小模型上是最优折中。

**Norm w/o Zero（排除零优势组）是关键**：包含零优势组会缩小 batch 标准差，放大其他优势值，导致训练不稳定。排除零优势组后，2B LLM avg 从 ~43 提升至 ~47。

---

## Ch5：局限性与延伸阅读

### 5.1 论文明确的局限性

**Teacher-bounded zone**：BCQ 需要教师成功解决 hard question。教师和学生同时失败时只有 NCQ 可用，而 NCQ 单独贡献有限（Tab. 3 中 NCQ-only vs BCQ-only）。扩展 ZPD 超越当前教师覆盖范围是最重要的开放问题。

**与 Dynamic Sampling 的张力**：ZPPO 存储 all-wrong group 给予第二次机会，DAPO 的 dynamic sampling 则丢弃它们。论文建议先 ZPPO reformulation 后 dynamic sampling 可能产生协同。

**Scope 限制**：仅覆盖单轮推理正确性，不覆盖多步推理、agentic 推理、鲁棒性、推理时效率、会话能力、多传感感知、多轮对话。

**继承偏差**：基于 Qwen3.5 预训练，未修改或过滤预训练数据中的偏差；奖励仅针对推理正确性。

### 5.2 延伸阅读

- [4] Qwen3.5 Team (2026) — 基座模型系列
- [5] DeepSeek-R1 (Guo et al., 2025) — 基础 RL 推理框架
- [8] DAPO (Yu et al., 2025) — clip-higher, token-level loss, dynamic sampling
- [36] GRPO (Shao et al., 2024) — 群组相对策略优化
- [37] REINFORCE++ (Hu, 2025) — 两步 batch-level advantage normalization
- [42] Vygotsky (1978) — Zone of Proximal Development
- [44] PPO (Schulman et al., 2017) — 近端策略优化

---

*报告基于 arXiv:2606.18216 原文（Tab. 1–3, Fig. 1–6），所有数字已与论文原文核对。*