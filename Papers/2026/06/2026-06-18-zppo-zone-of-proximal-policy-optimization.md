# Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients

## 论文元数据
- 标题：Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients
- 作者：Byung-Kwan Lee, Ximing Lu, Shizhe Diao, Minki Kang, Saurav Muralidharan, Karan Sapra, Andrew Tao, Pavlo Molchanov, Yejin Choi, Yu-Chiang Frank Wang, Ryo Hachiuma (NVIDIA)
- arXiv ID：2606.18216
- 提交日期：2026-06-17
- 官方代码：无公开仓库（NVIDIA Technical Patent Filed, 2026）

---

## Ch1：论文概述与核心贡献

### 问题背景

大型语言模型（数十亿参数）在benchmark上的性能增益无法有效迁移到移动设备、AR眼镜、机器人等资源受限场景的小型模型部署。核心挑战在于将大型教师模型的能力压缩到小一个数量级的学生模型。

现有两大方案均存在根本性缺陷：
- **知识蒸馏**：学生模仿教师logit分布 → 小容量学生过度集中在教师最尖锐的少数模式（mode-seeking）→ 泛化脆弱
- **强化学习（GRPO）**：所有rollout均失败时产生零优势信号 → hard question被静默丢弃

当将教师响应注入学生策略梯度时，违反on-policy假设，引起策略漂移。

### ZPPO核心思想

ZPPO将教师知识从梯度空间转移到prompt空间。教师响应被编码为prompt的一部分（BCQ/NCQ重新表述），学生生成全新rollout，策略梯度仅计算学生自身输出。这一设计同时避免了蒸馏的mode-seeking和RL的零优势陷阱。

### 三个核心组件

**BCQ（Binary Candidate-included Question）**：将一条教师正确响应和一条学生错误响应匿名配对，放入prompt让学生甄别。所有梯度来自学生新输出，保持on-policy。

**NCQ（Negative Candidate-included Question）**：聚合学生所有错误rollout到同一prompt中，暴露共享失败模式。

**Prompt Replay Buffer**：FIFO队列（容量10,000），准入r̄_x < 0.5，毕业r̄_x ≥ 0.5。

### 核心结果

- 学生：Qwen3.5 (0.8B-9B)，教师：Qwen3.5-27B，31 benchmarks
- 0.8B VLM：ZPPO 50.3 vs GRPO 43.8（+6.5pp），相对基模型+9.3pp
- 0.8B LLM：ZPPO 33.1 vs GRPO 27.1（+6.0pp）
- 0.8B LLM+Video泛化：ZPPO +6.8pp vs 蒸馏退化-2.5pp
- 最困难问题毕业率：ZPPO 28% vs GRPO 4%

---

## Ch2：研究背景与动机

### 知识蒸馏的共同缺陷

**Off-policy蒸馏**。学生通过KL散度模仿教师logit分布。当教师容量远大于学生时，教师logit包含学生无法表示的高维细节，学生被迫集中到少数高概率token（mode collapse），丧失多样性和泛化能力。

**On-policy蒸馏**。实时从当前教师采样响应构建蒸馏数据。根本问题不变：logit匹配信号使小容量学生只能模仿少数sharp模式。

**Self-distillation**。学生蒸馏自身checkpoints，仍依赖logit模仿，同样加剧mode collapse。

三者共同根源：当学生容量不足以表示目标分布时，优化必然收敛到少数高概率模式的低维投影。

### GRPO的零优势陷阱

GRPO计算group-relative advantage：A^(g) = (r(x, y_S^(g)) - r̄_x) / (std_x + ε)。所有rollout失败时r̄_x=0 → 所有A^(g)=0 → 零梯度信号。这些hard question（r̄_x < 0.5）被静默丢弃。无法通过增加采样次数解决，因为更多rollout只会让r̄_x更稳定地接近0。

### Hybrid方案的问题

将教师响应混入rollout参与梯度计算违反on-policy假设。教师响应来自π_teacher而非π_student，梯度估计出现bias，长期训练引起策略漂移。

### ZPPO的定位

与Hybrid方案的根本差异：教师知识载体是prompt而非梯度。ZPPO的策略梯度仅涉及学生自身rollout，教师响应仅出现在prompt条件中，保持on-policy的无偏性。

---

## Ch3：ZPPO核心方法

### 3.1 GRPO的失败模式

**符号定义**。问题x，响应y ~ π_θ(·|x)，G个rollout，outcome reward r∈{0,1}，group mean r̄_x，标准差std_x。

**Group-relative advantage**：
```
A^(g) = (r(x, y_S^(g)) - r̄_x) / (std_x + ε)
```

**零优势陷阱**。当全部错误（r̄_x=0）或全部正确（r̄_x=1）时，所有A^(g)=0 → 零梯度。Hard question定义：r̄_x < 0.5。std在0.5处达到最大值（二元分布方差最大点），携带最强学习信号的区域反而是GRPO的盲区。

### 3.2 BCQ（Binary Candidate-included Question）

**候选压缩**。教师将正确/错误轨迹重写为短推理过程，使用共享压缩prompt和512 token上限，消除长度和格式线索。

**BCQ构造**：
1. 从教师rollout中均匀采样一条正确响应y_T^(+)，从学生错误rollout中采样一条学生错误响应y_S^(-)
2. 教师压缩两条轨迹
3. 匿名化放入<candidate>标签，随机打乱顺序
4. 附加指令："One is correct and another is wrong"
5. 学生从BCQ prompt采样全新rollout

**On-policy保持**。BCQ prompt中的教师文本是输入的一部分，不是梯度目标。学生所有生成token来自当前策略π_θ，梯度条件期望与on-policy要求一致。

### 3.3 NCQ（Negative Candidate-included Question）

**NCQ构造**：
1. 收集当前group所有错误rollout，解析最终答案
2. 在prompt中列出："The following answers are all WRONG: {parsed answer list}"
3. 每个教师压缩的错误轨迹作为<candidate>块附加
4. 学生采样全新rollout

**结构性差异**。在独立rollout group中，每个错误rollout的梯度独立计算，学生无法跨rollout识别错误模式。NCQ是第一个使独立失败变为集体可见信号的地方。

### 3.4 Prompt Replay Buffer

**存储**。仅问题x（图像+文本），不存储rollout。教师rollout每次访问重新采样，BCQ候选每次变化。

**准入**。r̄_x < 0.5。**毕业**。r̄_x ≥ 0.5。**容量**。10,000，FIFO淘汰。

**采样**。新问题 + replay样本（ρ_replay=0.25）。BCQ+NCQ上限ρ_aug=0.25。最困难问题优先。

---

## Ch4：实验结果与分析

### 4.1 实验设置

**模型**。学生Qwen3.5 (0.8B/2B/4B/9B)，教师Qwen3.5-27B（冻结）。训练数据：ZPPO-77K（77K image-question pairs多模态RL数据集）。

**训练管线**。基于GRPO + DAPO改进（clip-higher ε_low=0.20, ε_high=0.28, token-level loss, 无KL惩罚）。每步I=4次迭代。Batch-level advantage normalization排除零优势组（Norm w/o Zero）。

**基线**。Off-policy蒸馏、On-policy蒸馏、GRPO、GRPO†（+prompt replay buffer）。

**评估**。31 benchmarks：16 VLM（AI2D, BabyV, CharXiv, DynaM, EmbSp, InfoVQA, MVerse, MVision, MVista, MMMUPro, MM-Vet, OCREN, OCRZH, VisP, VBlind, WeMath）、10 LLM（AIME25, AIME26, CEval, GPQA-D, HLE, IMO-AB, MMLU, MMLU-Pro, MMLU-Rd, MultiCh）、5 Video（MMVU, MVBench, VMME, VMMES, VMMMU）。

### 4.2 VLM基准结果

**0.8B VLM详细结果**：

| Benchmark | Base | GRPO† | ZPPO | Δ |
|-----------|------|-------|------|---|
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

**2B VLM**：ZPPO 62.0 vs GRPO† 59.2 vs Base 56.8。ZPPO在所有16个benchmark取得列最优，最大增益DynaM +6.8pp、MVision +7.1pp。

**规模趋势**：ZPPO增益随规模递减——+9.3pp (0.8B) → +5.2pp (2B) → +4.0pp (4B) → +2.8pp (9B)。蒸馏VLM增益始终有限（最佳+0.9pp）。

### 4.3 LLM与Video基准（泛化分析）

**关键发现**：蒸馏损害泛化，ZPPO改进泛化。

| 方法 | 0.8B LLM | 0.8B Video | 2B LLM | 2B Video |
|------|---------|-----------|-------|---------|
| Base | 25.2 | 48.3 | 45.3 | 60.6 |
| GRPO† | 28.7 | 50.5 | 47.3 | 61.9 |
| ZPPO | 33.1 | 52.8 | 50.4 | 63.2 |

**蒸馏退化**（LLM+Video平均）：-2.5pp (0.8B), -1.8pp (2B), -0.9pp (4B), -0.3pp (9B)。

**ZPPO改进**：+6.8pp (0.8B), +4.3pp (2B), +2.7pp (4B), +2.7pp (9B)。

**单项亮点**：0.8B GPQA-D +16.9pp，2B IMO-AB +10.2pp。

### 4.4 组件消融

| 组件 | 0.8B VLM | 2B VLM |
|------|---------|-------|
| GRPO | 43.8 | 58.7 |
| GRPO† (+buffer) | 45.4 | 59.2 |
| GRPO+Both (BCQ+NCQ) | 45.2 | 58.9 |
| GRPO†+BCQ | 48.6 | 60.8 |
| GRPO†+NCQ | 46.2 | 60.1 |
| ZPPO | 50.3 | 62.0 |

**超可加性**。Replay单独、Reformulation单独均有限，但Replay × Reformulation产生超可加增益。BCQ贡献随规模递减，NCQ随规模递增。

### 4.5 毕业动力学

0% entry accuracy：ZPPO毕业28% (432/1568) vs GRPO 4% (73/2035)。Next-hardest (12.5%)：54% vs 14%。BCQ eligibility随学生规模增大而缩小（教师成功比例下降），NCQ在所有规模保持有效。

### 4.6 RL配方消融

**I=4是最优点**。I=16（GRPO默认）产生4倍更新但增益有限；I=1消除漂移但训练不足。**Norm w/o Zero**（排除零优势组）是关键：包含零优势组会缩小batch标准差，放大其他优势值。

---

## Ch5：局限性与延伸阅读

### 局限性

**Teacher-bounded zone**。BCQ需要教师成功解决hard question。教师和学生同时失败时只有NCQ，而NCQ单独贡献有限。扩展zone超越当前教师覆盖范围是最重要的开放问题。

**与Dynamic Sampling的张力**。ZPPO存储all-wrong组给予第二次机会，DAPO的dynamic sampling则删除它们。论文建议先ZPPO reformulation后dynamic sampling。

**Scope限制**。仅单轮推理正确性，不覆盖多步/agentic推理、鲁棒性、推理时效率、会话能力、多传感感知、多轮对话。

**继承偏差**。基于Qwen3.5预训练，未修改或过滤预训练数据中的偏差。奖励仅针对推理正确性。

### 延伸阅读

- [1] DeepSeek-R1 (Guo et al., 2025) — 基础RL推理框架
- [4] Qwen3.5 Team (2026) — 基座模型系列
- [8] DAPO (Yu et al., 2025) — clip-higher, token-level loss, dynamic sampling
- [36] GRPO (Shao et al., 2024) — 群组相对策略优化
- [42] Vygotsky (1978) — Zone of Proximal Development
- [37] REINFORCE++ (Hu, 2025) — 两步batch-level advantage normalization
- [18] Knowledge Distillation (Hinton et al., 2015) — 经典logit匹配蒸馏