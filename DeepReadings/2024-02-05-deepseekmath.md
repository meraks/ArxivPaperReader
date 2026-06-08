# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

> **论文信息**
> - **标题**：DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
> - **作者**：Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo
> - **机构**：DeepSeek-AI, Tsinghua University, Peking University
> - **arXiv**：[2402.03300](https://arxiv.org/abs/2402.03300) (v3, Apr 27 2024)
> - **官方代码**：[deepseek-ai/DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math)
> - **模型**：🤗 [deepseek-math-7b-base](https://huggingface.co/deepseek-ai/deepseek-math-7b-base) | [deepseek-math-7b-instruct](https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct) | [deepseek-math-7b-rl](https://huggingface.co/deepseek-ai/deepseek-math-7b-rl)

---

## Ch1. 论文概述与核心贡献

### 1.1 一句话总结

DeepSeekMath 证明了**数据质量 + 算法创新**可以让 7B 开源模型在数学推理上逼近 GPT-4 级别，核心手段是：（1）从 Common Crawl 中迭代挖掘 120B tokens 高质量数学语料；（2）提出 GRPO 算法，用免 critic 的组相对策略优化替代 PPO。

### 1.2 问题定位

数学推理一直是大语言模型的"硬骨头"。2024年初的状态：

- **闭源模型**（GPT-4, Gemini Ultra）遥遥领先，MATH ~50%+
- **开源模型**（Mistral 7B, LLaMA-2）挣扎在 MATH ~10-15%
- **Scaling 派**（Minerva 540B）靠模型大小硬堆，MATH 33.6%，但不开源、不可复现
- **数据派**（Llemma 34B）用 Proof-Pile-2 专训练，MATH 25.3%

核心困境：**如何在不堆参数的前提下，用公开数据+高效RL达到闭源水平？**

### 1.3 核心贡献

| # | 贡献 | 级别 |
|---|------|------|
| 1 | **DeepSeekMath Corpus**：从 Common Crawl 迭代挖掘 120B tokens（35.5M页）数学数据 | 工程范式 |
| 2 | **GRPO 算法**：免 critic 的 PPO 变体，组均值作为 baseline，大幅省显存 | 算法创新 |
| 3 | **Code→Math 迁移**：证明代码预训练对数学有显著正迁移 | 洞察 |
| 4 | **arXiv 无效发现**：arXiv 论文数据不能提升数学基准 | 反直觉发现 |
| 5 | **统一 RL 范式**：SFT/RFT/DPO/PPO/GRPO 统一为三要素框架 | 理论贡献 |

---

## Ch2. 研究背景与动机

### 2.1 数学推理的独特挑战

> **类比理解**：让LLM做数学题，相当于要求一个只学过语文的人参加数学竞赛——它需要同时掌握"理解题目"（语言能力）、"逻辑推理"（符号操作）、"计算验证"（数值精度）三个维度。传统预训练只覆盖第一个维度。

数学推理的三个阶段：

```
题面理解 → 符号推理 → 答案验证
   │            │            │
   ↓            ↓            ↓
语言能力     逻辑链      计算精度
(预训练已有) (需要专项训练) (需要工具辅助)
```

### 2.2 当时的技术路线对比

| 路线 | 代表工作 | 核心手段 | MATH性能 | 局限 |
|------|---------|---------|---------|------|
| **Scaling** | Minerva 540B | 堆参数+专用数据 | 33.6% | 不公开，推理成本高 |
| **专用预训练** | Llemma 34B | Proof-Pile-2 | 25.3% | 数据质量不够 |
| **工具增强** | ToRA 7B | PoT+TIR | ~40% | 依赖外部工具 |
| **RL微调** | WizardMath 7B | 强化学习 | ~30% | RL方法粗糙 |

DeepSeekMath 的定位：**同时优化数据质量（Pipeline）+ 算法效率（GRPO）+ 训练路径（Code→Math）**。

### 2.3 关键假设

论文背后有三个贯穿全文的假设：

1. **数据质量假设**：Common Crawl 中隐藏着远超现有专用数据集的高质量数学内容，关键是**怎么捞出来**
2. **迁移假设**：代码训练学到的结构化推理能力（变量操作、条件分支、形式化语法）可以正向迁移到数学推理
3. **RL效率假设**：PPO 的 value network 对数学RL不是必需的——组内相对比较足以提供有效的训练信号

---

## Ch3. DeepSeekMath Corpus — 数据收集Pipeline

### 3.1 整体架构

这是论文的第一个核心贡献：一个**迭代式、自举式**的数据挖掘流水线。

```
Common Crawl (40B HTML pages)
         │
         ▼
    ┌─────────┐
    │ Seed    │ ← OpenWebMath (13.6B tokens, 高质量数学网页)
    │ Corpus  │
    └────┬────┘
         │ 采样500K positive + 500K negative
         ▼
    ┌──────────┐
    │ fastText  │ ← 训练二分类器 (math vs non-math)
    │ Classifier│
    └────┬─────┘
         │ 对40B页面打分+排序
         ▼
    ┌──────────┐
    │ Recall   │ ← 取 top-K，保留数学相关页面
    │ + Filter │
    └────┬─────┘
         │
         ▼
    ┌──────────────┐
    │ Domain       │ ← 识别 >10%页面为数学的域名
    │ Discovery    │    人工标注额外数学URL路径
    └────┬─────────┘
         │ 添加新发现的页面到 seed
         ▼
    ┌──────────┐
    │ Re-train  │ ← 更新 fastText，回到 Step 2
    │ fastText  │
    └──────────┘
         │
         ▼  (4轮迭代，35.5M pages, 120B tokens)
    DeepSeekMath Corpus
```

### 3.2 为什么是迭代式？

> **类比理解**：这像是在淘金——第一轮你知道金子长什么样（OpenWebMath），用这个标准去河里筛（Common Crawl）。筛着筛着，你发现新矿脉（数学域名），于是扩大搜索范围。再筛一轮，又发现新矿脉。四轮之后，你能找到的金子基本都找到了。

单轮 fastText 的局限：
- 只能找到与 seed 分布相似的页面
- 会漏掉呈现形式不同的数学内容（如中文数学论坛、特定领域的数学博客）
- Domain discovery 步骤是解决这个问题的关键——让人工标注者发现"新类型"的数学内容

### 3.3 四轮迭代数据

| 迭代轮次 | 新增页面 | 累计页面 | 与上轮重叠 |
|---------|---------|---------|-----------|
| Round 1 | ~8M | 8M | - |
| Round 2 | ~12M | 20M | ~60% |
| Round 3 | ~10M | 30M | ~85% |
| Round 4 | ~5.5M | 35.5M | **98%** |

第4轮与第3轮有98%重叠 → **信号饱和，停止迭代**。这是一个工程上非常严谨的停止条件。

### 3.4 去污染（Decontamination）

**这是模型评估诚信的关键一步**。论文对所有 benchmark 进行了严格的 n-gram 去重：

| Benchmark | 策略 |
|-----------|------|
| GSM8K, MATH, CMATH, AGIEval | 10-gram 精确匹配 → 移除整页 |
| 短文本基准（<10 words） | 3-gram 及以上精确匹配 → 移除整页 |

> **关键洞察**：不做去污染会导致 benchmark 性能虚高。2024年初多个开源数学模型的性能被质疑存在评估泄露问题。DeepSeekMath 的10-gram去重标准是当时最严格的。

### 3.5 语料质量验证

论文用 1.3B 模型在 150B tokens 上做了受控实验，比较不同语料：

| 语料 | 大小 | GSM8K | MATH | 中文CMATH |
|------|------|-------|------|----------|
| No math training | - | 2.9% | 3.0% | 12.3% |
| MathPile (mostly arXiv) | 8.9B | 2.7% | 3.3% | 1.2% |
| OpenWebMath | 13.6B | 11.5% | 8.9% | 16.8% |
| Proof-Pile-2 | 51.9B | 14.3% | 11.2% | 19.9% |
| **DeepSeekMath Corpus** | **120.2B** | **23.8%** | **13.6%** | **41.5%** |

**三个关键发现**：

1. **质量 > 大小**：DeepSeekMath Corpus 即使只消耗与 Proof-Pile-2 等量的 tokens，仍然显著更好——因为数据质量更高
2. **arXiv论文无效**：MathPile（主要是arXiv论文）几乎不提升数学能力（GSM8K 2.7% vs 无训练的2.9%）——这是**反直觉发现**，说明论文 ≠ 推理训练数据
3. **多语言优势**：DeepSeekMath Corpus 同时提升中英文数学，而英文专用语料可能损害中文性能

---

## Ch4. 预训练 — DeepSeekMath-Base

### 4.1 训练配置

| 配置项 | 值 |
|--------|-----|
| **初始化** | DeepSeek-Coder-Base-v1.5 7B |
| **总训练tokens** | 500B |
| **上下文长度** | 4096 |
| **优化器** | AdamW (β1=0.9, β2=0.95, weight_decay=0.1) |
| **峰值学习率** | 4.2×10⁻⁴ |
| **Batch size** | 10M tokens |

### 4.2 数据混合

```
DeepSeekMath Corpus  56% ██████████████████████████████████████████████████████
GitHub Code          20% ████████████████████
arXiv                10% ██████████
Natural Language     10% ██████████
AlgebraicStack        4% ████
```

**设计意图**：
- **代码+NL维持**：防止灾难性遗忘，保持通用能力和代码能力
- **arXiv虽无效但保留**：论文发现 arXiv 不能提升数学基准，但保留10%以维持学术语言的覆盖——这是一个实用主义的设计
- **56%数学**：远高于 Llemma 的数学占比，数据质量使高比例训练可行

### 4.3 基准模型性能（无工具，CoT评估）

| 模型 | 参数 | GSM8K | MATH |
|------|------|-------|------|
| GPT-4 (2024.01) | 未公开 | 92.0% | 52.9% |
| Gemini Ultra | 未公开 | 94.4% | 53.2% |
| Minerva | 540B | 58.8% | 33.6% |
| Llemma | 34B | 54.0% | 25.3% |
| Mistral 7B | 7B | 40.3% | 14.3% |
| **DeepSeekMath-Base** | **7B** | **64.2%** | **36.2%** |

> **震撼点**：7B 模型在 GSM8K 和 MATH 上全面超越 540B 的 Minerva（**77倍参数差距**）。这证明了"数据质量 × 训练路径"可以系统性补偿模型规模。

### 4.4 Code→Math 迁移实验

论文设计了精巧的受控实验来验证代码预训练的增益：

**实验设置**：1.3B 模型，两种训练路径：
- A. 通用预训练 400B → 数学 150B
- B. 代码预训练 400B → 数学 150B

| 路径 | GSM8K | MATH | MMLU-STEM |
|------|-------|------|-----------|
| A: General→Math | ~16% | ~10% | ~26% |
| **B: Code→Math** | **~24%** | **~14%** | **~33%** |

**而且代码训练的收益对工具使用场景同样存在**——这意味着代码训练不仅教会了模型"结构化思考"，还教会了"与工具交互"的潜在技能。

### 4.5 通用能力保持

| 能力维度 | 基准 | DeepSeekMath-Base | DeepSeek-Coder-Base (前身) |
|---------|------|-------------------|--------------------------|
| 语言理解 | MMLU | 54.9% | - |
| 推理 | BBH | 59.5% | - |
| 代码生成 | HumanEval | 40.9% | ~45% |
| 代码生成 | MBPP | 52.6% | ~55% |

数学训练后代码能力轻微下降（约4-5pp），但通用推理能力（BBH 59.5%）和语言理解（MMLU 54.9%）不仅没降，还超过了前身模型——**数学训练产生了正向的通用推理迁移**。

---

## Ch5. GRPO — Group Relative Policy Optimization

### 5.1 动机：PPO 的显存之痛

这是论文最著名的贡献。PPO（Proximal Policy Optimization）训练LLM时需要维护4个模型：

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Policy Model │  │ Reference    │  │ Value Model  │  │ Reward Model │
│   (训练中)    │  │  Model       │  │  (Critic)    │  │              │
│   ~7B × 2    │  │   ~7B × 2    │  │   ~7B × 2    │  │   ~7B × 2    │
│  (optimizer) │  │              │  │  (optimizer) │  │              │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
      ~14GB            ~14GB            ~14GB            ~14GB
                              ↓
                    总计约 56GB 显存（7B模型）
```

> **类比理解**：PPO训练像同时养4只7B参数的大象——每只大象都需要单独的笼子（显存）和饲养员（optimizer）。GRPO 的洞见是：Critic 这头大象其实可以用一个简单的计数器（组均值）替代。

### 5.2 GRPO 核心思想

对每个问题 q，采样 k 个输出 {o₁, ..., o_k}，计算各自的奖励 {r₁, ..., r_k}，然后用**组内相对排名**定义 advantage：

```
A_i = (r_i - mean(r)) / std(r)
```

**关键**：不需要 value function 来估计 baseline——组均值就是 baseline。

> **直觉**：对于同一个问题，如果你生成的4个答案中，有3个都是垃圾（奖励低），只有1个是好的（奖励高），那这个好答案的 advantage 就是正的——即使绝对奖励值不高。这就是"相对优势"的核心。

### 5.3 数学形式

**目标函数**：

$$J_{GRPO}(\theta) = \mathbb{E} \left[ \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} \cdot A_i, \;\; \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) \cdot A_i\right) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref}) \right]$$

其中 advantage：

$$A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_k\})}{\text{std}(\{r_1, ..., r_k\})}$$

**KL 散度估计**（Schulman 无偏估计器）：

$$D_{KL}(\pi_\theta \| \pi_{ref}) = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - \log\frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - 1$$

### 5.4 GRPO vs PPO 对比

| 维度 | PPO | GRPO | 改进 |
|------|-----|------|------|
| 模型数量 | 4 (policy+ref+value+reward) | 2 (policy+ref) | -50% |
| 显存消耗 | ~56GB (7B) | ~28GB (7B) | -50% |
| 训练速度 | 基准 | 约1.5-2× | 更快 |
| Advantage计算 | GAE (需要value) | 组内标准化 | 无偏+简单 |
| 奖励模型 | 需要 | 需要 | 相同 |
| KL约束 | PPO-clip + KL penalty | PPO-clip + Schulman KL | 更精确的KL估计 |

### 5.5 GRPO 实验效果

| 模型 | 方法 | GSM8K | MATH | CMATH | Gaokao-Math |
|------|------|-------|------|-------|------------|
| DeepSeekMath-Base | - | 64.2% | 36.2% | 70.3% | 23.2% |
| DeepSeekMath-Instruct | SFT | 82.9% | 46.8% | 81.2% | 42.1% |
| DeepSeekMath-RL | SFT+GRPO | **88.2%** | **51.7%** | **84.6%** | **45.8%** |
| + Self-Consistency (64) | SFT+GRPO | **91.7%** | **60.9%** | - | - |

> **GRPO 增量**（vs Instruct）：
> - GSM8K: +5.3pp
> - MATH: +4.9pp
> - CMATH: +3.4pp
> - Gaokao-Math: +3.7pp

而且这些增益仅使用了**指令微调数据的一个子集**（7.8K Math Instruct 子集，源自 GSM8K 和 MATH 的训练集），没有使用任何新的问题提示词。

### 5.6 域外泛化

GRPO 不仅在训练数据的分布上有效，还泛化到了**域外任务**：

> GRPO 使用 GSM8K+MATH 训练数据 → CMATH（中文数学）和 Gaokao-Math（高考数学）同样提升

这说明 GRPO 学习到的是**通用的推理能力改进**，而非对特定数据分布的过拟合。

---

## Ch6. 统一 RL 范式

### 6.1 三要素框架

论文的另一个理论贡献是将所有主流对齐方法统一为一个三要素框架：

$$
\mathcal{J} = \mathbb{E} \left[ \sum_t c_t \cdot \log \pi_\theta(a_t|s_t) \right]
$$

三个要素：

| 要素 | 符号 | 含义 |
|------|------|------|
| **数据源** | 来自哪个分布采样 | online rollout vs offline dataset |
| **奖励函数** | r(s,a) | 定义什么是"好" |
| **梯度系数** | c_t | 每个token的权重/缩放 |

### 6.2 方法的统一视图

| 方法 | 数据源 | 奖励函数 | 梯度系数 |
|------|--------|---------|---------|
| **SFT** | 离线数据集 | 1 (好)/ 0 (坏) | 1 |
| **RFT** | 离线（模型自己生成） | 答案正确性 | 1 |
| **Online RFT** | Online rollout | 答案正确性 | 1 |
| **DPO** | 离线偏好对 | 隐式（通过偏好对） | P_ref / P_θ |
| **PPO** | Online rollout | RM打分 | A_t (GAE) |
| **GRPO** | Online rollout | RM打分 | A_i (组内标准化) |

> **核心洞察**：所有这些方法的区别仅在于这三个要素的选择。SFT 极端简单（固定权重1），PPO 复杂（需要价值函数估计 advantage），GRPO 在中间找到了工程最优解（用组统计量替代价值函数）。

---

## Ch7. 实验结果与分析

### 7.1 RL 提升的本质：Maj@K vs Pass@K

论文中最重要的分析之一：

| 指标 | 定义 | RL的影响 |
|------|------|---------|
| **Pass@K** | K次采样中**任意一次**通过的概率 | RL 几乎不提升 |
| **Maj@K** | K次采样中**多数投票**的准确率 | **RL 显著提升** |

> **深层含义**：RL（GRPO）并不让模型"更聪明"——它不提升模型的基础推理能力上限（Pass@K不变）。RL 做的是让正确答案在输出分布中**更集中地出现**（Maj@K提升），即提升输出分布的**鲁棒性**。

```
RL之前：采样100次 → 正确答案出现了20次但散布在噪声中 → Maj@100可能不对
RL之后：采样100次 → 正确答案出现了20次但更集中 → Maj@100更容易正确
```

> **类比理解**：RL不是让枪法更准（Pass@K不变），而是让靶子更大（正确答案的密度更高）。

### 7.2 迭代RL

论文探索了**迭代 GRPO**：

```
Round 1: 用初始 RM 评分 → GRPO 训练 → 得到 Model v1
Round 2: 用 Model v1 作为新 RM → GRPO 训练 → 得到 Model v2
```

结果：性能持续提升，说明"用训练中的模型自举评分"是可行的——这是后来 DeepSeek-R1 的"自我进化"思想的前身。

### 7.3 工具辅助推理

| 配置 | GSM8K | MATH |
|------|-------|------|
| DeepSeekMath-Base (CoT) | 64.2% | 36.2% |
| DeepSeekMath-Base (PoT) | 66.9% | 31.4% |
| DeepSeekMath-RL (CoT) | 88.2% | 51.7% |
| DeepSeekMath-RL (PoT) | 89.5% | 58.8% |

**工具在低能力模型上可能降级 MATH**（36.2% → 31.4%，因为PoT需要将数学问题翻译为代码，这是一个额外的能力要求）。但在 RL 后的高能力模型上，工具显著增益（51.7% → 58.8%）。这说明**工具使用能力本身也需要足够的基础推理能力做支撑**。

### 7.4 形式化定理证明

在 miniF2F 上评估（Isabelle 形式化证明）：

| 指标 | DeepSeekMath-Base 7B |
|------|---------------------|
| miniF2F-valid | 25.8% |
| miniF2F-test | 24.6% |

一个仅经过非形式化数学训练的7B模型，在few-shot下能完成1/4的形式化定理证明——这暗示了非形式化与形式化推理之间存在隐式的技能迁移。

---

## Ch8. 代码实现

### 8.1 DeepSeekMath Corpus 的 fastText Pipeline

```python
# ⚠️ 概念实现 — 基于论文 Section 2.1 描述
# 来源：arxiv.org/abs/2402.03300

import fasttext
import numpy as np
from pathlib import Path

class IterativeMathCollector:
    """
    迭代式数学数据收集器
    
    参数:
        seed_path: 初始种子数据 (OpenWebMath路径)
        cc_path: Common Crawl 存储路径
        max_iterations: 最大迭代次数 (论文用4)
        overlap_threshold: 饱和阈值 (论文用98%)
    """
    
    def __init__(self, seed_path, cc_path, max_iterations=4, overlap_threshold=0.98):
        self.seed = self._load_seed(seed_path)
        self.cc_path = cc_path
        self.max_iterations = max_iterations
        self.overlap_threshold = overlap_threshold
        self.collected_pages = set()
    
    def _train_classifier(self, positive_pages, negative_pages):
        """训练 fastText 二分类器"""
        # 写入训练数据
        with open('/tmp/fasttext_train.txt', 'w') as f:
            for page in positive_pages[:500000]:  # 论文用500K
                f.write(f'__label__math {page}\n')
            for page in negative_pages[:500000]:
                f.write(f'__label__nonmath {page}\n')
        
        model = fasttext.train_supervised(
            input='/tmp/fasttext_train.txt',
            epoch=5,
            lr=1.0,
            wordNgrams=2,
            dim=100
        )
        return model
    
    def _score_and_rank(self, classifier, pages):
        """对页面打分并排序，取top-K"""
        scores = []
        for page in pages:
            # fastText predict返回 (label, probability)
            label, prob = classifier.predict(page.strip(), k=1)
            if label[0] == '__label__math':
                scores.append((page, prob[0]))
        
        # 按概率排序，取高分页面
        scores.sort(key=lambda x: x[1], reverse=True)
        return [p for p, s in scores]
    
    def _discover_domains(self, new_pages):
        """识别数学集中域名，触发人工标注"""
        domain_counts = {}
        for page in new_pages:
            domain = self._extract_domain(page)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # 识别 >10%页面为数学的域名
        math_domains = {
            d for d, c in domain_counts.items()
            if c / len(new_pages) > 0.10
        }
        # TODO: 人工标注math_domains下的额外数学URL
        return math_domains
    
    def run(self):
        """主循环"""
        prev_pages = set()
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n=== Iteration {iteration} ===")
            
            # Step 1: 训练分类器
            negatives = self._sample_negative(self.cc_path, len(self.seed))
            classifier = self._train_classifier(self.seed, negatives)
            
            # Step 2: 召回
            all_pages = self._load_cc_pages(self.cc_path)  # 40B pages
            scored = self._score_and_rank(classifier, all_pages)
            
            # Step 3: 去重 + 去污染
            new_pages = []
            for page in scored:
                if not self._is_contaminated(page):  # 10-gram去重
                    new_pages.append(page)
            
            # Step 4: 发现新域名
            math_domains = self._discover_domains(new_pages)
            print(f"  Discovered {len(math_domains)} math domains")
            
            # Step 5: 计算重叠
            overlap = len(set(new_pages) & prev_pages) / len(new_pages)
            print(f"  New pages: {len(new_pages)}, Overlap: {overlap:.1%}")
            
            # Step 6: 更新种子+检查饱和
            self.seed.extend(new_pages)
            prev_pages = set(new_pages)
            
            if overlap >= self.overlap_threshold:
                print(f"  Converged! {overlap:.1%} overlap with previous round")
                break
        
        return self.seed  # 35.5M pages, 120B tokens
```

### 8.2 GRPO 算法实现

```python
# ⚠️ 概念实现 — 基于论文 Section 4 的 GRPO 公式
# 来源：arxiv.org/abs/2402.03300

import torch
import torch.nn.functional as F

def grpo_loss(
    policy_model,
    reference_model,
    input_ids,
    attention_mask,
    rewards,        # shape: (batch_size, k) — k次采样的奖励
    epsilon=0.2,    # PPO clip range
    beta=0.04,      # KL penalty coefficient
):
    """
    Group Relative Policy Optimization 损失函数
    
    参数:
        policy_model: 当前策略模型
        reference_model: 冻结的参考模型 (init = SFT模型)
        input_ids: 输入序列 (batch_size, k, seq_len)
        attention_mask: 注意力掩码
        rewards: 每个输出的奖励 (batch_size, k)
        epsilon: PPO-clip范围
        beta: KL散度系数
    
    返回:
        loss: GRPO标量损失
    """
    B, K, L = input_ids.shape  # batch, group_size, seq_len
    
    # Step 1: 计算 advantages (组内标准化)
    # A_i = (r_i - mean(r)) / std(r)
    reward_mean = rewards.mean(dim=1, keepdim=True)   # (B, 1)
    reward_std = rewards.std(dim=1, keepdim=True)     # (B, 1)
    advantages = (rewards - reward_mean) / (reward_std + 1e-8)  # (B, K)
    
    # Step 2: 获取 log-probabilities
    # 展平为 (B*K, L)
    flat_input = input_ids.view(B * K, L)
    flat_mask = attention_mask.view(B * K, L)
    
    # Policy log-prob
    policy_logits = policy_model(flat_input, flat_mask).logits  # (B*K, L, V)
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    # 取每个token的实际词的概率
    per_token_log_probs = torch.gather(
        policy_log_probs, dim=-1,
        index=flat_input.unsqueeze(-1)
    ).squeeze(-1)  # (B*K, L)
    # 按序列求和得到 log P(o|q)
    seq_log_probs = (per_token_log_probs * flat_mask).sum(dim=-1)  # (B*K)
    
    # Reference log-prob
    with torch.no_grad():
        ref_logits = reference_model(flat_input, flat_mask).logits
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_per_token = torch.gather(
            ref_log_probs, dim=-1,
            index=flat_input.unsqueeze(-1)
        ).squeeze(-1)
        ref_seq_log_probs = (ref_per_token * flat_mask).sum(dim=-1)  # (B*K)
    
    # Step 3: 计算 ratio = π_θ(o|q) / π_old(o|q)
    # π_old 是上一次迭代的policy（用reference_model近似）
    ratios = torch.exp(seq_log_probs - ref_seq_log_probs.detach())  # (B*K)
    ratios = ratios.view(B, K)  # (B, K)
    
    # Step 4: PPO-clip 目标
    # min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
    ppo_objective = torch.min(
        ratios * advantages,
        clipped_ratios * advantages
    )  # (B, K)
    
    # Step 5: KL 散度（Schulman无偏估计器）
    # KL[π_θ||π_ref] = π_ref/π_θ - log(π_ref/π_θ) - 1
    ref_over_policy = ref_seq_log_probs.view(B, K) - seq_log_probs.view(B, K)
    kl_penalty = torch.exp(ref_over_policy) - ref_over_policy - 1  # (B, K)
    
    # Step 6: 总损失
    loss = -(ppo_objective - beta * kl_penalty).mean()
    
    return loss


def grpo_training_step(policy, ref, optimizer, batch, k=64):
    """
    单步GRPO训练
    
    参数:
        k: 每个问题的采样数（论文典型值: 64）
    """
    questions = batch['questions']  # (B,)
    
    # Step 1: 对每个问题采样k个输出
    all_outputs = []
    all_rewards = []
    for q in questions:
        outputs = policy.generate(q, num_return_sequences=k)  # (k, seq_len)
        rewards = compute_math_reward(q, outputs)              # (k,)
        all_outputs.append(outputs)
        all_rewards.append(rewards)
    
    outputs = torch.stack(all_outputs)  # (B, k, seq_len)
    rewards = torch.stack(all_rewards)  # (B, k)
    
    # Step 2: 计算GRPO损失
    optimizer.zero_grad()
    loss = grpo_loss(
        policy_model=policy,
        reference_model=ref,
        input_ids=outputs,
        attention_mask=(outputs != pad_token_id).float(),
        rewards=rewards,
    )
    
    # Step 3: 反向传播
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### 8.3 奖励函数设计

```python
def compute_math_reward(question, outputs):
    """
    数学推理的奖励函数
    
    论文使用基于规则的奖励（非学习式的RM），因为：
    1. 数学问题有"客观正确答案"
    2. 避免RM过拟合或奖励hacking
    """
    rewards = []
    for output in outputs:
        # 提取最终答案（\boxed{}中的内容）
        answer = extract_boxed_answer(output)
        ground_truth = get_ground_truth(question)
        
        if answer == ground_truth:
            # 正确答案: 奖励 = 1
            # 论文未使用格式奖励，纯结果导向
            reward = 1.0
        else:
            reward = 0.0
        
        rewards.append(reward)
    
    return torch.tensor(rewards)


def extract_boxed_answer(text):
    """从模型输出中提取 \\boxed{} 内的答案"""
    import re
    # 匹配 \boxed{answer}
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    # 降级：取最后一行
    return text.strip().split('\n')[-1].strip()
```

### 8.4 使用官方模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载 RL 模型
model_name = "deepseek-ai/deepseek-math-7b-rl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# CoT prompt (论文推荐的提示模板)
question = "Find the sum of all integer solutions to |x-3| + |x+2| = 7."
prompt = f"{question}\nPlease reason step by step, "
prompt += "and put your final answer within \\boxed{{}}.\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.0,  # greedy decoding (论文用greedy)
    do_sample=False,
)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

---

## Ch9. 局限性与延伸阅读

### 9.1 论文局限性

1. **GRPO 仅验证于数学领域**：论文仅在数学推理任务上验证了 GRPO。虽然域外泛化（英文→中文数学）效果好，但 GRPO 在对话、代码、创意写作等任务上的适用性未经验证。后续 DeepSeek-R1 的工作扩展了这一验证。

2. **奖励函数简单**：论文使用 rule-based 二元奖励（对/错），限制了 GRPO 在需要细粒度反馈的任务上的适用（如创意写作、翻译质量评估）。

3. **7B 规模上限**：仅验证了 7B 规模。GRPO 在更大模型（70B+）上的扩展性和与其他 RL 方法的相对优势未量化。

4. **数据规模 vs 数据质量**：论文证明了数据质量 > 数据规模，但质量的定义（fastText 数学分类）仍然粗糙。后续是否有更好的数学内容识别方法？

5. **无开源 GRPO 训练代码**：官方仓库只释放了模型权重和推理代码，未释放 GRPO 训练代码。（后续 OpenRLHF、VeRL 等框架实现了 GRPO）

### 9.2 延伸阅读

| 论文 | 关系 | 核心内容 |
|------|------|---------|
| [Minerva (2022)](https://arxiv.org/abs/2206.14858) | 对比对象 | 540B 数学专用模型，DeepSeekMath 在 77× 更小参数下超越 |
| [Llemma (2023)](https://arxiv.org/abs/2310.10631) | 同期工 | 34B Proof-Pile-2 训练，证明专用预训练的有效性 |
| [DeepSeek-Coder (2024)](https://arxiv.org/abs/2401.14196) | 前身 | DeepSeekMath 的初始化基础，16B tokens 代码语料 |
| [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) | 后继 | 将 GRPO 扩展到通用推理，引入"冷启动"数据+多阶段训练 |
| [DeepSeekMath-V2 (2025)](https://github.com/deepseek-ai/DeepSeek-Math-V2) | 直接后继 | 更大的模型和数据，MATH >70% |
| [WizardMath (2023)](https://arxiv.org/abs/2308.09583) | RL对比 | 用 Evol-Instruct + PPO 提升数学，验证了 RL 的有效性 |
| [Proximal Policy Optimization (2017)](https://arxiv.org/abs/1707.06347) | 理论基础 | PPO 原论文，GRPO 的改进起点 |
| [Direct Preference Optimization (2023)](https://arxiv.org/abs/2305.18290) | 统一范式中的方法 | DPO 作为隐式 RL，被统一范式纳入 |
| [VeRL](https://github.com/volcengine/verl) | GRPO 开源实现 | 字节跳动的 RLHF 框架，包含 GRPO 高性能实现 |
| [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF) | GRPO 开源实现 | 开源 RLHF 框架，支持 GRPO |

### 9.3 后续影响

DeepSeekMath 在 AI 社区的影响远超一篇数学论文：

1. **GRPO 成为 RLHF 新范式**：GRPO 提出后，被 DeepSeek-R1、Qwen 等大量后续工作采用。免 critic 的设计大幅降低了 RL 训练门槛。

2. **数据 Pipeline 方法论**：迭代式 fastText 挖掘被推广——不仅适用于数学，也适用于代码、法律、医学等垂直领域的数据收集。

3. **Code→Math 迁移成为共识**：代码预训练 → 数学微调的训练路径成为后续数学模型的标配（如 Qwen2.5-Math）。

4. **Pass@K vs Maj@K 分析方法**：论文对 RL 增益本质的分析（提升鲁棒性而非基本能力）成为后续 RL scaling 研究的基础分析框架。

5. **GRPO + 迭代RL → DeepSeek-R1**：GRPO 的迭代 RL 思想 + "用模型自举评分"直接催生了 DeepSeek-R1 的多阶段训练范式。

---

## 附录：关键数字速查

| 数字 | 含义 |
|------|------|
| 7B | 模型参数量 |
| 120B | DeepSeekMath Corpus 总 tokens |
| 35.5M | 收集的网页数 |
| 4 | 数据挖掘迭代轮次 |
| 500B | 预训练总 tokens |
| 4096 | 上下文窗口 |
| 4.2×10⁻⁴ | 学习率 |
| 10M | Batch size (tokens) |
| 56% | 数学数据在训练中的占比 |
| 51.7% | DeepSeekMath-RL MATH 准确率 |
| 88.2% | DeepSeekMath-RL GSM8K 准确率 |
| 60.9% | MATH w/ self-consistency (64 samples) |
| 77× | Minerva 540B vs DeepSeekMath 7B 参数比 |
| 64 | GRPO 每组采样数 (k) |
| 0.2 | GRPO clip ε |
| 0.04 | GRPO KL 惩罚系数 β |