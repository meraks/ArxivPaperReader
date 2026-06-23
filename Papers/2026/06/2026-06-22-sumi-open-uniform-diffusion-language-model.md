# Sumi: 从零预训练的 7B 统一扩散语言模型

**论文信息**
- **标题**：Sumi: Open Uniform Diffusion Language Model from Scratch
- **作者**：Mengyu Ye, Keito Kudo, Wataru Ikeda, Ryosuke Matsuda, Keisuke Sakaguchi, Jun Suzuki
- **机构**：东北大学 (Tohoku University)
- **arXiv ID**：2606.19005
- **发布日期**：2026-06-18
- **官方代码**：[sumi](https://github.com/tohoku-nlp/sumi)
- **HuggingFace**：[sumi-7b](https://huggingface.co/tohoku-nlp/sumi-7b)

---

## Ch1 论文概述与核心贡献

Sumi 是首个在 7B 参数规模和 1.5T token 预训练预算下，从零开始预训练的统一扩散语言模型 (Uniform Diffusion Language Model, UDLM)。UDLM 允许任意 token 在任意扩散步骤中被更新，理论上相比自回归 (AR) 模型和掩码扩散模型具有更强的生成灵活性。

在此之前，仅掩码扩散模型 (MDLM) 如 LLaDA-8B 已在大规模下验证了可行性，而 UDLM 要么是从预训练 AR 模型适配而来 (DiffusionGemma)，要么仅停留在小规模实验阶段。Sumi 的发布填补了这一空白，为社区提供了一个干净的 UDLM scaling 行为参考点。

### 核心贡献

1. **首次大规模 UDLM 从零预训练**：Sumi 在 7B 参数/1.5T token 规模下完成从零预训练，是首个达到此规模的统一扩散语言模型
2. **全面的开源发布**：模型权重、中间检查点、完整训练配方、数据混合规格全部公开
3. **与 AR 模型的竞争性对比**：在知识、推理、编程基准上与同 token 预算的 AR 模型 (Falcon-7B、Llama 2-7B、OLMo-7B) 性能相当
4. **诚实的不足报告**：在常识基准上显著弱于 AR 模型，归因于教育偏重 (education-heavy) 的数据混合策略

### 训练资源

- **硬件**：288 张 NVIDIA H100 GPU
- **GPU 小时**：43,308 小时（预训练 35,776 + 中期训练 7,531）
- **预算**：约 1.5T tokens

### 模型性能概览

在知识基准 (MMLU 51.1)、推理 (GSM8K* 32.8)、编程 (HumanEval* 22.6) 上与同规模 AR 模型竞争，但在常识推理 (RACE 41.4) 上明显偏弱。

---

## Ch2 研究背景与动机

### 2.1 语言模型生成的三种范式

#### 自回归 (Autoregressive, AR) 模型

AR 模型 (如 GPT 系列、Llama) 采用因果注意力机制，从左到右逐 token 生成。单向生成过程无法回溯修改已生成的 token，一旦出现错误只能继续或重启整个生成过程。AR 模型已在 7B-400B+ 参数规模和万亿级 token 上验证了 scaling behavior，并有多个开源实现。

#### 掩码扩散 (Masked Diffusion) 模型

掩码扩散模型 (如 LLaDA、SSD-LM) 从全掩码序列出发，逐步被预测 token 填充。每一步填充的 token 在后续步骤中不再修改，只能填充新的位置。掩码扩散在 8B 规模和 2T token 上已证明与 AR 模型竞争力相当。

#### 统一扩散 (Uniform Diffusion) 模型

UDLM 不限定哪些 token 可以被修改——任意位置、任意步骤都可以被更新。这种机制理论上支持自我修正和自我完善，但也意味着训练和推理变得更加复杂。UDLM 的数学模型基于广义插值离散扩散 (GIDD) 框架。

### 2.2 UDLM 的研究空白

尽管 UDLM 理论上有吸引力，但此前不存在从零开始大规模预训练的 UDLM。已有的工作包括：
- **DiffusionGemma**：从预训练的 Gemma 模型适配为 UDLM，不是原生 UDLM
- **小规模实验**：1.7B 参数级别或在数据充足条件下的小规模预训练
- **计算最优实验**：在较小规模上研究了 UDLM 的计算最优配置

缺乏大规模原生 UDLM 意味着社区无法系统研究：
- UDLM 的 scaling behavior（与 AR 模型的差异）
- 原生 UDLM 的生成动力学（去噪路径的规律）
- UDLM 的可控性和涌现能力

### 2.3 从零预训练 UDLM 的价值

Sumi 从零开始预训练（而非从 AR 模型适配）提供了干净的基准：
- 消除了模型适配带来的混淆因素
- 可以从训练过程初期追踪 UDLM 的学习动态
- 为后续研究 UDLM scaling、数据混合、训练策略提供对比基线

### 2.4 相关工作定位

| 工作             | 类型                          | 规模            |   从零预训练？    |   开源？    |
| -------------- | --------------------------- | ------------- | :---------: | :------: |
| LLaDA          | Masked Diffusion            | 8B / 2T       |      ✅      |    ✅     |
| DiffusionGemma | Uniform Diffusion (adapted) | 7B / -        | ❌ (从 AR 适配) |    ✅     |
| Small UDLMs    | Uniform Diffusion           | ≤1.7B / ≤1T   |      ✅      |    部分    |
| **Sumi (本文)**  | **Uniform Diffusion**       | **7B / 1.5T** |    **✅**    | **✅ 完整** |

---

## Ch3 Sumi 模型架构与训练目标

### 3.1 架构细节

Sumi 采用标准 LLaMA 风格架构，关键配置如下：

| 组件 | 配置 |
|------|------|
| 层数 | 36 |
| 隐藏层维度 | 4,096 |
| FFN 维度 | 12,288 (SwiGLU 激活) |
| 注意力头数 | 32 |
| KV 头数 (GQA) | 8 |
| 头维度 | 128 |
| 位置编码 | RoPE, θ=500,000 |
| Tokenizer | OLMo 3 (词表 100,278) |
| 精度 | bfloat16 |
| Embedding | untied (与 lm_head 分离) |
| Dropout | 不使用 |
| Attention Sink | off-by-one softmax |

**RoPE 基频选择**：θ=500,000 相比标准 RoPE 的 10,000 更大，使得低频位置的高频编码更细致，有助于模型学习长距离依赖。这是 Sumi 能够从 1,184 扩展到 4,864 序列长度的重要因素。

**off-by-one softmax**：在 softmax 计算前为 logits 增加一个小的偏置项，防止注意力 logits 全部集中在特定 token 上（attention sink 现象）。这一技术此前在 AR 模型的长期推理中被证明有效。

#### 架构变体

Sumi 全部使用标准 Transformer 块，没有引入特殊模块如 MoE、线性注意力或状态空间模型。这一设计选择保证了架构的简洁性，使得性能归因更加清晰——所有改进来自 UDLM 训练范式本身，而非架构创新。

### 3.2 训练目标：GIDD

Sumi 的数学基础是广义插值离散扩散 (Generalized Interpolating Discrete Diffusion, GIDD) 框架。与将连续扩散直接离散化的方法不同，GIDD 直接在离散 token 空间上定义扩散过程。

对于数据分布 $q(x_0)$，GIDD 通过以下过程建模：

1. **前向过程**：从均匀噪声分布采样，逐步向数据 $x_0$ 添加噪声
2. **反向过程**：学习去噪网络 $p_\theta(x_{t-1}|x_t)$ 逐步恢复干净数据

Sumi 的关键改进是 **SNR-reparameterized ELBO**——将 GIDD 的证据下界重新参数化为信噪比 (Signal-to-Noise Ratio) 的函数。这一重参数化使得：
- 噪声调度 (noise schedule) 与模型优化解耦
- 不同噪声水平的学习目标可以通过 SNR 统一衡量
- 训练更加稳定，对噪声调度超参数的选择更鲁棒

**噪声调度**：使用均匀噪声 (uniform noise)，log-SNR 范围 λ ∈ [-9, 9]。这一较宽的范围覆盖了从几乎干净到几乎完全噪声的扩散过程。

### 3.3 训练配置

| 超参数 | 值 |
|--------|-----|
| 优化器 | AdamW (β₁=0.9, β₂=0.95) |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| Auxiliary loss | z-loss (系数 1e-5) |
| 学习率 | WSD: warmup 2,000 → 2e-4 → cooldown 2,000 → 2e-5 |
| 全局 batch size | ~5.5M tokens/step (4,608 seqs @ 1,184) |
| Batch 调整 | 中期训练 seq len 增加时同步减少 seq count |

**WSD 调度**：Warmup-Stable-Decay 调度在达到峰值学习率后保持长期稳定（稳定阶段），然后在训练结束时快速衰减 (cooldown)。相比余弦调度，WSD 在训练过程中保持了恒定的学习率，使得模型在大部分训练时间内以最大学习率优化。

**z-loss**：在 softmax logits 上增加辅助损失，防止 logits 数值过大。系数 1e-5 确保辅助损失对主损失的影响可忽略。

### 3.4 三阶段训练流程

Sumi 的训练分为三个连续的阶段：

1. **阶段 1 — 预训练** (1.3T tokens @ seq_len 1,184)
   - 使用标准 LLaMA 架构在 1,184 序列长度上预训练
   - 训练约 35,776 GPU-hours
   - 使用教育评分高的公共语料

2. **阶段 2 — 中期训练 (短序列)** (130B tokens @ seq_len 1,184)
   - 相同的序列长度，但数据混合转向推理和编程
   - 开始引入数学和推理相关数据

3. **阶段 3 — 中期训练 (长序列)** (120B tokens @ seq_len 4,864)
   - 序列长度扩展到 4,864（约 4 倍于预训练）
   - 更新 RoPE 以支持更长的上下文
   - batch 大小从 4,608 seqs 降为 1,152 seqs，保持 ~5.6M tokens/step

总 GPU 小时：43,308。

### 3.5 训练稳定性

Sumi 的训练 loss 曲线平滑，没有出现梯度爆炸或 loss 发散。in-training monitor scores（在训练过程中采样的子集评估指标）也保持稳定。这归因于：
- SNR-reparameterized ELBO 的稳定性优势
- z-loss 对 logits 数值的约束
- 保守的 WSD 调度

---

## Ch4 训练数据与数据混合

### 4.1 预训练数据来源

Sumi 的预训练数据全部来自 **llm-jp-corpus-v4** 和 **v4.1** ——日本 llm-jp 社区构建的大规模多语言预训练语料。具体使用的子集：

- **llm-jp-corpus-v4 英文部分** (排除 FineWeb 子集)
- **code_olmo-starcoder**：编程代码数据
- **swallow_code_v2**：Python 编程数据
- **en_fineweb-rescored** (v4.1)：FineWeb 文档经教育评分后重新排序

**教育评分机制**：使用 Qwen3-32B 模型对 FineWeb 文档的教育价值进行评分（从 0 到 5），然后将评分最高的子集作为预训练数据。这一筛选策略使得 Sumi 的预训练数据偏向于教科书、教程、学术文章等高教育价值内容。

### 4.2 中期训练数据混合

经过预训练后，Sumi 进入中期训练，数据混合显著转向推理密集型任务：

| 数据类型 | Token 量 | 比例 |
|---------|:--------:|:----:|
| 编程 (Coding) | 81.4B | **32.5%** |
| 数学 (Math) | 74.3B | **29.7%** |
| 通用 (General) | 52.4B | **21.0%** |
| 推理 (Reasoning) | 42.0B | **16.8%** |

**关键调整**：
- **代码降采样**：原始 `nemotron_pretraining_code_v2` 有 491B token，被显著降采样到 63B token，以避免编程数据过度主导
- **数学增强**：从 `en_megamath-web-pro-max-oss` 添加高质量数学数据，经过 math_score 过滤，最终仅 74.3B token
- **推理过滤**：仅保留长度 ≤ 4,096 tokens 的推理样本，丢弃过长样本（约 55% 被过滤）

### 4.3 教育偏重策略的影响

教育偏重数据混合的最直接后果是 **常识基准性能显著偏弱**：

- PIQA: 66.4 (vs Falcon-7B 80.5, Llama 2-7B 78.7)
- HellaSwag: 60.0 (vs Falcon-7B 76.3, Llama 2-7B 76.2)
- WinoGrande: 60.0 (vs Falcon-7B 71.6, OLMo-7B 71.3)

这些差距不能仅用 token 预算解释（Llama 2-7B 2T vs Sumi 1.5T 差距不大），而更可能是数据分布差异。论文明确指出，教育偏重数据混合是主因但不是唯一原因——UDLM 本身的生成范式也可能影响常识任务表现。

**教育偏重同时带来正面效应**：在知识密集任务（MMLU 51.1）、数学推理（GSM8K 32.8）、编程（HumanEval 22.6）上，Sumi 在同 protocol 模型中实现了最优或接近最优的性能。

### 4.4 数据公开性

所有数据均来自公开可用的语料库：
- llm-jp-corpus-v4 / v4.1：公开可用
- StarCoder、swallow_code_v2：公开代码数据集
- en_megamath-web-pro-max-oss：公开数学语料

论文提供了完整的数据混合规格（Figure 1），社区可以完全复现训练数据的组成。

---

## 第5章 实验结果与分析

### 5.1 评估设置

Sumi 的评估基于修改版的 `lm-evaluation-harness` 框架，主要修改在于扩散模型的评分机制。生成过程中使用固定长度的 canvas（最优长度为 2048 tokens），采用 `<EOS>` 和 `<BOS>` 标记作为生成边界。Prompt 被冻结在 canvas 开头，`<|beginoftext|>` 分隔符锚定在 `max_new_tokens` 位置之后，剩余位置作为双向上下文进行去噪。

对比模型遵循相同评估 protocol，包括：
- Falcon-7B（1.5T tokens）
- Llama 2-7B（2T tokens）
- OLMo-7B（2.5T tokens）

LLaDA-8B 和 Llama 3-8B 作为参考模型，但使用不同的评估 protocol。

### 5.2 核心评估结果

| 类别 | Benchmark | Sumi-7B | Falcon-7B | Llama 2-7B | OLMo-7B |
|------|-----------|---------|-----------|------------|---------|
| **General Knowledge** | MMLU | **51.1** | 27.2 | 46.0 | 28.0 |
| | RACE | **41.4** | 38.3 | 39.5 | 37.9 |
| | TruthfulQA | **46.6** | 34.3 | 38.8 | 35.9 |
| **Reasoning & Math** | GSM8K* | **32.8** | 5.3 | 13.5 | 3.8 |
| | ARC-Easy | 70.0 | 70.8 | **73.8** | 68.8 |
| | ARC-Challenge | 43.0 | 43.2 | **45.1** | 40.3 |
| | BBH* | 31.8 | 27.1 | **39.6** | 29.8 |
| | GPQA | **26.1** | 24.6 | 24.3 | 24.8 |
| **Coding** | HumanEval* | **22.6** | 0.0 | 12.8 | 13.4 |
| | MBPP* | **26.6** | 12.4 | 23.2 | 21.4 |
| **Commonsense** | PIQA | 66.4 | **80.5** | 78.7 | 79.8 |
| | HellaSwag | 60.0 | **76.3** | 76.2 | 75.6 |
| | WinoGrande | 60.0 | 71.6 | **74.7** | 71.3 |

注：带 * 的任务使用 generation 模式评估（5-shot）。**加粗** 表示同 protocol 模型中的最佳值（LLaDA-8B 和 Llama 3-8B 使用不同 protocol，未在表中列出）。

在知识基准（MMLU）上，Sumi-7B 达到 51.1%，在同 protocol 模型中表现最佳。推理与数学基准中，GSM8K* 达到 32.8%。编程基准 HumanEval* 达到 22.6%，同样在同 protocol 模型中领先。常识基准（PIQA、HellaSwag、WinoGrande）上 Sumi-7B 显著弱于所有 AR 基线，这一差距被归因于教育偏重的数据混合。

### 5.3 Canvas 长度敏感性分析

Canvas 长度对模型性能的影响因任务而异。GSM8K 对 canvas 长度最为敏感，不同长度下的性能波动较大。这表明需要较长的上下文窗口来支持多步推理。部分常识任务对 canvas 长度相对不敏感，较短的 canvas 即可达到稳定性能。

长 canvas（训练长度 2.5 倍以内）对大部分任务仍保持流畅性。GSM8K 在短 canvas 下性能急剧下降，在长 canvas 下也出现波动。编程任务仅在短 canvas 下性能下降，长 canvas 下仍表现良好。

### 5.4 Token 预测类别分布

对比分析自回归（AR）模型与扩散模型的 token 预测分布：
- AR 模型倾向于频繁预测高频 token
- Sumi 和 LLaDA 的 token 预测分布更加均匀

这一差异反映了扩散模型的去噪机制与 AR 模型的因果性生成机制的本质不同。

### 5.5 与 AR 模型的对比

Sumi 与同规模 AR 模型的关键对比：

- **知识（MMLU 51.1%）**：同 protocol 模型中最优，比 OLMo-7B 的 28.0% 高 23.1 个百分点
- **推理（GSM8K* 32.8%）**：比 Falcon-7B 的 5.3% 高 27.5 个百分点
- **编程（HumanEval* 22.6%）**：比 Falcon-7B 的 0.0% 高 22.6 个百分点
- **常识（HellaSwag 60.0%）**：比 OLMo-7B 的 75.6% 低 15.6 个百分点

Sumi 在知识密集和推理密集型任务上展现了 UDLM 的竞争优势，但常识任务的差距提示 UDLM 需要不同的数据策略。

---

## 第6章 代码实现详解

### 6.1 仓库结构

Sumi 的官方代码仓库 ([tohoku-nlp/sumi](https://github.com/tohoku-nlp/sumi)) 结构清晰：

```
src/sumi_eval/
  --model sumi
  generate_until
model/
  modeling_sumi.py
  generation_sumi.py
  configuration_sumi.py
```

其中 `model/` 目录包含核心模型定义：
- **modeling_sumi.py**：Sumi 的 Transformer 模型定义，包括 GIDD 训练目标的前向传播
- **generation_sumi.py**：基于 canvas 的生成逻辑
- **configuration_sumi.py**：模型配置类（层数、隐藏维度等）

### 6.2 模型加载

Sumi 以自定义模型 (custom model) 的形式集成到 HuggingFace Transformers 中（推荐版本 5.8.1）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "tohoku-nlp/sumi-7b",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/sumi-7b")
```

注意 `trust_remote_code=True` 是必需的——因为 Sumi 使用了自定义模型类，需要从 HuggingFace Hub 加载并执行自定义代码。

### 6.3 Generation 流程

Sumi 的生成机制基于固定长度的 canvas（默认 2048 tokens），流程如下：

1. **创建 canvas**：创建一个固定长度的 token 序列，长度为 `canvas_length`
2. **插入 prompt**：将用户输入的 prompt 冻结在 canvas 开头
3. **锚定分隔符**：在 prompt 结束位置后的 `max_new_tokens` 位置插入 `<|endoftext|><|beginoftext|>` 分隔符序列
4. **双向去噪**：所有剩余位置（未冻结、未锚定的位置）作为双向上下文，通过去噪过程逐步生成
5. **裁剪输出**：生成的序列在第一个 `<|endoftext|>` token 处裁剪，返回最终结果

```python
# Generation 参数
# fill_mode=anchor
# max_new_tokens (maximum answer length)
# sampler=adaptive
```

生成的返回值包含：
- `out.sequences`：裁剪后的最终生成序列（到第一个 `<|endoftext|>` 为止）
- `out.canvas`：完整的未裁剪 canvas

### 6.4 评估集成

评估通过将 Sumi 注册为 `lm-eval` 的自定义模型进行：

```bash
src/sumi_eval/
  --model sumi
  generate_until
```

对于多项选择（loglikelihood）任务，使用标准 lm-eval 参数。对于生成任务，指定 `fill_mode=anchor`、`sampler=adaptive` 和 `max_new_tokens`。

代码任务需要额外 `--confirm_run_unsafe_code` 标志。

### 6.5 代码示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "tohoku-nlp/sumi-7b",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/sumi-7b")

# 生成示例
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=32,
    fill_mode="anchor",
    sampler="adaptive"
)
print(tokenizer.decode(out.sequences[0]))
```

### 推理示例

```python
# 使用 from_pretrained 加载
model = AutoModelForCausalLM.from_pretrained(
    "tohoku-nlp/sumi-7b",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/sumi-7b")

# 推理时指定 canvas 长度（默认 2048）
# 提示：训练中使用的 canvas 长度为 2048，推荐保持此值
prompt = "Explain the concept of diffusion models"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(
    **inputs,
    max_new_tokens=256,
    fill_mode="anchor"
)
generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
```

---

## 第7章 局限性与延伸阅读

### 7.1 局限性

#### 常识性能偏弱

Sumi 最显著的局限是常识基准上的弱势——PIQA 66.4%、HellaSwag 60.0%、WinoGrande 60.0%，均低于所有同 protocol AR 模型。论文归因于教育偏重数据混合，但也指出 UDLM 的生成范式本身可能不是常识任务的最优选择。作者计划通过 SFT 来缓解这一问题。

#### Canvas 长度约束

生成时的 canvas 长度必须与训练分布匹配。训练使用的序列长度分布决定了生成时的最优 canvas 长度（2048 tokens）。过长的 canvas 可能引入训练分布外的噪声，过短则损失上下文信息。

#### 分隔符锚定机制

当前方案需要将 `<|endoftext|><|beginoftext|>` 分隔符锚定在 `max_new_tokens` 位置后，这是一种临时约束——因为 Sumi 的预训练没有使用注意力掩码 (attention mask)，模型无法区分 prompt 区域和生成区域。作者计划发布 SFT 版本来缓解这一限制。

#### 生成质量问题

由于 UDLM 的去噪本质，Sumi 的生成质量在某些任务上不如 AR 模型流畅。去噪过程可能在长序列中累积误差，导致生成的 token 之间出现语义不一致。SFT 可以直接改善生成质量。

#### Scaling 规律未充分研究

作为首个大规模 UDLM，Sumi 的 scaling 规律（参数规模与训练预算的关系）尚未得到充分研究。目前仅有一个数据点（7B/1.5T），难以得出 generalizable 的结论。

### 7.2 延伸阅读

#### LLaDA (Nie et al., 2026)

LLaDA 是掩码扩散语言模型，在 8B/2T 规模上达到与 Llama 3-8B 竞争的性能。与 Sumi 不同，LLaDA 使用的是一次填充 (fill-in-one-shot) 的掩码扩散策略，一旦 token 被填充就不允许修改。阅读 LLaDA 有助于理解掩码扩散与统一扩散的差异。

#### DiffusionGemma (Wang et al., 2025)

DiffusionGemma 是从预训练 Gemma 模型适配的 UDLM。与 Sumi "从零预训练" 不同，它使用了 AR 预训练权重作为初始化。比较两者可以分离"从零训练"和"模型适配"的影响。

#### UDLM (Ou et al., 2025)

原始 UDLM 论文，提出了 uniform diffusion 的基本框架和计算最优配置。Sumi 在其基础上扩展到 7B/1.5T 规模。结合阅读可以理解 UDLM 从理论到工程的演变路径。

#### Discrete Diffusion Foundations

Austin et al. (2021) D3PM、Campbell et al. (2022) 连续时间离散扩散、Ou et al. (2025) GIDD。这些 foundational work 提供了离散扩散的数学基础，理解它们有助于深入理解 Sumi 的 SNR-reparameterized ELBO。
