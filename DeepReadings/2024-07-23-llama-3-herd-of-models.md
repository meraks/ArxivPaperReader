# Llama 3 深度精读报告（第 1–3 章）

> **论文**：《The Llama 3 Herd of Models》  
> **作者**：Llama Team, AI @ Meta（Grattafola 等 400+ 作者）  
> **发表**：arXiv:2407.21783, July 2024 (v3 Nov 2024)  
> **代码**：https://github.com/meta-llama/llama-models  
> **协议**：Llama 3 Community License（开放权重）

---

## 第 1 章 概述

### 1.1 模型家族

Llama 3 是一个包含 **8B、70B、405B** 三个规模的语言模型家族（统称 Llama 3 Herd），采用标准的 **Dense Transformer** 架构（而非 Mixture-of-Experts）。所有模型均支持多语言、编码、推理与工具使用。其旗舰模型为 405B 参数，上下文窗口最长可达 **128K tokens**。

论文中的所有实验结果均基于 **Llama 3.1** 系列模型，论文中为简练统称为 Llama 3。模型发布时间线如 Table 1 所示：

| Model | Finetuned | Multilingual | Long context | Tool use | Release |
|-------|:---------:|:-----------:|:-----------:|:-------:|:-------:|
| Llama 3 8B | ✗ | ✗ | ✗ | ✗ | Apr 2024 |
| Llama 3 8B Instruct | ✓ | ✗ | ✗ | ✗ | Apr 2024 |
| Llama 3 70B | ✗ | ✗¹ | ✗ | ✗ | Apr 2024 |
| Llama 3 70B Instruct | ✓ | ✗ | ✗ | ✗ | Apr 2024 |
| Llama 3.1 8B | ✗ | ✓ | ✓ | ✗ | Jul 2024 |
| Llama 3.1 8B Instruct | ✓ | ✓ | ✓ | ✓ | Jul 2024 |
| Llama 3.1 70B | ✗ | ✓ | ✓ | ✗ | Jul 2024 |
| Llama 3.1 70B Instruct | ✓ | ✓ | ✓ | ✓ | Jul 2024 |
| Llama 3.1 405B | ✗ | ✓ | ✓ | ✗ | Jul 2024 |
| Llama 3.1 405B Instruct | ✓ | ✓ | ✓ | ✓ | Jul 2024 |

¹ 8B 与 70B 在预训练阶段使用了多语言数据，但发布时针对英文使用场景。

### 1.2 核心结果

Table 2 展示了 Llama 3 三个模型与 GPT-4、GPT-4o、Claude 3.5 Sonnet 等闭源模型的关键基准对比：

| Benchmark | Llama 3 8B | Llama 3 70B | Llama 3 405B | GPT-4 (0125) | GPT-4o | Claude 3.5 Sonnet |
|-----------|:---------:|:----------:|:-----------:|:-----------:|:-----:|:----------------:|
| MMLU (5-shot) | 69.4 | 83.6 | 87.3 | 85.1 | 89.1 | 89.9 |
| MMLU-Pro (5-shot, CoT) | 48.3 | 66.4 | 73.3 | 64.8 | 74.0 | 77.0 |
| IFEval | 80.4 | 87.5 | 88.6 | 84.3 | 85.6 | 88.0 |
| HumanEval (0-shot) | 72.6 | 80.5 | 89.0 | 86.6 | 90.2 | 92.0 |
| MBPP EvalPlus (0-shot) | 72.8 | 86.0 | 88.6 | 83.6 | 87.8 | 90.5 |
| GSM8K (8-shot, CoT) | 84.5 | 95.1 | 96.8 | 94.2 | 96.1 | 96.4 |
| MATH (0-shot, CoT) | 51.9 | 68.0 | 73.8 | 64.5 | 76.6 | 71.1 |
| ARC Challenge (0-shot) | 83.4 | 94.8 | 96.9 | 96.4 | 96.7 | 96.7 |
| GPQA (0-shot, CoT) | 32.8 | 46.7 | 51.1 | 41.4 | 53.6 | 59.4 |
| BFCL (tool use) | 76.1 | 84.8 | 88.5 | 88.3 | 80.5 | 90.2 |
| MGSM (0-shot, CoT) | 68.9 | 86.9 | 91.6 | 85.9 | 90.5 | 91.6 |
| ZeroSCROLLS/QuALITY | 81.0 | 90.5 | 95.2 | 95.2 | 90.5 | 90.5 |
| NIH/Multi-needle | 98.8 | 97.5 | 98.1 | 100.0 | 100.0 | 90.8 |

几个值得关注的观察：

- **405B vs GPT-4**：Llama 3 405B 在 MMLU 87.3 超越 GPT-4 的 85.1，在 GSM8K 96.8 领先 GPT-4 的 94.2，MATH 73.8 也优于 GPT-4 的 64.5。与 GPT-4o 相比互有胜负。
- **小模型领先**：8B 和 70B 版本在同参数量级中属于最佳水平，显著优于 Gemma 2 9B、Mistral 7B、Mixtral 8x22B 等竞品。
- **主要差距**：GPQA 51.1 仍明显落后于 GPT-4o 的 53.6 和 Claude 3.5 的 59.4，说明在需要研究生级专业知识的高难度推理任务上仍有提升空间。
- **Benchmark 饱和**：ARC Challenge 三个最强模型均接近 97%，MMLU 前三名差距已缩窄至 2–3 个百分点，仅凭传统 benchmark 区分高端模型越来越困难。

### 1.3 三大核心杠杆

论文提出开发高质量基础模型的三个关键杠杆：

1. **数据 (Data)**：相比 Llama 2（1.8T tokens），预训练数据量提升至约 **15T tokens**（多语言），并大幅改进了数据预处理和质量管理流程。
2. **规模 (Scale)**：旗舰模型 405B 使用约 **3.8 × 10²⁵ FLOPs** 的算力进行预训练，约为 Llama 2 最大版本的 50 倍。
3. **管理复杂性 (Managing Complexity)**：选择标准 Dense Transformer 架构（而非 MoE）以最大化训练稳定性；后训练采用相对简单的 SFT + Rejection Sampling + DPO 流程，而非更复杂且不稳定的 RLHF。

### 1.4 许可与发布

所有三个规模的模型（含 405B 的预训练与后训练版本）以及 Llama Guard 3（输入/输出安全模型）均以 Llama 3 Community License 开放发布。

---

## 第 2 章 预训练数据

### 2.1 数据治理管线

Llama 3 的预训练语料来源多样，包含截至 2023 年底的知识，总量约 **15T tokens**（多语言）。论文采用了多层数据治理流程：

**PII 与安全过滤。** 移除含有大量个人身份信息（PII）的域名，以及被 Meta 安全标准评为有害的域名和已知成人内容站点。

**文本提取与清理。** 构建了自定义 HTML 解析器，在去除 boilerplate 时的精确度优于流行的第三方 HTML 解析器。对于数学内容（常以预渲染图片形式呈现，数学内容含在 alt 属性中）和代码内容做了专门的结构保留处理。实验发现 **markdown 对模型性能有害**，因此移除了所有 markdown 标记，保留纯文本。

**去重 (De-duplication)。** 采用三轮去重：
- **URL 级别**：全数据集去重，保留每个 URL 的最新版本。
- **文档级别**：使用全局 MinHash 去重，移除近似重复文档。
- **行级别**：类似 ccNet 的行级去重，移除每 3000 万文档桶中出现超过 6 次的行。

**启发式过滤。** 包括：
- 重复 n-gram 覆盖比率：移除日志或错误消息等重复内容。
- "脏词"计数：过滤未被域名屏蔽覆盖的成人网站。
- Token 分布 KL 散度：移除包含异常过多离群 token 的文档。

**基于模型的质量过滤。** 使用了多个质量分类器：
- **fasttext 分类器**：训练识别被 Wikipedia 引用的文本。
- **DistilRoberta 分类器**：基于 Llama 2 的 chat 模型标注数据训练。

**代码与推理数据。** 构建了专门的代码和数学内容提取管线。代码和推理分类器均为基于 Llama 2 标注的 DistilRoberta 模型，通过 prompt tuning 定向捕捉含数学推导、STEM 推理和代码的网页。

### 2.2 数据配比

论文通过知识分类和 scaling law 实验来确定最优数据配比。最终配比为：

- **50%** 通用知识
- **25%** 数学与推理数据
- **17%** 代码数据
- **8%** 多语言数据

### 2.3 多语言数据

使用 fasttext 语言识别模型将文档分类为 **176 种语言**。按语言分别进行文档级和行级去重，并应用语言特定的启发式和基于模型的过滤。使用多语言 Llama 2 分类器进行质量排序，通过实验确定多语言 token 数量的最优配比。

### 2.4 退火数据 (Annealing)

在预训练末期，利用少量高质量代码和数学数据进行 **退火（annealing）**：将学习率线性衰减至 0，使用 40B tokens，其中 30% 为高质量领域数据，70% 为默认数据配比。

退火效果因模型规模而异：
- Llama 3 8B：GSM8k 提升 **24.0%**，MATH 提升 **6.4%**
- Llama 3 405B：提升可忽略不计

这说明大模型具有强大的上下文学习和推理能力，无需特定领域训练样本即可获得强性能。

---

## 第 3 章 模型架构

### 3.1 架构总览

Llama 3 采用标准 Dense Transformer 架构（Vaswani et al., 2017），与 Llama 和 Llama 2 相比未做显著架构变更，性能提升主要来自数据质量和多样性的改进以及训练规模的扩大。

| 超参数 | 8B | 70B | 405B |
|:-------|:---:|:---:|:----:|
| Layers | 32 | 80 | 126 |
| Model Dimension | 4,096 | 8,192 | 16,384 |
| FFN Dimension | 14,336 | 28,672 | 53,248 |
| Attention Heads | 32 | 64 | 128 |
| Key/Value Heads | 8 | 8 | 8 |
| Peak Learning Rate | 3 × 10⁻⁴ | 1.5 × 10⁻⁴ | 8 × 10⁻⁵ |
| Activation Function | SwiGLU | SwiGLU | SwiGLU |
| Vocabulary Size | 128,000 | 128,000 | 128,000 |
| Positional Embeddings | RoPE (θ = 500,000) | RoPE (θ = 500,000) | RoPE (θ = 500,000) |

### 3.2 关键设计变更

相较于 Llama 2，Llama 3 做了以下几项关键改动：

**Grouped Query Attention (GQA)。** 使用 **8 个 KV heads**。这能有效提升推理速度，并减少解码过程中 KV 缓存的显存占用。

**Document Mask。** 使用注意力掩码防止同一序列中不同文档之间的自注意力。在标准预训练中影响有限，但在超长序列的持续预训练中非常重要。

**128K Token 词表。** 将 **tiktoken** 中的 100K tokens 与 **28K 额外非英语 token** 合并。相比 Llama 2 分词器，英文压缩率从 3.17 提升至 3.94 字符/token。额外加入的 28K 非英语 tokens 改善了多语言压缩率和下游性能，且对英文 tokenization 无影响。

**RoPE 基频提升至 500,000。** 支持更长的上下文。Xiong et al. (2023) 证明该值对高达 32,768 的上下文长度有效。

**上下文扩展至 128K。** 在标准预训练（8K 上下文）之后，通过持续预训练（continued pre-training）将上下文窗口扩展至 128K tokens。

### 3.3 Scaling Laws

论文开发了一套 **两阶段 scaling law 方法**：

1. 建立计算最优模型的 **负对数似然（NLL）与训练 FLOPs** 之间的线性相关性。
2. 建立 **NLL 与基准准确率** 之间的 Sigmoid 关系，利用了 scaling law 模型和较早的 Llama 2 模型。

**实验设置。** 在 6 × 10¹⁸ 至 10²² FLOPs 的计算预算下预训练模型，模型规模覆盖 40M 到 16B 参数。使用余弦学习率调度、2,000 步线性 warmup、峰值 LR 在 2 × 10⁻⁴ 至 4 × 10⁻⁴ 之间。

**发现。** 计算最优模型拟合结果为：

$$N^{\star}(C) = 0.29 \times C^{0.53}$$

外推至 **3.8 × 10²⁵ FLOPs** 表明：训练 **402B 参数模型** 在 **16.55T tokens** 上最为计算最优。基于此，论文最终选择了 **405B 参数** 的旗舰模型。一个重要的观察是：IsoFLOPs 曲线在最小值附近随计算预算增加而变得平坦，意味着旗舰模型的性能对模型规模与训练 token 之间的权衡较小变化相对稳健。

### 3.4 训练基础设施

**计算资源。** Llama 3 405B 使用最多 **16K H100 GPUs**（700W TDP，80GB HBM3），运行在 Meta 的 Grand Tion AI 服务器平台上。每台服务器搭载 8 块 GPU，通过 NVLink 互联。训练任务通过 Meta 的全球级训练调度器 **MAST** 调度。

**存储。** Meta 的通用分布式文件系统 **Tectonic** 构建了存储网络，提供 **240 PB** 存储（7,500 台 SSD 服务器），可持续吞吐量 **2 TB/s**，峰值 **7 TB/s**。

**网络。** Llama 3 405B 使用基于 **RoCE（RDMA over Converged Ethernet）** 的网络架构（Arista 7800 + Minipack2 交换机），400 Gbps 互联，三层 Clos 网络拓扑。较小模型使用 Nvidia Quantum2 Infiniband 网络。

### 3.5 4D 并行策略

Llama 3 采用 **4D 并行（4D parallelism）** 组合：

1. **Tensor Parallelism (TP)**：将单个权重张量拆分到多个设备上。
2. **Pipeline Parallelism (PP)**：按层垂直划分模型，不同设备处理流水线的不同阶段。
3. **Context Parallelism (CP)**：沿序列维度划分输入，对极长序列进行显存优化。
4. **Data Parallelism (DP / FSDP)**：分片模型参数、优化器和梯度，实现数据并行。

并行分组顺序为 **[TP, CP, PP, DP]**，按网络带宽需求从高到低排列：TP 限制在同一服务器内，DP（FSDP）可跨多跳网络。

配置示例（405B）：
- TP=8, CP=1→16, PP=16, DP=64→128
- Sequence length: 8,192 → 131,072
- Batch size: 16M tokens/global batch
- **TFLOPs/GPU: 380–430 BF16 MFU 38–43%**

**Pipeline 优化。** 论文针对 batch size 约束、显存不均衡、计算不均衡等挑战做了改进，支持灵活设置连续微批次数 N，并通过 interleaved schedule 减少 pipeline bubble。采用异步点对点通信，大幅加速训练。

**数值稳定性。** 在反向传播中使用 **FP32 梯度累积**，并在 FSDP 数据并行 worker 间以 FP32 reduce-scatter 梯度。最终在序列长度为 8K 时无需 activation checkpointing 即可训练。