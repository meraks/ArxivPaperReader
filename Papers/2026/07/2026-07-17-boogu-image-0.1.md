> **论文**：Boogu-Image-0.1: Boosting Open-Source Unified Multimodal Understanding and Generation · **作者**：Boogu Team · **arXiv**：2607.13125 · **日期**：2026-07-14 · **许可**：Apache 2.0
>
> **代码仓库**：https://github.com/Boogu-Project/Boogu-Image
>
> **模型权重**：https://huggingface.co/Boogu（Base, Turbo, Edit, Edit-Turbo 变体）

# Boogu-Image-0.1：面向开源统一多模态理解与生成的理解驱动型图像生成系统

## 第 1 章 概述

### 1.1 核心问题

图像生成正从 Text-to-Image（文本到图像）向 Requirement-to-Image（需求到图像）范式转变。用户不再满足于单句提示词，而是期望模型理解复杂意图、隐式约束、多级指令和跨模态上下文线索。现有开源模型主要聚焦于视觉质量提升（如 FLUX、Qwen-Image），但系统性地将"理解"（understanding）作为一等设计目标的方案仍属空白。同时，闭源系统（GPT-Image-2、Nano-Banana-Pro）通过系统级集成获得优势，但内部实践鲜有披露。

Boogu-Image-0.1 的核心洞察是：将**理解**提升至与生成能力同等重要的位置，通过理解能力的系统性注入（更好的文本编码器 + Agentic 推理时策略 + 高质量 Caption），在不显著增加参数量的前提下大幅提升生成质量。仅使用 2.086 亿张唯一图像、约 $400K 训练成本，即达到接近顶级闭源系统的性能。

### 1.2 关键结果速览

| 指标 | Boogu-Image-0.1 | 对比基准 |
|------|-----------------|---------|
| Boogu Arena Elo | 1048 (Turbo-Thinking) | GPT-Image-2: 1196, Nano-Banana-Pro: 1087 |
| Qwen-Image-Bench CN Overall | 53.57 (Base-Thinking) | 开源最优，接近闭源 |
| Qwen-Image-Bench EN Overall | 53.73 (Base-Thinking) | 开源最优，接近闭源 |
| LongText-Bench AVG | 0.971 (Turbo-Thinking) | 中文第二（0.985），仅次于 Seedream-4.5 |
| ImgEdit-Bench Overall | 4.64 (Edit-Thinking) | 全部评估模型中排名第一 |
| GenEval Overall | 0.85 (Base) | 开源模型前列，HiDream-O1: 0.90 |
| DPG-Bench Overall | 88.35 (Turbo) | 开源模型前列，HiDream-O1: 89.83 |
| 训练数据量 | 208.62M 唯一图像 | 显著少于同类开源模型 |
| 训练成本 | ~$400K | 仅为大规模训练方案的一小部分 |
| 开源协议 | Apache 2.0 | 权重、代码、配方完全开源 |

### 1.3 论文图表总览

| 编号 | 内容 | 章节 |
|------|------|------|
| **Figure 1** | Boogu Arena 性能对比 + T2I 变体权衡图 | 第 1 章 |
| **Figure 2** | 模型生成图像多样例展示 | 第 1 章 |
| **Figure 3** | 文本渲染定性结果 | 第 5 章 |
| **Figure 4** | 摄影与风格迁移定性结果 | 第 5 章 |
| **Figure 5** | 公共基准与人类偏好秩反转分析 | 第 5 章 |
| **Figure 6** | Boogu Arena 四面板 Elo 评分 | 第 5 章 |
| **Figure 7** | Boogu Arena vs LMArena 一致性 | 第 5 章 |
| **Figure 8-10** | Boogu Arena 三类别定性对比（摄影、风格化、文本渲染） | 第 5 章 |
| **Figure 11-12** | 密集文本渲染 2K 分辨率定性对比 | 第 5 章 |
| **Figure 13-15** | 图像编辑定性对比（ImgEdit、场景文本编辑、风格化） | 第 5 章 |
| **Table 1** | Qwen-Image-Bench 中文提示词结果 | 第 5 章 |
| **Table 2** | Qwen-Image-Bench 英文提示词结果 | 第 5 章 |
| **Table 3** | LongText-Bench 结果 | 第 5 章 |
| **Table 4** | GenEval 补充结果 | 第 5 章 |
| **Table 5** | DPG-Bench 补充结果 | 第 5 章 |
| **Table 6** | ImgEdit-Bench 图像编辑结果 | 第 5 章 |
| **Table 11** | 训练超参数配置 | 第 4 章 |

## 第 2 章 研究背景与动机

### 2.1 从 Text-to-Image 到 Requirement-to-Image

图像生成模型的发展经历了三个阶段：

1. **早期 Diffusion 阶段**（2021-2023）：DALL-E、Stable Diffusion 等模型专注于将简短提示词映射为视觉内容，用户输入以关键词为主
2. **高品质生成阶段**（2024-2025）：FLUX、Midjourney 等模型大幅提升视觉逼真度，开始支持更长、更详细的提示词
3. **需求理解阶段**（2026-）：GPT-Image-2、Nano-Banana-Pro 等闭源系统展示了系统级智能——不仅仅是更好的生成，而是理解用户意图，甚至通过 web search 增强知识

当前开源的差距不仅在于视觉质量，更在于**理解能力**。用户指令从简单关键词演变为复杂的多层级描述（如"一张蛋糕店橱窗展示海报，正方形构图，小红书清新ins风格"），包含隐式的风格、构图、色彩、布局要求，这对模型的跨模态理解能力提出了更高要求。

### 2.2 现有方法的局限

**闭源系统**通过多方面集成获得优势：更强的文本编码器、Agentic 改写器、推理时缩放、路由策略，但没有公开这些实践细节。

**开源方案**主要集中在三个方面：
- FLUX / Ideogram：提升视觉保真度
- Qwen-Image / Z-Image：附加 LLM 辅助改写提示词
- Hunyuan-Image：优化训练管线

但存在系统性不足：文本编码器选择不够重视、提示词改写欠缺原则、caption 策略未被充分优化、缺乏推理时智能路由。

### 2.3 核心设计理念

Boogu 的设计围绕一个中心理念：**理解是连接模糊用户需求与精确视觉生成的桥梁**。理解被分解为三个互补维度：

1. **理解用户意图**：通过更强大的文本编码器（传感器）和更智能的提示词改写器（翻译器）来实现
2. **理解训练图像**：通过更好的 caption 策略来确保模型学到正确的概念、属性和组合模式
3. **智能推理时策略**：通过 Agentic 管线动态选择模型变体和应用推理时增强技术

## 第 3 章 系统设计

### 3.1 整体架构

Boogu-Image-0.1 采用基于 Diffusion Transformer (DiT) 的架构，以 Flow Matching 为生成框架。核心组件包括：

- **文本编码器**：Qwen3-VL-8B，平衡文本理解能力与参数量
- **DiT 生成主干**：基于流匹配的扩散 Transformer
- **Agentic 推理层**：推理时管线，包含 Prompt Rewriter、Model Router、Reflection
- **轻量模块**：基于单流 Transformer 的 Refiner/Prompt Tuning 组件

模型支持三种分辨率：512×512、1024×1024、2048×2048。

### 3.2 文本编码器选择

Boogu 团队将文本编码器系统地视为文本到图像模型的"传感器"（sensor）。通过分析不同规模编码器的影响，确认更强的 LLM 提供了更强的文本编码能力。最终选择 Qwen3-VL-8B 作为文本编码器，平衡编码能力与开源社区可接受的参数量预算。

这一选择的关键意义在于：**文本编码器的能力是生成质量的上限前哨**——编码器无法准确捕获的语义细节，DiT 生成器不可能正确还原。Boogu 的研究表明，这在长提示词、多约束指令、隐式上下文等场景中尤为关键。

### 3.3 Agentic Image Generation

Boogu 在推理时通过 Agent 封装生成过程，实现"理解驱动生成"：

1. **Agentic Prompt Rewriting**：Agent 解释用户原始需求，将其重写为更精确、更适合生成器的格式。与传统的"附加 LLM 辅助改写"不同，Boogu 的改写遵循"翻译器而非增强器"原则——保留用户的原始意图而非过度创作

2. **Model Router**：Agent 在 Base（高品质）和 Turbo（快速推理）等变体之间动态选择。简单请求走快速通道，复杂请求分配更强模型，优化速度-质量权衡

3. **Reflection**：应用推理时反思机制，对生成结果进行多轮修正，进一步提升与用户意图的匹配度

4. **推理时缩放**：如图 1（右）所示，增加推理计算量持续提升 T2I 生成质量，遵循可预测的 scaling 曲线

### 3.4 Prompt Rewriter 的设计原则

Boogu 将 Prompt Rewriter 定位为"翻译器，而非增强器"：

- **不改变意图**：保留用户原始需求的核心语义
- **不添加信息**：对模糊部分做合理推理（填充常识上下文），但不过度创作
- **结构化输出**：将自然的用户语言转换为生成器友好的结构化格式
- **中文适配**：对中文输入做专门优化，保留中文特有的表达方式和文化意象

这一原则通过更强 VLM 骨干（同 Qwen3-VL-8B）和更优的改写规则实现。因为 Prompt Rewriter 在生成器上游运行，其改进会传播到所有下游模块。

### 3.5 Model Router

不同用户请求对生成能力的要求差异巨大：渲染一个孤立物体远比组合多主体场景容易。Model Router 识别请求复杂度，动态选择：

- **Base 变体**：最高质量，适合长文本渲染、密集场景
- **Turbo 变体**：快速推理，适合简单请求
- **Thinking 变体**：通过提示词缩放（Prompt Scaling）增强文本渲染精度

实验显示（Figure 1 右），增加推理时间一致性地提升质量，且不同变体在推理时质量曲线上占据不同位置。

### 3.6 Caption 策略

Boogu 认为 caption 策略从根本上决定了模型能学到什么：生成器只能从 caption 中实际描述的概念、属性和组合模式中学习。遵循三条原则：

1. **全面覆盖**：不仅描述显著物体，还要覆盖风格、空间布局、数量、细粒度属性
2. **图像-文本对齐**：确保 caption 准确反映图像内容
3. **多样性**：不同训练样本的 caption 风格和粒度保持多样，避免模型过拟合到单一描述模式

## 第 4 章 训练与推理优化

### 4.1 训练管线

Boogu-Image-0.1 采用分阶段训练策略，从零开始训练（scratch）：

**基础训练阶段**：
- 数据：187M 开源图像 + 47M Boogu Syllabus 数据
- 分辨率：512×512 → 1024×1024 → 2048×2048 渐进式
- 训练 Epoch：1（Base 512×512），2（Turbo 1024×1024）
- Global Batch Size：1280（512×512），2048（1024×1024），1024（2048×2048）

**关键训练超参数**：

| 超参数 | T2I 512×512 | T2I 1024×1024 | T2I 2048×2048 |
|--------|-------------|--------------|--------------|
| 学习率 | 1e-4 | 3e-5 | 3e-5 |
| Warmup Steps | 1200 | 800 | 800 |
| 学习率调度 | Constant with Warmup | — | — |
| Adam β1, β2 | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 |
| Weight Decay | 0.01 | 0.01 | 0.01 |
| Max Grad Norm | 1.0 | 1.0 | 1.0 |
| SNR 策略 | Dynamic Logit Normal | — | Rectified Dynamic Logit Normal |
| 混合精度 | bf16 | bf16 | bf16 |
| 分片策略 | HYBRID_SHARD_ZERO2 | — | HYBRID_SHARD |

**图像编辑（I2I）训练**：
- 数据：11M T2I + 11M I2I 数据
- 分辨率：2048×2048
- Global Batch Size：448
- 参考图像 Dropout：0.01

### 4.2 数据管线

Boogu 的数据管线遵循"质量优先于数量"原则：
- 总数据量仅 208.62M 唯一图像——显著少于许多大规模训练方案
- 通过 Boogu Syllabus（47M 策划数据）注入高质量训练样本
- 数据过滤和 Caption 设计经过系统优化

### 4.3 Boosted Orthogonal Guidance (BOG)

BOG 是一种免训练推理时增强方法，针对 DiT 的 Classifier-Free Guidance (CFG) 的已知问题：高引导尺度导致过饱和、色调扁平化、细节平滑等伪影。

**核心洞察**：DiT 的预测输出在本质上是**空间矩阵**（shape [B, C, H, W]），而不是扁平向量。传统 CFG 采用向量级归一化，丢弃了空间结构这一关键归纳偏置。BOG 将每个去噪步的预测视为 2D 矩阵，应用矩阵级正交分解。

**算法流程**：
1. 计算原始引导方向：$$\Delta D(i) = D_\theta(x_t, t, c) - D_\theta(x_t, t, c_\emptyset)$$
2. 动量平滑：应用 Rolling-Sum Momentum（η=0.9, ρ=0.1），避免方向抖动
3. 矩阵归一化：通过 Newton-Schulz 迭代近似计算最优半正交逼近，保留矩阵的秩
4. 矩阵正交分解：将引导方向分解为与条件预测平行和正交的分量，保留信息量更大的正交份量
5. 应用修正后的 BOG 引导：$$D_{BOG}(i) = D_\theta(x_t, t, c) + (\omega - 1) \hat{\Delta D}(i)$$

**可配置参数**：BOG Interval ($\Delta_{BOG}$) 控制 BOG 应用频率。默认 $\Delta_{BOG}=2$（偶步 BOG，奇步标准 CFG），在增强效果和结构稳定性间取得平衡。

BOG 生成的图像具有更丰富的微结构、更优的色调深度和更自然的"电影感"光影氛围，但可能增加结构扭曲和文本渲染错误的概率。

## 第 5 章 实验评估

### 5.1 评估方法论

Boogu 团队重新思考了开源研究评估的标准。通过对 6 个近期 T2I 模型的分析（Figure 5），他们发现 GenEval 和 DPG-Bench 与人类偏好存在显著秩反转——最强的人类偏好模型 GPT-Image-2 在两项基准上仅排中游。原因包括：基准与真实应用场景脱节、基准接近饱和（得分差异缺乏统计意义）、数据污染（训练集可能包含公开评估集）。

基于此，Boogu 主要依据**人类偏好评估**（Boogu Arena Elo）和**新发布的未污染基准**（Qwen-Image-Bench），传统基准（GenEval, DPG-Bench）仅作补充参考。

### 5.2 Boogu Arena

Boogu 自建了与 LMArena 协议一致的盲评基准：

- **提示词设计**：三个核心类别（摄影/电影感、文本渲染、风格化艺术，每类约 400 关键词）× 提示词长度（短:中:长=3:4:3）× 用户角色（新手:中级:专业=5:3:2，27 角色）
- **双语提示词**：1200 提示词同时提供中文和英文版本
- **盲评机制**：匿名模型配对投票，"A更好/B更好/平局好/平局差"四选项
- **总计**：9 模型 × 4000+ 投票

**Boogu Arena 结果**（Figure 6）：

| 模型 | 总体 Elo | 摄影 Elo | 文本 Elo | 风格化 Elo |
|------|---------|---------|---------|----------|
| GPT-Image-2 (Closed) | 1196 | 1137 | 1246 | 1298 |
| Nano-Banana-Pro (Closed) | 1087 | 1119 | 1103 | 985 |
| **Boogu-Image-0.1-Turbo-Thinking** | **1048** | **1056** | **1029** | **1060** |
| Seedream-5.0-Lite (Closed) | 1032 | 1020 | 1017 | 1086 |
| **Boogu-Image-0.1-Turbo** | **1021** | **1028** | **1008** | **1030** |
| Qwen-Image-Max-2025-12-30 | 988 | 952 | 1028 | 958 |
| Z-Image-Turbo | 960 | 1010 | 965 | 969 |
| Qwen-Image-2.0-2026-03-03 | 946 | 951 | 902 | 869 |
| HiDream-O1-Image | 868 | 850 | 889 | 850 |

Boogu 模型在所有类别中领先开源层级，仅落后于 GPT-Image-2 和 Nano-Banana-Pro。Thinking 变体在文本渲染类别中的增益最为显著（+21 Elo vs 非 Thinking）。

**Boogu Arena vs LMArena 一致性**（Figure 7）：Pearson r=0.986，Spearman ρ=1.0，表明 Boogu Arena 忠实复现了 LMArena 的模型排序。

### 5.3 Qwen-Image-Bench

Qwen-Image-Bench 是 Boogu 模型开发冻结后发布的新基准，不易受数据泄漏影响，被视为更可靠的测试平台。

**中文提示词结果**（Table 1）：

| 模型 | 许可 | Quality | Aesthetics | Alignment | Realism | Creativity | Overall |
|------|------|:-------:|:----------:|:---------:|:-------:|:----------:|:-------:|
| GPT-Image-2 | Closed | 58.65 | 67.53 | 65.85 | 57.38 | 75.23 | **64.69** |
| Nano-Banana-2.0 | Closed | 54.77 | 61.08 | 62.40 | 54.28 | 67.05 | 59.82 |
| Nano-Banana-Pro | Closed | 55.67 | 60.26 | 61.25 | 54.07 | 66.23 | 59.45 |
| **Boogu-Image-0.1-Base-Thinking** | **Apache-2.0** | **50.58** | **55.20** | **55.99** | **47.35** | **56.74** | **53.57** |
| **Boogu-Image-0.1-Turbo-Thinking** | Apache-2.0 | 50.89 | 54.36 | 56.28 | 47.67 | 53.33 | 53.13 |
| Qwen-Image-2512 | Apache-2.0 | 51.76 | 54.74 | 52.72 | 47.00 | 50.19 | 52.06 |
| **Boogu-Image-0.1-Turbo** | Apache-2.0 | 51.24 | 53.50 | 53.54 | 46.11 | 48.91 | 51.53 |
| **Boogu-Image-0.1-Base** | Apache-2.0 | 50.41 | 53.14 | 52.91 | 45.42 | 48.62 | 50.96 |
| HunyuanImage-3.0 | Other | 50.35 | 53.57 | 52.00 | 44.31 | 49.12 | 50.81 |

**英文提示词结果**（Table 2）：

| 模型 | 许可 | Quality | Aesthetics | Alignment | Realism | Creativity | Overall |
|------|------|:-------:|:----------:|:---------:|:-------:|:----------:|:-------:|
| GPT-Image-2 | Closed | 59.09 | 68.48 | 65.78 | 59.40 | 75.34 | **65.23** |
| **Boogu-Image-0.1-Base-Thinking** | **Apache-2.0** | **51.52** | **55.89** | **55.58** | **47.23** | **56.24** | **53.73** |
| **Boogu-Image-0.1-Turbo-Thinking** | Apache-2.0 | 51.92 | 56.28 | 55.67 | 47.89 | 52.70 | 53.58 |

Boogu 开源模型在中文和英文提示下均为开源最优。Thinking 变体在 Creativity 维度提升最显著（CN: 48.62 → 56.74，+8.12 点），符合理解驱动设计的预期。

### 5.4 LongText-Bench

LongText-Bench 评估长文本渲染能力（海报、幻灯片、文档等）。

| 模型 | 许可 | AVG | EN | ZH |
|------|------|:---:|:--:|:--:|
| Seedream-4.5 | Closed | **0.988** | 0.989 | 0.987 |
| **Boogu-Image-0.1-Turbo-Thinking** | **Apache-2.0** | **0.971** | 0.957 | **0.985** |
| **Boogu-Image-0.1-Turbo** | Apache-2.0 | 0.961 | 0.944 | 0.977 |
| **Boogu-Image-0.1-Base** | Apache-2.0 | 0.961 | 0.952 | 0.969 |
| GPT-Image-2 | Closed | 0.961 | 0.960 | 0.961 |
| Qwen-Image-2512 | Apache-2.0 | 0.961 | 0.956 | 0.965 |

Boogu 中文文本渲染表现突出（Turbo-Thinking ZH 0.985，仅次于 Seedream-4.5）。但 Base 模型在密集文本渲染（>100 字符）下比 Turbo 表现更稳定，Turbo 在密集区域可能引入可见伪影。

### 5.5 ImgEdit-Bench 图像编辑

| 模型 | 许可 | Add | Adjust | Extract | Replace | Remove | BG | Style | Hybrid | Action | Overall |
|------|------|:---:|:------:|:-------:|:-------:|:-----:|:--:|:-----:|:------:|:------:|:-------:|
| **Boogu-Image-0.1-Edit-Thinking** | **Apache-2.0** | 4.59 | 4.64 | **4.32** | 4.69 | **4.85** | 4.60 | 4.94 | **4.26** | **4.83** | **4.64** |
| JoyAI-Image-Edit | Apache-2.0 | 4.63 | 4.52 | **4.32** | 4.71 | 4.76 | 4.53 | 4.88 | 4.09 | 4.69 | 4.57 |
| FireRed-Image-Edit | Apache-2.0 | 4.55 | 4.66 | 4.34 | **4.75** | 4.58 | 4.45 | **4.97** | 4.07 | 4.71 | 4.56 |
| **Boogu-Image-0.1-Edit** | Apache-2.0 | **4.71** | 4.50 | 3.69 | 4.65 | 4.75 | 4.44 | 4.94 | 4.04 | 4.90 | 4.51 |
| Seedream-5.0-Lite | Closed | 4.93 | **4.69** | 3.01 | 4.41 | 4.45 | **4.65** | 4.93 | 3.91 | 4.82 | 4.42 |
| Nano-Banana-Pro | Closed | 4.44 | 4.62 | 3.42 | 4.60 | 4.63 | 4.32 | 4.97 | 3.64 | 4.69 | 4.37 |

Boogu-Image-0.1-Edit-Thinking 在 ImgEdit-Bench 上以 4.64 的总分排名第一，在 Remove（4.85）、Hybrid（4.26）、Action（4.83）子任务上领先。Thinking 变体 vs Base 的增益集中体现在 Extract 任务（4.32 vs 3.69），表明显式推理有助于解析和定位复杂编辑指令。

### 5.6 传统基准结果

**GenEval**（Table 4，补充参考）：

| 模型 | Single Obj. | Two Obj. | Counting | Colors | Position | Attribute | Overall |
|------|:-----------:|:--------:|:--------:|:------:|:--------:|:---------:|:-------:|
| HiDream-O1-Image | 1.00 | 0.99 | 0.79 | 0.89 | 0.93 | 0.78 | **0.90** |
| GPT-Image-2 | 0.99 | 0.98 | 0.85 | 0.93 | 0.85 | 0.77 | 0.89 |
| FLUX.2-Dev | 1.00 | 0.99 | 0.79 | 0.93 | 0.73 | 0.78 | 0.87 |
| **Boogu-Image-0.1-Base** | 0.99 | 0.95 | 0.80 | 0.84 | 0.85 | 0.68 | **0.85** |
| **Boogu-Image-0.1-Turbo-Thinking** | 1.00 | 0.94 | 0.88 | 0.83 | 0.70 | 0.68 | 0.84 |

**DPG-Bench**（Table 5，补充参考）：

| 模型 | Global | Entity | Attribute | Relation | Other | Overall |
|------|:------:|:------:|:---------:|:--------:|:-----:|:-------:|
| HiDream-O1-Image | 95.15 | 92.32 | 93.74 | 92.88 | 90.25 | **89.83** |
| **Boogu-Image-0.1-Turbo** | 88.54 | 91.67 | 92.19 | 93.20 | 93.83 | **88.35** |
| Qwen-Image | 91.32 | 91.56 | 92.02 | 94.31 | 92.73 | 88.32 |
| FLUX.2-Dev | 92.20 | 91.36 | 93.28 | 93.52 | 89.72 | 87.57 |
| GPT-Image-2 | 87.27 | 91.91 | 90.85 | 91.59 | 91.58 | 85.98 |

### 5.7 定性比较

Boogu 在 Boogu Arena 的三个类别中提供了系统的定性比较（Figures 8-12）：

- **摄影/电影感**（Figure 8）：Boogu-Image-0.1-Turbo 在提示词对齐和视觉质量上接近 GPT-Image-2 和 Nano-Banana-Pro
- **风格化艺术**（Figure 9）：支持日系动漫、厚涂油画、扁平插画等多种风格
- **文本渲染**（Figure 10）：短到中文本（中 100 字/英 100 词）精确渲染，准确率接近顶级闭源系统
- **密集文本渲染**（Figure 11-12，Base 2K 分辨率）：Base 模型在密集排版场景下输出最连贯

图像编辑方面（Figures 13-15），Boogu-Image-0.1-Edit 在添加、移除、替换、提取等任务中表现一致，在场景文本编辑（中英文字替换/翻译/重排）上尤为突出。

## 第 6 章 代码实现详解

### 6.1 仓库结构

官方仓库 https://github.com/Boogu-Project/Boogu-Image 包含：

- **模型权重发布**：Base, Turbo, Edit, Edit-Turbo 变体
- **推理代码**：支持 T2I 生成和 I2I 编辑
- **ComfyUI 集成**：原生支持 ComfyUI（无需额外安装）
- **许可证**：Apache 2.0

模型权重托管在 HuggingFace（https://huggingface.co/Boogu），包括：
- Boogu/Boogu-Image-0.1-Base
- Boogu/Boogu-Image-0.1-Turbo
- Boogu/Boogu-Image-0.1-Edit
- Boogu/Boogu-Image-0.1-Edit-Turbo

### 6.2 模型变体

| 变体 | 用途 | 特点 |
|------|------|------|
| **Base** | 高质量 T2I 基础模型 | 最高质量，密集文本渲染最优（2K 分辨率） |
| **Turbo** | 快速 T2I 推理 | 4 步蒸馏推理，速度-质量平衡 |
| **Edit** | 图像编辑 | 基于 Base 的 I2I 模型 |
| **Edit-Turbo** | 快速图像编辑 | 4 步蒸馏编辑推理 |
| **Thinking 变体** | 增强版（各变体可选） | 推理时提示词缩放，增强指令遵循 |

### 6.3 轻量模块

Boogu 使用基于单流 Transformer 的轻量模块作为 Refiner 和 Prompt Tuning 组件。该模块采用经典设计（RMS Norm + RoPE + QKV 投影 + Feed Forward），改动极小（Figure 37）。

## 第 7 章 局限性与未来方向

### 7.1 主要局限

**密集文本渲染**：虽然 Base 模型在 2K 分辨率下能处理超密集文本（>100 字符），但 Turbo 变体在密集区域仍会引入可见伪影。LongText-Bench 主要评估短序列（<100 字符）的 OCR 级准确率，不评估视觉保真度。

**BOG 副作用**：Boosted Orthogonal Guidance 尽管增强了电影感氛围，但可能增加结构扭曲的概率，并降低文本渲染的成功率。推荐 $\Delta_{BOG}=2$ 的默认设置在增强效果和稳定性间取得平衡。

**评估标准不完善**：传统基准（GenEval, DPG-Bench）与人类偏好脱节；ImgEdit-Bench 依赖 VLM 评分，无法完全对齐人类审美；LongText-Bench 主要评估 OCR 准确率而非视觉质量。

**编辑能力偏好反转**：ImgEdit-Bench 上 Nano-Banana-Pro 得分 4.37（低于 Boogu 的 4.64），但人类评估显示 Nano-Banana-Pro 在真实编辑场景中表现更好，说明当前 VLM 评分工具的局限性。

### 7.2 未来方向

1. **更高分辨率**：支持更大分辨率以满足专业设计需求
2. **更快推理**：进一步蒸馏和优化 Turbo 变体
3. **更好的评估标准**：建立与人类偏好更一致的自动评估方法
4. **场景文本理解深化**：在密集排版场景中提升文本渲染的一致性
5. **推理时缩放**：探索更高效的推理时计算分配策略
6. **更广泛的编辑能力**：覆盖多步编辑、多区域编辑、主题一致生成等更丰富的编辑操作

### 7.3 对开源社区的启示

Boogu-Image-0.1 的核心经验教训：

1. **理解是成本效益最高的改进方向**：在有限计算预算下，系统性提升理解能力带来的增益远超单纯增加参数或数据
2. **评估方法论至关重要**：使用与人类偏好对齐的评估标准，避免被饱和基准给出虚假信心
3. **工程细节即胜负手**：数据过滤策略、caption 设计、提示词改写规则等看似细节的决策在大规模部署中至关重要
4. **Agentic 推理时策略的低成本高回报**：不修改模型权重即可通过推理时管线获得显著质量提升
