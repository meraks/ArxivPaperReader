> **论文**：Mage-VL: An Efficient Codec-Native Streaming Multimodal Foundation Model
> **作者**：Microsoft Mage Team
> **arXiv ID**：2607.24904
> **发表时间**：2026-07
> **许可协议**：CC BY 4.0
> **代码仓库**：https://github.com/microsoft/Mage
> **模型仓库**：https://huggingface.co/collections/microsoft/mage

## 第 1 章 概述

### 1.1 一句话定位

Mage-VL 提出了一种 codec-native（编解码原生的）流式多模态基础模型，通过将视频建模为稀疏 codec token 流并在视觉前端消除时空冗余，在 4B 参数规模下实现了卓越的离线推理性能与高达 3.5× 的实时流式推理加速。

### 论文图表总览

| 编号 | 内容 | 章节 |
|------|------|------|
| **Figure 1** | 引言：VLM 的 Moravec 悖论示意 | 第 1 章 |
| **Figure 2** | Mage-ViT 架构：codec 驱动的稀疏 patch 选择 | 第 3 章 |
| **Figure 3** | 主动流式推理架构：System 1 门控 + System 2 LLM | 第 4 章 |
| **Figure 4** | AI4AI 描述提示优化流水线 | 第 6 章 |
| **Figure 5** | 流式事件数据构建格式 | 第 4 章 |
| **Figure 6** | 变分辨率缩放：Mage-ViT 单调提升 vs 固定分辨率基线退化 | 第 5 章 |
| **Figure 7** | 传统 HEVC vs 神经 DCVC-RT 的 codec 鲁棒性对比 | 第 5 章 |
| **Table 1** | Mage-ViT 图像+视频表示质量对比 | 第 5 章 |
| **Table 2** | 图像理解基准对比（文档/OCR/通用VQA/空间智能） | 第 5 章 |
| **Table 3** | 视频理解+时序定位+空间推理基准对比 | 第 5 章 |

### 1.2 核心贡献

1. **Codec-Native 主动流式架构**：将视频流建模为 codec 驱动的稀疏 token 流，通过 System 1（轻量级事件门控）+ System 2（因果 LLM 解码器）实现事件驱动的实时交互，视觉 token 消耗减少 75%+（64 帧预算仅用 4096 token）。

2. **Mage-ViT：高效视觉编码器**：从零训练，仅用约 560M 无标注图像和 100M 无标注视频帧即匹配或超越 SigLIP2（训练于数十亿图像-文本对）等旗舰视觉编码器。同时支持变分辨率输入，视觉 token 预算越大性能单调提升。

3. **AI4AI 数据流水线与 7 项经验发现**：提出 prompt-code 联合优化 + 多智能体评估 + 人工审核的闭环数据优化；系统性给出预训练数据效率、变分辨率缩放律、codec 加速、VideoQA SFT 冗余性、运动-空间协同等七项经验性发现。

### 1.3 关键结果速览

- **图像理解**：Mage-VL-4B 在 DocVQA (95.14%) 和 ChartQA (84.88%) 上超越同参数规模 Qwen3-VL-4B (94.69%/83.96%)，在 CV-Bench-3D (94.75%) 上大幅领先（Qwen3-VL-4B: 92.30%）
- **视频理解**：在 LongVideoBench (61.3% vs 57.7%)、MLVU (68.7% vs 61.5%)、VideoMME (64.0% vs 59.7%) 上大幅领先 Qwen3-VL-4B（MV-Bench 65.1% vs 66.7% 略低）
- **空间推理**：在 VSI-Bench (64.3% vs 53.3%)、EmbSpatial (82.67% vs 77.50%) 上显著领先同参数模型
- **效率**：codec 流式 tokenization 实现最高 3.5× 的 wall-clock 推理加速
- **跨参数级竞争力**：4B 参数在绝大多数基准上超越 Phi-4-reasoning-vision（15B，3.75× 参数），仅在 AI2D、CharXiv-Reason 等少数基准上略低

## 第 2 章 研究背景与动机

### 2.1 VLM 的 Moravec 悖论

现有 Vision-Language Models（VLM）在复杂离线视觉推理（如代码解读、密集文档理解、几何问题求解）上表现出色，但在简单流式感知任务上的计算效率极低。论文将其称为 Moravec 悖论的现代表现——高阶推理易而基础感知难。例如 Qwen-VL、InternVL、LLaVA-OneVision 等模型对稀疏采样的关键帧做逐帧密集编码，当视频时长增加时计算量呈线性增长。

### 2.2 现有方法的局限性

**均匀帧采样的结构性缺陷**：主流 VLM 的视觉编码器（SigLIP2、MoonViT 等）预训练于大规模静态图像-文本对。处理视频时采用固定分辨率的均匀帧采样策略，完全忽略时空冗余——静态背景区域在相邻帧间反复编码，即使只有小部分场景发生变化。计算量随视频时长快速增长。

**现有 token 压缩方法的不足**：DynamicViT、ToMe、FastV 等自适应剪枝/合并方法在编码后压缩，仍有大量冗余；LLaMA-VID、SlowFast-LLaVA 等视频 VLM 选择整帧或压缩特征，无法解决帧间大量重复编码的问题。代码化表示（CoViAR、Video-LaVIT）已展示压缩域识别的潜力，但尚未扩展到端到端 VLM 训练。

### 2.3 生物视觉的启示

自然视觉系统在严格资源约束下运作：视网膜前端过滤冗余背景信号、优先响应动态运动；高级感知通过双系统过程——快速低延迟的「何时反应」通路和深思熟虑的复杂推理系统——协同工作。Mage-VL 受此高效双过程设计启发。

## 第 3 章 Mage-ViT：从零训练的 Codec 视觉编码器

### 3.1 架构设计

Mage-ViT 采用 **Codec-ViT 架构**，由三个核心组件构成：

**Codec 驱动的 patchifier**：将输入划分为 $16 \times 16$ 像素的 patch，然后基于 codec 的位分配估计进行稀疏选择。对多帧输入，估计每帧每个 patch 的重要性张量 $S \in \mathbb{R}_{\ge 0}^{T \times H \times W}$。默认 codec 为 HEVC/H.265，$S$ 定义为 P-frame 的运动向量模长和残差能量的加权组合。I-frame 的 patch 全部保留，P-frame 仅保留 top-$k$ 重要的 patch。对 64 帧、每帧 $16 \times 16$ tokens 的片段，token 预算 $B=4096$，实现约 75% 的 token 减少。

**ViT trunk**：24 层 pre-norm Vision Transformer，隐藏维度 1024，16 注意力头，GELU MLP 4× 扩展，使用 Flash Attention 2。

**共享 3D 旋转位置编码**：在未剪枝的时空网格上应用 3D RoPE，即使大量 patch 被 codec 前端丢弃，也能保留空间-时间位置关系。

### 3.2 预训练

**数据**：约 560M 图像（来自 LAION-400M、COYO-700M、OBELICS 等）和 100M 视频帧（HowTo100M、Panda-70M），涵盖自然照片、文档、图表、室内场景、机器人轨迹等。

**目标**：大规模聚类判别（cluster-discrimination）——提取 MetaCLIP 特征后做 K-means 聚类，优化负采样聚类判别损失，使语义相关的样本聚在相近表示空间中。

**两阶段训练**：
- **Stage 1**：变分辨率图像预训练（224–448 分辨率，1:2–2:1 宽高比），使用 AdamW、学习率 $10^{-3}$、cosine decay
- **Stage 2**：联合图像和视频预训练，视频 256 分辨率、64 帧/clip、token 预算 4096，学习率降至 $5 \times 10^{-5}$

### 3.3 与旗舰编码器的对比

**图像基准（线性探针）**：

| 方法 | 骨干 | DTD | CIFAR-10 | SUN397 | Food-101 | ImageNet-1K |
|:----|:---|:---:|:--------:|:-----:|:--------:|:----------:|
| SigLIP2 | ViT-L/16 | 85.90 | 98.31 | 82.36 | **95.85** | 85.92 |
| MetaCLIP2 | ViT-L/14 | 85.48 | 98.76 | 79.75 | 93.93 | 82.74 |
| AIMv2 | ViT-L/14 | 86.28 | 98.95 | 81.12 | 94.76 | 85.26 |
| DINOv3 | ViT-L/14 | 86.81 | 99.17 | 80.11 | 94.96 | 85.38 |
| MoonViT | ViT-SO400M/14 | 82.77 | 96.81 | 77.83 | 88.56 | 77.18 |
| OV-Encoder | ViT-L/14 | 85.48 | 99.06 | 80.60 | 94.79 | 84.54 |
| **Mage-ViT** | ViT-L/16 | 85.96 | **99.33** | 82.01 | 95.60 | 85.69 |

Mage-ViT 在 CIFAR-10 上取得最高（99.33%），ImageNet-1K 上紧追 SigLIP2（85.69% vs 85.92%）。SigLIP2 预训练于超 10B 图像-文本对，而 MoonViT（Kimi-VL 编码器）性能显著低于 Mage-ViT。

**视频基准（注意探针）**：

| 方法 | 骨干 | 模式 | Diving-48 | HMDB-51 | K400 |
|:----|:---|:---:|:--------:|:-------:|:----:|
| SigLIP2 | ViT-L/16 | Chunk | 62.30 | 84.61 | 83.93 |
| DINOv3 | ViT-L/14 | Chunk | 61.30 | 79.70 | 83.90 |
| MoonViT | ViT-SO400M/14 | Chunk | 43.95 | 77.17 | 79.73 |
| OV-Encoder | ViT-L/14 | Codec | **65.32** | 84.80 | 84.87 |
| **Mage-ViT** | ViT-L/16 | Chunk | 60.45 | 85.13 | 84.83 |
| **Mage-ViT** | ViT-L/16 | Codec | 64.14 | **85.17** | 84.66 |

在 codec 稀疏采样模式下，Mage-ViT 较 chunk 模式在 Diving-48 上提升 3.69pp（60.45% → 64.14%），验证了 codec 驱动稀疏 tokenization 对运动密集型任务的优势。

## 第 4 章 Mage-VL：统一流式多模态模型

### 4.1 统一架构

Mage-VL 将 Mage-ViT 作为共享视觉编码器，通过轻量级 2 层 MLP projector 连接到 Qwen3-4B-Instruct-2507 因果语言解码器。单个模型同时支持：

- **静态图像**：单张空间 token 序列
- **离线视频**：按时间顺序拼接的 codec-token 窗口
- **主动流式**：System 1 门控 + System 2 解码器

### 4.2 主动流式机制

一个轻量级**认知门控**（cognition gate）持续评估流入 codec-window 的视觉特征。给定当前流式表示 $\mathbf{h}_t$，门控预测 $p_{\text{speak}} = g(\mathbf{h}_t)$，当 $p_{\text{speak}} \ge \tau$ 时触发生成。背景段和非响应时刻门控保持关闭。

流式推理时，新增 codec-window 由感知通路增量处理。认知门控评估累积的流式记忆，响应生成使用最近的 $N$ 个 codec 视觉段的局部滑动窗口。一个文本查询可在任何时刻插入，而主动响应由认知门控控制。

### 4.3 训练数据

| 数据类型 | 数量 | 说明 |
|:--------|:---:|:------|
| 图像-描述对 | ~350M | LLaVA-OneVision-1.5 中训练语料 85M + 过滤约 10 亿对得到 265M，均用优化提示重描述 |
| 图像指令样本 | ~54M | LLaVA-OneVision-1.5、FineVision、OpenBee、LLaVA-OneVision-2 空间/GUI 数据 |
| 视频-描述样本 | ~7.95M | 4.2M ≤30s, 2.7M 30–60s, 0.7M 60–180s, 350K 10–15min |
| 流式事件样本 | ~3.35M | 2.8M 来自 180s 描述语料，0.54M 来自 MatchTime/LiveCC/StreamingVLM |

### 4.4 五阶段渐进训练

| 阶段 | 目标 | 核心数据 | 关键设计 |
|:---|:-----|:--------|:--------|
| Stage 1 | 多模态对齐 | 350M 图像描述 + 4.2M 短视频描述 | 较 LLaVA-OneVision2 多 4 倍描述数据；视频描述从一开始就引入 |
| Stage 2 | 指令微调 + 时序基础 | 54M 图像指令 + 3.4M 视频描述（30-180s） | 提高 OCR/文档/图表/空间等权重，短时视频保持 |
| Stage 3 | 时域扩展 | 20M 图像指令 + 长视频数据 | 教会跨分钟的场景演进和长程依赖 |
| Stage 4 | Codec 长上下文适应 | 350K 长视频 codec 流 + 40M 图像指令 | 将视频转换为 codec-native 表示，token 预算下降为 1/8 |
| Stage 5 | 主动流式对齐 | 3.35M 流式样本 | 冻结所有骨干，仅训练认知门控；使用类别加权 token 交叉熵损失 |

Stage 5 的损失函数为：
$$
\mathcal{L}_{\text{gate}} = -\sum_t w_{g_t} \log p(g_t \mid g_{<t}, \mathcal{M}_{\text{per}}), \quad g_t \in \{\text{silent}, \text{speak}\}
$$

其中 $\mathcal{M}_{\text{per}}$ 是事件保持特征提取器（EPFE）的循环状态记忆。

## 第 5 章 实验结果

### 5.1 Mage-ViT 表示质量

**图像基准（线性探针）**：

| 方法 | 骨干 | DTD | CIFAR-10 | SUN397 | Food-101 | ImageNet-1K |
|:----|:---|:---:|:--------:|:-----:|:--------:|:----------:|
| SigLIP2 | ViT-L/16 | 85.90 | 98.31 | 82.36 | **95.85** | 85.92 |
| MetaCLIP2 | ViT-L/14 | 85.48 | 98.76 | 79.75 | 93.93 | 82.74 |
| AIMv2 | ViT-L/14 | 86.28 | 98.95 | 81.12 | 94.76 | 85.26 |
| DINOv3 | ViT-L/14 | 86.81 | 99.17 | 80.11 | 94.96 | 85.38 |
| MoonViT | ViT-SO400M/14 | 82.77 | 96.81 | 77.83 | 88.56 | 77.18 |
| OV-Encoder | ViT-L/14 | 85.48 | 99.06 | 80.60 | 94.79 | 84.54 |
| **Mage-ViT** | ViT-L/16 | 85.96 | **99.33** | 82.01 | 95.60 | 85.69 |

Mage-ViT 在 CIFAR-10 上取得最高（99.33%），ImageNet-1K 上紧追 SigLIP2（85.69% vs 85.92%）。SigLIP2 预训练于超 10B 图像-文本对，而 MoonViT（Kimi-VL 编码器）性能显著低于 Mage-ViT。

**视频基准（注意探针）**：

| 方法 | 骨干 | 模式 | Diving-48 | HMDB-51 | K400 |
|:----|:---|:---:|:--------:|:-------:|:----:|
| SigLIP2 | ViT-L/16 | Chunk | 62.30 | 84.61 | 83.93 |
| DINOv3 | ViT-L/14 | Chunk | 61.30 | 79.70 | 83.90 |
| MoonViT | ViT-SO400M/14 | Chunk | 43.95 | 77.17 | 79.73 |
| OV-Encoder | ViT-L/14 | Codec | **65.32** | 84.80 | 84.87 |
| **Mage-ViT** | ViT-L/16 | Chunk | 60.45 | 85.13 | 84.83 |
| **Mage-ViT** | ViT-L/16 | Codec | 64.14 | **85.17** | 84.66 |

在 codec 稀疏采样模式下，Mage-ViT 较 chunk 模式在 Diving-48 上提升 3.69pp（60.45% → 64.14%），验证了 codec 驱动稀疏 tokenization 对运动密集型任务的优势。

### 5.2 图像理解对比

| 基准 | Mage-VL-4B | Qwen3-VL-4B | Phi-4-MM-5.6B | Phi-4-R-V-15B |
|:----|:---------:|:----------:|:------------:|:-------------:|
| **文档理解** | | | | |
| DocVQA-val | **95.14** | 94.69 | 92.79 | 76.20 |
| InfoVQA-val | **80.33** | 79.50 | 71.84 | 55.41 |
| AI2D w/o Mask | 91.87 | 92.20 | 91.35 | **93.81** |
| ChartQA | **84.88** | 83.96 | 83.76 | 83.40 |
| OCRBench | **81.80** | 81.60 | 81.70 | 73.90 |
| MultiDocVQA-val | **87.46** | 87.21 | 46.84 | 58.35 |
| ChartQAPro | **32.57** | 26.79 | 0.13 | 25.38 |
| CharXiv-Reason | 35.20 | 26.10 | 0.10 | **36.10** |
| **通用 VQA** | | | | |
| MMBench-EN-dev | 84.02 | 83.25 | 65.81 | **84.19** |
| MMStar | **67.32** | 62.04 | 61.24 | 59.63 |
| RealWorldQA | 70.46 | **70.85** | 60.65 | 70.72 |
| MME-Perception | **1709.54** | 1703.50 | 1409.66 | 1590.21 |
| SeedBench-Image | **79.32** | 78.87 | 74.48 | 77.97 |
| CV-Bench | **87.79** | 85.37 | 57.09 | 81.31 |
| **空间智能** | | | | |
| CV-Bench-2D | **82.13** | 81.00 | 56.12 | 80.11 |
| CV-Bench-3D | **94.75** | 92.30 | 56.92 | 82.50 |
| EmbSpatial | **82.67** | 77.50 | 41.51 | 72.67 |
| BLINK | 65.11 | 65.10 | 35.24 | 57.80 |
| CrossPoint | **80.00** | 26.90 | 12.20 | 47.73 |
| MMSI-Bench | 28.20 | **31.00** | 28.80 | 25.70 |

Mage-VL-4B 在文档理解（DocVQA 95.14% vs 94.69%）、图表推理（ChartQAPro 32.57% vs 26.79%）、通用 VQA（MMStar 67.32% vs 62.04%）、3D 空间推理（CV-Bench-3D 94.75% vs 92.30%、CrossPoint 80.00% vs 26.90%）上系统性领先同参数规模 Qwen3-VL-4B（MMSI-Bench 28.20% vs 31.00%、RealWorldQA 70.46% vs 70.85% 略低），且以 4B 参数在绝大多数基准上超越 15B Phi-4-reasoning-vision。

### 5.3 视频理解对比

| 基准 | Mage-VL-4B | Qwen3-VL-4B | Phi-4-MM-5.6B | Phi-4-R-V-15B |
|:----|:---------:|:----------:|:------------:|:-------------:|
| **视频 QA** | | | | |
| MV-Bench | 65.1 | **66.7** | 44.9 | 49.2 |
| NextQA | **83.1** | 79.8 | 54.1 | 69.0 |
| VideoMME | **64.0** | 59.7 | 44.7 | 55.3 |
| LongVideoBench | **61.3** | 57.7 | 41.14 | 51.2 |
| LVBench | **41.8** | 39.2 | 25.31 | 34.4 |
| MLVU-dev | **68.7** | 61.5 | 44.18 | 51.8 |
| VideoEval-Pro | **45.2** | 20.7 | 14.35 | 16.8 |
| JumpScore | **45.60** | 3.82 | 3.61 | 1.53 |
| **时序定位** | | | | |
| Timelens-Charades | **50.7** | 43.1 | 4.09 | 20.6 |
| Timelens-ActivityNet | **45.4** | 28.4 | 2.03 | 23.0 |
| Timelens-QVHighlight | **57.4** | 34.9 | 2.47 | 11.6 |
| **空间推理** | | | | |
| VSI-Bench | **64.3** | 53.3 | 24.09 | 25.5 |

在视频理解上，Mage-VL-4B 在绝大多数基准上显著超越所有对比模型。特别值得注意的是 JumpScore（视频时间跳跃检测）上 45.60% vs Qwen3-VL-4B 的 3.82%（11.9× 提升），以及 VideoEval-Pro 上 45.2% vs 20.7%（2.2× 提升）。时序定位方面，Timelens-ActivityNet 45.4% vs Qwen3-VL-4B 28.4%（+17.0pp）、Timelens-QVHighlight 57.4% vs 34.9%（+22.5pp）。

### 5.4 效率分析

在 codec-native tokenization 下，Mage-VL 提供三个配置点：
- **tc32**：最大视觉预算，最强性能
- **tc16**：平衡配置
- **tc8**：低延迟优先

相较均匀帧采样，codec 流式 tokenization 在保持同等性能的同时实现最高 3.5× wall-clock 推理加速。

### 5.5 Codec 鲁棒性

Mage-ViT 默认使用 HEVC/H.265 codec，但对神经 codec（DCVC-RT）同样兼容。在视频 QA 基准上，HEVC 和 DCVC-RT 性能高度一致（MV-Bench 65.1 vs 66.5, NextQA 83.1 vs 82.6, VideoMME 64.0 vs 64.3），而 DCVC-RT 的 token 消耗更低（平均 30.8 vs 33.4 tokens），验证了 Mage-ViT 的 codec 族间鲁棒性。

## 第 6 章 代码实现与开源生态

### 6.1 开源发布

Mage 模型家族以 MIT 许可证在 GitHub 开源（https://github.com/microsoft/Mage），包括：

- **Mage-VL**：codec-native 流式多模态理解模型（4B），代码和 checkpoints 待发布
- **Mage-Flow**：文本到图像生成与指令编辑模型（4B），已发布技术报告

模型权重托管于 HuggingFace（https://huggingface.co/collections/microsoft/mage）。

### 6.2 AI4AI 数据流水线

论文详细描述了 AI4AI 数据优化流水线：使用 GPT-5 作为评分智能体，从完整性、冗余性、连贯性、OCR 保真度四个维度评估描述质量；GitHub Copilot 作为提示优化智能体基于诊断分析提出最小修改；经过 10 轮 evaluate–refine–verify 迭代得到最终系统提示。优化后的描述在 OCR、文档、图表、感知和通用视觉推理基准上持续改善性能。

## 第 7 章 经验发现与局限性

### 7.1 七项关键经验发现

1. **大规模预训练不是视觉编码器的必要条件**：560M 无标注图像 + 100M 视频帧的 Mage-ViT 可匹配数十亿图像-文本对训练的 SigLIP2，说明数据效率和时间对齐比单纯规模更重要。

2. **变分辨率预训练实现单调性能缩放**：Mage-ViT 的表示质量随视觉 token 预算增加单调提升（196→676 tokens，ImageNet 85.69%→86.3%+），而 SigLIP2 等固定分辨率编码器在超出预训练分辨率后性能退化。

3. **Codec-native tokenization 建立卓越的精度-效率前沿**：实现最高 3.5× 加速，token 消耗降至均匀帧采样的 1/8 以下。

4. **显式长视频 VideoQA SFT 是冗余的**：仅在密集视频描述 + 短视频 SFT 上微调即可获得强零样本长视频 QA 能力。

5. **动态视频训练显著增强静态 2D/3D 空间推理**：运动感知训练提升了 CV-Bench-3D、EmbSpatial 等静态空间基准的性能，验证运动与空间结构本质协同。

6. **AI4AI 数据流水线系统性提升数据质量**：agentic 闭环反馈和 prompt-code 协同设计在 OCR、文档、图表等下游基准上带来持续改进。

7. **Zero-Vision SFT 释放多模态 RL 能力**：跳过视觉 SFT 而纯文本推理 SFT 为多模态 RL 提供了计算高效的路径。

### 7.2 局限性

- 当前专注于 4B 参数预算，更大参数规模的 scaling 行为尚未探索
- 生成任务（图像/视频生成）在本工作中未涉及（由同一家族的 Mage-Flow 处理）
- 认知门控的延迟优化（如门控延迟阈值的自动学习）尚未充分研究
- Codec 编码的时间开销在极端低延迟场景下仍需优化