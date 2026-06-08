---
title: "ETCHR: Editing To Clarify and Harness Reasoning"
authors: Beichen Zhang*, Yuhong Liu*, Jinsong Li, Yuhang Zang†, Jiaqi Wang†, Dahua Lin†
institutions: CUHK, Shanghai AI Laboratory, Shanghai Jiao Tong University, Shanghai Innovation Institute, CPII under InnoHK
arxiv: https://arxiv.org/abs/2605.23897
code: https://github.com/InternLM/ETCHR
date: 2026-05-22
keywords: [视觉推理, 图像编辑, 多模态大语言模型, 强化学习, 解耦架构]
---

# ETCHR: Editing To Clarify and Harness Reasoning


> **论文信息**
> - **标题**：ETCHR: Editing To Clarify and Harness Reasoning
> - **作者**：Beichen Zhang, Yuhong Liu, Jinsong Li, Yuhang Zang, Jiaqi Wang, Dahua Lin
> - **arXiv**：[2605.23897](https://arxiv.org/abs/2605.23897)
> - **官方代码**：无官方实现

---

> **通过编辑图像来澄清和利用推理能力——一个解耦的、推理感知的图像编辑器，为多模态大语言模型提供视觉推理辅助。**

---

## 一、研究背景与动机

### 1.1 问题陈述

多模态大语言模型（MLLM）在处理文本 Chain-of-Thought（CoT）推理时已取得显著进展，但当问题依赖于**"看哪里"**（where to look）或**"场景如何变化"**（how a scene would change）时，纯文本推理遇到了根本性瓶颈。

具体而言，以下三类任务对 MLLM 构成严峻挑战：

| 任务类型 | 核心难点 | 示例 |
|----------|---------|------|
| 细粒度感知 | 需要聚焦到图像中极小区域 | "红色汽车旁边的路标上写了什么？" |
| 逻辑推理 | 需要在图像上标记路径/状态 | "从迷宫入口到出口的最短路径经过哪些格子？" |
| 空间/3D 理解 | 需要视角变换来回答 | "从另一个角度看，A 在 B 的左边还是右边？" |

> **类比理解**：想象你在参加一场考试，题目要求你"指出图中两条线的交点坐标"。如果只给你一张模糊的照片让你口述答案，你会很困难；但如果允许你拿一支笔在图上画出辅助线、标记关键位置，问题就变得简单多了。ETCHR 的核心思想就是给 MLLM 配备一支"智能画笔"——一个能理解问题并自动编辑图像的专用编辑器。

### 1.2 现有方法的局限

当前"think with images"范式主要有两条路线：

| 方面 | Tool-based 方法（如 DeepEyes、V*、Thyme） | Unified 方法（如 ThinkMorph、Zebra-CoT） |
|------|-------------------------------------------|------------------------------------------|
| **原理** | 调用固定工具（裁剪、缩放、框选） | 单一模型交替生成文本和图像 token |
| **灵活性** | ❌ 仅支持固定的低层次局部操作 | ✅ 理论上支持任意变换 |
| **编辑质量** | ✅ 工具操作精确 | ❌ 生成式图像质量低于专用编辑器 |
| **感知能力** | ✅ 保留原始图像信息 | ❌ 生成头在感知任务上落后于专用理解模型 |
| **覆盖范围** | ❌ 无法处理迷宫路径、拼图还原、3D 视角变换 | ✅ 范围广但质量差 |

### 1.3 两个关键 Gap

作者通过实验揭示了现成图像编辑器作为推理助手时的两个核心缺陷：

**Gap 1：语言侧 Gap（Language-Side Gap）**

现成编辑器被训练为被动指令跟随者——它们期望显式指令如"在红色汽车周围加一个边框"，但无法从抽象问题如"垃圾桶在黑色椅子的左侧还是右侧？"自动映射到合适的视觉变换。

实验证实：使用 concrete instruction（具体指令）的性能显著优于 abstract question（抽象问题）条件。

**Gap 2：生成侧 Gap（Generation-Side Gap）**

编辑正确率随推理深度增长而急剧下降。在迷宫任务中，路径长度 $L \in \{1, 3, 5, 7, 10\}$ 时：
- $L=1$：准确率接近完美
- $L=10$：准确率趋近于零（即使 prompt 中提供了正确路径）

> **类比理解**：语言侧 Gap 就像是你请了一个只会照菜谱做菜的厨师（需要精确指令），但你只告诉他"我想吃点开胃的"（抽象问题）。生成侧 Gap 则像是即使给了菜谱，当菜品步骤从 3 步增加到 10 步时，厨师出错的概率也会指数级上升。

---

## 二、核心贡献

**1. 揭示两个系统性 Gap**：首次系统性地分析了现成图像编辑器作为推理助手的局限性——语言侧 Gap 和生成侧 Gap，为后续研究提供了清晰的问题框架。

**2. 提出 ETCHR 框架**：一个**解耦的、问题条件化的、推理感知的图像编辑器**，与下游理解模型完全分离，可即插即用地适配不同开源/闭源 MLLM，无需训练理解模型。

**3. 两阶段训练方案**：Stage I 通过 SFT 实现推理模仿（Reasoning Imitation），解决语言侧 Gap；Stage II 通过基于 VLM 奖励的强化学习（Pref-GRPO）实现推理增强（Reasoning Enhancement），缓解生成侧 Gap。

**4. Edit-Verify-Reason 推理流程**：引入反思验证机制，在编辑后让理解模型验证编辑是否包含有用信息，避免错误编辑反而误导推理。

**5. 广泛验证**：跨越 5 个任务族（细粒度感知、图表理解、逻辑推理、拼图还原、3D 理解），在 Qwen3-VL-8B、Gemini-3.1-Flash-Lite 和 1T 参数 MoE 模型 Kimi K2.5 上均取得显著提升。

---

## 三、方法详解

### 3.1 整体架构

ETCHR 采用**解耦架构**，将视觉推理辅助分解为三个独立阶段：

```
┌─────────────────────────────────────────────────────┐
│                 ETCHR Framework                      │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │  Editor   │───>│ Verify   │───>│  Reasoning   │  │
│  │ (FLUX.2)  │    │ (MLLM)   │    │  (MLLM)      │  │
│  │ Stage I+II│    │          │    │              │  │
│  └──────────┘    └──────────┘    └──────────────┘  │
│       ↑                                ↑            │
│  [p_task + q] + i              i_edit or i          │
│                                                     │
│  Training: SFT (Stage I) + Pref-GRPO (Stage II)    │
└─────────────────────────────────────────────────────┘
```

**设计哲学**：编辑器和理解模型**完全解耦**。编辑器负责"让图像更容易理解"，理解模型负责"基于图像回答问题"。这种分离意味着：
- 编辑器可以即插即用，适配任何 MLLM（开源/闭源均可）
- 无需修改或训练理解模型
- 编辑器可以针对推理任务专门优化

### 3.2 Stage I: Reasoning Imitation（推理模仿）

**目标**：将被动渲染器转换为问题条件化编辑器。

**训练数据**：覆盖 5 个推理族

| 推理族 | 数据源 | 编辑类型 | 核心挑战 |
|--------|--------|---------|---------|
| 细粒度感知 | V* 数据集 | 边界框叠加 | 定位微小目标 |
| 图表理解 | RefChartQA | 边界框叠加 | 定位数据区域 |
| 逻辑推理 | 自建迷宫语料 | 路径叠加 | 多步路径规划 |
| 拼图推理 | Spatial-SSRL | 图像还原 | 空间关系推理 |
| 3D 理解 | DL3DV-10K | 视角变换 | 3D→2D 投影 |

**关键技术：Task-level Meta-Prompt**

在问题 $q$ 前添加任务特定的 meta-prompt $p_{\text{task}}$，作为"软路由器"：

$$\text{input} = [p_{\text{task}},\ q,\ \text{image}]$$

这个设计的作用是**分区潜在空间**，抑制跨任务梯度冲突。例如：
- 细粒度感知任务：$p_{\text{task}}$ = "Locate and highlight the relevant region..."
- 迷宫任务：$p_{\text{task}}$ = "Draw the correct path through the maze..."
- 3D 任务：$p_{\text{task}}$ = "Transform the viewpoint to show..."

**训练配置**：

```python
# Stage I SFT 配置
base_model = "FLUX.2-klein-base-9B"  # 9B 参数 Diffusion Transformer
lora_rank = 768                       # LoRA 应用于所有 DiT 线性层
learning_rate = 1e-4
epochs = 1
cfg_scale = 1                         # 无分类器自由引导
inference_steps = 30
framework = "DiffSynth-Studio"
```

> **为什么选择 FLUX.2-klein-base？**
> 1. 它具有 MLLM 风格的编码器，具备足够的语言理解能力来解析复杂问题
> 2. 作为专用图像编辑模型，其生成质量远高于统一模型的生成头
> 3. Diffusion 架构天然支持精细的空间编辑控制

### 3.3 Stage II: Reasoning Enhancement（推理增强）

**目标**：通过强化学习进一步提升编辑质量，特别是对推理深度的鲁棒性。

**数据筛选**：从每个族中选择 2,000 个实例（共 10,000 个），筛选标准：

$$\mathcal{C}(i, i_{\text{gt}}, q, a) = \mathbf{1}[\mathcal{M}(i, q) \neq a \ \wedge \ \mathcal{M}(i_{\text{gt}}, q) = a]$$

即仅保留满足以下条件的样本：
- 理解模型 $\mathcal{M}$ 在原始图像 $i$ 上回答**错误**
- 理解模型 $\mathcal{M}$ 在 ground-truth 编辑 $i_{\text{gt}}$ 上回答**正确**

> **类比理解**：这就像是一个"错题本"策略——只练习那些学生做错了、但老师给了正确提示后能做对的题目。如果原始图像就能答对（太简单）或即使有 ground-truth 编辑也答不对（太难），都不适合用来训练编辑器。

**双奖励设计**：

| 奖励 | 公式 | 优势 | 劣势 |
|------|------|------|------|
| Editing Guidance ($r_{\text{guide}}$) | $\mathbf{1}[\mathcal{M}(i_{\text{edit}}, q) = a]$ | 最忠实于端到端目标 | 受 $\mathcal{M}$ 能力天花板限制 |
| Editing Correctness ($r_{\text{correct}}$) | $\mathbf{1}[\mathcal{J}(i, i_{\text{edit}}, q) = 1]$ | 可超越 $\mathcal{M}$ 天花板 | Judge 模型存在噪声 |

组合奖励：

$$\mathcal{R} = 0.5 \cdot r_{\text{guide}} + 0.5 \cdot r_{\text{correct}}$$

> **为什么需要双奖励？** $r_{\text{guide}}$ 直接衡量编辑是否帮助了下游推理，但如果理解模型本身做不出某道题，这个奖励就无法提供梯度信号。$r_{\text{correct}}$ 通过 VLM-as-Judge 评估编辑本身的质量，可以突破理解模型的天花板，但 Judge 评估可能存在噪声。两者互补。

**优化算法：Pref-GRPO**

Pref-GRPO 是 GRPO 的成对偏好扩展版本：

```python
# Pref-GRPO 核心逻辑（伪代码）
G = 8  # rollout group size
for batch in dataloader:
    # 生成 G 个编辑候选
    edits = [editor.edit(image, question) for _ in range(G)]
    
    # 计算每个编辑的奖励
    rewards = [0.5 * r_guide(edit) + 0.5 * r_correct(edit) for edit in edits]
    
    # 成对偏好比较 + GRPO 梯度更新
    # 选择最佳和最差编辑构建偏好对
    best = edits[argmax(rewards)]
    worst = edits[argmin(rewards)]
    loss = pref_grpo_loss(best, worst, rewards)
    loss.backward()
    optimizer.step()
```

**RL 配置**：

```python
lora_rank = 128       # 比 Stage I 的 768 小，避免过拟合
lora_alpha = 128
learning_rate = 1e-4
rollout_group = 8     # G=8
judge_model = "Qwen3-VL-8B-Instruct"  # 同时作为理解模型和 Judge
```

### 3.4 Edit-Verify-Reason 推理流程

推理时采用三步流程：

```
输入: 图像 i, 问题 q, 元提示 p_task
│
├── Step 1: Edit（编辑）
│   i_edit = Editor(p_task + q, i)
│
├── Step 2: Verify（验证）
│   如果 M 认为 i_edit 包含回答 q 所需的视觉信息:
│       use_edited = True
│   否则:
│       use_edited = False  # 回退到原始图像
│
└── Step 3: Reason（推理）
    如果 use_edited:
        answer = M(i_edit, q)
    否则:
        answer = M(i, q)  # 安全回退
```

> **为什么验证如此关键？** 作者发现编辑错误具有**非对称成本**：一个正确的编辑能提供决定性的视觉引导，但一个错误的编辑会引入结构化的混淆因素（structured confounders），MLLM 几乎无法覆盖这些混淆。验证机制就像是一个"安全网"——当编辑不可靠时宁可不用，也不要引入误导。

---

## 四、实验结果

### 4.1 实验设置

| 维度 | 配置 |
|------|------|
| **评估指标** | Pass@1 准确率 |
| **任务族** | 5 个：细粒度感知、图表理解、逻辑推理、拼图还原、3D 理解 |
| **Backbone** | Qwen3-VL-8B, Gemini-3.1-Flash-Lite, Kimi K2.5 (1T MoE) |
| **对比方法** | Tool-based (DeepEyesV2, Thyme), Unified (ThinkMorph-7B, Bagel-Zebra-CoT), Closed-source (Nano Banana 2) |

### 4.2 主要结果

**跨模型性能提升（Pass@1）**：

| 理解模型 | 无编辑（Baseline） | + ETCHR | **提升** |
|----------|-------------------|---------|---------|
| Qwen3-VL-8B | 55.95 | 60.77 | **+4.82 (+8.6%)** |
| Gemini-3.1-Flash-Lite | 65.08 | 70.55 | **+5.47 (+8.4%)** |
| Kimi K2.5 (1T MoE) | 76.55 | 81.16 | **+4.61 (+6.0%)** |

**关键发现**：ETCHR 的提升在**所有规模的模型**上都保持一致，从 8B 到 1T 参数，说明解耦编辑器的增益不受理解模型能力限制。

**按任务族细分（Qwen3-VL-8B）**：

| 任务族 | Baseline | + ETCHR | **Δ** |
|--------|----------|---------|-------|
| COCO Jigsaw（拼图还原） | 41.1 | 57.2 | **+16.1 (+39.2%)** |
| Maze（逻辑推理） | 27.5 | 38.5 | **+11.0 (+40.0%)** |
| DL3DV-2k（3D 理解） | 70.8 | 78.6 | **+7.8 (+11.0%)** |
| Frozen Lake（逻辑推理） | 9.5 | 13.5 | **+4.0 (+42.1%)** |
| Fine-grained Perception | ~60+ | ~63+ | +3~4 |

**亮点**：结构性任务（拼图 +16.1、迷宫 +11.0）提升最大，正是 Tool-based 方法无法覆盖的领域。

### 4.3 与其他范式对比

| 方法 | 范式 | 平均 Pass@1 | 支持的任务族数 |
|------|------|-------------|---------------|
| DeepEyesV2 | Tool-based | 51.34 | 2/5（仅感知+图表） |
| Thyme | Tool-based | 49.69 | 2/5 |
| ThinkMorph-7B | Unified | 44.05 | 5/5（质量差） |
| Bagel-Zebra-CoT | Unified | 38.27 | 5/5（质量差） |
| Nano Banana 2 | Closed-source | ~58 | 5/5 |
| **ETCHR + Qwen3-VL-8B** | **Decoupled** | **60.77** | **5/5** |

**关键结论**：
- Tool-based 方法在感知/图表上与 ETCHR 可比，但在逻辑/拼图/3D 上**完全无法工作**（固定动作空间不支持）
- Unified 方法覆盖面广但质量低，落后于专用 backbone **10+ 个百分点**
- ETCHR 在感知/图表上与闭源编辑器可比，在结构性任务上**优势更大**

### 4.4 消融实验

| 组件 | 发现 |
|------|------|
| **Stage I SFT** | 在所有 5 个任务族上均有显著提升，是主要贡献来源 |
| **Stage II RL** | 在感知/图表上额外增加 ~1 个百分点，在结构性任务上几乎无增益（GRPO 采样对结构性编辑的语义多样性有限） |
| **单奖励 vs 双奖励** | $r_{\text{guide}}$ 或 $r_{\text{correct}}$ 单独使用均不如组合；Correctness 提供保真度底线，Guidance 提升任务上限 |
| **Verify 机制** | 移除验证导致性能下降，尤其在编辑正确率较低的任务上；验证有效过滤了误导性编辑 |
| **Task-level Meta-Prompt** | 移除后跨任务干扰显著增加，平均下降 2-3 个百分点 |

---

## 五、关键发现与洞察

### 5.1 解耦设计的优势

ETCHR 最核心的设计洞察是**解耦**（Decoupling）。将编辑器和理解模型分开带来了三个关键优势：

1. **即插即用**：无需修改理解模型，可适配任何 MLLM（包括闭源 API 模型如 Gemini）
2. **各取所长**：编辑器用专用 Diffusion 模型（生成质量高），理解模型用专用 MLLM（理解能力强）
3. **独立优化**：编辑器可以独立训练和改进，不影响理解模型的稳定性

> **实践建议**：如果你的应用场景涉及视觉推理，不要试图让一个模型同时做好理解和编辑。分离关注点（Separation of Concerns）在架构设计中是经典原则，ETCHR 证明了它同样适用于多模态 AI。

### 5.2 双奖励的互补性

Editing Guidance 和 Editing Correctness 的互补关系揭示了一个重要原则：

- **忠实性信号**（Guidance）：直接优化端到端目标，但受限于理解模型的能力天花板
- **质量信号**（Correctness）：评估中间产出的质量，可以突破天花板但引入评估噪声

这种"结果驱动 + 过程约束"的双信号设计在 RL 训练中具有普遍参考价值。

### 5.3 验证机制的非对称成本洞察

> "A correct edit gives decisive visual guidance, while an incorrect one introduces structured confounders that MLLMs struggle to override."

这一发现对整个"视觉推理辅助"领域都有指导意义：**错误的视觉辅助比没有辅助更糟糕**。MLLM 对编辑图像中的结构化错误极为敏感，因为编辑后的图像看起来很"真实"，模型难以区分哪些是原始信息、哪些是编辑引入的噪声。

### 5.4 RL 的局限性

一个诚实且重要的发现是 Stage II RL 在结构性任务上几乎没有增益。作者分析原因是 GRPO 采样的语义多样性不足以覆盖复杂的结构性编辑（如迷宫路径规划、拼图还原）。这暗示了**当前 RL 方法在处理需要长程依赖和高精度的视觉编辑任务时的局限性**。

---

## 六、个人评述

### 6.1 优势

- **架构优雅**：解耦设计简洁明了，"编辑→验证→推理"的三步流程直觉清晰
- **实用价值高**：即插即用，无需修改理解模型，可以直接与 GPT-4V、Gemini 等闭源模型配合使用
- **实验充分**：覆盖 5 个差异化的任务族、3 个不同规模的理解模型、多种对比基线
- **开源**：代码和模型在 GitHub 上开源（InternLM/ETCHR），便于社区复现和扩展

### 6.2 局限性

- **编辑质量的天花板**：Diffusion 模型的编辑正确率在深度推理任务上仍然较低（如 Frozen Lake 仅 9.5→13.5），说明"生成侧 Gap"并未完全解决
- **RL 阶段收益有限**：Stage II 在结构性任务上几乎无增益，说明当前的 RL 方法（Pref-GRPO）对复杂视觉编辑的优化能力不足
- **依赖理解模型作为 Judge**：Judge 模型的能力限制了奖励信号的质量，形成了一定的循环依赖
- **计算成本**：编辑器推理（FLUX.2 的 30 步 Diffusion）+ 验证 + 推理，整体延迟较高，可能不适合实时应用
- **任务族的 meta-prompt 需要手动设计**：虽然称为"软路由器"，但 task-level prompt 仍需人工定义，扩展到新任务类型需要额外工程

### 6.3 总体评价

ETCHR 提出了一个**概念清晰、工程实用**的视觉推理增强框架。"解耦编辑"这一设计选择是论文最大的亮点——它避开了统一模型"既要又要"的困境，让编辑器和理解模型各司其职。双奖励设计和验证机制也体现了对问题本质的深刻理解。

不过，实验结果也暴露了一些根本性挑战：编辑质量随推理深度急剧下降的问题并未被完全解决，RL 阶段的有限增益暗示需要更强大的优化方法。这是一篇**好的开局之作**，为"视觉推理辅助"这一方向奠定了基础，但距离真正解决复杂视觉推理问题还有很长的路。

---

## 七、参考文献

1. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*.
2. Wu, J., et al. (2025). DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning. *arXiv:2504.11830*.
3. Shao, L., et al. (2024). V*: Guided Visual Search as a Core Capability in Multimodal LLMs. *CVPR 2024*.
4. Labs, B. (2025). FLUX.2: Next-Generation Text-to-Image Generation. Black Forest Labs.
5. Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv:2402.03300*.
6. Team, InternLM (2025). InternLM-XComposer: A Vision-Language Large Model for Advanced Text-image Comprehension and Composition. *arXiv:2309.15112*.
7. Achiam, J., et al. (2023). GPT-4 Technical Report. *arXiv:2303.08774*.

