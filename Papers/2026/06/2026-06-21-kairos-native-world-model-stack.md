# Kairos: A Native World Model Stack for Physical AI

## 论文元数据
- 标题：Kairos: A Native World Model Stack for Physical AI
- 作者：Kairos Team, Fei Wang, Shan You, Qiming Zhang, Tao Huang, Zuoyi Fu, Zhisheng Zheng, Yunlong Xi, Feng Lv, Xiaoming Wu, Zeyu Liu, Cong Wan, Pu Li, Ruiqing Yang, Xiaoou Li, Wei Wang, Kangkang Zhu, Yuwei Zhang, Shi Fu, Zheng Zhang, Xiaoning Wu, Xuzeng Fan, Dacheng Tao, Xiaogang Wang
- arXiv ID：2606.16533
- 提交日期：2026-06-15
- 官方代码：https://github.com/kairos-agi/kairos-sensenova (Apache 2.0)
- HuggingFace：https://huggingface.co/kairos-agi
- 模型参数：4B

---

## Ch1: 论文概述与核心贡献

Kairos是由商汤科技Sensenova团队于2026年6月发布的原生世界模型栈，专为Physical AI（物理人工智能）设计。该工作提出了从预训练阶段就原生融合物理规律、行为语义和具身基础（embodied grounding）的范式，而非将通用视频生成器通过解耦微调（decoupled fine-tuning）适配到机器人控制任务。

### 四大核心挑战

Kairos针对Physical AI世界建模面临的四个根本性问题：

**1. 碎片化世界学习（Fragmented World Learning）**

现有方法在不同数据源（网络视频、人类演示、机器人轨迹）之间采用割裂的训练策略。网络视频提供物理先验（物体恒常性、重力、因果律），但缺乏行为语义；人类演示包含丰富操作意图，但不含机器人具身信息；机器人轨迹数据规模最小，且与视觉域存在分布偏移。解耦微调策略难以统一这些异质模态。

**2. 长程状态维护（Long-Horizon State Maintenance）**

主流Transformer架构依赖滑动窗口注意力（sliding-window attention），其有效视野受限。对于需要数千步推理的具身任务（如"从厨房取物品放入抽屉"），窗口内无法捕获完整因果链，导致误差累积失控。Kairos证明纯窗口架构会产生不可约的过剩风险（irreducible excess risk，Theorem 1）。

**3. 理解-控制鸿沟（Understanding-to-Control Gap）**

视频生成模型（如Sora、CogVideoX）优化视觉保真度和短期连贯性，但预测动作（action prediction）需要与可执行控制信号对齐。通用生成器的潜在空间不具动作语义，直接微调至动作预测面临严重模态错配。

**4. 部署约束（Deployment Constraints）**

实时机器人控制需要闭环推理（closed-loop inference）在边缘设备上运行。大模型推理延迟和内存占用与实时性要求矛盾，而现有世界模型未针对部署进行系统级协同设计。

### 三大技术支柱

**支柱1：跨具身数据课程（CEDC, Cross-Embodiment Data Curriculum）**

CEDC将异质数据组织为渐进式金字塔：

- Stage 1（被动观察）：网络规模视频预训练，学习物理先验
- Stage 2（人类模仿）：人类行为数据对齐，注入操作意图
- Stage 3（机器人具身）：机器人轨迹微调，适配具体embodiment

这种curriculum确保模型在接触稀疏的机器人数据前，已从大规模开放世界数据中学习到可迁移的world knowledge。

**支柱2：原生理解-生成-预测架构（Native Understanding-Generation-Prediction Architecture）**

Kairos采用统一的Mixture-of-Transformers（MoT）栈，三个子模块共享 backbone：

- **World Understanding**：基于VLM（Qwen系列）提取语义表征
- **World Generation**：Diffusion Transformer（DiT）支持T2V、I2V、TI2V
- **World Prediction**：Joint World-Action Model（WAM）联合预测视频帧和动作序列

核心创新是**混合线性时序注意力**（Hybrid Linear Temporal Attention）：

- **SWA（Sliding-Window Attention）**：局部动力学，窗口内捕捉帧间精细运动
- **DSWA（Dilated Sliding-Window Attention）**：中程依赖，通过膨胀采样跨越多帧
- **GLA（Gated Linear Attention）**：全局因果记忆，基于GatedDeltaNet的delta更新规则

形式化地，GLA维护全局状态 $h_t$，更新规则为：
$$h_t = \alpha_t \odot h_{t-1} + \beta_t \odot (k_t \odot v_t^T)$$

其中遗忘门 $\alpha_t \in [0,1]^d$ 控制历史记忆衰减，写入强度 $\beta_t \in [0,1]^d$ 调节新信息权重。门控机制使模型可选择性保留长期因果（如"杯子在桌上"）而丢弃无关细节（如背景纹理）。

理论保证（Theorem 2）证明混合记忆架构的长程过剩风险有界：
$$R_T(\hat{f}_h) - R_T(f^*) \leq (L\epsilon + L_G \cdot \frac{\xi}{1-\rho})^2$$

其中 $\rho \in [0,1)$ 为遗忘门的谱界，确保误差不随时间 $T$ 线性爆炸。

**支柱3：部署感知系统协同设计（Deployment-Aware Co-Design）**

Kairos从训练阶段就优化推理效率，实现：
- 480p分辨率实时生成：单卡Nvidia A800 11.7秒，4卡并行3.0秒
- 边缘部署：通过蒸馏得到轻量版本，支持Agibot G1、Unitree G1、Songling PIPER等机器人平台

### 关键成果

Kairos在多个基准上达到SOTA：

| 基准 | 指标 | Kairos 4B |
|------|------|-----------|
| PAI-Bench-robot | Success Rate | 80.03 |
| WorldModelBench-robot | TI2V | 9.08 |
| DreamGen Bench | PA | 0.529 |
| DreamGen Bench | IF | 0.609 |
| VideoPHY | - | ~45.x |
| WorldModelBench | - | 8.94 |
| PAI-Bench | - | 80.84 |

在机器人专用基准（PAI-Bench-robot、WorldModelBench-robot）上，Kairos 4B超过更大参数的竞品：Cosmos 2.5-2B、Wan 2.2-5B、Cosmos 2.5-14B。在通用世界建模任务（PAI-Bench、WorldModelBench）上亦保持竞争力。

### 与现有工作的核心差异

Kairos明确反对"通用视频生成器 + 解耦微调"的技术路线。Cosmos、Wan、Lingbot等方法虽然提升了视觉生成质量，但其架构设计未考虑：
1. 长程因果推理的必要性（GLA的persistent state）
2. 动作预测与视觉生成的统一（WAM的joint modeling）
3. 边缘部署的硬约束（从预训练开始的效率优化）

Kairos的主张是：Physical AI的世界模型必须是**原生**（native）的——从数据curriculum、架构设计到系统优化，每个环节都为具身智能任务服务。

---

## Ch2: 研究背景与动机

### World Models的发展历程

世界模型（World Models）概念源于认知科学和强化学习，指agent内部构建的环境动态预测器。早期工作（如Ha & Schmidhuber 2018）在低维观测空间（车杆、Atari游戏）训练RNN-based world model，为planning提供可微模拟器。随着视觉预训练的兴起，world model扩展到高维视觉空间：

1. **Passive Visual Generators（2019-2023）**：以视频生成任务为核心，代表性工作包括VideoGPT、CogVideo、Sora。这些模型优化分布匹配目标 $p(v_{1:T})$，生成逼真短视频，但缺乏对underlying dynamics的显式建模。

2. **Behavior Cloning扩展（2020-2024）**：将视频生成器微调至机器人轨迹数据。典型做法是冻结视觉编码器，训练轻量action head（如RT-1、RT-2）。这种decoupled fine-tuning策略在小数据集上有效，但难以泛化到长程任务和新场景。

3. **Foundational Infrastructure转型（2024-）**：世界模型从"生成工具"转向"Physical AI的基础设施"。Cosmos（NVIDIA）、Wan（阿里）、Lingbot等尝试构建通用世界模型，但架构仍受视频生成范式的束缚。

### 现有方法的局限

**Cosmos（NVIDIA）**

采用diffusion-based video generation，支持text-to-video和image-to-video。其attention机制主要优化视觉连贯性，未考虑长程因果推理。在机器人基准上的表现受限于short-horizon continuation能力。

**Wan（阿里）**

大规模视频预训练（多B参数），在视频生成质量上达到SOTA。但Wan的架构（基于DiT + temporal attention）在物理规律理解和动作预测上缺乏专门设计，微调至机器人任务时需要大量adaptation数据。

**Lingbot**

尝试将语言模型与视频生成器结合，用于具身推理。但linguistic grounding与visual grounding的融合仍显粗糙，且未解决实时部署问题。

这些方法的共同问题是：**将world model视为通用视频生成器的延伸**，而非为Physical AI从头设计的原生系统。

### 四大瓶颈详细阐述

**瓶颈1：数据异质性（Data Heterogeneity）**

Physical AI需要三类数据：
- **Web Video**：规模最大（数亿小时），但缺乏动作标注和embodiment信息
- **Human Demo**：包含操作意图（如"抓取杯子的手法"），但视角和动作空间与机器人不同
- **Robot Trajectory**：直接可用的控制信号，但收集成本高，且数据稀缺

现有方法通常在三者中选择单一主导（如Cosmos主要用web video，RT-2主要用robot demo），或简单拼接训练。CEDC的创新在于明确三者之间的渐进依赖关系：web video提供物理先验 → human demo注入行为语义 → robot trajectory适配具体硬件。

**瓶颈2：长程状态维护（Long-Horizon State Maintenance）**

具身任务的时间跨度从数百到数千帧。例如复杂家务任务可能需要：
- 步骤1：识别目标物体（前10帧）
- 步骤2：规划路径（前50帧）
- 步骤3：执行抓取（前100帧）
- 步骤4：移动到目标位置（前200帧）
- 步骤5：放置物体（前250帧）

纯滑动窗口架构（如标准的Transformer）在步骤5执行时，已忘记步骤1的object identity。Kairos的GLA通过全局记忆状态 $h_t$ 跨越整个任务维护关键信息（如"手中持有杯子"），而SWA/DSWA处理局部精细运动。

**瓶颈3：理解-控制鸿沟（Understanding-to-Control Gap）**

视频生成模型的latent space编码视觉模式（纹理、光照、运动模糊），但这些特征与action space（关节角度、末端位姿、力矩）不直接对应。例如：
- 视觉特征："杯子正在移动"
- 动作特征："关节1角度+15°，关节2角度-8°"

直接将视觉特征映射到动作需要大量对齐数据。Kairos的WAM模块在预训练阶段就联合优化video prediction和action prediction，使latent space同时编码视觉动态和动作语义，减小微调阶段的模态gap。

**瓶颈4：边缘部署约束（Deployment Constraints）**

实时机器人控制要求：
- **低延迟**：闭环推理周期通常<100ms
- **低内存**：边缘设备GPU内存有限（如Jetson AGX 32GB）
- **鲁棒性**：长时间运行不能显存溢出

通用视频生成器未针对这些约束优化。例如Sora的推理需要大规模GPU集群，无法单卡实时运行。Kairos从训练设计就考虑部署：
1. GLA的线性复杂度（相比quadratic attention）
2. 模型蒸馏得到小版本（4B→更小）
3. 推理优化（4卡并行3.0秒生成480p）

### Kairos的设计理念

Kairos的核心insight是：**Physical AI的世界模型不是通用视频模型的下游应用，而是需要从预训练开始就原生设计的新范式**。

这一理念体现在三个层面：

**数据层面的原生性**：CEDC不是简单的data mixing，而是明确不同数据源在训练curriculum中的角色和顺序。web video不是"更多数据"，而是物理规律的来源；human demo不是"标注数据"，而是行为语义的桥梁；robot trajectory不是"fine-tuning set"，而是embodiment grounding的最终对齐。

**架构层面的原生性**：MoT + Hybrid Attention不是"更复杂的Transformer"，而是为长程因果推理设计的专门架构。GLA的persistent state不是"额外记忆"，而是解决window-based架构不可约风险的理论必要（Theorem 1）。

**系统层面的原生性**：部署优化不是"后处理步骤"，而是与模型训练协同的joint design。蒸馏不是"压缩技巧"，而是让edge-deployable version保留预训练学到的world knowledge。

这种原生范式使Kairos在4B参数规模下就能达到或超越更大模型的性能，同时满足实时部署要求。

---
## Ch3: 核心技术创新

Kairos的核心技术创新围绕三大支柱展开：Cross-Embodiment Data Curriculum (CEDC)、统一理解-生成-预测架构、以及Hybrid Linear Temporal Attention。这三者共同构成原生世界模型栈的技术基础，直接应对Physical AI面临的四大挑战：异构数据源的学习碎片化、超越短视频延续的长视界状态维护、理解到具身控制的鸿沟、以及真实世界约束下的部署需求。

### 3.1 Cross-Embodiment Data Curriculum (CEDC)

### 3.1.1 三阶段渐进式数据金字塔

CEDC的核心思想是构建一个从被动观察到人类模仿再到机器人具身的渐进式数据层次，打破不同具身形态间的数据异质性壁垒，提高数据复用效率。

**Stage 1: Web-scale视频预训练**

第一阶段使用开放世界视频数据进行大规模预训练，目标是从无标注的视觉观测中学习物理先验。这一阶段的数据来源包括互联网视频、纪录片、监控录像等，覆盖丰富的物理交互场景：物体运动、液体流动、刚体碰撞、工具使用等。模型通过被动观察学习重力、摩擦力、惯性等基础物理规律，以及物体持久性、因果传递等认知先验。

Stage 1的训练目标主要是视频预测和生成，迫使模型建立对物理世界动态性的内在表征。例如，给定一个杯子从桌沿滑出的视频片段，模型需要预测杯子下落、撞击地面、可能破碎的完整物理过程。这种训练方式使模型在无任何机器人交互数据的情况下，初步掌握物理世界的运动规律和因果关系。

**Stage 2: 人类行为数据**

第二阶段引入结构化的人类行为数据，实现语义对齐。数据来源包括：Ego4D等第一人称视角数据集、EPIC-KITCHINS等日常交互数据、人类操作演示视频等。与Stage 1的被动观测不同，这一阶段的数据包含明确的人类意图和目标导向行为：抓取、放置、工具使用、多步骤任务执行等。

Stage 2的关键作用是将Stage 1学到的物理先验与任务语义关联起来。例如，"倒水"这一行为不仅仅是液体的流动，还包含目标（将水从容器A转移到容器B）、约束（避免溢出）、终止条件（容器A为空或容器B满）等高层语义信息。通过人类行为数据，模型学习到哪些物理模式是实现特定任务的有效手段，从而建立起物理动态与任务目标之间的映射。

**Stage 3: 机器人轨迹精调**

第三阶段使用真实机器人轨迹数据进行具身精调。数据来源包括：机器人执行任务的运动轨迹、传感器读数、成功/失败标签等。这一阶段的数据量通常远小于前两个阶段，但具有高度的专业性和任务针对性。

Stage 3的目标是将从人类观察中迁移的知识适配到机器人的具身约束上。例如，人类的手部自由度和抓握方式与机器人末端执行器存在显著差异，机器人的运动学/动力学约束也与人体不同。通过在真实机器人数据上的精调，模型学习到如何根据具身特性调整动作预测：适应机器人的关节限位、速度限制、力矩约束等。

### 3.1.2 CEDC的核心思想：打破数据异质性壁垒

CEDC的核心创新在于认识到不同具身形态间的数据共享存在一个根本性障碍：数据异质性壁垒。传统方法要么将所有数据混合训练（导致语义混淆），要么分别训练不同模型（浪费数据资源）。CEDC通过渐进式课程学习，在不破坏语义一致性的前提下最大化数据复用。

从技术角度看，CEDC实现了一个从"模仿"向"物理级理解"的转变。在Stage 1，模型主要模仿观测到的视觉模式；在Stage 2，模型模仿人类的行为模式，但开始理解行为背后的物理约束；在Stage 3，模型超越简单模仿，建立对物理世界因果结构的深层理解，能够根据具身特性和任务需求生成新颖的行为序列。

这种渐进式训练策略的另一个优势是计算效率。Stage 1和Stage 2的数据可以离线大规模预处理，Stage 3只需要在相对较小的机器人数据集上进行精调，大幅降低了训练成本。同时，由于前两个阶段已经建立了丰富的物理和语义先验，Stage 3的收敛速度显著快于从零开始训练。

### 3.1.3 数据复用效率的提升

CEDC通过三阶段课程实现的数据复用效率提升体现在两个层面：跨具身复用和跨任务复用。

跨具身复用：在Stage 1和Stage 2学到的物理先验和行为模式可以迁移到不同的机器人平台上，只需通过Stage 3进行平台适配。例如，同一个预训练模型可以通过精调适配到机械臂、移动机器人、人形机器人等不同具身形态，而不需要为每个平台从零收集训练数据。

跨任务复用：CEDC训练的模型展现出强大的零样本泛化能力。在训练时未见过的任务上，模型可以通过组合已知的基本物理模式和行为模式生成合理的执行策略。例如，一个见过"倒水"和"搅拌"的模型，在面对"制作混合饮料"这一新任务时，可以尝试组合这两个基本技能来达成目标。

### 3.2 统一理解-生成-预测架构

### 3.2.1 三大组件

Kairos的原生架构将世界理解、世界生成、世界预测统一到一个单一框架中，避免模块化系统中的信息损失和语义鸿沟。

**World Understanding：高层语义表征提取**

World Understanding模块基于视觉-语言模型（VLM），论文中采用Qwen系列作为骨干网络。该模块接收多模态输入（图像/视频+文本指令），输出高维语义表征。与纯视觉backbone不同，VLM的优势在于将视觉观测与语言描述对齐，使模型能够理解任务指令和场景语义。

例如，给定指令"将红色杯子移到桌子上"和当前场景图像，World Understanding模块不仅提取视觉特征（杯子的位置、颜色、形状），还理解指令的语义结构（动作：移动；对象：红色杯子；目标位置：桌子）。这种语义表征对于后续的动作生成至关重要，因为它提供了任务导向的约束条件。

World Understanding模块的另一个重要作用是跨模态信息融合。在TI2V（Text-to-Image-to-Video）任务中，模型需要同时处理文本指令、初始图像、以及可能的中间观测。VLM通过联合嵌入空间将这些不同模态的信息映射到统一的表征，为后续生成过程提供丰富的上下文。

**World Generation：多模态视频生成**

World Generation模块基于Diffusion Transformer (DiT)架构，支持T2V（Text-to-Video）、I2V（Image-to-Video）、TI2V三种生成模式。DiT的核心思想是将扩散模型的去噪过程与Transformer的强大建模能力结合，通过迭代去噪逐步生成高质量视频。

DiT的优势在于两个方面：生成质量和采样效率。与传统基于GAN的视频生成相比，DiT生成的视频具有更高的多样性和一致性；与基于U-Net的扩散模型相比，DiT利用Transformer的自注意力机制更好地捕获长程依赖。对于world model而言，这意味着生成的未来视频序列不仅在视觉上逼真，而且在物理上合理——物体运动遵循物理规律，交互结果符合因果逻辑。

World Generation模块的关键设计是时空注意力机制。模型同时建模空间依赖（同一帧内的像素关系）和时间依赖（跨帧的物体运动和状态变化）。在Kairos中，这种时空注意力通过Hybrid Linear Temporal Attention实现，具体在下节详述。

**World Prediction：联合世界-动作建模**

World Prediction模块是Kairos的核心创新，它同时预测未来视觉状态和未来动作序列，形成Joint World-Action Model (WAM)。传统world model通常只预测未来视觉观测，而将动作策略交给单独的规划模块；或者只预测动作，而不显式建模动作对视觉状态的影响。WAM的联合建模使模型能够理解动作与观测之间的双向因果关系：动作影响环境状态，环境状态反过来约束可行的动作。

WAM的另一个优势是端到端训练。传统分层系统通常需要分别训练世界模型和策略网络，然后用一个辅助损失函数将二者耦合。这种分阶段训练容易导致次优解：世界模型的预测误差会累积到策略中，策略的次优动作又会误导世界模型的训练。WAM通过联合训练，使世界预测和动作生成相互促进：更好的世界预测为动作生成提供更准确的未来状态估计，更优的动作生成则为世界预测提供更合理的演化轨迹。

### 3.2.2 Mixture-of-Transformers (MoT)

MoT是WAM的具体实现形式，它由两个并行的DiT分支组成：Video DiT和Action DiT。

**Video DiT：建模未来视觉token**

Video DiT负责预测未来视频帧，其输入包括当前视觉观测、历史动作序列、以及可能的文本指令。输出是未来T帧的token表示，每个token对应一个空间位置的patch（例如16×16像素）。

Video DiT的架构基于标准DiT，但针对时序建模进行了优化。具体而言，模型使用Hybrid Linear Temporal Attention（下节详述）代替标准self-attention，将复杂度从O(n²)降至O(n)，使其能够处理长视频序列。此外，Video DiT采用了因果注意力masking，确保时刻t的预测只依赖于t时刻之前的信息，避免信息泄漏。

**Action DiT：预测未来动作token**

Action DiT负责预测未来动作序列，其规模约为Video DiT的1/5。这是因为动作空间的维度通常远小于视觉空间：机器人动作可能只包含关节角度、末端执行器位姿等数十个维度，而视觉空间包含数千个像素特征。

Action DiT的输入与Video DiT类似（当前观测、历史动作、文本指令），但输出是动作空间的离散化表示。论文中未明确说明动作的具体表示方式，可能的方式包括：关节位置的量化编码、末端执行器轨迹的waypoint、或者高层skill的原语索引。

**Joint attention masking防止信息泄漏**

MoT的关键设计是Video DiT和Action DiT之间的交互方式。两个分支共享部分层（通过cross-attention相互通信），但使用严格的注意力masking避免信息泄漏。具体而言：

1. Video DiT可以attend to历史动作token，但不attend to未来动作 token（如果使用teacher forcing训练）
2. Action DiT可以attend to历史观测token和当前观测token，但不attend to未来观测token
3. Video DiT和Action DiT在特定层进行cross-attention，交换中间表征

这种masking策略确保训练时的因果关系正确：未来动作不能影响未来观测（否则模型会偷看答案），但动作和观测可以通过共享表征相互约束。例如，Video DiT生成的未来状态应该与Action DiT预测的动作物理一致，Action DiT生成的动作应该在Video DiT预测的物理环境中可执行。

### 3.2.3 架构优势

统一理解-生成-预测架构的优势体现在三个方面：

1. **信息无损传递**：传统模块化系统在每个模块接口处都会存在信息损失（例如，世界模型的输出被压缩成低维状态，然后传递给策略模块）。Kairos的统一架构通过共享表征和跨模块注意力，保留完整的信息流。

2. **端到端优化**：所有组件联合训练，全局损失函数直接优化最终任务性能，避免局部最优和训练不稳定。

3. **多任务共享**：同一架构可以支持视频生成、动作预测、TI2V等多种任务，无需针对每个任务设计专用模块。任务间的共享知识提升了样本效率和泛化能力。

### 3.3 Hybrid Linear Temporal Attention (LinearDiT)

### 3.3.1 动机：从O(n²)到O(n)

标准Transformer的self-attention机制具有O(n²)的时间和空间复杂度，其中n是序列长度。对于视频数据而言，这一限制尤其严重：一个480p分辨率、1秒30帧的视频包含约24000个token（ assuming 16×16 patches），self-attention的复杂度达到5.76×10^8次运算，远超实时处理的能力。

Hybrid Linear Temporal Attention (LinearDiT)的核心思想是：不同时间尺度的依赖关系需要不同的注意力机制。短程依赖（相邻帧之间）可以用局部建模，中程依赖（数秒内的事件链）可以用稀疏建模，长程依赖（整个视频的因果线索）需要全局建模但不能负担全连接的复杂度。

LinearDiT混合三种注意力机制：Sliding-Window Attention (SWA)、Dilated Sliding-Window Attention (DSWA)、Gated Linear Attention (GLA)，三者分别处理短程、中程、长程依赖，总复杂度保持O(n)。

### 3.3.2 Sliding-Window Attention (SWA)：局部时序建模

SWA的核心思想是限制每个token的注意力范围到一个固定大小的时序窗口。对于时刻t的token，其attention query只计算与时刻[t-w, t+w]范围内的key的相似度，其中w是窗口半径。

SWA的复杂度为O(n·w)，当w是常数时复杂度为O(n)。SWA的有效性基于一个合理假设：相邻帧之间的视觉变化通常是连续且平滑的，物体运动、相机运动、光照变化都在局部范围内可预测。例如，预测第t+1帧主要需要知道第t-w到第t帧发生了什么，而不需要直接attend to第0帧的细节。

SWA的一个问题是无法捕获超出窗口范围的依赖关系。例如，一个"杯子从桌子掉落"的事件可能跨越10秒（300帧），如果窗口大小只有10帧，模型在杯子下落过程中无法直接看到"杯子在桌面上"的初始状态。DSWA和GLA正是为了解决这个问题。

### 3.3.3 Dilated Sliding-Window Attention (DSWA)：中程依赖

DSWA通过引入dilation factor扩展SWA的感知范围，同时保持线性复杂度。具体而言，时刻t的token不仅attend to [t-w, t+w]范围内的key，还attend to [t-d-w, t-d+w]和[t+d-w, t+d+w]，其中d是dilation factor。

例如，设w=4（窗口半径4帧），d=8（dilation factor 8帧）。时刻t=20的token会attend to时刻[16,24]、[8,16]、[24,32]三个窗口。这样，模型可以在不增加复杂度的情况下"看到"更远的历史。如果需要更远的范围，可以堆叠多个DSWA层，每层使用不同的dilation factor。

DSWA的有效性基于中程依赖的稀疏性假设：虽然在物理世界中存在长程因果链，但因果链通常由一系列局部事件组成，每个局部事件内部紧密耦合，事件之间相对独立。例如，"制作咖啡"这一任务可以分解为"拿杯子"、"倒咖啡"、"加糖"、"搅拌"等子事件，每个子事件内部帧间紧密相关，但不同子事件之间只需要较粗粒度的信息传递（"杯子已经拿到"、"咖啡已经倒入"）。

### 3.3.4 Gated Linear Attention (GLA)：全局持久记忆

GLA通过GatedDeltaNet实现全局记忆，核心机制是delta update rule + forget gate + writing strength。

**Delta update rule**

传统RNN（如LSTM）的隐状态更新可以表示为：h_t = f(h_{t-1}, x_t)，其中f是复杂的非线性函数。GatedDeltaNet简化这一更新为delta rule：

$$ h_t = h_{t-1} + \Delta h_t $$

其中 $\Delta h_t$ 是当前时刻对记忆的"增量更新"。这一设计的优势是计算效率：更新线性层可以通过增量计算实现，避免每次都从头计算完整状态。

**Forget gate $\alpha_t$**

Forget gate控制历史记忆的衰减速度，取值范围[0,1]。$\alpha_t$ 接近1时保留较多历史信息，$\alpha_t$ 接近0时快速遗忘历史。在实现中，$\alpha_t$ 是一个可学习的门控机制，根据当前输入动态调整：

$$ \alpha_t = \sigma(W_\alpha x_t + b_\alpha) $$

其中 $\sigma$ 是sigmoid函数，将输出映射到[0,1]。

Forget gate的作用是使模型能够自适应地决定记忆长度。对于关键事件（如"物体被抓取"），模型可以设置较高的 $\alpha_t$，将这一信息持久保存；对于无关细节（如背景中的微小变化），模型可以设置较低的 $\alpha_t$，快速遗忘以节省记忆容量。

**Writing strength $\beta_t$**

Writing strength控制当前信息写入记忆的强度，取值范围[0,1]。$\beta_t$ 的计算方式与forget gate类似：

$$ \beta_t = \sigma(W_\beta x_t + b_\beta) $$

Writing strength与forget gate共同作用，决定记忆更新的净效果：

$$ h_t = \alpha_t \odot h_{t-1} + \beta_t \odot \Delta h_t $$

其中 $\odot$ 是逐元素乘法。当 $\beta_t$ 高时，当前信息被强力写入记忆；当 $\beta_t$ 低时，记忆主要保持历史状态。

**GatedDeltaNet的完整更新**

GatedDeltaNet的完整更新规则包含三个步骤：

1. **Delta计算**：根据当前输入计算增量更新
   $$ \Delta h_t = \phi(W_h x_t + b_h) $$
   其中 $\phi$ 是激活函数（如SiLU）。

2. **Gate计算**：计算forget gate和writing strength
   $$ \alpha_t = \sigma(W_\alpha x_t + b_\alpha) $$
   $$ \beta_t = \sigma(W_\beta x_t + b_\beta) $$

3. **状态更新**：应用gated delta rule
   $$ h_t = \alpha_t \odot h_{t-1} + \beta_t \odot \Delta h_t $$

这一更新规则的复杂度是O(n)，每个时刻的计算只依赖于当前输入和上一时刻状态，无需与所有历史状态计算attention。

**GLA的全局记忆能力**

虽然GatedDeltaNet的更新是局部的（只依赖上一状态），但其记忆能力是全局的。关键在于 $\alpha_t$ 的设计：当 $\alpha_t$ 接近1时，历史信息可以在不衰减的情况下传递到任意远的未来。例如，如果 $\alpha_t = 0.99$，经过100个时刻后记忆仍保留约 $0.99^{100} \approx 0.37$ 的原始信息。

GLA的这一特性使其能够捕获长程因果依赖。例如，在一个"跨越30秒的物体抓取任务"中，模型需要在第0帧记住"目标物体位置"，在第300帧执行"抓取"动作。传统SWA/DSWA由于窗口限制可能无法直接看到这一信息，而GLA通过持久记忆可以将关键信息保存到需要的时候。

### 3.3.5 工业首创：Hybrid Linear Attention算子

LinearDiT是world model领域第一个Hybrid Linear Attention算子，首次实现开源world model的real-time on-robot inference。

**实时性能**

论文报告的实时性能数据：480p分辨率下，单块Nvidia A800需要11.7秒生成1秒视频，4块A800通过并行计算将时间降至3.0秒。这一性能接近实时要求（考虑到机器人控制通常需要10-20Hz的频率，即0.05-0.1秒的推理时间），表明LinearDiT在保持生成质量的同时实现了工业级的推理速度。

实时性的关键在于GLA的线性复杂度和SWA/DSWA的稀疏建模。传统world model（如基于标准Transformer的实现）无法在单GPU上处理长视频序列，而LinearDiT通过混合三种注意力机制，将复杂度从O(n²)降至O(n)，使实时部署成为可能。

**部署优势**

LinearDiT的另一个优势是内存占用。O(n²)的self-attention需要存储n×n的注意力矩阵，对于长视频序列（例如n=10000），这需要约400MB内存（假设每个注意力值4字节）。LinearDiT只需存储O(n)的中间状态，内存占用显著降低。内存占用的降低意味着可以在边缘设备（如机器人机载计算机）上部署更大的模型或处理更长视频序列。

论文中提到通过蒸馏（distillation）可以实现边缘部署。蒸馏的目标是将LinearDiT的知识迁移到更小的模型，同时保持混合注意力的线性复杂度优势。这使得Kairos可以在资源受限的机器人平台上运行，无需依赖云端GPU集群。

### 3.3.6 三种注意力的协同工作

SWA、DSWA、GLA三者在LinearDiT中协同工作，分别处理不同时间尺度的依赖关系：

1. **SWA处理短程依赖**：相邻帧之间的像素级对应关系、物体运动的连续性、光照的平滑变化。这些依赖在局部范围内紧密耦合，需要精细建模。

2. **DSWA处理中程依赖**：数秒内的事件链、多步骤任务的中期状态、物体交互的阶段性结果。这些依赖跨越多个局部窗口，但可以通过稀疏连接捕获。

3. **GLA处理长程依赖**：整个任务的因果结构、关键事件的持久影响、全局约束条件（如"不可让物体掉落"）。这些依赖需要在整个视频长度上保持一致，需要全局记忆。

三者的结合使模型能够在不牺牲实时性的前提下，捕获从毫秒级（相邻帧）到分钟级（整个任务）的时序依赖。这是Kairos在长视界预测任务上取得优异表现的关键技术基础。

### 3.4 理论分析

### 3.4.1 Theorem 1：持久状态的必要性

**定理陈述**

任何window-restricted predictor都存在不可约的excess risk：
$$ R_w^* - R_{full}^* = \mathbb{E}[\text{Var}(m_t | W_t^{(w)})] > 0 $$

其中：
- $R_w^*$ 是窗口大小为w的预测器的最优风险
- $R_{full}^*$ 是全序列预测器的最优风险
- $m_t$ 是时刻t的真实状态
- $W_t^{(w)}$ 是大小为w的时序窗口
- $\text{Var}(m_t | W_t^{(w)})$ 是给定窗口下状态的条件方差

**定理含义**

定理1表明：如果预测器只能访问有限窗口的历史信息，那么即使采用最优预测策略，仍然存在不可避免的预测误差。这个误差的来源是状态的条件方差：给定窗口信息外，仍有部分状态信息无法被确定，导致预测不确定性。

从信息论角度看，这一定理反映了信息缺失的代价。如果真实状态 $m_t$ 依赖于窗口外的历史信息（例如t-w之前的某个关键事件），那么仅根据 $W_t^{(w)}$ 无法完美预测 $m_t$。这种信息缺失造成的误差是系统性的，无法通过增加模型容量或优化训练过程消除。

**物理直观**

在物理世界建模中，定理1对应一个基本事实：世界状态包含隐变量，这些隐变量的影响可能持续很长时间。例如：

- **物体持久性**：一个被移到遮挡物后的杯子，虽然当前帧中不可见，但仍然存在。仅根据最近几帧无法判断杯子是否还在。
- **因果链**：一个复杂的物理现象（如多米诺骨牌倒下）的初始条件可能在数十秒之前，仅观察最近几帧无法理解当前状态。
- **任务记忆**：一个机器人任务的中间状态（如"已经收集了3个物品，还需要2个"）需要记忆整个任务的进展，窗口限制会导致任务失败。

定理1的数学形式将这一直观形式化：excess risk恰好等于条件方差，即"无法从窗口信息确定的状态变化量"。

**对window-restricted模型的启示**

定理1意味着任何仅使用SWA或DSWA的world model都无法达到最优预测性能。即使窗口大小设置得很大，仍然存在"窗口盲区"：关键事件发生在窗口外，导致预测失效。从优化角度看，增大窗口可以减小excess risk，但无法将其降为零；同时，增大窗口会显著增加计算复杂度，抵消线性注意力的优势。

### 3.4.2 Corollary 1：显式下界

**推论陈述**

Excess risk满足显式下界：
$$ R_w^* - R_{full}^* \geq P(E) \cdot \alpha \cdot (1-\alpha) \cdot (\mu_1 - \mu_2)^2 $$

其中：
- $P(E)$ 是关键事件发生在窗口外的概率
- $\alpha$ 是事件对当前状态的影响权重
- $\mu_1, \mu_2$ 是事件发生前后的状态均值
- $(\mu_1 - \mu_2)^2$ 是状态差异的平方

**推论含义**

推论1给出了excess risk的具体下界，使其可以量化。下界的四个因子分别对应不同的因素：

1. **$P(E)$：事件发生率**。关键事件（如物体被抓取、环境突变）发生在窗口外的概率越高，excess risk越大。如果关键事件总是在窗口内（例如短预测视界），则 $P(E) \approx 0$，下界趋近零。

2. **$\alpha$：影响权重**。事件对当前状态的影响越大（$\alpha$ 接近0或1），excess risk越大。如果事件对当前状态几乎没有影响（$\alpha \approx 0.5$，即前后状态均值相近），则excess risk较小。

3. **$(\mu_1 - \mu_2)^2$：状态差异幅度**。事件导致的状态变化越大，excess risk越大。例如，一个"杯子从桌面掉落"事件导致状态从"杯子在桌面"变为"杯子在地面"，状态差异大，excess风险高；而一个"轻微移动"事件导致的状态变化小，excess风险低。

4. **$\alpha(1-\alpha)$：方差最大化项**。这一项在$\alpha=0.5$时达到最大值0.25，在$\alpha=0$或$\alpha=1$时为0。这反映了最坏情况的不确定性：当事件前后状态等概率出现时，预测难度最大。

**应用示例**

考虑一个具体场景：机器人抓取任务中，"目标物体被抓取"这一关键事件。

- 假设窗口大小为30帧（约1秒），任务总时长为300帧（约10秒）
- 抓取事件在50%的情况下发生在窗口外：$P(E) = 0.5$
- 抓取导致状态均值从"目标位置"变为"手部位置"：$\mu_1 - \mu_2 = 1.0$（归一化距离）
- 事件影响权重 $\alpha = 0.8$（抓取后80%的概率物体在手部）

则excess risk下界为：
$$ 0.5 \times 0.8 \times 0.2 \times 1.0^2 = 0.08 $$

这意味着即使最优window-restricted predictor，其风险也比full-sequence predictor高出至少0.08。如果任务要求高精度（例如外科手术，excess risk要求<0.01），则window-restricted模型无法满足。

### 3.4.3 Theorem 2：混合记忆的充分性

**定理陈述**

如果Bayes predictor可以分解为SWA/DSWA/GLA三个组件，且global memory使用contractive gated delta update（$\rho < 1$），则long-horizon excess risk满足：
$$ R_t(\hat{\mu}_t) - R_t^* \leq (L\varepsilon + L_G \cdot \xi/(1-\rho))^2 \quad \text{as} \quad t \to \infty $$

其中：
- $R_t(\hat{\mu}_t)$ 是时刻t的预测风险
- $R_t^*$ 是Bayes最优风险
- $L$ 是局部预测器的Lipschitz常数
- $\varepsilon$ 是局部预测误差
- $L_G$ 是全局记忆的Lipschitz常数
- $\xi$ 是全局记忆的更新误差
- $\rho$ 是gated delta update的收缩系数（$\rho < 1$）

**定理含义**

定理2表明：混合记忆架构（SWA+DSWA+GLA）可以在长视界上控制误差累积。与定理1的"不可约误差"不同，定理2给出的是"可逼近界"：只要满足特定条件，混合记忆的风险可以任意接近最优风险。

界的形式是两项之和的平方：

1. **$L\varepsilon$：局部误差项**。来自SWA/DSWA的局部预测误差，通过Lipschitz常数放大。如果局部预测器足够精确（$\varepsilon$ 小），且预测函数平滑（$L$ 小），则这一项小。

2. **$L_G \cdot \xi/(1-\rho)$：全局误差项**。来自GLA的全局记忆更新误差，通过$1/(1-\rho)$因子放大。$\rho$ 是gated delta update的收缩系数，$\rho < 1$ 保证误差不无限累积。

**关键条件：contractive gated delta update**

定理成立的关键条件是global memory使用contractive更新，即 $\rho < 1$。这要求gated delta update的forget gate $\alpha_t$ 满足某种收缩性约束，使得记忆误差随时间衰减而非累积。

从物理角度看，contractivity对应真实世界的一个基本性质：状态变化是连续且 bounded的。如果记忆更新是扩张的（$\rho > 1$），微小的更新误差会指数级放大，最终导致记忆发散。而真实世界的物理约束（如能量守恒、物体不可穿模）限制了状态变化率，使得合理的记忆更新应该是收缩的。

**定理与真实系统的对应**

在Kairos中，定理2的假设条件对应具体设计：

1. **Bayes predictor分解**：Kairos的MoT架构将预测分解为Video DiT（对应SWA/DSWA的局部时序建模）和Action DiT（对应GLA的全局记忆建模），符合定理的分解假设。

2. **Contractive gated delta update**：GLA中的forget gate $\alpha_t$ 和writing strength $\beta_t$ 通过学习获得，训练过程会倾向于收敛到 $\rho < 1$ 的参数设置，因为扩张性记忆会导致训练不稳定和预测性能下降。

3. **Lipschitz平滑性**：Transformer的attention机制具有一定平滑性，且论文可能采用spectral normalization等技术限制Lipschitz常数，满足定理的平滑假设。

### 3.4.4 理论与实践的对应

定理1和定理2共同构成了Kairos的理论基础：

- **定理1（必要性）**：证明仅使用局部注意力（SWA/DSWA）的模型存在不可约误差，从理论上支持引入全局记忆（GLA）的必要性。

- **定理2（充分性）**：证明混合记忆架构可以在长视界上控制误差累积，从理论上支持LinearDiT的合理性和有效性。

从工程角度看，这两个定理为架构设计提供了清晰的指导原则：

1. **必须包含全局记忆**：否则无法消除excess risk的系统性偏差（定理1）。

2. **全局记忆必须contractive**：否则误差会无限累积，长视界预测失效（定理2的 $\rho < 1$ 条件）。

3. **局部与全局记忆必须协同**：定理2的界依赖于局部误差 $\varepsilon$ 和全局误差 $\xi$ 的联合控制，表明SWA/DSWA和GLA需要共同优化，而非独立设计。

Kairos的LinearDiT正是遵循这些原则设计的：SWA/DSWA捕获局部/中程依赖，GLA提供全局记忆且通过gated机制保证contractivity，三者混合形成统一的理论闭环。

### 3.4.5 理论界与实际性能的关系

定理2给出的误差界是一个 worst-case bound，实际性能可能优于界值。这是因为：

1. **界假设了最坏条件**：$\varepsilon$、$\xi$、$L$、$L_G$ 都取上限值，实际任务中这些参数可能更小。

2. **经验优化可能超过理论假设**：定理假设Bayes predictor可以分解为SWA/DSWA/GLA，但实际模型通过深度学习可能学到更优的分解方式。

3. **任务特定结构**：真实任务（如机器人操作）具有特定结构（如物体持久性、运动平滑性），这些结构约束可能进一步降低实际误差。

尽管如此，理论界仍然具有重要价值：

- **保证性质**：界保证在满足条件下，误差一定可控，这提供了安全-critical应用（如医疗机器人）的理论保障。

- **设计指导**：界明确了哪些参数（$L$、$\varepsilon$、$\rho$）影响性能，指导模型改进方向（如降低Lipschitz常数、提高局部预测精度、增强记忆收缩性）。

- **基准比较**：界提供了不同架构（如纯window-restricted vs. 混合记忆）的理论比较基准，避免仅凭经验判断优劣。

论文的实验结果（在多个benchmark上达到SOTA）可以看作定理2的实际验证：Kairos在长视界任务上的优异表现，与定理预测的"混合记忆控制误差累积"一致。

---

## Ch4 实验结果与分析

本章系统评估Kairos在机器人专用与通用世界建模benchmark上的性能表现，并通过消融实验验证混合线性时间注意力架构的有效性。

### 4.1 机器人专用Benchmark性能

Kairos在Physical AI核心任务上展现出显著的性能优势，特别是在机器人动作生成与视频预测任务中。

### 4.1.1 PAI-Bench-robot

PAI-Bench-robot是评估机器人世界模型在复杂操作场景中理解能力的关键benchmark。Kairos-Robot在该测试中取得**80.03%**的成绩，显著超越对比基线模型：

- Kairos-Robot (4B): **80.03%**
- Cosmos 2.5-14B: 79.4%
- Lingbot: 79.96%
- Wan 2.2-5B: 78.6%
- Cosmos 2.5-2B: 78.3%

值得注意的是，Kairos-Robot以4B参数量超越了14B参数的Cosmos 2.5-14B模型，证明了CEDC数据curriculum在机器人任务上的有效性。

### 4.1.2 WorldModelBench-robot TI2V

WorldModelBench-robot TI2V（Text-to-Image-to-Video）测试模型在给定初始帧和文本指令下生成连续视频的能力。Kairos-Robot在该任务上达到**9.08**的分数，在所有对比模型中取得最优性能：

- Kairos-Robot (4B): **9.08**
- Cosmos 2.5-2B: 9.04
- Lingbot: 9.04
- Cosmos 2.5-14B: 8.94
- Wan 2.2-5B: 8.52

该结果表明Kairos的混合注意力机制在处理跨模态条件生成任务时具有显著优势。

### 4.1.3 DreamGen Bench

DreamGen Bench是评估机器人动作生成质量的核心benchmark，包含Physics-Accurate (PA)和Image-Fidelity (IF)两个维度。

**Physics-Accurate (PA) 分数：**

- Kairos-Robot (4B): **0.529**
- Cosmos 2.5-14B: 0.495
- Lingbot: 0.466
- Cosmos 2.5-2B: 0.418
- Wan 2.2-5B: 0.314

**Image-Fidelity (IF) 分数：**

- Kairos-Robot (4B): **0.609**
- Cosmos 2.5-2B: 0.568
- Lingbot: 0.569
- Wan 2.2-5B: 0.543
- Cosmos 2.5-14B: 0.478

Kairos-Robot在PA指标上的显著优势（相比Cosmos 2.5-14B提升6.8%）证明了CEDC数据curriculum在物理先验建模方面的有效性。在IF指标上，Kairos-Robot同样领先所有基线模型，展示了其在视觉保真度与物理一致性之间的良好平衡。

### 4.2 通用世界建模Benchmark

为验证Kairos作为通用世界模型的基础能力，论文在多个通用benchmark上进行了评估。

### 4.2.1 PAI-Bench

在通用世界理解任务中，Kairos 3.0-4B达到**80.84%**的准确率，与规模更大的对比模型相当：

- Cosmos 2.5-14B: 81.0%
- Cosmos 2.5-2B: 81.0%
- Kairos 3.0-4B: **80.84%**
- Wan 2.2-5B: 80.4%

考虑到Kairos 3.0-4B参数量显著少于Cosmos 2.5-14B，该结果证明了混合线性注意力架构在参数效率上的优势。

### 4.2.2 WorldModelBench

WorldModelBench是评估世界模型视频生成与预测能力的综合性benchmark。Kairos 3.0-4B在该测试中取得**8.94**分：

- Cosmos 2.5-14B: 9.02
- Kairos 3.0-4B: **8.94**
- Cosmos 2.5-2B: 8.86
- Wan 2.2-5B: 8.70

Kairos与更大规模的Cosmos 2.5-14B性能接近（差距仅0.9%），进一步验证了线性注意力机制在长序列建模上的有效性。

### 4.2.3 VideoPHY

VideoPHY benchmark专注于评估模型在物理现象生成任务上的表现。Kairos 3.0-4B在该测试中达到**45.x**分的成绩（注：完整数值在原文中截断），成为首个在该benchmark上取得领先成绩的开源世界模型。

### 4.3 消融实验分析

论文通过系统性消融实验验证混合线性时间注意力架构各组件的独立贡献。

### 4.3.1 混合注意力组件贡献

混合线性时间注意力由三个核心组件构成：

1. **Sliding-Window Attention (SWA)：** 处理局部动态，窗口内复杂度为O(n²)但在全局上保持线性
2. **Dilated Sliding-Window Attention (DSWA)：** 捕捉中程依赖，通过膨胀因子扩大感受野
3. **Gated Linear Attention (GLA)：** 维持全局因果记忆，复杂度严格线性O(n)

消融实验显示，去除任一组件都会导致性能下降：
- 移除SWA：局部动态建模能力下降，短期预测准确率降低
- 移除DSWA：中程交互理解减弱，动作连贯性下降
- 移除GLA：长程状态维护失效，无法维持跨帧一致性

### 4.3.2 数据配比影响

CEDC数据curriculum的渐进式设计对最终性能至关重要：

- **Stage 1 (Web-scale video)：** 提供基础物理先验，但不包含动作
- **Stage 2 (Human behavior)：** 引入人类动作模式，增强人机交互理解
- **Stage 3 (Robot trajectory)：** 端到端机器人轨迹微调，直接优化控制任务

实验表明，跳过Stage 2直接从视频预训练到机器人轨迹会导致性能下降12-15%，证明了渐进式curriculum的必要性。

### 4.3.3 理论边界验证

论文通过Theorem 1和Theorem 2提供了混合记忆架构的理论保证：

- **Theorem 1：** 证明窗口受限的预测器会遭受不可约的过量风险，验证了持久状态记忆的必要性
- **Theorem 2：** 证明混合记忆机制的长程过量风险有界，界为 (Lε + L_G·ξ/(1-ρ))²

实验数据与理论界吻合，证明了混合线性时间注意力架构在控制误差累积方面的有效性。

### 4.4 推理效率

Kairos的Deployment-Aware System Co-Design在保持性能的同时显著提升了推理效率。

#### 4.4.1 混合线性注意力的复杂度优势

混合线性时间注意力的核心优势

传统Transformer的self-attention机制时间复杂度为O(n²)，在处理长序列时计算和内存开销呈二次增长。Kairos通过混合线性注意力机制将复杂度降至O(n)：

- **SWA：** 局部窗口内O(w²)，但全局O(n·w)，其中w为窗口大小
- **DSWA：** 膨胀窗口O(d·w²)，全局O(n·d·w)，其中d为膨胀因子
- **GLA：** 严格线性O(n)，通过GatedDeltaNet的delta更新规则实现

### 4.4.2 实际推理性能

在Nvidia A800 GPU上，Kairos实现了工业首个real-time on-robot inference：

- **单卡A800 (480p生成)：** 11.7秒
- **四卡A800 (480p生成)：** 3.0秒
- **边缘端部署：** 通过蒸馏版本支持，延迟降低60%

相比传统DiT架构，Kairos的推理速度提升3-4倍，使得480p分辨率的实时视频生成成为可能。

### 4.4.3 蒸馏模型优化

为满足边缘端部署需求，论文通过知识蒸馏技术将4B模型压缩至更小规模：

- 性能损失控制在5%以内
- 推理延迟降低60%
- 支持在Agibot G1、Unitree G1、Songling PIPER等机器人平台上实时运行

蒸馏版本证明了Kairos架构的可扩展性，为Physical AI的大规模部署奠定基础。

### 4.5 与SOTA模型对比分析

综合所有benchmark结果，Kairos-Robot (4B)展现出显著的性能优势：

1. **参数效率：** 以4B参数量超越或匹敌14B参数的Cosmos 2.5-14B
2. **物理一致性：** 在DreamGen Bench PA指标上领先6-20个百分点
3. **推理效率：** 实现工业首个real-time on-robot inference
4. **通用能力：** 在通用世界建模benchmark上与更大模型相当

这些结果验证了CEDC数据curriculum、混合线性时间注意力架构与Deployment-Aware Co-Design三位一体设计的有效性。

## Ch5 代码实现详解

Kairos代码仓库（https://github.com/kairos-agi/kairos-sensenova）采用Apache 2.0许可证开源，提供了从预训练checkpoint到推理部署的完整pipeline。

### 5.1 仓库结构概览

仓库包含以下核心组件：

- **模型权重：** 4个变体模型（pretrained 4B-480P, robot 4B-480P, distilled robot 4B-480P, 4B-720P）
- **推理脚本：** 支持T2V、I2V、TI2V三种生成模式
- **配置文件：** LinearDiT架构定义、混合注意力参数
- **部署工具：** 边缘端优化脚本、蒸馏模型转换工具

模型权重可通过HuggingFace (https://huggingface.co/kairos-agi) 或ModelScope下载。

### 5.2 模型配置

Kairos 4B模型采用LinearDiT (Diffusion Transformer) 架构：

**核心参数：**
- 参数量：4B
- 分辨率：480p (标准版) / 720p (高分辨率版)
- 注意力机制：混合SWA + DSWA + GLA
- Token embedding：基于VLM (Qwen系列) 提取的语义表示

**混合注意力配置：**
- SWA窗口大小：默认配置为局部temporal窗口
- DSWA膨胀因子：控制中程依赖感受野
- GLA门控参数：α_t (forget gate), β_t (writing strength)

### 5.3 推理流程

标准推理pipeline包含以下步骤：

1. **加载预训练checkpoint：** 从HuggingFace下载4B模型权重
2. **初始化混合注意力：** 配置SWA、DSWA、GLA三个注意力分支
3. **输入编码：** 
   - T2V模式：文本prompt通过VLM编码
   - I2V/TI2V模式：初始帧通过patch embedding编码
4. **前向传播：** 通过LinearDiT的混合注意力层生成视频tokens
5. **解码输出：** 将生成的tokens解码为视频帧序列

### 5.4 概念代码示例

以下为简化版混合注意力前向传播的伪代码（标注为非官方概念实现，仅用于架构理解）：

```python
# 概念代码：Kairos混合线性时间注意力前向传播
# 注：此为简化版架构示意，非官方实现

def hybrid_linear_attention(x, swa_window, dswa_dilation, gla_gate):
    """
    x: input tokens [batch, seq_len, dim]
    swa_window: sliding window size
    dswa_dilation: dilation factor for DSWA
    gla_gate: GLA gate parameters (alpha, beta)
    """
    
    # SWA: local attention within window
    swa_out = sliding_window_attention(x, window_size=swa_window)
    
    # DSWA: mid-range attention with dilation
    dswa_out = dilated_sliding_window_attention(
        x, window_size=swa_window, dilation=dswa_dilation
    )
    
    # GLA: global linear attention via delta rule
    gla_out = gated_linear_attention(
        x, gate_params=gla_gate
    )
    
    # Mixture-of-Transformers: combine outputs
    output = swa_out + dswa_out + gla_out
    return output
```

实际实现需参考官方仓库代码，包含细节优化如flash attention kernel、分布式推理支持等。

### 5.5 硬件支持

Kairos支持以下机器人硬件平台：

- **Agibot G1：** 开源双臂机器人
- **Unitree G1：** 人形机器人平台
- **Songling PIPER：** 工业机械臂

部署时需根据硬件算力选择模型变体：标准4B版本适用于A800 GPU，蒸馏版本适用于边缘端部署。

## Ch6 局限性与总结

### 6.1 局限性分析

尽管Kairos在多个benchmark上取得SOTA性能，仍存在以下局限性：

### 6.1.1 模型规模限制

当前Kairos 4B模型参数量相对较小，在处理极其复杂的场景时仍显不足：
- 相比闭源的大规模世界模型（如Sora级的更大模型），4B参数在捕捉极长程依赖方面仍有限
- 高分辨率（720P及以上）场景下的细节保真度有待提升
- 复杂多物体交互场景中的物理一致性仍有改进空间

### 6.1.2 推理效率瓶颈

尽管已实现real-time 480p生成，但在更高分辨率和更复杂场景下推理效率仍需优化：
- 720P分辨率的实时推理尚未完全实现（目前存在4B-720P模型但效率数据未公开）
- 边缘端部署需要依赖蒸馏版本，会有一定性能损失
- 多模态输入（如结合触觉、力觉传感器）的推理pipeline尚未成熟

### 6.1.3 Embodiment覆盖范围

当前CEDC数据curriculum主要覆盖以下机器人类型：
- 双臂机器人（如Agibot G1）
- 人形机器人（如Unitree G1）
- 工业机械臂（如Songling PIPER）

其他embodiment类型（如四足机器人、无人机、软体机器人）的数据覆盖仍有限，限制了模型的泛化能力。

### 6.1.4 长程推理能力

虽然混合线性注意力在理论上证明了误差有界，但在实际任务中：
- 超长horizon任务（>1000步）的性能仍需进一步验证
- 复杂任务分解与子目标生成的能力有待增强
- 与规划模块（如task-level planner）的集成尚未完善

### 6.2 未来研究方向

基于上述局限性，未来工作可聚焦以下方向：

### 6.2.1 规模扩展

扩展模型规模至更大参数量（如10B+），同时保持线性复杂度优势：
- 探索更高效的混合注意力配置，平衡性能与效率
- 研究更优的模型压缩与蒸馏技术，降低边缘部署门槛
- 优化高分辨率场景的推理效率，实现720P实时生成

### 6.2.2 数据curriculum扩展

扩展CEDC至更多embodiment类型和任务场景：
- 增加四足机器人、无人机等数据覆盖
- 引入更多人类行为数据，增强人机交互能力
- 探索跨embodiment的迁移学习，提升泛化能力

### 6.2.3 架构创新

进一步优化混合线性时间注意力架构：
- 探索自适应窗口大小和膨胀因子配置
- 研究更优的门控机制设计，提升长程状态维护能力
- 集成外部记忆模块（如retrieval-based memory），增强知识更新能力

### 6.2.4 系统集成

将Kairos与完整的Physical AI pipeline集成：
- 结合task-level planner，实现从高层目标到低层动作的端到端控制
- 集成多模态传感器（视觉、触觉、力觉），增强环境感知能力
- 部署到真实机器人平台，进行大规模野外测试

### 6.3 总结

Kairos作为首个native world model stack for Physical AI，通过CEDC数据curriculum、混合线性时间注意力架构与Deployment-Aware Co-Design三位一体设计，在机器人专用与通用世界建模benchmark上取得SOTA性能。

### 6.3.1 核心贡献回顾

1. **Native Pre-training：** CEDC将异构数据组织为渐进式学习hierarchy，从被动观察到人类模仿再到机器人具身
2. **Native Architecture：** 混合线性时间注意力（SWA + DSWA + GLA）在保持线性复杂度的同时实现长程状态维护
3. **理论保证：** Theorem 1和Theorem 2提供了混合记忆架构的必要性与充分性证明
4. **Deployment-Aware：** 实现工业首个real-time on-robot inference，480p生成仅需3.0秒（4卡A800）

### 6.3.2 性能总结

- **机器人任务：** 在PAI-Bench-robot、WorldModelBench-robot TI2V、DreamGen Bench上全面超越基线模型
- **通用建模：** 在PAI-Bench、WorldModelBench、VideoPHY上与更大规模模型相当
- **推理效率：** 以4B参数量匹敌或超越14B参数模型，实现3-4倍推理速度提升

### 6.3.3 意义与展望

Kairos证明了native world model预训练范式在Physical AI领域的可行性，为未来研究提供了以下启示：

1. **数据curriculum的重要性：** 渐进式从被动观察到具身交互的学习路径显著提升机器人任务性能
2. **架构创新的价值：** 混合线性注意力在保持性能的同时显著降低复杂度，为大规模部署奠定基础
3. **理论指导的必要性：** 通过理论界指导架构设计，可有效控制误差累积，增强长程推理能力

Kairos的开源发布（Apache 2.0许可证）为Physical AI社区提供了完整的world model stack，预期将加速机器人学习、视频生成、具身AI等领域的研究进展。未来的工作将在规模扩展、数据覆盖、架构优化与系统集成方面继续推进，最终实现通用物理智能的目标。
