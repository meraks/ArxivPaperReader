# 深度精读报告：Hallucination in World Models is Predictable and Preventable

## 论文元数据
- 标题：Hallucination in World Models is Predictable and Preventable
- 作者：Nicklas Hansen, Xiaolong Wang (UC San Diego)
- arXiv ID：2606.27326
- 发表/提交日期：2026-06-25
- 官方代码：[MMBench2](https://github.com/nicklashansen/mmbench2)
- 代码发现方式：PDF / 网页检索

---

## 第1章：论文概述与核心贡献

现代生成式世界模型（generative world model）能够渲染高度逼真、动作可控的未来帧序列。论文从摘要即点明这类模型的一个核心缺陷：rollout **视觉上流畅（visually fluent）却偏离真实动力学（drift from ground-truth dynamics）**。作者将这种失败命名为 **hallucination（幻觉）**——该术语借自语言模型文献 \[Ji et al., 2023; Huang et al., 2025\]，但论文强调在世界模型中后果更为严重：幻觉轨迹会被直接喂给下游规划器与策略 \[Schrittwieser et al., 2020; Hafner et al., 2023\]，因此 rollout 期间的静默幻觉会直接转化为控制阶段的静默错误决策。

### 1.1 核心论点

论文的一句话主张是：

> Hallucination in world models is **inherently a data coverage issue**, and the same signals used to detect it can also be used for mitigation.

这与主流直觉形成对照——一种自然解读把幻觉视作**架构问题**，认为靠更大骨干网络与更多训练算力即可解决 \[Hoffmann et al., 2022\]。论文反驳这一观点：幻觉**集中在状态-动作空间的低覆盖区域** \[Levine et al., 2020; Gadre et al., 2023\]，因而（1）可由运行时可得的轻量信号预测，（2）可通过调整训练数据的采样而非改造架构来预防。实验进一步揭示：**单一根源——覆盖缺口——解释了模型流水线每个阶段**（tokenizer、动作条件、多步 rollout）的失败，并以三种不同的失效模式呈现（Figure 1）。

### 1.2 主要贡献清单

论文在 Introduction 末尾将贡献归纳为五点：

1. **MMBench2 数据集**：427 小时、210 任务的可视化世界模型数据集，包含 ground-truth 动作/奖励、live 模拟器与行为多样化数据（含人类游戏数据）。
2. **阶段式幻觉刻画**：将生成式世界模型的幻觉绑定到 tokenizer、action marginalization、rollout 三个阶段。
3. **三种无标签幻觉预测器**：tokenizer round-trip residual、flow instability、inter-seed variance，无需任何标签或额外训练即可在运行时检测幻觉。
4. **覆盖感知训练（coverage-aware training）**：单一重采样配方在三个预测器上同时降低幻觉，并以零额外成本提升 rollout 保真度。
5. **定向数据收集框架**：将预测器用作好奇心奖励，使预训练的 350M 世界模型仅用 50 条真实环境轨迹即可适应完全未见的环境。

整套工作同时发布了数据集、训练/评估代码、模型 checkpoint 以及一个用于开放式交互的浏览器界面（Webpage: [mmbench2](https://nicklashansen.com/mmbench2)）。

---

## 第2章：研究背景与动机

### 2.1 现代生成式世界模型的能力现状

相关工作章节（Appendix A）梳理了世界模型的两条脉络。一是抽象 latent 动力学模型 \[Ha and Schmidhuber, 2018; Hafner et al., 2020, 2023; Hansen et al., 2022, 2024\]；二是渲染完整像素观测的高容量生成模型 \[Micheli et al., 2023; Alonso et al., 2024; Valevski et al., 2024\]。近作进一步将规模推向异构视频语料 \[Bruce et al., 2024; NVIDIA, 2025; Wan et al., 2025\] 与实时可玩的神经环境 \[Quevedo et al., 2024; Rami, 2024; Parker-Holder et al., 2025\]。本文的 350M 模型直接基于 **Dreamer 4** \[Hafner et al., 2025\]——一种集成 tokenizer+dynamics、强动作条件化的架构，但作者强调其提出的信号与干预手段**本质上是模型无关的**，原则上适用于任何现代生成式世界模型。

### 2.2 幻觉问题的严重性与研究空白

幻觉已在语言模型 \[Ji et al., 2023; Huang et al., 2025\]、图像 \[Li et al., 2023\] 与视频生成 \[Huang et al., 2024\] 中被广泛研究。但论文指出一个被忽视的问题：**自回归世界模型会在 rollout 的哪个位置、为何失败，目前所知甚少**。Introduction 将此归因于理解幻觉所需的三类资源缺一不可、而现有任一基准都不具备：

- 对训练语料的完全控制；
- 跨越多任务、多领域的行为多样化数据；
- 用于通过在线交互探测覆盖缺口的 live 环境。

这正构成了 MMBench2 的设计动机。

### 2.3 现有方法的不足：重架构、轻覆盖

论文从两条相邻文献归纳出"现有方法为何不足"：

**不确定性估计文献**通过 deep ensembles \[Lakshminarayanan et al., 2017\]、MC dropout \[Gal and Ghahramani, 2016\]、post-hoc OOD 检测器 \[Lee et al., 2018; Liu et al., 2020\]，以及离线 model-based RL 中基于不确定性的惩罚策略 \[Yu et al., 2020; Kidambi et al., 2020\] 来刻画不确定性；Plan2Explore \[Sekar et al., 2020\] 等工作用 ensemble disagreement 作为单任务 RL 的探索信号。相比之下，本文的预测器**直接派生自现有世界模型、针对生成式世界模型会幻觉的三个阶段**（encoder、dynamics、decoder），既不需要辅助网络也不需要标签。

**覆盖感知训练与数据收集文献**认为数据规模与组成是生成模型性能的一阶杠杆 \[Hoffmann et al., 2022; Gadre et al., 2023\]；离线 RL 中数据覆盖已被证明约束策略改进 \[Levine et al., 2020; Kumar et al., 2020\]，催生了 ExoRL、RLUnplugged、V-D4RL 等精选数据集；好奇心驱动探索 \[Pathak et al., 2017, 2019; Burda et al., 2019\] 是其在线对应物。本文将好奇心机制从"驱动单任务策略探索"改造为"为生成式世界建模收集数据"，并以两种互补方式落地：均匀任务重采样，以及把预测器用作好奇心奖励。

---

## 第3章：MMBench2 数据集与模型架构

### 3.1 MMBench2 数据集

MMBench2 在 MMBench \[Hansen et al., 2026\]（200 任务在线 RL 基准，提供 4k 专家演示与 live 环境）之上扩展，提供 65,600 条轨迹、427 小时 224×224、15fps 视频（合计 23M 帧），跨 210 个连续控制任务、10 个领域。其与既有数据集的差异（Table 4）在于**同时**满足：ground-truth 动作与奖励标签、每个任务均有 live 模拟器、混合质量行为、跨任务领域与具身的广覆盖。

**任务构成**。10 个领域为：DMControl、DMControl Extended、Meta-World、ManiSkill3、MuJoCo、MiniArcade、Box2D、RoboDesk、OGBench、Continuous Atari（CALE）。Table 5 给出每个领域的任务数、动作维度区间、回合长度区间与奖励形式：

| 任务领域 | 任务数 | 动作维度 (min/max) | 回合长度 (min/max) | 奖励 |
|---|---|---|---|---|
| DMControl | 23 | 1 / 12 | 500 / 500 | dense/sparse |
| DMControl Ext. | 16 | 1 / 7 | 500 / 500 | dense/sparse |
| Meta-World | 49 | 4 / 4 | 100 / 100 | dense |
| ManiSkill3 | 37 | 1 / 12 | 25 / 500 | dense/sparse |
| MuJoCo | 6 | 1 / 8 | 50 / 1000 | dense/sparse |
| MiniArcade | 24 | 1 / 2 | 200 / 500 | dense/sparse |
| Box2D | 8 | 2 / 4 | 500 / 500 | dense |
| RoboDesk | 6 | 5 / 5 | 100 / 100 | dense |
| OGBench | 14 | 2 / 8 | 100 / 1000 | dense |
| Atari | 27 | 3 / 3 | 1000 / 1000 | sparse |

观测统一为 224×224 RGB 帧。动作维度在 1–16 之间，论文将每个动作向量 **zero-pad 到最大维度 d_a=16** 并提供逐维度 validity mask，使模型能忽略每个任务中非激活的维度。200 任务构成预训练集，另 10 个（本文新建）作为完全未见的迁移任务 held-out。

**覆盖的极不均匀性**。预训练语料每任务含 260 个回合，但回合长度差异巨大（ManiSkill3 部分任务 25 步、Atari 达 1000 步），导致 Figure 3 的重尾分布：**前 20 任务占总帧数的 26%（以 Atari 为主），末 20 任务仅占 0.7%（短时序操控任务），每任务帧数中位数为 65k**。这种不均正是后续"覆盖感知训练"的靶点。

**七种行为类型**。Appendix D 列举了数据采集行为，旨在最大化行为多样性（单一专家演示恰恰缺乏多样性，论文实验证明这是幻觉的重要来源）：

- **Random policy**：动作在 [−1,1] 均匀采样，动作多样但任务表现差；
- **No-op actions**：a=0，刻画无 agent 干预的纯动力学；
- **Expert actions**：直接采样专家策略 π⋆，任务表现高但多样性低；
- **Transformed expert**：以一定概率施加五种变换之一（scale ε∈[0,1]、action dropout、flip sign、全零动作、重复上一动作），提供反事实转移；
- **Structured noise**：在 π⋆ 上叠加三种噪声之一（高斯、Ornstein-Uhlenbeck 时序相关、与随机策略的凸组合），参数按回合采样；
- **Curiosity-driven**：用 CEM 规划在预训练世界模型中选最大化幻觉预测器 u_norm,r 的轨迹；
- **Human play**：人通过键盘 web 界面交互，共采集 1,400 条轨迹。

除人类数据外，单回合内频繁在上述行为间切换以最大化多样性（例如先施加固化噪声探索低频区域，再切回专家策略生成恢复行为）。所有数据记录 ground-truth 动作与环境奖励。

### 3.2 350M Dreamer 4 架构概览

模型遵循 Dreamer 4 的两阶段范式：先训练 video tokenizer（基于掩码自编码），再在冻结 tokenizer 产出的空间 latent token 上训练基于 block-causal Transformer、用 shortcut flow-matching 训练的 dynamics 模型。总参数 350M，由三部分构成。

**Tokenizer（2×50M ≈ 100M）**。非对称 encoder–decoder Transformer。Encoder 对 224×224 帧以 stride 14 做 patchify（每帧 256 个 patch token），前置 64 个可学习 latent query，将 latent 流投影到 **tanh 限制的 64 维 bottleneck**，得到每帧 code $z \in [-1,1]^{64\times 64}$（即 4,096 维 latent/帧）。Decoder 仅依据 latent code 重建图像。训练目标为掩码重建 \[He et al., 2022\]：每帧从 $U(0,0.9)$ 采样被掩码 patch 比例并替换为可学习 mask token，损失（pixel MSE + LPIPS \[Zhang et al., 2018\]）仅在掩码位置计算。Table 12 的超参数：embedding dim 512、heads 8、depth 12、MLP ratio 4、latent 数 64、bottleneck dim 64、MAE keep range \[0.1, 1.0\]、LPIPS 权重 0.2。Encoder 与 Decoder 各 50M 参数。

**Dynamics model（~230M，含预测头共 250M）**。block-causal Transformer，每个 timestep 的输入序列为 `[ACTION×1, SHORTCUT×1, SPATIAL×32, REGISTER×4, AGENT×4]`：action token 由 16 维 padded action 经 2 层 MLP 得到；shortcut-conditioning token 拼接离散化噪声水平 σ 与步长 $d=1/2^{step}$ 两路嵌入；spatial token 是经 factor k=2 空间打包后的 latent（(64,64)→(32,128)，把空间轴注意力成本减半、通道维翻倍，decode 时做逆解包）；register token \[Darcet et al., 2024\] 为 4 个可学习 "sink" token；agent token 由逐任务 CLIP 嵌入广播到时间轴初始化，作为 reward/BC 头的读出。Table 12：embedding dim 1024、heads 8、depth 16、MLP ratio 4、self-consistency fraction 0.25、context corruption τ_ctx=0.1。模型以 shortcut flow-matching \[Frans et al., 2025\] 训练，下一帧采样**最少 4 个 Euler 子步**即可完成（实现配置见 §3.3）。

**Heads（~20M）**。Reward predictor：经 agent token 的 attention pooling 读出，预测 L=8 多步 symlog two-hot 分布，255 个 bin，范围 [−10,10]，梯度回传穿过 dynamics 模型。BC policy：确定性高斯策略（对角协方差），MSE 损失预测 16 维 ground-truth 动作——这是对 Dreamer 4（仅考虑离散动作）的偏离。两个头都以 CLIP-ViT/B 嵌入（openai/clip-vit-base-patch32，512 维）的逐任务语言指令为条件。

**Block-causal 层与注意力设计**。每层由空间自注意力（per-frame token 序列）、因果时间自注意力（沿时间轴）与 SiLU-gated MLP（ratio 4）组成；注意力使用 RoPE \[Su et al., 2024\]、QK-normalization \[Henry et al., 2020\]、RMSNorm \[Zhang and Sennrich, 2019\] 预归一化且归一化层无 bias。Tokenizer 中采用 modality-aware mask：encoder 中 latent query 可见所有 token、patch query 仅在图像模态内自注意；decoder 中方向反转。Dynamics 中 action/shortcut/spatial/register token 互相可见，而 agent token 非对称隔离（agent query 可见一切，非 agent query 看不到 agent 的 key）。

### 3.3 训练配置

8× NVIDIA H100 GPU 训练。Tokenizer 预训练 300k 步（14 GPU days），dynamics 预训练 180k 步（24 GPU days）；微调后最终 checkpoint 分别训练 380k 与 210k 步，合计 58 GPU days，context 长度 T=24。优化器 AdamW，学习率 $1\times10^{-4}$，权重衰减 $1\times10^{-2}$；Tokenizer 有效 batch size 96、Dynamics 有效 batch size 512。所有损失项（pixel MSE、LPIPS、dynamics 的 empirical/self-consistency 双分支、reward two-hot 交叉熵）各自按自身 running RMS 归一化后再加权，使损失权重与绝对尺度解耦。

Shortcut flow-matching 细节（Appendix F）：离散化噪声水平 σ 用整数索引 $\{0,\dots,k_{\max}\}$，$k_{\max}=64$（0 为纯噪声、$k_{\max}$ 为干净）；步长以整数索引 $\{0,\dots,\log_2 k_{\max}\}$ 对应 $d=1/2^{step}$。每个 batch 的 ρ_self=0.25 部分施加自一致性 bootstrap：在 (σ,step) 处于 no_grad 下跑两次 step+1 的粗半步，取平均速度作为当前步预测速度的 stop-gradient 目标；其余 0.75 在最细步用经验单步回归。推理时以 step size $d=0.125$（即 K=8 子步）的 shortcut Euler 积分器逐帧采样，从 $z\sim\mathcal{N}(0,I)$ 出发：

$$b = \frac{\hat{x}_1 - z}{1-\sigma}, \qquad z \leftarrow z + b\cdot d$$

并对过去 token 施加 context corruption。需要说明：正文 §3 称该机制"**至少 4 个 Euler 子步**"即可生成下一帧（描述能力下限），而 Appendix F 给出的实际推理配置为 $d=0.125$（K=8 子步）；两者并不冲突，前者是 shortcut 模型支持的最少步数，后者是论文实际部署的步长。

---

## 第4章：幻觉模式与预测信号

生成式世界模型想象未来靠三步串联：encoder 把观测映射为 latent → action-conditioned dynamics 头预测下一 latent → decoder 把重建渲染回像素。**每个阶段都是在一个有限的状态-动作空间切片上训练的学习函数，都可能在被要求外推时独立失败**。因三阶段顺序组合，早期引入的幻觉（如被破坏的编码）会被后续阶段传播并放大，故**指认某一失效源于哪个阶段，是诊断与修复的前提**。

### 4.1 三种幻觉模式

Section 4.1 给出三类失效，分别绑定流水线的不同阶段（见 Figure 1）：

**(i) Perceptual hallucination（感知幻觉，绑定 tokenizer）**。即使尚未做任何 dynamics rollout（**horizon H=0 时即存在**），tokenizer 对观测的重建就已经偏离观测本身。机制上，encoder–decoder 对把 OOD 场景结构投影到其 latent "词汇表"中最近的 in-distribution 样例——例如一个未见过的迷宫布局被重建时，agent 与 goal 位置正确，但墙体换成训练时见过的另一种布局。dynamics 头随后把这个被破坏的场景当作 ground-truth 进行 rollout。该失效**仅是冻结 encoder–decoder 对的属性**。

**(ii) Action-marginalized hallucination（动作边缘化幻觉，绑定 dynamics）**。给定 context，预测的下一 latent 对输入动作**几乎不敏感**。rollout 视觉上合理，却塌缩到一个被动作边缘化的未来，使模型更像视频生成器而非可控世界模型。论文用**动作干预**来暴露该模式：在评估时对 batch 内的动作流做随机 shuffle，测量由此引起的 flow MSE 变化——一个会这样幻觉的模型，其 flow MSE 在干预下**几乎不动**。

**(iii) Scene-diverging hallucination（场景发散幻觉，绑定多步 rollout）**。自回归 rollout 会随预测 horizon 累积复合误差，但场景发散特指**物理上不可能事件**被预测出来（如 Pong 中球得分后"传送"回比赛）。该模式在数据覆盖差的状态中最频繁。

三类失效**探测模型互不相交的部分**：tokenizer、dynamics 的动作条件化、多步误差累积。

### 4.2 覆盖缺口：单一根源

Section 4.3 用数据中心的统一视角串联三类模式：每种失效机制上都是模型对状态-动作空间某区域"见得太少"的后果——感知幻觉是 tokenizer 重建分布的覆盖缺口；动作边缘化是动作条件化转移的覆盖缺口；场景发散是模型被要求想象的轨迹上的覆盖缺口。Figure 4 在四个任务上展示了数据覆盖与幻觉（tokenizer round-trip residual $u_r$）的关系：**幻觉集中在低覆盖区域，尤其是已访问状态分布的外围**。这一单一根源是"单一重加权配方即可同时改善三类信号"的理论依据。

### 4.3 三种幻觉预测信号

Section 4.2 提出三个机制各异但都强预测幻觉的指标，**无需任何标签或额外训练**，适合运行时检测。作者把三者收敛视为特征而非冗余：三个机制不同的预测器对幻觉事件一致同意，本身即是底层信号真实的证据。

**Predictor 1：tokenizer round-trip residual（$u_r$）**。

$$u_r = \lVert \hat{z} - \mathrm{Encode}(\mathrm{Decode}(\hat{z})) \rVert$$

对 dynamics 预测的下一 latent $\hat{z}$ 做一次 decode→encode 往返，取 latent 空间残差。它针对感知幻觉：一个解码帧落在 tokenizer manifold 之外（被破坏的场景布局或虚构物体）将无法在 re-encode 中存活，产生大的 $u_r$。

**Predictor 2：flow instability（$u_f$）**。测量 dynamics 头在给定 (context, action) 下，去噪器对干净目标的预测 $\hat{x}_1$ 在**相邻 Euler 积分子步间的变化量**（在后半段子步上取平均）。一个尖锐、良条件的 dynamics 头会快速收敛到稳定的 $\hat{x}_1$（低 $u_f$）；条件化几乎不提供信号的头会在子步间持续振荡（高 $u_f$）。

**Predictor 3：inter-seed variance（$u_s$）**。在固定过去与动作下，跨 $N$ 条独立去噪轨迹的下一 latent 预测的方差，针对场景发散幻觉：刻画跨噪声种子的认知不确定性——种子分歧大的区域，正是多步 rollout 会扇出的区域。

**Dynamism normalization**。朴素使用上述信号会被场景活跃度混淆：高运动转移会同时抬高 $u_r$、$u_f$、$u_s$。论文改报 dynamism-normalized 变体：

$$u_{\mathrm{norm}} = u \big/ m, \qquad m = \text{每步 latent 表示的 RMS 变化量}$$

$m$ 取数据集上的逐任务平均，或在线数据收集时取 running estimate。归一化后，每个预测器刻画的是**相对于场景里"发生了多少"的模型不确定性**。$u_{\mathrm{norm},r}$、$u_{\mathrm{norm},f}$、$u_{\mathrm{norm},s}$ 作为后续分析的主预测器。

### 4.4 相关性证据（详见 §5）

Section 5.1 计算开环 rollout ΔPSNR 与各预测器的 Spearman 相关：三者在 9k 条 held-out 24 帧序列上均呈强负相关，$u_{\mathrm{norm},r}$ ρ=−0.81、$u_{\mathrm{norm},f}$ ρ=−0.79、$u_{\mathrm{norm},s}$ ρ=−0.80（Figure 5；论文正文与摘要概括为 ρ≈−0.80），说明三者追踪同一底层误差。Appendix E（Table 7）的逐任务 AUROC 进一步证实：归一化版本**一致优于未归一化版本**（如 raw $u_f$ 仅 0.752/0.854）及依赖 latent 场景运动 $m$、逐任务帧数的基线（frame-count 基线仅 0.596/0.534）。

---

## 第5章：实验结果与分析

实验在 MMBench2 预训练语料上训练 350M 动作条件世界模型，训练数据约 20M 帧（200 任务），另留 3M 帧测试；虽 MMBench2 提供低维状态信息，本文**严格只考虑 RGB 观测**。评估覆盖四个训练阶段：(1) 不带奖励标签的 action-conditioned 预训练；(2) 覆盖感知"中期训练"（在上阶段基础上加本文重加权采样）；(3) 带奖励标签的 action-conditioned 建模；(4) 经定向数据收集在 seen/unseen 任务上微调。

### 5.1 评估指标定义

- **Reconstruction PSNR ↑**：仅评估 encoder/decoder 质量，不涉及 dynamics。
- **Rollout PSNR gain (dB) ↑**：相对"整段 horizon 重复最后一帧"基线的 rollout 质量增益；该朴素基线依任务可能出奇地强。**ΔPSNR ≤ 0 判定为场景发散**。
- **Action shuffle ratio ↑**：以一步 teacher-forced flow MSE 相对 batch-shuffled 动作的比值度量动作敏感性；**比值 ≤ 1.1 判定动作被忽略（边缘化）**。
- **Downstream task performance (normalized score) ↑**：以 CEM 做闭环 MPC 评估世界模型对下游任务的效用，plan horizon H=32、每 16 步重规划、3 次 CEM 迭代、种群 32、每候选 2 条 rollout、BC 先验暖启动；因奖励量级跨任务差数个数量级，报告归一化得分 $s\in[0,1]$。

### 5.2 Q1 检测：三预测器强相关（Figure 5）

如 §4.4 所述，三预测器对 rollout ΔPSNR 的 Spearman 相关分别为 −0.81 / −0.79 / −0.80。Table 7 的 AUROC（200 预训练任务的 held-out 数据，对 "action-ignored" 与 "scene-divergent" 两个二值标签）：

| 预测器 | Action ignored ↑ | Scene divergent ↑ |
|---|---|---|
| $u_{\mathrm{norm},r}$ | 0.887 | 0.919 |
| $u_{\mathrm{norm},f}$ | 0.868 | 0.939 |
| $u_{\mathrm{norm},s}$ | 0.873 | 0.934 |
| Scene motion $m$（latent） | 0.803 | 0.927 |
| kNN distance（global） | 0.814 | 0.731 |
| $u_f$（raw） | 0.752 | 0.854 |
| n frames（baseline） | 0.596 | 0.534 |

归一化三预测器在两类标签上均达 0.87–0.94，显著优于 raw 与 frame-count 基线。

### 5.3 Q2 预训练：覆盖感知训练同时改善三类信号（Table 1）

把基模型预训练各延长 30k 步（tokenizer 与 dynamics 各一），比较两种采样：跨帧均匀（默认）与**跨任务均匀（本文）**。Table 1 给出相对基模型在 200 任务 held-out 轨迹上的均值变化（Tok_FT=tokenizer 微调 30k 步用覆盖感知训练后再以默认采样微调 dynamics 30k 步；Dyn_FT 反之；Both=两者各 30k 步用覆盖感知训练）：

| 指标 | Tok_FT | Dyn_FT | Both |
|---|---|---|---|
| Recon PSNR (dB) ↑ | +0.46 | −0.01 | +0.44 |
| Action-shuffle ratio ↑ | +0.02 | +0.27 | +0.29 |
| Rollout ΔPSNR (dB) ↑ | +0.42 | +0.68 | +0.88 |
| $u_{\mathrm{norm},r}$ ↓ | −0.07 | −0.16 | −0.20 |
| $u_{\mathrm{norm},f}$ ↓ | −0.03 | −0.06 | −0.07 |
| $u_{\mathrm{norm},s}$ ↓ | −0.06 | −0.13 | −0.14 |

tokenizer 与 dynamics 都从中受益，**两者同时采用覆盖感知训练取得最佳总成绩**——三类预测器与 rollout ΔPSNR 同向改善（其中 Rollout ΔPSNR +0.88 dB），印证"单一覆盖杠杆可同时移动所有信号"。论文也实验了 loss re-weighting，但发现采样干预更优。

### 5.4 Q3 定向数据收集：好奇心数据媲美 expert/human（Table 2）

构造 10 个 seen + 10 个 unseen 任务集，每种行为各采 50 回合/任务，仅改变行为策略；微调 tokenizer 50k 步、dynamics 30k 步。10 unseen 任务的离线指标与闭环 MPC（Table 2，测试集为等量专家轨迹与人类数据）：

| Method | Tok_FT/Dyn_FT | Recon PSNR ↑ | ΔPSNR ↑ | Shuf. ↑ | $u_{\mathrm{norm},r}$ ↓ | Task perf (MPC) ↑ |
|---|---|---|---|---|---|---|
| Random policy | — | — | — | — | — | 0.118 |
| Base | ○/○ | 17.37 | −12.44 | 1.12 | 3.860 | — |
| Coverage-aware | ○/○ | 17.21 | −12.52 | 1.29 | 3.769 | 0.276 |
| No-op actions | ●/● | 34.74 | +0.66 | 1.55 | 1.486 | 0.163 |
| Random policy | ●/● | 35.81 | +2.66 | 2.00 | 1.201 | 0.228 |
| Expert policy | ●/● | 35.86 | +2.84 | 2.04 | 1.131 | 0.362 |
| Human play | ●/● | 37.11 | +3.89 | 2.42 | 1.002 | 0.362 |
| Curiosity ($u_{\mathrm{norm},r}$) | ●/● | 36.05 | +3.00 | 2.00 | 1.144 | 0.325 |
| All (combined) | ●/● | 37.91 | +4.02 | 2.34 | 0.975 | 0.390 |

（Tok_FT/Dyn_FT 列 ○ 表示仅预训练、● 表示已微调。）关键结论：预训练已具备一定零样本迁移能力（MPC 0.276，是 random 策略 0.118 的 2.3×）；**仅采 50 回合/任务、用 $u_{\mathrm{norm},r}$ 好奇心驱动收集，把 MPC 提升到 0.325，达到 expert/human 上界（0.362）的约 90%，且不使用任何特权行为**；All(combined) 进一步达到 0.390。

**在线收集闭环**：在与 live 环境交互时，候选轨迹先在世界模型中 rollout，按预测幻觉打分，得分最高的轨迹在环境中执行，从而"按构造"覆盖此前引发幻觉的转移。论文以预测 horizon H=32、每 K=16 步重规划进行闭环收集。Figure 6/10 展示 no-op/random/expert/curiosity/human 五种策略在多任务上的状态密度差异。

### 5.5 Q4 泛化：零样本 vs 微调，以及现成 tokenizer（Table 3）

Discussion 部分回答"为何不同数据源信息量不同"：世界建模需要宽广的状态-动作覆盖（random 策略不保证），而下游任务是目标导向、状态-动作分布窄得多，故闭环 MPC 主要测量该窄子空间的模型精度——这"可以说是设计使然"，并引出"未来如何更好评估世界模型"的开放问题。

对于感知幻觉，Table 3 比较自研 tokenizer（微调前后）与四个 off-the-shelf tokenizer（PSNR / LPIPS，10 seen + 10 unseen 任务集）：

| Tokenizer | Params | Latent/帧 | Seen PSNR ↑ | Unseen PSNR ↑ | Δ(S−U) | Seen LPIPS ↓ | Unseen LPIPS ↓ |
|---|---|---|---|---|---|---|---|
| Ours Base | 102M | 4096 | 38.29 | 17.34 | +20.95 | 0.011 | 0.389 |
| Ours Coverage-aware | 102M | 4096 | 38.93 | 17.12 | +21.81 | 0.008 | 0.348 |
| Ours Post-FT | 102M | 4096 | 39.66 | 38.04 | +1.62 | 0.007 | 0.010 |
| SD-VAE-MSE | 84M | 3136 | 33.32 | 32.39 | +0.93 | 0.031 | 0.030 |
| Cosmos-CV8x8x8 (1.0) | 106M | 2048 | 32.80 | 32.72 | +0.08 | 0.050 | 0.042 |
| Wan 2.1 VAE | 127M | 4096 | 36.45 | 36.62 | −0.17 | 0.010 | 0.010 |
| DC-AE-f32c32 | 323M | 2048 | 31.49 | 32.15 | −0.66 | 0.035 | 0.031 |

自研 tokenizer 在 200 任务训练集上**大幅领先**最强现成 tokenizer Wan 2.1 VAE（seen 38.29 vs 36.45），但在未见任务上泛化较差（unseen 17.34 vs 36.62）；一旦在未见集上微调，自研 Post-FT 在两端都反超（seen 39.66、unseen 38.04）。结论：现成 tokenizer 对世界建模确有前景，但**域内微调仍有切实收益**。此外 Table 11 表明：是否联合训练 reward head（梯度回传至 dynamics）对结果无显著差异（如 Rollout ΔPSNR 3.01 vs 3.14），reward-free 控制即可。

### 5.6 核心定量小结

- 三预测器 Spearman ρ≈−0.80、AUROC 0.87–0.94；
- 覆盖感知训练在 Rollout ΔPSNR 上 +0.88 dB（Both 配方）、三类 $u_{\mathrm{norm}}$ 同向下降；
- 好奇心驱动 50 回合/任务把未见任务 MPC 从 0.276 提到 0.325，达 expert/human（0.362）约 90%；
- 总训练成本 58 GPU days（8×H100），context T=24。

---

## 第6章：代码实现详解

> 本章依据论文官方仓库 [MMBench2](https://github.com/nicklashansen/mmbench2) 的 README 结构梳理（仓库结构与训练流程信息来自研究材料的 README 摘要）。论文 PDF 附录 F（Implementation Details）提供了与代码对应的实现细节，已在第 3 章给出。**下列代码片段为概念示意，旨在把论文公式落到可读伪代码，并非从官方源码逐行复制——请勿当作官方实现引用。**

### 6.1 仓库结构

`src/` 目录包含以下脚本，覆盖数据、训练、评估与交互全链路：

| 脚本 | 职责（依据 README/论文方法） |
|---|---|
| `download_dataset.py` | 下载 HF 数据集 `nicklashansen/mmbench2` |
| `download_checkpoints.py` | 下载 HF 模型 `nicklashansen/mmbench2-models`（base / coverage_aware / combined） |
| `preprocess_dataset.py` | 预处理为训练所需格式 |
| `train_tokenizer.py` | 阶段一：训练 video tokenizer（MAE+LPIPS，每损失项 running-RMS 归一化） |
| `train_dynamics.py` | 阶段二：在冻结 tokenizer 上训练 shortcut flow-matching dynamics |
| `uncertainty.py` | 计算三种幻觉预测器 $u_r,u_f,u_s$ 及 dynamism 归一化 |
| `curiosity.py` | 用 $u_{\mathrm{norm},r}$ 作好奇心奖励、CEM 规划的在线数据收集 |
| `collect_data.py` | 执行七种行为策略的数据采集 |
| `interactive.py` | 浏览器交互界面，支持开放式 world-model 探索 |

环境与运行：conda 环境、推理需 ≥4GB 显存的 CUDA GPU；训练复现论文 8×H100 配置（tokenizer 300k 步、dynamics 180k 步）；通过 `run_interactive.sh [base|coverage_aware|combined]` 切换 checkpoint。检查点含义：base（预训练）、coverage_aware（覆盖感知微调）、combined（微调 + 全数据源）。

### 6.2 核心方法的概念示意

**Tokenizer round-trip residual（对应 Predictor 1）**：

```python
# ⚠️ 非官方概念实现，未经验证
def tokenizer_round_trip_residual(dynamics_predicted_latent, tokenizer):
    z_hat = dynamics_predicted_latent                      # dynamics 预测的下一 latent
    recon = tokenizer.decode(z_hat)                        # latent -> 帧
    z_re = tokenizer.encode(recon)                         # 帧 -> latent（往返）
    u_r = (z_hat - z_re).norm(dim=-1)                      # latent 空间残差
    return u_r
```

**Flow instability（对应 Predictor 2，对子步间 $\hat{x}_1$ 变化取后半段平均）**：

```python
# ⚠️ 非官方概念实现，未经验证
def flow_instability(x1_per_substep):
    # x1_per_substep[i] 是第 i 个 Euler 子步处对干净目标的预测
    half = len(x1_per_substep) // 2
    deltas = [x1_per_substep[k+1] - x1_per_substep[k]
              for k in range(half, len(x1_per_substep) - 1)]
    u_f = sum(d.norm(dim=-1) for d in deltas) / len(deltas)
    return u_f
```

**Inter-seed variance（对应 Predictor 3，固定过去/动作、跨 N 条去噪轨迹）**：

```python
# ⚠️ 非官方概念实现，未经验证
def inter_seed_variance(z_next_seeds):
    # z_next_seeds: [N, ...] N 条独立去噪轨迹的下一 latent 预测
    u_s = z_next_seeds.var(dim=0).sum(dim=-1)             # 跨种子方差
    return u_s
```

**Dynamism normalization 与三预测器归一化**：

```python
# ⚠️ 非官方概念实现，未经验证
def dynamism_normalize(u, m):
    # m = 每步 latent 表示的 RMS 变化量（数据集逐任务平均，或在线 running estimate）
    return u / m
# 主预测器：u_norm_r, u_norm_f, u_norm_s
```

**Shortcut Euler 推理（Appendix F，步长 d=0.125，K=8 子步）**：

```python
# ⚠️ 非官方概念实现，未经验证
def shortcut_euler_sample(x_hat1_fn, sigma, d=0.125, K=8):
    z = torch.randn_like(...)                              # z ~ N(0, I)
    for _ in range(K):
        x_hat1 = x_hat1_fn(z, sigma)                       # 预测干净目标
        b = (x_hat1 - z) / (1 - sigma)                     # 速度场
        z = z + b * d                                      # Euler 推进
    return z
```

### 6.3 训练流程（两阶段）

1. **阶段一 tokenizer 预训练**：`train_tokenizer.py`，全训练语料，MAE+LPIPS、各损失项 running-RMS 归一化、300k 步 / 14 GPU days、有效 batch 96。
2. **阶段二 dynamics 预训练**：`train_dynamics.py`，冻结 tokenizer，shortcut flow-matching（self-consistency fraction 0.25）、context T=24、180k 步 / 24 GPU days、有效 batch 512。
3. **中期训练/微调**：覆盖感知训练（跨任务均匀采样）把 tokenizer、dynamics 各延长 30k 步；reward/BC 头在预训练后初始化（reward 梯度回传穿过 dynamics，L=8 symlog two-hot，255 bin，\[−10,10\]；BC 确定性高斯 + MSE）。
4. **定向数据收集**：`curiosity.py` 以 $u_{\mathrm{norm},r}$ 为目标、CEM（H=32、每 16 步重规划）规划候选轨迹并在 live 环境执行，每任务 50 回合。

### 6.4 运行交互式界面

`run_interactive.sh [base|coverage_aware|combined]` 选择 checkpoint 启动 `interactive.py` 的浏览器界面，对应论文随数据发布的开放式 world-model 交互接口（论文 Webpage 同时提供视频与资源）。

---

## 第7章：局限性与延伸阅读

### 7.1 局限性（论文 Limitations 小节）

论文明确列出三方面局限：

- **规模**：研究在 **350M 参数、210 个仿真控制任务**上完成。结论能否迁移到十亿参数级模型仍是开放的经验问题。
- **数据真实性**：未涉及带传感器噪声与部分可观测性的**真实机器人数据**。
- **领域范围**：限于连续控制领域；作者同时承认训练大型世界模型的算力成本，以及对更丰富数据集的需求。

### 7.2 未来方向

Conclusion 指出：幻觉**首要是一个数据覆盖问题**，三预测器以 ρ≈−0.80 追踪 rollout ΔPSNR，并衍生两条互补配方（覆盖感知采样同时降低三类失效；把预测器用作好奇心奖励使预训练模型迁移到完全未见任务）。Discussion 进一步抛出"如何更好评估世界模型"的开放问题——下游任务目标导向、分布窄，意味着某些行为比其他行为更值得建模，评估口径需重新审视。

### 7.3 完整参考文献

> 以下为论文完整参考文献列表，按原文顺序排列。

1. Rishabh Agarwal, Dale Schuurmans, and Mohammad Norouzi. An optimistic perspective on offline reinforcement learning. In *International Conference on Machine Learning*, pages 104–114. PMLR, 2020.
2. Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Diffusion for world modeling: Visual details matter in atari. In *Thirty-eighth Conference on Neural Information Processing Systems*, 2024.
3. Bowen Baker, Ilge Akkaya, Peter Zhokov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune. Video pretraining (VPT): Learning to act by watching unlabeled online videos. *Advances in Neural Information Processing Systems*, 35:24639–24654, 2022.
4. Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. OpenAI Gym, 2016.
5. Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In *Forty-first International Conference on Machine Learning*, 2024.
6. Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. In *International Conference on Learning Representations (ICLR)*, 2019.
7. Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine. Deep reinforcement learning in a handful of trials using probabilistic dynamics models. In *NeurIPS*, 2018.
8. Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In *International Conference on Learning Representations (ICLR)*, 2024.
9. Sudeep Dasari, Frederik Ebert, Stephen Tian, Suraj Nair, Bernadette Bucher, Karl Schmeckpeper, Siddharth Singh, Sergey Levine, and Chelsea Finn. RoboNet: Large-scale multi-robot learning. In *Conference on Robot Learning*, pages 885–897. PMLR, 2019.
10. Jesse Farebrother and Pablo Samuel Castro. CALE: Continuous arcade learning environment. *Advances in Neural Information Processing Systems*, 37:134927–134946, 2024.
11. Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. In *The Thirteenth International Conference on Learning Representations*, 2025.
12. Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. DataComp: In search of the next generation of multimodal datasets. *Advances in Neural Information Processing Systems*, 36:27092–27112, 2023.
13. Yarin Gal and Zoubin Ghahramani. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, volume 48, pages 1050–1059, 2016.
14. Caglar Gulcehre, Ziyu Wang, Alexander Novikov, Thomas Paine, Sergio Gómez, Konrad Zolna, Rishabh Agarwal, Josh S. Merel, Daniel J. Mankowitz, Cosmin Paduraru, Gabriel Dulac-Arnold, Jerry Li, Mohammad Norouzi, Matthew Hoffman, Nicolas Heess, and Nando de Freitas. RL Unplugged: A suite of benchmarks for offline reinforcement learning. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.
15. David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. In *Advances in Neural Information Processing Systems 31*, pages 2451–2463. Curran Associates, Inc., 2018.
16. Danijar Hafner, Timothy P. Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. *ArXiv*, abs/1912.01603, 2020.
17. Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. *arXiv preprint arXiv:2301.04104*, 2023.
18. Danijar Hafner, Wilson Yan, and Timothy Lillicrap. Training agents inside of scalable world models. *arXiv preprint arXiv:2509.24527*, 2025.
19. Nicklas Hansen, Xiaolong Wang, and Hao Su. Temporal difference learning for model predictive control. In *International Conference on Machine Learning*, 2022.
20. Nicklas Hansen, Hao Su, and Xiaolong Wang. TD-MPC2: Scalable, robust world models for continuous control. In *International Conference on Learning Representations (ICLR)*, 2024.
21. Nicklas Hansen, Hao Su, and Xiaolong Wang. Learning massively multitask world models for continuous control. In *International Conference on Learning Representations (ICLR)*, 2026.
22. Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 16000–16009, 2022.
23. Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for transformers. In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 4246–4253, 2020.
24. Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, DDL Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*, 10, 2022.
25. Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. *ACM Transactions on Information Systems*, 43(2):1–55, 2025. doi: 10.1145/3703155.
26. Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024.
27. Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model: Model-based policy optimization. *ArXiv*, abs/1906.08253, 2019.
28. Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12):1–38, 2023. doi: 10.1145/3571730.
29. Harini Kannan, Danijar Hafner, Chelsea Finn, and Dumitru Erhan. RoboDesk: A multi-task reinforcement learning benchmark. <https://github.com/google-research/robodesk>, 2021.
30. Alexander Khazatsky, Karl Pertsch, Suraj Nair, et al. DROID: A large-scale in-the-wild robot manipulation dataset. 2024.
31. Rahul Kidambi, Aravind Rajeswaran, Praneeth Netrapalli, and Thorsten Joachims. MOREL: Model-based offline reinforcement learning. *Advances in Neural Information Processing Systems*, 33:21810–21823, 2020.
32. Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative Q-learning for offline reinforcement learning. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.
33. Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.
34. Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2018.
35. Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. *arXiv preprint arXiv:2005.01643*, 2020.
36. Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2023.
37. Weitang Liu, Xiaoyun Wang, John D. Owens, and Yixuan Li. Energy-based out-of-distribution detection. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.
38. Cong Lu, Philip J. Ball, Tim G. J. Rudner, Jack Parker-Holder, Michael A. Osborne, and Yee Whye Teh. Challenges and opportunities in offline reinforcement learning from visual observations. *Transactions on Machine Learning Research*, 2023.
39. Loïc Magne, Anas Awadalla, Guanzhi Wang, Yinzhen Xu, Joshua Belofsky, Fengyuan Hu, Joohwan Kim, Ludwig Schmidt, Georgia Gkioxari, Jan Kautz, Yisong Yue, Yejin Choi, Yuke Zhu, and Linxi "Jim" Fan. NitroGen: An open foundation model for generalist gaming agents, 2026.
40. Vincent Micheli, Eloi Alonso, and François Fleuret. Transformers are sample-efficient world models. In *The Eleventh International Conference on Learning Representations*, 2023.
41. NVIDIA. Cosmos world foundation model platform for physical AI. *arXiv preprint arXiv:2501.03575*, 2025.
42. Open X-Embodiment Collaboration. Open X-Embodiment: Robotic learning datasets and RT-X models. <https://arxiv.org/abs/2310.08864>, 2023.
43. Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. OGBench: Benchmarking offline goal-conditioned RL. In *International Conference on Learning Representations (ICLR)*, 2025.
44. Jack Parker-Holder, Shlomi Fruchter, and Google DeepMind. Genie 3: A new frontier for world models. *Google DeepMind Blog*, August 2025.
45. Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In *International Conference on Machine Learning*, pages 2778–2787. PMLR, 2017.
46. Deepak Pathak, Dhiraj Gandhi, and Abhinav Gupta. Self-supervised exploration via disagreement. In *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 2019.
47. Julian Quevedo, Quinn McIntyre, Spruce Campbell, Xinlei Chen, Robert Wachen, Decart, and Etched. Oasis: A universe in a transformer. Decart, October 2024.
48. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International Conference on Machine Learning*, pages 8748–8763. PMLR, 2021.
49. Rami. Lucid v1: A world model that does go brrr on consumer hardware. Substack, November 2024.
50. Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering Atari, Go, Chess and Shogi by planning with a learned model. *Nature*, 588(7839):604–609, 2020.
51. Ramanan Sekar, Oleh Rybkin, Kostas Daniilidis, Pieter Abbeel, Danijar Hafner, and Deepak Pathak. Planning to explore via self-supervised world models. In *Proceedings of the 37th International Conference on Machine Learning*, volume 119, pages 8583–8592. PMLR, 2020.
52. Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063, 2024.
53. Stone Tao, Fanbo Xiang, Arth Shukla, et al. ManiSkill3: GPU parallelized robotics simulation and rendering for generalizable embodied AI. *Robotics: Science and Systems*, 2025.
54. Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, et al. DeepMind Control Suite. Technical report, DeepMind, 2018.
55. Emanuel Todorov, Tom Erez, and Yuval Tassa. MuJoCo: A physics engine for model-based control. In *2012 IEEE/RSJ International Conference on Intelligent Robots and Systems*, pages 5026–5033. IEEE, 2012.
56. Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines, 2024.
57. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 2017.
58. Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe Hansen-Estruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, and Sergey Levine. BridgeData V2: A dataset for robot learning at scale. In *Conference on Robot Learning*, pages 1723–1736. PMLR, 2023.
59. Team Wan, Ang Wang, Baole Ai, et al. Wan: Open and advanced large-scale video generative models. *arXiv preprint arXiv:2503.20314*, 2025.
60. Denis Yarats, David Brandfonbrener, Hao Liu, Michael Laskin, Pieter Abbeel, Alessandro Lazaric, and Lerrel Pinto. Don't change the algorithm, change the data: Exploratory data for offline reinforcement learning. In *International Conference on Learning Representations (ICLR)*, 2022.
61. Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-World: A benchmark and evaluation for multi-task and meta reinforcement learning. In *Conference on Robot Learning*, 2019.
62. Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Zou, Sergey Levine, Chelsea Finn, and Tengyu Ma. MOPO: Model-based offline policy optimization. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.
63. Biao Zhang and Rico Sennrich. Root mean square layer normalization. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.
64. Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 586–595, 2018.

### 7.4 补充附录：微调任务列表与消融实验

**微调任务列表（Table 6）**。论文在 seen + unseen 共 20 个任务上进行微调实验：

| 类别 | 任务名 | 领域 |
|------|--------|------|
| Seen | cup-catch, finger-turn-easy, mw-push, ms-push-cube, lunarlander-hover, og-point-maze, og-point-bottleneck, pygame-point-maze-var1, pygame-pong, pygame-bird-attack | DMControl / Meta-World / ManiSkill / Box2D / OGBench / MiniArcade |
| Unseen | cup-catch-var1, finger-turn-easy-var1, ms-push-banana, og-point-var1, og-point-var2, pygame-point-maze-var4, pygame-reacher-easy, pygame-dungeon-explorer1, pygame-foraging, pygame-whirlpool | DMControl / ManiSkill / OGBench / MiniArcade |

**专家测试集与人类测试集消融（Table 8/9）**。Table 2 报告的离线指标在专家轨迹与人类数据等量混合的测试集上计算。论文 Table 8（仅专家测试集）与 Table 9（仅人类测试集）的 MPC 性能一致（均为 0.325 for curiosity），表明好奇心策略在两种分布上均有鲁棒表现。

**Reward 微调的影响（Table 11）**。是否在 dynamics 训练中联合训练 reward head（梯度回传至 dynamics）对结果无显著影响：Rollout ΔPSNR 3.01 vs 3.14（无 reward vs 有 reward），表明 reward-free control 即可获得接近的性能。

**完整超参数表（Table 12）**。已整合于 §3.2–§3.3 架构与训练配置部分。

### 7.5 公式与代码校验

> 以下校验基于 PDF 原文逐项比对，确认 MD 文档中的公式和代码正确。

| 公式/代码 | 原文定义 | MD 实现 | 结论 |
|-----------|---------|---------|------|
| $u_r = \|\hat{z} - \mathrm{Encode}(\mathrm{Decode}(\hat{z}))\|$ | §4.2 "the latent-space residual of a single decode→encode round-trip" | `tokenizer_round_trip_residual` 计算 `(z_hat - z_re).norm(dim=-1)` | ✅ |
| $u_f$：相邻 Euler 子步间 $\hat{x}_1$ 变化，后半段取平均 | §4.2 "how much the denoiser's clean-target prediction moves between successive Euler integration substeps, averaged over the second half" | `flow_instability` 取后半段子步的 $\hat{x}_1$ 差值范数平均 | ✅ |
| $u_s$：跨 $N$ 条独立去噪轨迹的 latent 方差 | §4.2 "inter-seed variance of the next-latent prediction across N independent denoising trajectories" | `inter_seed_variance` 计算 `.var(dim=0).sum(dim=-1)` | ✅ |
| $u_{\mathrm{norm}} = u / m$，$m$ 为每步 latent RMS 变化量 | §4.2 "per-step RMS change of the latent representation" | `dynamism_normalize` 返回 `u / m` | ✅ |
| Shortcut Euler: $b = (\hat{x}_1 - z)/(1-\sigma),\; z \leftarrow z + b\cdot d$ | Appendix F "b = (x̂₁ - z)/(1 - σ), z ← z + b·d" | `shortcut_euler_sample` 循环 K=8 次、d=0.125 的 Euler 更新 | ✅ |
| $d=0.125$, $K=8$ 子步 | Appendix F "step size d=0.125 (i.e., K=8 substeps)" | `d=0.125, K=8` | ✅ |
| 自洽 bootstrap 比例 $\rho_{\mathrm{self}}=0.25$ | Appendix F "self_consistency_fraction = 0.25" | §3.3 明确记录为 0.25 | ✅ |
| $k_{\max}=64$ 离散噪声水平 | Appendix F "kmax=64" | §3.3 明确记录 | ✅ |

所有公式和代码均与原文一致，无需修正。

---

*本报告所有数值均取自论文 PDF 原文（arXiv:2606.27326）及其 Table 1/2/3/6/7/8/9/10/11/12 与正文、附录 A–F；代码为概念示意而非官方源码。*
