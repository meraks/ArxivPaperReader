> **论文**：NCP-ArchPreview Technical Report: Moving towards Latent Space Language Models through Next Concept Prediction
> **作者**：The Intern-NCP Team（Jiaqi Cao, Chiyu Chen, Shuang Cheng, Xu Cheng, Beiya Dai, Yufan Feng, Kewen Ge, Ruijun Ge, Jiayi Huang, Yang Jiao, Dahua Lin, Zhouhan Lin, Yifan Liu, Yuliang Liu, Biqing Qi, Mowen Ruan, Junzhe Shen, Yunchong Song, Hao Sun, Zhongbo Tian, Yixuan Wang, Rubin Wei, Jiaxin Xiong, Kangyu Yang, Qian Yao, Qi Zhang, Bowen Zhou）
> **机构**：Shanghai AI Lab；LUMIA Lab, Shanghai Jiao Tong University
> **arXiv ID**：2609.10715（cs.CL）
> **发表时间**：2026-09-11
> **代码仓库**：https://github.com/LUMIA-Group/ncp_olmo_eval
> **模型权重**：https://huggingface.co/collections/ArchSpace-Collection/ncp-archpreview

## 第 1 章 论文概述与核心贡献

### 1.1 一句话定位

NCP-ArchPreview 把语言模型内部自发涌现的语义抽象从"下一 token 预测的副产品"提升为**一等预测目标**：在标准 NTP 之外并行训练 Next Concept Prediction（NCP），用自学习的乘积量化概念词表把跨多 token 的离散概念作为监督对象，在 OLMo-3-7B backbone 上以 8.94B 参数、5.73T token 尺度验证了这一范式的可扩展性——仅用 51.3% 的训练 token 即达到 baseline 的最终预训练 loss（1.95× 收敛加速），下游 26 项指标宏平均提升 2.45 分。

### 论文图表总览

| 编号 | 内容 | 章节 |
|:-----|:-----|:-----|
| **Figure 1** | Stage-1 / Stage-2 训练动力学对比（OLMo-3-7B vs NCP-ArchPreview） | 第 5 章 |
| **Figure 2** | 模型架构总览（Token Encoder / Concept Module / Token Decoder + IRC/CRC） | 第 3 章 |
| **Figure 3** | 前 200B token 训练 loss 消融（三条 vanilla 基线 + 三个递进配置） | 第 5 章 |
| **Figure 4** | 缩放律曲线：NCP-ArchPreview vs OLMo-3（1.74× 计算效率） | 第 5 章 |
| **Figure 5** | OLMo-3 layer-wise Q/K norm 下的数值不稳定现象 | 第 4 章 |
| **Figure 6** | 数值不稳定消融（AdamW / Muon / per-head Q/K norm） | 第 4 章 |
| **Figure 7** | VQ / LoRA / Full 适配的训练吞吐与单卡显存占用 | 第 6 章 |
| **Figure 8** | 3B 规模 NTP loss 曲线（含 MTP 对照与早期交叉点） | 第 6 章 |
| **Figure 9** | 中训配方代理指标与下游表现的拟合关系（附录 C） | 第 6 章 |
| **Figure 10** | IsoFLOP 曲线（缩放梯子实验，附录 D） | 第 5 章 |
| **Table 1** | 按任务域分组的下游性能主表（Stage-1/Stage-2 × Vanilla/NCP，26 项 + 10 项 BPB） | 第 5 章 |
| **Table 2** | 参数量/计算量对齐的模型配置（$P_{\text{blk}}$ / $F_{\text{blk}}$ 记账） | 第 5 章 |
| **Table 3** | 1B 模型层级残差连接消融（IRC / CRC / input-level / Block AttnRes） | 第 4 章 |
| **Table 4** | 三种适配策略的参数量对比（Full / LoRA / VQ） | 第 6 章 |
| **Table 5** | 代码域适配结果 + 通用能力保留 | 第 6 章 |
| **Table 6** | 数学域适配结果 + 通用能力保留 | 第 6 章 |
| **Table 7** | 知识域适配结果 + 通用能力保留 | 第 6 章 |
| **Table 8** | 3B 规模：depth 对齐 OLMo-3-3B 与 NCP-ArchPreview 的 MTP 对照 | 第 6 章 |
| **Table 9** | 块并行 drafter 的平均接受长度（MAL） | 第 6 章 |
| **Table 10** | 完整架构配置表 | 第 7 章 |
| **Table 11** | 中训配方筛查的精确输入（source-balanced NLL + 下游） | 第 6 章 |
| **Table 12** | 缩放梯子实验设置与验证 loss（27 个配置点，五档 FLOPs 预算） | 第 5 章 |
| **Table 13** | 生成式 benchmark 的 prompting / sampling 配置 | 第 7 章 |
| **Table 14** | Stage-1 下游性能随 checkpoint 的演化 | 第 5 章 |

### 1.2 核心贡献

1. **首个万亿 token 尺度的 latent-space 语言模型验证**：在 OLMo-3-7B 之上构造 8.94B 参数的 NCP-ArchPreview（16 层 Token Encoder + 8 层 Concept Module + 16 层 Token Decoder），从预训练起点即启用 NCP，跨 5.73T token 完成验证——相比 ConceptLM 的 1.5B 从零训练与 8B 续训，规模推进了一个量级以上。
2. **乘积量化构造离散概念词表**：每个概念被切为 $S = 32$ 段，每段一个含 $N = 128$ 个码字（维度 128）的 codebook，定义 $N^S = 128^{32}$ 种组合，在不使用巨型单体 codebook 的前提下扩大离散概念空间；概念词表直接从 Token Encoder 的 hidden state 学习，不依赖外部编码器。
3. **可微的概念预测路径**：Concept Module 对每段 codebook 输出分布，并取分布下的**码字加权期望**作为预测概念，既保持完全可微，又把输出约束在已学码本空间内，避免连续回归目标的自由漂移。
4. **层级残差路由（IRC + CRC）**：IRC 允许模块内跨深度组合状态（初始化退化为标准残差），CRC 以目标状态条件化的 softmax 系数 + 对角缩放跨模块传递表示；1B 消融显示相对无层级残差的参照降低 0.0323 loss，额外解析训练 FLOPs 仅 +0.051%。
5. **对齐消融与复用证据链**：参数/计算对齐实验证明增益不来自额外参数或算力（用 size-aligned 基线 **85% 的计算量**逼近其表现），缩放律给出 **1.74× 计算效率**；17M 参数的 VQ 接口支撑 code/math/knowledge 三域适配，概念表示注入 DFlash2 drafter 使平均接受长度提升 **4.17%**。
6. **开源释放**：完全预训练权重集合（Stage-1 每 100k 步 checkpoint、drafter、最终 Stage-1/Stage-2 checkpoint）+ 评测代码。

### 1.3 关键结果速览

| 维度 | 结果 | 来源 |
|:-----|:-----|:-----|
| 收敛效率（Stage-1） | 仅用 **51.3%** 训练 token 达到 OLMo-3-7B 最终 loss，**1.95×** 加速（loss 差 0.091） | Sec 4.2.2 / Figure 1a |
| 收敛效率（Stage-2） | 用 **66.2%** token 匹配 baseline loss，**1.51×**；最终 loss 低 0.027 | Sec 4.2.2 / Figure 1b |
| 下游宏平均（Stage-1，26 项） | 46.59 → **49.04（+2.45 分）** | Table 1 |
| 数学域最大提升（Stage-1） | GSM8K 39.27 → **45.26（+5.99 分）**；全局最大单项为 PiQA +8.60 | Table 1 |
| 域级提升（Stage-1） | MATH AVG +3.75、Code AVG +2.64、MC-Non-STEM AVG +4.63 | Table 1 |
| 下游宏平均（Stage-2） | 56.98 → 57.57（**+0.59 分**），Code AVG 反向 −0.65 | Table 1 |
| 似然指标 | BPB AVG Stage-1 0.824 → 0.811；Stage-2 0.793 → **0.763** | Table 1 |
| 计算等价性 | 以 **85%** 计算量（34/40 $F_{\text{blk}}$）逼近参数对齐的 40 层 vanilla 基线 | Sec 4.3.1 / Figure 3 |
| 缩放律 | 相对 compute-optimal OLMo-3 取得 **1.74×** 计算效率 | Sec 4.5 / Figure 4 |
| 层级残差性价比 | 1B/150B token 上 IRC+CRC 降 loss **0.0323**，仅 +0.051% 解析 FLOPs | Table 3 |
| VQ 域适应 | 17M 可训练参数、**0 新增参数**；code AVG +2.65（唯一四任务全改善）、math AVG +4.27、TriviaQA EM +9.19 | Tables 5–7 |
| VQ 工程收益 | 吞吐 **15,632** tokens/s/GPU（LoRA 10,400 / Full 7,739）；显存 32.4%（LoRA 49.0% / Full 90.3%） | Figure 7 |
| 多 token 预测 | 3B 规模 NCP-ArchPreview 2.3707 vs depth 对齐基线 2.3805（**−0.0098**）且 FLOPs 更少，收益可与 MTP 叠加 | Table 8 |
| 推理加速 | 概念条件化使块并行 drafter 宏平均 MAL 5.933 → **6.180（+4.17%）**，仅 +0.04M 参数 | Table 9 |
| 中训代理指标 | source-balanced NLL 与下游呈强拟合（function generation → HumanEval，**R² = 0.999**） | Appendix C / Figure 9 |

## 第 2 章 研究背景与动机

### 2.1 标准 NTP 的监督粒度局限

现代语言模型在 hidden state 中确实涌现出高层抽象——语义概念与潜在的世界表示。但在标准 Next Token Prediction 之下，这些抽象纯粹是**间接副产品**：监督被严格限制在细粒度的 token 上，没有任何显式目标去指导语义结构如何在跨多 token 的跨度上展开。这意味着模型要"顺带"学会概念，而没有人告诉它"下一个概念应该是什么"。

这一局限在表示学习的历史上有明确对照：视觉合成领域用 latent diffusion 把生成从原始像素转移到紧凑连续表示，显著提升了建模效率与可扩展性；语言领域却仍以表面 token 为唯一监督对象。论文的动机正是把前者的思路移植到语言建模：**在潜在空间里设立预测目标**。

### 2.2 潜在表示预测的技术谱系

论文把已有工作沿两个维度区分：**表示如何形成**，以及**被训练去表示什么**。

**（A）表示如何形成——结构 vs 自适应**：

| 路线 | 代表工作 | 机制 |
|:-----|:---------|:-----|
| 固定层级结构 | Hourglass Transformer、ContextLM、MegaByte | 预定义多尺度分辨率或 byte patch |
| 动态/输入自适应分块 | Byte Latent Transformer (BLT)、DLCM、H-Net | 潜在边界或 patch 大小随输入变化 |

这些工作的共同点是主要改变**计算粒度**或**潜在单元的构造机制**，而非监督目标本身。

**（B）预测什么目标——连续空间 vs 离散概念**：

| 路线 | 代表工作 | 预测对象 |
|:-----|:---------|:---------|
| 连续句级嵌入空间 | Large Concept Model | 共享连续 sentence-embedding |
| 离散概念词表 | ConceptLM | 从学习到的概念词表预测离散概念 |
| 多 token 表面监督 | Multi-Token Prediction (MTP) | 多个未来位置的**表面 token** |

**Joint-Embedding Predictive Architectures（JEPA）** 提供了一条有原则的替代路径：直接在潜在空间预测未来表示，以捕捉不变的语义结构。该范式已在图像表示学习（I-JEPA）与大规模视频建模（V-JEPA、V-JEPA 2）上取得突破。但在语言建模中，**显式以潜在目标为学习对象仍然欠缺探索**。MTP 虽然监督多个未来位置，其损失依然系在单个表面 token 上——它扩展了预测的**数量**，没有改变监督的**粒度**。

### 2.3 本文的定位：ConceptLM 的规模化推进

ConceptLM 是本文最直接的来源：它引入了 NCP 作为离散概念级目标，与语言模型联合学习乘积量化概念词表，并以预测概念条件化 token 生成。其规模验证停留在**从零训练的 1.5B 参数模型**，并以续训方式给一个 8B 模型加上 NCP。

NCP-ArchPreview 的差异化定位有三点：

1. **规模**：在 OLMo-3-7B 之上构造 **8.9B** 模型，跨 **5.73T token** 预训练并延续到中训阶段；
2. **时点**：NCP 从**预训练第一步**就启用，而非事后续训——概念通路全程参与表示塑形；
3. **架构配套**：为支撑跨粒度信息流，配套设计了层级残差路由（IRC/CRC），并用参数/计算对齐的消融把增益来源拆解为"latent 架构"与"NCP 目标"两部分。

论文明确区分了自己的机制性主张：**NCP 是显式的、张成多 token 的概念级目标**，它同时保留了标准的 token 级自回归生成能力——这不是用一个新范式替换 NTP，而是在 NTP 之外增加一条语义粒度的监督通路。

### 2.4 三个需要被验证的问题

论文的实验设计实际上是围绕三个可证伪的问题组织的：

1. **它能否规模化？** —— 概念词表、概念预测与层级残差在万亿 token 与十亿参数级是否仍有效（→ 第 5 章主结果与缩放律）。
2. **增益是不是"买"来的？** —— 是否只是多了参数或算力（→ 第 5 章参数/计算对齐消融）、是否只是残差连接的功劳（→ 第 4 章层级残差消融）。
3. **概念空间有没有复用价值？** —— 预训练结束后它能否支撑低成本域适应与推理加速（→ 第 6 章 VQ 适配、MTP 对照与投机解码）。

## 第 3 章 模型架构：Token–Concept–Token 三层潜在空间设计

### 3.1 总体结构

NCP-ArchPreview 的核心主张是：语言模型内部自发涌现的语义抽象（concept）不应只是 NTP 的副产品，而应成为显式的、可被监督的预测目标。为此，模型在标准自回归骨干中插入一条概念通路，把骨干拆成三个模块：

| 模块 | 层数 | 角色 | 序列粒度 |
|:-----|:----:|:-----|:---------|
| Token Encoder | 16 层 causal Transformer | 产出 token 级 hidden state $h_{1:T}$ | token |
| Concept Module | 8 层 causal Transformer | 在概念空间自回归预测下一个 concept | concept（$M = \lfloor T/k \rfloor$） |
| Token Decoder | 16 层 causal Transformer | 融合概念信号后预测下一个 token | token |

模型总参数 8.94B，保留了 OLMo-3-7B backbone 的主配置：hidden size 4096、FFN size 11,008、32 个 attention head、head 维度 128、词表 100,278、最大训练长度 8,192 token、RoPE base 500,000、SwiGLU + RMSNorm（$\epsilon = 10^{-6}$）。token 级局部注意力窗口为 4,096 token，每第 4 层使用 full attention；attn/hidden dropout 均为 0；参数精度 BF16。完整的架构配置见论文 Table 10。

关键设计约束是**序列压缩**：概念通路以 $k = 4$ 个 token 状态为一段做均值池化，因此 Concept Module 的序列长度只有 token 序列的约四分之一。这一点同时带来两个后果——概念模块的每层参数量与标准 Transformer 块相当（$P_{\text{blk}}$），但每层的分析计算量不到标准块的四分之一（$< 0.25\,F_{\text{blk}}$）。后文的参数/计算对齐消融正是建立在这个不对称性之上。

### 3.2 用向量量化构造离散概念词表

概念表示的第一步是压缩。设概念压缩因子为 $k$，$M = \lfloor T/k \rfloor$，Token Encoder 产出 $h_{1:T}$ 后，第 $m$ 个概念由连续 $k$ 个 token 状态均值池化得到：

$$c_m = f_c\!\left(h_{(m-1)k+1 : mk}\right) = \frac{1}{k}\sum_{i=1}^{k} h_{(m-1)k+i}, \qquad c_m \in \mathbb{R}^{d}$$

由此得到的连续概念序列 $c_{1:M}$ 比 token 序列短 $k$ 倍。

若直接把 $c_m$ 当作回归目标，模型可以输出表示空间中任意漂移的向量——损失函数只约束"距离目标多远"，并不定义"什么样的输出才算合法概念"。为解决这一病态性，论文用向量量化（VQ）构造有限的概念空间，作为概念预测的结构化目标空间。为了在不使用巨型单体 codebook 的前提下扩大离散空间容量，采用**乘积量化**（product quantization）：每个概念向量被切成 $S$ 段特征，

$$c_m = \mathrm{concat}\left(c_m^1, \dots, c_m^S\right), \qquad c_m^s \in \mathbb{R}^{d/S}$$

第 $s$ 段拥有自己的 codebook $E^s = \{e^s_1, \dots, e^s_N\}$，分配为该段最近的码字：

$$n_m^s = \arg\min_{n \in \{1,\dots,N\}} \left\| c_m^s - e_n^s \right\|_2^2, \qquad d_m^s = e_{n_m^s}^{s}$$

$$d_m = \mathrm{concat}\left(d_m^1, \dots, d_m^S\right)$$

每个分段 codebook 只有 $N$ 个条目，但乘积量化定义了 $N^S$ 种码字组合，因此在保持 codebook 体积可控的同时大幅提升离散概念空间容量。NCP-ArchPreview 的实例配置为 $S = 32$ 段、每段 $N = 128$ 个码字、码字维度 128，组合容量达到 $128^{32}$。概念词表本身从 Token Encoder 的 hidden state 出发学习，不依赖任何外部编码器。

### 3.3 概念预测：在学到的词表上做可微期望

Concept Module 在概念空间中自回归地预测下一个概念。给定历史 $c_{<m}$，堆叠的 Transformer 层产出 latent hidden state：

$$u_m = \mathrm{ConceptModule}_{\theta_c}\!\left(c_{<m}\right)$$

对每个乘积量化分段 $s$，一个分段专属的 prediction head 在 codebook 条目上给出分布：

$$\pi_m^s = \mathrm{softmax}\!\left(\mathrm{PredictionHead}^s(u_m)\right), \qquad \pi_m^s \in \mathbb{R}^{N}$$

关键细节在于**不取 argmax、不采样**：预测的分段是分布下的期望，

$$\hat{c}_m^s = \sum_{n=1}^{N} \pi_{m,n}^s \, e_n^s, \qquad \hat{c}_m = \mathrm{concat}\left(\hat{c}_m^1, \dots, \hat{c}_m^S\right)$$

这一"码字加权组合"设计让预测路径完全可微，同时把输出限制在已学 codebook 的线性组合之内。于是 NCP 目标是在一个受约束的潜在空间中做监督，而不是回归到无约束、自由漂移的连续目标——这正是 3.2 节所述病态性的直接解法。推理时，先前预测出的 concept 被自回归地反馈回概念通路。

### 3.4 把预测概念注入 token 流：重复、因果位移与残差

概念级的信号必须与 Token Decoder 消费的 token 级状态对齐。做法是先把每个预测 concept 重复 $k$ 次，再做因果位移。实现取 $\Delta = k$，对 token 级预测位置 $t = 1, \dots, T-1$：

$$b_t = \begin{cases} 0, & 1 \le t < \Delta \\ \hat{c}_{\lfloor (t-\Delta)/k \rfloor + 2}, & \Delta \le t < T \end{cases}$$

位移的目的是**防止信息泄漏**：每个预测 concept 只在"用于预测它的 token 状态"已被处理之后才被注入（前缀 $b_t = 0$ 就是零前缀）。

概念与 token 的 hidden state 共享同一维度 $d$，因此直接做逐元素残差相加完成融合：

$$\tilde{h}_t = h_t + b_t$$

Token Decoder 随后从融合状态自回归预测下一个 token：

$$p(x_{t+1} \mid x_{\le t}) = P_{\theta_d}\!\left(\tilde{h}_{\le t}\right)$$

### 3.5 层级残差连接：模块内与跨模块

三个模块工作在不同的深度与序列粒度上，普通残差只在单模块内做固定加法，缺乏跨粒度的信息通路。论文受 MUDDFormer 的动态稠密连接启发，设计了两类**层级残差**，并保持每个模块只有单一 hidden 流。

**模块内残差（IRC, Intra-Module Residual Connections）**：设模块 $s \in \{\text{TokenEncoder}, \text{ConceptModule}, \text{TokenDecoder}\}$，第 $\ell$ 层 Transformer 块的输入为 $H^s_\ell$，块输出（残差加法之前）为 $R^s_\ell = F^s_\ell(H^s_\ell)$。标准 Transformer 固定使用 $H^s_{\ell+1} = H^s_\ell + R^s_\ell$；IRC 把它推广为多深度组合。候选集为

$$X^s_\ell = \left\{ H^s_1,\; H^s_1 + R^s_1,\; \dots,\; H^s_\ell + R^s_\ell \right\}$$

一个轻量 MLP 作用于当前块输出 $R^s_\ell$，为每个候选状态产生一个系数：

$$w^s_\ell = \mathrm{MLP}^s_\ell\!\left(R^s_\ell\right), \qquad H^s_{\ell+1} = \sum_{j=1}^{\ell+1} w^s_{\ell,j} X^s_{\ell,j}$$

系数**不做归一化**，因此 IRC 可以表达不同深度状态的有符号组合。初始化时令 $w^s_\ell = [0, \dots, 0, 1]$，即 IRC 初始选择最近的残差更新状态、退化为标准残差连接，训练过程中再学习是否引入更早层的信息。

**跨模块残差（CRC, Cross-Module Residual Connections）**：设源模块 $u$ 导出表示 $S^u = \{S^u_j\}_{j=1}^{K}$，目标模块 $s$ 的状态为 $T^s_\ell$。当两个模块序列粒度不同时，先通过分块或重复对齐源表示。一个轻量 MLP 作用于**目标状态**，产出源各深度上的归一化系数：

$$\alpha^{s \leftarrow u}_\ell = \mathrm{softmax}\!\left(\mathrm{MLP}^{s \leftarrow u}_\ell\!\left(T^s_\ell\right)\right), \qquad M^{s \leftarrow u}_\ell = \sum_{j=1}^{K} \alpha^{s \leftarrow u}_{\ell,j} \, \mathrm{LN}\!\left(S^u_j\right)$$

$$T^s_{\ell,\text{out}} = T^s_\ell + D^{s \leftarrow u}_\ell \odot M^{s \leftarrow u}_\ell, \qquad D^{s \leftarrow u}_\ell \in \mathbb{R}^{d}$$

注意 CRC 的系数条件是**目标状态**（target-conditioned depth selection），与 IRC 条件于当前块输出不同。论文使用三条跨模块连接：

1. Token Encoder → Concept Module
2. Token Encoder → Token Decoder
3. Concept Module → Token Decoder

其中第三条遵循与预测概念通路相同的因果位移，因此不会暴露未来 token 信息。对角缩放 $D$ 以小值初始化，模型起始状态接近原始 backbone，随后逐步学会利用新增通路。IRC 与 CRC 合起来构成完整的层级残差机制。

### 3.6 架构设计的三点直觉

1. **概念是"目标"而不是"中间变量"**：概念通路有自己独立的预测损失（见第 4 章），Token Encoder 必须保留对未来概念预测有用的信息，这使 token 表示被一个更粗粒度的语义目标反向塑形。
2. **压缩即算力红利**：概念通路每 4 个 token 只跑一次，概念模块 8 层中每层计算量低于标准块的四分之一，因此额外参数（8 层 × $P_{\text{blk}}$）换来的计算开销远小于参数比例——这是 4.3 节"85% 计算量逼近参数对齐基线"的结构性原因。
3. **残差通路的初始化即恒等**：IRC 初始化退化为标准残差、CRC 对角缩放小值初始化，保证新增机制不会在训练早期破坏 backbone 的已有能力。

## 第 4 章 训练目标、层级残差与优化稳定性

### 4.1 端到端联合训练

概念通路不是外挂的辅助头，而是与 token 通路在同一前向图中端到端优化。对输入序列 $x_{1:T}$，一次前向依次为：Token Encoder 产出 token 状态 $h_{1:T}$ → 池化算子（式 2）产出连续概念状态 $c_{1:M}$ → VQ 模块把每个概念映射到结构化 codebook 表示 → Concept Module 从概念历史预测下一概念 → 因果位移后的预测被注入 Token Decoder。codebook、Token Encoder、Concept Module、Token Decoder 由 NTP 与 NCP 两个目标共同更新。

### 4.2 三个损失函数与梯度流向

**VQ 损失（codebook 学习）**。乘积量化的每个分段 codebook 被训练去拟合连续概念表示的分布：

$$L_{\mathrm{VQ}} = \frac{1}{MS}\sum_{m=1}^{M}\sum_{s=1}^{S}\left\| \mathrm{sg}\!\left(c_m^s\right) - d_m^s \right\|_2^2$$

其中 $\mathrm{sg}(\cdot)$ 表示 stop-gradient，$d_m^s$ 是分配给 $c_m^s$ 的最近码字。stop-gradient 阻止 VQ 目标更新 Token Encoder：该目标只把被选中的码字推向连续概念表示，让 codebook 跟踪潜在分布，而不直接改动 token 级 hidden state。

**NCP 损失（概念预测）**。Concept Module 被训练去预测下一个连续概念，采用均方误差：

$$L_{\mathrm{NCP}} = \frac{1}{M-1}\sum_{m=2}^{M}\left\| \hat{c}_m - \mathrm{sg}\!\left(c_m\right) \right\|_2^2$$

目标 $c_m$ 被 detach。与 VQ 不同，$L_{\mathrm{NCP}}$ **同时更新 Concept Module 与 Token Encoder**——但 Token Encoder 只通过喂给 Concept Module 的前序概念表示 $c_{<m}$ 间接获得 NCP 梯度。这个梯度路径是刻意的：它鼓励 Token Encoder 保留"对预测下一个概念有用"的信息，也就是把 token 表示向概念可预测的方向塑形，而不是让 token 表示自由演化。

**NTP 损失（token 预测）**。融合表示 $\tilde{h}_t$（式 12）走标准的因果下一 token 目标：

$$L_{\mathrm{NTP}} = -\frac{1}{T-1}\sum_{t=1}^{T-1}\log p_{\theta_d}\!\left(x_{t+1} \mid \tilde{h}_{\le t}\right)$$

零前缀与位移（式 11）保证 NTP 损失永远不会条件于"由它要监督的那些 token 计算出来的概念"。NTP 为模型中每个权重提供稠密监督，同时更新 token 通路与概念通路，并保持 backbone 的标准自回归语言建模行为不被破坏。

**联合目标**。三者加权求和：

$$L_{\text{total}} = L_{\mathrm{NTP}} + \alpha L_{\mathrm{NCP}} + \beta L_{\mathrm{VQ}}$$

其中系数 $\alpha$、$\beta$ 控制两个辅助目标的权重，三个目标在训练中联合优化。

| 目标 | 监督对象 | 更新范围 | 是否 detach 目标 |
|:-----|:---------|:---------|:----------------|
| $L_{\mathrm{NTP}}$ | 每个位置的下一 token | token + 概念通路全部权重 | — |
| $L_{\mathrm{NCP}}$ | 下一个概念 $\hat{c}_m$ vs $c_m$ | Concept Module + Token Encoder（经 $c_{<m}$ 间接） | ✅ $c_m$ 被 detach |
| $L_{\mathrm{VQ}}$ | 码字 vs 连续概念 $c_m^s$ | 仅 codebook 条目 | ✅ $\mathrm{sg}(c_m^s)$ |

### 4.3 优化器配置与注意力 logit 稳定性

**Muon + AdamW 混合**。矩阵型参数使用 Moonlight Muon（构建在 Muon 之上），其余参数（embedding、bias 等非 Muon 参数）使用 AdamW。矩阵参数的更新写作：

$$W_t = W_{t-1} - \eta_t\left( \lambda_{\mathrm{Muon}} \frac{O_t}{\sqrt{\max(d_{\text{in}}, d_{\text{out}})}} + \lambda_{wd} W_{t-1} \right)$$

其中 $O_t$ 是正交化后的 Muon 更新，$\eta_t$ 是学习率计划，$\lambda_{\mathrm{Muon}}$ 控制更新尺度，$\lambda_{wd}$ 为 weight decay。默认学习率为 $6 \times 10^{-5}$，并采用与 OLMo-3-7B 相同的 cosine 计划。

**观察到的数值不稳定**。沿用 OLMo-3 的 **layer-wise Q/K normalization** 训练 NCP-ArchPreview 时，论文观察到（Figure 5）：

- attention logit 持续增长（最大绝对值与最大正值同时上升）；
- Q/K head block 的矩阵范数出现显著不平衡，少数 outlier head 主导 pre-softmax 分数，使注意力分布高度集中；
- 放大的 Q/K 状态伴随不均匀的梯度流、异常的 Q/K 梯度范数与周期性的全局梯度范数尖峰。

**消融与归因**（Figure 6）。论文比较了匹配设置的 AdamW 与 Muon baseline，并评估把 layer-wise Q/K normalization 换成 **per-head Q/K normalization** 的 Muon 变体（其余训练设置不变）。结论是：

1. Muon baseline 本身就能复现该不稳定，而 NCP-ArchPreview 在稳定性上表现更好；
2. 主因是 **full-matrix Muon 更新与 layer-wise Q/K normalization 的交互**——full-matrix Muon 会耦合多个 attention head 的更新，而 layer-wise normalization 只约束它们的**聚合尺度**，不独立控制每个 head 的尺度或 QK 对齐，于是少数 head 可以逐步主导并产出过大的点积；
3. **per-head Q/K normalization** 能一致地同时抑制 Q/K 范数离群与 attention logit 增长。

值得强调的是工程取舍：论文的主实验**仍然使用 OLMo-3-7B 的 layer-wise Q/K normalization**，以保持与原始 baseline 的受控对比；per-head 变体只作为稳定性干预在消融中报告，不用于产主结果。

### 4.4 层级残差的消融证据

层级残差是本章最"可量化"的设计选择。论文在 **1B 规模**的 NCP-ArchPreview 模型上、训练 150B token 后比较多种连接变体，报告最终 200 个记录步的 loss 均值，并以"无层级残差"版本作为 loss 与计算量的共同参照（Table 3）：

| 变体 | Avg. LM loss | Δ loss vs. 参照 | Δ FLOPs vs. 参照 |
|:-----|:---:|:---:|:---:|
| 无层级残差（参照） | 2.2588 | 0.0000 | 0.000% |
| **IRC + CRC（完整设计）** | **2.2265** | **−0.0323** | **+0.051%** |
| IRC only | 2.2315 | −0.0273 | +0.024% |
| IRC + Input-Level Cross-Module Connections | 2.2292 | −0.0296 | +0.024% |
| IRC + All-Stage Softmax | 2.2295 | −0.0294 | +0.024% |
| Block AttnRes ($S = 4$) + Cross-Module Connections | 2.2409 | −0.0180 | +0.026% |

从表中可以读出三条结论：

1. **IRC + CRC 收益最大**：相对无层级残差版本降低 0.0323 loss，额外解析训练 FLOPs 仅 +0.051%——这是极高的性价比；
2. **IRC 单独就贡献了绝大部分收益**（−0.0273，仅 +0.024% FLOPs）；在此之上补三条 input-level 跨模块连接再降 0.0023（−0.0296），且不增加被计入的张量收缩；
3. **Block AttnRes 变体弱于 IRC 系列**：在可比的 +0.026% FLOPs 下只降低 0.0180。论文同时提醒，这些解析 FLOPs **不包含** source-state materialization、内存流量、reduction 操作与小 kernel 启动开销，因此完整的层级连接设计在运行时与显存上的额外成本可能高于解析 FLOPs 所反映的水平——这是评价该机制真实代价时必须注意的口径限制。

## 第 5 章 预训练主结果与对齐消融

### 5.1 实验设置

- **backbone 与数据**：以 OLMo-3-7B 为 backbone，遵循 OLMo-3 的分阶段数据课程。Stage-1 使用 **Dolma 3 Mix**，Stage-2 在 **Dolma 3 Dolmino** 上继续训练；Stage-1 累计 **5.73T token**，Stage-2 中训 **100B token**。
- **评测协议**：沿用 OLMo-Core 中的 OLMo 评测协议，覆盖 30 个 benchmark family，包括 MMLU、GSM8K、MATH-500、HumanEval、MBPP、ARC、HellaSwag 等，考察事实知识、数学推理、代码生成、常识推理、阅读理解与语言建模能力。主表（Table 1）把 26 个 higher-is-better 指标与 10 个 lower-is-better 似然（BPB）指标分开，BPB 均值单独计算且不计入 Overall AVG；Overall AVG 是 26 个组成指标的**无权平均**，而非各分段 AVG 的平均。
- **NCP-ArchPreview 配置**：hidden 4096 / FFN 11,008 / 32 heads；32 个 token 级 Transformer 层均分为 16 层 Token Encoder 与 16 层 Token Decoder；Concept Module 8 层、chunk size 4；VQ 32 个 codebook × 128 条目 × 维度 128；概念表示通过归一化残差在**每个** decoder 层注入；总参数约 8.94B；最大上下文 8,192 token。
- **域适应数据**（用于 5.4 节）：从 Stage-1 checkpoint 出发分别在 Magicoder（code）、Orca-Math（math）、TriviaQA-RC（knowledge）上继续训练。

### 5.2 训练动力学：loss 优势与收敛加速

同数据对比下，NCP-ArchPreview 在 Stage-1 与 Stage-2 **全程**保持比 OLMo-3-7B 更低的 token 级语言建模 loss（Figure 1）：

| 阶段 | loss 差距 | 达到 baseline 最终 loss 所需 token | 收敛加速 |
|:-----|:---:|:---:|:---:|
| Stage-1（5.73T token） | 结束时间隙 0.091 | **51.3%** 的训练 token | **1.95×** |
| Stage-2（中训） | 最终 loss 低 0.027 | **66.2%** 的训练 token | **1.51×** |

Stage-1 的 loss 差距随训练推进**扩大**，说明优势不是早期噪声而是持续累积的优化效率差异；Stage-2 的优势仍然存在但幅度收窄（1.51× vs 1.95×），且曲线波动更小。

### 5.3 下游性能（Table 1）

Stage-1 结束时，NCP-ArchPreview 在几乎全部评测指标上超过 OLMo-3-7B，Overall AVG 高 **2.45 分**：

| 域 | Stage-1 Vanilla | Stage-1 NCP | Δ | Stage-2 Vanilla | Stage-2 NCP | Δ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| MMLU AVG | 54.50 | 56.73 | **+2.24** | 58.32 | 59.94 | +1.62 |
| MATH AVG | 20.79 | 24.54 | **+3.75** | 55.63 | 57.39 | +1.76 |
| Code AVG | 25.15 | 27.79 | **+2.64** | 39.42 | 38.77 | −0.65 |
| MC-STEM AVG | 84.47 | 86.93 | +2.47 | 89.65 | 88.67 | −0.98 |
| MC-Non-STEM AVG | 70.08 | 74.71 | **+4.63** | 76.85 | 77.76 | +0.91 |
| GenQA AVG | 54.29 | 54.76 | +0.47 | 53.49 | 54.04 | +0.55 |
| **Overall AVG**（26 项） | **46.59** | **49.04** | **+2.45** | **56.98** | **57.57** | **+0.59** |
| Likelihood AVG（BPB ↓，10 项） | 0.824 | 0.811 | −0.014 | 0.793 | 0.763 | −0.030 |

逐项细节中，Stage-1 提升最大的单项包括：

- **GSM8K 39.27 → 45.26（+5.99）**，是**数学域**最大的单项提升；全部 26 项中绝对提升最大的两项是 PiQA 72.25 → 80.85（**+8.60**）与 MultiPL-E MBPP 28.67 → 33.96（+5.29），GSM8K 的 +5.99 位列第三；
- PiQA 72.25 → 80.85（+8.60）、MultiPL-E MBPP 28.67 → 33.96（+5.29）、HumanEval 27.10 → 31.38（+4.28）、GSM-Symbolic 18.85 → 22.80（+3.95）、SocialIQA 65.35 → 69.24（+3.89）、ARC-C 77.99 → 81.57（+3.58）、MMLU-Humanities 64.73 → 68.28（+3.55）；
- MATH 域的相对提升达到 **18%**（论文正文表述），MATH AVG 绝对提升 +3.75 分；
- 似然指标上，Stage-1 的 HumanEval Gold BPB 由 0.384 降至 0.364（−0.020）、DROP 由 4.474 降至 4.397（−0.077）、Natural Questions 0.917 → 0.884（−0.033）。

Stage-2 的情况更复杂：Overall AVG 只高 0.59 分，MATH（+1.76）、MC-Non-STEM（+0.91）、GenQA（+0.55）改善，但 **Code AVG 反而下降 0.65 分**（HumanEval −3.69、BigCodeBench −1.16、MultiPL-E HumanEval −1.02、MultiPL-E MBPP −0.79）。论文给出的解释是：Stage-2 数据中 code 占比只有约 **10%**，对聚合混合分布的更好拟合可以在提升总体均值的同时削弱 underrepresented 域（如 code）的表现。

总体 AVG 之外，似然块在 Stage-2 的改善反而更大（BPB AVG 0.793 → 0.763，−0.030），但其中 MT-MBPP Gold（+0.025）、CoQA（+0.040）、GSM8K Gold（+0.024）、SQuAD（+0.015）等子项出现反向或分化，说明 Stage-2 的 loss/likelihood 改善与具体任务的格式与评分统计量并不一致。

### 5.4 参数对齐与计算对齐消融

要回答"提升是否只是因为多了参数或算力"，论文构造了三个 OLMo-3 基线（Table 2）。记一个标准 OLMo-3-7B Transformer 块的参数量为 $P_{\text{blk}}$、计算量为 $F_{\text{blk}}$：

| 模型 | Token Encoder | Token Decoder | Concept Module | 参数 | 计算 |
|:-----|:---:|:---:|:---:|:---:|:---:|
| **NCP-ArchPreview** | $16P_{\text{blk}}$ | $16P_{\text{blk}}$ | $8P_{\text{blk}}$ | $40P_{\text{blk}}$ | $34F_{\text{blk}}$ |
| Vanilla | $32P_{\text{blk}}$ | – | – | $32P_{\text{blk}}$ | $32F_{\text{blk}}$ |
| Vanilla size-aligned | $40P_{\text{blk}}$ | – | – | $40P_{\text{blk}}$ | $40F_{\text{blk}}$ |
| Vanilla computation-aligned | $34P_{\text{blk}}$ | – | – | $34P_{\text{blk}}$ | $34F_{\text{blk}}$ |

这一对照组的构造逻辑正来自第 3 章的压缩结构：Concept Module 每 4 个 token 状态合成一个概念，序列长度降为约四分之一，因此每个 Concept Module 块参数量约等于一个标准 OLMo-3-7B 块（$P_{\text{blk}}$），而分析训练 FLOPs 不到其四分之一。VQ codebook 与残差组件的参数与计算占比可忽略，构造对齐基线时被省略。

结论（Figure 3，前 200B token）：

1. NCP-ArchPreview **显著优于 Vanilla 与 Vanilla computation-aligned**，因此提升不能仅用"额外计算量"解释；
2. NCP-ArchPreview **逼近 Vanilla size-aligned 的表现，却只用了其 34/40 = 85% 的计算量**。

模块消融则把增益拆解到组件级别。从标准 OLMo-3-7B 出发逐步引入：Vanilla → Vanilla + Concept Module（CM）→ Vanilla + CM + Residual → Vanilla + CM + Residual + NCP（即 NCP-ArchPreview）。三者**逐级改善**训练 loss，且完整模型接近 Vanilla size-aligned 基线。这组实验同时支撑了摘要中的"增益同时来自 latent 架构与 NCP 目标"这一表述。

### 5.5 缩放律：1.74× 计算效率

论文在多个 FLOPs 预算上做缩放梯子实验（Figure 4、Figure 10 与 Table 12）：每个预算下搜索训练超参以及模型规模/训练数据的分配，报告该预算下找到的最佳验证 loss。结论是 NCP-ArchPreview 相对 compute-optimal 训练的 OLMo-3 取得 **1.74× 的计算效率**。

方法学上有一处必须注意的口径差异：常见的 $C \approx 6 N_{\text{param}} D$ 近似对"层激活不均匀"的架构并不准确——NCP-ArchPreview 的 Concept Module 每个 chunk 只运行一次，因此论文改用

$$C = F_{\text{tok}} \cdot D$$

其中 $F_{\text{tok}}$ 是每 token 的解析训练 FLOPs、$D$ 是训练 token 数，且 FLOPs 估计**排除 embedding 参数**。超参搜索策略遵循已有观察（固定 FLOPs 预算下不同模型/数据分配的最优超参相近）：先在每个预算上选一个代表模型搜索学习率与 batch size，再在同一预算的其他模型上于其邻域内搜索最佳学习率。

Table 12 给出的缩放梯子共 27 个配置点（1e19: 5 点、3e19: 5 点、5e19: 6 点、8e19: 6 点、1e20: 5 点），覆盖 $1\times10^{19}$ 到 $1\times10^{20}$ FLOPs 五档预算，隐藏宽度 $H$ 从 896 到 2176、encoder/Concept Module/decoder 深度组合从 4/4/4 到 8/8/8，$F_{\text{tok}}$ 从 0.824 到 7.641 GF/token、训练 token 数 $D$ 从 41.375B 到 3.502B。几个代表性点：在 $1\times10^{19}$ 预算下最佳验证 loss 为 2.759（$H = 1408$、4/4/4、1.771 GF/token、5.645B token）；在 $1\times10^{20}$ 预算下为 **2.449**（$H = 1792$、8/8/8、5.401 GF/token、18.516B token）。值得注意的规律是：同一预算内把层级深度从 4/4/4 改为 8/8/8 通常带来明显收益（例如 $1\times10^{19}$ 第 5 点 8/4/8 反而 2.762），说明**深度分配**本身是缩放过程中的一个重要旋钮。

### 5.6 loss 与下游能力的关系

论文专门检验"训练 loss 是否准确反映模型能力"，得到的是一个**分阶段不同**的结论：

- **Stage-1 上吻合**：随训练推进 loss 逐步下降，下游分数持续上升，说明能力在整个预训练过程中稳步提升；
- **Stage-2 上背离**：三个中训配置 V1–V3 的最终训练 loss 逐级更低，但**下游表现逐级更差**。

论文把这一现象归因于 Stage-2 训练数据分布与下游任务数据分布之间的错配，并在附录 C 用代理数据集进一步研究（见 6.4 节）。这个反直觉结果对实践的含义是明确的：**在中训阶段用 loss 选数据配方是不可靠的**，需要任务原生的代理指标。

## 第 6 章 附加结果：域适应、多 token 预测与投机解码加速

本章的四个实验回答的是同一个问题：**学到的概念空间除了降低预训练 loss，还能不能复用？**

### 6.1 VQ 训练作为轻量域适应接口

概念词表在预训练结束后仍保留价值。论文比较三种适配策略（Table 4）：

| 策略 | 可训练参数 | 新增参数 | 训练对象 |
|:-----|:---:|:---:|:-----|
| +Full | 8.9B | 0 | 全部权重 |
| +LoRA | 17M | 17M | 低秩适配器（参数匹配对照） |
| **+VQ** | **17M** | **0** | **冻结 token 级 backbone，只更新 VQ codebook 与概念预测头** |

关键差异在于 +VQ **不新增任何参数**（更新的是已存在的 codebook 与预测头），而 LoRA 需要向模型注入 17M 新参数。所有适配实验都从对应的 Stage-1 checkpoint 出发（OLMo-3-7B 与 NCP-ArchPreview 各自）。

**代码域（Table 5）**。NCP-ArchPreview 的 VQ 适配把 code 平均分从 30.04 提升到 **32.69（+2.65）**，是三种适配中**唯一在四个代码任务上全部改善**的设置：

| 模型 / 策略 | HumanEval | HumanEval+ | MBPP | MBPP+ | Code AVG | 通用 AVG |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| NCP-ArchPreview（起点） | 31.27 | 23.17 | 36.09 | 29.63 | 30.04 | 68.68 |
| +Full | 43.27 (+12.00) | 23.78 (+0.61) | 39.53 (+3.44) | 7.14 (−22.49) | 28.43 (−1.61) | 68.20 (−0.48) |
| +LoRA | 36.49 (+5.22) | 28.05 (+4.88) | 39.02 (+2.93) | 21.96 (−7.67) | 31.38 (+1.34) | 68.57 (−0.11) |
| **+VQ** | **33.94 (+2.67)** | **28.66 (+5.49)** | 36.94 (+0.85) | **31.22 (+1.59)** | **32.69 (+2.65)** | 68.26 (−0.42) |

对比之下，Full 训练虽然大幅提升 HumanEval（+12.00），但 MBPP+ **塌陷 −22.49 分**，聚合反而低于起点（−1.61）；LoRA 也在 MBPP+ 上损失 7.67 分。这类适配后回退与域适应/持续学习中的干扰现象一致。

**数学域（Table 6）**。VQ 把 math 平均分从 30.56 提升到 **34.83（+4.27）**：GSM8K 46.78 → 52.69（+5.91）、MATH-500 14.34 → 16.96（+2.62）。

| 策略 | GSM8K | MATH-500 | Math AVG | 通用 AVG 变化 |
|:---|:---:|:---:|:---:|:---:|
| +Full | 61.03 (+14.25) | 12.40 (−1.94) | 36.72 (+6.16) | −0.97 |
| +LoRA | 53.07 (+6.29) | 14.34 (±0.00) | 33.71 (+3.15) | −0.42 |
| **+VQ** | 52.69 (+5.91) | **16.96 (+2.62)** | **34.83 (+4.27)** | **+0.39** |

这里出现了一个清晰的权衡结构：VQ 的 math 增益（+4.27）低于 Full（+6.16），但**遗忘更小**——通用平均分变化 +0.39（VQ）对比 −0.42（LoRA）、−0.97（Full）。也就是说，VQ 用 1.89 分的目标域性能换取 1.36 分更高的通用能力。VQ 在 math 与通用两个平均分上**同时**优于参数量匹配的 LoRA。

**知识域（Table 7）**。VQ 把 TriviaQA exact match 从 40.28 提升到 **49.47（+9.19）**，同时通用平均分几乎不变（68.68 → 68.71，+0.03）。但目标域增益小于 Full（+18.00）与 LoRA（+17.48）。论文的解释是机制性的：Transformer 的 FFN 层被视为 key-value memory，事实召回主要由 backbone MLP 承载；**VQ 训练冻结了这些 MLP 参数**，只调整概念词表，因此无法直接改写 backbone 中编码的事实关联。

跨模型看，NCP-ArchPreview 在知识域获得了比 OLMo-3-7B 更大的 TriviaQA 增益（Full：+18.00 vs +8.53；LoRA：+17.48 vs +8.06），且遗忘更少——OLMo-3-7B 的 Full/LoRA 适配使通用平均分分别下降 1.04 与 1.80 分，其中 SciQ MC 单项损失 4.90 与 9.60 分。

**吞吐与显存（Figure 7）**。在 8 卡、micro-batch size = 1 的设置下：

| 策略 | 吞吐（tokens/s/GPU） | 相对 Full | 显存占用（MBS=1, 8 GPU） | 显存占用（MBS=2, 单卡） |
|:---|:---:|:---:|:---:|:---:|
| +Full | 7,739 | 1.00× | 90.3% | OOM（>100%） |
| +LoRA | 10,400 | 1.34× | 49.0% | 87.9% |
| **+VQ** | **15,632** | **2.02×** | **32.4%** | **51.6%** |

VQ 相对 LoRA 吞吐高 1.50×、相对 Full 高 2.02×。值得注意的是 VQ 与 LoRA **都只优化 17M 参数**，但 VQ 吞吐高 50%——差异来自 LoRA 需要经过适配器路径的回传与额外激活，而 VQ 只更新 codebook 与预测头。显存方面 VQ 占用最低（32.4%，相对 Full −64%、相对 LoRA −34%），单卡 MBS=2 时 VQ 51.6%、LoRA 87.9%、Full 直接 OOM。

综合起来，+VQ 的工程定位很清楚：**参数量效率接近 LoRA，但吞吐、显存与抗遗忘都更好；代价是目标域增益不如全量训练**。

### 6.2 与多 token 预测（MTP）的关系

5.2 节观察到 NCP 的概念表示含有对多 token 预测有用的信息。论文在 **3B 规模**上做了受控对比：3B-scale NCP-ArchPreview 组织为 8 层 Token Encoder + 4 层 Concept Module + 7 层 Token Decoder；为公平比较，把 OLMo-3-3B 的 16 层也按 8/1/7 分组并在对应深度加上**相同的层级残差连接**，记作 OLMo-3-3B + Residual。四个变体（两种架构 × 是否加 MTP 层）都在预训练语料的前 150B token 上训练。MTP 层遵循 DeepSeek-V3 的做法：主模型保留 NTP 目标，新增层预测"后两个位置"的 token，损失权重 0.3。

| 模型 | 平均 LM loss（前 150B token） | Δ vs. OLMo-3 + Residual | 训练 FLOPs | Δ FLOPs |
|:---|:---:|:---:|:---:|:---:|
| OLMo-3-3B + Residual | 2.3805 | 0.0000 | $3.245\times10^{21}$ | 0.00% |
| OLMo-3-3B + Residual + MTP | 2.3739 | −0.0065 | $3.750\times10^{21}$ | +15.55% |
| **NCP-ArchPreview** | **2.3707** | **−0.0098** | $3.227\times10^{21}$ | **−0.57%** |
| **NCP-ArchPreview + MTP** | **2.3664** | **−0.0141** | $3.731\times10^{21}$ | +14.98% |

三个结论：

1. NCP-ArchPreview 相对 depth 对齐的 OLMo-3-3B + Residual 改善 **0.0098**，**大于**给 OLMo-3-3B + Residual 加 MTP 带来的 0.0065 改善——且完成于**更少**的 FLOPs（−0.57%）；
2. NCP-ArchPreview + MTP 比 OLMo-3-3B + Residual + MTP 低 **0.0075**，训练 FLOPs 还略少（3.731 vs 3.750 ×$10^{21}$）；
3. 两者的收益**可叠加**，说明概念级监督与"多预测几个 token"的辅助监督捕捉的不是同一信息。

训练早期存在一次交叉：OLMo-3-3B + Residual + MTP 起初 loss 更低，NCP-ArchPreview 在大约第 2,285 次更新附近穿过它；在最后 8k 次更新中，曲线稳定保持 NCP+MTP < NCP < OLMo-3-3B+Residual+MTP < OLMo-3-3B+Residual 的顺序。

### 6.3 概念表示加速块并行投机解码

最近的工作用块扩散（block-diffusion）式 drafter 一次前向提出整个 token 块以降低起草延迟，但瓶颈在于跨多个未来位置很难维持连贯的预测轨迹。论文检验：Concept Module 产出的状态能否提升块并行 drafter 的**接受长度**。

**注入方式**。对以位置 $t$ 结束的已验证前缀，取最后一个**完整** Target chunk 的概念表示 $c_{\lfloor t/k \rfloor}$；在 drafter 第 $\ell$ 层，把它加入每个提议位置：

$$\tilde{s}^{(\ell)}_{t,j} = s^{(\ell)}_{t,j} + \tanh\!\left(g^{(\ell)}\right) \odot \mathrm{RMSNorm}_\ell\!\left(c_{\lfloor t/k \rfloor}\right), \qquad j = 1, \dots, K$$

门控 $g^{(\ell)}$ 零初始化，drafter 因此可以从零开始逐步引入概念信号。该改动只给 1.1B 参数的 drafter **增加 0.04M 参数**，且不引入额外的 Target 模型计算。

**实验设置**。baseline 组合了 DFlash2 与 DFlare 的架构改进：按 DFlash2 使用 two-tap 动态卷积建模相邻 draft 位置依赖，并用轻量 path selector 在每位置的 top-16 候选中选出连贯序列；按 DFlare 让每个 drafter 层学习五个 Target hidden state 的加权组合，从而在不同语义深度上访问 Target 表示。Baseline 与 Baseline+Concept 使用**相同的**在线 Target 蒸馏流程、数据顺序、随机种子与训练预算（序列长度 8192、每序列 512 个训练锚点、全局 batch 512、固定 5B token 子集重复 10 个 epoch），评测时用精确投机验证、提议视野 16 个 draft token。指标为平均接受长度（MAL）——验证轮中提交的 token 数（含 Target 修正或 bonus token）的平均。

| 基准 | Baseline MAL | Baseline + Concept MAL | 相对提升 |
|:---|:---:|:---:|:---:|
| GSM8K | 6.351 | 6.537 | **+2.93%** |
| MATH | 6.105 | 6.240 | +2.22% |
| HumanEval | 5.432 | 5.845 | **+7.59%** |
| MBPP | 5.844 | 6.099 | +4.37% |
| **宏平均** | **5.933** | **6.180** | **+4.17%** |

四个基准上一致提升，HumanEval 幅度最大（+7.59%）。论文的解释是：chunk 级概念表示以更宽的预测上下文补充了 token 级 Target 特征，从而提升块并行提议与 Target 的一致性。

### 6.4 中训数据配方的代理指标筛查（附录 C）

Stage-2 出现"loss 更低但下游更差"的背离（5.6 节）后，论文构造了一个可复用的代理评测流程来筛查中训数据配方，避免每次都跑完整下游评测套件：

- **构建方式**：从 **177,202** 条专家轨迹（跨 **63** 个来源、**6** 个域）构造扩展 held-out 集，对应约 **100M** teacher-suffix token；所有轨迹由**同一个冻结的 Qwen3.7-Plus teacher** 按统一生成协议产出。prompt、专家后缀、来源标识与能力标注在评测候选 checkpoint 之前就固定，保证每个配方在完全相同的评分视图上比较（无效生成被剔除，另保留一个更严格的 trusted 子集做敏感性分析）。
- **评分口径**：item 级分数先在来源内平均，来源级均值再在能力叶节点内平均，即 source-balanced 评分，避免单一来源主导。
- **对照**：三个匹配预算的 Stage-2 配方（V1/V2/V3），外加 Stage-1 checkpoint 作为跨阶段参照（不参与配方选择）。
- **结果**：代理指标（source-balanced 轨迹 NLL）与下游表现之间拟合出很强的关系——例如 function-generation 能力叶到 HumanEval 的映射报告 $\mathbf{R^2 = 0.999}$。论文同时强调口径必须匹配任务格式：HellaSwag 就是反例，其 task-native 评分统计量与 NLL 的关系明显偏离趋势，说明"用错统计量"会破坏代理指标的可解释性。
- **实用结论**：一个新的中训配方可以用**单次冻结前向**、通过比较其能力级 NLL 与 task-native 统计量来筛查。

## 第 7 章 代码实现与工程细节

### 7.1 官方开源产物

论文明确列出释放内容，并对齐到三个仓库/集合：

| 产物 | 位置 | 说明 |
|:-----|:-----|:-----|
| 评测代码 | `https://github.com/LUMIA-Group/ncp_olmo_eval` | 论文首页标注的官方评测仓库 |
| 模型权重 | `https://huggingface.co/collections/ArchSpace-Collection/ncp-archpreview` | Stage-1 每 100,000 训练步的 checkpoint、对应的 drafter 模型，以及最终 Stage-1 / Stage-2 checkpoint |
| 推理与部署 | `https://github.com/InternLM/lmdeploy` | 论文首页引用的推理部署脚本（通用推理引擎，非本论文专属仓库） |

论文正文对释放范围的表述是：为加速非 vanilla 基础架构的开源研究，正式释放 NCP-ArchPreview 作为**完全预训练**的 latent-space 基础模型，包括模型权重、推理脚本、训练配方与中间评测 checkpoint。需要区分的是：`lmdeploy` 是通用推理引擎而非本论文的实现仓库；本论文专属的开源资产是权重集合与 `ncp_olmo_eval` 评测代码。

### 7.2 完整架构配置（Table 10）

| 组件 | 配置 |
|:-----|:-----|
| 模型类别 | Latent-space 自回归语言模型 |
| 总参数 | 8.94B |
| Token Encoder | 16 层 causal Transformer |
| Concept Module | 8 层 causal Transformer |
| Token Decoder | 16 层 causal Transformer |
| Hidden 维度 | 4,096 |
| FFN 维度 | 11,008 |
| Attention heads / KV groups | 32 / 32 |
| Head 维度 | 128 |
| 词表大小 | 100,278 |
| 最大训练长度 | 8,192 token |
| 位置编码 | RoPE，base = 500,000 |
| token 级局部注意力窗口 | 4,096 token |
| 全局注意力模式 | 每第 4 层使用 full attention |
| 激活函数 | SwiGLU |
| 归一化 | RMSNorm，$\epsilon = 10^{-6}$ |
| 注意力归一化 | Layer-wise QK RMSNorm |
| attention / hidden dropout | 0 |
| 概念压缩因子 $k$ | 每 4 个 token 状态合成 1 个 concept |
| 概念序列长度 | $M = T/k$（padding 与 masking 之后） |
| PQ 分段数 $S$ | 32 |
| 每段码字数 $N$ | 128 |
| 码字维度 | 128 |
| 潜在词表容量 | $N^S = 128^{32}$ 种码字组合 |
| token 级目标 | Next-token prediction（NTP） |
| 潜在空间目标 | Next-concept prediction（NCP） |
| 概念预测器 | 专用 Concept Module |
| 概念反馈方式 | 因果位移 + 重复 $k$ 次 + 加到 token 状态上 |
| 模块内残差路由 | Token Encoder / Concept Module / Token Decoder 各自启用 IRC |
| 跨模块残差路由 | Token Encoder→Concept Module、Token Encoder→Token Decoder、Concept Module→Token Decoder |
| 参数精度 | BF16 |

### 7.3 前向训练流程（Algorithm 1 的结构）

论文附录 B.2 给出端到端训练前向过程的伪代码，输入为 token 序列 $x_{1:T}$ 与压缩因子 $k$；其中用 $S_e$ 与 $S_c$ 分别表示 Token Encoder 与 Concept Module 导出的中间层状态集合，两个算子承担粒度转换：

- $\mathrm{ChunkPool}_k$：把 token 级残差状态转换为概念分辨率（对应式 2 的均值池化）；
- $\mathrm{CausalShiftAndRepeat}$：把概念级状态转回 token 分辨率，且**不暴露未来概念**（对应式 11 的因果位移 + 重复）。

流程的语义可以概括为六步：

1. Token Encoder 处理 $x_{1:T}$，产出 $h_{1:T}$ 并导出层状态集合 $S_e$；
2. $\mathrm{ChunkPool}_k$ 把 $h_{1:T}$ 池化为连续概念 $c_{1:M}$；
3. VQ 把每个 $c_m$ 量化到码字 $d_m$（乘积量化，逐段最近邻）；
4. Concept Module 在概念历史 $c_{<m}$ 上预测下一概念 $\hat{c}_m$，同时导出层状态集合 $S_c$；
5. CRC 把 $S_e$、$S_c$ 按粒度对齐后注入目标模块（Token Encoder→Concept Module / Token Encoder→Token Decoder / Concept Module→Token Decoder），IRC 在每个模块内跨深度组合状态；
6. $\hat{c}_{1:M}$ 经 $\mathrm{CausalShiftAndRepeat}$ 变回 token 分辨率并以残差加入 $h_{1:T}$，Token Decoder 从融合状态预测下一 token。

三个损失（$L_{\mathrm{NTP}}$、$L_{\mathrm{NCP}}$、$L_{\mathrm{VQ}}$）在同一次前向中计算并联合回传。IRC/CRC 的具体形式对应第 4 章的式 14–20。

### 7.4 工程层面的可复用经验

1. **非均匀层激活的 FLOPs 记账**。NCP-ArchPreview 的 Concept Module 每个 chunk 只运行一次，$C \approx 6N_{\text{param}}D$ 会明显失真。论文改用 $C = F_{\text{tok}} \cdot D$ 并排除 embedding 参数，同时显式说明残差连接的解析 FLOPs **不含** source-state materialization、内存流量、reduction 与小 kernel 启动开销。任何要复用这类"非均匀深度激活"架构的工程实现，都应把记账口径与运行时开销分开评估。
2. **矩阵优化器与注意力归一化的交互**。full-matrix Muon 耦合多个 attention head 的更新，配合 layer-wise Q/K normalization 时会出现 attention logit 增长、head 级离群与梯度尖峰；per-head Q/K normalization 是有效的稳定化手段。这是一条与架构正交、但对大规模训练稳定性至关重要的经验。
3. **VQ 适配的显存/吞吐红利**。冻结 backbone、只更新 17M 参数（codebook + 预测头）时，吞吐达到 15,632 tokens/s/GPU（8 卡、MBS=1），显存占用 32.4%，单卡 MBS=2 仍只有 51.6%——对照 LoRA 的 87.9% 与 Full 的 OOM。对于需要在同一 checkpoint 上反复做多域适配的场景，这是明显的成本结构差异。
4. **评测口径必须显式固定**。主表把 26 个 higher-is-better 指标与 10 个 BPB 指标分组，BPB 均值不计入 Overall AVG，Overall AVG 取 26 项无权平均而非分段均值平均；每个 delta 定义为 NCP 减对应 Vanilla。附录 C 的代理筛查同样要求先固定 prompt、专家后缀、来源标识与能力标注再评测候选 checkpoint。这类"先定口径再比数"的做法是复现与交叉比较的前提。
5. **可复现性细节**：few-shot 样例选择与生成使用全局随机种子 42；MMLU 四个域分数与聚合分数使用 5-shot 多选题设置；代码补全在隔离的任务专属环境中用官方测试用例与依赖执行；MultiPL-E HumanEval 等指标按论文附录 E 的配置（shots、样本数、统计量）评测。

## 第 8 章 局限性与延伸阅读

### 8.1 论文自述的局限

**（1）未覆盖长上下文训练。** 本报告的实验集中在 5.73T token 预训练 + 100B token 中训、**标准上下文长度**之下。长上下文训练不在本架构预览的范围内，属于计划中的完整训练配方。论文给出的一个正面预期是：由于概念通路本身工作在**压缩序列**上，更长的上下文可能为 latent-space 建模提供特别有利的设置——概念预测与层级残差路由在长依赖上的表现，是后续要研究的问题。

**（2）语言建模 loss 到下游能力的转换不稳定。** NCP-ArchPreview 在预训练与中训两个阶段都保持 token 级 loss 优势，两个 checkpoint 的聚合下游也都为正，但幅度不同：**预训练后 +2.45 分**，**中训后仅 +0.59 分**且逐 benchmark 波动明显。论文承认这种阶段依赖与任务依赖的 loss–能力关系在近期工作中也被观察到，并把"中训配方、能力感知的数据选择、以及结合 loss 与下游/代理评测的 checkpoint 选择准则"列为后续方向。

**（3）增益在部分域上不成立。** Stage-2 的 Code AVG 下降 0.65 分，作者归因于 code 数据在中训混合中仅约 10%。这是数据配比问题而非架构失效，但意味着"latent 架构全面更优"的说法需要按域限定。

**（4）解析 FLOPs 不等于真实成本。** 层级残差的 +0.051% 只统计张量收缩，不含状态物化、内存流量与 kernel 启动开销；第 6 章也显示 VQ 适配的优势更多体现在吞吐与显存而非目标域极限性能。

### 8.2 结果边界的诚实标注

以下几点属于**推论/边界说明**，读报告时不应与论文的直接结论混淆：

1. **1.74× 计算效率的口径**：该数字来自缩放律拟合（Figure 4/10）在搜索了超参与模型/数据分配后的最佳验证 loss 对比，因此它衡量的是"在给定 FLOPs 预算下可达到的最佳 loss 等价关系"，而不是同一配置下的直接 wall-clock 加速比。
2. **85% 计算的对照对象**：NCP-ArchPreview 是用 $34F_{\text{blk}}$ 逼近 $40F_{\text{blk}}$ 的 size-aligned baseline，而它的参数量（$40P_{\text{blk}}$）与后者相同——因此这一条证明的是**计算效率**，不是"以更少参数达到同等效果"。
3. **概念容量 $128^{32}$** 是乘积量化定义的**理论**组合上界，论文并未声称模型实际使用了全部组合；实际有效容量取决于 codebook 利用率与分布，论文未报告 codebook 利用率指标。
4. **VQ 域适应的知识域上限**：TriviaQA +9.19 明显低于 Full（+18.00）与 LoRA（+17.48），机制解释（FFN 为事实记忆载体、VQ 冻结 FFN）是作者的归因，论文未做针对性的神经元级验证。
5. **drafter 加速的适用面**：+4.17% 宏平均接受长度是在固定的 5B token 训练子集、16 token 提议视野、1.1B drafter 的特定设置下测得，且只覆盖 GSM8K/MATH/HumanEval/MBPP 四个基准，不能外推为通用推理加速比。

### 8.3 与相关工作的定位

- **层级/潜在语言建模**：Hourglass Transformer 与 MegaByte 使用固定层级结构（预定义多尺度分辨率或 byte patch）；BLT、DLCM、H-Net 使用动态或输入自适应分块；ContextLM 学习预测式上下文嵌入。这些工作主要改变**计算粒度或潜在单元的构造机制**。
- **抽象级预测目标**：Joint-Embedding Predictive Architectures（JEPA）在图像（I-JEPA）与视频（V-JEPA / V-JEPA 2）上取得成功，Large Concept Model 把整句映射到共享连续嵌入空间做抽象级自回归；ConceptLM 是本文最直接的来源——它首次提出 NCP 作为离散概念级目标，学习乘积量化概念词表并与语言模型联合训练，但只从零训练到 **1.5B** 参数、并以续训方式给一个 8B 模型加上 NCP。
- **本文的差异**：NCP-ArchPreview 是在 OLMo-3-7B 基础上、**从预训练开始**就启用 NCP、跨 5.73T token 的 8.9B 实现。用一句话概括贡献边界：它把 ConceptLM 的机制从"概念验证规模"推到"万亿 token 规模"，并提供参数量/计算量对齐的对照证据。
- **残差连接谱系**：DenseFormer（输入无关的跨深度平均权重）、DeepCrossAttention（输入相关权重 + 深度交叉注意力）、MUDDFormer（按 query/key/value/residual 分别预测 token 条件化稠密连接权重）、Attention Residuals（对前序层输出做 softmax 注意力，含 blockwise 变体）、Depth-Attention（在自注意力内部混合更早的 value 状态）。本文的 IRC 采用 MUDDFormer 的单流 DD 表述以实现模块内全历史复用；CRC 则额外提供**目标条件化**的跨模块深度选择。
- **多 token 预测**：MTP 为多个未来 token 加辅助头，但损失仍锚定在单个表面 token 上；本文的 NCP 把监督对象换成显式离散概念，并在 3B 对照中显示两者收益可叠加。

### 8.4 值得延伸的阅读线索

| 方向 | 建议阅读 | 与本文的关系 |
|:-----|:---------|:-------------|
| 离散概念预测的起点 | ConceptLM（Liu et al., 2026） | 本文 NCP 机制的直接来源，规模对比的基准 |
| 潜在目标预测 | I-JEPA / V-JEPA 2 / V-JEPA | JEPA 范式在视觉与视频的成功，本文的语言版对应 |
| 层级语言建模 | Hourglass Transformer、MegaByte、BLT、DLCM、H-Net | 潜在单元"如何构造"的另一条技术路线 |
| 连续概念空间 | Large Concept Model | 连续 vs 离散概念空间的对照 |
| 稠密连接与残差 | DenseFormer、MUDDFormer、Attention Residuals、Depth-Attention | IRC 的设计来源与替代方案 |
| 概念表示的下游用法 | DFlash 2、DSpark、块扩散投机解码 | 概念表示作为 drafter 条件信号的工程延伸 |
| 事实记忆与编辑 | Transformer FFN as key-value memory、知识神经元、模型编辑 | 解释 VQ 适配在知识域受限的机制背景 |
