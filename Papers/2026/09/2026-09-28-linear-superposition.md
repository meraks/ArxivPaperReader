# Your Transformer Can Hold Two Thoughts at Once 精读

> **论文**：Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs
> **作者**：9 位作者（HF Daily Papers 2026-09-27/28 连续上榜，64 upvotes）
> **arXiv ID**：2609.29845（v1）
> **发表时间**：2026-09-27
> **代码仓库**：论文未标注官方仓库链接

## 第 1 章 概述

### 1.1 一句话定位

本文提出并验证**叠加线性假说（Superposition Linearity Hypothesis）**：把两条文本流的 token embedding 逐位平均后送入冻结的预训练 Transformer，输出分布近似两条流各自 next-token 分布的叠加——线性叠加是 Transformer 的固有架构属性而非训练习得，预训练会使其衰减，轻量微调可恢复，引导解码可从单次前向解缠出两条连贯延续。

### 论文图表总览

| 编号 | 内容 | 所在章节 |
|------|------|---------|
| **Figure 1** | 排名 CDF：真值 token 在混合分布中 top-10 约 30–40%、top-100 达 60–65% | 第 3 章 |
| **Figure 2** | 训练动态：hidden-state additivity 误差随预训练单调增长 | 第 3 章 |
| **Table 1** | 近似比 $\mathcal{R}_{\mathcal{D}}<1$ 全模型一致（KL/JS/Wasserstein） | 第 3 章 |
| **Table 2** | 注意力贴补分层分析（predictable vs content） | 第 3 章 |
| **Table 5** | Joint Contrastive 解码：LAMBADA 混合 0.18→0.43 | 第 5 章 |
| **Table 6（附录 G）** | Two-Head/Mixed Distillation 替代解码对比 | 第 5 章 |
| **表（附录 G.5）** | 吞吐：Separate ≈ batch-2 独立推理、2× 于顺序推理 | 第 6 章 |
| **表（附录 G.4）** | 单流质量权衡（NLL/PPL） | 第 6 章 |

### 1.2 核心贡献

1. **固有叠加线性的证据**：标准预训练 LLM（Pythia/Qwen/Llama/OLMo/Gemma）对 embedding 平均的混合输入，双流真值 token 以远超频率基线一个数量级的比率存活于输出分布顶部；混合分布整体近似线性混合（$\mathcal{R}<1$ 全模型一致）。
2. **架构属性而非习得能力**：沿 Pythia 预训练轨迹追踪，叠加保真度在初始化时最高、随训练单调衰减——语言建模目标持续放大残差流非线性；逐层线性度呈 U 形，深层（$\ell \gtrsim 2L/3$）保持 >0.95 的近线性。
3. **轻量恢复**：自蒸馏微调（<0.025% 预训练数据量）将 Pythia-2.8B 的 KL 散度 1.86→0.27、近似比 0.42→0.06，且恢复集中发生在 content 位置（中位排名 284→5）而非走频率捷径。
4. **解缠解码**：识别出几何均值障碍（$\propto\sqrt{P_A P_B}$ 惩罚单流高概率 token），提出 Joint Contrastive 解码——小模型逐流引导 + 联合训练，Llama-3.2-3B 混合 LAMBADA 0.182→0.430。
5. **部署价值**：单次前向产两条延续，理论 2× 吞吐 + 每流 KV-cache 减半；实测 Separate 模式与 batch-2 独立推理差距 <3%。

### 1.3 关键结果速览

- **叠加存活率**：未修改模型 top-10 约 30–40%、top-50 约 50–60%、top-100 达 60–65%（词表 ≥50,000；频率基线 top-10 仅 2.63%）。
- **微调恢复**：Pythia-2.8B KL 1.86→0.27、$\mathcal{R}_{\mathrm{KL}}$ 0.42→0.06、真值入 top-5 ≈30%→>60%；代价是单流 LAMBADA 0.544→0.357。
- **解码解缠**：Llama-3.2-3B 混合 LAMBADA 0.182→0.430（小模型单流基线 0.540）；流间 Jaccard 降至 0.067。
- **吞吐**：Separate 109.6 token/s vs 顺序 56.0（Pythia-2.8B+160M），≈2×。
- **N=3 扩展**：三流混合 $\mathcal{R}_{\mathrm{KL}}$ 仅 +0.04~+0.09，性质保持。

## 第 2 章 研究背景与动机

### 2.1 从逐层线性到端到端线性

decoder-only Transformer 的残差流已知的强线性结构是本研究的起点：Razzhigaev et al. (2024) 表明相邻层间的转移常可由仿射映射很好近似。本文把问题从「层间几何」推到「端到端输入输出行为」：模型对输入线性组合的响应是否近似对应输出的线性组合？这是对主流范式的直接挑战——主流把推理视为单一连贯语义流，多流处理需多次前向、串行处理或改架构避免干涉。

### 2.2 假设的形式化与排名直觉

对两条长度 $T$ 的序列，混合输入为 $z_t = \frac{1}{2}(E(x_t^{(A)})+E(x_t^{(B)}))$（式 1），经冻结骨干的标准因果注意力处理。假说的可检验形式：$P_{\text{mix}}(z) \approx \frac{1}{2}(P(x|A)+P(x|B))$。排名分析提供直觉校准：完美线性时双流真值 token 应居排名前二；强非线性网络中 embedding 之和预期与两个语义都正交，真值应落入词表尾部（rank $\sim |\mathcal{V}|/2$）。实测（第 3 章）远靠近前者。

### 2.3 相关工作：外加复用 vs 固有属性

多路复用路线——DataMUX（特征复用+特征化解码）、MIMONets（VSA 绑定/解绑键）、RevMUX（可逆适配器）——把叠加视为**需要外加结构的工程能力**（专用层、绑定机制、等距正则）。推理时利用路线——Superposed Decoding（草稿 token 混合多续写）、superposition prompting（RAG 多文档路径单次前向）。本文的定性差异：叠加是标准预训练 Transformer 的**固有**输入输出效应，简单 embedding 平均在现成模型上已保留大量信号，微调是「恢复」被预训练退化的性质而非「注入」新能力；并首次刻画其训练动态（初始化最强、随训练衰减）与注意力机制归因。

## 第 3 章 固有线性的实证证据

### 3.1 排名分析：真值 token 在混合分布中的存活

对未修改的预训练模型（Pythia、Qwen、Llama、OLMo、Gemma 家族；TinyStories + FineWeb 随机文本对），将两条流的 embedding 逐位平均后过冻结骨干，统计各自真值 next token 在混合输出分布 $P_{\text{mix}}$ 中的排名。直觉校准：若叠加完全线性，两条流的真值 token 应占据排名前二；若网络强非线性，平均后的表征预期与两个语义都正交，真值应落入词表尾部（rank $\sim |\mathcal{V}|/2$）。

实测结果（词表 $\ge 50{,}000$）：真值 token 落入 **top-10 约 30–40%**、**top-50 约 50–60%**、**top-100 达 60–65%**。混合状态没有塌缩成噪声，而是把搜索空间收窄到同时包含两条流有效延续的小邻域。

**频率基线控制**（排除 Zipf 先验解释）：用单流 A 的 logits 去排无关流 B 的同位真值 token——top-3 仅 1.12%、top-10 仅 2.63%、top-100 仅 10.41%，且无关流同位 token 重叠仅 0.2%。叠加前向的 top-10 恢复率（30–40%）比频率基线高一个数量级，效应不可归因于词频分布或词表利用不足。

### 3.2 分布形状保持：Superposition Approximation Ratio

排名只说明真值存活，是否整个分布都近似线性混合？定义目标分布

$$P_{\text{target}}(x|A,B)=\tfrac{1}{2}\left(P(x|A)+P(x|B)\right)$$

并用归一化近似比消除模型族/上下文差异：

$$\mathcal{R}_{\mathcal{D}}=\frac{\mathbb{E}_{(A,B)}\left[\mathcal{D}(P_{\text{target}}\|P_{\text{mix}})\right]}{\mathbb{E}_{(A,B)}\left[\mathcal{D}(P(\cdot|A)\|P(\cdot|B))\right]}$$

$\mathcal{R}_{\mathcal{D}}<1$ 即混合输出比「两条无关上下文之间的距离」更接近理想线性混合。**Table 1**（FineWeb，KL/JS 用温度平滑 $\tau=1.5$，Wasserstein 在 top-256 token 上以 embedding 余弦距离为 ground metric）：

| Model | KL (值/比) | JS (值/比) | Wasserstein (值/比) |
|:---|:---:|:---:|:---:|
| **L=32** | | | |
| Pythia-160M | 0.91 / 0.35 | 0.23 / 0.40 | 0.27 / 0.63 |
| Pythia-410M | 1.54 / 0.39 | 0.36 / 0.50 | 0.30 / 0.69 |
| Pythia-2.8B | 1.86 / 0.42 | 0.40 / 0.54 | 0.31 / 0.71 |
| Llama-3.1-8B | 2.16 / 0.37 | 0.48 / 0.57 | 0.32 / 0.69 |
| **L=512** | | | |
| Pythia-160M | 0.92 / 0.31 | 0.24 / 0.39 | 0.25 / 0.61 |
| Pythia-410M | 1.49 / 0.35 | 0.37 / 0.49 | 0.27 / 0.65 |
| Pythia-2.8B | 1.69 / 0.37 | 0.37 / 0.51 | 0.30 / 0.69 |
| Llama-3.1-8B | 1.98 / 0.34 | 0.45 / 0.54 | 0.31 / 0.67 |

三个散度族的 $\mathcal{R}$ 全部 <1 且跨上下文长度稳定；$P(z)$ 与 $P_{\text{target}}$ 的总变差距离在前 ~20 token 略高后在整个上下文窗口保持恒定——叠加所需的几何性质不随上下文复杂化而失效。

### 3.3 训练动态：线性是架构 gift、预训练是 eroder

沿 Pythia 预训练轨迹追踪 hidden-state additivity 误差 $\bar{\mathcal{E}}$（混合隐状态与两单流隐状态之和的归一化 $\ell_2$ 距离，逐层均值中心化，越低越线性）：**$\bar{\mathcal{E}}$ 在初始化 checkpoint 最小，随预训练单调增长**——叠加线性在随机初始化时最强，语言建模目标持续放大残差流中的非线性交互。

逐层线性度分析（Razzhigaev 2024 线性得分）呈 U 形深度剖面：0–5 层高线性（嵌入整合）、中层（6–20）降至 ~0.65、**后三分之一层恢复到 >0.95**。这个「终端线性」把高层特征对齐到 unembedding 矩阵，几何上解释了排名保持：准线性的最后阶段防止叠加信号 $z \approx x+y$ 在到达输出 logits 前塌缩。

**N=3 流扩展**：三条流等比混合后 $\mathcal{R}_{\mathrm{KL}}$ 增加 +0.04~+0.09，性质保持、仅定量退化（L=512 退化更大）——同一干涉机制跨流数运作。

### 3.4 注意力贴补：解离频率先验与注意力形状

为什么 softmax 注意力没有破坏叠加？在 Qwen2.5-3B（FineWeb-Edu，$T=128$）上比较 embedding mixing 与两种单流扰动：**donor patch**（逐层逐头用无关文本 C 的 post-softmax 注意力权重替换 A 自身的，Q/K/V、RoPE、value 路径不变——保自然注意力结构、解耦内容）与 **permutation patch**（对 A 自身注意力每行在因果前缀内随机置换——保行和与权重多重集、毁结构）。按 token 类型分层（predictable ~65% vs content ~35%）：

| Setup | predictable 中位排名 / top-1% | content 中位排名 / top-1% |
|:---|:---:|:---:|
| Embedding mixing (Base) | 66 / 24.8 | 284 / 4.2 |
| Donor attention patch (Base) | 33 / 33.0 | 11 / 5.4 |
| Permutation patch (Base) | 3,079 / 1.3 | 19,246 / 0.0 |
| Embedding mixing (Fine-tuned) | 8 / 21.6 | 5 / 22.8 |

四个结论：

1. **聚合指标被 predictable 位置主导**：只要扰动保住注意力结构（embedding mixing 与 donor patch 都是），可预测多数位置上任何扰动都表现尚可——LM-head 频率先验占优。content 位置的排名存活才是区分扰动优劣的战场。
2. **Donor patch 的两面性**：作为自一致性指标很良性（整体中位排名 8、$\mathcal{R}_{\mathrm{KL}}=0.27$），作为任务指标是灾难——LAMBADA 200 题 argmax 准确率 73%→0.5%、目标中位排名 2,350。LAMBADA 目标恰是长叙事末位的 content 词，predictable 多数救不了场。
3. **Permutation 解离两个成分**：保留频率先验但毁注意力结构后 $\mathcal{R}_{\mathrm{KL}}$ 0.27→0.68、top-10 一致率 53%→10%、中位排名 8→8,148——频率先验单独不够；donor patch 之所以良性是频率先验 + 自然注意力的结构形状（对角/局部带、attention sink、头特化）的联合贡献，两者缺一不可。
4. **Embedding mixing 携带更多**：聚合自一致性差于 donor（19 vs 8，符合双流共用残差流的预期——完美混合下自然基线排名 ~1.5 而非 1，且每层 Q/K/V 从第 0 层起就被双输入污染），但在 LAMBADA 上反超 donor 4.5× 准确率（2.25% vs 0.5%）与 7× 排名（339 vs 2,350）——embedding 叠加在困难 content 位置保留的逐案例信号，严格多于「频率先验+注意力形状」的组合，且同时对两条无关流做到。

## 第 4 章 微调恢复线性：自蒸馏

### 4.1 目标函数

学生模型 $M_{\text{student}}$ 以预训练权重初始化，冻结的同模型为 teacher。对文本对 $(x^{(A)}, x^{(B)})$，目标分布为 teacher 独立预测的算术平均，学生处理混合 embedding 并最小化 KL：

$$P_{\text{target}}=\tfrac{1}{2}\left(M_{\text{teacher}}(x^{(A)})+M_{\text{teacher}}(x^{(B)})\right),\qquad \mathcal{L}=D_{KL}\left(P_{\text{target}}\;\|\;M_{\text{student}}(z)\right)$$

应用于 Pythia/Qwen/Llama（FineWeb 子集约 200k 步）——**数据量不足预训练的 0.025%**。

### 4.2 恢复效果

| 指标 | 微调前 | 微调后 |
|:---|:---:|:---:|
| Pythia-2.8B 平均 KL | 1.86 | **0.27** |
| 近似比 $\mathcal{R}_{\mathrm{KL}}$ | 0.42 | **0.06** |
| 真值 token 入 top-5 | ≈30% | **>60%** |
| content 位置中位排名 | 284 | **5** |
| content 位置 top-1 一致 | 4.2% | **22.8%** |

关键在第 3.4 节的分层视角：恢复**不是靠抬高高频可预测 token**——predictable 位置中位排名基本不变（66→8 的小改善），真正的大幅恢复发生在 content 位置（284→5）。微调后的模型在真正并行处理两条流的语义内容，不是走频率捷径。

### 4.3 代价：单流质量受损

同一目标可测地降低单流 next-token 预测质量：Pythia-2.8B LAMBADA 0.544→0.357、Qwen2.5-3B 0.602→0.460；FineWeb 上 NLL/PPL 全量权衡见第 6 章。分布保真与解码质量之间的缺口正是第 5 章的主题。

## 第 5 章 解码：几何均值障碍与解缠

### 5.1 几何均值障碍（开放问题）

微调恢复了目标 token 的概率质量，但**直接从混合分布采样**产生语义不一致的序列——模型在 A、B 两流的 token 间交替。根源在式 (6)：logit 近似平均时，混合概率按几何均值缩放

$$P'_{\text{target}}(t)\;\propto\;\exp\!\bigl(\tfrac{1}{2}(\ell_{A}(t)+\ell_{B}(t))\bigr)\;\propto\;\sqrt{P_{A}(t)\,P_{B}(t)}$$

一条流高概率、另一条低概率的 token 被几何均值重罚——即使双真值稳定进 top-5，用标准解码让两流同时 rank-1 仍极难。**完全克服该障碍是留给未来的开放问题**；以下是两个 proof-of-concept。

### 5.2 Joint Contrastive 解码

用小辅助模型 $M_{\text{small}}$ 提供逐流引导，解缠 logits 为

$$\tilde{\ell}^{(A)}=\ell_{\text{large}}(z)+\alpha\,\ell_{\text{small}}(A)-\beta\,\ell_{\text{small}}(B),\qquad \tilde{\ell}^{(B)}=\ell_{\text{large}}(z)+\alpha\,\ell_{\text{small}}(B)-\beta\,\ell_{\text{small}}(A)$$

标量 $\alpha,\beta$ 初始化为 1 并与骨干联合训练（对称逐流交叉熵损失）。**Table 5**（LAMBADA 混合前向平均准确率 / FineWeb 流间 Jaccard 重叠）：

| Backbone | Guide | 方法 | single (large/small) | **mixed** | Jaccard |
|:---|:---|:---|:---:|:---:|:---:|
| Qwen2.5-3B | Qwen2.5-0.5B | Pretrained | 0.592 / 0.437 | 0.168 | 0.126 |
| Qwen2.5-3B | Qwen2.5-0.5B | **Joint Contrastive** | 0.592 / 0.437 | **0.345** | **0.061** |
| Llama-3.2-3B | Llama-3.2-1B | Pretrained | 0.643 / 0.540 | 0.182 | 0.094 |
| Llama-3.2-3B | Llama-3.2-1B | **Joint Contrastive** | 0.643 / 0.540 | **0.430** | **0.067** |
| Pythia-2.8B | Pythia-160m | Pretrained | 0.544 / 0.225 | 0.065 | 0.109 |
| Pythia-1.4B | Pythia-160m | **Joint Contrastive** | 0.499 / 0.225 | **0.110** | **0.080** |

Llama-3.2-3B 的混合准确率达 0.430（对照小模型单流基线 0.540）——单次前向同时产出两条连贯延续的可行性证据，剩余缺口与式 (6) 的障碍一致而非微调不足。

### 5.3 替代解码变体（附录 G）

- **Two-Head 对称破缺**：可学投影 $W_A, W_B = I + \mathcal{N}(0, 10^{-3})$ 打破平均的置换对称性，双头联合交叉熵 + 动态损失平衡（EMA 0.99、$\gamma=5.0$、$\alpha_t/\beta_t$ 截断到 $[0.1, 10]$，防塌缩到单流）。Llama 0.105/J0.235、Pythia-2.8B 0.164/J0.117、Pythia-6.9B 0.080——均弱于 Joint Contrastive。
- **Mixed Distillation**（无引导蒸馏）：Qwen 0.207、Llama 0.174、Pythia-2.8B 0.088——也弱。
- **推理时 logit 算术**（冻结两模型、只梯度优化 $\alpha,\beta$）：最弱，纯后处理不足以分离。

结论：解缠需要大模型混合表征与小模型逐流方向性引导的**联合训练**，事后线性组合或对称破缺都不够。

## 第 6 章 部署视角：吞吐、显存与质量权衡

### 6.1 吞吐与显存（附录 G.5，A100 实测，token/s）

四种同输入 token 数的解码模式对比：**Separate**（融合 embedding $z=\frac{1}{2}(E(A)+E(B))$ 单骨干双头）、**Guided**（大骨干+小模型逐流引导）、**Big(b=2)**（batch 2 独立推理）、**2×Big(seq)**（顺序独立推理）：

| Pair | Separate | Guided | Big(b=2) | 2×Big(seq) |
|:---|:---:|:---:|:---:|:---:|
| Pythia-2.8B + 160M | 109.6±0.8 | 79.4±0.5 | 113.1±0.6 | 56.0±0.3 |
| Pythia-1.4B + 160M | 144.7±1.0 | 93.9±0.6 | 145.4±1.0 | 71.7±0.5 |
| Llama-3B + 1B | 81.5±0.6 | 52.4±0.3 | 81.4±0.5 | 41.3±0.2 |
| Qwen-3B + 0.5B | 64.9±0.5 | 39.4±0.2 | 63.6±0.3 | 32.4±0.2 |

Separate 与 batch-2 独立推理差距 <3% 且对顺序推理约 2× 加速——但两者任务本质不同（融合隐状态同时产两条延续 vs 独立批处理）。显存：Separate 比单流前向仅小幅增加（Pythia-2.8B 7.15 vs 6.34 GB；Llama-3B 9.26 vs 6.72 GB）；Guided 需双模型驻留（Llama-3B+1B 峰值 12.93 GB）。理论上多流压缩进单表征还使每活跃流的 KV-cache 足迹减半。

### 6.2 单流质量权衡（附录 G.4，FineWeb NLL/PPL）

| Family | 方法 | NLL↓ | PPL↓ |
|:---|:---|:---:|:---:|
| Qwen2.5 | 独立大模型 (3B) | 2.521 | 12.44 |
| Qwen2.5 | 独立小模型 (0.5B) | 2.942 | 18.96 |
| Qwen2.5 | Joint Contrastive | 2.961 | 19.31 |
| Qwen2.5 | 微调蒸馏 | 4.182 | 65.52 |
| Llama-3.2 | 独立大模型 (3B) | 2.416 | 11.20 |
| Llama-3.2 | 独立小模型 (1B) | 2.605 | 13.53 |
| Llama-3.2 | Joint Contrastive | 2.691 | 14.75 |
| Llama-3.2 | 微调蒸馏 | 4.269 | 71.44 |
| Pythia | 独立大模型 (2.8B) | 2.645 | 14.09 |
| Pythia | 独立小模型 (160M) | 3.385 | 29.53 |
| Pythia | Joint Contrastive | 3.392 | 29.72 |
| Pythia | 微调蒸馏 | 3.783 | 43.95 |

Joint Contrastive 的单流质量落在小模型水平（大模型引导解码后 2.961 vs 小模型 2.942），纯蒸馏损伤最大（PPL 65+）。

### 6.3 生成质量（TinyStories LLM-as-Judge，GPT-5.2 评分 1–10）

Pythia-2.8B 独立基线 vs 叠加（预训练）vs 叠加（微调）：Grammar 4.52→2.59→3.69；Consistency 3.57→2.70→3.53；Creativity 5.72→2.08→2.13。微调恢复语法与一致性接近独立水平，创造力差距仍大。

### 6.4 训练成本（附录 A）

蒸馏微调 150k 步：Pythia-2.8B 约 114 小时 @2×A100 80GB；Qwen2.5-3B 132 小时；Llama-3.2-3B 128 小时。Joint Contrastive（Pythia-1.4B+160M 引导）68 小时 @单卡 A100。超参：蒸馏用 AdamW lr 1e-3（头）/1e-4（骨干）、温度 2.0、seq 128；JC 用 lr 5e-5、batch 16、seq 512、1000 步 warmup、bf16+SDPA，Qwen/Llama 取 30k 步 checkpoint、Pythia 取 700k。

## 第 7 章 局限性与延伸阅读

### 7.1 局限性

- **短上下文与单语**：分析实验 $L \le 128$（分布度量扩展到 512），语料基本单语；更长上下文与多语言涉及更复杂 embedding 几何，未测。
- **仅文本**：多模态 embedding 混合（文本 token + 图像 patch）的结构性挑战完全未探索。
- **微调的代价**：恢复线性的同时单流质量可测下降（Pythia-2.8B LAMBADA 0.544→0.357；纯蒸馏 PPL 14→44+）；「分布保真」与「解码质量」的缺口由几何均值障碍解释，但完全克服是开放问题。
- **解码解缠不完全**：Joint Contrastive 的混合准确率（0.43）仍低于大模型单流（0.643），且需小模型联合训练；纯推理时方案（logit 算术、Two-Head）更弱。
- **训练成本**：单次蒸馏 150k 步约 114–132 小时 @2×A100 80GB。

### 7.2 延伸阅读

| 方向 | 代表工作 | 关系 |
|:-----|:---------|:-----|
| 残差流线性 | Razzhigaev et al. 2024（逐层仿射近似） | 本文线本性度量与 U 形深度剖面的方法基础 |
| 多路复用 | DataMUX (2022)、MIMONets (2023)、RevMUX (2024) | 外加复用结构 vs 本文固有属性+恢复 |
| 推理时叠加 | Superposed Decoding (Shen 2024)、superposition prompting (Merth 2024) | 叠加的工程利用 |
| 特征超叠加 | Elhage et al. 2022（superposition 假说） | 单 neuron 多特征视角的表征瓶颈 |
| 任务叠加 | Xiong et al. 2024（ICL 中的任务叠加） | 上下文学习中的相关现象 |
