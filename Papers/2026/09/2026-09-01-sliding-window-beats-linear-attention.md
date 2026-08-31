> **论文**：Sliding-window beats linear attention
> **作者**：Alexia Jolicoeur-Martineau, Rhea Sanjay Sukthanker, Pashmina Cameron, Emy Gervais（前三作者：Microsoft Applied Sciences Group；第四作者：Independent）
> **arXiv ID**：2608.28444（cs.CL，交叉 cs.LG）
> **发表时间**：2026-08-28（v1）
> **许可协议**：未标注（arXiv 预印本）
> **代码仓库**：无官方实现（论文未提供代码）

## 第 1 章 概述

### 1.1 一句话定位

本文首次将"带 attention sinks 的 Sliding Window Attention（SWA，window-size $w=64$、sink 数 $s=4$）"作为直接基线，与 12 种后训练（post-training）线性注意力方法在 11 个预训练 LLM（1.3B–70B 参数）上系统对比，证明这个 0 token 后训练、0 个后训练阶段的 training-free 方案可恢复教师模型 MMLU 性能的 93.2%、六基准平均性能的 99.0%，并在长上下文任务（S-NIAH、BABILong）上取得后训练线性注意力 2–10 倍的准确率，因此强烈建议实践者直接切换 SWA 而非后训练线性模型来降低推理内存成本。

### 论文图表总览

| 编号 | 内容 | 章节 |
|------|------|------|
| **Figure 1** | 三类 attention mask 示意：Full Attention (FA)、LoLCATs/Liger-GLA（线性注意力 + SWA 混合）、带 sinks 的 SWA | 第 3 章 |
| **Figure 2** | 四种注意力（FA、SWA、Linear、Linear+SWA 即 LoLCATs）在 128–256K context 下的解码吞吐（tokens/s）与内存成本（KV cache 或循环状态，MiB） | 第 4 章 |
| **Figure 3** | Figure 2 的补充细节图：各 attention 类型的速度与内存随 context（128–256K）变化曲线 | 第 4 章 |
| **Figure 4** | 四种注意力（FA、SWA、Linear、Linear+SWA）在 128–256K context 下的 FLOPs 对比 | 第 4 章 |
| **Table 1** | 13 行方法汇总（12 种后训练线性化方法 + SWA(64,4)）：后训练 token 数、阶段数、MMLU-5shot 恢复率 (%)、六基准平均恢复率 (%) | 第 1、4 章 |
| **Table 2** | 各蒸馏线性化模型、教师模型、教师 + SWA（sink=4，window 64 或 128）在 6 个基准上的完整分数 (%)，标出最佳非教师模型 | 第 4 章 |
| **Table 3** | S-NIAH 准确率：context 0.5K/1K/2K/4K × window 128/256/512，基座 Llama 3.1 8B，对比 SWA、LoLCATs、Liger-GLA | 第 4 章 |
| **Table 4** | BABILong 准确率（QA1–QA5 平均）：context 0K/1K/2K/4K、window 256，对比 SWA 与 LoLCATs | 第 4 章 |
| **Table 5** | 作者自行后训练的新线性化模型（LoLCATs 式两阶段蒸馏，约 0.1B token cleaned-Alpaca；attention 变体 GLA、Gated DeltaNet、QRWKV6；基座 Qwen3、Phi-4-mini-reasoning、Phi-4-reasoning-plus）与 training-free SWA(64,4) 的对比 | 第 4 章 |
| **Table 6** | BABILong 完整结果：QA1–QA5 各子任务在 0K–4K context 上的明细 | 第 4 章 |

### 1.2 核心贡献

1. **补上缺失的基线对比**：首次把带 sinks 的 training-free SWA 与后训练线性注意力在同一口径（学生/教师性能恢复率）下直接对比，覆盖 SUPRA、Hedgehog、LoLCATs、Liger-GLA、MOHAWK、Mamba in the Llama、DiJiang、ARWKV、Llamba、QLinAtt、QRWKV6、QRWKV7 共 12 种方法、11 个预训练模型（Phi-1.5-1.3B 至 Llama 3.1-70B / Qwen2.5-72B-Instruct / QwQ，1.3B–70B 参数），短上下文与长上下文基准兼备。
2. **短上下文结论**：SWA(64,4) 以 0 token 后训练、0 阶段实现 MMLU-5shot 恢复率 93.2%（13 行方法中最高）与六基准平均恢复率 99.0%（仅次于 QRWKV6 的 99.1%，后者需 350–700M token、3 个阶段）；在 11 组模型对比中有 9 组取得最佳平均下游性能（Table 2）。
3. **长上下文结论**：S-NIAH 4K context 下 SWA 恢复全注意力准确率的 17.2–23%，而 LoLCATs 准确率至多 5.8%、Liger-GLA 至多 0.8%（Table 3）；BABILong 4K 下 SWA 恢复基线性能的 25%（绝对准确率 15%）而 LoLCATs 仅 5%（3%）（Table 4）；摘要级结论为长上下文任务 SWA 达线性注意力的 2–10 倍。
4. **方法论纠偏**：指出既有线性注意力论文普遍只与无 sink 的 SWA 比较，而无 sink 的 SWA 在首批 token 移出窗口后会灾难性崩溃，因此这类对比不成立；同时指出后训练线性化模型的长上下文行为此前研究不足。
5. **系统效率实测**：在 4 层 Transformer（embedding 维 1024、16 个 head × 64 维、batch 1、float16、NVIDIA RTX PRO 6000 Blackwell Max-Q）上于 128–256K context 实测四种注意力的速度与内存（Figure 2）：FA 解码吞吐在 context 超过 1K 后随长度下降，SWA 全程最快且内存达到 window-size 后恒定，window=64 时内存低于 Linear、Linear+SWA（LoLCATs）与 window=512 的 SWA，据此给出工程建议——直接切换 SWA 而非后训练线性模型。

### 1.3 关键结果速览

**短上下文知识与推理（Table 1 / Table 2）**

- MMLU-5shot 恢复率：SWA(64,4) 为 93.2%（0 token、0 阶段），高于 QRWKV6 的 92.4%（350–700M token、3 阶段）、Llamba 的 91.5%（8–12B token、3 阶段）、DiJiang 的 88.7%（40B token、1 阶段）、LoLCATs 的 83.2%（40M token、2 阶段）、Hedgehog 的 36.9%（40M token、2 阶段）；各方法后训练 token 需求跨度为 20M（Liger-GLA）至 100B（SUPRA）。
- 六基准（MMLU-5shot、ARC-C、ARC-E、HellaSwag、PIQA、Winogrande）平均恢复率：SWA 为 99.0%，QRWKV6 为 99.1%，LoLCATs 为 97.5%，Liger-GLA 为 92.0%，Hedgehog 为 73.9%。
- 11 组模型对比中 SWA 平均下游性能 9 组最佳；例外为 Phi-1.5-1.3B 上 LoLCATs 62.5% 对 SWA 62.4%，Qwen2.5-32B-Instruct 上 QRWKV6 77.3%（持平教师）对 SWA 76.6%；MMLU 唯一失利为 Llama2.0-7B 上 DiJiang 40.7% 对 SWA 39.8%（Table 2）。

**长上下文推理（Table 3 / Table 4）**

- S-NIAH（基座 Llama 3.1 8B，context 0.5K–4K，window 128/256/512）：所有 window 与 context 组合下 SWA 得分均不低于对手；4K 时 SWA 恢复全注意力准确率的 17.2–23%，LoLCATs 至多 5.8%、Liger-GLA 至多 0.8%（Table 3）。
- BABILong（基座 Llama 3.1 8B，window 256）：2K 时 SWA 19% 对 LoLCATs 10%，4K 时 SWA 15% 对 LoLCATs 3%（恢复率 25% 对 5%）；0K 时 LoLCATs 56% 略高于 SWA 55%，两者恢复基线性能 74–76%，1K 时 22% 对 20%（Table 4）。
- 摘要级结论：长上下文任务（Needle-in-a-Haystack 与 BABILong）上 SWA 准确率为线性注意力后训练方法的 2–10 倍。

**速度与内存（Figure 2，context 128–256K）**

- 速度（tokens/s）：FA 在 context 超过 1K 后解码吞吐随长度下降，其余方法基本平坦；SWA 最快，window=64 快于 window=512，且两者都快于 Linear 与 Linear+SWA（LoLCATs）。
- 内存（MiB）：FA 随 context 线性增长；SWA 增长至 window-size 后保持恒定；内存排序为 SWA(64) 最低，其次 Linear，再次 Linear+SWA（LoLCATs），最后 SWA(512)。

## 第 2 章 研究背景

### 2.1 全注意力与 KV cache 问题

Transformer（Vaswani et al., 2017）是现代 LLM 的骨干：词表大小为 $V$ 的离散 token 经嵌入映射 $V \to D$ 得到形状 $[L, D]$ 的序列，模型交替施加沿维度 $D$ 的 MLP 与沿维度 $L$ 的 Self-Attention（两者均使用残差连接 $x \leftarrow x + f(x)$），最终投影回 $[L, V]$ 预测下一 token 概率。Self-Attention 将 $[L, D]$ 数据线性投影为 queries $\mathbf{q}$、keys $\mathbf{k}$、values $\mathbf{v}$，第 $t$ 个位置的输出为

$$x_t=\frac{\sum_{i=1}^{t}\exp(\mathbf{q}_t\mathbf{k}_i^\top/\sqrt{d})\mathbf{v}_i}{\sum_{i=1}^{t}\exp(\mathbf{q}_t\mathbf{k}_i^\top/\sqrt{d})},\quad t\in[1,\ldots,L]$$

（式 1），即对全部历史位置的 softmax 加权值聚合，$d$ 为缩放的 head 维度。该形式有若干变体，如 Multi-Query Attention（Shazeer, 2019）与 Grouped Query Attention（Ainslie et al., 2023）。复杂度方面，MLP 为 $\mathcal{O}(LD^2)$，而 Self-Attention 为 $\mathcal{O}(DL^2)$——对 context 长度 $L$ 呈二次增长。自回归推理时必须用 KV cache 保存全部历史 keys/values，才能把每生成一个新 token 的代价压到 $\mathcal{O}(L)$，但该缓存随序列长度无限增长：每个新 token 都比上一个更贵，内存占用与处理时间随 context 持续上升，不可持续。

围绕这一点已有大量工程优化：注意力高效实现（FlashAttention、FlashAttention-2；PagedAttention 等服务级内存管理）与 KV cache 压缩（Scissorhands、SnapKV、Quest、H2O），但二次复杂度问题本身并未消除。本文聚焦两条根本性替代路线：Sliding Window Attention（SWA）与 Linear Attention（LA）。

### 2.2 滑动窗口注意力与 attention sinks

SWA（Beltagy et al., 2020, Longformer）不再 attend 全部历史 token，而只 attend 最近 $w$ 个 token（窗口大小）：

$$\mathbf{x}_t=\frac{\sum_{i=\max(1,t-w+1)}^{t}\exp(\mathbf{q}_t\mathbf{k}_i^\top/\sqrt{d})\mathbf{v}_i}{\sum_{i=\max(1,t-w+1)}^{t}\exp(\mathbf{q}_t\mathbf{k}_i^\top/\sqrt{d})},\quad t\in[1,\ldots,L]$$

（式 2）。这一约束看似极强，但类比卷积网络中的有效感受野（Luo et al., 2016）：每层 SA 只看局部窗口，$l$ 层堆叠后有效感受野扩展为 $lw$。经验上，SWA 还能促使模型学习局部感受野之外的依赖，从而改善长期记忆与长度外推（Cabannes et al., 2026）。

attention sinks 现象（Xiao et al., 2024；Barbero et al., 2025）：LLM 会给序列前几个 token 分配异常高的注意力，尽管这些 token 在语义上并不相关——它们充当"多余注意力的存放仓库"。若 attention mask 不包含这些 sink token，性能会灾难性退化；因此当 SWA 窗口移过前几个 token 后，无 sink 的 SWA 表现极差。简单修复（Xiao et al., 2024）：在剩余 $w-4$ 个滑动窗口 token 之外，额外 attend 前 $s=4$ 个 sink token。本文记号 SWA($w$, $s$) 中，$w$ 为窗口大小（实验取 64–512），$s$ 为 sink 数（固定为 4）。SWA 天然无需训练即可工作；后训练可进一步提升其性能（SWAA，Yu et al., 2025），可学习 attention sink（gpt-oss，Agarwal et al., 2025）也需额外后训练——本文聚焦的正是完全 training-free 的 SWA with sinks。

### 2.3 线性注意力与后训练路线

Linear Attention（Katharopoulos et al., 2020）的核心是设计变换 $\phi$，使 $\exp(\mathbf{q}_t\mathbf{k}_i^\top)\approx\phi(\mathbf{q}_t)\phi(\mathbf{k}_i)^\top$，从而把注意力改写为固定大小循环状态的形式：

$$\mathbf{x}_t=\frac{\phi(\mathbf{q}_t)\sum_{i=1}^{t}\phi(\mathbf{k}_i)^\top\mathbf{v}_i}{\phi(\mathbf{q}_t)\sum_{i=1}^{t}\phi(\mathbf{k}_i)^\top}=\frac{\phi(\mathbf{q}_t)\,\mathbf{s}_t}{\phi(\mathbf{q}_t)\,\mathbf{z}_t}$$

推理时按递推更新

$$\mathbf{s}_t=\mathbf{s}_{t-1}+\phi(\mathbf{k}_t)^\top\mathbf{v}_t,\qquad \mathbf{z}_t=\mathbf{z}_{t-1}+\phi(\mathbf{k}_t)^\top$$

（式 3）。每 token 推理成本变为关于 $L$ 的 $\mathcal{O}(1)$，只需存储并更新不随时间增长的 $\mathbf{s}$ 与 $\mathbf{z}$，从根本上消除了 KV cache 与二次复杂度。代价同样明显：表达力（expressivity）低于 softmax 注意力；模型必须在持续覆写自身状态的同时不遗忘/忽略重要信息（"记什么、忘什么"的不可解问题）；且训练昂贵、现有软硬件生态大多围绕 softmax 注意力构建。实践中衍生出大量变体（Mamba、RWKV 系列、RetNet、GLA、Griffin、DeltaNet、Mamba-2/SSD 等），kernel 选择也从 ReLU、ELU 等简单函数不断发展。一个好 kernel 需要三个性质：expressiveness、spikiness、monotonicity；为此 Hedgehog（Zhang et al., 2024）提出可学习投影加双侧指数变换

$$\phi(x)\leftarrow(\exp(f(x)),\exp(-f(x)))$$

（式 4），其中 $f$ 是从 $D$ 到 $D/2$ 的线性投影。

由于从头训练线性注意力模型成本高，主流做法是把预训练 LLM 的二次 Self-Attention 替换为线性注意力再后训练恢复性能——一般需要数十亿 token（Bick et al., 2024a）。LoLCATs（Zhang et al., 2025a）将该代价压缩到 40M token：用 LoRA（Hu et al., 2021）做低秩适配，采用 Hedgehog 这类高表达力 kernel，并将线性注意力与小窗口 SWA 混合，从而恢复大部分全注意力性能。这条路线由此成为高效后训练线性注意力的代表；Table 1 汇总的各方法后训练开销从 Liger-GLA 的 20M token、LoLCATs 的 40M token 到 Mamba in the Llama 的 20B token、DiJiang 的 40B token 与 SUPRA 的 100B token 不等，后训练阶段数为 1–3 个。
## 第 3 章 方法

本文的方法论贡献不在于提出新架构，而在于补上一个被既有文献系统性遗漏的对比：**带 attention sinks 的 training-free SWA 从未被纳入后训练线性注意力的评估基线**。本章给出这一"缺失的对比"的论证（3.1 节）、SWA($w$, $s$) 的记号与配置（3.2 节）、以及覆盖 11 个基座模型与短/长上下文基准的对比协议（3.3 节）。

### 3.1 缺失的对比：sink-free SWA 是一个注定失败的基线

后训练线性注意力文献（SUPRA、Hedgehog、LoLCATs、Liger-GLA、MOHAWK、Mamba in the Llama、DiJiang、ARWKV、Llamba、QLinAtt、QRWKV6、QRWKV7 共 12 种方法）在评估时通常会把 SWA 当作"简单基线"纳入，但它们对比的大多是**不含 attention sinks 的 SWA（sink-free SWA）**（论文原文："most linear attention papers only compare linear-attention to regular (sink-free) SWA"）。

第 2.2 节的 attention sinks 现象（Xiao et al., 2024；Barbero et al., 2025）决定了这类基线的结局：预训练 LLM 会把大量冗余注意力分配给序列开头的少数 token，这些 token 的功能是充当"多余注意力的存放仓库"而非携带语义信息。一旦滑动窗口滑过序列开头、attention mask 不再覆盖这些 sink token，模型性能立即灾难性退化——这是机制使然，与滑动窗口思想本身能否胜任注意力无关。

由此产生双重失真：

1. **基线必然崩溃**：sink-free SWA 在长序列上的失败是结构性的，"线性注意力优于 SWA"的既有结论无法区分"线性注意力优于局部注意力"与"任何不含 sinks 的 attention mask 都会失败"这两种解释，对比因此不具说服力。
2. **长上下文行为缺测**：既有线性化文献对后训练模型的长上下文能力（needle 检索、长程推理类任务）系统评估不足，而这恰是循环状态式注意力最需要被检验的场景。

本文把 SWA($w$, $s$)（$s=4$）作为同等地位的候选补入同口径对比，并以 S-NIAH 与 BABILong 填补第二条缺口（结果见第 4 章）。

### 3.2 SWA(w, s)：记号、配置与零成本实现

记号 SWA($w$, $s$) 中 $w$ 为窗口大小，$s$ 为 sink 数。滑动窗口注意力（式 2）为

$$\mathbf{x}_t=\frac{\sum_{i=\max(1,t-w+1)}^{t}\exp(\mathbf{q}_t\mathbf{k}_i^\top/\sqrt{d})\mathbf{v}_i}{\sum_{i=\max(1,t-w+1)}^{t}\exp(\mathbf{q}_t\mathbf{k}_i^\top/\sqrt{d})},\quad t\in[1,\ldots,L]$$

在此基础上加入 sinks：每个时间步 attend 的 token 总数为 $w$，其中滑动窗口贡献最近的 $w-s$ 个 token，另有序列开头的前 $s$ 个 token 作为 sinks 恒被 attend、永不移出有效集合。

实验配置为：

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| $w$（窗口大小） | 64–512 | 短上下文主配置 64（Table 1/2），长上下文实验取 128/256/512（Table 3/4） |
| $s$（sink 数） | 恒为 4 | 遵循 Xiao et al. (2024) 的简单修复，不做调参 |
| 后训练 tokens | 0 | 不更新任何参数 |
| 后训练阶段数 | 0 | 无 LoRA、无蒸馏、无权重替换 |
| 专用 kernel | 无 | 仅修改 attention mask 即可实现 |

换言之，SWA 的全部"方法"就是在一个已预训练好的模型上换 mask：分数通过直接对教师模型施加 mask 获得，KV cache 只需保留 $w$ 个 token（含 sinks）。对照组的 12 种线性化方法则分别需要 20M–100B token 的后训练数据与 1–3 个后训练阶段（Table 1），例如 LoLCATs 需 40M token、2 阶段，DiJiang 需 40B token、1 阶段，SUPRA 需 100B token、1 阶段。

### 3.3 对比协议：模型、基准与公平性原则

**基座模型**：11 个预训练 LLM，参数跨度 1.3B–70B，每个基座对应其文献中已有的线性化 student：

| 基座模型 | 参数规模 | 对比的线性化方法（student） |
| --- | --- | --- |
| Phi-1.5 | 1.3B | MOHAWK、LoLCATs(+SWA) |
| Mistral-7B-v0.1 | 7B | SUPRA、LoLCATs、Liger-GLA |
| Llama 2.0 7B | 7B | DiJiang |
| Llama 3.0 8B | 8B | Hedgehog、LoLCATs、Liger-GLA |
| Llama 3.0 8B Instruct | 8B | Mamba2(+SWA)（Mamba in the Llama） |
| Llama 3.1 8B | 8B | LoLCATs、Llamba |
| Llama 3.1 70B | 70B | LoLCATs |
| Qwen2.5 7B Instruct | 7B | ARWKV、QLinAtt、QRWKV6-RoPE、QRWKV7、QRWKV7-RoPE |
| Qwen2.5 32B Instruct | 32B | QRWKV6 |
| QwQ 32B | 32B | QRWKV6 |
| Qwen2.5 72B Instruct | 72B | QRWKV6、QRWKV7 |

**短上下文基准**（6 项）：MMLU-5shot、ARC-Challenge（acc-norm）、ARC-Easy（acc）、HellaSwag（acc-norm）、PIQA（acc）、WinoGrande（acc），另报六基准平均（Avg）。

**长上下文基准**（2 项，基座均为 Llama 3.1 8B）：

- S-NIAH：S-NIAH-1/2/3 三种 needle 任务 × context 0.5K/1K/2K/4K，对比 SWA($w\in\{128,256,512\}$, 4)、LoLCATs(+SWA)、Liger-GLA(+SWA)，以 Full Attention 为教师参照。
- BABILong：QA1–QA5 五类推理任务 × context 0K/1K/2K/4K，window 256，对比 SWA(256, 4) 与 LoLCATs。

**指标**：统一采用 student/teacher 性能恢复率（%），即同一基座上线性化模型（或加 mask 的教师）分数除以教师分数，各基准分别计算后取六基准平均。

**公平性原则**：纳入对比的所有方法均为"将注意力机制整体替换"的纯线性化路线——要么全部注意力层换成线性注意力，要么换成线性注意力 + 小窗口 SWA 混合（LoLCATs(+SWA)、Liger-GLA(+SWA)、Mamba2(+SWA) 属后者）；**不含保留全注意力层的混合模型**，这类 FA-hybrid 架构被明确排除在对比之外（其地位另见 6.1 节局限性）。附录 A.2 中作者自行后训练的新线性化模型（Qwen3-8B、Phi-4-mini-reasoning、Phi-4-reasoning-plus 三个基座；Gated DeltaNet、GLA、QRWKV6 三种注意力变体；约 0.1B token cleaned-Alpaca 两阶段蒸馏）同样遵循该口径，用以补充验证结论不依赖于第三方报告数字。

## 第 4 章 实验设置

### 4.1 基座模型

论文在 11 个预训练基座模型上对比训练-free SWA 与线性注意力后训练方法，模型规模覆盖 1.3B 至 70B：

| 模型族 | 模型 | 参数量 |
|--------|------|--------|
| Phi | Phi-1.5-1.3B | 1.3B |
| Mistral | Mistral-7B-v0.1 | 7B |
| Llama 2 | Llama2.0-7B | 7B |
| Llama 3 | Llama3.0-8B / Llama3.0-8B-Instruct | 8B |
| Llama 3.1 | Llama-3.1-8B / Llama-3.1-70B | 8B / 70B |
| Qwen 2.5 | Qwen2.5-7B-Instruct / Qwen2.5-32B-Instruct / Qwen2.5-72B-Instruct | 7B / 32B / 72B |
| QwQ | QwQ-32B | 32B |

### 4.2 评估基准

短上下文评估采用 6 个基准：**MMLU**（5-shot）、**ARC-C**（acc-norm）、**ARC-E**（acc）、**HellaSwag**（acc-norm）、**PIQA**（acc）、**WinoGrande**（acc）。这 6 个基准是线性注意力文献的标准评测集合，便于与 LoLCATs、QRWKV 等方法直接横向比较。

长上下文推理采用两个任务：

- **S-NIAH**（Single Needle-in-a-Haystack）：在最长 4K 上下文中检索单个/多个埋入信息，分为 S-NIAH-1、S-NIAH-2、S-NIAH-3 三档复杂度；
- **BABILong**：基于 bAbI 任务的长上下文推理基准，取 QA1-QA5 五个子任务的平均准确率，上下文长度从 0K 到 4K。

### 4.3 对比方法与配置

- **Teacher**：原始全注意力预训练模型（无任何训练），作为性能上界；
- **SWA(w, s)**：训练-free 滑动窗口注意力，窗口大小 w ∈ {64, 128, 256, 512}，attention sinks 数量 s 恒为 4（前 4 个 token 始终参与注意力）；
- **线性注意力后训练方法**：SUPRA、Hedgehog、LoLCATs、Liger-GLA、MOHAWK、Mamba in the Llama、DiJiang、ARWKV、Llamba、QLinAtt、QRWKV6、QRWKV7，后训练 token 量从 20M 到 100B 不等；
- **附录 A.2 自训对比**：作者还自行以 LoLCATs 式两阶段蒸馏（约 0.1B tokens 的 cleaned-Alpaca）训练 Gated DeltaNet、GLA、QRWKV6 三个线性注意力变体，基座为 Qwen3-8B、Phi-4-mini-reasoning、Phi-4-reasoning-plus。

论文明确声明：为公平与简洁，不对比包含全注意力层的混合模型。

### 4.4 速度与内存测试环境

速度/内存/FLOPs 实验在 **NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition** 上进行。测试网络为 4 层 Transformer（embedding size 1024，16 个头、每头维度 64），batch size 1，float16 精度。后端配置：全注意力（FA）与 SWA 使用 FlashAttention；线性注意力使用 ThunderKittens；LoLCATs 使用 ThunderKittens 的融合 SWA+线性注意力 kernel（窗口 256）。

## 第 5 章 实验结果与分析

### 5.1 短上下文：性能恢复率对比（Table 1）

Table 1 汇总了各方法相对 Teacher 基线的性能恢复率（student 分数 ÷ teacher 分数）。SWA(64, 4) 以 **0 token 后训练**取得 93.2% 的 MMLU 恢复率与 99.0% 的平均恢复率，是唯一同时在这两项上领先的配置：

| 方法 | 后训练 Tokens | Stages | MMLU 恢复 (%) | 平均恢复 (%) |
|------|:---:|:---:|:---:|:---:|
| SUPRA | 100B | 1 | 53.0 (0.0) | 88.1 (0.0) |
| Hedgehog | 40M | 2 | 36.9 (0.0) | 73.9 (0.0) |
| LoLCATs | 40M | 2 | 83.2 (2.2) | 97.5 (1.3) |
| Liger-GLA | 20M | 1 | 62.2 (5.8) | 92.0 (2.8) |
| MOHAWK | 3-5B | 3 | 56.9 (0.0) | 92.4 (0.0) |
| Mamba in the Llama | 20B | 2 | 67.7 (0.0) | 86.7 (0.0) |
| DiJiang | 40B | 1 | 88.7 (0.0) | - |
| ARWKV | 60M/830M | 2/3 | 84.1 (0.0) | 94.7 (0.0) |
| Llamba | 8-12B | 3 | 91.5 (0.0) | 98.6 (0.0) |
| QLinAtt | 350-700M | 3 | 74.0 (0.0) | 92.9 (0.0) |
| QRWKV6 | 350-700M | 3 | 92.4 (2.7) | 99.1 (0.8) |
| QRWKV7 | 350-700M | 3 | 86.4 (6.8) | 96.1 (4.1) |
| **SWA(64, 4 sinks)** | **0** | **0** | **93.2 (3.5)** | **99.0 (0.5)** |

括号内为多个基座模型间的标准差。SWA 以 0 训练成本取得最高的 MMLU 恢复率（93.2%），平均恢复率 99.0% 与 QRWKV6 的 99.1% 基本持平；在「训练效率 × 性能」权衡上，SWA 是唯一无需任何训练即可达到该水平的方案。

### 5.2 逐模型完整结果（Table 2）

Table 2 给出 11 个基座模型 × 6 基准的完整数值。列序：MMLU (5-shot)、ARC-C (acc-norm)、ARC-E (acc)、HellaSwag (acc-norm)、PIQA (acc)、WinoGrande (acc)、Avg。

**Phi-1.5-1.3B**

| 类型 | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Teacher | 42.5 | 48.0 | 76.2 | 62.6 | 76.6 | 72.9 | 63.1 |
| SWA(64,4) | 39.3 | 48.1 | 76.2 | 61.5 | 76.4 | 72.8 | 62.4 |
| MOHAWK | 24.2 | 44.1 | 74.0 | 60.2 | 75.5 | 71.7 | 58.3 |
| LoLCATs(+SWA) | 39.2 | 46.9 | 77.0 | 62.3 | 76.9 | 72.7 | 62.5 |

**Mistral-7B-v0.1**

| 类型 | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Teacher | 62.5 | 54.3 | 80.1 | 81.2 | 80.8 | 75.1 | 72.3 |
| SWA(64,4) | 56.3 | 53.8 | 80.3 | 80.6 | 80.9 | 75.4 | 71.2 |
| SUPRA | 33.1 | 45.8 | 75.9 | 77.1 | 80.1 | 70.3 | 63.7 |
| LoLCATs(+SWA) | 51.4 | 54.9 | 81.7 | 80.7 | 81.5 | 74.0 | 70.7 |
| Liger-GLA(+SWA) | 36.3 | 49.3 | 78.7 | 76.3 | 80.1 | 70.1 | 65.1 |

**Llama2.0-7B**

| 类型 | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Teacher | 45.9 | 45.1 | 75.5 | 76.2 | 78.1 | 69.3 | 65.0 |
| SWA(64,4) | 39.8 | 44.8 | 75.5 | 75.3 | 78.1 | 69.5 | 63.8 |
| DiJiang | 40.7 | 42.7 | 62.6 | 69.4 | 77.5 | - | - |

**Llama3.0-8B**

| 类型 | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Teacher | 65.5 | 53.9 | 80.8 | 79.1 | 78.5 | 73.3 | 71.9 |
| SWA(64,4) | 59.8 | 54.2 | 80.9 | 78.8 | 78.8 | 73.7 | 71.0 |
| Hedgehog | 24.2 | 40.6 | 71.1 | 50.7 | 77.4 | 54.3 | 53.1 |
| LoLCATs(+SWA) | 52.8 | 54.9 | 81.7 | 79.7 | 80.9 | 74.1 | 70.7 |
| Liger-GLA(+SWA) | 43.4 | 52.5 | 81.1 | 76.3 | 80.3 | 72.0 | 67.6 |

**Llama3.0-8B-Instruct**

| 类型 | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Teacher | 66.8 | 55.9 | 82.2 | 75.5 | 77.7 | 71.3 | 71.6 |
| SWA(64,4) | 61.4 | 55.7 | 82.2 | 75.3 | 78.1 | 71.3 | 70.7 |
| Mamba2(+SWA) | 45.2 | 48.0 | 74.1 | 70.8 | 75.8 | 58.6 | 62.1 |

**Llama-3.1-8B**

| 类型 | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Teacher | 65.4 | 55.1 | 81.9 | 79.2 | 78.9 | 74.2 | 72.5 |
| SWA(64,4) | 60.6 | 55.1 | 82.1 | 78.9 | 79.3 | 75.0 | 71.8 |
| LoLCATs(+SWA) | 54.9 | 54.4 | 82.4 | 79.1 | 81.0 | 69.7 | 70.3 |
| Llamba | 60.0 | 54.6 | 82.5 | 77.6 | 80.9 | 73.3 | 71.5 |

**Llama-3.1-70B**

| 类型 | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Teacher | 78.9 | 61.4 | 85.0 | 85.7 | 82.5 | 81.6 | 79.1 |
| SWA(64,4) | 73.2 | 61.2 | 85.1 | 85.4 | 82.3 | 81.7 | 78.2 |
| LoLCATs(+SWA) | 67.7 | 60.5 | 85.0 | 84.6 | 82.1 | 73.7 | 75.6 |

**Qwen2.5-7B-Instruct**

| 类型 | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Teacher | 74.2 | 55.4 | 81.4 | 80.4 | 79.7 | 70.9 | 73.7 |
| SWA(64,4) | 70.3 | 55.4 | 81.8 | 79.8 | 79.4 | 71.3 | 73.0 |
| ARWKV | 62.4 | 52.2 | 79.7 | 76.8 | 79.2 | 68.7 | 69.8 |
| QLinAtt | 54.9 | 53.4 | 79.6 | 75.3 | 78.8 | 68.9 | 68.5 |
| QRWKV6-RoPE | 66.1 | 56.7 | 81.7 | 78.9 | 79.9 | 70.6 | 72.3 |
| QRWKV7 | 65.7 | 56.3 | 81.4 | 79.0 | 80.3 | 71.1 | 72.3 |
| QRWKV7-RoPE | 68.2 | 55.8 | 81.7 | 79.3 | 79.8 | 71.8 | 72.8 |

**Qwen2.5-32B-Instruct**

| 类型 | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Teacher | 83.2 | 58.5 | 82.2 | 85.3 | 80.8 | 73.0 | 77.2 |
| SWA(64,4) | 79.5 | 58.8 | 81.9 | 84.9 | 80.7 | 73.6 | 76.6 |
| QRWKV6 | 76.6 | 60.9 | 84.3 | 83.0 | 81.2 | 78.2 | 77.3 |

**QwQ-32B**

| 类型 | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Teacher | 79.9 | 55.6 | 81.0 | 84.1 | 79.8 | 70.5 | 75.2 |
| SWA(64,4) | 79.9 | 55.6 | 81.0 | 84.1 | 79.8 | 70.5 | 75.2 |
| QRWKV6 | 74.3 | 56.4 | 80.6 | 83.0 | 80.4 | 73.2 | 74.7 |

**Qwen2.5-72B-Instruct**

| 类型 | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Teacher | 84.6 | 63.1 | 86.0 | 87.4 | 83.3 | 76.4 | 80.1 |
| SWA(64,4) | 81.8 | 63.7 | 86.4 | 87.2 | 83.2 | 76.3 | 79.8 |
| QRWKV6 | 77.5 | 63.8 | 86.5 | 85.7 | 82.5 | 79.6 | 79.3 |
| QRWKV7 | 66.7 | 57.2 | 83.3 | 78.2 | 80.1 | 73.9 | 73.2 |

**关键观察**：

- SWA 在 **9/11 个基座模型**上取得最佳平均下游性能。两个例外：① LoLCATs 在 Phi-1.5-1.3B 上以 62.5 微超 SWA 的 62.4；② QRWKV6 在 Qwen2.5-32B-Instruct 上达到 77.3（与 Teacher 持平），SWA 为 76.6。
- 在 MMLU 上，SWA 在除 Llama2.0-7B（DiJiang 40.7 vs SWA 39.8）外的所有模型上均为最佳。
- 值得注意的反常现象：QwQ-32B 上 SWA(64,4) 的 6 项指标与 Teacher 完全相同（79.9/55.6/81.0/84.1/79.8/70.5），说明该模型窗口内注意力覆盖已足以复现全部评估性能。
- 线性注意力方法在部分模型上出现**灾难性退化**（如 Llama3.0-8B 上 Hedgehog 平均仅 53.1、GLA 在 Qwen3-8B 上仅 51.3），而 SWA 从未出现此类塌缩。

### 5.3 附录 A.2：自训线性注意力变体（Table 5）

作者以 ~0.1B tokens 的 cleaned-Alpaca 两阶段蒸馏自行训练线性注意力变体（Gated DeltaNet、GLA、QRWKV6），结论一致：SWA 全面优于，线性变体几乎全部塌缩至随机水平附近：

**Qwen3-8B**

| 类型 | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Teacher | 74.9 | 56.7 | 83.5 | 75.0 | 76.6 | 68.4 | 72.5 |
| SWA(64,4) | 70.8 | 56.9 | 83.6 | 74.3 | 76.5 | 67.6 | 71.6 |
| Gated DeltaNet | 27.5 | 46.8 | 76.6 | 59.1 | 73.9 | 52.9 | 56.1 |
| GLA | 25.8 | 38.0 | 70.5 | 50.9 | 72.0 | 50.6 | 51.3 |
| QRWKV6 | 24.6 | 41.5 | 74.7 | 53.7 | 72.6 | 52.0 | 53.2 |

**Phi-4-mini-reasoning**

| 类型 | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Teacher | 57.4 | 48.0 | 71.4 | 64.8 | 69.3 | 59.1 | 61.7 |
| SWA(64,4) | 54.8 | 48.1 | 71.0 | 63.3 | 69.0 | 59.5 | 60.9 |
| Gated DeltaNet | 24.3 | 34.7 | 64.0 | 45.5 | 68.7 | 51.1 | 48.1 |
| GLA | 25.2 | 30.7 | 59.2 | 40.9 | 66.5 | 51.8 | 45.7 |
| QRWKV6 | 25.1 | 31.4 | 61.4 | 41.8 | 67.6 | 52.0 | 46.6 |

**Phi-4-reasoning-plus**

| 类型 | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Teacher | 77.8 | 56.5 | 82.5 | 78.7 | 80.1 | 76.7 | 75.4 |
| SWA(64,4) | 74.4 | 56.5 | 82.4 | 78.4 | 80.3 | 77.1 | 74.9 |
| Gated DeltaNet | 25.9 | 44.5 | 75.3 | 60.1 | 75.7 | 52.6 | 55.7 |
| GLA | 23.7 | 45.0 | 75.9 | 60.0 | 74.3 | 52.8 | 55.3 |
| QRWKV6 | 25.7 | 40.1 | 72.7 | 57.3 | 74.6 | 53.2 | 53.9 |

三个现代架构上，Gated DeltaNet/GLA/QRWKV6 经 0.1B tokens 蒸馏后平均分全部落在 45.7-56.1 区间（接近或低于随机水平），而 SWA(64,4) 保持在 Teacher 的 96-99% 水平。

### 5.4 长上下文：S-NIAH（Table 3）

基座模型为 Llama 3.1 8B，对比 SWA、LoLCATs(+SWA)、Liger-GLA(+SWA) 在窗口 128/256/512 与上下文 0.5K-4K 下的 S-NIAH-1/2/3 表现：

| 模型 | 窗口 | S-NIAH-1 (.5K/1K/2K/4K) | S-NIAH-2 (.5K/1K/2K/4K) | S-NIAH-3 (.5K/1K/2K/4K) |
|------|:---:|:---:|:---:|:---:|
| SWA | 128 | 35.0/20.2/15.0/12.6 | 100/33.0/22.8/9.2 | 99.8/56.2/41.4/17.2 |
| LoLCATs(+SWA) | 128 | 29.4/9.6/3.0/0 | 100/17.4/7.2/4.2 | 98.2/14.6/3.2/1.6 |
| Liger-GLA(+SWA) | 128 | 28.4/0.2/0.2/0.2 | 100/1.6/0.6/1.0 | 97.6/2.0/1.2/0.8 |
| SWA | 256 | 91.8/34.8/21.0/14.6 | 100/51.4/33.4/13.8 | 100/68.0/50.2/19.6 |
| LoLCATs(+SWA) | 256 | 84.8/26.2/10.2/2.2 | 100/37.0/12.2/8.2 | 100/30.6/6.0/2.2 |
| Liger-GLA(+SWA) | 256 | 83.8/0.0/0.0/2.8 | 100/1.0/4.6/0.8 | 97.6/1.0/3.8/0.6 |
| SWA | 512 | 100/69.0/33.6/19.0 | 100/85.2/51.0/23.0 | 100/90.4/64.8/23.0 |
| LoLCATs(+SWA) | 512 | 100/65.6/24.6/8.8 | 100/71.8/17.4/16.6 | 100/51.0/10.6/5.8 |
| Liger-GLA(+SWA) | 512 | 73.4/1.0/0.0/0.0 | 100/2.6/4.0/0.0 | 97.6/1.2/1.2/0.0 |
| Full Attention | ∞ | 100/100/100/100 | 100/100/100/100 | 100/99.8/100/99.8 |

在**所有窗口大小与上下文长度**下，SWA 的得分均 ≥ 线性注意力方法。4K 上下文下，SWA 恢复了全注意力准确率的 **17.2%-23%**（S-NIAH-3），而 LoLCATs 最高仅 5.8%、Liger-GLA 最高仅 0.8%——差距达 3-20 倍。

### 5.5 长上下文：BABILong（Table 4）

BABILong 平均（QA1-QA5 均值，窗口 256，基座 Llama 3.1 8B）：

| 模型 | 0K | 1K | 2K | 4K |
|------|:---:|:---:|:---:|:---:|
| SWA(256, 4) | 55 | 20 | 19 | 15 |
| LoLCATs(+SWA) | 56 | 22 | 10 | 3 |
| Full Attention | 74 | 70 | 67 | 60 |

短上下文（0K-1K）下 LoLCATs 略高（56 vs 55、22 vs 20），但上下文拉长后 SWA 优势显著扩大：2K 时 19 vs 10，4K 时 **15 vs 3**。相对全注意力基线：0K 时两者恢复 74%-76%；4K 时 SWA 恢复 **25%**，LoLCATs 仅恢复 **5%**。

完整的 QA1-QA5 分任务结果见附录（Table 6）：

| 模型 | QA1 (0K/1K/2K/4K) | QA2 | QA3 | QA4 | QA5 |
|------|:---:|:---:|:---:|:---:|:---:|
| SWA(256,4) | 84/21/16/12 | 36/14/11/6 | 26/16/21/11 | 82/16/17/18 | 49/35/29/30 |
| LoLCATs(256) | 100/22/5/3 | 25/10/4/1 | 30/17/13/4 | 65/21/9/1 | 62/42/17/6 |
| Full attention | 94/88/82/74 | 59/49/48/44 | 47/53/48/44 | 82/76/74/67 | 90/86/83/73 |

### 5.6 速度与内存（Figure 3/4）

速度与内存实验（4 层 Transformer、embedding 1024、16 头 × 64 维、batch 1、fp16、RTX PRO 6000 Blackwell Max-Q）：

- **解码速度**：全注意力在上下文超过 1K 后速度持续下降；SWA、线性注意力、LoLCATs 均保持近似恒定速度；SWA 是全部方法中最快的（窗口 64 比 512 更快，两者均快于线性注意力与 LoLCATs）。
- **内存**：全注意力的 KV cache 随上下文线性增长；SWA 的内存增长到窗口大小后封顶不变；窗口 64 的 SWA 内存最低，其次为纯线性注意力、LoLCATs，最后是窗口 512 的 SWA。
- 因此，在窗口 ≤512 时 SWA 同时拥有更高速度与相似或更低的内存占用（Figure 3），且 FLOPs 随上下文增长远低于全注意力（Figure 4）。

### 5.7 综合结论

论文的核心实证结论可以概括为三条：

1. **短上下文**：训练-free 的 SWA(64, 4) 在 11 个模型上平均恢复 Teacher 性能的 99.0%，与需要 350M-100B tokens 后训练的 SOTA 线性注意力方法（QRWKV6 99.1%）持平，且以 0 训练成本胜出；
2. **长上下文**：SWA 对线性注意力后训练的领先从 2 倍到 10 倍不等（S-NIAH 4K：SWA 17.2-23% vs LoLCATs ≤5.8%；BABILong 4K：15 vs 3），线性注意力在长上下文上几乎完全失效；
3. **效率**：SWA 无需后训练、无需专用 kernel，解码速度最快，内存随窗口封顶——是「性能 × 成本」权衡下的最优解。
## 第 6 章 局限性与结论

### 6.1 局限性

1. **仅评估 training-free SWA**：本文所有 SWA 结果均来自零后训练的直接 mask 替换，因此报告的是该路线的 performance 下界。既有工作已表明后训练可以继续抬升 SWA 上限——SWAA（Yu et al., 2025）对 SWA 做后训练，gpt-oss（Agarwal et al., 2025）使用可学习 attention sinks——而本文的对比尚未触及这些增强配置。
2. **未考虑 FA 混合架构**：部分层保留全注意力、其余层线性化（或 SWA 化）的混合模型未纳入对比（见 3.3 节公平性原则），其在内存-性能权衡中的位置留待后续。
3. **覆盖范围有限**：结论基于 1.3B–70B 参数、6 项短上下文基准与 2 项长上下文基准，未评估超大规模模型、agentic 工作流以及多模态/视频等场景。
4. **跨模态先例待检验**：视频扩散领域已有 Sliding Tile Attention 的正面先例——在 HunyuanVideo 上取得 3.53× 速度提升并恢复 97% VBench 分数——提示局部窗口注意力思想可迁移至时空 token 序列，但本文未做多模态实验。

### 6.2 未来工作

- **SWA 后训练对比**：在相同后训练 token 预算下系统比较"后训练 SWA"与"后训练线性注意力"，检验 3.2 节零成本优势在训练后是否保持。
- **缩放定律**：研究不同后训练 token 量（20M–100B）下各方法的性能恢复曲线与缩放规律，确定线性化方法何时代价过高。
- 承接 6.1 节第 2–4 条：FA 混合层的影响、超大规模与 agentic 场景、多模态/视频扩散中的局部注意力。

### 6.3 结论

本文在 11 个基座模型（1.3B–70B）、6 项短上下文基准与 2 项长上下文基准上，首次将带 4 个 attention sinks 的 training-free SWA 与 12 种后训练线性注意力方法置于同一恢复率口径下对比，得到三个层次的结论：

1. **短上下文**：SWA(64, 4) 以 0 token 后训练、0 个后训练阶段恢复教师模型六基准平均性能的 99.0%（MMLU-5shot 恢复 93.2%，为 Table 1 全部 13 行方法中最高），在 11 组模型对比中 9 组取得最佳平均下游性能，优于或持平所有需要 20M–100B token 后训练的线性化方法。
2. **长上下文**：SWA 的优势进一步放大——S-NIAH 4K context 下 SWA 恢复全注意力准确率的 17.2–23%，而 LoLCATs 至多 5.8%、Liger-GLA 至多 0.8%；BABILong 4K 下 SWA 恢复基线的 25% 对 LoLCATs 的 5%；摘要级结论为长上下文任务上 SWA 准确率达后训练线性注意力的 2–10 倍。
3. **效率**：SWA 解码吞吐全程最高（快于 Linear、Linear+SWA 与各窗口配置的对比组），KV cache 内存增长至 window-size 后恒定，且无需后训练流程与专用 kernel。

据此作者给出强实践建议：为降低推理内存成本，应直接切换到 SWA with sinks 而非后训练线性模型——在 $w=64$、$s=4$ 的配置下，以封顶于 64 个 token 的固定 KV 内存与全程最高的解码吞吐，即可保留 99.0% 的平均短上下文性能，并获得长上下文上 2–10 倍于后训练线性注意力的准确率。
