> **论文**：Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning
> **作者**：Heng Wang, Jielin Qiu, Wenting Zhao, Cheng Qian, Liangwei Yang, Jiawei Han, Heng Ji（UIUC）；Silvio Savarese, Shelby Heinecke, Huan Wang（Salesforce AI Research）
> **arXiv ID**：2609.03430
> **发表时间**：2026-09-03
> **许可协议**：Apache-2.0（官方代码）
> **代码仓库**：https://github.com/SalesforceAIResearch/Random-Attention

## 第 1 章 概述

### 1.1 一句话定位

这篇论文直接检验了 KV cache 驱逐领域最核心的隐含前提——"用一个分数估计每个缓存 token 日后的重要性、再保住 top-K"是否真的有效——并证明**选择信号（selection signal）的贡献几乎为零**：只要保护 prompt、其余位置在每个 attention head 内做 i.i.d. 均匀随机驱逐（Random Attention），就能匹配乃至超越最强的启发式驱逐器，同时在 vLLM 部署下获得显著更高的吞吐。

### 论文图表总览

论文正文含 4 张主表（另含附录 Table 5–11），无独立 Figure 编号图形（图内嵌于各表所在小节）：

| 编号 | 内容 | 章节 |
|------|------|------|
| Table 1 | 主结果：3 模型 × 5 任务列（MATH500 / GPQA-D / AIME / HMMT / LiveCodeBench，共 6 任务含 AIME 2025/2026 合并；Qwen3-14B 在附录 Table 5），Random 与最强基线对比，60 组基线对比中 31 组显著领先 | 第 4 章 |
| Table 2 | matched prompt-protection 实验：给所有方法统一加上"prompt 全保"规则前后的精度 | 第 5 章 |
| Table 3 | passcode 植入探针：一次性罕见事实的检索率（R-KV 找回 84%，随机为 0） | 第 5 章 |
| Table 4 | vLLM 服务化吞吐：32k 生成长度下各方法 tok/s（比 TriAttention 快 32–43%） | 第 6 章 |
| 附录 Table 5–11 | Qwen3-14B 复现（T5）、matched protection 其余设置（T6）、生成长度与变异性（T7/T8）、驱逐轮耗时（T9）、等内存吞吐（T10/T11） | 附录/第 5-6 章 |

### 1.2 核心贡献

1. **反直觉发现**：KV 驱逐中的"选择信号"几乎无用。随机驱逐（保护 prompt + 每头均匀随机）在与 SnapKV / R-KV / VaSE / TriAttention 完全同预算的对比中，匹配乃至超越这些精心设计的启发式选择器。
2. **机制解释一（prompt 脆弱性）**：prompt 是 cache 中最脆弱的部分；方法间差距主要来自是否保住 prompt。matched-protection 实验（Table 2 / Table 6）显示，一旦所有方法都保护 prompt，大部分差距消失。
3. **机制解释二（推理轨迹的两级冗余自保护）**：推理轨迹在文本层（模型会复述自己需要的信息）和跨 attention head 层（每个 head 各持有一份副本）具有冗余；planted-fact 探针证明跨头副本的检索率是超加性的（superadditive）——单头几乎检不回，多头几乎总能检回。
4. **效率红利**：无打分 pass 使 Random Attention 在 vLLM 集成下比最强基线 TriAttention 快 32–43% tok/s（32k 生成长度），且无需校准、无需调参。
5. **方法论定位（零假设）**：Random Attention 既是可部署的方法，也是该领域的 null hypothesis——今后任何信号选择器都必须在 matched budget + matched prompt protection 的协议下击败它才有说服力。

### 1.3 关键结果速览

- **随机驱逐匹配最强基线**：4 个模型（Qwen3-4B/14B/32B、Phi-4-reasoning）× 6 个推理任务上，Random Attention 与最强启发式基线整体可比；在数学、科学类任务上表现尤为突出，竞赛数学（AIME/HMMT）上无人真正拉开差距（名义领先均在噪声内，30 题任务上 run-to-run 波动约 ±5 点）。
- **31/60 显著领先**：60 组基线对比（baseline comparisons）中，Random Attention 在 31 组显著领先，仅 1 组显著落后（TriAttention 在 Qwen3-32B code 任务上领先约 3 点）。
- **vLLM 吞吐 +32–43%**：32k token 生成长度下，比最强基线 TriAttention 每秒多产出 32–43% 的 token（Qwen3-14B/4B/32B 与 Phi-4 上分别为 +37%/+43%/+40%/+32%）；等内存比较下相对 Full attention 达到约 3–10× 吞吐，K=1024 高压缩下最高 28.8×。
- **两机制解释**：(1) prompt 是脆弱部分——所有方法都保 prompt 后，差距大部分消失（如 SnapKV 在 MATH500 K=1024 上 0.703→0.829，Recency 诊断基线 0.246→0.843）；(2) 推理轨迹通过两级冗余自我保护——planted-fact 探针显示一个事实保留在单个 head 内几乎无法检索（最佳单头仅 3%），保留在多个 head 中则几乎总能检索（两头 60%、三头 83%、八头 99%）。真正留给选择信号的，只剩"只陈述一次、从不复述的罕见事实"，而推理轨迹很少产生这种信息。

## 第 2 章 研究背景与动机

### 2.1 KV cache 瓶颈与推理模型的长 CoT 问题

推理模型（reasoning model）通过生成长思维链（chain-of-thought）解决难题，输出动辄上万 token。KV cache 随生成长度**线性增长**，在 32k 级生成下构成严重的显存瓶颈：它既限制 batch size（从而限制吞吐），也在长任务上直接顶爆显存。这推动了 KV cache eviction 研究线：维持一个固定预算 K 的缓存，每轮驱逐时丢弃"预计日后不重要"的条目。论文的实验设置即围绕这一现实：4 模型 × 6 任务，最长生成 32,768 token，预算约 4× 压缩（LiveCodeBench 约 3×）。

### 2.2 现有驱逐范式：打分 + 保 top-K

现有方法共享同一个模板：为每个缓存 token 计算重要性分数 $s(i)$，保住分数最高的 K 个。差异只在分数设计（§2 Preliminaries 逐一梳理）：

- **StreamingLLM**：无打分，直接保 recency 窗口 + 开头的 "attention sink" 位置——纯位置先验。
- **H2O**：累积注意力分数——用历史注意力质量作为重要性代理。
- **SnapKV**：只看最近 $w$ 个 query 的注意力（observation window），认为"最近被关注过的 token 日后也会被关注"。
- **R-KV**：在 SnapKV 式分数上叠加冗余惩罚——惩罚与已保留条目高度相似的候选，试图留住"信息多样性"。
- **VaSE**：value 范围——用 V 向量的取值范围作为重要性信号，无需 query。
- **TriAttention**：三角函数特征 + key 范数 $\|k\|$ 构造的分数，是当前最强（也最贵）的基线。

这条研究线的全部创造力都集中在"更好的分数"上。但论文指出：**从来没有工作直接检验过这个前提本身**——分数（选择信号）到底对压缩后的精度贡献了多少？

### 2.3 与稀疏注意力的区别

论文在 §2 明确区分两条容易混淆的技术线：

- **稀疏注意力选择（sparse-attention selection）**：每一步计算 attention 时动态挑选要参与的 KV（如 token 稀疏化 / query 感知选择），KV 仍完整存在，只是算得少——省的是计算。
- **KV cache 驱逐（eviction）**：把条目从 cache 中物理删除，后续步骤永久不可见——省的是显存。删除是不可逆的，这决定了驱逐必须依赖对未来重要性的估计，也正是"打分"存在的理由。

Random Attention 属于驱逐线：它同样维持预算 K（加 buffer r 的周期性驱逐协议），但把分数换成随机数。

### 2.4 研究问题

由此论文的核心研究问题是：**在受控同预算比较下，选择信号是否真的决定压缩后的精度？** 若答案是否定的，那么整条"设计更好分数"的研究线的边际价值需要重新评估，而免打分方法在效率上的优势（少一次遍历全 cache 的打分 pass）就成为纯粹的收益。这正是第 3 章 Random Attention 作为"零假设探测器"的动机。

## 第 3 章 Random Attention 方法

### 3.1 两个结构选择

Random Attention 的全部设计只有两条结构规则，不含任何可学习或可调的分数：

1. **保护 prompt**：prompt 的全部 KV 条目无条件保留，不参与驱逐（后续 §5.1 证明这是性能的真正来源）。
2. **其余位置每头 i.i.d. 均匀随机**：对生成部分（推理轨迹）的每个 token，在每个 attention head 内独立地、均匀随机地决定去留——各 head 的随机选择相互独立。

记序列由长度 $\ell_p$ 的 prompt 和生成部分 $\mathcal{G} = \{\ell_p+1, \dots, \ell\}$ 组成，缓存预算为 $K$（另有 buffer $r$ 的周期性驱逐协议：每生成 $r$ 个 token 触发一轮驱逐，回到预算 $K$）。

### 3.2 形式化

对照 §2 的通用"打分 + top-K"模板，可以把现有驱逐方法统一写成每轮驱逐时按分数保留 K 个条目：

$$
\mathcal{S}_h \;=\; \operatorname{TopK}\big(\,\{\,s_h(i)\,\}_{i \in \mathcal{G}},\; K\,\big)
$$

其中 $s_h(i)$ 是 head $h$ 上位置 $i$ 的重要性分数（SnapKV 用 observation window 注意力、VaSE 用 value 范围、TriAttention 用三角特征 + $\|k\|$，等等）。Random Attention 把这个分数替换为每头独立的均匀随机数：

$$
s_h(i) \;=\; u_{h,i}, \qquad u_{h,i} \;\overset{\text{i.i.d.}}{\sim}\; \mathrm{Uniform}(0,1)
$$

再加上 prompt 保护规则，最终每轮驱逐的保留集合为：

$$
\mathcal{S}_h \;=\; \mathcal{P} \;\cup\; \operatorname{TopK}\big(\{\,u_{h,i}\,\}_{i \in \mathcal{G}},\; K - |\mathcal{P}|\big)
$$

其中 $\mathcal{P}$ 为 prompt 的全部条目，$\cup$ 表示并集（prompt 条目优先占用预算）。

值得注意的是，虽然方法不含显式的位置先验，均匀随机 + top-K 的组合会自然产生一个**隐式的 soft recency 窗口**（附录 D 的 keep-log 分析）：一条已生成 $n$ 轮驱逐前、位于预算内深处的条目，每轮驱逐的存活概率约为 $(K-\ell_p)/(K+r-\ell_p)$，在 $K=1024$、$r=64$ 时约为 $0.94^n$——越老的 token 越可能已被换出，但并非硬截断。

### 3.3 每轮代价：1 rand + 1 topk

由于分数是均匀随机数，每轮驱逐的全部计算就是**一次随机数生成加一次 top-K 选择**。对照附录 Table 9 的实测：Random Attention 每轮驱逐耗时 0.30 ms（占 decode 时间 0.57%），而 TriAttention 需要 1.47–1.64 ms（2.49%/2.67%）。这个差距在服务化场景被急剧放大（§6）：vLLM 下每 request 每 64 token 压缩一次、一个 workload 约触发 62,000 次压缩，且压缩与生成之间有 barrier 同步——省掉打分 pass 省的是 62k 次的开销；PagedAttention 分页存储还让 content-dependent 的打分（要读 KV 内容）格外昂贵。

### 3.4 作为零假设的定位

论文刻意把 Random Attention 定位为双重身份：

- **可部署的方法**：无校准、无超参调优、免打分，vLLM 集成下吞吐最高。
- **领域的 null hypothesis**：任何新的信号选择器，必须在 matched budget + matched prompt protection 协议下显著击败随机驱逐，才能证明其选择信号真有价值。此前文献中"随机基线远落后于启发式"的结论，论文归因于协议差异——那些随机基线没有保护 prompt（§5.1：Recency 诊断基线在无 prompt 保护时 GPQA-D 低至 0.093，加上保护规则后逼近最优）。

### 3.5 与 StreamingLLM / H2O 的承接关系

Random Attention 在方法谱系上承接了两条早期脉络：它继承了 **StreamingLLM** 的"免打分"传统（StreamingLLM 用 recency 窗口 + sink 位置先验，同样不计算重要性分数），但把硬位置先验换成每头独立的均匀随机；同时它回应了 **H2O** 开启的"分数即一切"范式——H2O 用累积注意力作为分数，而本文的实验表明这类分数相对随机的增量几乎为零。可以说 Random Attention 是"回到 StreamingLLM 式的免打分路线，但用随机性替换位置先验，并第一次配上严格的 prompt 保护与同预算统计检验"。

## 第 4 章 实验设置与主结果

### 4.1 评估协议

论文在 4 个模型（Qwen3-4B、Qwen3-14B、Qwen3-32B、Phi-4-reasoning 14B）和 6 个推理任务上评估随机驱逐与信号式驱逐的差距，任务覆盖数学、科学与代码：

| 任务 | 题目数 | 每格独立采样次数 R | 性质 |
|------|:------:|:------:|------|
| MATH500 | 500 | 2 | 数学（Hendrycks 2021 / Lightman 2024） |
| GPQA-Diamond | 198 | 4 | 科学（Rein 2024） |
| AIME 2025 + 2026 | 各 30（合并报告） | 16 | 竞赛数学 |
| HMMT | 60 | 16 | 竞赛数学（经 MathArena） |
| LiveCodeBench-v6 medium | 383 | 4 | 代码（pass@1 真实执行） |

生成采用各模型官方采样设置（Qwen3 系 temperature 0.6，Phi-4-reasoning 0.8；nucleus p=0.95）。主网格将每个任务的预算固定在典型推理轨迹约 4 倍压缩（LiveCodeBench 约 3 倍压缩）；最大生成长度 32,768 tokens。所有方法用 FlashAttention-2 内核评估（无 PagedAttention）。统计显著性采用逐题聚类的 paired percentile bootstrap（95% CI）+ exact sign test，显著低于随机驱逐的格子置灰标注。

每个任务的每头预算 K：MATH500 K=1024，GPQA-D K=2048，AIME/HMMT K=4096，LiveCodeBench K=3072。

基线包括 SnapKV、R-KV、VaSE、TriAttention（全部在相同预算下按其发布配置运行），Full attention（不驱逐）作为上限。作为对照，无驱逐上限（Full）在各任务上的表现即表 1 首行——例如 Qwen3-4B 上 Full 达到 MATH500 0.939、GPQA-D 0.562、AIME 0.642、HMMT 0.462、LiveCodeBench 0.807。论文以最终框选答案是否正确为判定标准（沿用 Chang et al. 2026 与 Gao et al. 2026 的协议）。

### 4.2 主结果：随机驱逐在 60 个基线格中 31 格显著领先

Table 1 报告了 Qwen3-4B、Phi-4-reasoning、Qwen3-32B 三个模型的主网格结果。配对检验下，随机驱逐在表中 60 个基线单元里 31 个显著领先、1 个显著落后。

| 模型 / 方法 | MATH500 (K=1024) | GPQA-D (K=2048) | AIME (K=4096) | HMMT (K=4096) | LiveCodeBench (K=3072) |
|------|:---:|:---:|:---:|:---:|:---:|
| **Qwen3-4B** | | | | | |
| Full（上限） | 0.939 | 0.562 | 0.642 | 0.462 | 0.807 |
| SnapKV | 0.703 | 0.369 | 0.418 | 0.395 | 0.507 |
| R-KV | 0.810 | 0.482 | 0.494 | 0.371 | 0.712 |
| VaSE | 0.809 | 0.461 | 0.596 | 0.421 | 0.700 |
| TriAttention | 0.864 | 0.533 | 0.592 | 0.437 | 0.755 |
| **Random Attention（本文）** | **0.874** | **0.530** | **0.610** | **0.438** | **0.744** |
| **Phi-4-reasoning** | | | | | |
| Full（上限） | 0.922 | 0.707 | 0.677 | 0.444 | 0.697 |
| SnapKV | 0.844 | 0.442 | 0.502 | 0.343 | 0.314 |
| R-KV | 0.909 | 0.636 | 0.643 | 0.440 | 0.621 |
| VaSE | 0.853 | 0.562 | 0.520 | 0.354 | 0.373 |
| TriAttention | 0.891 | 0.684 | 0.633 | 0.431 | 0.652 |
| **Random Attention（本文）** | **0.910** | **0.678** | **0.662** | **0.430** | **0.667** |
| **Qwen3-32B** | | | | | |
| Full（上限） | 0.950 | 0.703 | 0.715 | 0.559 | 0.886 |
| SnapKV | 0.816 | 0.476 | 0.541 | 0.450 | 0.609 |
| R-KV | 0.857 | 0.638 | 0.613 | 0.472 | 0.779 |
| VaSE | 0.868 | 0.597 | 0.680 | 0.524 | 0.797 |
| TriAttention | 0.887 | 0.683 | 0.677 | 0.508 | 0.834 |
| **Random Attention（本文）** | **0.891** | **0.683** | **0.664** | **0.509** | **0.806** |

（来源：论文 Table 1；Qwen3-14B 的完整复现见论文 Appendix A Table 5，模式一致——随机驱逐在 MATH500 与 GPQA-D 显著超越 VaSE 和 SnapKV，中间规模不改变结论。）

中间规模 Qwen3-14B（论文 Appendix A Table 5）的结果如下，作为跨规模一致性验证：

| 模型 / 方法 | MATH500 (K=1024) | GPQA-D (K=2048) | AIME (K=4096) | HMMT (K=4096) | LiveCodeBench (K=3072) |
|------|:---:|:---:|:---:|:---:|:---:|
| Full（上限） | 0.951 | 0.645 | 0.706 | 0.557 | 0.856 |
| SnapKV | 0.812 | 0.506 | 0.460 | 0.467 | 0.622 |
| R-KV | 0.816 | 0.600 | 0.571 | 0.427 | 0.788 |
| VaSE | 0.852 | 0.543 | 0.668 | 0.495 | 0.813 |
| TriAttention | 0.891 | 0.625 | 0.654 | 0.493 | 0.843 |
| **Random Attention（本文）** | **0.870** | **0.628** | **0.642** | **0.505** | **0.820** |

在该规模上 3 个基线格显著领先：TriAttention 在 LiveCodeBench（+2.6 分，p=0.007）、TriAttention 在 MATH500（+2.1 分，p=0.02）、VaSE 在 AIME（+2.6 分，p=0.007），与其在 Qwen3-32B AIME 的名义优势一致。

**数学与科学推理上，选择信号买不到任何东西。** 在 MATH500 和 GPQA-D 上，随机驱逐在全部三个模型上显著超越 VaSE 与 SnapKV，并在 Qwen3-4B 上显著超越 R-KV。没有任何选择器在这些任务上显著击败随机驱逐：TriAttention 名义领先的 0.3-0.6 分（两个 GPQA-D 格）落在噪声范围内。

**竞赛数学上同样无人拉开差距。** AIME 与 HMMT 样本更少更难，比较噪声更大：16 次采样下 SnapKV 在每个模型上都显著落后于随机驱逐，R-KV（Qwen3 系）与 VaSE（Phi-4-reasoning）同理，但没有任何选择器显著领先。名义领先双向出现：VaSE 在 Qwen3-32B AIME 领先 1.7 分、HMMT 领先 1.5 分，而两种方法在 30 题集合上的 run-to-run 标准差即达 ±5 分。预算收紧后这些任务才开始分离，且方向对随机驱逐有利（见 §4.3）。

**代码推理上，多数信号式选择器因更长的 prompt 而崩溃。** LiveCodeBench 是唯一出现大差距的任务：SnapKV 在每个模型上输给随机驱逐 20-35 分；VaSE 在 Phi-4-reasoning 上崩至 0.373（落后 29 分），且在 3 个模型中的 2 个被置灰；R-KV 同样被置灰。TriAttention 与随机驱逐在 Qwen3-4B 与 Phi-4-reasoning 上打平，在 Qwen3-32B 上领先约 3 分——这是主网格中唯一的显著基线胜。原因在 prompt 长度：LiveCodeBench prompt 平均 557 tokens，是同 tokenizer 下 MATH500 的约 6 倍，最长者可消耗 K=3072 预算的近一半。因此未能保住 prompt 的选择器在代码任务上损失最大；第 5 章将证明保住 prompt 即可弥合 SnapKV 与 VaSE 的差距。随机驱逐本身固定钉住全部 prompt token，在代码任务上其预算在"选择"开始前就有很大一块被 prompt 占用——大量代码 prompt 是脚手架内容（I/O 格式、harness 指令），更聪明的规则或可压缩而非整段钉住，作者将此留给未来工作，因为随机驱逐作为零假设的价值恰在于无可调参数。

### 4.3 压缩压力：预算越紧，随机驱逐与最强基线的差距越大

Figure 2 给出了 Qwen3-4B 与 Phi-4-reasoning 在四个数学/科学任务上从 2 倍到 16 倍压缩的扫描。两个模型家族讲同一个故事：2 倍压缩时所有方法都贴近 Full attention；预算收紧后，随机驱逐始终与 TriAttention 持平，而两者对 VaSE 的领先逐步拉开。LiveCodeBench 未纳入扫描，因为其 prompt 在极小预算下无法容纳。

## 第 5 章 为什么选择信号买不到什么：两级冗余解释

推理场景的 KV cache 内容可分两类。prompt 只陈述一次、从不复述；工作状态（解依赖的中间推理步骤）随推理持续被写出与重写。第 5 章证明：前者脆弱，方法间的差异取决于是否保住它；后者冗余到随机抽取即可保住模型仍需要的内容。

### 5.1 prompt 是缓存中脆弱的部分

各方法对 prompt 的态度不同：TriAttention 默认保留全部输入；VaSE、R-KV、SnapKV 默认只保留 sink tokens 而把其余位置完全交给分数。因此跨论文比较同时也在比较"保护机制"（Chen et al. 2026）。论文通过给每个方法统一施加"保住 prompt"规则，将分数与保护解耦（Table 2）：

| 方法 | Qwen3-4B MATH500 分数单独 → +prompt | Qwen3-4B GPQA-D 分数单独 → +prompt | Phi-4 MATH500 分数单独 → +prompt | Phi-4 GPQA-D 分数单独 → +prompt |
|------|:---:|:---:|:---:|:---:|
| SnapKV | 0.703 → 0.829（+12.6） | 0.369 → 0.492（+12.3） | 0.844 → 0.889（+4.5） | 0.442 → 0.667（+22.5） |
| R-KV | 0.810 → 0.812（+0.2） | 0.482 → 0.471（−1.1） | 0.909 → 0.902（−0.7） | 0.636 → 0.655（+1.9） |
| VaSE | 0.809 → 0.812（+0.3） | 0.461 → 0.470（+0.9） | 0.853 → 0.895（+4.2） | 0.562 → 0.664（+10.2） |
| Recency window | 0.246 → 0.843（+59.7） | 0.093 → 0.519（+42.6） | 0.665 → 0.884（+21.9） | 0.323 → 0.658（+33.5） |
| **Random Attention** | 0.459 → 0.874（+41.5） | 0.231 → 0.530（+29.9） | 0.759 → 0.910（+15.1） | 0.434 → 0.678（+24.4） |

（来源：论文 Table 2。规则按每方法分数实际丢失多少 prompt 来补偿——逐轮日志见论文 Appendix D。）

统一规则后，三个基线在所有设置中彼此落在 2.2 分以内。Phi-4-reasoning 上它们还落在随机驱逐约 2 分以内（随机驱逐本身不做任何排序）；Qwen3-4B 上三者仍比随机驱逐低 4-6 分，而 R-KV 与 VaSE 本就在 Qwen3-4B 保住了大部分 prompt，因此几乎不从规则获益。代码推理（两个模型）、Qwen3-4B 竞赛数学与 32B GPQA-D（Appendix C Table 6）呈现相同模式：规则弥合了所有大且方法特定的差距，残余差距更小且朝随机驱逐有利方向。

两个无信号行从另一面印证同一结论。无规则时纯 recency 窗口把全部预算给近期轨迹、不给 prompt，分数可低至 0.093；随机驱逐在仅以均匀概率保 prompt 时跌至 0.231-0.760。加规则后两者都不再损失关键内容：随机驱逐在每个设置中都成为最佳策略，纯 recency 窗口也来到最佳基线 2 分以内。丢掉 prompt 是灾难性的，随机削减轨迹则不是——这正是 prompt 脆弱性的含义。该混杂也解释了过往文献中"随机保留远落后于信号式选择"的结论（Yuan et al. 2026; Liu et al. 2025）：那些随机基线差是因为丢了 prompt。

### 5.2 工作状态自我保护：跨头冗余与超加性池化

除文本级冗余（推理轨迹会复述仍在使用的值，Cai et al. 2025 已指出）外，还有第二层跨头冗余：每个 KV head 各自缓存每个 token 的副本，驱逐按 head 独立决策，只有全部 KV head 恰好都丢掉某 token 时它才真正丢失。

论文用 planted-fact 探针实验展示模型如何利用第二层冗余。合成事实（例如"令 z_q = 4729"，每次使用全新变量与值）被植入真实模型生成的 MATH500 推理轨迹，问题在事实植入 1,536 tokens 之后才出现——期间缓存被驱逐约 15 次。实验控制的是哪些 KV head 保留该事实：条件 = 选定一组 head 钉住事实 token，其余 head 全部驱逐事实，背景驱逐为标准的逐头均匀采样。两个指标衡量存活：Retrieval（贪心解码在问题处复现值的轨迹比例）与分级 recall R（公式 4，衡量保留副本相对"永不驱逐事实"的对数概率增益）。

探针揭示模型以两种方式利用冗余：

**副本跨 head 池化，且呈强超加性。** 注意力 head 存在分工：Qwen3-4B 的 8 个 KV head 中仅 3 个能单独保留可用的事实轨迹（与 Wu et al. 2025 的 retrieval-head 分工一致），且单独都很弱——最佳单 head 在 3% 试验中产出该值，次佳 1%。但读出不依赖任何单一 head：同样的两个 head 合起来在 60% 试验中产出事实，三个 head 达 83%，全部 8 个达 99%（Figure 3a）。池化因此是强超加性的，一对 head 的价值是各自单打独斗之和的数倍；它甚至跨事实生效——两个保存在不同 head、答案都需要用到的值，合起来 R=0.31，而各自单独仅 0.10 与 0.16（Figure 3b）。对驱逐策略的直接推论：只要**某些** head 保住副本，值就可继续使用——这正是独立逐头随机抽取所最大化的性质。在真实 MATH500 轨迹上这层额外覆盖甚至不需要：共享抽取（所有 head 用同一随机位置集）在 4 倍与 8 倍压缩下与随机驱逐仅差 0.3 分（Appendix D），因为文本级冗余已保住复述副本；跨头层承担的是文本未复述的内容，即探针所在的场景。

**副本形态无关紧要。** 将事实逐 token 打散摊到各 head，使任意两个连续 token 都不共享 head、任何 head 都无连续可读片段：retrieval 几乎不变（0.33 对完整句子的 0.39），recall 基本不变（R=0.75 对 0.76）。真实 MATH500 轨迹上同样不敏感：连续块从 1 增至 64 tokens 无损；块大小 256 才出现精度下降（此时预算每 head 只剩 4 或 2 个块，且 2 块更差）——重要的是每 head 块数而非块长（Figure 3c）。两发现合起来说明：答案取决于某个需要值的可用副本是否在**某处**存活，而非哪个副本、在哪个 head、以什么形态。

### 5.3 留给选择信号的残余空间：一次性事实

无信号策略唯一无法覆盖的情形：只陈述一次、从不复述、却在很久以后才需要的事实。论文在问题前 57 轮压缩时宣布一次口令（passcode），让各策略自行取舍，报告两个数字（Table 3）：

| 策略 | Retrieval | log p |
|------|:---:|:---:|
| Random Attention（本文） | 0.000 | −18.35 |
| VaSE | 0.344 | −3.88 |
| SnapKV | 0.004 | −11.11 |
| R-KV | 0.836 | −0.71 |
| TriAttention | 0.016 | −11.11 |

（来源：论文 Table 3。log p=0 表示模型必然产出正确口令；−18 表示口令实际上已从缓存消失。）

随机驱逐从未复现口令；retrieval 精确跟踪各选择信号所打分的注意力统计量——R-KV 的 importance 累加全历史注意力，84% 情况下找到口令；VaSE 的采样注意力约 1/3 找到；SnapKV 与 TriAttention 的近期窗口信号几乎从不。Wang (2026) 已证明无冗余时随机缓存必在 pointer-chasing 场景丢失信息。然而找针能力既非聚合精度的充分条件也非必要条件：R-KV 是最佳找针者，在 Table 1 却只领先一列；TriAttention 是主表最强基线，在此几乎一无所获。真实轨迹上该场景罕见，因为模型持续复述仍在使用的信息——但这是任何内容相关信号仍可增值的方向，也是后续研究应聚焦的地方。

## 第 6 章 效率评估与部署

### 6.1 vLLM 分页服务：比 TriAttention 快 32-43%

第一个设置遵循 TriAttention 协议在 vLLM + PagedAttention 下评估（仅对比 TriAttention，因为只有 TriAttention 与 R-KV 有 vLLM 端口，且 Mao et al. 2026 已证明同预算下 TriAttention 比 R-KV 高效）。两者在同一 H200 上以 K=2048、1k-token prompt、32k-token 生成、128 请求运行；Qwen3-32B 因权重更大、缓存池更小，压缩运行被限制为 96 并发请求，Full attention 无此限制：

| 方法 | Qwen3-4B | Phi-4-reasoning | Qwen3-14B | Qwen3-32B |
|------|:---:|:---:|:---:|:---:|
| Full（不驱逐） | 1296 tok/s（1.00×） | 780（1.00×） | 925（1.00×） | 346（1.00×） |
| TriAttention | 1494（1.15×） | 1212（1.55×） | 1303（1.41×） | 700（2.02×） |
| **Random Attention（本文）** | **2046（1.58×）** | **1737（2.23×）** | **1819（1.97×）** | **923（2.67×）** |
| 本文相对 TriAttention | +37% | +43% | +40% | +32% |

（来源：论文 Table 4。每请求 32k tokens 时限制 GPU 可容纳请求数的是 KV cache 而非算力，缓存越小并发越高。）

随机驱逐在四模型上达到 Full attention 吞吐的 1.6-2.7 倍、同一内核上比 TriAttention 高 32-43%。这些运行距容量平台约 7%：提供 512 请求仅使 Qwen3-4B 吞吐提升 7%、Qwen3-14B 变动 <1%；容量极限下对 TriAttention 的优势保持（+41%、+42%）。优势不依赖该工作点：短生成（压缩尚未回本）与轻负载下同样成立。

### 6.2 等内存比较：免打分 pass 值多少钱

第二个设置在 HuggingFace Transformers + FlashAttention-2（无分页）下给每种方法能装入单张 143 GB H200 的最大 batch（K=3072、32k 生成）：

| 方法 | Qwen3-4B batch | Qwen3-4B tok/s（×Full） | Qwen3-14B batch | Qwen3-14B tok/s（×Full） |
|------|:---:|:---:|:---:|:---:|
| Full（不驱逐） | 28 | 178（1.00） | 20 | 164（1.00） |
| SnapKV | 186 | 1624（9.14） | 111 | 1133（6.92） |
| R-KV | 186 | 1223（6.88） | 111 | 925（5.65） |
| VaSE | 190 | 1617（9.10） | 114 | 1162（7.10） |
| TriAttention | 182 | 670（3.77） | 109 | 483（2.95） |
| **Random Attention（本文）** | **200** | **1779（10.01）** | **120** | **1436（8.78）** |

（来源：论文 Appendix G Table 10。压缩缓存可容纳 109-200 条并发序列，Full 仅 28/20——3-10 倍加速主要来自容量。）

残余排序来自打分 pass：随机驱逐不算任何分数、装入最大 batch、峰值占用最小（101/89 GB），达到 Full attention 吞吐的 10.0 倍与 8.8 倍；16k 生成下排序不变。更紧的 MATH500 预算 K=1024 下同一协议达 28.8 倍（584 序列、5110 tok/s，比 SnapKV 多 16%、比 VaSE 多 20%——Table 11）。注意 TriAttention 行跑的是其 PyTorch 移植版（论文声明尚无优化内核），3 倍差距反映未融合实现而非方法本身；§6.1 的 vLLM 对比（用其发布内核）才是方法级比较。

单轮驱逐计时（Appendix G Table 9，K=1024、4096 解码步、空闲节点、单流）：随机驱逐 0.30 ms/轮（占解码 0.57%），SnapKV 0.37 ms、R-KV 0.58 ms、VaSE 0.74 ms、TriAttention 1.47-1.64 ms（占 2.49-2.67%）。

**为何免打分在服务场景值 32-43%？** 单独计时打分 pass 很便宜：单流下额外打分只占解码时间几个百分点。服务场景将该小成本放大两倍：其一，压缩不断堆积——128 并发请求每个每 64 tokens 压缩一次，vLLM 几乎每个解码步都在压缩某个请求（每工作负载约 62,000 次），且每次压缩发生在批量步间的同步点，128 个请求全部等待其中一个被压缩；其二，分页服务下内容相关打分比普通批量解码更贵——cache 统计类选择器需额外遍历分页 KV 状态，注意力权重类选择器必须重算或显式暴露注意力统计（融合内核不物化它们），随机驱逐两者都不需要、只做压缩本身。Qwen3-14B 上 TriAttention 的 32k 运行比随机驱逐多花 910 秒，相当于每次压缩约 15 ms 的整批等待，而随机驱逐每次远低于 1 ms。

### 6.3 代码实现与工程要点

官方实现（Apache-2.0）位于 https://github.com/SalesforceAIResearch/Random-Attention（29 stars），核心策略在代码中名为 `random_pp`。仓库包含驱逐引擎、评估 harness、显著性检验、效率基准、vLLM 移植与机制研究工具，即论文全部数字的产出代码；`data/`、`figures/`、`kvcompress/`、`scripts/` 为顶层目录。评估 harness 与引擎骨架继承自 VaSE（MIT），TriAttention 基线与 vLLM 基准构建于 TriAttention（Apache-2.0）。

把驱逐器移植到分页运行时的工程量与分数本身无关：物理驱逐须把每个请求幸存的 KV 重写进其 block、释放尾部、在物理槽收缩时保持 rotary 位置逻辑一致、并在 preemption 下保持正确。R-KV 的发布移植横跨 13 个上游文件约 849 行接线并依赖 vLLM V1 model runner；分数决定还需要什么：读 cache 的分数（VaSE 的 value range、TriAttention 的校准 key 统计）需跨 block table 的 gather；读注意力权重的分数（SnapKV、R-KV、VaSE 的注意力比例填充）在融合分页内核中根本读不到——权重从不物化，必须重算为显式 window-query 乘积（chunked，因瞬时量达 heads×window×cache）或改注意力内核本身。随机驱逐的 keep-set 是槽位索引的随机排列，什么都不读，运行时现有压缩路径即全部集成——加入 TriAttention 插件只花了一个函数。

仓库还记录了两个会静默测错东西的工具陷阱：vLLM 自带吞吐基准忽略请求的输出长度（默认 128-token 输出达不到压缩阈值，实际在测 Full attention 解码）；集成的去重守卫在一次良性欠预算轮后可能禁用后续所有压缩。论文用不影响选择与内核语义的记账修改修正两者，每个报告运行均由 applied-event 计数器验证。

分页表格与等内存表格存在预算差异（K=2048 与 K=3072）；32k 点在 K=3072 重跑保持排序（Qwen3-4B/14B：随机驱逐 2011/1696 tok/s 对 TriAttention 1437/1223，+40%/+39%），预算差异不驱动任一比较。

## 第 7 章 局限性与延伸

### 7.1 代码任务的长 prompt 占预算问题

LiveCodeBench 是全部 6 个任务中唯一存在大差距的任务，论文将原因归于 **prompt 长度**：LCB 的 prompt 平均 557 token，约为 MATH500 的 6 倍，最长的可占掉 K=3072 预算的一半。当 prompt 占据预算的大头时：(a) 留给推理轨迹的余量被压缩；(b) "保护 prompt"这一随机方法唯一的规则本身变成了性能约束。主表中唯一的显著基线胜利正发生在这里——TriAttention 在 Qwen3-32B code 上领先约 3 点。延伸方向上，这提示**自适应预算分配**（prompt 与生成部分如何分摊 K）是比"更好的分数"更值得投入的轴；并发工作 Prefix Sliding（recency + prompt 且探索训练时融入）也在同一方向上。

### 7.2 罕见一次性事实：留给选择信号的窄缝

§5.3 的 passcode 探针（把一个一次性事实植在 57 轮驱逐之前）划出了随机驱逐的真实失败域：Random Attention 检索率为 0.000（logprob −18.35），而 R-KV 凭其冗余惩罚设计找回 84%（VaSE 0.344、SnapKV 0.004、TriAttention 0.016）。也就是说，"只陈述一次、从不复述的罕见事实"确实是选择信号的价值所在——只是推理轨迹很少产生这种信息（两级冗余意味着重要信息通常被复述、被多头复制）。此外论文引用 Wang (2026) 指出随机 cache 在 pointer-chasing 类任务上必然失败。延伸问题：**needle-finding 能力 ≠ aggregate strength**——长程单点依赖的 agent / 工具调用场景（文件句柄、变量名回指）可能是随机驱逐需要让位给信号方法的边界，值得单独评估。

### 7.3 matched protection 后的残留差距

prompt 保护解释了大部分差距但不是全部（附录 C）：Qwen3-32B GPQA-D 上仍有 4–6 点残差；AIME 上 SnapKV / R-KV / VaSE 分别差 5.4 / 11.1 / 3.5 点。这说明在竞赛数学的高压缩档位，选择信号仍有可测的（尽管不大的）贡献空间。

### 7.4 系统与工程局限

- **并发 vLLM port 未合入上游**：官方仓库提供的是独立 vLLM port，而非合入 vLLM 主干。且 preemption（并发抢占）会导致压缩状态丢失，实践中必须 cap 并发（224 @ 4B/14B、96 @ 32B @ 8k 生成）——这是部署上的真实约束，牺牲了部分弹性调度。
- **短生成场景不回本**：1k 输入 / 8k 输出的短生成下，Random Attention 仅达 Full attention 吞吐的 0.52×/0.70×/0.96×（4B/14B/32B）与 0.76×（Phi-4）——驱逐本身有开销，收益要到长生成才显现。
- **测量工具链陷阱**：论文报告了两个工具坑——vLLM 自带 bench 忽略 output length、dedup guard 会禁用后续压缩——后来者复现时需规避。

### 7.5 统计与适用范围

竞赛数学任务（AIME 每组仅 30 题）的 run-to-run 波动约 ±5 点，名义领先多在噪声内；论文用 paired problem-clustered bootstrap 95% CI + exact sign test 控制了这一点，但读者解读单格数字时仍需谨慎。更宏观地，本文结论限定于"推理模型的长 CoT 场景"——正是两级冗余成立的场景；将其外推到非推理生成（摘要、翻译、对话）前，冗余假设需要重新验证。
