# Jet-Long: Efficient Long-Context Extension with Dynamic Bifocal RoPE — 深度精读报告

> 论文编号：arXiv:2607.07740 | 机构：NVIDIA Research | 年份：2026
> 作者：Haozhan Tang, Zerui Wang, Yuxian Gu, Song Han, Han Cai

---

## 第一章 论文概述与核心贡献

### 1.1 论文元数据

| 项目 | 内容 |
|------|------|
| 标题 | Jet-Long: Efficient Long-Context Extension with Dynamic Bifocal RoPE |
| 作者 | Haozhan Tang, Zerui Wang, Yuxian Gu, Song Han, Han Cai |
| 机构 | NVIDIA Research |
| arXiv ID | 2607.07740 |
| 提交日期 | 2026-07-10 |
| 分类 | cs.LG, cs.AI |
| 源码 | https://github.com/jet-ai-projects/jet-long；Jet-Nemotron 发布页：https://github.com/NVlabs/Jet-Nemotron（论文未给出 star 数） |

### 1.2 论文速览

Jet-Long 是 NVIDIA 于 2026 年提出的一种**无需微调的零样本（zero-shot / tuning-free）长上下文扩展方法**，核心创新是 **Dynamic Bifocal RoPE（动态双焦旋转位置编码）**。该方法无需任何额外训练或微调，即可将原本在 32K 上下文长度上训练的大语言模型稳定外推到 128K 甚至更长的上下文，**同时保持接近原生注意力机制的推理吞吐量（prefill 为 FA3 的 83–88%，8B decode 为 FA3 的 1.04–1.08×）**。

论文从**算法设计**（动态分组外推）→ **数值合并**（容斥原理）→ **工程实现**（CuTe 融合核）三个层面构成了完整的技术栈：

1. **算法** — 把位置编码分组映射改造为**随输入长度自适应**的双焦机制，近场保持原始高分辨率，远场按动态分组系数 G 压缩到训练分布内
2. **数值** — 基于容斥原理的精确注意力合并，消除近/远场重叠区域的重复计数偏差
3. **工程** — 融合 CuTe kernel 将全部计算下沉到 GPU 内部；融合实现相对未融合版本大幅提速，但 prefill 仍为 FA3 的 0.83–0.88×、匹配 CuTe 控制的 0.88–0.93×，8B decode 为 FA3 的 1.04–1.08×

### 1.3 核心贡献总结

1. **Dynamic Bifocal RoPE**：局部窗口保留原始 RoPE（精确位置感知），远程窗口使用**动态分组系数** G = max(1, ceil(L / w_pretrained))，使远程位置始终落在训练分布内
2. **容斥注意力合并（Inclusion-Exclusion Attention Merge）**：通过三次 FlashAttention 调用的精确数值组合，实现距离感知路由，无需布尔掩码矩阵
3. **在线 RoPE 校正旋转（On-the-Fly RoPE Correction Rotation）**：KV cache 仅存储未压缩的原始位置编码，通过即时旋转变换得到外推向量；生成时新 token 的 key 仍照常写入缓存，校正旋转仅避免重写已缓存的历史 key，也无需重算历史 hidden states
4. **融合 CuTe 核**：将双焦计算融合为单个 CuTe 实现（论文未明确是否为单次 kernel launch），在 64K–128K 区间 prefill 为 FA3 的 0.83–0.88×、匹配 CuTe 控制的 0.88–0.93×，8B decode 为 FA3 的 1.04–1.08×
5. **主要基线领先**：在 Qwen3-1.7B/4B/8B 的 RULER 与 PG-19 全规模上最优，HELMET-RAG 在 4B/8B 最优（1.7B 落后 Self-Extend 0.73 pp），并泛化到混合注意力架构 Jet-Nemotron

### 1.4 关键结果速览

| 指标 | 结果 |
|------|------|
| RULER 领先幅度（vs 次优基线） | **+4.79 pp** (1.7B) / **+2.18 pp** (4B) / **+2.03 pp** (8B) |
| HELMET-RAG 最优结果 | 42.28% (1.7B) / 53.61% (4B) / **57.34%** (8B) |
| PG-19 Perplexity（越低越好） | 11.72 (1.7B) / 9.85 (4B) / **8.73** (8B) |
| 128K 处 PG-19 ppl (vs Base 8B) | **8.51** (vs Base 79.37 — 崩溃) |
| 多键检索 MK2 1.7B (vs DNTK) | **61.80%** (vs 16.40% — 领先 45.40 pp) |
| 多键检索 MK3 8B (vs Self-Extend) | **70.80%** (vs 39.80% — 领先 31.00 pp) |
| Prefill 吞吐量 (64K–128K) | FA3 的 **0.83–0.88×**；匹配 CuTe 控制的 0.88–0.93× |
| Generation 吞吐量 (8B, 64K–128K) | FA3 的 **1.04–1.08×**；与 FA2 差距在 3.6% 以内 |
| Jet-Nemotron-2B RULER 提升 | 42.93 → **52.94** (+10.01 pp) |
| Jet-Nemotron-4B RULER 提升 | 42.16 → **53.47** (+11.31 pp) |

### 1.5 图表总览

| 图表编号 | 类型 | 内容描述 | 对应章节 |
|---------|------|---------|---------|
| Figure 1 | 结果图 | Qwen3-1.7B-Base 的 RULER 准确率与 PG-19 perplexity 随上下文长度变化 | Introduction / Section 4.2（效率比较见 4.8） |
| Figure 2(a) | 示意图 | 动态双焦位置映射（局部窗口 + 远程映射 f(k)=⌊k/G⌋） | Section 3.1 |
| Figure 2(b) | 示意图 | KV Cache 不变性：在线校正旋转的流程 | Section 3.2 |
| Figure 2(c) | 示意图 | 容斥合并的三次 FA 调用示意图 | Section 3.3 |
| Figure 3 | 折线图 | RULER 准确率 vs 序列长度（Base/YaRN/DNTK/Jet-Long；DCA/Self-Extend 见 Table 1） | Section 4.2 |
| Table 1 | 主结果表 | RULER / HELMET-RAG / PG-19 六方法全对比 | Section 4.2 |
| Table 13 | 附录表 | PG-19 perplexity 分长度剖析（4K–128K，Appendix G.1） | Section 4.2 |
| Table 14 | 附录表 | RULER 13项子任务逐项准确率（64K，Appendix G.2） | Section 4.2 |
| Table 15 | 附录表 | Jet-Nemotron 混合架构扩展结果（含 96K，Appendix G.3） | Section 4.5 |
| Table 16 | 附录表 | w0 局部窗口消融（Qwen3-4B/8B，含 96K，Appendix G.4） | Section 4.6 |
| Table 6 | 数据表 | 位置别名 vs 频率插值对比（Qwen3-1.7B） | Section 4.7 |
| Table 7 | 数据表 | 推理吞吐量：Prefill + Generation（8B 详细） | Section 4.8 |
| Table 8 | 附录表 | Meta-Llama-3-8B-Base 跨家族 RULER（Appendix C） | Appendix C |
| Table 9 | 附录表 | post-trained Qwen3 RULER（Appendix D） | Appendix D |
| Table 17 | 附录表 | 完整推理效率（8B selected measurements，Appendix H） | Appendix H |
| Table 18 | 附录表 | 四种 Baseline 的配置参数（Appendix I） | Appendix I |

---

## 第二章 研究背景与动机

### 2.1 长上下文 LLM 部署挑战

现代大语言模型越来越广泛地部署在需要长上下文的场景中：

- **检索增强生成（RAG）**：从大规模文档库中检索相关片段，拼接后输入模型
- **代码仓库级理解**：对完整代码库的分析、修改与重构
- **Agent 工作流**：多轮推理、规划、工具调用轨迹的累积

这些场景的输入长度经常达到 **100K+ tokens**，远超模型预训练窗口（通常为 4K–32K）。直接在长上下文上训练成本极高——FlashAttention 等高效核虽使内存线性化，但**计算仍随序列长度二次增长**；长上下文数据稀缺，且微调往往损害短上下文性能。因此，**零样本上下文扩展（zero-shot context extension）** 成为开源权重模型的主要部署路径。

### 2.2 RoPE 位置编码的外推瓶颈

当前主流 LLM（Qwen、LLaMA、Mistral 等）广泛使用 **Rotary Position Embedding（RoPE）**，通过对 query 与 key 向量施加位置相关的旋转来注入相对位置信息：

$$
\theta_i = 10000^{-2i/d}
$$

位置 m 在第 i 个二维子空间上的旋转角为 $m\theta_i$。RoPE 存在两个导致外推失败的根本原因：

**(i) 位置分布外（Position OOD）**：推理时 m 超出训练长度 $L_{train}$（这里为 32K），低频维度的旋转角度 $m\theta_i$ 处于训练时从未出现的角度区间，导致 attention 得分行为异常。

**(ii) 注意力扩散与位置偏差（Attention Diffusion）**：key 集增大后，softmax 分布平坦化，概率质量分散到无关 token；同时模型呈现 U 形位置注意力偏差，中间位置的证据被系统性忽视。

例如，Qwen3-8B Base 在 128K 处 PG-19 perplexity 飙升至 **79.37**（而 4K 处仅为 8.64），外推失效极为严重。

### 2.3 现有零样本扩展方法及其不足

| 方法类别 | 代表方法 | 核心思路 | 根本局限 |
|---------|---------|---------|---------|
| 频率缩放 | NTK/DNTK, YaRN | 增大 RoPE base 或按维度分段缩放频率 | 固定缩放因子：激进则短上下文受损，保守则长上下文失效 |
| 分组位置 | Self-Extend | 近场精确 + 远场分组共用位置索引 | 分组大小 G 固定，无法自适应 |
| 分块注意力 | DCA | 将序列分成训练长度内的块，块间使用非对称索引 | 块大小固定，超参敏感 |

**共同局限**：所有现有方法都使用**单一固定缩放因子**，在短上下文保真度与长上下文覆盖之间被迫取舍。

---

## 第三章 Jet-Long 方法详解

### 3.1 动态双焦分解（Dynamic Bifocal Decomposition）

Jet-Long 采用双窗口（双焦）分解架构：一个局部窗口（大小 $w_0$）保留经典 RoPE，一个远程窗口使用改写的位置映射 $f(\cdot)$ 将位置映射回训练范围内。

对于位置 q 的 query 和位置 k 的 key，预 softmax attention 得分为：

$$
S(q,k) = \begin{cases} \text{RoPE}(\mathbf{x}_q, q)^\top \text{RoPE}(\mathbf{x}_k, k) & \text{if } q-k \leq w_0, \\ \text{RoPE}(\mathbf{x}_q, f(q))^\top \text{RoPE}(\mathbf{x}_k, f(k)) & \text{if } q-k > w_0. \end{cases}
$$

当 $L \leq w_{pretrained}$ 时，$f(x)=x$，上式退化为未修改的基础模型。

### 3.2 动态外推因子

Jet-Long 的核心创新在于**分组系数 G 随当前序列长度 L 动态自适应**：

$$
G = \max\left(1,\; \left\lceil\frac{L}{w_{pretrained}}\right\rceil\right)
$$

其中 $w_{pretrained}=32768$（32K）是 Qwen3 系列的预训练窗口。远程位置映射采用底除操作：

$$
f(x) = \left\lfloor\frac{x}{G}\right\rfloor
$$

| 输入长度范围 | G | 有效远程位置数 | 说明 |
|------------|-----|--------------|------|
| L ≤ 32K | 1 | 与原始相同 | 等价原模型，无外推 |
| 32K < L ≤ 64K | 2 | ≤ 32K | 2 个真实位置共享 1 个位置 ID |
| 64K < L ≤ 96K | 3 | ≤ 32K | 3 个真实位置共享 1 个位置 ID |
| 96K < L ≤ 128K | 4 | ≤ 32K | 4 个真实位置共享 1 个位置 ID |

相比 Self-Extend（固定 G=8），动态 G 在短输入时保持高位置分辨率，长输入时自动增大分组，**最大化任意序列长度下的位置精度**。

### 3.3 KV Cache 管理与在线校正旋转

动态外推面临一个缓存管理挑战：如果 G 在生成过程中变化，重写 KV cache 将需要丢弃并重新计算整个缓存。

Jet-Long 通过严格的不变量避免这一开销——**缓存只存储未压缩的 base key 及其精确绝对位置 k**：

$$
\mathbf{k}_{cache}[k] = \text{RoPE}(\mathbf{x}_k, k)
$$

当远程窗口需要位置 $f(q)$ 的 query 和位置 $f(k)$ 的 key 时，不是重新计算向量，而是通过即时校正旋转应用位置偏移：

$$
\Delta q = f(q) - q = \left\lfloor\frac{q}{G}\right\rfloor - q, \qquad 
\Delta k = f(k) - k = \left\lfloor\frac{k}{G}\right\rfloor - k
$$

利用 RoPE 的角度可加性 $R_a R_b = R_{a+b}$：

$$
\mathbf{q}_{remote} = \text{RoPE}(\mathbf{q}_{local}, \Delta q), \qquad
\mathbf{k}_{remote} = \text{RoPE}(\mathbf{k}_{cache}[k], \Delta k)
$$

这步操作是**常数时间**的，不对已缓存的历史 key 做重写（新 token 仍照常追加），因此生成可以在长度边界上流畅运行。

### 3.4 容斥预填充（Inclusion-Exclusion Prefill）

为避免物化完整的二次注意力矩阵（在长序列预填充时耗尽内存），Jet-Long 将**容斥原理**与 FlashAttention 的 LogSumExp 统计量结合。

合并需要三次标准 attention 调用：
- **A**：滑动窗口注意力（局部 $w_0$），使用 base query 和 base key
- **B**：完整因果注意力，使用远程 query 和远程 key
- **C**：滑动窗口注意力（局部 $w_0$），使用远程 query 和远程 key

调用 B 和 C 对局部 key 应用相同的远程旋转。最终输出通过稳定化处理合并：

$$
\mathbf{O}_{final} = \frac{W_A \mathbf{O}_A + W_B \mathbf{O}_B - W_C \mathbf{O}_C}{W_A + W_B - W_C}
$$

其中权重 $W_X = \exp(\boldsymbol{\ell}_X - M)$，$M = \max(\boldsymbol{\ell}_A, \boldsymbol{\ell}_B, \boldsymbol{\ell}_C)$。这种合并实现了精确的基于距离的路由，无需布尔掩码矩阵，保留了 FlashAttention 的内存效率。

### 3.5 融合 CuTe 核实现

为降低三次 FA 调用的执行开销，作者基于 NVIDIA CuTe（CUTLASS Tensor primitives）给出融合实现（论文未明确说明是否为单次 kernel launch）：

- 分组位置计算 $\lfloor x/G \rfloor$ 在 kernel 内部完成
- 在线旋转与容斥合并融入 FlashAttention 的 tiling 流水线
- 预填充为 FA3 的 0.83–0.88×、匹配 CuTe 控制的 0.88–0.93×（64K–128K 区间）
- 8B 逐 token 生成达到 FA3 的 1.04–1.08×

---

## 第四章 实验结果与分析

### 4.1 实验设置

**骨干模型**：

| 模型 | 参数量 | 训练窗口 | 扩展目标 |
|------|--------|---------|---------|
| Qwen3-1.7B-Base | 1.7B | 32K | 128K |
| Qwen3-4B-Base | 4B | 32K | 128K |
| Qwen3-8B-Base | 8B | 32K | 128K |

**评测基准**：
- **RULER**：综合长上下文能力，13 项子任务（多跳检索、聚合、变量追踪等）
- **HELMET-RAG**：HELMET 研究中最高效（most efficient）的下游长上下文性能预测器
- **PG-19**：长文档语言建模 perplexity

**基线方法**：Base（不扩展）、DNTK、YaRN、DCA、Self-Extend

**硬件**：NVIDIA H100 Tensor Core GPU

### 4.2 主结果（Table 1 — RULER / HELMET-RAG / PG-19）

| 方法 | RULER 1.7B | RULER 4B | RULER 8B | HELMET-RAG 1.7B | HELMET-RAG 4B | HELMET-RAG 8B | PG-19 ppl 1.7B | PG-19 ppl 4B | PG-19 ppl 8B |
|------|:----------:|:--------:|:--------:|:---------------:|:--------------:|:--------------:|:--------------:|:-------------:|:-------------:|
| Base | 60.93 | 69.94 | 73.13 | 36.20 | 44.33 | 47.24 | 16.13 | 14.84 | 12.80 |
| DNTK | 69.14 | 79.75 | 83.54 | 41.27 | 50.81 | 55.55 | 12.60 | 10.78 | 9.13 |
| YaRN | 52.99 | 70.11 | 78.49 | 32.24 | 43.63 | 53.41 | 16.39 | 12.08 | 9.91 |
| DCA | 67.80 | 80.19 | 81.08 | 41.77 | 51.91 | 56.12 | 11.77 | 9.89 | 8.77 |
| Self-Extend | 67.86 | 80.84 | 84.71 | 43.01 | 52.98 | 56.86 | 11.85 | 9.95 | 8.81 |
| **Jet-Long** | **73.93** | **83.02** | **86.74** | **42.28** | **53.61** | **57.34** | **11.72** | **9.85** | **8.73** |

Jet-Long 在 **8/9 个指标上取得最优**（RULER 全规模、PG-19 全规模、HELMET-RAG 4B/8B），仅 HELMET-RAG 1.7B 次优（落后 Self-Extend 0.73 pp）。RULER 领先次优方法幅度为：

$$
\Delta = +4.79\ \text{pp (1.7B vs DNTK)},\quad +2.18\ \text{pp (4B vs Self-Extend)},\quad +2.03\ \text{pp (8B vs Self-Extend)}
$$

在 PG-19 上，Jet-Long 与 DCA/Self-Extend 处于同一水平，并无显著额外提升——这符合预期，因为 perplexity 主要测量整体语言建模质量，而非长距离检索能力。

### 4.3 PG-19 分长度剖析（Table 13 — Appendix G.1）

| 方法 | 4K | 8K | 16K | 32K | 64K | 96K | 128K | 均值 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|
| Base 1.7B | 11.52 | 11.82 | 11.59 | 12.49 | 12.60 | 17.99 | 71.00 | 16.39 |
| Jet-Long 1.7B | 11.52 | 11.82 | 11.59 | 12.49 | 11.53 | 11.65 | 11.41 | 11.71 |
| Base 4B | 9.74 | 9.90 | 9.75 | 10.45 | 10.40 | 21.44 | 104.66 | 15.64 |
| Jet-Long 4B | 9.74 | 9.90 | 9.75 | 10.45 | 9.67 | 9.81 | 9.62 | 9.85 |
| Base 8B | 8.64 | 8.77 | 8.61 | 9.25 | 9.18 | 19.21 | 79.37 | 13.56 |
| Jet-Long 8B | 8.64 | 8.77 | 8.61 | 9.25 | 8.63 | 8.71 | 8.51 | 8.73 |

关键观察：
- **32K 以内**：Jet-Long 与 Base 数值完全相同（G=1，动态因子退化为恒等映射）
- **32K 以外**：Base 的 perplexity 呈指数级恶化（8B 在 128K 处 79.37），而 Jet-Long 保持平坦（8B 在 128K 处 8.51，**低于 4K 处的 8.64**）
- Jet-Long 在各个模型规模上都优于所有基线方法

### 4.4 逐任务剖析（Table 14 — RULER 13 sub-tasks at 64K，Appendix G.2）

| 任务 | 缩写 | 描述 | Jet-Long 1.7B | DNTK 1.7B | Jet-Long 8B | Self-Extend 8B |
|------|:----:|------|:-------------:|:----------:|:-----------:|:--------------:|
| Single Query NIAH 1 | S1 | 单针检索（深） | 100.00 | 100.00 | 100.00 | 100.00 |
| Single Query NIAH 2 | S2 | 单针检索（中） | 100.00 | 93.00 | 100.00 | 99.80 |
| Single Query NIAH 3 | S3 | 单针检索（浅） | 100.00 | 99.40 | 100.00 | 99.80 |
| Multi-Key NIAH 1 | MK1 | 多键检索（1键） | 93.20 | 78.40 | 94.20 | 87.40 |
| Multi-Key NIAH 2 | MK2 | 多键检索（2键） | **61.80** | 16.40 | **92.20** | 57.80 |
| Multi-Key NIAH 3 | MK3 | 多键检索（3键） | 1.60 | 1.60 | **70.80** | 39.80 |
| Multi-Value NIAH | MV | 多值检索 | 84.45 | 81.60 | 96.60 | 96.55 |
| Multi-Query NIAH | MQ | 多查询检索 | 82.25 | 81.05 | 92.15 | 94.05 |
| Variable Tracking | VT | 变量追踪 | 73.56 | 54.24 | 98.60 | 99.52 |
| Common Word Extraction | CWE | 常见词提取 | 0.32 | 0.38 | 47.32 | 41.44 |
| Frequent Word Extraction | FWE | 频繁词提取 | 75.47 | 70.93 | 87.53 | 81.60 |
| QA 1 | QA1 | 上下文问答（硬） | 39.40 | 41.20 | 56.00 | 59.00 |
| QA 2 | QA2 | 上下文问答（软） | 28.00 | 26.80 | 39.80 | 39.60 |
| **均值** | | | **64.62** | 57.31 | **82.71** | 76.64 |

最突出的差距出现在**多键检索（MK2/MK3）**：
- MK2 1.7B：Jet-Long **61.80** vs DNTK **16.40**（领先 45.40 pp）
- MK3 8B：Jet-Long **70.80** vs Self-Extend **39.80**（领先 31.00 pp）

多键检索要求模型精确定位并同时检索多个分散在上下文中的证据，对位置编码的外推保真度要求极高。Jet-Long 的**动态 G + 容斥合并**组合在此类任务上将差距放大到极致。

### 4.5 混合架构泛化：Jet-Nemotron（Table 15 — 含 96K，Appendix G.3）

| 方法 | 4K | 8K | 16K | 32K | 64K | 96K | 128K | **均值** |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:-------:|
| Jet-Nemotron-2B | 82.35 | 68.93 | 61.21 | 46.59 | 20.34 | 12.52 | 8.54 | 42.93 |
| **+ Jet-Long** | 82.35 | 68.93 | 61.21 | 46.58 | **40.33** | **37.40** | **33.78** | **52.94** |
| Jet-Nemotron-4B | 84.08 | 69.24 | 62.35 | 48.87 | 17.48 | 7.44 | 5.65 | 42.16 |
| **+ Jet-Long** | 84.08 | 69.24 | 62.35 | 48.88 | **39.93** | **36.69** | **33.14** | **53.47** |

32K 以内 Jet-Long 退化为原始 attention（数值在舍入误差范围内一致，差异 ≤0.01 pp）。32K 以上，裸混合基线的 RULER 崩溃至 8.54/5.65（128K），而 Jet-Long 保留了 33.78/33.14，**分别提升 +25.24 pp 和 +27.49 pp**。在七个长度的均值上，Jet-Long 将 2B 的 RULER 从 42.93 提升至 52.94（**+10.01 pp**），将 4B 从 42.16 提升至 53.47（**+11.31 pp**）。这表明 Jet-Long 可应用于与其他注意力架构（线性注意力等）交织的混合结构；实验仅与未扩展模型比较，未证明其优于其他混合架构扩展方法。

---

## 第五章 消融实验与分析

### 5.1 局部窗口大小 w0 消融（Table 16 — 含 96K，Appendix G.4）

局部窗口 $w_0$ 是 Jet-Long 唯一的超参数，控制近场精细位置的范围。消融实验在 Qwen3-4B/8B 上进行，覆盖 $w_0 \in \{0, 256, 512, 1024, 2048, 4096, 8192\}$：

| 长度 | w0=0 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|:----:|:----:|:---:|:---:|:----:|:----:|:----:|:----:|
| **Qwen3-4B** | | | | | | | |
| 64K | 4.10 | 76.65 | 76.63 | 76.49 | 76.57 | 75.91 | 74.69 |
| 96K | 2.07 | 70.71 | 71.19 | 70.77 | 71.09 | 70.57 | 69.11 |
| 128K | 1.08 | 64.13 | 65.63 | 65.57 | 64.74 | 64.06 | 63.52 |
| **Qwen3-8B** | | | | | | | |
| 64K | 4.91 | 83.51 | 83.23 | 83.31 | 83.24 | 82.71 | 81.40 |
| 96K | 2.54 | 74.29 | 74.94 | 75.20 | 75.49 | 74.84 | 73.82 |
| 128K | 1.55 | 69.48 | 70.49 | 70.68 | 71.01 | 71.03 | 69.65 |

关键发现：
- **w0=0 严重退化**（局部分支仅剩自注意力、远程分组注意力仍启用），RULER 降至 1.08–4.91，验证了局部窗口的必要性
- **所有 w0 ∈ {512, 1024, 2048, 4096} 均在单长度最佳值的 2 pp 以内**，在模型均值上在 1 pp 以内
- **w0=8192 轻微退化**（1.4–2.1 pp），因为局部窗口消耗了更大的上下文比例
- 实践者可以固定 w0=2048 无需逐部署调整

### 5.2 频率插值 vs 位置别名（Table 6）

论文区分了将远程位置保持在训练分布内的两种策略：

- **位置别名（Aliasing，默认 Jet-Long）**：$f(x) = \lfloor x/G \rfloor$，将多个真实位置映射到同一个整数位置 ID
- **频率插值（Frequency）**：保留原始位置索引，但缩放 RoPE 频率（类似 YaRN style）

在 Qwen3-1.7B-Base 上对比：

| 策略 | S1 | S2 | S3 | MK1 | MK2 | MK3 | MV | MQ | VT | CWE | FWE | QA1 | QA2 | **Avg 64K** |
|:----:|:--:|:--:|:--:|:---:|:---:|:---:|:--:|:--:|:--:|:---:|:---:|:---:|:---:|:----------:|
| Aliasing | 100.00 | 100.00 | 100.00 | 93.20 | 61.80 | 1.60 | 84.45 | 82.25 | 73.56 | 0.32 | 75.47 | 39.40 | 28.00 | **64.62** |
| Frequency | 100.00 | 95.40 | 98.40 | 75.20 | 49.00 | 1.20 | 81.40 | 81.10 | 64.36 | 0.78 | 55.13 | 28.20 | 19.00 | 57.63 |

| 策略 | 128K 均值 |
|:----:|:---------:|
| Aliasing | **59.71** |
| Frequency | 55.41 |

在 64K 处，别名领先 **6.99 pp**；在 128K 处领先 **4.30 pp**。在测试的 64K 与 128K 上别名平均更好（映射消融未覆盖其他长度，不能据此断言 $L \leq 128K$ 全范围都更优）。但在 MK2 和 QA2 等少数任务上，频率插值在 128K 处略优于别名，暗示混合映射在更长上下文下可能有益。

### 5.3 推理效率分析（Table 7 / Table 17 — Qwen3-8B on H100）

论文以 FA3 为 H100 主参照，FA2 为次参照，FA4 为互补实现，并加入一个匹配的 CuTe 控制（不含双焦注意力）。以下数值取自正文 Table 7 与附录 H 的 Table 17（均以 FA3 为 1.00×）。

**预填充（Prefill）吞吐量（tok/s；括号为相对 FA3）**：

| 方法 | 4K | 32K | 64K | 96K | 128K |
|------|:---:|:---:|:---:|:---:|:---:|
| FA3 (primary baseline) | 33915 (1.00×) | 22869 (1.00×) | 17701 (1.00×) | 14615 (1.00×) | 12269 (1.00×) |
| FA2 (secondary) | 31211 (0.92×) | 17796 (0.78×) | 12238 (0.69×) | 9332 (0.64×) | 7433 (0.61×) |
| FA4 (complementary) | 33455 (0.99×) | 22358 (0.98×) | 16690 (0.94×) | 13693 (0.94×) | 11400 (0.93×) |
| CuTe control (no bifocal) | 33235 (0.98×) | 22431 (0.98×) | 16860 (0.95×) | 13810 (0.94×) | 11494 (0.94×) |
| Jet-Long (unfused) | 31242 (0.92×) | 17810 (0.78×) | 10909 (0.62×) | 8548 (0.58×) | 6932 (0.57×) |
| **Jet-Long CuTe** | **33830 (1.00×)** | **22759 (1.00×)** | **15612 (0.88×)** | **12575 (0.86×)** | **10382 (0.85×)** |

**生成（Generation）吞吐量（tok/s；括号为相对 FA3）**：

| 方法 | 64K | 96K | 128K |
|------|:---:|:---:|:---:|
| FA3 (primary baseline) | 68.33 (1.00×) | 61.44 (1.00×) | 56.05 (1.00×) |
| FA2 (secondary) | 74.80 (1.09×) | 67.01 (1.09×) | 60.16 (1.07×) |
| CuTe control (no bifocal) | 71.48 (1.05×) | 64.14 (1.04×) | 58.38 (1.04×) |
| Jet-Long (unfused) | 14.20 (0.21×) | 10.49 (0.17×) | 8.32 (0.15×) |
| **Jet-Long CuTe** | **73.90 (1.08×)** | **65.00 (1.06×)** | **58.03 (1.04×)** |

原生长度（≤32K）G=1 绕过扩展代码，匹配 FA3（0.97–1.00× FA3）。扩展长度（64K–128K）：

- **Unfused 基线**付出双焦代价：三次 FA 调用 + 核外合并使 prefill 降至 0.57–0.62× FA3（相对 FA2 为 0.89–0.93×），生成降至 0.15–0.21× FA3
- **Fused CuTe 核**大幅降低双焦实现开销：prefill 相对未融合实现提升 1.43–1.56×，但仍为 FA3 的 0.83–0.88×（相对匹配 CuTe 控制为 0.88–0.93×，即双焦规则仍有 7–12% 的 prefill 开销）；生成达到 FA3 的 1.04–1.08×（与 FA2 差距在 3.6% 以内）

---

## 第六章 深入分析与讨论

### 6.1 方法原理的统一视角

Jet-Long 可以看作对 Self-Extend 路线的两项关键改进：

| 维度 | Self-Extend | Jet-Long | 改进效果 |
|------|------------|---------|---------|
| 分组系数 | 固定 G（G=8） | 动态 G = ceil(L/32K) | 短输入保真，长输入稳定 |
| 合并策略 | 简单拼接 | 容斥原理精确合并 | 多键检索收益最大；1.7B MK2 为 +45.40 pp，8B MK2/MK3 为 +34.40/+31.00 pp（论文未给出统一区间） |
| 工程实现 | Python 层操作 | 融合 CuTe kernel | 大幅降低开销，但相对 FA3 仍有 7–12% prefill 成本 |

### 6.2 超参数鲁棒性

Jet-Long 唯一的超参数是局部窗口 $w_0$。消融显示：
- $w_0 \in [512, 4096]$ 范围内的 RULER 差异 ≤2 pp
- 固定 $w_0=2048$ 即可在所有模型上达到接近最佳的性能

相比之下，DNTK 需要调节 RoPE base 缩放因子，YaRN 需要调节温度参数和 ramp 因子，DCA 需要选择块大小。Jet-Long 的**超参数鲁棒性**是其实际部署优势之一。

### 6.3 位置别名 vs 频率插值的深层分析

位置别名本质上将离散位置映射到 LLM 在训练中实际**见过**的整数位置集合。"Aliasing 优于 Frequency"的实验结果暗示：LLM 可能学习了一个离散的位置角度网格，连续频率插值引入的平滑性反而破坏了模型对此网格的预期。这为 RoPE-based LLM 的位置编码理解提供了新的视角。

---

## 第七章 局限性与未来方向

### 7.1 当前局限

1. **训练长度先验依赖**：动态分组公式 $G = \max(1, \lceil L / w_{pretrained} \rceil)$ 要求已知模型的确切预训练窗口。对渐进式训练或未知长度的模型可能不适用。

2. **Aliasing 的固有上限**：当外推到 256K、512K（G=8, 16）时，每个粗化位置承载的真实 token 数线性增长，位置碰撞不可避免加剧。论文在 Qwen3-30B-A3B-Instruct-2507 上已评测 512K/1M（Table 3），但主 Qwen3-1.7B/4B/8B 实验上限为 128K（G=4）；更极端的倍率仍需验证。

3. **需要 softmax + RoPE 基础**：方法要求模型使用 RoPE + softmax attention。对 NoPE、ALiBi 或其他位置编码方案不直接适用。不过实验已证明可推广到混合线性注意力架构。

4. **效率结论绑定 CuTe/NVIDIA 硬件**：0.83–0.88× FA3 的 prefill 结果依赖 CuTe kernel + H100。在非 NVIDIA 平台或缺乏等价 kernel 时，效率表现可能不同。

5. **复杂推理覆盖不足**：主结果以检索和语言建模为主，缺少对长跨度数学证明、多步因果推理等场景的评测。

### 7.2 未来方向

1. **在更长上下文验证映射策略**：论文已评测到 512K/1M，但位置别名 vs 频率混合映射的效果仍需在该范围验证
2. **与训练基方法结合**：零样本方法与长上下文微调并非互斥，结合可能进一步突破 aliasing 上限
3. **泛化到其他架构**：扩展到 Multi-head Latent Attention（MLA）和稀疏注意力变体
4. **Jet-Nemotron 后续**：方法已成功扩展到 Jet-Nemotron 混合架构，未来可探索 Jet-Long 在完整 Jet 模型家族中的应用

---

## 附录：论文中未列出但值得注意的细节

- **基线配置**：DNTK 使用 factor=4.0、YaRN 使用 factor=4.0 target=128K、DCA 使用 chunk_size=20480 local_window=4096、Self-Extend 使用 group_size=8 window_size=1024
- **Qwen3 基础模型**：RoPE base $\theta=10^6$
- **评测方法**：RULER 和 HELMET-RAG 使用贪心解码，PG-19 使用 stride=1024 的 teacher-forced perplexity
- **Appendix H 效率数据**：Table 17 给出 Qwen3-8B-Base 的 selected prefill/decode 测量（含 96K），并非 1.7B/4B/8B 全部数据；Appendix B 只是指标定义
