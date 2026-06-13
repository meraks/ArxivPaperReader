# KV Cache Optimization Strategies for Scalable and Efficient LLM Inference — 深度精读

---

> **论文信息**
> - **标题**：KV Cache Optimization Strategies for Scalable and Efficient LLM Inference
> - **作者**：Y. Xu, N. K. Khaira, T. Singh
> - **arXiv**：[2603.20397](https://arxiv.org/abs/2603.20397)
> - **官方代码**：无官方实现（综述论文）

---

## Ch1: 论文概述与核心贡献

### 1.1 一句话总结

这是一篇对 KV Cache 优化技术的**全景式系统综述**，将 30+ 篇前沿论文的方法归纳为 5 大类（驱逐、压缩、混合存储、新注意力机制、组合策略），并为 7 种实际部署场景提供了选型建议。核心结论：**没有一种技术在所有场景下都最优**——自适应多阶段 pipeline 是下一前沿。

### 1.2 为什么 KV Cache 是 LLM 推理的命门？

Transformer 自回归生成有一个根本性的冗余：每个新 token 生成时，都要重新计算与**所有历史 token** 的注意力。对于第 $t$ 个 token，注意力计算复杂度为：

$$
\mathcal{O}(t \cdot d_{\text{model}})
$$

而 KV Cache 的核心思想极其简单：**将已计算的 Key 和 Value 矩阵缓存起来，新 token 只需计算与自己的 Q 的注意力**。这消除了 O(N²) 的重复计算。

单个 token 的 KV cache 大小（每层每头）：

$$
\text{KV}_{\text{per\_token}} = 2 \times d_h \times B \text{ bytes}
$$

完整模型的 KV cache 大小（上下文长度 $T$）：

$$
\text{KV}_{\text{total}} = 2 \times L \times H_{\text{kv}} \times d_h \times T \times B \text{ bytes}
$$

其中 $L$=层数，$H_{\text{kv}}$=KV 头数（GQA 时小于 Q 头数），$d_h$=头维度，$B$=每元素字节数（FP16 为 2 bytes）。

**问题的核心**：这个缓存的大小随上下文长度**线性增长**。在 2026 年，上下文窗口已从 GPT-3 的 2K 爆炸到百万 token 级别：

| 模型 | 上下文 | KV Cache 大小 (估算) |
|------|--------|---------------------|
| GPT-3 (175B) | 2K | ~3 GB |
| GPT-4 (推测) | 32K | ~50 GB |
| Gemini 1.5 Pro | 1M | ~1.5 TB |
| Llama 4 | 10M | **不可行** |

### 1.3 五大类别速览

| 类别 | 核心思想 | 代表方法 | 关键权衡 |
|------|---------|---------|---------|
| **Cache Eviction** | 扔掉不重要的 token | H2O, SnapKV, RocketKV | 内存↔精度 |
| **Cache Compression** | 用更少比特存储 | KIVI (2-bit), KVQuant | 内存↔解压开销 |
| **Hybrid Memory** | GPU/CPU/磁盘分级存 | PagedAttention, Oneiros | 速度↔容量 |
| **New Attention** | 替换 O(N²) 的 softmax | Linear, Log-Linear, KIMI | 精度↔复杂度 |
| **Combination** | 多技术混合 | FlexGen, ShadowKV, TailorKV | 效率↔复杂度 |

---

## Ch2: 类别一 — Cache Eviction（缓存驱逐）

### 2.1 核心思想

选择性丢弃不重要的 KV 对，保持缓存大小可控。大多数方法依赖**注意力分数**来估计 token 重要性。

**关键洞察**：并非所有历史 token 对新 token 的生成同等重要。研究表明，只有少数"重击手"（Heavy Hitters）token 贡献了大部分注意力权重。

### 2.2 代表方法详解

**H2O (Heavy-Hitter Oracle)** — 注意力累积分数

H2O 识别"重击手"token：在整个生成过程中累积注意力分数最高的 token。同时保留最近的 token（局部上下文窗口）。驱逐策略是贪心的——保留 top-K 重击手 + 最近 W 个 token。

累积注意力分数定义为（论文 Eq. 1）：

$$
\text{Score}(K_i, V_i) = \sum_{t=1}^{T} \alpha_{t,i} = \sum_{t=1}^{T} \text{softmax}_j \left( \frac{Q_t K_j^\top}{\sqrt{d_k}} \right)_i
$$

算法细节：
1. 初始化空缓存
2. 对于每个新 token $t$，如果缓存有空间，则加入
3. 否则，计算所有缓存 token（包括新 token）的累积注意力分数
4. 驱逐分数最低的 token
5. 重复步骤 2-4

**SnapKV** — 投票机制找关键 token

SnapKV 在 prefill 阶段完成后运行。核心创新是"投票"机制：用一个观察窗口（observation window）内的 token 来"投票"哪些历史 token 重要。对每个注意力头，SnapKV 聚类重要的 key 特征，然后保留聚类后的 token 及其周围上下文。

具体实现：
1. 选择一个观察窗口（通常为最近 $W$ 个 token）
2. 对每个注意力头，计算观察窗口内 token 的注意力分数
3. 对每个历史 token，累加来自观察窗口的注意力分数（投票）
4. 保留分数最高的 token 及其周围上下文（通过 1D pooling 聚类）
5. 形成新的 KV cache 用于解码

**HASHEVICT** — 注意力前驱逐（零计算开销）

最轻量的方案：在计算注意力**之前**就决定要驱逐哪些 token。使用 SimHash（LSH 的一种），将每个 token 的 key embedding 哈希为二进制码，通过汉明距离近似注意力相关性。与 query token 汉明距离最大的（最不相似的）被驱逐。

**RocketKV** — 两阶段压缩

粗粒度 → 细粒度的层级式驱逐。Stage 1 用 SnapKV 做永久性粗驱逐。Stage 2 用 Hybrid Sparse Attention (HSA) 做动态细粒度选择。HSA 将 token 分页，存储每页的 max/min key 值来快速近似 token 重要性。

HSA 机制：每个 attention head 维护一个 KV cache 页面。每个页面存储当前所有 token 的 key 值的 min/max。计算新 token 的 query 时，通过比较 query 与每个页面的 min/max 汉明距离，快速近似 token 重要性，避免计算完整注意力。

**Ada-KV** — 注意力头自适应预算

关键发现：不同注意力头有不同的"稀疏度"——有的头注意力集中在少数 token 上（sparse heads），有的分散在大量 token 上（dispersed heads）。Ada-KV 动态分配驱逐预算：sparse heads 可以用更少 token，省下的预算给 dispersed heads。

### 2.3 驱逐方法全景表（对应论文 Table 2）

| 方法 | 机制 | 阶段 | 核心描述 |
|------|------|------|---------|
| **H2O** | 驱逐 Top-K 非关键 token | Decoding | 保留 Heavy-Hitter (H2) token + 最近 token 的平衡 |
| **SnapKV** | 保留关键 token 及其上下文 | Post-Prefill | 观察窗口投票 + 1D pooling 聚类 |
| **NACL** | 混合代理与随机驱逐 | Prefill | 编码阶段单次全局驱逐 |
| **InfiniPot** | 持续上下文蒸馏 (CCD) | Prefill | CaP (催化提示) + NuC (压缩下新颖度) 评分，固定内存预算 |
| **HASHEVICT** | 无注意力 LSH 排序 | Decoding | SimHash + 汉明距离近似 token 重要性，避免全注意力计算 |
| **MorphKV** | 相关性感知选择 | Decoding | 恒定缓存大小处理长回复，消除早期 token 偏置 |
| **RocketKV** | 两阶段压缩 | Prefill + Decoding | Stage 1: SnapKV 永久粗驱逐；Stage 2: HSA 动态细粒度选择 (页级 max/min key) |
| **KVzip** | 上下文重构评分 | Prefill | 自监督重构任务 ("Repeat the previous content") 得到 token 重要性 |
| **Ada-KV** | 自适应预算分配 | Post-Prefill | 按头动态分配驱逐预算：sparse heads 少给，dispersed heads 多给 |

---

## Ch3: 类别二 — Cache Compression（缓存压缩）

### 3.1 核心思想

不驱逐 token，而是用更少的比特存储——通过量化（quantization）或低秩分解减少每个 token 的内存占用。

### 3.2 代表方法详解

**KIVI — 2-bit 非对称量化**

KIVI 的关键洞察：Key cache 和 Value cache 有不同的数值特征——

- **Key cache**：少数通道的幅值非常大（outlier channels），需要 per-channel 量化
- **Value cache**：没有明显的 outlier 模式，per-token 量化即可

量化策略：
1. **Key cache**：按 channel 量化（每个 channel 独立量化）
2. **Value cache**：按 token 量化（每个 token 独立量化）
3. **组织结构**：将 KV cache 划分为 group（如 32 token）+ residual（最近 token 全精度 FP16）
4. **动态合并**：随着生成推进，residual 合并到 group 中

量化公式：
$$
\text{Quantized}(x) = \text{round}\left(\frac{x - z}{s}\right), \quad x \in \mathbb{R}
$$

其中 $z$ 是 zero-point，$s$ 是 scaling factor。KIVI 使用自适应量化，确保每个 channel/token 的动态范围都得到充分利用。

**MiniCache — 跨层压缩**

发现相邻层的 KV cache 高度相似。MiniCache 利用这种跨层冗余——不单独存储每一层的 KV cache，而是存储共享表示 + 层间残差。

具体实现：
1. 选择一个参考层（如第 0 层）
2. 存储该层的 KV cache（共享表示）
3. 计算其他层的残差（当前层 KV cache - 共享表示）
4. 仅存储残差，显著减少存储需求
5. 推理时，通过组合共享表示 + 残差得到完整 KV cache

**KVQuant — 面向千万级上下文**

专为**超长上下文**（>10M tokens）设计。使用 per-channel 量化 + 非均匀量化网格，在 4-bit 精度下保持几乎无损的模型质量。

量化策略：
1. **Per-channel 量化**：每个 channel 独立量化，捕获不同 channel 的动态范围
2. **非均匀量化网格**：使用自适应量化器，动态调整量化参数
3. **4-bit 精度**：在保持高精度的同时，实现 16× 的压缩比
4. **超长上下文支持**：即使在 10M tokens 上下文下，也能保持模型质量

**PALU — 低秩投影压缩**

不量化，而是用低秩矩阵分解压缩 KV cache：

将投影矩阵 $W \in \mathbb{R}^{d_{\text{model}} \times d_h}$ 通过 SVD 分解：

$$
W \approx A \times B, \quad A \in \mathbb{R}^{d_{\text{model}} \times r}, \quad B \in \mathbb{R}^{r \times d_h}, \quad r \ll d_h
$$

推理时：
1. 计算潜在表示 $H = X A \in \mathbb{R}^{r}$（只缓存小的 $H$）
2. 注意力计算需要时，动态重构：$K = H B$ 或 $V = H B$

PALU 使用 **Matrix Fusion** 将重构矩阵 $B$ 离线融合到其他权重中，避免运行时重构开销。对于使用 RoPE 的 Key，采用自定义高效 GPU kernel 动态重构。还提出 **Group-Head Low-Rank Decomposition (G-LRD)**：跨头联合分解精度高、逐头分解重构快。

**MiniCache — 跨层压缩**

发现相邻层的 KV cache 高度相似。MiniCache 利用这种跨层冗余——不单独存储每一层的 KV cache，而是存储共享表示 + 层间残差。

**KVQuant — 面向千万级上下文**

专为**超长上下文**（>10M tokens）设计。使用 per-channel 量化 + 非均匀量化网格，在 4-bit 精度下保持几乎无损的模型质量。

**PALU — 低秩投影压缩 (SVD-based)**

不量化，而是用低秩矩阵分解压缩 KV cache。将投影矩阵 $W \in \mathbb{R}^{d_{\text{model}} \times d_h}$ 通过 SVD 分解：

$$
W \approx A \times B, \quad A \in \mathbb{R}^{d_{\text{model}} \times r}, \quad B \in \mathbb{R}^{r \times d_h}, \quad r \ll d_h
$$

推理时：
1. 计算潜在表示 $H = X A \in \mathbb{R}^{r}$（只缓存小的 $H$）
2. 注意力计算需要时，动态重构：$K = H B$ 或 $V = H B$

PALU 使用 **Matrix Fusion** 将重构矩阵 $B$ 离线融合到其他权重中，避免运行时重构开销。对于使用 RoPE 的 Key，采用自定义高效 GPU kernel 动态重构。还提出 **Group-Head Low-Rank Decomposition (G-LRD)**：跨头联合分解精度高、逐头分解重构快。

### 3.3 压缩方法对比（对应论文 Table 6）

| 方法 | 内存减少 | 吞吐提升 | 精度损失 | 权衡/备注 |
|------|---------|---------|---------|----------|
| **KIVI** | 2.6× 峰值内存 | 2.35×–3.47× | <2% 大多数模型 | 单 KV 头模型需 4-bit 保精度；量化初始化有开销 |
| **MiniCache** | 最高 41% | ~5× | 极小 | 仅支持两层合并，限制更高压缩 |
| **PALU** | ~50% KV 压缩 | 1.89×(RoPE) / 2.91×(量化) | 相当基线 | Key 重构有开销（尤其是 RoPE） |
| **KVQuant** | 3.7–6.9× | ~1.7× | <0.1 perplexity (3-bit) | 长上下文训练难；反量化更复杂 |

---

## Ch4: 类别三 — Hybrid Memory Solutions（混合存储）

### 4.1 核心思想

**不减少缓存内容，而是改变缓存放哪里**。利用 GPU → CPU → 磁盘的多级存储层次，将不常用的 KV cache 从昂贵的 GPU 显存（HBM）卸载到 CPU 内存或 SSD。

### 4.2 代表方法详解

**PagedAttention (vLLM) — KV Cache 的分页式管理**

受操作系统虚拟内存的启发，PagedAttention 将 KV cache 分割为固定大小的"页"（block）。每个请求的 KV cache 不再占据一大块连续显存，而是由多个非连续的页组成。

**关键优势**：
- **零内存碎片**：不再有内部碎片（internal fragmentation）
- **高效共享**：多个请求可以共享相同的 prefix 页（如 system prompt）
- **批量增大**：同一批次可以容纳更多并发请求

**实际效果**：相比传统方案，吞吐量提升 2-4×。

PagedAttention 设计了一个 **Page Table**，将逻辑页（logical page）映射到物理页（physical page）。每个 attention head 维护自己的 page table。计算注意力时，只加载需要的页面到 GPU 显存。

PagedAttention 支持 **prefix caching**，多个请求可以共享相同的 prefix KV cache（如 system prompt），进一步提高吞吐量。

**Oneiros — 参数内存重映射**

在 Grace Hopper (GH200) 等具有高 CPU-GPU 带宽的硬件上，Oneiros 将**模型参数**的内存空间重新映射用于 KV cache 存储。当 KV cache 膨胀时，动态"借用"参数内存区域，实现最高 86.7% 的吞吐量提升。

具体实现：
1. 识别哪些参数可以安全卸载（通常是 inactive 模型的参数）
2. 如果所有模型都是 active，则均匀分布地从每个模型卸载一个层
3. 利用自回归解码的层序执行特性，充分利用 GPU 计算时间
4. 异步迁移参数从 CPU 到 GPU
5. 释放的显存用于 KV cache

Oneiros 实现了**硬件-软件协同**，充分利用 GH200 系统上 450–900 GB/s 的 CPU-GPU 带宽。

**LayerKV — 分层 KV Cache 管理**

观察到**不同层的 KV cache 重要性不同**。LayerKV 对重要层保留完整 cache，对非重要层激进卸载。TTFT（Time to First Token）提升最高 69×。

具体实现：
1. 分析每个层的 KV cache 重要性（如注意力分布）
2. 选择最小数量的重要层保留在 GPU
3. 其他层卸载到 CPU
4. 使用双缓冲 overlapping computation 和 communication
5. 集成 SLO-aware 调度器，平衡延迟和吞吐量

**FlexGen — 单 GPU 极限卸载**

为资源极度受限的场景设计（单张消费级 GPU）。将模型权重、激活值和 KV cache 拆分到 GPU → CPU → 磁盘，通过线性规划求解最优的数据放置策略。使用 zig-zag 块调度最大化 GPU 上模型权重的复用。

具体实现：
1. **多级存储层次**：GPU（高速）、CPU（中等）、磁盘（低速）
2. **线性规划调度**：为每个张量（权重、激活值、KV cache）分配最佳存储位置
3. **Zig-zag 块调度**：最大化 GPU 上模型权重的复用
4. **动态调整**：根据当前任务需求调整存储策略

**ShadowKV — 值卸载 + 键压缩**

结合压缩和混合存储：将 Value cache 卸载到 CPU（因为 Value 没有低秩结构），将 Key cache 用 SVD 压缩后保留在 GPU（pre-RoPE key 具有强低秩特性）。解码时通过 landmark vector 预估哪些 chunk 需要从 CPU 取回。

具体实现：
1. **Key cache 压缩**：使用 SVD 分解，保留主要信息
2. **Value cache 卸载**：迁移到 CPU 内存
3. **Landmark vector**：用于快速估计哪些 KV chunk 需要从 CPU 取回
4. **稀疏注意力**：仅计算与 landmark vector 附近的 key 的注意力
5. **动态取回**：根据需要从 CPU 异步加载 Value cache

**Oneiros — 参数内存重映射**

在 Grace Hopper (GH200) 等具有高 CPU-GPU 带宽的硬件上，Oneiros 将**模型参数**的内存空间重新映射用于 KV cache 存储。当 KV cache 膨胀时，动态"借用"参数内存区域，实现最高 86.7% 的吞吐量提升。

**LayerKV — 分层 KV Cache 管理**

观察到**不同层的 KV cache 重要性不同**。LayerKV 对重要层保留完整 cache，对非重要层激进卸载。TTFT（Time to First Token）提升最高 69×。

**FlexGen — 单 GPU 极限卸载**

为资源极度受限的场景设计（单张消费级 GPU）。将模型权重、激活值和 KV cache 拆分到 GPU → CPU → 磁盘，通过线性规划求解最优的数据放置策略。使用 zig-zag 块调度最大化 GPU 上模型权重的复用。

**ShadowKV — 值卸载 + 键压缩**

结合压缩和混合存储：将 Value cache 卸载到 CPU（因为 Value 没有低秩结构），将 Key cache 用 SVD 压缩后保留在 GPU（pre-RoPE key 具有强低秩特性）。解码时通过 landmark vector 预估哪些 chunk 需要从 CPU 取回。

### 4.3 混合存储方法对比（对应论文 Table 7）

| 方法 | 卸载目标 | 机制 | 关键优化 | 性能指标 |
|------|---------|------|---------|---------|
| **PagedAttention** | CPU DRAM | 虚拟内存式分页 | 消除碎片、支持前缀共享 | 2–4× 吞吐，无损 |
| **InfiniGen** | CPU DRAM | 注意力推测 + 预取 | 仅取回关键 KV | 1.63–32.9× 加速，>15% 相对 KV 缓存需额外 GPU 存推测权重 |
| **LayerKV** | CPU DRAM | 分层管理 + SLO 感知调度 | 计算通信重叠，最小化 TTFT | **TTFT 提升 69×**，高负载解码轻微吞吐权衡 |
| **INF2** | NVMe SSD (CSD) | Attention-Near-Storage | 利用 CSD 内部带宽 | 3.46× 吞吐，KV I/O 降 >80%，需 CSD 硬件 |
| **KVPR** | CPU DRAM | GPU 部分重计算 + 异步传输 | 掩盖 PCIe 延迟 | 延迟降 35.8%，吞吐增 46.2%，仅解码阶段 |
| **Oneiros** | CPU DRAM | 参数内存重映射 | 多租户场景回收空闲模型显存 | TBT 降 44.8–82.5%，TTFT 降 20.7–99.3%，吞吐增 6.6–86.7%（vs vLLM） |
| **CLO** | CPU DRAM | 头级近似缓存 + 零拷贝传输 | 消除 CPU 瓶颈，跑满 PCIe | - |

| 方法 | 内存减少 | 吞吐/延迟提升 | 精度 | 权衡 |
|------|---------|-------------|------|------|
| **FlexGen** | 10× | 40–100× 最大吞吐 (vs DS/HF) | 4-bit 可忽略 | 高延迟；PCIe/磁盘带宽成瓶颈；小批量不适用 |
| **Q-Hitter** | 20× | 33× (vs HF Accelerate) | 全质量保留 | 计算量化误差与反量化开销 |
| **ShadowKV** | 6× GPU 显存 | 3.04× | 稀疏预算 >1.56% 时高精度 | 部分依赖 PCIe 带宽 |
| **TailorKV** | 73.8% GPU 显存 | 8–18× (vs 标准卸载) | 接近无损 | Prefill 瓶颈；系统复杂度高 |

---

## Ch5: 类别四 — 新型注意力机制

### 5.1 核心思想

不优化现有 KV cache——**从根本上改变注意力计算方式**，使复杂度从 O(N²) 降为 O(N) 或 O(N log N)。

### 5.2 注意力机制复杂度全景（对应论文 Table 5）

| 方法 | 机制 | 训练复杂度 | 解码复杂度(每步) | 解码内存 | 特点 |
|------|------|-----------|----------------|---------|------|
| **Softmax Attention** | Scaled Dot-Product & MHA | O(T²) | O(T) | O(T) | 高表达力、高开销 |
| **Linear Attention** | 核函数线性化 | O(T) | O(1) | O(1) | 表达力受限、开销最低 |
| **Log-Linear Attention** | 对数增长隐藏状态 | O(T log T) | O(log T) | O(log T) | 效率与表达力平衡 |
| **Local Linear Attention** | 局部线性回归 | O(T²) | ~O(T) | O(T) | 优于 softmax 的 bias-variance 权衡 |
| **KIMI Linear** | KDA + MHA 混合 (3:1) | 主要 O(T) | O(1) | O(1) | **超越全注意力精度**，需混合保持全局信息流 |

### 5.3 代表方法详解

**Linear Attention — O(1) 内存的注意力**

核心思想：将 softmax 替换为核函数的点积。利用矩阵乘法的结合律交换计算顺序，将 O(N²) 降为 O(N)（论文 Eq. 6-8）：

$$
V'_i = \frac{\phi(Q_i)^\top S_i}{\phi(Q_i)^\top Z_i}, \quad
S_i = \sum_{j=1}^i \phi(K_j) V_j^\top, \quad
Z_i = \sum_{j=1}^i \phi(K_j)
$$

其中 $\phi(\cdot)$ 是特征映射函数（如 $\phi(x) = \text{elu}(x) + 1$ 或 $\phi(x) = \text{ReLU}(x)$）。

关键优势：$S_i$ 和 $Z_i$ 是固定大小的状态矩阵（$d_k \times d_v$ 和 $d_k$），每步更新只需 O(1) 计算与内存，无需存储完整 KV 序列。

**Log-Linear Attention — 对数增长的隐藏状态**

Linear Attention 的 O(1) 内存过于激进（表达能力不足），Softmax Attention 的 O(T) 过于昂贵。Log-Linear 在两者之间找到一个平衡点：隐藏状态大小**对数增长**，解码复杂度 O(log T)。Log-Linear 维持了较高的表达能力，同时显著降低了计算和内存开销。

**Local Linear Attention — 局部线性回归视角**

将注意力机制重新解释为回归问题：
- Softmax Attention ≈ 局部常数回归（只看附近 keys 的 value 平均）
- Linear Attention ≈ 全局线性回归（拟合一条全局直线）
- **Local Linear Attention** ≈ 局部线性回归（在每个 query 周围拟合局部线性模型）

这使 LLA 在表达能力上**超越 Softmax Attention**，同时保留线性注意力的效率优势。

**KIMI Linear — 动态门控 + 混合架构**

最先进的线性注意力方案。核心创新是 Kimi Delta Attention (KDA)：

$$
S_t = (I - \beta_t k_t k_t^\top) \text{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^\top
$$

其中 $\alpha_t$ 是遗忘门（控制旧信息保留），$\beta_t$ 是更新率（控制新信息的影响）。两者都由小型神经网络从输入 token 动态计算。

输出计算：
$$
o_t = S_t^\top q_t \in \mathbb{R}^{d_v}
$$

**关键设计**：KIMI Linear 不是纯线性注意力——它采用 KDA 和全注意力 **3:1 混合**。全注意力层提供全局信息流，KDA 层提供效率。在 1M 上下文时吞吐量提升 6×，KV 占用减少 75%，且**超越全注意力精度**。

**Local Linear Attention — 局部线性回归视角**

将注意力机制重新解释为回归问题：
- Softmax Attention ≈ 局部常数回归（只看附近 keys 的 value 平均）
- Linear Attention ≈ 全局线性回归（拟合一条全局直线）
- **Local Linear Attention** ≈ 局部线性回归（在每个 query 周围拟合局部线性模型）

这使 LLA 在表达能力上**超越 Softmax Attention**，同时保留线性注意力的效率优势。

**KIMI Linear — 动态门控 + 混合架构**

最先进的线性注意力方案。核心创新是 Kimi Delta Attention (KDA)：

$$
S_t = (I - \beta_t k_t k_t^\top) \text{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^\top
$$

其中 $\alpha_t$ 是遗忘门（控制旧信息保留），$\beta_t$ 是更新率（控制新信息的影响）。两者都由小型神经网络从输入 token 动态计算。

输出计算：
$$
o_t = S_t^\top q_t \in \mathbb{R}^{d_v}
$$

**关键设计**：KIMI Linear 不是纯线性注意力——它采用 KDA 和全注意力 **3:1 混合**。全注意力层提供全局信息流，KDA 层提供效率。在 1M 上下文时吞吐量提升 6×，KV 占用减少 75%，且**超越全注意力精度**。

---

## Ch6: 类别五 — 组合策略

### 6.1 核心理念

**没有任何单一技术在所有场景下都是最优的**。组合策略将驱逐 + 压缩 + 混合存储混合使用。

### 6.2 代表方法

**FlexGen** — 压缩 + 卸载（4-bit 量化 + GPU/CPU/磁盘三级存储 + 线性规划调度）

FlexGen 是一个**系统化的 KV cache 优化框架**，将模型权重、激活值和 KV cache 拆分到多级存储中。通过线性规划求解最优的数据放置策略，实现高达 100× 的吞吐量提升。

具体实现：
1. **多级存储层次**：GPU（高速）、CPU（中等）、磁盘（低速）
2. **线性规划调度**：为每个张量（权重、激活值、KV cache）分配最佳存储位置
3. **Zig-zag 块调度**：最大化 GPU 上模型权重的复用
4. **动态调整**：根据当前任务需求调整存储策略

**Q-Hitter** — 稀疏 + 量化（同时考虑注意力重要性和量化鲁棒性来选择 token，达到 20× 内存压缩）

Q-Hitter 采用**双指标 token 选择机制**：
1. **注意力评分**：反映 token 的重要性（用于 future predictions）
2. **量化误差评分**：反映 token 的量化鲁棒性（量化损失）
3. **统一评分**：平衡两种指标，选择 top-K token
4. **稀疏 KV cache**：仅存储高评分 token，显著减少内存和 I/O 开销

**ShadowKV** — 压缩 + 混合存储（键 SVD 压缩 + 值 CPU 卸载 + 解码时稀疏取回）

ShadowKV 采用**混合存储策略**：
1. **Key cache 压缩**：使用 SVD 分解，保留主要信息
2. **Value cache 卸载**：迁移到 CPU 内存
3. **Landmark vector**：用于快速估计哪些 KV chunk 需要从 CPU 取回
4. **稀疏注意力**：仅计算与 landmark vector 附近的 key 的注意力
5. **动态取回**：根据需要从 CPU 异步加载 Value cache

**TailorKV** — 自适应混合（离线识别哪些层适合量化、哪些适合稀疏化，在线动态选择策略）

TailorKV 利用**层级注意力分布差异**：
1. **离线分析**：计算每个层的注意力分布（top-k 比例）
2. **分类策略**：
   - **量化友好层**：注意力分布分散 → 采用 aggressive quantization
   - **稀疏友好层**：注意力集中 → 采用 sparsity-based offloading
3. **在线动态选择**：根据 runtime 情况调整策略
4. **双缓冲技术**：overlap CPU-GPU 数据传输与 GPU 计算

### 6.3 组合策略优势

组合策略将多种优化技术结合起来，实现比任何单一技术更高的效率。这些方法通常结合驱逐、压缩和混合存储，以平衡内存、吞吐量和精度。

组合策略的优势包括：
1. **互补性**：不同技术互补，共同提高性能
2. **自适应性**：根据 runtime 情况动态调整策略
3. **鲁棒性**：多种策略保证系统稳定性
4. **扩展性**：适用于从边缘设备到云端服务器的各种场景

---

## Ch7: 部署场景选型指南

论文最实用的贡献之一是将每种技术映射到 7 种真实部署场景（对应论文 Section 5）：

### 7.1 七种场景速查

| 场景 | 推荐方法 | 不推荐 / 注意事项 |
|------|---------|------------------|
| **超长上下文 (>1M) 单请求** | 驱逐: RocketKV / 压缩: KVQuant / KIMI Linear | — |
| **最小模型修改 (Plug-and-play)** | Ada-KV, SnapKV, KIVI (即插即用) | 新型注意力(需重新训练) |
| **高吞吐服务** | PagedAttention, Oneiros, ShadowKV | FlexGen(面向延迟不敏感场景) |
| **边缘/内存受限设备** | InfiniPot (专为移动/边缘设计), TailorKV (单 RTX 3090 跑 128k 8B) | 混合存储: PagedAttention 需高端 GPU，Oneiros 需 450–900 GB/s CPU-GPU 带宽 |
| **多轮对话** | RocketKV-MT (保留所有 KV 供下轮), KVzip (查询无关压缩), ShadowKV (多轮能力) | H2O (永久驱逐不可逆) |
| **Prefill-Heavy / 低 TTFT** | NACL, HASHEVICT, LayerKV (TTFT 69×) | — |
| **精度关键推理** | PagedAttention (无损), 混合存储方案 | 驱逐/压缩 (有损), 线性注意力 |

### 7.2 硬件特定约束 (论文 Section 5.8)

| 硬件配置 | 推荐方案 |
|----------|---------|
| **高 PCIe 带宽 (NVIDIA GH200)** | Oneiros, CLO (利用高 CPU-GPU 带宽) |
| **CSD (计算存储设备, FPGA/ASIC SSD)** | INF2 (首选，近存计算) |
| **消费级单 GPU (内存受限)** | FlexGen (GPU/CPU/磁盘三级存储 + 线性规划调度) |

### 7.3 三个通用原则

1. **超长上下文 → 驱逐 + 压缩**：当上下文超过百万 token 时，只有激进减少存储量才能生存
2. **高并发 → 混合存储**：通过分级存储扩展容量而非精度，保持无损
3. **新架构设计 → 新型注意力**：如果可以从头训练，O(N) 注意力是未来方向

### 7.4 场景具体建议

**超长上下文 (>1M) 单请求：**
- 驱逐方法：RocketKV（两阶段压缩），有效处理超长上下文
- 压缩方法：KVQuant（4-bit per-channel），实现高压缩比
- 注意力机制：KIMI Linear（混合架构），实现高吞吐量和低内存占用

**最小模型修改 (Plug-and-play)：**
- 驱逐方法：Ada-KV（自适应预算分配），无需修改模型结构
- 压缩方法：SnapKV（投票机制），无需修改模型结构
- 压缩方法：KIVI（2-bit 非对称量化），无需修改模型结构

**高吞吐服务：**
- 混合存储方法：PagedAttention（分页管理），实现高吞吐量
- 混合存储方法：Oneiros（参数内存重映射），实现高吞吐量
- 混合存储方法：ShadowKV（值卸载 + 键压缩），实现高吞吐量

**边缘/内存受限设备：**
- 驱逐方法：InfiniPot（持续上下文蒸馏），专为移动/边缘设备设计
- 组合方法：TailorKV（自适应混合），适用于资源受限的 GPU

**多轮对话：**
- 驱逐方法：RocketKV-MT（多轮版本），保留所有 KV 供下轮
- 压缩方法：KVzip（上下文重构），实现高效的多轮对话
- 组合方法：ShadowKV（值卸载 + 键压缩），实现多轮对话能力

**Prefill-Heavy / 低 TTFT：**
- 驱逐方法：NACL（混合代理和随机驱逐），实现高效的 prefill
- 驱逐方法：HASHEVICT（注意力前驱逐），实现低延迟
- 混合存储方法：LayerKV（分层 KV Cache 管理），实现低 TTFT

**精度关键推理：**
- 混合存储方法：PagedAttention（无损），保持高精度
- 混合存储方法：各种混合存储方案，实现无损推理

---

## Ch8: 总结与展望（对应论文 Section 6）

### 8.1 关键洞察

这篇综述的价值不在于"提出了新方法"，而在于**首次系统化地揭示了 KV Cache 优化的设计空间**：

1. **驱逐 vs 压缩**：驱逐是"扔掉"（信息不可恢复），压缩是"折叠"（可解压但精度有限）
2. **存储 vs 精度**：混合存储是唯一无损方案，但严重依赖硬件（GPU-CPU带宽）
3. **即插即用 vs 重新训练**：驱逐/压缩/混合存储可以即插即用，新型注意力需要重新训练
4. **组合 > 单一**：最优方案几乎总是多技术的混合（ShadowKV, TailorKV）

论文总结的三大核心观察：
- **超长上下文 (>1M) 由驱逐/压缩/组合主导**：RocketKV、KVzip、ShadowKV 通过粗粒度 token 选择 + 细粒度重构/低秩近似，实现高压缩比且精度损失极小
- **混合内存方案是高吞吐服务的基石**：PagedAttention 已成事实标准；Oneiros、LayerKV、INF2、CLO 进一步针对多租户、TTFT、近存计算、PCIe 饱和等场景优化
- **新型注意力机制需混合部署**：纯 Linear Attention 表达力不足；KIMI Linear 的 3:1 混合架构证明了"线性注意力 + 少量全注意力"可超越纯 Softmax

### 8.2 2026 年的趋势

- **百万 token 上下文**已成为现实（Gemini 1.5 Pro, Kimi），KV Cache 压力指数级增长
- **PagedAttention** 已成为事实标准，几乎所有推理框架都采用分页方案
- **KIMI Linear** 证明了线性注意力可以在**精度上超越 Softmax Attention**——这是范式级突破
- **自适应多阶段 pipeline** 是正在兴起的方向：不再预先选择一种策略，而是运行时动态切换

### 8.3 局限性与未来方向

1. **硬件依赖过强**：很多方法的性能高度依赖 PCIe 带宽、GPU 型号、是否支持 CSD
2. **缺乏统一的 Benchmark**：不同论文在不同硬件/模型上评估，难以直接比较
3. **长尾效应**：百万 token 上下文在实际应用中仍罕见，优化可能过度
4. **与 speculative decoding 的协同**：KV Cache 优化和投机解码的结合研究不足

---

## 延伸阅读

- **H2O**: arXiv:2306.14048 — Heavy-Hitter Oracle
- **SnapKV**: arXiv:2404.14469 — 基于投票的长上下文 KV 驱逐
- **PagedAttention (vLLM)**: arXiv:2309.06180 — 分页式 KV Cache 管理
- **KIVI**: 2-bit 非对称量化 KV Cache
- **KIMI Linear**: arXiv:2510.26692 — 超越全注意力的线性注意力
- **ShadowKV**: arXiv:2410.21465 — 键压缩+值卸载
- **FlexGen**: arXiv:2303.06865 — 单 GPU 极限卸载
- **RocketKV**: arXiv:2502.14051 — 两阶段 KV 压缩

---

*阅读日期: 2026-06-05 | 论文长度: 24页 | 涵盖方法: 30+ 篇 | 阅读深度: Deep Read*
