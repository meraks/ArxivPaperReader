# KV Cache Optimization Strategies for Scalable and Efficient LLM Inference — 深度精读


> **论文信息**
> - **标题**：KV Cache Optimization Strategies for Scalable and Efficient LLM Inference
> - **作者**：Y. Xu, N. K. Khaira, T. Singh
> - **arXiv**：[2603.20397](https://arxiv.org/abs/2603.20397)
> - **官方代码**：无官方实现（综述论文）

---

> **论文**: KV Cache Optimization Strategies for Scalable and Efficient LLM Inference
> **作者**: Y. Xu, N. K. Khaira, T. Singh
> **arXiv**: [2603.20397](https://arxiv.org/abs/2603.20397) | 2026年3月
> **页数**: 24页，14张图 | **类型**: 系统综述 (Survey)
> **代码**: 无官方实现（综述论文，涵盖30+篇独立论文的方法）

---

## Ch1: 论文概述与核心贡献

### 1.1 一句话总结

这是一篇对 KV Cache 优化技术的**全景式系统综述**，将 30+ 篇前沿论文的方法归纳为 5 大类（驱逐、压缩、混合存储、新注意力机制、组合策略），并为 7 种实际部署场景提供了选型建议。核心结论：**没有一种技术在所有场景下都最优**——自适应多阶段 pipeline 是下一前沿。

### 1.2 为什么 KV Cache 是 LLM 推理的命门？

Transformer 自回归生成有一个根本性的冗余：每个新 token 生成时，都要重新计算与**所有历史 token** 的注意力。对于第 N 个 token：

$$
\text{计算量} \propto N \times d_{\text{model}}
$$

而 KV Cache 的核心思想极其简单：**将已计算的 Key 和 Value 矩阵缓存起来，新 token 只需计算与自己的 Q 的注意力**。这消除了 O(N²) 的重复计算。

$$
\text{KV}_{\text{per\_token}} = 2 \times H \times D \times B \times L
$$

其中 H=注意力头数，D=头维度，L=层数，B=每元素字节数。

**问题的核心**：这个缓存的大小随上下文长度**线性增长**。在 2026 年，上下文窗口已从 GPT-3 的 2K 爆炸到百万 token 级别：

| 模型 | 上下文 | KV Cache 大小 (估算) |
|------|--------|---------------------|
| GPT-3 (175B) | 2K | ~3 GB |
| GPT-4 (推测) | 32K | ~50 GB |
| Gemini 1.5 Pro | 1M | ~1.5 TB |
| Llama 4 | 10M | **不可行** |

> **类比理解**：KV Cache 就像你写论文时保留的所有"参考文献草稿"。一开始只有几页笔记，查找很快。但当参考文献积累到 1000 条时，你的书桌（GPU 显存）就堆满了。KV Cache 优化的本质是回答三个问题：(1) 哪些笔记可以扔？(驱逐) (2) 哪些笔记可以压缩成简要记录？(压缩) (3) 不常用的笔记能否放到书架上？(混合存储)

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

$$
\text{Score}(token_i) = \sum_{t} \text{Attention}(token_i \mid query_t)
$$

**SnapKV** — 投票机制找关键 token

SnapKV 在 prefill 阶段完成后运行。核心创新是"投票"机制：用一个观察窗口（observation window）内的 token 来"投票"哪些历史 token 重要。对每个注意力头，SnapKV 聚类重要的 key 特征，然后保留聚类后的 token 及其周围上下文。

**HASHEVICT** — 注意力前驱逐（零计算开销）

最轻量的方案：在计算注意力**之前**就决定要驱逐哪些 token。使用 SimHash（LSH 的一种），将每个 token 的 key embedding 哈希为二进制码，通过汉明距离近似注意力相关性。与 query token 汉明距离最大的（最不相似的）被驱逐。

> **类比理解**：HASHEVICT 就像在图书馆找书时先看书的封面颜色。红色封面的书不太可能是你要找的蓝色主题的书，所以直接跳过——不需要翻开看内容。这比一本本翻阅（计算完整注意力）快得多。

**RocketKV** — 两阶段压缩

粗粒度 → 细粒度的层级式驱逐。Stage 1 用 SnapKV 做永久性粗驱逐。Stage 2 用 Hybrid Sparse Attention (HSA) 做动态细粒度选择。HSA 将 token 分页，存储每页的 max/min key 值来快速近似 token 重要性。

**Ada-KV** — 注意力头自适应预算

关键发现：不同注意力头有不同的"稀疏度"——有的头注意力集中在少数 token 上（sparse heads），有的分散在大量 token 上（dispersed heads）。Ada-KV 动态分配驱逐预算：sparse heads 可以用更少 token，省下的预算给 dispersed heads。

### 2.3 驱逐方法全景表

| 方法 | 阶段 | 机制 | 适用场景 |
|------|------|------|---------|
| H2O | Decoding | Heavy-Hitter + 最近token | 单轮生成 |
| SnapKV | Post-Prefill | 投票+聚类 | 长上下文 |
| NACL | Prefill | 全局单次驱逐 | Prefill-heavy |
| InfiniPot | Prefill | 持续上下文蒸馏 | 边缘设备 |
| HASHEVICT | Decoding | LSH预筛选 | 低延迟 |
| MorphKV | Decoding | 相关性感知 | 长回复 |
| RocketKV | Prefill+Decoding | 两阶段层级 | 通用 |
| KVzip | Prefill | 自监督重建 | 多轮对话 |
| Ada-KV | Post-Prefill | 头自适应预算 | 即插即用 |

---

## Ch3: 类别二 — Cache Compression（缓存压缩）

### 3.1 核心思想

不驱逐 token，而是用更少的比特存储——通过量化（quantization）或低秩分解减少每个 token 的内存占用。

### 3.2 代表方法详解

**KIVI — 2-bit 非对称量化**

KIVI 的关键洞察：Key cache 和 Value cache 有不同的数值特征——

- **Key cache**：少数通道的幅值非常大（outlier channels），需要 per-channel 量化
- **Value cache**：没有明显的 outlier 模式，per-token 量化即可

KIVI 将 KV cache 组织为"组"（如 32 token 一组）+ "残差"（最近 token 保持全精度 FP16）。组被量化为 2-bit，残差保持 FP16。随着生成推进，残差合并到组中。

**MiniCache — 跨层压缩**

发现相邻层的 KV cache 高度相似。MiniCache 利用这种跨层冗余——不单独存储每一层的 KV cache，而是存储共享表示 + 层间残差。

**KVQuant — 面向千万级上下文**

专为**超长上下文**（>10M tokens）设计。使用 per-channel 量化 + 非均匀量化网格，在 4-bit 精度下保持几乎无损的模型质量。

**PALU — 低秩投影压缩**

不量化，而是用低秩矩阵分解压缩 KV cache：

$$
K_{\text{compressed}} = K \times P, \quad P \in \mathbb{R}^{D \times r}, \quad r \ll D
$$

在推理时，用压缩的 K 和 V 重构注意力分数，计算开销远小于完整矩阵乘法。

### 3.3 压缩方法对比

| 方法 | 技术 | 压缩比 | 精度损失 |
|------|------|--------|---------|
| KIVI | 2-bit 非对称量化 | ~8× | 极小 |
| MiniCache | 跨层共享 | ~2× | 几乎无损 |
| KVQuant | 4-bit per-channel | ~4× | 接近无损 |
| PALU | 低秩投影 | 视 r 而定 | 可控 |

> **类比理解**：压缩 vs 驱逐就像整理衣柜的两种策略。驱逐是"扔掉不穿的衣服"（信息永久丢失），压缩是"用真空袋压缩冬天的羽绒服"（信息保留，只是占用更少空间，需要时解压取出）。

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

**Oneiros — 参数内存重映射**

在 Grace Hopper (GH200) 等具有高 CPU-GPU 带宽的硬件上，Oneiros 将**模型参数**的内存空间重新映射用于 KV cache 存储。当 KV cache 膨胀时，动态"借用"参数内存区域，实现最高 86.7% 的吞吐量提升。

**LayerKV — 分层 KV Cache 管理**

观察到**不同层的 KV cache 重要性不同**。LayerKV 对重要层保留完整 cache，对非重要层激进卸载。TTFT（Time to First Token）提升最高 69×。

**FlexGen — 单 GPU 极限卸载**

为资源极度受限的场景设计（单张消费级 GPU）。将模型权重、激活值和 KV cache 拆分到 GPU → CPU → 磁盘，通过线性规划求解最优的数据放置策略。使用 zig-zag 块调度最大化 GPU 上模型权重的复用。

**ShadowKV — 值卸载 + 键压缩**

结合压缩和混合存储：将 Value cache 卸载到 CPU（因为 Value 没有低秩结构），将 Key cache 用 SVD 压缩后保留在 GPU（pre-RoPE key 具有强低秩特性）。解码时通过 landmark vector 预估哪些 chunk 需要从 CPU 取回。

### 4.3 混合存储方法对比

| 方法 | 策略 | 吞吐量提升 | 精度损失 |
|------|------|-----------|---------|
| PagedAttention | 分页管理 | 2-4× | 无损 |
| Oneiros | 参数内存重映射 | 最高 86.7% | 无损 |
| LayerKV | 分层卸载 | TTFT 69× | 无损 |
| FlexGen | GPU+CPU+磁盘 | 最高 100× | 4-bit 可忽略 |
| ShadowKV | 值卸载+键压缩 | 3.04× | 稀疏预算>1.56%时高精度 |

---

## Ch5: 类别四 — 新型注意力机制

### 5.1 核心思想

不优化现有 KV cache——**从根本上改变注意力计算方式**，使复杂度从 O(N²) 降为 O(N) 或 O(N log N)。

### 5.2 注意力机制复杂度全景

| 方法 | 训练复杂度 | 解码复杂度(每步) | 解码内存 | 表达能力 |
|------|-----------|----------------|---------|---------|
| **Softmax Attention** | O(T²) | O(T) | O(T) | 最强 |
| **Linear Attention** | O(T) | O(1) | O(1) | 弱于 softmax |
| **Log-Linear Attention** | O(T log T) | O(log T) | O(log T) | 平衡 |
| **Local Linear Attention** | O(T²) | ~O(T) | O(T) | 优于 softmax |
| **KIMI Linear** | 主要是 O(T) | O(1) | O(1) | **超越全注意力** |

### 5.3 代表方法详解

**Linear Attention — O(1) 内存的注意力**

核心思想：将 softmax 替换为核函数的点积。利用矩阵乘法的结合律交换计算顺序，将 O(N²) 降为 O(N)：

$$
\text{Attention}(Q, K, V) \approx \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1})}
$$

先算 K^T V（固定大小矩阵），再与 Q 相乘——每次新 token 只需 O(1) 计算。

**Log-Linear Attention — 对数增长的隐藏状态**

Linear Attention 的 O(1) 内存过于激进（表达能力不足），Softmax Attention 的 O(T) 过于昂贵。Log-Linear 在两者之间找到一个平衡点：隐藏状态大小**对数增长**，解码复杂度 O(log T)。

**Local Linear Attention — 局部线性回归视角**

将注意力机制重新解释为回归问题：
- Softmax Attention ≈ 局部常数回归（只看附近 keys 的 value 平均）
- Linear Attention ≈ 全局线性回归（拟合一条全局直线）
- **Local Linear Attention** ≈ 局部线性回归（在每个 query 周围拟合局部线性模型）

这使 LLA 在表达能力上**超越 Softmax Attention**，同时保留线性注意力的效率优势。

**KIMI Linear — 动态门控 + 混合架构**

最先进的线性注意力方案。核心创新是 Kimi Delta Attention (KDA)：

$$
S_t = (I - \beta_t k_t k_t^T) \text{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^T
$$

其中 α_t 是遗忘门（控制旧信息保留），β_t 是更新率（控制新信息的影响）。两者都由小型神经网络从输入 token 动态计算。

**关键设计**：KIMI Linear 不是纯线性注意力——它采用 KDA 和全注意力 **3:1 混合**。全注意力层提供全局信息流，KDA 层提供效率。在 1M 上下文时吞吐量提升 6×，KV 占用减少 75%，且**超越全注意力精度**。

> **类比理解**：新型注意力机制就像从"每次查阅所有历史档案"（Softmax O(N²)）进化到"维护一个不断更新的摘要本"（Linear O(1)）。KIMI Linear 则更进一步——维护多个动态更新的摘要，同时偶尔回到原档案查阅（混合策略）。

---

## Ch6: 类别五 — 组合策略

### 6.1 核心理念

**没有任何单一技术在所有场景下都是最优的**。组合策略将驱逐 + 压缩 + 混合存储混合使用。

### 6.2 代表方法

**FlexGen** — 压缩 + 卸载（4-bit 量化 + GPU/CPU/磁盘三级存储 + 线性规划调度）

**Q-Hitter** — 稀疏 + 量化（同时考虑注意力重要性和量化鲁棒性来选择 token，达到 20× 内存压缩）

**ShadowKV** — 压缩 + 混合存储（键 SVD 压缩 + 值 CPU 卸载 + 解码时稀疏取回）

**TailorKV** — 自适应混合（离线识别哪些层适合量化、哪些适合稀疏化，在线动态选择策略）

---

## Ch7: 部署场景选型指南

论文最实用的贡献之一是将每种技术映射到 7 种真实部署场景：

### 7.1 七种场景速查

| 场景 | 推荐方法 | 不推荐 |
|------|---------|--------|
| **超长上下文(>1M)单请求** | 驱逐(RocketKV) / 压缩(KVQuant) / KIMI Linear | — |
| **最小模型修改** | Ada-KV, SnapKV, KIVI (即插即用) | 新型注意力(需重新训练) |
| **高吞吐服务** | PagedAttention, Oneiros, ShadowKV | FlexGen(面向延迟不敏感) |
| **边缘/内存受限** | InfiniPot, TailorKV | 混合存储(需要高端 GPU) |
| **多轮对话** | RocketKV-MT, KVzip, ShadowKV | H2O(持久驱逐不可逆) |
| **Prefill-Heavy** | NACL, HASHEVICT, LayerKV(TTFT 69×) | — |
| **精度关键推理** | PagedAttention(无损), 混合存储 | 驱逐/压缩(有损), 线性注意力 |

### 7.2 三个通用原则

1. **超长上下文 → 驱逐 + 压缩**：当上下文超过百万 token 时，只有激进减少存储量才能生存
2. **高并发 → 混合存储**：通过分级存储扩展容量而非精度，保持无损
3. **新架构设计 → 新型注意力**：如果可以从头训练，O(N) 注意力是未来方向

---

## Ch8: 总结与展望

### 8.1 关键洞察

这篇综述的价值不在于"提出了新方法"，而在于**首次系统化地揭示了 KV Cache 优化的设计空间**：

1. **驱逐 vs 压缩**：驱逐是"扔掉"（信息不可恢复），压缩是"折叠"（可解压但精度有限）
2. **存储 vs 精度**：混合存储是唯一无损方案，但严重依赖硬件（GPU-CPU带宽）
3. **即插即用 vs 重新训练**：驱逐/压缩/混合存储可以即插即用，新型注意力需要重新训练
4. **组合 > 单一**：最优方案几乎总是多技术的混合（ShadowKV, TailorKV）

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
