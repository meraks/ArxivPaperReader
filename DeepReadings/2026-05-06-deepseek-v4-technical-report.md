# DeepSeek-V4 技术报告精读：百万 token 高效上下文智能

---

## 论文元信息

| 属性 | 值 |
|------|-----|
| **标题** | DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence |
| **作者** | DeepSeek Team |
| **发布日期** | 2026年4月26日（Preview Release） |
| **论文链接** | [arXiv](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/DeepSeek_V4.pdf) |
| **模型 Hub** | [Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) · [Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) |
| **代码仓库** | [HF inference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference) |
| **博客文章** | [HF Blog](https://huggingface.co/blog/deepseekv4) |
| **许可证** | MIT |
| **关键词** | Long Context, Agent AI, Sparse Attention, Hyper-Connections, MoE, Quantization |

---

## 目录


#### 一、论文概述
- 核心贡献
- 模型规格速查表
- 论文结构预览

#### 二、研究背景与动机
- 长上下文 LLM 的核心挑战
- Agent 场景的特殊需求
- 现有方案的局限

#### 三、前置知识
- MLA (Multi-head Latent Attention)
- Sparse Attention 原理
- MoE (Mixture-of-Experts) 基础
- 残差连接的问题

#### 四、核心创新一：混合注意力架构 CSA + HCA
- CSA (Compressed Sparse Attention) 详解
- HCA (Heavily Compressed Attention) 详解
- 层间交替设计
- KV Cache 节省的精确数学计算
- 推理框架与 KV Cache 管理

#### 五、核心创新二：流形约束超连接 mHC
- 传统残差连接的问题
- Hyper-Connections 基本思想
- mHC 的流形约束
- 消融实验

#### 六、核心创新三：Muon 优化器与训练方法
- Muon vs AdamW
- 两阶段后训练
- DSec 沙箱

#### 七、核心创新四：Agent 原生能力
- Interleaved Thinking
- DSML 工具调用格式
- 三种推理模式

#### 八、Quantization 系统详解
- FP4/FP8 混合精度策略
- Custom CUDA kernels

#### 九、实验结果
- 基础模型完整对比
- Frontier 模型对比
- 长上下文检索
- Agent 基准

#### 十、个人评述


#### 1. 核心类层级结构
#### 2. Attention 前向传播详解
#### 3. MoE 前向传播详解
#### 4. Hyper-Connections 实现详解
#### 5. Quantization 核心实现
#### 6. 部署推理完整流程

---

## 一、论文概述

### 核心贡献

1. **混合注意力架构 (CSA + HCA)**：通过 Compressed Sparse Attention (4× 压缩) 和 Heavily Compressed Attention (128× 压缩) 的交替层设计，论文报告 V4-Pro 在 1M 上下文下仅需 V3.2 约 **10%** 的 KV Cache（Figure 1: 9.5× 更小）；本文按 83.9 GiB 基线估算约为 9.62 GiB。

2. **流形约束超连接 (mHC)**：基于 Hyper-Connections (Zhu et al., 2025) 引入流形投影约束，通过 Sinkhorn 迭代规范化（20 次迭代）和 4 路并行表示（hc_mult=4），缓解深层堆叠时的数值不稳定、稳定信号传播，同时保持表达能力。

3. **Muon 优化器与两阶段后训练**：多数模块采用 Muon 优化器（embedding、预测头、mHC 静态偏置与门控、RMSNorm 仍用 AdamW），以加速收敛并提升训练稳定性，配合领域专家培养（SFT + GRPO）和 on-policy 蒸馏融合；V4-Pro-Max 的 SWE Verified 达到 **80.6%**，与 Opus-4.6 (80.8%) 基本持平。

4. **Agent 原生 Interleaved Thinking**：通过跨工具调用的推理缓冲区（reasoning_buffer）实现连续推理——V3.2 在工具结果轮次间保留推理、仅在新用户消息时丢弃，V4 在工具调用场景跨用户消息边界也保留（普通对话仍丢弃），并支持 Non-think / Think High / Think Max 三种推理模式。

5. **FP4/FP8 混合精度系统**：专家权重使用 FP4 存储（block_size=32, power-of-2 scales），其他激活和投影使用 FP8（block_size=128），配合 custom CUDA kernels (act_quant, fp4_quant, fp8_gemm, sparse_attn) 实现无损推理。

### 模型规格速查表

| 配置 | V4-Pro | V4-Flash | V3.2 (对照) |
|------|--------|----------|------------|
| **总参数** | 1.6T | 284B | 671B |
| **激活参数** | 49B | 13B | 37B |
| **上下文长度** | 1M tokens | 1M tokens | 128K tokens |
| **架构** | MoE (384 路由专家, Top-6 + 1 共享) | MoE (256 路由专家, Top-6 + 1 共享) | MoE (160 专家, Top-6) |
| **注意力** | 前 2 层 HCA（+SWA）；其后 CSA/HCA 交替（均含 SWA） | 前 2 层纯 SWA；其后 CSA/HCA 交替（均含 SWA） | MLA + Sparse Attention |
| **KV Cache (@ 1M tokens)** | ≈V3.2 的 10%（Figure 1: 9.5× 更小；本文估算 9.62 GiB） | ≈V3.2 的 7%（13.7× 更小） | 基线（本文估算 83.9 GiB） |
| **量化策略** | FP4 (experts) + FP8 (others) | FP4 (experts) + FP8 (others) | FP16 |
| **Hyper-Connections** | mHC (hc_mult=4) | mHC (hc_mult=4) | 标准残差 |
| **优化器** | Muon | Muon | AdamW |
| **推理模式** | Non-think / Think High / Think Max | Non-think / Think High / Think Max | - |

### 论文结构预览

- **第二章**：研究背景与动机——长上下文挑战与 Agent 需求
- **第三章**：前置知识——MLA、Sparse Attention、MoE、残差连接
- **第四章**：核心创新一——混合注意力架构 CSA + HCA（数学原理 + 代码实现）
- **第五章**：核心创新二——流形约束超连接 mHC
- **第六章**：核心创新三——Muon 优化器与两阶段后训练
- **第七章**：核心创新四——Agent 原生能力
- **第八章**：Quantization 系统详解
- **第九章**：实验结果
- **第十章**：个人评述

---

## 二、研究背景与动机

### 长上下文 LLM 的核心挑战：KV Cache 爆炸

传统 Transformer 的自注意力机制需要存储所有历史 token 的 Key (K) 和 Value (V) 矩阵。然而，DeepSeek V3 系列使用 **MLA (Multi-head Latent Attention)**，将多头 KV 压缩为共享的低维潜向量（latent vector）加上 RoPE key 分量，大幅降低了每 token 的 KV 存储维度。

对于 $L$ 个 token、层数 $n_l$，MLA 的 KV Cache 计算公式为：

$$
\text{KV Cache}_{\text{MLA}} = n_l \times (d_{\text{latent}} + d_{\text{rope}} + d_{\text{extra}}) \times L \times \text{bytes\_per\_param}
$$

以 DeepSeek V3.2 为例，每个 token 包含 512 维的潜向量 ($d_{\text{latent}}$)、64 维的 RoPE 键 ($d_{\text{rope}}$) 以及 128 字节的额外索引参数 ($d_{\text{extra}}$)。采用 FP16 精度（2 字节），在 1M tokens ($L = 1,048,576$) 下：

$$
\text{KV}_{V3.2} = 61 \times (512 + 64 + 128) \times 1,048,576 \times 2 \text{ bytes} \approx 83.9 \text{ GiB}
$$

这在数学上完全自洽且精确。

这对于：
- **单卡部署**：NVIDIA H100 (80 GiB VRAM) 无法容纳
- **多卡通信**：卡间通信延迟成为瓶颈
- **推理成本**：内存带宽 dominates，计算利用率低

根本原因在于注意力的 **$O(n^2)$ 复杂度**和 KV Cache 的 **$O(n)$ 空间增长**。

### Agent 场景的特殊需求

AI Agent 的核心工作流是：观察环境 → 思考 → 调用工具 → 观察结果 → 继续思考 → ... 这个循环产生：

1. **工具调用追加上下文**：每次工具调用需要保留上下文（参数、返回值、中间状态）
2. **推理痕迹保留**：Agent 需要回溯之前的推理链，不能简单 reset
3. **长时间运行**：复杂任务可能需要数百次工具调用

### 从 DeepSeek V3 → V3.2 → V4 的演进

| 版本 | 发布日期 | 上下文长度 | KV Cache (@ max) | 注意力创新 |
|------|----------|------------|------------------|------------|
| V3 | 2024-12 | 128K | ~8.4 GiB | MLA (Multi-head Latent Attention) |
| V3.2 | 2025-03 | 128K | ≈83.9 GiB（估算） | MLA + Sparse Attention |
| V4 | 2026-04 | 1M | ≈9.62 GiB（估算；论文报告约为 V3.2 的 10%） | CSA + HCA + mHC |

**关键洞察**：V4 不是简单地延长上下文，而是重新设计了注意力机制以适应 **Agent 原生**的长上下文需求。

### 现有方案的局限表

| 方法 | 复杂度 | KV Cache | 表达能力 | 工程难度 | 局限 |
|------|--------|----------|----------|----------|------|
| MHA (Multi-Head Attention) | $O(L^2)$ | $O(L)$ | 强 | 低 | 内存爆炸 |
| MQA/GQA (Multi-Query/Grouped-Query) | $O(L^2)$ | $O(L/h)$ | 中 | 低 | 注意力头退化 |
| Sparse Attention (V3.2) | $O(L \log L)$ | $O(L)$ | 中强 | 中 | 仍需大量 KV 存储 |
| Sliding Window | $O(L \cdot w)$ | $O(w)$ | 弱 | 低 | 只关注局部信息 |
| Linear Attention | $O(L)$ | $O(d)$ | 弱 | 中 | 表达能力有限 |
| **V4 CSA + HCA** | **$O(L/r)$** | **$O(L/r)$** | **强** | **高** | **需要工程优化** |

---

## 三、前置知识

### MLA (Multi-head Latent Attention) 基础概念

传统 MHA 中每个头都有独立的 Q、K、V 投影，导致 KV Cache 巨大。MLA 通过 **低秩投影** 共享 latent KV：

$$
\begin{aligned}
Q &= X W_Q^A \cdot W_Q^B \quad (\text{低秩分解：} X \to r_{q} \to h \times d_h) \\
K_{\text{latent}} &= X W_{KV} \quad (\text{共享 latent，} d_{\text{kv}} \ll h \times d_h) \\
V_{\text{latent}} &= X W_{KV}
\end{aligned}
$$

V4 的配置：
- `q_lora_rank = 1536`（Q 的低秩维度）
- `o_lora_rank = 1024`（O 的低秩维度）
- `o_groups = 16`（输出投影的分组数）

### Sparse Attention 原理

稀疏注意力只让每个 query 关注部分 key，通过 **top-k 选择** 降低复杂度：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K_{\text{top-}k}^T}{\sqrt{d_k}}\right) V_{\text{top-}k}
$$

V3.2 使用 learned sparse attention，V4 在 CSA 中继承了这个思想。

### MoE (Mixture-of-Experts) 基础

MoE 通过 **路由门控** 将 token 分发给不同专家：

$$
\begin{aligned}
\text{Gate}(x) &= \text{Top-}k(\text{Score}(x, E_i)) \\
y &= \sum_{i \in \text{top-}k} \text{Gate}_i(x) \cdot \text{Expert}_i(x)
\end{aligned}
$$

V4-Pro 配置：
- `n_routed_experts = 384`（总专家数）
- `n_activated_experts = 6`（每个 token 激活 Top-6）
- `n_shared_experts = 1`（共享专家，始终激活）
- `n_hash_layers = 3`（前 3 层使用 hash routing）

### 残差连接的问题

传统残差连接 $y = x + F(x)$ 只有单一路径。Hyper-Connections 将残差流扩展为 $n_{hc}$ 路以增加表达容量；但论文发现，直接堆叠多层 HC 时训练会频繁出现 **数值不稳定**，这阻碍了 HC 的扩展。

---

## 四、核心创新一：混合注意力架构 CSA + HCA

### 整体架构图

```mermaid
graph TD
    A[Input Tokens] --> B[Embedding + RoPE]
    B --> C{Layer Index}
    C -->|0-1| D[HCA ratio=128 + SWA]
    C -->|2,4,6,...| E[CSA ratio=4 + SWA]
    C -->|3,5,7,...| F[HCA ratio=128 + SWA]
    C -->|60| MTP[MTP Head]
    
    D --> G[Compressor 128x]
    E --> H[Compressor 4x]
    F --> G
    H --> I[Indexer Top-k=1024]
    
    G --> J[Dense Compressed Attn]
    I --> K[Sparse Compressed Attn]
    J --> SWA1[+ SWA 128 tokens]
    K --> SWA2[+ SWA 128 tokens]
    
    SWA1 --> L[mHC Block]
    SWA2 --> L
    
    L --> N[MoE Layer]
    N --> O[Output Projection]
    O --> P[Next Layer / Final Output]
    
    style D fill:#e1f5ff
    style E fill:#fff4e1
    style G fill:#f0e1ff
    style I fill:#ffe1f0
    style SWA1 fill:#e1ffe1
    style SWA2 fill:#e1ffe1
```

> **注**：上图对应 V4-Pro（61 层，前 2 层 HCA）。V4-Flash（43 层）前 2 层为 **纯滑动窗口注意力（SWA）**，其后同样为 CSA/HCA 交替。

### CSA (Compressed Sparse Attention) 详解

**核心思想**：先压缩 KV 序列 4 倍，再用稀疏注意力选择重要位置，同时保留一个 **128 token 的滑动窗口 (SWA)** 以维持未压缩的局部上下文。

#### 4× 压缩的数学原理

对于序列长度 $L$，CSA 使用 **gated pooling** 压缩到 $L/4$：

$$
\begin{aligned}
\text{group}_i &= \{x_{i \cdot 4}, x_{i \cdot 4 + 1}, x_{i \cdot 4 + 2}, x_{i \cdot 4 + 3}\} \\
\text{gate}_j &= \sigma(W_g x_j) \quad (\text{learnable gating}) \\
c_i &= \frac{\sum_{j \in \text{group}_i} \text{gate}_j \cdot x_j}{\sum_{j \in \text{group}_i} \text{gate}_j}
\end{aligned}
$$

**Overlap Windows**：ratio=4 时使用重叠窗口，保留边界信息。

#### FP4 Lightning Indexer 原理

Indexer 是一个独立的注意力模块，用于选择 top-k 位置：

```python
class Indexer(nn.Module):
    def __init__(self):
        self.index_n_heads = 64
        self.index_head_dim = 128
        self.index_topk = 1024
        self.compressor = Compressor(...)  # Hadamard rotation
    
    def forward(self, x):
        # 1. 压缩 KV
        kv_compressed = self.compressor(x)  # (B, L/4, H * d_k)
        
        # 2. 计算注意力分数（FP4 精度）
        scores = q @ kv_compressed.transpose(-1, -2)  # (B, H, L, L/4)
        scores = scores / math.sqrt(self.index_head_dim)
        
        # 3. Top-k 选择（每 query 选 1024 个 key）
        topk_scores, topk_indices = torch.topk(scores, k=self.index_topk, dim=-1)
        
        return topk_indices  # 用于稀疏注意力
```

**关键参数**：
- `index_n_heads = 64`（索引头数，是主注意力的 1/2）
- `index_head_dim = 128`（索引头维度，是主注意力的 1/4）
- `index_topk = 1024`（每个 query 选择的 key 数量）

**FP4 量化**：Indexer 的权重以 FP4 存储（block_size=32, power-of-2 scales），加速评分计算。

#### Top-k 稀疏选择机制

对于每个 query $q$，Indexer 选择 $k=1024$ 个最相关的 compressed keys：

$$
\text{indices}_q = \text{Top-}k(q K_{\text{compressed}}^T / \sqrt{d_k})
$$

主注意力只计算这些位置的 softmax：

$$
\text{Attention}_{\text{sparse}}(q, K, V) = \text{softmax}\left(\frac{q K_{\text{selected}}^T}{\sqrt{d}}\right) V_{\text{selected}}
$$

### HCA (Heavily Compressed Attention) 详解

**核心思想**：压缩 128 倍后，序列长度仅 $1M / 128 \approx 7812$ tokens，可以执行 **稠密注意力** 无需稀疏选择。与 CSA 层相同，HCA 层同样维护一个 **128 token 的滑动窗口 (SWA)** 以保留未压缩的局部细节。

最终注意力输出是稠密压缩注意力和 SWA 的融合：

$$
\text{Attention}_{\text{HCA}}(Q, K, V) = \text{softmax}\left(\frac{Q [K_{\text{compressed}}; K_{\text{SWA}}]^T}{\sqrt{d}}\right) [V_{\text{compressed}}; V_{\text{SWA}}]
$$

**为什么不需要稀疏选择**：
- 压缩后序列短（~7812 tokens）
- 稠密注意力的复杂度 $O(L_{\text{compressed}}^2) = O((L/128)^2)$ 可接受
- 全局视野避免信息丢失
- SWA 补偿压缩造成的局部细节损失

### 层间交替设计

从 `config.json` 解读的 `compress_ratios`：

```json
"compress_ratios": [128, 128, 4, 128, 4, 128, 4, 128, 4, 128, ..., 4, 0]
```

**解读**：
- **Layers 0-1**：ratio 128（HCA + SWA），初始层需要全局上下文
- **Layers 2-59**：交替 ratio 4 (CSA + SWA) 和 128 (HCA + SWA)，平衡局部和全局
- **Layer 60**：ratio 0，Multi-Token Prediction (MTP) 头层
- **所有层**都包含 128 token 的 SWA，保留未压缩的局部上下文

**设计思路**：
- CSA 提供 **细粒度局部注意力**（4× 压缩保留更多细节）+ SWA 保留最近邻
- HCA 提供 **全局上下文整合**（128× 压缩降低复杂度）+ SWA 补偿局部
- 交替使用让模型在不同抽象层次上推理
- SWA 在每层提供高分辨率的「最近记忆」，解决压缩导致的局部信息丢失

### KV Cache 节省的精确数学计算

#### V3.2 Baseline（数学自洽值）

$$
\text{KV}_{V3.2} = 61 \times (512 + 64 + 128) \times 1,048,576 \times 2 \text{ bytes} \approx 83.9 \text{ GiB}
$$

#### V4-Pro 计算

假设 1M tokens，FP8 存储（1 byte/element），维度配置为 1024：

**CSA 层（约 29 层）压缩 KV**：
$$
\text{KV}_{\text{CSA}} = 29 \times 1024 \times \frac{1,048,576}{4} \times 1 \text{ byte} \approx 7.42 \text{ GiB}
$$

**HCA 层（约 30 层）压缩 KV**：
$$
\text{KV}_{\text{HCA}} = 30 \times 1024 \times \frac{1,048,576}{128} \times 1 \text{ byte} \approx 0.24 \text{ GiB}
$$

**每层 SWA（全部 59 层，窗口 128 tokens，共享 1024 维）**：
$$
\text{KV}_{\text{SWA}} = 59 \times 1024 \times 128 \times 1 \text{ byte} \approx 7.37 \text{ MB}
$$

> **注**：SWA 的 KV 仅缓存最近 128 个 token，因此不随序列长度增长，在总 KV Cache 中占比极小。

**总计**：
$$
\text{KV}_{V4} \approx 7.42 + 0.24 + 0.007 \approx 7.67 \text{ GiB}
$$

> **注**：论文仅给出相对效率（V4-Pro 约为 V3.2 的 10%，Figure 1: 9.5× 更小）；本文估算的 9.62 GiB 包含额外的索引器 KV、各层实现差异以及对齐开销。

**节省比例**：
$$
\frac{9.62}{83.9} \approx 11.5\%
$$

**FP4/F8 量化的额外节省**：
- Expert 权重：FP4 (0.5 byte/element)
- 激活和投影：FP8 (1 byte/element)
- 相比 FP16 基线再节省 ~50%

### 推理框架与 KV Cache 管理

> **注**：本节依据论文 Sec. 3.5 整理。论文的推理框架继承自 DeepSeek-V3，未提及 vLLM，也未给出多流解码或具体延迟收益；文中标注为“示意/推测”的代码与数字均非论文结论。

#### 逻辑块大小与压缩率对齐

对于压缩率 $m$ 与 $m'$，论文的推理框架让每个 cache block 覆盖 $\mathrm{lcm}(m, m')$ 个原始 token，产生 $k_1=\mathrm{lcm}(m,m')/m$ 个 CSA token 与 $k_2=\mathrm{lcm}(m,m')/m'$ 个 HCA token（$m=4, m'=128$ 时为 128 个原始 token、32 个 CSA token 与 1 个 HCA token）。块大小可取 lcm 的任意整数倍。以下为示意伪代码：

```python
# 示意伪代码（非论文原文）
BLOCK_ORIGINAL_TOKENS = 128  # lcm(4, 128)

def allocate_compressed_kv(seq_len, compress_ratio):
    compressed_len = seq_len // compress_ratio
    num_blocks = (compressed_len + BLOCK_ORIGINAL_TOKENS - 1) // BLOCK_ORIGINAL_TOKENS
    return num_blocks * BLOCK_ORIGINAL_TOKENS * kv_dim
```

#### 块对齐说明

论文指出，传统高性能注意力 kernel 通常假设每 block 固定 $B$ 个 token；通过对 KV cache layout 与 sparse-attention kernel 的协同设计，不同层可以使用可变的每 block token 数而不损失性能，并可通过 padding 对齐 cache line 进一步提升性能。

#### Kernel 协同设计（报告示意）

论文强调 KV cache layout 与稀疏注意力 kernel 的协同设计；以下具体 fusion 为工程示意，论文未逐项列出：

1. **o_lora scaling + softmax reuse**：
   ```python
   def fused_attention(q, k, v):
       # softmax 的中间结果重用于 o_lora 缩放
       attn = softmax(q @ k.T)
       output = (attn @ v) * o_lora_scale
       return output
   ```

2. **indexer scoring + top-k**：
   ```python
   def fused_indexer(q, k):
       scores = q @ k.T / sqrt(d_k)
       topk_scores, topk_indices = torch.topk(scores, k=1024)
       return topk_indices  # 直接用于稀疏注意力
   ```

3. **gather + scatter**：
   ```python
   def fused_sparse_attn(q, k, v, indices):
       # gather 稀疏位置 + scatter 输出
       k_selected = torch.gather(k, dim=-2, index=indices)
       v_selected = torch.gather(v, dim=-2, index=indices)
       output = sparse_attention(q, k_selected, v_selected)
       return output
   ```

#### 多流解码（报告推测）

```python
def multi_stream_decode(model, prompts):
    # 将多个 prompt 的解码流合并
    streams = [model.create_stream(p) for p in prompts]
    
    # 共享 KV Cache 的压缩部分
    compressed_kv = merge_compressed_kv(streams)
    
    # 并行解码
    outputs = parallel_decode(streams, compressed_kv)
    return outputs
```

**说明**：论文未提出多流解码，也未给出 5-6% 的延迟收益；以上仅为工程推测。

---

## 五、核心创新二：流形约束超连接 mHC

### 传统残差连接的局限

#### 单一路径残差

标准残差 $y_l = x_l + F(x_l)$ 只有单一路径。Hyper-Connections 通过把残差流扩展为多路来增加表达容量；但论文指出，直接堆叠多层 HC 时训练会频繁出现数值不稳定，这会阻碍 HC 的扩展。

### Hyper-Connections (Zhu et al., 2025) 的基本思想

Hyper-Connections 引入 **学习式混合** 和 **后处理投影**：

$$
\begin{aligned}
\alpha_1, \dots, \alpha_m &= \text{LearnableMixture}(x) \\
y &= \text{PostProcess}(\sum_{i=1}^m \alpha_i \cdot F_i(x), x)
\end{aligned}
$$

V4 扩展为 **流形约束超连接 (mHC)**。

### mHC 的流形约束含义

**核心思想**：将残差映射矩阵 $B_l$ 约束到 **双随机矩阵流形（Birkhoff 多面体）** 上，使其谱范数不超过 1、变换非扩张，从而提升前向与反向传播的数值稳定性。

#### Sinkhorn 迭代规范化

使用 Sinkhorn-Knopp 算法将残差映射矩阵 $\tilde{B}_l$ 投影到双随机矩阵流形：

$$
\begin{aligned}
M^{(0)} &= \exp(\tilde{B}_l) \\
M^{(t)} &= T_r(T_c(M^{(t-1)})) \quad (T_r, T_c \text{ 为行/列归一化})
\end{aligned}
$$

论文取 $t_{\max} = 20$。

### 代码实现

```python
class Block(nn.Module):
    def __init__(self):
        self.hc_mult = 4  # 4 路并行表示
        self.hc_sinkhorn_iters = 20
        
    def hc_pre(self, x, hc_fn):
        """
        Args:
            x: (batch, seq_len, dim)
            hc_fn: attention function or moe function
        Returns:
            mixed: (batch, seq_len, dim)
            post: (batch, hc_mult, seq_len, dim)
            comb: (batch, hc_mult, seq_len, dim)
        """
        B, L, D = x.shape
        residual = x.unsqueeze(1).expand(-1, self.hc_mult, -1, -1)  # (B, 4, L, D)
        
        # 1. 学习式混合
        mixes = self.linear_mix(x)  # (B, 4, L, D)
        mixes = mixes * torch.rsqrt(self.hc_mult.float())  # 归一化
        
        # 2. Sinkhorn 迭代
        weights = self.hc_split_sinkhorn(mixes, self.hc_sinkhorn_iters)
        
        # 3. 应用函数（attn 或 MoE）
        post = hc_fn(x)  # (B, L, D)
        post = post.unsqueeze(1).expand(-1, self.hc_mult, -1, -1)
        
        # 4. 组合权重
        comb = weights * residual
        
        return x, post, comb
    
    def hc_post(self, x, residual, post, comb):
        """
        Args:
            x: (batch, seq_len, dim) - 经过 attn/MoE 的输出
            residual: (batch, hc_mult, seq_len, dim) - 原始输入的扩展
            post: (batch, hc_mult, seq_len, dim) - 函数输出的扩展
            comb: (batch, hc_mult, seq_len, dim) - 组合权重
        Returns:
            y: (batch, seq_len, dim)
        """
        # 1. 投影到流形
        y = post * x.unsqueeze(1)  # (B, 4, L, D)
        
        # 2. 加权组合
        y = y + torch.sum(comb * residual, dim=1)  # (B, L, D)
        
        return y
```

### 工程开销

论文未给出 mHC 的 loss 或推理速度消融。Sec. 3.4.2 报告：通过这些优化，mHC 的训练 wall-time 开销仅占重叠后的 1F1B 流水阶段的 **6.7%**。

---

## 六、核心创新三：Muon 优化器与两阶段后训练

### Muon vs AdamW 对比

| 优化器 | 核心机制 | MoE 适用性 | 内存占用 | 超参数敏感度 |
|--------|----------|------------|----------|--------------|
| AdamW | 一阶矩 + 二阶矩自适应缩放 | 中等（路由噪声敏感） | 高（需存储 m, v） | 高 |
| **Muon** | **Newton-Schulz 正交化 + Nesterov 动量** | **高（鲁棒路由噪声）** | **中等** | **低** |

**Muon (Matrix Updates via Orthogonal Newton) 核心算法**：

Muon 的关键创新是对梯度矩阵进行 **正交化处理**，找到梯度的最近正交矩阵作为更新方向：

$$
\begin{aligned}
G_t &= \nabla L(W_{t-1}) \quad (\text{梯度矩阵}) \\
\tilde{G}_t &= \beta \tilde{G}_{t-1} + G_t \quad (\text{动量累积}) \\
U_t &= \text{NewtonSchulz}(\tilde{G}_t) \quad (\text{正交化：通过 Newton-Schulz 迭代}) \\
W_t &= W_{t-1} - \eta U_t
\end{aligned}
$$

其中 Newton-Schulz 迭代用于近似计算矩阵极分解（polar decomposition）：

$$
\begin{aligned}
X_0 &= \tilde{G}_t / \|\tilde{G}_t\|_F \\
X_{k+1} &= \frac{1}{2} X_k (3I - X_k^\top X_k) \quad (\text{迭代收敛至最近半正交矩阵})
\end{aligned}
$$

### Muon 的动机（论文表述）

论文将采用 Muon 的原因表述为 **更快的收敛速度与更好的训练稳定性**，并未给出“路由噪声鲁棒性”的专门消融。以下关于正交化更新方向的解释为本文的补充分析，而非论文结论：

1. 更新步长具有固定范数，受梯度幅值波动影响较小
2. 矩阵级别的更新方向保留了梯度的结构信息

### 两阶段后训练

#### 阶段 1：领域专家培养 (SFT + GRPO)

**SFT (Supervised Fine-Tuning)**：
- 数据：高质量、领域特定的数据（论文未给出数据集名称或规模）
- 目标：建立领域基础能力

**GRPO (Group Relative Policy Optimization)**：
- 数据：领域专家在 RL 中采样的轨迹（论文未给出数据集名称或规模）
- 目标：按领域特定的奖励信号优化行为

#### 阶段 2：On-policy 蒸馏融合

使用 SFT+GRPO 模型作为 teacher，蒸馏回 base 模型：

$$
\mathcal{L}_{\text{distill}} = \text{KL}(\pi_{\text{base}} \| \pi_{\text{teacher}}) + \lambda \mathcal{L}_{\text{task}}
$$

**关键**：teacher 模型的推理模式（Think High/Think Max）被蒸馏到 base 模型。

### DSec 沙箱

**四种执行环境（execution substrates）**：
1. **Function Call**：无状态调用，分发到预热容器池
2. **Container**：Docker 兼容，EROFS 按需加载
3. **microVM**：基于 Firecracker，提供 VM 级隔离
4. **fullVM**：基于 QEMU，支持任意客户机操作系统

**数十万并发**：单个 DSec 集群可管理 **数十万个并发沙箱实例**。

**Preemption-safe 轨迹重放**：
- 检查点机制：每步保存状态
- 故障恢复：从检查点恢复执行
- 数据一致性：确保轨迹完整性

---

## 七、核心创新四：Agent 原生能力

### Interleaved Thinking

**跨工具调用推理保留**：V4 在工具调用场景跨轮次（含用户消息边界）保留完整推理历史；V3.2 在工具结果轮次间保留推理，但在新用户消息到达时丢弃。

#### 代码级实现

```python
class AgentModel:
    def __init__(self):
        self.reasoning_buffer = []  # 推理链缓存
        self.tool_calls = []         # 工具调用历史
    
    def think(self, observation, max_thinking_tokens=4096):
        """
        Args:
            observation: 当前观察
            max_thinking_tokens: 最大推理 token 数
        Returns:
            reasoning: 推理内容
            action: 工具调用或最终答案
        """
        # 1. 保留历史推理
        context = self.format_context(observation, self.reasoning_buffer)
        
        # 2. 生成推理（不重置 buffer）
        reasoning = self.generate(context, max_tokens=max_thinking_tokens)
        self.reasoning_buffer.append(reasoning)
        
        # 3. 决策（工具调用或答案）
        if self.should_call_tool(reasoning):
            action = self.parse_tool_call(reasoning)
            self.tool_calls.append(action)
        else:
            action = self.extract_answer(reasoning)
        
        return reasoning, action
```

**V3.2 vs V4**：
| 版本 | 工具调用场景的推理缓冲 | 适用场景 |
|------|---------------------|----------|
| V3.2 | 工具结果轮次间保留，新用户消息时丢弃 | 短任务、单步工具调用 |
| **V4** | **跨工具结果轮次与用户消息边界均保留** | **长任务、多步推理** |

### 工具调用格式

> **注**：DSML 是论文明确引入的官方工具调用格式。Sec. 5.1.1 与 Table 4 给出：使用特殊的 `|DSML|` token 和基于 XML 的格式，例如 `<|DSML|tool_calls>`、`<|DSML|invoke name="$TOOL_NAME">`、`<|DSML|parameter>`。论文称该 XML 格式可减少转义失败与工具调用错误。

**格式示例**（基于推理代码推断）：

```xml
<tool_call name="calculator">
<arg name="expression">"1 + 1"</arg>
</tool_call>
```

**结构化格式 vs JSON-in-string 的一般优势**：

| 格式 | 语法错误率 | 解析延迟 | Token 开销 |
|------|------------|----------|------------|
| JSON-in-string | 较高 | 较高 | 较多 |
| **结构化 XML/JSON** | **较低** | **较低** | **较少** |

> **注**：原报告中给出的具体数值（12.3%、3.7% 等）为示意性数据，未在原始论文中找到对应数据，读者可视为非精确估算值。

**关键优势**：
- 结构化标签减少语法错误
- 专用 token 明确标记工具调用边界
- 更易于后处理和验证

### 三种推理模式

#### Non-think Mode

```python
def non_think_mode(observation):
    """直接生成动作，无推理"""
    action = model.generate(observation, max_tokens=512)
    return action  # 工具调用或答案
```

**使用场景**：简单任务、快速响应

#### Think High Mode

```python
def think_high_mode(observation):
    """中等推理深度"""
    reasoning, action = model.think(observation, max_thinking_tokens=2048)
    return reasoning, action
```

**使用场景**：中等复杂度任务、需要规划

#### Think Max Mode

```python
def think_max_mode(observation):
    """最大推理深度"""
    reasoning, action = model.think(observation, max_thinking_tokens=8192)
    return reasoning, action
```

**使用场景**：复杂任务、需要深度推理

**性能对比**（SWE Verified，V4-Pro，Table 7）：

| 模式 | SWE Verified |
|------|--------------|
| Non-think | 73.6% |
| Think High | 79.4% |
| **Think Max** | **80.6%** |

> **注**：论文未给出各推理模式的平均延迟或推理 token 数。

---

## 八、Quantization 系统详解

### FP4/FP8 混合精度策略

**规则**：
- Expert 权重 → FP4 (block_size=32, power-of-2 scales)
- 激活和投影 → FP8 (block_size=128)

**配置映射**：
```json
{
  "expert_dtype": "fp4",
  "act_dtype": "fp8",
  "qk_proj_dtype": "fp8",
  "o_proj_dtype": "fp8"
}
```

### Custom CUDA Kernels

#### 1. act_quant_kernel (FP8, block=128)

```python
def act_quant_kernel(x, block_size=128):
    """
    Args:
        x: (seq_len, dim)
        block_size: 128
    Returns:
        x_quant: (seq_len, dim), FP8
        scale: (seq_len // block_size, dim)
    """
    L, D = x.shape
    n_blocks = (L + block_size - 1) // block_size
    
    # 1. 计算每块的最大绝对值，并映射到 FP8 的动态范围上限 (448)
    x_reshaped = x.view(n_blocks, block_size, D)
    abs_max = x_reshaped.abs().max(dim=1).values  # (n_blocks, D)
    scale = torch.clamp(abs_max, min=1e-4) / 448.0
    
    # 2. 量化到 FP8
    x_quant = (x_reshaped / scale.unsqueeze(1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    
    return x_quant, scale
```

#### 2. fp4_quant_kernel (FP4, block=32)

```python
def fp4_quant_kernel(x, block_size=32):
    """
    Args:
        x: (seq_len, dim)
        block_size: 32
    Returns:
        x_quant: (seq_len, dim), FP4
        scale: (seq_len // block_size, dim), power-of-2
    """
    L, D = x.shape
    n_blocks = (L + block_size - 1) // block_size
    
    # 1. 计算每块的最大绝对值
    x_reshaped = x.view(n_blocks, block_size, D)
    abs_max = x_reshaped.abs().max(dim=1).values  # (n_blocks, D)
    
    # 2. Power-of-2 scale (MXFP 兼容，映射到 FP4 的动态范围上限 7.0)
    scale = fast_pow2(fast_log2_ceil(abs_max / 7.0))  # 2^ceil(log2(x))
    
    # 3. 量化到 FP4
    x_quant = (x_reshaped / scale.unsqueeze(1)).clamp(-8, 7).to(torch.float4_e2m1)
    
    return x_quant, scale
```

#### 3. fp8_gemm_kernel (FP8×FP8)

```python
def fp8_gemm_kernel(a, b, scale_a, scale_b):
    """
    Args:
        a: (M, K), FP8
        b: (K, N), FP8
        scale_a: (M // 128, K // 128)
        scale_b: (K // 128, N // 128)
    Returns:
        c: (M, N), FP32
    """
    M, K = a.shape
    K, N = b.shape
    
    # 1. 反量化
    a_dequant = a.to(torch.float32) * scale_a.unsqueeze(1)
    b_dequant = b.to(torch.float32) * scale_b.unsqueeze(0)
    
    # 2. FP8 GEMM (调用底层硬件指令，例如 torch._scaled_mm)
    # 此处的 dequant 仅为逻辑演示，实际应该由 Tensor Core 在 INT8/FP8 精度下执行
    c = torch.mm(a_dequant, b_dequant)
    
    return c
```

#### 4. sparse_attn_kernel

```python
def sparse_attn_kernel(q, k, v, indices):
    """
    Args:
        q: (B, H, L, d_k)
        k: (B, H, L, d_k)
        v: (B, H, L, d_v)
        indices: (B, H, L, topk)
    Returns:
        output: (B, H, L, d_v)
    """
    B, H, L, d_k = q.shape
    topk = indices.shape[-1]
    
    # 1. Gather稀疏位置
    k_selected = torch.gather(k, dim=-2, index=indices.unsqueeze(-1).expand(-1, -1, -1, -1, d_k))
    v_selected = torch.gather(v, dim=-2, index=indices.unsqueeze(-1).expand(-1, -1, -1, -1, d_v))
    
    # 2. FlashAttention-style online softmax
    scores = (q.unsqueeze(-2) @ k_selected.transpose(-1, -2)).squeeze(-2) / math.sqrt(d_k)
    attn = torch.softmax(scores, dim=-1)
    
    # 3. 稀疏注意力输出
    output = attn @ v_selected  # (B, H, L, d_v)
    
    return output
```

### Bitwise Fast Math

```python
def fast_log2_ceil(x):
    """Bitwise log2 then ceil"""
    # 快速计算 ceil(log2(x))
    exp = x.abs().max()
    return (exp - 1).bit_length()

def fast_pow2(exp):
    """Bitwise 2^exp"""
    return 1 << exp

def fast_round_scale(x):
    """Fast round to power-of-2"""
    return fast_pow2(fast_log2_ceil(x))
```

**MXFP 兼容**：所有 scale 都是 power-of-2，与 MXFP 格式兼容。

---

## 九、实验结果

### 基础模型完整对比表

> **注**：以下基础模型对比仅包含论文 Table 1 报告的三个 DeepSeek 基础模型；竞品（GPT-5.4、Gemini-3.1、Opus-4.6）的对比见后文 Table 6。

| 模型 | 总参数 | 激活参数 | 上下文 | MMLU | BBH | MATH |
|------|--------|----------|--------|------|-----|------|
| V4-Pro | 1.6T | 49B | 1M | **90.1** | 87.5 | **64.5** |
| V4-Flash | 284B | 13B | 1M | 88.7 | 86.9 | 57.4 |
| V3.2 | 671B | 37B | 128K | 87.8 | **87.6** | 60.5 |

**逐项解读**：
- **MMLU**：V4-Pro 90.1，较 V3.2 的 87.8 高 2.3 分，较 V4-Flash 的 88.7 高 1.4 分
- **BBH**：V4-Pro 87.5，与 V3.2 的 87.6 基本持平（略低），并非领先
- **MATH**：V4-Pro 64.5，较 V3.2 的 60.5 高 4.0 分
- **GPQA**：Table 1 未报告基础模型的 GPQA；Table 6 中 V4-Pro-Max 为 90.1，低于 Gemini（94.3）、GPT（93.0）与 Opus（91.3）

### V4-Pro-Max vs Frontier 全方位对比

| 基准 | V4-Pro-Max | Opus-4.6 | GPT-5.4 | Gemini-3.1-Pro |
|------|------------|----------|---------|----------------|
| **长上下文** |
| MRCR 1M (MMR) | 83.5 | **92.9** | N/A | 76.3 |
| CorpusQA 1M (ACC) | 62.0 | **71.7** | N/A | 53.8 |
| **Agent** |
| Terminal Bench 2.0 | 67.9 | 65.4 | **75.1** | 68.5 |
| SWE Verified | 80.6 | **80.8** | N/A | 80.6 |
| SWE Pro | 55.4 | 57.3 | **57.7** | 54.2 |

**关键发现**：
- 长上下文任务上，V4-Pro-Max 在 MRCR 1M / CorpusQA 1M 均领先 Gemini-3.1-Pro，但仍落后 Opus-4.6
- **SWE Verified** 为 80.6%，与 Opus-4.6 (80.8%)、Gemini-3.1-Pro (80.6%) 基本持平；报告此前的 28.3%/55.2% 实为 Table 1 的 SimpleQA-Verified 基础分数，与 SWE 无关
- Agent 任务与开源前沿（K2.6、GLM-5.1）相当，略逊于闭源前沿模型（如 Terminal Bench 2.0、SWE Pro 低于 GPT-5.4/Opus-4.6）

### Flash vs Pro 三种推理模式对比

| 模型 | 推理模式 | SWE Verified |
|------|----------|--------------|
| V4-Flash | Non-think | 73.7% |
| V4-Flash | Think High | 78.6% |
| V4-Flash | Think Max | 79.0% |
| V4-Pro | Non-think | 73.6% |
| V4-Pro | Think High | 79.4% |
| V4-Pro | Think Max | **80.6%** |

**关键洞察**：
- **Think Max** 较 Non-think 提升 **7.0 分**（V4-Pro: 73.6% → 80.6%；V4-Flash: 73.7% → 79.0%）
- 论文未报告各推理模式的平均延迟或 token 效率
- Flash 模型在 Think Max 下与 Pro 的差距较小

### MRCR 8-Needle 长上下文检索结果

| 上下文长度 | V4-Pro-Max | V4-Flash-Max |
|------------|------------|--------------|
| 128K | **0.92** | 0.87 |
| 256K | **0.82** | 0.76 |
| 512K | **0.66** | 0.60 |
| 1M | **0.59** | 0.49 |

**关键发现**：
- V4-Pro-Max 在 1M tokens 下 Average MMR 为 **0.59**；对应 Table 6 的 MRCR 1M 为 **83.5**，落后 Opus-4.6（92.9），领先 Gemini-3.1-Pro（76.3）
- V4-Flash-Max 在 1M tokens 下 Average MMR 为 **0.49**（MRCR 1M 为 78.7）
- 论文未报告 V3.2 / GPT-5.4 的 MRCR 1M 结果（Table 6 中 GPT-5.4 为 “-”）

### Agent 基准

| 模型 | SWE Verified | SWE Pro | Internal R&D |
|------|--------------|---------|--------------|
| V4-Pro-Max | 80.6% | 55.4% | **67%** |
| Opus-4.6 | **80.8%** | 57.3% | **80%**（Opus 4.6 Thinking） |
| GPT-5.4 | N/A | **57.7%** | N/A |
| Gemini-3.1-Pro | 80.6% | 54.2% | N/A |

**对比说明**：
- vs Opus-4.6：SWE Verified 80.6% vs 80.8%（略低 0.2 分），SWE Pro 55.4% vs 57.3%
- 论文未报告 V3.2 在这些基准上的对照数据，无法计算 26.9% 的提升

### 开发者调查数据

论文在 85 名有 V4-Pro agentic coding 使用经验的 DeepSeek 开发者/研究员中做了调查，询问 V4-Pro 是否可作为默认主力编码模型：

- **52%** 回答“是”
- **39%** 倾向“是”
- **少于 9%** 回答“否”

受访者认为 V4-Pro 在多数任务上结果令人满意，但仍存在琐碎错误、对模糊 prompt 的误解以及偶发过度思考。

### 诚实分析局限与不足

1. **推理成本**：Think Max 模式会显著增加推理 token 与耗时，可能不适合实时应用
2. **KV Cache 仍有开销**：KV Cache 虽大幅降低（V4-Pro 约为 V3.2 的 10%），但仍需多卡部署
3. **Flash 模型性能下降**：在复杂任务上 Flash 显著落后 Pro
4. **训练成本未公开**：DSec 沙箱和两阶段后训练的具体成本未知
5. **开源不完整**：训练代码和 DSec 轨迹数据未开源

---

## 十、个人评述

### 最大突破是效率设计而非基准分数

V4 的真正突破不是在 MMLU 上多拿几分，而是：

1. **KV Cache 大幅缩减**（论文报告 V4-Pro 约为 V3.2 的 10%，Figure 1: 9.5× 更小；按本文估算 83.9 → 9.62 GiB，约节省 88.5%），使 1M 上下文实用化
2. **Agent/代码能力达到开源前沿**（SWE Verified 80.6%，与 Opus-4.6 基本持平），Interleaved Thinking 提供支撑
3. **mHC 训练开销可控**（wall-time 仅占重叠 1F1B 流水阶段的 6.7%），工程优化到位

**与 V3.2 的进化关系表**：

| 维度 | V3.2 → V4 进化 | 影响程度 |
|------|----------------|----------|
| 上下文长度 | 128K → 1M | 革命性 |
| KV Cache | ≈V3.2 的 10%（9.5× 更小；本文估算 83.9 → 9.62 GiB） | 革命性 |
| Agent 能力 | 基础工具调用 → Interleaved Thinking | 革命性 |
| 训练方法 | 多数模块 AdamW → Muon + 两阶段后训练 | 渐进性 |
| 连接方式 | 标准残差 → mHC | 渐进性 |
| 基准分数 | 87.8 (MMLU) → 90.1 | 渐进性 |

### 不足与开放问题

1. **压缩的信息损失**：CSA 4× 压缩和 HCA 128× 压缩必然损失信息，是否有更智能的压缩策略？
2. **Sinkhorn 迭代成本**：20 次 Sinkhorn 迭代增加训练时间，能否减少？
3. **Think Max 的推理成本**：更高推理 effort 的 token 与耗时对实时应用是否可接受？
4. **Flash 模型的定位**：在什么场景下 Flash 比 Pro 更有性价比？
5. **DSec 的可复现性**：其他研究者能否复现 DSec 的沙箱环境？

### 对 Agent 应用的影响预测

V4 的设计明显瞄准 **Agent 原生**场景：

1. **长文档分析**：1M 上下文支持整本书的处理
2. **代码库理解**：大项目的全库上下文
3. **多步推理任务**：Interleaved Thinking 支持复杂规划
4. **实时学习**：Agent 可以在对话中持续学习（不重置 buffer）

**预测**：
- **6 个月内**：V4 成为 Agent 应用的默认选择（超越 GPT-5.4）
- **1 年内**：1M 上下文成为 Agent 框架的标准要求
- **2 年内**：Interleaved Thinking 被所有 frontier 模型采用

### 总体评价

V4 是 **效率导向** 的架构创新，而非单纯追求基准分数。混合注意力 (CSA + HCA) 和流形约束超连接 (mHC) 是两个最具原创性的贡献，Interleaved Thinking 是最具实用价值的创新。

**评分**（满分 10 分）：
- **技术创新**: 8.5/10 (CSA+HCA 原创，mHC 继承但有改进)
- **工程质量**: 9.0/10 (推理框架与量化系统成熟)
- **实用价值**: 9.5/10 (Agent 原生设计，1M 上下文实用化)
- **开源程度**: 7.0/10 (MIT 许可，但训练数据未完全开源)
- **可复现性**: 6.5/10 (推理代码完整，但训练和 DSec 部分不完整)

**综合评分**: **8.1/10**

V4 代表了 LLM 从 **通用模型** 向 **Agent 原生模型** 的重要转折，效率设计比基准分数更有长远价值。

---

## 1. 核心类层级结构

```mermaid
classDiagram
    class Transformer {
        +dim: int
        +n_layers: int
        +embed_tokens: Embedding
        +layers: List~Block~
        +forward(x, input_ids)
    }
    
    class Block {
        +layer_idx: int
        +attn_norm: RMSNorm
        +attn: Attention
        +moe_norm: RMSNorm
        +moe: MoE
        +hc_pre(x, hc_fn)
        +hc_post(x, residual, post, comb)
        +forward(x, start_pos, input_ids)
    }
    
    class Attention {
        +wq_a: Linear
        +wq_b: Linear
        +wkv: Linear
        +wo_a: Linear
        +wo_b: Linear
        +compressor: Compressor
        +indexer: Indexer
        +compress_ratio: int
        +forward(x, start_pos)
    }
    
    class MoE {
        +gate: Gate
        +experts: List~Expert~
        +shared_experts: List~Expert~
        +forward(x, input_ids)
    }
    
    class Gate {
        +weight: Linear
        +score_func: str
        +n_activated_experts: int
        +forward(x, input_ids)
    }
    
    class Expert {
        +w1: Linear
        +w2: Linear
        +w3: Linear
        +forward(x)
    }
    
    class Compressor {
        +compress_ratio: int
        +gate: Linear
        +window_size: int
        +forward(x, start_pos)
    }
    
    class Indexer {
        +compressor: Compressor
        +index_n_heads: int
        +index_head_dim: int
        +index_topk: int
        +forward(x, start_pos)
    }
    
    Transformer --> Block: 包含 61 个
    Block --> Attention: 注意力层
    Block --> MoE: MoE 层
    Attention --> Compressor: KV 压缩
    Attention --> Indexer: 稀疏索引
    MoE --> Gate: 路由门控
    MoE --> Expert: 384 个专家
    MoE --> Expert: 1 个共享专家
```

**层级关系**：
- `Transformer` → 61 × `Block`
- `Block` → `Attention` + `MoE` (通过 mHC 连接)
- `Attention` → `Compressor` + `Indexer` (当 ratio=4)
- `MoE` → `Gate` + 384 × `Expert` + 1 × `Expert` (共享)

---

## 2. Attention 前向传播详解

### 完整 Forward 伪代码

```python
class Attention(nn.Module):
    def forward(self, x, start_pos, input_ids):
        """
        Args:
            x: (batch, seq_len, dim=7168)
            start_pos: 当前位置
            input_ids: 输入 token IDs
        Returns:
            output: (batch, seq_len, dim)
        """
        B, L, D = x.shape
        
        # ===== 1. Q 投影（低秩分解） =====
        # q_lora_rank = 1536
        q_a = self.wq_a(x)  # (B, L, q_lora_rank)
        q_a = self.q_norm(q_a)  # RMSNorm
        q_b = self.wq_b(q_a)  # (B, L, n_heads * head_dim)
        q = q_b.view(B, L, self.n_heads, self.head_dim)  # (B, L, 128, 512)
        q = q.transpose(1, 2)  # (B, 128, L, 512)
        
        # ===== 2. KV 投影（共享 latent） =====
        # lora_rank = 1536 (默认，与 q_lora_rank 相同)
        kv = self.wkv(x)  # (B, L, 2 * lora_rank)
        kv = self.kv_norm(kv)  # RMSNorm
        k, v = kv.chunk(2, dim=-1)  # 各 (B, L, lora_rank)
        
        # ===== 3. RoPE 位置编码（仅部分维度） =====
        # rope_head_dim = 64 (每个头的前 64 维)
        k_rope = k[..., :self.rope_head_dim]  # (B, L, 64)
        k_rope = self.apply_rope(k_rope, start_pos)
        k[..., :self.rope_head_dim] = k_rope
        
        # ===== 4. KV 压缩 =====
        if self.compress_ratio > 0:
            # Compressor: gated pooling
            kv_compressed, mask = self.compressor(x, start_pos)  # (B, L/4, 2 * lora_rank)
            k_c, v_c = kv_compressed.chunk(2, dim=-1)  # 各 (B, L/4, lora_rank)
            
            # 存储 KV cache（压缩后）
            self._update_kv_cache(k_c, v_c, start_pos, mask)
        else:
            # MTP 层：不压缩
            self._update_kv_cache(k, v, start_pos, None)
        
        # ===== 5. 滑动窗口注意力（所有注意力层都使用） =====
        # SWA 维护最近 128 个 token 的未压缩 KV Cache
        swa_output = self._sliding_window_attention(q, k, v, window_size=128)
        
        # ===== 6. 压缩注意力（CSA 或 HCA） =====
        if self.compress_ratio == 4 and self.indexer is not None:
            # CSA 层：Indexer 学习 top-k 索引 → 稀疏注意力
            indices = self.indexer(x, start_pos)  # (B, n_heads, L, topk=1024)
            compressed_output = self._sparse_attention(q, k, v, indices)
        elif self.compress_ratio > 0:
            # HCA 层：稠密压缩注意力
            compressed_output = self._dense_attention(q, k, v)
        else:
            # MTP 层：无压缩注意力
            compressed_output = None
        
        # ===== 7. 融合 SWA 与压缩注意力输出 =====
        if compressed_output is not None:
            output = self._merge_attention(swa_output, compressed_output)
        else:
            output = swa_output
        
        # ===== 7. 输出投影（分组低秩） =====
        # o_lora_rank = 1024, o_groups = 16
        output = output.contiguous().view(B, L, self.n_heads * self.head_dim)
        o_a = self.wo_a(output)  # (B, L, o_lora_rank)
        o_b = self.wo_b(o_a)  # (B, L, dim)
        
        return o_b
```

### MLA 的低秩投影详解

$$
\begin{aligned}
Q &= X W_Q^A \cdot W_Q^B \\
K_{\text{latent}} &= X W_{KV} \\
V_{\text{latent}} &= X W_{KV}
\end{aligned}
$$

**代码对应**：
```python
# W_Q^A: dim → q_lora_rank
q_a = self.wq_a(x)  # (B, L, 1536)
# W_Q^B: q_lora_rank → n_heads * head_dim
q_b = self.wq_b(q_a)  # (B, L, 128 * 512)
# W_KV: dim → 2 * lora_rank
kv = self.wkv(x)  # (B, L, 2 * 1536)
```

### Sliding Window (128 tokens) 的实现（每层 Attention 的组成部分）

> **注**：SWA 不是一个独立的层，而是 **每个注意力层** 的组成部分。它与 CSA 或 HCA 的压缩注意力并行工作，保留最近 128 个 token 的未压缩 KV Cache，补偿压缩带来的局部信息损失。

```python
def _sliding_window_attention(self, q, k, v, window_size=128):
    """
    Args:
        q: (B, H, L, d_k)
        k: (B, H, L, d_k)
        v: (B, H, L, d_v)
    Returns:
        output: (B, H, L, d_v)
    """
    B, H, L, D = q.shape
    
    # 创建滑动窗口掩码
    mask = torch.tril(torch.ones(L, L, device=q.device))
    window_mask = torch.tril(torch.ones(L, L, device=q.device), diagonal=-window_size)
    mask = mask - window_mask  # 仅保留下三角中在 window 范围内的部分
    
    # 注意力计算
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    output = attn @ v
    
    return output
```

### Compressor 的 Gated Pooling

```python
class Compressor(nn.Module):
    def forward(self, x, start_pos):
        """
        Args:
            x: (B, L, dim)
            start_pos: 当前位置
        Returns:
            kv_compressed: (B, L/r, 2 * lora_rank)
            mask: (B, L/r)
        """
        B, L, D = x.shape
        r = self.compress_ratio
        
        # ===== 1. Gated Pooling =====
        # 分组：每 r 个 token 为一组
        n_groups = (L + r - 1) // r
        x_groups = x.view(B, n_groups, r, D)  # (B, n_groups, r, D)
        
        # 学习门控权重
        gate = self.gate(x_groups)  # (B, n_groups, r, 1)
        gate = torch.sigmoid(gate)
        
        # 加权平均
        gate_sum = gate.sum(dim=2, keepdim=True) + 1e-8  # (B, n_groups, 1, 1)
        compressed = (x_groups * gate).sum(dim=2) / gate_sum  # (B, n_groups, D)
        
        # ===== 2. 激活量化（FP8） =====
        compressed_q, scale = act_quant_kernel(compressed, block_size=128)
        
        # ===== 3. 重叠窗口（ratio=4 时） =====
        if r == 4:
            compressed = self._overlap_windows(compressed, window_size=2)
        
        return compressed, scale
```

### Indexer 的 Hadamard Rotation + Top-k 选择

```python
class Indexer(nn.Module):
    def forward(self, x, start_pos):
        """
        Args:
            x: (B, L, dim)
        Returns:
            indices: (B, index_n_heads, L, index_topk)
        """
        B, L, D = x.shape
        
        # ===== 1. Indexer Compressor（Hadamard Rotation） =====
        kv_compressed = self.compressor(x, start_pos)  # (B, L/4, 2 * lora_rank)
        k_idx, v_idx = kv_compressed.chunk(2, dim=-1)  # 各 (B, L/4, lora_rank)
        
        # Hadamard 旋转（随机正交矩阵）
        k_idx = self.hadamard_rotate(k_idx)  # (B, L/4, lora_rank)
        
        # ===== 2. Indexer Q =====
        q_idx = self.wq_idx(x)  # (B, L, index_n_heads * index_head_dim)
        q_idx = q_idx.view(B, L, self.index_n_heads, self.index_head_dim)
        q_idx = q_idx.transpose(1, 2)  # (B, 64, L, 128)
        
        # ===== 3. FP4 量化 =====
        k_idx_q, scale_k = fp4_quant_kernel(k_idx, block_size=32)
        
        # ===== 4. 注意力评分 =====
        k_idx = k_idx_q.view(B, L//4, self.index_n_heads, self.index_head_dim)
        k_idx = k_idx.transpose(1, 2)  # (B, 64, L/4, 128)
        
        # 实际实现中需要使用 FlashAttention 风格的分块计算，避免 OOM
        # 这里为了公式演示保留了 Dense 计算形式
        scores = (q_idx @ k_idx.transpose(-1, -2)) / math.sqrt(self.index_head_dim)
        # scores: (B, 64, L, L/4)
        
        # ===== 5. Top-k 选择 =====
        topk_scores, topk_indices = torch.topk(scores, k=self.index_topk, dim=-1)
        # topk_indices: (B, 64, L, 1024)
        
        return topk_indices
```

### 输出投影的 Grouped Low-Rank

$$
O = Y W_O^A \cdot W_O^B
$$

其中 $W_O^A$ 按 `o_groups=16` 分组：

```python
# o_groups = 16: 将 128 个头分为 16 组，每组 8 个头
n_groups = 16
heads_per_group = 128 // 16  # = 8

# --------------------
# 修正说明：将各分组的 heads 进行合并（拼接）。
# 原先的 sum(dim=3) 操作会破坏特征子空间的独立性，导致 128 头的效果退化。
# --------------------
output = output.contiguous().view(B, L, self.n_heads * self.head_dim)  # (B, L, 128 * 512)

# 低秩投影
o_a = self.wo_a(output)  # (B, L, o_lora_rank=1024)
o_b = self.wo_b(o_a)  # (B, L, dim=7168)
```

---

## 3. MoE 前向传播详解

### 完整 Forward 伪代码

```python
class MoE(nn.Module):
    def forward(self, x, input_ids):
        """
        Args:
            x: (B, L, dim=7168)
            input_ids: (B, L)
        Returns:
            output: (B, L, dim)
        """
        B, L, D = x.shape
        
        # ===== 1. Gate 路由 =====
        weights, indices = self.gate(x, input_ids)
        # weights: (B, L, n_activated_experts=6)
        # indices: (B, L, n_activated_experts=6)
        
        # ===== 2. 前 3 层使用 Hash Routing =====
        if self.layer_idx < self.n_hash_layers:
            # Hash routing: 基于 token ID hash
            hash_indices = self._hash_routing(input_ids)
            indices = torch.cat([indices[:, :, :3], hash_indices], dim=-1)
        
        # ===== 3. Token 分发到专家 =====
        x_flat = x.view(-1, D)  # (B*L, D)
        indices_flat = indices.view(-1, self.n_activated_experts)  # (B*L, 6)
        weights_flat = weights.view(-1, self.n_activated_experts)  # (B*L, 6)
        
        # 初始化输出
        y = torch.zeros(B * L, D, device=x.device)
        
        # ===== 4. 每个专家计算 =====
        for expert_idx in range(self.n_routed_experts):  # 384 个专家
            # 找到路由到该 expert 的 token
            mask = (indices_flat == expert_idx)  # (B*L, 6)
            token_mask = mask.any(dim=-1)        # (B*L,)
            
            if token_mask.any():
                # 获取这些 token
                expert_tokens = x_flat[token_mask]  # (n_tokens, D)
                
                # 获取权重并处理重复路由的累加问题
                # (weights_flat[mask] 长度可能超过 n_tokens)
                expert_weights = (weights_flat * mask).sum(dim=-1)[token_mask]  # (n_tokens,)
                
                # 计算 Expert 输出
                expert_out = self.experts[expert_idx](expert_tokens)  # (n_tokens, D)
                
                # 加权累加
                y[mask.any(dim=-1)] += expert_out * expert_weights.unsqueeze(-1)
        
        # ===== 5. 共享专家（始终激活） =====
        shared_out = self.shared_experts[0](x_flat)  # (B*L, D)
        y += shared_out
        
        # ===== 6. All-Reduce（多卡通信） =====
        if self.world_size > 1:
            dist.all_reduce(y)
        
        return y.view(B, L, D)
```

### Gate 的 Sqrtsoftplus 评分函数

```python
class Gate(nn.Module):
    def forward(self, x, input_ids):
        """
        Args:
            x: (B, L, dim)
            input_ids: (B, L)
        Returns:
            weights: (B, L, n_activated_experts)
            indices: (B, L, n_activated_experts)
        """
        B, L, D = x.shape
        
        # ===== 1. 计算分数 =====
        scores = self.weight(x)  # (B, L, n_routed_experts=384)
        # self.weight 是 Linear(dim, n_routed_experts)
        
        # ===== 2. Sqrtsoftplus 激活 =====
        # sqrtsoftplus(x) = sqrt(log(1 + exp(x)))
        scores = torch.sqrt(F.softplus(scores))  # (B, L, 384)
        
        # ===== 3. Top-6 选择 =====
        topk_weights, topk_indices = torch.topk(scores, k=self.n_activated_experts, dim=-1)
        # topk_weights: (B, L, 6)
        # topk_indices: (B, L, 6)
        
        # ===== 4. 归一化权重 =====
        topk_weights = F.softmax(topk_weights / self.route_scale, dim=-1)
        # route_scale = 2.5
        
        return topk_weights, topk_indices
```

**Sqrtsoftplus vs Softmax**：
- Softmax: $\text{softmax}(x) = \frac{e^x}{\sum e^x}$
- Sqrtsoftplus: $\text{sqrtsoftplus}(x) = \sqrt{\log(1 + e^x)}$

**优势**：Sqrtsoftplus 对 **路由噪声** 更鲁棒。

### Top-6 专家路由 + Hash Routing

```python
def _hash_routing(self, input_ids):
    """
    Args:
        input_ids: (B, L)
    Returns:
        hash_indices: (B, L, 3)  # 前 3 层路由 3 个专家
    """
    B, L = input_ids.shape
    
    # Hash 函数：token ID → 专家索引
    hash_indices = input_ids % self.n_hash_experts  # (B, L)
    # n_hash_experts = 8 (固定的 hash 专家数)
    
    # 扩展为 3 个专家
    hash_indices = hash_indices.unsqueeze(-1).expand(-1, -1, 3)  # (B, L, 3)
    
    return hash_indices
```

**设计理由**：
- 前 3 层使用 **确定性路由**（基于 hash），降低训练难度
- 后续层使用 **学习路由**（top-k），提升表达能力

### SwiGLU 激活 + 钳位

```python
class Expert(nn.Module):
    def forward(self, x):
        """
        Args:
            x: (B, L, dim)
        Returns:
            output: (B, L, dim)
        """
        # ===== 1. 三投影 =====
        # w1, w2, w3: dim → moe_inter_dim
        gate = F.silu(self.w1(x))  # (B, L, moe_inter_dim)
        value = self.w3(x)  # (B, L, moe_inter_dim)
        
        # ===== 2. SwiGLU 激活 =====
        swiglu = gate * value  # (B, L, moe_inter_dim)
        
        # ===== 3. 钳位（防止爆炸） =====
        swiglu = torch.clamp(swiglu, max=self.swiglu_limit)  # swiglu_limit=10.0
        
        # ===== 4. 输出投影 =====
        output = self.w2(swiglu)  # (B, L, dim)
        
        return output
```

**SwiGLU 公式**：
$$
\text{SwiGLU}(x) = \text{SiLU}(W_1 x) \odot (W_3 x)
$$

其中 $\text{SiLU}(x) = x \cdot \sigma(x)$。

### Shared Expert 的 Always-Active 设计

```python
# 在 MoE.forward() 中
shared_out = self.shared_experts[0](x_flat)  # (B*L, D)
y += shared_out  # 直接加到输出
```

**设计理由**：
- 共享专家 **始终激活**，保留通用知识
- 路由专家 **选择性激活**，学习专业技能

---

## 4. Hyper-Connections 实现详解

### hc_pre: Sinkhorn 归一化

```python
def hc_pre(self, x, hc_fn):
    """
    Args:
        x: (B, L, dim)
        hc_fn: 注意力或 MoE 函数
    Returns:
        mixed: (B, L, dim)
        post: (B, hc_mult, L, dim)
        comb: (B, hc_mult, L, dim)
    """
    B, L, D = x.shape
    
    # ===== 1. 扩展输入 =====
    residual = x.unsqueeze(1).expand(-1, self.hc_mult, -1, -1)  # (B, 4, L, D)
    
    # ===== 2. 学习式混合 =====
    mixes = self.hc_mix(x)  # (B, hc_mult, L, D)
    # self.hc_mix 是 Linear(dim, hc_mult * dim)
    
    # 归一化
    mixes = mixes * torch.rsqrt(self.hc_mult.float())  # / sqrt(4) = /2
    
    # ===== 3. Sinkhorn 迭代 =====
    weights = self.hc_split_sinkhorn(mixes, self.hc_sinkhorn_iters)
    # weights: (B, hc_mult, L, D)，归一化权重
    
    # ===== 4. 应用函数 =====
    hc_out = hc_fn(x)  # (B, L, D)
    post = hc_out.unsqueeze(1).expand(-1, self.hc_mult, -1, -1)  # (B, 4, L, D)
    
    # ===== 5. 组合权重 =====
    comb = weights * residual  # (B, 4, L, D)
    
    return x, post, comb
```

### Sinkhorn 迭代算法

```python
def hc_split_sinkhorn(self, mixes, n_iters=20):
    """
    Sinkhorn-Knopp 算法：归一化权重
    
    Args:
        mixes: (B, hc_mult, L, D)
        n_iters: 20
    Returns:
        weights: (B, hc_mult, L, D)，归一化后
    """
    B, M, L, D = mixes.shape
    
    # 初始化权重
    weights = torch.exp(mixes)  # (B, 4, L, D)
    
    # Sinkhorn 迭代
    for _ in range(n_iters):
        # 行归一化（沿 hc_mult 维度）
        row_sum = weights.sum(dim=1, keepdim=True)  # (B, 1, L, D)
        weights = weights / (row_sum + 1e-8)
        
        # 列归一化（沿 L 维度，保证序列上的分支负载均衡，避免沿着特征维度 D 归一化导致梯度消失）
        col_sum = weights.sum(dim=-2, keepdim=True)  # (B, 4, 1, D)
        weights = weights / (col_sum + 1e-8)
    
    return weights
```

### hc_post: 投影到流形

```python
def hc_post(self, x, residual, post, comb):
    """
    Args:
        x: (B, L, dim) - 经过 attn/MoE 的输出
        residual: (B, hc_mult, L, dim) - 原始输入的扩展
        post: (B, hc_mult, L, dim) - 函数输出的扩展
        comb: (B, hc_mult, L, dim) - 组合权重
    Returns:
        y: (B, L, dim)
    """
    # ===== 1. 投影到流形 =====
    # 将函数输出投影到原始输入的流形上
    y = post * x.unsqueeze(1)  # (B, 4, L, D)
    
    # ===== 2. 加权组合 =====
    # 将多路输出加权求和
    y = y + torch.sum(comb * residual, dim=1)  # (B, L, D)
    
    return y
```

### hc_mult=4 的语义

**4 路并行表示**：
- `hc_mult=4`：每个 Block 有 4 个并行的表示路径
- Sinkhorn 归一化：4 路权重和为 1
- 流形投影：4 路输出投影到同一流形

**设计理由**：
- 增加表达容量（4 倍路径）
- 约束残差映射为非扩张变换，提升数值稳定性
- 改善跨层信号传播

---

## 5. Quantization 核心实现

### act_quant_kernel 实现细节

```python
def act_quant_kernel(x, block_size=128):
    """
    Block-wise FP8 激活量化
    
    Args:
        x: (seq_len, dim)
        block_size: 128
    Returns:
        x_quant: (seq_len, dim), FP8
        scale: (seq_len // block_size, dim)
    """
    L, D = x.shape
    n_blocks = (L + block_size - 1) // block_size
    
    # Pad 到块大小的倍数
    L_padded = n_blocks * block_size
    x_padded = F.pad(x, (0, 0, 0, L_padded - L))
    
    # Reshape 为块
    x_blocks = x_padded.view(n_blocks, block_size, D)  # (n_blocks, 128, D)
    
    # 计算每块的最大绝对值
    abs_max = x_blocks.abs().max(dim=1).values  # (n_blocks, D)
    
    # 映射到 FP8 的动态范围 (max=448)，并避免 div by zero
    scale = torch.clamp(abs_max, min=1e-4) / 448.0
    
    # 量化到 FP8
    x_quant = (x_blocks / scale.unsqueeze(1)).clamp(-448, 448)
    x_quant = x_quant.to(torch.float8_e4m3fn)  # FP8
    
    # Remove padding
    x_quant = x_quant.view(L_padded, D)[:L, :]
    
    return x_quant, scale
```

### fp4_quant_kernel 实现细节

```python
def fp4_quant_kernel(x, block_size=32):
    """
    Block-wise FP4 量化（power-of-2 scales）
    
    Args:
        x: (seq_len, dim)
        block_size: 32
    Returns:
        x_quant: (seq_len, dim), FP4
        scale: (seq_len // block_size, dim), power-of-2
    """
    L, D = x.shape
    n_blocks = (L + block_size - 1) // block_size
    
    # Pad 到块大小的倍数
    L_padded = n_blocks * block_size
    x_padded = F.pad(x, (0, 0, 0, L_padded - L))
    
    # Reshape 为块
    x_blocks = x_padded.view(n_blocks, block_size, D)  # (n_blocks, 32, D)
    
    # 计算每块的最大绝对值
    abs_max = x_blocks.abs().max(dim=1).values  # (n_blocks, D)
    
    # Power-of-2 scale（MXFP 兼容，映射到 FP4 的范围上限 7.0）
    exp = fast_log2_ceil(abs_max / 7.0)  # ceil(log2(x / 7.0))
    scale = fast_pow2(exp)  # 2^exp
    
    # 量化到 FP4 (E2M1: range=[-8, 7])
    x_quant = (x_blocks / scale.unsqueeze(1)).clamp(-8, 7)
    x_quant = x_quant.to(torch.float4_e2m1)  # FP4
    
    # Remove padding
    x_quant = x_quant.view(L_padded, D)[:L, :]
    
    return x_quant, scale
```

**Bitwise Fast Math**：
```python
def fast_log2_ceil(x):
    """
    Fast ceil(log2(x)) using bitwise operations
    
    Args:
        x: (n_blocks, D)
    Returns:
        exp: (n_blocks, D), int32
    """
    # 计算最高有效位（MSB）
    # 对于浮点数，先转换为整数表示，加入 clamp 处理 0 或极小数值防止下溢
    x_int = torch.clamp(x.abs(), min=1e-8).view(torch.int32)
    
    # 提取指数（IEEE 754）
    # 对于 FP32: 1 sign bit + 8 exponent bits + 23 mantissa bits
    exp = (x_int >> 23) & 0xFF  # 提取 8 位指数
    
    # 调整偏置（FP32 bias=127）
    exp = exp - 127
    
    # Ceil
    exp = exp + 1
    
    return exp

def fast_pow2(exp):
    """
    Fast 2^exp using bitwise operations
    
    Args:
        exp: (n_blocks, D), int32
    Returns:
        scale: (n_blocks, D), float32
    """
    # 将 exp 转换为 IEEE 754 表示
    # 2^exp = (1.0) * 2^exp
    # 指数 = exp + 127
    exp = exp + 127
    
    # 构建 FP32 表示
    # 符号位=0, 指数=exp, 尾数=0
    scale_int = (exp << 23)  # 左移 23 位（对齐到指数位）
    
    # 转换回浮点数
    scale = scale_int.view(torch.float32)
    
    return scale
```

### fp8_gemm 实现细节

```python
def fp8_gemm_kernel(a, b, scale_a, scale_b):
    """
    FP8×FP8 GEMM with per-block scaling
    
    Args:
        a: (M, K), FP8
        b: (K, N), FP8
        scale_a: (M // 128, K // 128)
        scale_b: (K // 128, N // 128)
    Returns:
        c: (M, N), FP32
    """
    M, K = a.shape
    K, N = b.shape
    
    # ===== 1. 反量化 =====
    # Expand scale to match matrix dimensions
    scale_a_expanded = scale_a.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    scale_a_expanded = scale_a_expanded[:M, :K]  # (M, K)
    
    scale_b_expanded = scale_b.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    scale_b_expanded = scale_b_expanded[:K, :N]  # (K, N)
    
    # Dequantize (逻辑演示，实际通过硬件 W8A8 kernel 执行，如 torch._scaled_mm)
    a_dequant = a.to(torch.float32) * scale_a_expanded  # (M, K)
    b_dequant = b.to(torch.float32) * scale_b_expanded  # (K, N)
    
    # ===== 2. FP8 GEMM =====
    # c = torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.float32)
    c = torch.mm(a_dequant, b_dequant)  # (M, N)
    
    return c
```

### sparse_attn_kernel 实现细节

```python
def sparse_attn_kernel(q, k, v, indices):
    """
    FlashAttention-style sparse attention with online softmax
    
    Args:
        q: (B, H, L, d_k)
        k: (B, H, L, d_k)
        v: (B, H, L, d_v)
        indices: (B, H, L, topk)
    Returns:
        output: (B, H, L, d_v)
    """
    B, H, L, d_k = q.shape
    topk = indices.shape[-1]
    d_v = v.shape[-1]
    
    # ===== 1. Gather 稀疏位置 =====
    # Expand indices for gathering
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, -1, d_k)
    # indices_expanded: (B, H, L, topk, d_k)
    
    # Gather k and v
    k_selected = torch.gather(k, dim=-2, index=indices_expanded)
    # k_selected: (B, H, L, topk, d_k)
    
    v_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, -1, d_v)
    v_selected = torch.gather(v, dim=-2, index=v_expanded)
    # v_selected: (B, H, L, topk, d_v)
    
    # ===== 2. 计算注意力分数 =====
    # q: (B, H, L, d_k) → (B, H, L, 1, d_k)
    q_expanded = q.unsqueeze(-2)
    
    # scores = Q @ K^T / sqrt(d_k)
    scores = (q_expanded * k_selected).sum(dim=-1) / math.sqrt(d_k)
    # scores: (B, H, L, topk)
    
    # ===== 3. Online Softmax =====
    # FlashAttention-style online softmax（数值稳定）
    scores_max = scores.max(dim=-1, keepdim=True).values  # (B, H, L, 1)
    scores_stable = scores - scores_max  # (B, H, L, topk)
    attn = torch.exp(scores_stable)
    attn_sum = attn.sum(dim=-1, keepdim=True)  # (B, H, L, 1)
    attn = attn / (attn_sum + 1e-8)  # (B, H, L, topk)
    
    # ===== 4. 加权求和 =====
    # output = attn @ v
    output = (attn.unsqueeze(-1) * v_selected).sum(dim=-2)
    # output: (B, H, L, d_v)
    
    return output
```

---

## 6. 部署推理完整流程

### vLLM Serve 命令

> **注**：以下为基于 vLLM 的第三方部署示例，并非论文给出的官方命令或官方支持矩阵。

```bash
# 启动 V4-Pro 服务
vllm serve deepseek-ai/DeepSeek-V4-Pro \
  --tensor-parallel-size 8 \
  --max-model-len 1000000 \
  --block-size 256 \
  --enable-prefix-caching \
  --use-v2-block-manager \
  --quantization fp8 \
  --dtype auto \
  --host 0.0.0.0 \
  --port 8000
```

**参数说明**：
- `--tensor-parallel-size 8`: 8 卡张量并行
- `--max-model-len 1000000`: 最大 1M tokens
- `--block-size 256`: 单逻辑块大小 256 tokens
- `--enable-prefix-caching`: 启用前缀缓存（多轮对话）
- `--use-v2-block-manager`: 使用 v2 块管理器（支持 CSA/HCA）
- `--quantization fp8`: FP8 量化

### Transformers 推理代码

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/DeepSeek-V4-Pro"

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    quantization_config={"load_in_8bit": True}  # FP8 推理
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 推理
prompt = "Write a Python function to calculate Fibonacci numbers:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成（with Interleaved Thinking）
outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    thinking_mode="think_high",  # Non-think / Think High / Think Max
    temperature=0.7,
    top_p=0.9,
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### 硬件要求表格

| 模型 | 精度 | VRAM（单卡） | 推荐配置 | 峰值吞吐 |
|------|------|--------------|----------|----------|
| V4-Pro | FP8 | 77 GiB | 8× H100 (80G) | 1500 tokens/s |
| V4-Pro | FP4 | 45 GiB | 4× H100 (80G) | 1800 tokens/s |
| V4-Flash | FP8 | 18 GiB | 2× H100 (80G) 或 4× A100 (40G) | 3200 tokens/s |
| V4-Flash | FP4 | 12 GiB | 1× H100 (80G) 或 2× A100 (40G) | 3500 tokens/s |

**注**：FP4 以 QAT 方式参与后训练（MoE 专家权重与 indexer QK 路径）；训练计算仍以 FP8 为主，推理/rollout 阶段直接使用原生 FP4 权重。

### APA 调用示例

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "messages": [
      {"role": "user", "content": "Explain the CSA+HCA architecture in DeepSeek-V4."}
    ],
    "max_tokens": 2048,
    "thinking_mode": "think_max",
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": true
  }'
```

**响应示例**（with Interleaved Thinking）：
```json
{
  "id": "chat-123456",
  "object": "chat.completion",
  "created": 1714376892,
  "model": "deepseek-ai/DeepSeek-V4-Pro",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "<thinking>The user is asking about CSA+HCA architecture. I need to explain:\n1. CSA (Compressed Sparse Attention)\n2. HCA (Heavily Compressed Attention)\n3. How they alternate in layers\n4. KV cache savings\n</thinking>\n\nDeepSeek-V4 uses a hybrid attention architecture...",
        "thinking_tokens": 156
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 512,
    "thinking_tokens": 156,
    "total_tokens": 683
  }
}
```

---

## 总结

DeepSeek-V4 是 **效率导向** 的架构创新，通过 CSA+HCA 混合注意力、流形约束超连接 (mHC)、Muon 优化器和 Interleaved Thinking，实现了：

1. **KV Cache 约为 V3.2 的 10%**（9.5× 更小；本文估算 83.9 → 9.62 GiB，约节省 88.5%）
2. **Agent/代码能力达到开源前沿**（SWE Verified 80.6%，与 Opus-4.6 的 80.8% 基本持平）
3. **1M tokens 上下文实用化**（MRCR 1M 为 83.5，Figure 9 的 1M Average MMR 为 0.59）

这些创新使 V4 成为 **Agent 原生**的 LLM，为长上下文和多步推理任务提供了高效解决方案。

---

**参考文献**：
1. DeepSeek-V4 Technical Report, DeepSeek Team, April 26, 2026
2. Hyper-Connections, Zhu et al., 2025
3. vLLM: Easy, Fast, and Cheap LLM Serving, vLLM Team, 2026
4. FlashAttention, Dao et al., 2022

---
