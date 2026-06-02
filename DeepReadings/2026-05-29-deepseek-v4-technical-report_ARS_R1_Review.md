# ARS Academic Paper Reviewer — Full Mode Review Report (R1)

**Paper Under Review:** DeepSeek-V4 Technical Report  
**Review Date:** 2026-05-30  
**Reviewer Mode:** Full (EIC + 3 Reviewers + Devil's Advocate + Methodology + Source Verifier)  
**Round:** R1 (First Review)

---

## 1. EIC 总体评分与结论

| 项目 | 评分 |
|------|------|
| **总体评分** | **34 / 100** |
| **结论** | **MAJOR REVISION (拒稿边缘)** |

**EIC Summary:**
本报告存在系统性的事实准确性危机。经与原始论文（DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence, April 24, 2026）及多个独立可信来源交叉验证，发现以下致命缺陷：

1. **核心 benchmark 数据大面积造假/严重误读**：报告中 SWE Verified 55.2%、SWE Pro 38.4% 等关键数据与论文原文（SWE Verified 80.6%、SWE Pro 55.4%）严重不符，差距达 25-45 个百分点。
2. **论文标题错误**：报告篡改原论文副标题（Intelligence → for Agentic AI）。
3. **Muon 优化器公式根本性错误**：将 Newton-Schulz 正交化降级为简单 Nesterov momentum。
4. **代码不可运行**：9 个核心代码片段（CC1-CC9）全部存在严重 bug，包括不存在的 PyTorch dtype、shape mismatch、逻辑错误等。
5. **大量编造内容**：DSec 沙箱、开发者调查、消融实验数据、DSML 对比表、Interleaved Thinking 细节等均为虚构。
6. **关键 named technique 遗漏**：完全未提及 DSA (DeepSeek Sparse Attention)、Attention Sink、FP4-aware training、Anticipatory Routing、32T/33T token 预训练规模等论文核心内容。

报告在结构组织和可读性方面表现尚可，但**技术准确性和事实可溯源性已跌破学术底线**。在修正所有 MUST FIX 问题前，不建议作为技术参考使用。

---

## 2. 三位独立 Reviewer 评分与意见

### Reviewer #1 — 架构与模型专家 (Score: 30/100)

**评分细项：**
- 技术准确性：2/10
- 论文覆盖度：3/10
- 可读性：7/10

**核心意见：**
本报告对 DeepSeek-V4 的架构描述存在严重扭曲。最令人震惊的是**完全遗漏了 DSA (DeepSeek Sparse Attention)** —— 这是 CSA 层的核心技术组件，论文中明确说明 "CSA applies DeepSeek Sparse Attention (DSA)"，而报告通篇只字未提，仿佛 CSA 凭空发明了稀疏选择机制。

此外，HCA 的架构描述严重不完整：
- HCA 并非纯粹的 128× 压缩 + 稠密注意力。论文描述 HCA 会**拼接滑动窗口局部 token** 到压缩序列中，以保留局部上下文。报告完全遗漏这一关键设计。
- 报告对 "compress_ratios" 的解读 `[128, 128, 4, 128, 4, ...]` 缺乏原文支撑，且未说明 V4-Flash 和 V4-Pro 在初始层配置上的差异（Flash 起始用 sliding window，Pro 起始用 HCA）。

KV Cache 的数学推导是报告作者自行计算（9.62 GiB），而非引用论文数据。论文明确给出的数字是 **V4-Pro KV cache = 10% of V3.2**，报告的计算结果（11.5%）与论文矛盾，且推导过程中的假设（如 30 层 CSA / 30 层 HCA 的分割）未给出来源。

### Reviewer #2 — 训练与优化专家 (Score: 32/100)

**评分细项：**
- 训练方法准确性：2/10
- 优化器描述：1/10
- 后训练流程：4/10

**核心意见：**
**Muon 优化器的描述是本报告最严重的学术错误之一。** 报告将 Muon 降级描述为：

```
m_t = β m_{t-1} + η ∇L(w_{t-1})
w_t = w_{t-1} - m_t + β m_{t-1}   (Nesterov)
```

这完全是普通 Nesterov Momentum 的公式，**与 Muon 的核心贡献毫无关系**。论文 Algorithm 1 的真实 Muon 更新规则是：

```
M_t = μ M_{t-1} + G_t
O'_t = HybridNewtonSchulz(μ M_t + G_t)   // 关键：Newton-Schulz 正交化
O_t = O'_t · √(max(n,m)) · γ
W_t = (1 - ηλ) W_{t-1} - η O_t
```

报告遗漏了 Newton-Schulz 迭代（将梯度矩阵正交化为极分解因子 U·V^T），这是 Muon 区别于所有其他优化器的本质特征。这种错误不是笔误，而是**对算法原理的根本性误解**。

两阶段后训练也被严重简化且部分错误：
- 论文描述的是 **多个领域专家分别进行 SFT+RL(GRPO)**，然后 **On-Policy Distillation (OPD)** 统一回单一模型。报告将其扭曲为"领域专家培养(SFT+GRPO) + on-policy 蒸馏融合"，模糊了"多专家分化→统一"的关键结构。
- "DSec 沙箱"在论文中**完全不存在**，是报告编造的虚构概念。
- 预训练 token 数量（Flash: 32T, Pro: 33T）—— 论文明确披露的数据 —— 报告完全遗漏。

### Reviewer #3 — 长上下文与推理专家 (Score: 38/100)

**评分细项：**
- 注意力机制准确性：4/10
- 量化描述：5/10
- Agent 能力描述：3/10

**核心意见：**
报告对量化系统的描述部分正确但遗漏关键细节：
- **遗漏 RoPE 维度使用 BF16、剩余维度使用 FP8 的混合 KV 存储格式**。这是论文明确说明的 KV cache 减半技术，报告只字未提。
- **遗漏 FP4-aware training**。论文强调专家权重的 FP4 不是简单 PTQ，而是训练时即使用 FP4 量化感知训练，报告将其描述为简单的推理量化。

"Interleaved Thinking" 的描述极度过度的具体化：
- 论文中确实提到三种推理模式（Non-think / Think High / Think Max），但报告给出的 `reasoning_buffer` 实现、`AgentModel.think()` API、`thinking_mode` 参数等**均为编造**。`model.generate(..., thinking_mode="think_high")` 不是 HuggingFace transformers 的真实 API。
- SWE Verified 55.2% → 真实论文 80.6%。这一差距不是四舍五入，而是**错把 V3.2 的 baseline 当成了 V4 的数据，或完全凭空捏造**。

---

## 3. Devil's Advocate 风险/盲区分析

| 风险编号 | 风险类别 | 严重程度 | 描述 |
|---------|---------|---------|------|
| DA-1 | **数据伪造** | 🔴 致命 | 核心 benchmark（SWE Verified/Pro、MMLU 等）与论文原文差距巨大。若读者引用这些数字进行模型选型或学术研究，将导致严重错误决策。 |
| DA-2 | **代码安全风险** | 🔴 致命 | 报告中 9 个代码片段全部不可运行。若读者将其复制到生产环境，将立即触发 RuntimeError。尤其是 `torch.float4_e2m1` 不存在、`sparse_attn_kernel` shape mismatch 等。 |
| DA-3 | **标题篡改** | 🟠 严重 | 将论文标题从 "Context Intelligence" 改为 "Context for Agentic AI"，暗示论文专注于 Agent 能力，可能误导读者对论文主旨的理解。 |
| DA-4 | **虚构引用** | 🟠 严重 | "DSec 沙箱"、开发者调查、消融实验表等虚构内容若被二次传播，可能形成"信息回音室"，污染整个技术社区的知识库。 |
| DA-5 | **关键架构遗漏** | 🟠 严重 | 遗漏 DSA、Attention Sink、Anticipatory Routing 等技术导致读者无法复现模型行为，也无法理解论文完整技术栈。 |
| DA-6 | **量化描述不完整** | 🟡 中等 | 未区分 FP4-aware training 和 PTQ，可能导致读者在自研量化方案时走弯路。 |
| DA-7 | **过度具体化推测** | 🟡 中等 | 对 vLLM 内部实现（page size bucket、kernel fusion 细节）的描述缺乏论文支撑，可能是基于 V3 或其他来源的推测。 |

---

## 4. Methodology Reviewer — 代码正确性验证 (CC1-CC9)

### CC1: `fp4_quant_kernel` — ❌ FAIL
```python
x_quant = x_quant.to(torch.float4_e2m1)  # FP4
```
**问题：** PyTorch (截至 2.6) **不存在 `torch.float4_e2m1` dtype**。FP4 的支持仅存在于 NVIDIA Blackwell 硬件和 cuDNN/cutlass 底层库中，PyTorch 前端尚无原生 FP4 tensor 类型。代码运行时直接抛出 `AttributeError`。

### CC2: `act_quant_kernel` — ❌ FAIL
```python
x_quant = x_quant.view(L_padded, D)[:L, :]
```
**问题：** `torch.float8_e4m3fn` tensor 不能直接 `view` 后切片。FP8 是窄类型，reshape 操作可能触发 layout 错误。此外，scale 的形状 `(seq_len // block_size, dim)` 与后续 `fp8_gemm_kernel` 中声明的 `(M // 128, K // 128)` 不一致（后者是 2D block scale，前者是 1D per-row scale）。

### CC3: `fp8_gemm_kernel` — ❌ FAIL
```python
scale_a_expanded = scale_a.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
```
**问题：** 对于标准矩阵乘法维度（如 M=4096, K=4096），该操作将分配 `(4096, 4096)` 的 FP32 scale 矩阵，内存开销巨大（64MB per scale），完全违背量化省内存的初衷。真实的 FP8 GEMM 使用 CUTLASS 的 per-block scaling，不会在 Python 层做这种 naive expansion。

### CC4: `sparse_attn_kernel` — ❌ FAIL
```python
indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, -1, d_k)
k_selected = torch.gather(k, dim=-2, index=indices_expanded)
```
**问题：** `indices` 的形状是 `(B, H, L, topk)`。`unsqueeze(-1).expand(..., d_k)` 后是 `(B, H, L, topk, d_k)`。但 `torch.gather` 要求 index 的维度与 input 相同。`k` 的形状是 `(B, H, L, d_k)`（4 维），而 `indices_expanded` 是 5 维。**Shape mismatch**，运行时抛出 `IndexError`。

### CC5: `Indexer.forward` — ❌ FAIL
```python
kv_compressed = self.compressor(x, start_pos)  # (B, L/4, 2 * lora_rank)
k_idx, v_idx = kv_compressed.chunk(2, dim=-1)
k_idx = self.hadamard_rotate(k_idx)
k_idx_q, scale_k = fp4_quant_kernel(k_idx, block_size=32)
k_idx = k_idx_q.view(B, L//4, self.index_n_heads, self.index_head_dim)
```
**问题：** `lora_rank`（论文中约 512-1536）与 `index_n_heads * index_head_dim = 64 * 128 = 8192` **维度不匹配**。`k_idx_q.view(B, L//4, 64, 128)` 在 `lora_rank ≠ 8192` 时必然失败。Indexer 的 key 有独立的投影矩阵（`d_I` 维度），但报告代码完全遗漏了这一投影，直接从 KV latent 做 reshape。

### CC6: `MoE.forward` — ❌ FAIL
```python
for expert_idx in range(self.n_routed_experts):
    mask = (indices_flat == expert_idx)
    if mask.any():
        expert_tokens = x_flat[mask.any(dim=-1)]
        expert_weights = weights_flat[mask]
        y[mask.any(dim=-1)] += expert_out * expert_weights.unsqueeze(-1)
```
**问题：**
1. `expert_tokens` 和 `y[mask.any(dim=-1)]` 的索引逻辑不一致：前者取的是**至少有一个专家选中该 token** 的所有 token，但当前 expert 可能只处理其中一部分。
2. `expert_weights = weights_flat[mask]` 形状是 `(n_selected,)`，但 `mask` 是 `(B*L, 6)`，匹配后可能产生不同数量的元素。
3. 正确的实现应使用 `torch.index_select` 或 `torch_scatter`/`segment_sum`。

### CC7: `hc_post` — ❌ FAIL
```python
y = post * x.unsqueeze(1)  # (B, 4, L, D)
y = y + torch.sum(comb * residual, dim=1)
```
**问题：**
1. `post * x.unsqueeze(1)` 是**逐元素乘法 (Hadamard product)**，不是"投影到流形"。流形投影应该是线性投影（矩阵乘法）或 Sinkhorn 约束，而非逐元素相乘。
2. `comb * residual` 的语义不明：`comb` 已在 `hc_pre` 中被定义为 `weights * residual`，此处再次 `* residual` 相当于 `weights * residual^2`，与论文的 mHC 公式不符。
3. 论文中 mHC 的核心是 **Birkhoff Polytope（双随机矩阵流形）** 上的投影，报告代码完全未体现。

### CC8: `fast_log2_ceil` — ❌ FAIL
```python
exp = (x_int >> 23) & 0xFF
exp = exp - 127
exp = exp + 1
```
**问题：**
1. 对 `x = 0` 的处理：`(0).bit_length() = 0`，但 `fast_log2_ceil(0)` 应返回 `-inf` 或未定义。
2. 对 `x = 1.0`：`exp = 127`，`exp - 127 = 0`，`+1 = 1`，即 `ceil(log2(1.0)) = 1`，**错误**（应为 0）。
3. 对非规格化数（subnormal）的处理完全缺失。
4. `x_int = x.abs().view(torch.int32)` 在某些 PyTorch 版本上可能触发类型转换错误。

### CC9: `Gate.forward` + `MoE` 配置 — ❌ FAIL
```python
scores = torch.sqrt(F.softplus(scores))
topk_weights = F.softmax(topk_weights / self.route_scale, dim=-1)
```
**问题：**
1. 论文明确说计算亲和度分数的激活函数从 Sigmoid 改为 `Sqrt(Softplus(·))`，但归一化使用的是 **softmax 归一化**（报告正确）。然而，报告未说明 `route_scale = 2.5` 的来源。
2. 更关键的是：论文提到 "**无辅助损失的负载均衡策略**"（auxiliary-loss-free load balancing）+ "轻微序列级平衡损失"。报告完全遗漏了这一 MoE 训练的核心机制。
3. Hash Routing 的描述过于简化：论文说 "**前几个 Transformer 块**"替换为 hash routing，报告固定为 "前 3 层"，缺乏来源。

---

## 5. Source Verifier — 引用/事实准确性检查 (F1-F5)

### F1: 论文元数据 — ❌ FAIL
| 检查项 | 报告内容 | 论文原文 | 结论 |
|--------|---------|---------|------|
| 标题 | Towards Highly Efficient Million-Token Context **for Agentic AI** | Towards Highly Efficient Million-Token Context **Intelligence** | **标题被篡改** |
| 作者 | DeepSeek Team | DeepSeek-AI | 基本正确 |
| 发布日期 | 2026年4月24日（Preview Release） | April 24, 2026 | 正确 |
| 许可证 | MIT | MIT | 正确 |

### F2: 数值声明 — ❌ FAIL（多项）
| 指标 | 报告数值 | 论文/官方数据 | 差距 |
|------|---------|--------------|------|
| SWE Verified (V4-Pro-Max) | 55.2% | **80.6%** | -25.4 pp |
| SWE Pro (V4-Pro-Max) | 38.4% | **55.4%** | -17.0 pp |
| Terminal Bench 2.0 | 未提及 | **67.9%** | 遗漏 |
| MCPAtlas Public | 未提及 | **73.6%** | 遗漏 |
| MMLU (报告自称) | 88.7% | MMLU-Pro **87.5%** | +1.2 pp |
| V3.2 SWE Verified | 28.3% | **无此数据**（论文未报告 V3.2 的 SWE） | 疑似编造 |
| KV Cache (@1M) | 9.62 GiB (11.5% of 83.9) | **10% of V3.2** | 与论文矛盾 |
| 推理 FLOPs (@1M) | 未提及 | **27% of V3.2** | 重大遗漏 |
| Pro 预训练 token | 未提及 | **33T** | 遗漏 |
| Flash 预训练 token | 未提及 | **32T** | 遗漏 |
| LiveCodeBench | 未提及 | **93.5%** | 遗漏 |
| GPQA (报告自称) | 58.9% | **90.1%** | -31.2 pp |

### F3: 公式符号 — ❌ FAIL
| 公式 | 报告描述 | 论文原文 | 结论 |
|------|---------|---------|------|
| Muon | 简单 Nesterov momentum | Newton-Schulz 正交化 + 重缩放 | **根本性错误** |
| mHC Sinkhorn | 未说明约束流形的具体数学结构 | Birkhoff Polytope（双随机矩阵流形） | 不完整 |
| Sparse Attention | `Top-k(Q·K^T)` | DSA 包含 adaptive attention normalization + attention sink | 过度简化 |
| HCA | 纯 128× 压缩 + 稠密注意力 | 128× 压缩 + **拼接滑动窗口局部 token** | 遗漏关键组件 |

### F4: 模型名称/API — ❌ FAIL
| 检查项 | 报告内容 | 实际情况 | 结论 |
|--------|---------|---------|------|
| HF 模型名 | `deepseek-ai/DeepSeek-V4-Pro` | ✅ 存在 | 正确 |
| `thinking_mode` 参数 | `thinking_mode="think_high"` | ❌ transformers 无此参数 | 编造 |
| `AutoModelForCausalLM.from_pretrained` + `quantization_config` | `{"load_in_8bit": True}` 作为 FP8 | ❌ `load_in_8bit` 是 INT8/LLM.int8()，不是 FP8 | 概念混淆 |
| vLLM 参数组合 | `--quantization fp8 --dtype auto` | ⚠️ 部分正确但组合未必支持 CSA/HCA | 可疑 |

### F5: 引用准确性 — ❌ FAIL
| 引用 | 报告内容 | 实际情况 | 结论 |
|------|---------|---------|------|
| Hyper-Connections | Zhu et al., 2025 | mHC 论文实际为 **2026** (`arxiv:2512.24880`) | 年份错误 |
| vLLM | vLLM Team, 2026 | Kwon et al., **2023** (SOSP) | **年份错误+作者缺失** |
| FlashAttention | Dao et al., 2022 | ✅ 正确 | 正确 |
| DeepSeekMoE | 未引用 | 论文核心架构，应引用 | 遗漏 |
| DSA / DeepSeek Sparse Attention | 未引用 | 论文核心组件，应引用 | **重大遗漏** |

---

## 6. 合并 MUST FIX / SHOULD FIX 问题列表

### MUST FIX (MF)

| 编号 | 问题 | 严重程度 | 位置 | 修正要求 |
|------|------|---------|------|---------|
| **MF1** | **论文标题篡改** | 🔴 致命 | 第 1 行 | 将标题修正为 "Towards Highly Efficient Million-Token Context **Intelligence**" |
| **MF2** | **SWE Verified 数据错误** | 🔴 致命 | 第 827, 1061 行 | 修正为 **80.6%**（V4-Pro-Max），并注明来源（论文 Table 6） |
| **MF3** | **SWE Pro 数据错误** | 🔴 致命 | 第 1017, 1061 行 | 修正为 **55.4%**，并注明来源 |
| **MF4** | **Muon 公式根本性错误** | 🔴 致命 | 第 652-659 行 | 重写为论文 Algorithm 1 的真实 Muon：Hybrid Newton-Schulz + 重缩放 |
| **MF5** | **GPQA 数据错误** | 🔴 致命 | 第 996 行 | 修正为 **90.1%** |
| **MF6** | **DSA (DeepSeek Sparse Attention) 完全遗漏** | 🔴 致命 | 第四章整章 | 补充 DSA 的技术细节：Lightning Indexer、adaptive attention normalization、attention sink |
| **MF7** | **代码 CC1-CC9 全部不可运行** | 🔴 致命 | 第 309-961 行 | 修正所有代码片段，或明确标注为"示意性伪代码，不可直接运行" |
| **MF8** | **V3.2 SWE Verified 28.3% 无出处** | 🔴 致命 | 第 97, 1067 行 | 提供来源或删除。论文未报告 V3.2 的 SWE 数据 |
| **MF9** | **torch.float4_e2m1 不存在** | 🔴 致命 | 第 840, 902 行 | 说明 PyTorch 无原生 FP4 支持，需使用 NVIDIA cutlass/blackwell 专用 API |
| **MF10** | **KV Cache 10% 遗漏** | 🔴 致命 | 第 93, 382-417 行 | 删除自行推导的 9.62 GiB，改用论文明确给出的 "10% of V3.2" |
| **MF11** | **推理 FLOPs 27% 遗漏** | 🔴 致命 | 全文 | 补充论文明确给出的 "27% of V3.2 single-token FLOPs" |
| **MF12** | **预训练 token 32T/33T 遗漏** | 🟠 严重 | 全文 | 补充预训练规模 |

### SHOULD FIX (SF)

| 编号 | 问题 | 严重程度 | 位置 | 修正要求 |
|------|------|---------|------|---------|
| **SF1** | **HCA 滑动窗口局部 token 遗漏** | 🟠 严重 | 第 353-363 行 | 补充 HCA 拼接滑动窗口局部上下文的设计 |
| **SF2** | **Attention Sink 遗漏** | 🟠 严重 | 第四章 | 补充 CSA/HCA 中 attention sink（可学习 sink logits）的设计 |
| **SF3** | **RoPE-BF16 + 剩余-FP8 混合存储遗漏** | 🟠 严重 | 第 836-850 行 | 补充 KV cache 的混合精度存储策略 |
| **SF4** | **FP4-aware training 遗漏** | 🟠 严重 | 第 839-841 行 | 区分 FP4-aware training（训练时量化）和推理时 PTQ |
| **SF5** | **Anticipatory Routing 遗漏** | 🟠 严重 | 第六章 | 补充 MoE 训练稳定性技术 |
| **SF6** | **DSec 沙箱为虚构内容** | 🟠 严重 | 第 695-707 行 | 删除或明确标注为"作者推测，非论文内容" |
| **SF7** | **开发者调查数据为虚构** | 🟠 严重 | 第 1070-1082 行 | 删除或标注为虚构 |
| **SF8** | **消融实验数据为虚构** | 🟠 严重 | 第 623-633 行 | 删除或标注为虚构 |
| **SF9** | **DSML 对比表为虚构** | 🟠 严重 | 第 773-777 行 | 删除或标注为虚构 |
| **SF10** | **MRCR 8-Needle 数据为虚构** | 🟠 严重 | 第 1043-1055 行 | 删除或标注为虚构 |
| **SF11** | **Hash Routing 层数固定为 3 缺乏来源** | 🟡 中等 | 第 235, 1498 行 | 提供论文出处或改为"前若干层" |
| **SF12** | **vLLM 内部实现细节缺乏来源** | 🟡 中等 | 第 424-501 行 | 标注为基于公开信息的推测 |
| **SF13** | **引用年份错误** | 🟡 中等 | 参考文献 | 修正 vLLM (2023)、mHC (2026) 的引用 |
| **SF14** | **load_in_8bit 与 FP8 混淆** | 🟡 中等 | 第 2027-2033 行 | 说明 transformers 的 `load_in_8bit` 是 INT8 LLM.int8，非 FP8 |
| **SF15** | **Index Q 投影遗漏** | 🟡 中等 | 第 1411-1448 行 | 补充 Indexer 中 `d_I` 维度的独立投影矩阵 |

---

## 7. 六维度评分表

| 维度 | 权重 | 得分 (0-10) | 加权得分 | 说明 |
|------|------|------------|---------|------|
| **论文细节完整性** | 25% | 2.5 | 0.625 | 遗漏 DSA、Attention Sink、FP4-aware training、32T/33T token 等核心技术；编造 DSec、调查数据 |
| **代码深度** | 15% | 3.0 | 0.450 | 代码覆盖范围广但实现细节错误多，缺乏对真实约束（FP4 不支持、shape mismatch）的考虑 |
| **代码正确性** | 25% | 1.0 | 0.250 | **CC1-CC9 全部 FAIL**，存在 dtype 不存在、shape mismatch、逻辑错误等致命 bug |
| **技术准确性** | 20% | 2.0 | 0.400 | Muon 公式根本性错误、KV Cache 计算与论文矛盾、大量 benchmark 数据错误 |
| **可读性** | 10% | 7.5 | 0.750 | 结构清晰、类比丰富、目录组织良好，这是报告唯一的亮点 |
| **格式** | 5% | 7.0 | 0.350 | Markdown 格式规范，表格和公式排版整齐 |
| **总分** | 100% | — | **2.825** | 折算百分制 ≈ **28.25/100**（与 EIC 的 34/100 处于同一区间） |

---

## 8. ARS 七种失败模式检查清单 (M1-M7)

| 编号 | 失败模式 | 状态 | 证据 |
|------|---------|------|------|
| **M1** | **遗漏 named technique** | ❌ FAIL | 完全遗漏 DSA (DeepSeek Sparse Attention)、Attention Sink、FP4-aware training、Anticipatory Routing、Heterogeneous KV cache、TileLang DSL 等 |
| **M2** | **公式/符号不一致或错误** | ❌ FAIL | Muon 公式从 Newton-Schulz 正交化降级为 Nesterov momentum；mHC 未体现 Birkhoff Polytope；Sparse Attention 过度简化 |
| **M3** | **代码不可运行 (CC1-CC9)** | ❌ FAIL | 9/9 代码片段存在致命 bug：dtype 不存在、shape mismatch、逻辑错误、bitwise 错误 |
| **M4** | **关键数据无出处/无法追溯** | ❌ FAIL | SWE Verified 55.2%、SWE Pro 38.4%、V3.2 SWE 28.3%、开发者调查、消融实验、DSML 对比表、MRCR 数据等全部无出处或编造 |
| **M5** | **过度推测被当作事实** | ❌ FAIL | DSec 沙箱、Interleaved Thinking 实现细节、三种推理模式延迟数据、vLLM kernel fusion 细节等 |
| **M6** | **引用错误或缺失** | ❌ FAIL | vLLM 年份错误（2026→2023）、mHC 年份错误（2025→2026）、遗漏 DSA/DeepSeekMoE 引用 |
| **M7** | **标题/元数据篡改** | ❌ FAIL | 将 "Million-Token Context Intelligence" 改为 "Million-Token Context for Agentic AI" |

**M1-M7 综合结果：7/7 FAIL**

---

## 9. 附录：关键事实核查来源

1. **原始论文 PDF**: `https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf` (标题、架构、算法)
2. **DeepSeek 官方公告**: `https://api-docs.deepseek.com/news/news260424` (发布日期、模型规格)
3. **DeepSeek V4 Model Card**: `https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf` (架构确认)
4. **CSDN 技术报告精读** (victory0431, shibing624): 论文中文翻译和 benchmark 数据核实
5. **DeepSeek-V4 Paper Guide**: `http://deepseekv4paper.lol/` (关键数字交叉验证)
6. **NVIDIA Developer Blog**: `https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/` (架构官方解读)
7. **Muon 原始论文**: Jordan et al., 2024 + Liu et al., 2025 (Muon 算法验证)
8. **多个 benchmark 聚合站**: `deepseekv4.space`, `aiworkflows.tools`, `framia.pro` (SWE/MMLU/GPQA 等数据交叉验证)

---

*本审校报告由 ARS Academic Paper Reviewer (Full Mode) 生成。所有事实声明均通过与原始论文及独立可信来源的交叉验证确认。*
