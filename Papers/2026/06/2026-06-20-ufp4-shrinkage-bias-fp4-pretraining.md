# FP4 预训练中的收缩偏差：几何起源、系统性影响与 UFP4 配方
## —— arXiv 2606.20381 深度阅读报告

---

## 1. 引言

当前 FP4 硬件加速器（NVIDIA Blackwell B200、Rubin、AMD MI350）均采用 E2M1 非均匀格式。本文揭示了该格式存在一个根本性的几何缺陷：**Shrinkage Bias（收缩偏差）**——一种由 Round-to-Nearest-Even (RTNE) 量化在非均匀网格上引起的系统性负偏差。该偏差在层间**乘性累积**，并被随机 Hadamard 变换（RHT）加剧，导致显著的信号衰减。

本文的核心洞察是：**E2M1 的非均匀量化网格在 RTNE 下会产生负偏差，而均匀网格（E1M2、INT4）则可完全避免此问题。**

---

## 2. Shrinkage Bias 理论分析

### 2.1 几何起源

对于非均匀量化网格 $\{q_{i-1}, q_i, q_{i+1}\}$，RTNE 量化在 $q_i$ 周围的区间 $\mathcal{B}_i = [t_{i,left}, t_{i,right})$ 内会产生系统性偏差：

$$
\mathbb{E}[\rho_G(t) - t \mid t \in \mathcal{B}_i] = \frac{\ell_i - r_i}{2} = \frac{2q_i - q_{i-1} - q_{i+1}}{4}
$$

其中 $\ell_i = q_i - t_{i,left}$ 为左边界宽度，$r_i = t_{i,right} - q_i$ 为右边界宽度。

**关键条件**：当 $r_i > \ell_i$ 时，该期望为**负值**。

### 2.2 E2M1 格式的偏差

E2M1 格式的非负层级为：$\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$

在 $q = 2$ 层级处：
- 偏差 $= \frac{2 \times 2 - 1.5 - 3}{4} = -0.125$

**均匀网格（E1M2、INT4）**满足 $\ell_i = r_i$，因此**不存在此偏差**。

---

## 3. 乘性累积机制

对于 GEMM 运算 $Z = AB^T$，当操作数被量化时，信号被衰减：

$$
\eta \approx \alpha_A \alpha_B < 1
$$

其中 $\alpha_A, \alpha_B$ 为量化导致的信号衰减系数。

对于 $K$ 个量化的 GEMM 层，总衰减为：

$$
\prod_{k=1}^{K} \eta_k = \prod_{k=1}^{K} (1 - \delta_k) \approx \exp\left(-\sum_{k=1}^{K} \delta_k\right)
$$

**关键差异**：零均值噪声会相互抵消，而负偏差会**指数级累积**，导致信号持续衰减。

---

## 4. RHT 的双重影响

### 4.1 传统观点的局限

RHT 通常被认为可以"平滑"异常值，使数据更适合低比特量化。

### 4.2 本文发现

RHT 将异常值能量分散，使量化从**动态范围受限**转变为**局部分辨率受限**：

- **E2M1**：数据被推入最不对称的量化箱 → **负 ΔSQNR**（质量下降）
- **E1M2**：均匀网格将平滑分布转换为更高保真度 → **正 ΔSQNR**（质量提升）

**核心洞察**："RHT 并非普遍有益。"

---

## 5. UFP4 配方

基于上述分析，本文提出 UFP4 (Uniform FP4) 配方：

| 配方项 | 具体配置 |
|--------|----------|
| **量化格式** | E1M2 或 INT4（均匀网格） |
| **RHT 应用** | 全部 3 个 GEMM（fwd_y, bwd_dx, bwd_dw） |
| **平滑策略 (SR)** | 仅应用于 dY |

---

## 6. 三种 FP4 格式对比

| 格式 | 网格类型 | 层级示例（非负） | Shrinkage Bias | 硬件支持 |
|------|----------|------------------|-----------------|----------|
| **E2M1** | 非均匀 | $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ | **存在（负偏差）** | B200, MI350（原生） |
| **E1M2** | 均匀 | $\{0, 1, 2, 3, 4, 5, 6, 7\}$ | **无** | 需软件支持 |
| **INT4** | 均匀 | $\{0, 1, 2, 3, 4, 5, 6, 7\}$ | **无** | 广泛支持 |

---

## 7. 实验结果

### 7.1 模型规模对比

| 模型 | E2M1 相对损失退化 | UFP4 (E1M2) 相对损失退化 | 改进幅度 |
|------|---------------------|------------------|----------|
| **Dense 1.5B** | 1.2570% | **0.9673%** | -23.0% |
| **MoE 7.9B** | 2.3596% | **1.8469%** | -21.7% |
| **MoE 124B** | 1.7308% | **1.3863%** | -19.9% |

*注：数值为相对于 BF16 基准的损失退化百分比，越低越好。*

### 7.2 关键观察

1. **一致性优势**：UFP4 在所有模型规模上均优于 E2M1
2. **可扩展性**：从 1.5B 到 124B，改进幅度保持稳定（约 20%）
3. **MoE 友好**：MoE 架构同样受益于均匀量化网格

---

## 8. 核心代码实现（简化示意）

```python
def ufp4_quantize_rtne(tensor: torch.Tensor, scale_max: float,
                       format: str = "E1M2") -> torch.Tensor:
    """
    UFP4 均匀网格 RTNE 量化（简化示意）
    
    Args:
        tensor: 输入张量（先除以 scale_max 归一化到 [-1, 1]）
        scale_max: 全局缩放因子（per-tensor absmax）
        format: "E1M2" 或 "INT4"
    
    Returns:
        量化后的 float32 模拟张量（实际部署中打包为 4-bit）
    """
    # 归一化
    x = tensor / scale_max
    
    if format == "E1M2":
        # E1M2：1符号 + 1指数 + 2尾数 = 4-bit
        # 均匀网格（非 IEEE 标准，UFP4 自定义格式）：
        # 16 级均匀分布：{-7/7, -5/7, ..., 5/7, 7/7} × scale_max
        n_levels = 8  # 每半轴
        scale = 1.0 / n_levels  # 量化步长
        # RTNE：四舍五入到最近的网格点，平局时舍入到偶数
        x_q = torch.round(x / scale) * scale
        x_q = torch.clamp(x_q, -1.0, 1.0)
    elif format == "INT4":
        # INT4 均匀网格：16 级，[-1, 1] 区间
        # 层级：{-1.0, -13/15, -11/15, ..., 11/15, 13/15, 1.0}
        n_levels = 8  # 每半轴 8 级（含 0）
        scale = 1.0 / (n_levels - 1)  # 量化步长
        x_q = torch.round(x / scale) * scale
        x_q = torch.clamp(x_q, -(n_levels - 1) * scale, (n_levels - 1) * scale)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # 实际部署中这里会有 4-bit pack/unpack
    return x_q * scale_max


def ufp4_forward(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """UFP4 前向：fwd_y = quantize(A) @ quantize(B)^T"""
    A_q = ufp4_quantize_rtne(A, scale_max=A.abs().max())
    B_q = ufp4_quantize_rtne(B, scale_max=B.abs().max())
    return torch.nn.functional.linear(A_q, B_q)


# 反向传播：SR (Stochastic Rounding) 仅应用于 dY
# 实际训练中通过指令 `stochastic_round(dY, bits=4)` 实现
```

---

## 9. 硬件影响分析

### 9.1 当前硬件困境

| 硬件 | 原生 FP4 格式 | 问题 |
|------|--------------|------|
| **NVIDIA Blackwell B200** | E2M1 | Shrinkage Bias 烘入硬件 |
| **NVIDIA Rubin** | E2M1 | 继承相同偏差 |
| **AMD CDNA4 MI350X** | E2M1 | 专注于 E2M1 实现 |

### 9.2 产业建议

**建议未来加速器将 E1M2/INT4 均匀网格作为一等公民原语支持：**

1. **指令集扩展**：添加 UFP4 量化/反量化指令
2. **流水线优化**：针对均匀网格的快速查表
3. **存储格式**：支持 E1M2/INT4 的片上缓存格式

### 9.3 迁移路径

对于现有 E2M1 硬件，可通过**软件模拟**实现 UFP4：
- 量化时使用 E1M2 映射
- 利用现有 INT4 加速路径
- 性能开销取决于硬件对均匀网格的加速支持（论文提供融合 kernel 基准测试）

---

## 10. 结论

本文通过严格的数学分析和大规模实验，证明：

1. **E2M1 的 Shrinkage Bias 是几何必然**，而非实现瑕疵
2. **乘性累积**使其影响远超零均值噪声
3. **RHT 的效果依赖量化网格几何**：对均匀网格有益，对非均匀网格有害
4. **UFP4 配方**在实践中 consistently 优于 E2M1，且具备硬件可实现性

**深远影响**：未来 AI 加速器设计应重新审视量化格式的选择，均匀网格（E1M2/INT4）应成为默认选项。
