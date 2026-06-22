# Rethinking Shrinkage Bias in LLM FP4 Pretraining: 几何起源、系统性影响与 UFP4 配方

## 论文元数据
- 标题：Rethinking Shrinkage Bias in LLM FP4 Pretraining: Geometric Origin, Systemic Impact, and UFP4 Recipe
- 作者：Qian Zhao, Kunlong Chen, Changxin Tian, Zhonghui Jiang, Haitao Zhang, Chaofan Yu, Peijie Jiang, Mingliang Gong, Jia Liu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou (Ant Group, Ling Team)
- arXiv ID：2606.20381
- 提交日期：2026-06-19
- 页数/图表：18 pages, 12 figures
- 官方代码：无官方实现

---

## Ch1: 论文概述与核心贡献

### FP4 训练背景与硬件路径

FP4 训练承诺大幅降低 LLM 预训练的内存和计算成本。当前主流 FP4 硬件路径和配方均围绕 E2M1 数据元素展开，包括 NVIDIA Blackwell/Rubin 级别系统和 AMD MI350 系列 GPU。E2M1 格式（2 指数位 + 1 尾数位）仅提供 16 个可表示值，但即便在此极端约束下，现有研究表明通过随机 Hadamard 变换（RHT）、stochastic rounding（SR）和 2D weight scaling 等技术，FP4 训练仍可实现可行。

### Shrinkage Bias 的发现

本文识别出 E2M1 格式的根本性限制：非均匀格式如 E2M1 天然存在 Shrinkage Bias，这是由其量化 bin 的几何不对称性导致的系统性负舍入误差。该偏差在 RTNE（Round-To-Nearest-Even）量化器下表现为条件期望误差：

$$E[\rho_G(t) - t | t \in B_i] = \frac{\ell_i - r_i}{2} = \frac{2q_i - q_{i-1} - q_{i+1}}{4}$$

在 E2M1 的可表示值集合 $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ 中，量化 bin 在间距转换点（如 $q=2, q=4$）处严重不对称，导致负偏差：在 $q=2$ 处偏差为 $-0.125$，在 $q=4$ 处偏差为 $-0.25$。

### 偏差的乘性累积机制

Shrinkage Bias 在深层网络中通过乘性机制累积，而非零均值噪声的可相互抵消性质。每个量化 GEMM 引入的信号衰减因子为 $\eta \approx \alpha_A \alpha_B < 1$，经过 $K$ 层后累积为 $\prod \eta_k \approx \exp(-\sum \delta_k)$，导致指数级信号衰减。

### RHT 的双重影响

随机 Hadamard 变换（RHT）对不同量化网格产生截然相反的效果：
- 对均匀网格（E1M2/INT4）：RHT 通过将异常值主导的张量转换为更均匀分布，提高码本利用率，转化为更高量化保真度
- 对非均匀网格（E2M1）：RHT 将数据从动态范围受限 regime 推向局部分辨率受限 regime，E2M1 的不对称 bin 在此 regime 下大量释放量化噪声，导致 SQNR 下降

这一发现统一解释了现有 E2M1-based FP4 配方中观察到的训练不稳定性。

### UFP4 配方

基于上述理论发现，本文提出 UFP4（Uniform FP4）配方，核心设计包括：
1. **均匀网格格式**：使用 E1M2 或 INT4-style 均匀网格，从根本上避免 Shrinkage Bias
2. **全路径 RHT**：在所有三个训练 GEMM（fwd_y, bwd_dx, bwd_dw）上应用 RHT
3. **受限 SR**：Stochastic rounding 仅应用于 dY 张量

### 主要结果速览

在 Dense 1.5B、MoE 7.9B 和 MoE 124B 的长运行预训练中，UFP4 一致实现比强 E2M1 baseline 更低的 BF16 相对损失退化：

| 模型 | E2M1 相对损失退化 | UFP4 (E1M2) 相对损失退化 | 改进幅度 |
|------|---------------------|--------------------------|---------|
| Dense 1.5B | 1.2570% | 0.9673% | -23.0% |
| MoE 7.9B | 2.3596% | 1.8469% | -21.7% |
| MoE 124B | 1.7308% | 1.3863% | -19.9% |

（注：数值为 $|L_r - L_{BF16}| / L_{BF16}$，latest-1000-step，越低越好）

Scaling-law 分析（10M–324M MoE 模型）和消融实验进一步支持了 UFP4 的有效性。

---

## Ch2: Shrinkage Bias 理论分析

### 2.1 几何起源

#### E2M1 的非均匀层级

E2M1 格式的非负可表示值为 $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$，在归一化幅度空间形成非均匀网格。该格式提供 2 个指数位和 1 个尾数位，相比均匀网格在动态范围上有优势，但以牺牲局部分辨率均匀性为代价。

#### RTNE 量化器的不对称性

Round-To-Nearest-Even（RTNE）量化器确定性地选择最近网格点。对于任意 bin $B_i = [q_{i-1/2}, q_{i+1/2}]$（左右边界 $\ell_i, r_i$），条件期望误差为：

$$E[\rho_G(t) - t | t \in B_i] = \frac{\ell_i - r_i}{2}$$

当 bin 对称时（$\ell_i = r_i$），该误差为零；当 bin 不对称时，产生系统性偏差。

#### E2M1 的负偏差模式

在 E2M1 网格中，间距转换点处 bin 严重不对称：
- $q=2$ 处：左侧间距 0.5，右侧间距 1.0，$\ell_i - r_i = 1.0 - 1.5 = -0.5$，偏差 $-0.125$
- $q=4$ 处：左侧间距 1.0，右侧间距 2.0，$\ell_i - r_i = 3.5 - 4.5 = -1.0$，偏差 $-0.25$

这种系统性的负偏差即 Shrinkage Bias。

#### 均匀网格的零偏差性质

E1M2 和 INT4 格式采用均匀网格，所有 bin 满足 $\ell_i = r_i$，因此：

$$E[\rho_G(t) - t | t \in B_i] = 0$$

均匀网格从根本上避免了 Shrinkage Bias。

### 2.2 乘性累积机制

#### 单层信号衰减

考虑单个量化 GEMM 操作 $Y = XW^T$，其中 $X$ 和 $W$ 均被量化。量化引入的相对误差可分解为：
- 零均值噪声：会在层间传播时相互抵消
- 系统性负偏差：无法抵消，累积放大

每个量化 GEMM 引入信号衰减因子 $\eta \approx \alpha_A \alpha_B < 1$，其中 $\alpha_A, \alpha_B$ 为两个输入张量的平均保留率。

#### 多层累积效应

经过 $K$ 个量化 GEMM 后，总信号保留率为：

$$\prod_{k=1}^{K} \eta_k \approx \exp\left(-\sum_{k=1}^{K} \delta_k\right)$$

其中 $\delta_k = -\ln \eta_k > 0$。偏差的指数级累积导致深层网络中的显著信号衰减，这与零均值噪声的抵消行为形成鲜明对比。

#### 非叶节点输出的级联传播

前向传播的激活值（fwd_y）和反向传播的梯度（bwd_dx）作为非叶节点输出，其偏差在层间级联传播。每个中间层不仅累积本层的 Shrinkage Bias，还继承来自上游层的偏差放大效应。

### 2.3 RHT 的双重影响

#### RHT 的数据分布变换

随机 Hadamard 变换（RHT）通过正交旋转将异常值主导的张量转换为更均匀分布。在全精度下，RHT 精确保持结果；在低精度下，RHT 将数据从**动态范围受限 regime**（少数异常值主导动态范围分配）推向**局部分辨率受限 regime**（分布均匀，需要高局部分辨率）。

#### E2M1 在 RHT 后的性能下降

在局部分辨率受限 regime 下，E2M1 的不对称 bin 成为致命缺陷：
- RHT 提高了码本利用率（effective bucket ratio 从 0.56 提升至 0.97）
- 但均匀分布使更多样本落入 E2M1 的高度不对称 bin（如 $[1.5, 2.5]$ 区间）
- 系统性负偏差被大量释放，导致 SQNR 下降

实验数据显示，在 outlier-heavy 张量（linear_fc2/fwd_x）上：
- RHT 前：E2M1 领先（21.90 vs E1M2 19.94 dB）
- RHT 后：E1M2 领先（23.19 vs E2M1 20.00 dB）

#### E1M2 在 RHT 后的性能提升

E1M2 的均匀网格在局部分辨率受限 regime 下保持零偏差特性：
- RHT 改善的分布均匀性直接转化为更高量化保真度
- 无不对称 bin 释放系统性噪声

在 well-behaved 张量（linear_fc1/fwd_x）上，RHT 对 E1M2 几乎中性（ΔSQNR: -0.008 dB），对 E2M1 略有益（+0.007 dB）。

#### RHT 双面性的统一解释

RHT 的效果依赖于量化网格几何：
- **均匀网格**：RHT → 分布均匀化 → 码本利用率提升 → 正 ΔSQNR
- **非均匀网格**：RHT → 推入不对称 bin → Shrinkage Bias 释放 → 负 ΔSQNR

这一发现统一解释了为何 RHT 在某些配置下改善训练稳定性，而在其他配置下加剧不稳定性。

---

## Ch3: UFP4 配方

### 核心设计原理

UFP4 基于以下理论洞察：
1. **Shrinkage Bias 是几何必然**：E2M1 的非均匀网格在 RTNE 下必然产生系统性负偏差，无法通过 scaling 或其他技巧消除
2. **RHT 后数据处于局部分辨率受限 regime**：在此 regime 下，均匀网格的零偏差特性成为关键优势
3. **RHT 对均匀网格有益**：全路径 RHT 可充分利用均匀网格的几何优势

因此，UFP4 的核心设计策略是：**使用均匀网格消除 Shrinkage Bias，配合全路径 RHT 最大化量化质量，仅对关键路径应用 SR**。

### 三项关键设计

#### 3.1 E1M2/INT4 均匀网格

UFP4 采用 E1M2 或 INT4-style 均匀网格，而非主流的 E2M1 格式：
- **E1M2**：1 指数位 + 2 尾数位，非负值 $\{0, 1, 2, 3, 4, 5, 6, 7\}$，16 级均匀网格
- **INT4**：纯整数均匀网格 $\{0, 1, 2, 3, 4, 5, 6, 7\}$

两种格式均满足 $\ell_i = r_i$，从根本上避免 Shrinkage Bias。

#### 3.2 全路径 RHT

UFP4 在所有三个训练 GEMM 上应用 RHT：
- **fwd_y**：前向传播输出量化 $Y = \text{quant}(RHT(XW^T))$
- **bwd_dx**：反向传播输入梯度量化 $\Delta X = \text{quant}(RHT(\Delta Y W))$
- **bwd_dw**：反向传播权重梯度量化 $\Delta W = \text{quant}(RHT(X^T \Delta Y))$

相比 E2M1-based recipe 仅在 bwd_dw 上使用 RHT，UFP4 的全路径设计确保：
- 所有异常值主导张量都经过均匀化处理
- 均匀网格的几何优势在所有路径上得到充分利用

#### 3.3 SR 仅限 dY

Stochastic Rounding（SR）仅应用于前向传播的输出量化（dY），而不用于梯度量化。这一设计基于以下考虑：
- SR 对 dY 有助于保持前向传播的数值精度
- 避免 SR 在梯度路径上引入额外噪声，保持训练稳定性

### 配置细节

UFP4 的核心配置参数（Table 1）：

| Configuration | E2M1-based recipe | E1M2-based recipe (UFP4) |
|---|---|---|
| Format | E2M1 | E1M2/INT4-style uniform grid |
| Quant block size | 1×16 | 1×16 |
| Scale hierarchy | FP32 single-level | FP32 single-level |
| RHT scope | bwd_dw | fwd_y, bwd_dx, bwd_dw |
| RHT block size | 16 | 16 |
| SR scope | dY | dY |
| 2D weight scaling | ✗ | ✗ |

**关键参数说明**：
- **Block size 1×16**：per-token 量化，沿 hidden dimension 维度共享 scale
- **FP32 single-level scale**：不使用 hierarchical scaling，简化实现
- **RHT block size 16**：与 hidden dimension 对齐，确保正交变换的有效性

### 与 E2M1-based Recipe 的对比

UFP4 与现有 E2M1-based 配方的关键区别：

| 维度 | E2M1-based recipe | UFP4 |
|------|-------------------|------|
| 网格几何 | 非均匀（存在 Shrinkage Bias） | 均匀（零偏差） |
| RHT 范围 | 仅 bwd_dw | 全路径（fwd_y + bwd_dx + bwd_dw） |
| 理论保证 | 无 | 几何误差分析 + 系统性影响证明 |
| 实验结果 | 基准 | 一致改进（~20%） |

### Scale Hierarchy 与格式选择的正交性

UFP4 的设计强调格式选择（E1M2 vs E2M1）与 scale hierarchy（single-level vs 2D scaling）的正交性：
- Shrinkage Bias 是网格几何的内在属性，与 scale 配置无关
- 均匀网格的优势在任何 scale hierarchy 下都成立
- 实验采用 single-level scale 简化验证，但结论可推广至 hierarchical scaling

这一正交性设计使 UFP4 可灵活集成至现有的 scale 优化方案中。

---


## 第4章 实验验证

## 4.1 实验设置

实验分为两类：单张量/单GEMM量化诊断实验，以及端到端FP4预训练实验。端到端训练使用与E2M1参考配方匹配的辅助配置以隔离格式效果，包括：block size = 1×16，FP32单层级scale hierarchy，stochastic rounding (SR) 限制于 dY。模型规模涵盖 Dense 1.5B、MoE 7.9B 和 MoE 124B，训练跨度超过 100B tokens。

## 4.2 Q1: RHT改变首选4-bit网格

单张量量化实验显示，RHT的效果依赖于量化网格几何。在well-behaved张量（linear_fc1/fwd_x）上，RHT几乎中性（ΔSQNR: E1M2 -0.008 dB, E2M1 +0.007 dB）。但在outlier-heavy张量（linear_fc2/fwd_x）上，RHT逆转了格式排名：

- E2M1：旋转前 21.90 dB → 旋转后 20.00 dB（下降1.90 dB）
- E1M2：旋转前 19.94 dB → 旋转后 23.19 dB（提升3.25 dB）

E1M2的effective bucket ratio从0.56提升到0.97，表明RHT将数据从outlier-dominated regime推向resolution-limited regime后，均匀网格的几何对称性优势得以体现。单GEMM输出实验显示，同样的格式依赖逆转在GEMM输出中持续存在，验证了该现象的鲁棒性。

## 4.3 Q2: UFP4减小BF16损失差距

在Dense 1.5B、MoE 7.9B和MoE 124B三个规模上，UFP4（E1M2配方）相比E2M1基线一致降低了BF16相对损失退化：

| 模型 | E2M1 相对损失退化 | UFP4 (E1M2) 相对损失退化 | 改进幅度 |
|------|---------------------|--------------------------|---------|
| Dense 1.5B | 1.2570% | 0.9673% | -23.0% |
| MoE 7.9B | 2.3596% | 1.8469% | -21.7% |
| MoE 124B | 1.7308% | 1.3863% | -19.9% |

数值为 |L_r - L_BF16| / L_BF16，基于latest-1000-step平均，越低越好。三个规模上改进幅度稳定在20%左右，验证了均匀网格在减轻Shrinkage Bias方面的系统性优势。

## 4.4 Q3:跨模型规模的Scaling-law验证

为验证UFP4的scaling特性，训练了10M–324M参数规模的MoE模型。结果显示，E1M2曲线在所有规模下均低于E2M1，表明均匀网格的优势不依赖于模型规模。同时，FP4-to-BF16 gap随计算量增加而减小，符合低精度训练的scaling规律。

## 4.5 Q4:消融实验

在Dense 1.5B、E1M2格式、训练超过100B tokens的设置下，对各组件进行消融：

| Setting | SR | fwd_y | bwd_dx | bwd_dw | Mean LM loss | Δ loss |
|---------|:--:|:-----:|:------:|:------:|:------------:|:------:|
| No RHT | ✓ | – | – | – | 1.89202 | 0.00000 |
| RHT on bwd_dw | ✓ | – | – | ✓ | 1.88721 | -0.00481 |
| RHT on bwd_dx, bwd_dw | ✓ | – | ✓ | ✓ | 1.88912 | -0.00290 |
| RHT on fwd_y, bwd_dw | ✓ | ✓ | – | ✓ | 1.88558 | -0.00644 |
| Full RHT w/ SR (UFP4) | ✓ | ✓ | ✓ | ✓ | 1.88079 | -0.01123 |
| Full RHT w/o SR | – | ✓ | ✓ | ✓ | 1.88535 | 0.00000 |
| Full RHT w/ SR (UFP4) | ✓ | ✓ | ✓ | ✓ | 1.88079 | -0.00456 |

关键发现：
- 全路径RHT（fwd_y + bwd_dx + bwd_dw）贡献-0.00644（相比仅bwd_dw）
- SR贡献额外-0.00456（相比Full RHT w/o SR）
- 组合效应达到-0.01123的总改进

先前NVFP4配方避免旋转非叶节点输出，但在均匀网格下，全路径RHT成为有益。此外，实验验证了E2M1 range-restriction（max_fpx=2.0，仅保留{0, 0.5, 1.0, 1.5, 2.0}）仍劣于E2M1 reference，不能作为均匀网格的替代品。

## 4.6 Q5:融合Kernel效率

融合RHT+量化的kernel开销轻微：

- 融合版 vs 独立量化：SM90 ~1.06×，SM100 ~1.07×
- 非融合版 vs 融合版：SM90 ~1.62×，SM100 ~1.41×

融合kernel的额外开销可控，且避免了中间数据的存储与往返，适合端到端训练。

---

## 第5章 代码实现

论文未提供官方实现代码。以下为基于论文描述的概念性实现，仅供参考，未经验证。

```python
# ⚠️ 非官方概念实现，未经验证

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def ufp4_quantize(x: torch.Tensor, format: str = "E1M2") -> torch.Tensor:
    """
    UFP4 量化：E1M2/INT4 均匀网格，RTNE 舍入
    format: "E1M2" (uniform) 或 "INT4" (uniform)
    """
    # E1M2 格式：1 指数位 + 2 尾数位，[-8, 8] 范围，16 级均匀分布
    if format == "E1M2":
        # E1M2 非负层：{0, 1, 2, 3, 4, 5, 6, 7}，共 16 级（含符号）
        grid = torch.tensor([
            -8.0, -7.0, -6.0, -5.0,
            -4.0, -3.0, -2.0, -1.0,
            0.0, 1.0, 2.0, 3.0,
            4.0, 5.0, 6.0, 7.0
        ], device=x.device, dtype=x.dtype)
    elif format == "INT4":
        # INT4 均匀网格，16 级
        grid = torch.linspace(-7.0, 7.0, 16, device=x.device, dtype=x.dtype)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # RTNE: 四舍五入到最近网格点
    x_flat = x.flatten()
    distances = torch.abs(grid.unsqueeze(1) - x_flat.unsqueeze(0))
    closest_indices = torch.argmin(distances, dim=0)
    quantized = grid[closest_indices].reshape(x.shape)
    
    return quantized

class UFP4Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, block_size: int = 16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        # 权重和偏置
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 量化输入
        x_q = ufp4_quantize(x, format="E1M2")
        
        # 量化权重
        w_q = ufp4_quantize(self.weight, format="E1M2")
        
        # 矩阵乘法
        output = F.linear(x_q, w_q, self.bias)
        
        return output

def stochastic_round(x: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """
    Stochastic Rounding (SR) 用于 dY
    保持期望值，概率性地选择相邻量化点
    """
    # 简化的 SR 实现
    scale = x.abs().max() / (2 ** (num_bits - 1))
    x_norm = x / scale
    floor = torch.floor(x_norm)
    frac = x_norm - floor
    
    # 概率性选择
    rand = torch.rand_like(frac)
    rounded = torch.where(rand < frac, floor + 1, floor)
    
    return rounded * scale

class UFP4LinearWithRHT(nn.Module):
    def __init__(self, in_features: int, out_features: int, block_size: int = 16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # 预计算 Hadamard 矩阵
        self.hadamard_size = block_size
        self.register_buffer('hadamard', self._generate_hadamard(block_size))
    
    def _generate_hadamard(self, n: int) -> torch.Tensor:
        """生成 Hadamard 矩阵（简化版）"""
        H = torch.ones(1, 1)
        while H.size(0) < n:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        return H[:n, :n] / (n ** 0.5)  # 归一化
    
    def _apply_rht(self, x: torch.Tensor) -> torch.Tensor:
        """应用 Random Hadamard Transform"""
        # Block-wise RHT（简化实现，实际应按 block_size 分块）
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, -1, self.block_size)
        
        # 对每个 block 应用 Hadamard 变换
        transformed = torch.einsum('ij,bkj->bki', self.hadamard, x_reshaped)
        
        return transformed.reshape_as(x)
    
    def forward(self, x: torch.Tensor, apply_rht: bool = True) -> torch.Tensor:
        if apply_rht:
            x_rht = self._apply_rht(x)
        else:
            x_rht = x
        
        # 量化
        x_q = ufp4_quantize(x_rht, format="E1M2")
        w_q = ufp4_quantize(self.weight, format="E1M2")
        
        # 矩阵乘法
        output = F.linear(x_q, w_q, self.bias)
        
        return output

# ⚠️ 非官方概念实现，未经验证
```

该实现展示了UFP4的核心逻辑：E1M2均匀网格量化、RTNE舍入、可选的RHT。实际训练中还需实现：
1. Block-wise scale 计算（FP32 单层级）
2. 反向传播时对 dY 应用 Stochastic Rounding
3. 三个 GEMM 位置的 RHT（fwd_y, bwd_dx, bwd_dw）

---

## 第6章 硬件影响与展望

## 6.1 当前硬件困境

当前FP4加速器硬件路径几乎全部硬编码E2M1格式：
- NVIDIA Blackwell (B200) 和 Rubin 系统采用 E2M1
- AMD MI350 系列 GPU 采用 E2M1
- 这些硬件选择源于MXFP4/NVFP4的早期标准化，但论文显示该选择存在几何必然的Shrinkage Bias

## 6.2 E2M1 Range-Restriction的局限性

实验验证了通过限制E2M1的动态范围（max_fpx=2.0，仅保留{0, 0.5, 1.0, 1.5, 2.0}）无法替代真正的均匀网格。该设置仍劣于E2M1 reference，表明Shrinkage Bias是bin不对称性的结构性问题，而非range问题。

## 6.3 未来硬件方向

### 6.3.1 HiFloat4 (Ascend 960)
华为Ascend 960的HiFloat4格式采用均匀S1P2格式（1符号位 + 2尾数位），符合UFP4的均匀网格理念。这代表了硬件设计的一个积极方向。

### 6.3.2 硬件设计建议
论文建议未来的FP4训练加速器应：
1. 原生支持 E1M2/INT4 均匀网格格式
2. 保留 E2M1 用于推理和含异常值张量的场景
3. 融合 RHT + 量化 kernel 以最小化开销（当前已实现 ~1.06× 加速）

## 6.4 软件生态影响

UFP4配方与现有FP4训练栈兼容，主要改动：
1. 格式选择：E2M1 → E1M2/INT4 均匀网格
2. RHT scope：从 bwd_dw 扩展到 fwd_y + bwd_dx + bwd_dw
3. SR scope：保持 dY 限制

这些改动不涉及scale hierarchy或block size的重新设计，可复用现有的MXFP4/NVFP4基础设施。

## 6.5 研究展望

### 6.5.1 更低精度训练
FP4的成功验证了4-bit训练的可行性，未来可能探索：
- FP3 训练的几何偏差问题
- 混合精度训练（不同层使用不同格式）

### 6.5.2 理论扩展
Shrinkage Bias的形式化可扩展至：
- 其他非均匀格式（E3M0, E0M3）
- 不同的舍入策略（SR, dithering）

### 6.5.3 系统优化
- 硬件级 RHT 单元
- 动态格式选择（根据张量分布自适应切换 E2M1/E1M2）

---

**结论**：Shrinkage Bias是E2M1非均匀网格的几何必然，UFP4通过均匀网格、全路径RHT和SR限制，在三个规模模型上实现了约20%的损失退化降低。未来硬件应原生支持均匀网格格式以充分利用FP4训练潜力。
