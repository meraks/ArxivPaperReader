# Functional Attention: 从逐对亲和力到函数对应

> **论文信息**：Functional Attention: From Pairwise Affinities to Functional Correspondences  
> **作者**：Jiefang Xiao, Maolin Gao, Simon Weber, Guandao Yang, Daniel Cremers  
> **机构**：Technical University of Munich, Munich Center for Machine Learning, University of Oxford, UT Austin  
> **发表**：ICML 2026  
> **arXiv**：[2605.31559](https://arxiv.org/abs/2605.31559) | **代码**：[github.com/xjffff/FUNCATTN](https://github.com/xjffff/FUNCATTN)

---

## Ch1：论文概述与核心贡献

### 1.1 一句话总结

**Functional Attention 将 Transformer 中的注意力机制从「token 间的逐对相似度计算」重新定义为「函数空间之间的线性算子学习」——将算子估计从 O(n²) 降至 O(k²)（k≪n），整体复杂度从 O(n²d) 降至 O(ndk+dk²+k³)（k<d 时），并在实验中表现出良好的分辨率不变性。**

### 1.2 为什么要重新定义 Attention？

标准 attention 的核心操作是：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

这个操作有三个根本性问题：

1. **复杂度爆炸**：n×n 的 softmax 矩阵，O(n²d) 的计算和存储，处理长序列时不可承受
2. **将连续函数离散化为 token**：输入数据（如 PDE 解、3D 点云）本质上是**连续函数**在离散点上的采样，但 attention 把它们当成独立 token，忽略了底层的连续函数结构
3. **分辨率敏感**：改变采样密度（网格分辨率）会导致完全不同的 token 集合，attention 没有内建的机制来处理这种变化

> **类比理解**：
> 想象你在看一幅画。标准 attention 的做法是：逐个像素地比较任意两个像素之间的「关系」（n×n 矩阵），然后基于这些关系来更新每个像素。而 Functional Attention 的做法是：先用少量「画笔」（k 个基函数，k≈64-128）来描述整幅画，然后在画笔之间建立关系（k×k 矩阵），最后用这些关系来更新画的内容。画笔的数量不会因为画布尺寸（分辨率）改变而改变——这就是分辨率不变性的来源。

### 1.3 三大核心贡献

| 贡献 | 内容 | 意义 |
|------|------|------|
| **新范式** | 将 attention 重新定义为函数空间之间的线性算子估计 | 从 token-level 到 function-level 的质变 |
| **FUNCATTN 架构** | 可学习自适应基函数 + Tikhonov 正则化闭式解 | 实用的 k×k 算子实现 |
| **全面验证** | PDE 求解、3D 分割、回归、OOD 泛化、超分辨率 | 6 个 PDE benchmark 全面超越 Transolver |

---

## Ch2：研究背景与动机

### 2.1 标准 Attention 的隐藏假设

回顾标准 scaled dot-product attention：

给定输入 $X \in \mathbb{R}^{n \times d}$（n 个 token，每个 d 维），通过线性投影得到 Q, K, V：

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

然后计算：

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \in \mathbb{R}^{n \times n}$$
$$\text{Output} = AV \in \mathbb{R}^{n \times d}$$

这里有一个**隐藏的结构性假设**：输入的 n 个位置是独立的、离散的 token。对于 NLP 中的词序列，这很自然。但对于科学计算中的数据——PDE 在网格上的解、3D 点云、物理场的采样——这个假设是有问题的。

### 2.2 连续函数 vs 离散 Token

考虑一个简单的例子：求解热传导方程，输入是温度场 $u(x)$ 在 $n$ 个网格点上的值。如果换一个更密的网格（$n' = 4n$），物理本质上完全相同，但标准 attention 会处理完全不同规模的矩阵。这就是**分辨率依赖性**（discretization dependence）。

> **类比理解**：
> 用不同分辨率的相机拍同一座山——标准 attention 会因为你换了相机而完全改变内部计算方式。Functional Attention 则关注于「山」这个连续实体本身，无论你用多少像素去描述它。

### 2.3 Functional Maps 的启示

Functional Maps（Ovsjanikov et al., 2012）是 3D 形状分析中的经典框架。它的核心思想是：

- 两个形状之间的对应关系不应该是「点对点」的 n×n 映射
- 而应该是「函数对函数」的紧凑线性算子，在谱基下表示为 k×k 矩阵（k ≪ n）

具体来说，给定两个形状上的函数空间和各自的基函数 $\Phi_1, \Phi_2$（通常取 Laplace-Beltrami 算子的特征函数），functional map $C \in \mathbb{R}^{k \times k}$ 满足：

$$\Phi_2 C \approx \Pi \Phi_1$$

其中 $\Pi$ 是真实的点对应矩阵。$C$ 可以通过最小二乘求解：

$$C = \arg\min_C \|\Phi_2 C - \Pi \Phi_1\|_F^2 + \lambda R(C)$$

Functional Attention 巧妙地将这个几何处理论**搬到了 attention 的语境中**：attention 层中的 Q、K、V 投影天然形成了函数空间，而「基函数」可以通过学习得到。

### 2.4 现有线性 Attention 的局限

过去几年出现了大量降低 attention 复杂度的工作（Linear Transformer, Performer, Linformer, Nyströmformer 等），但它们通常：
- 依赖于特定的核函数近似（如 $\phi(Q)\phi(K)^T$）
- 缺乏对连续函数结构的显式建模
- 在处理非网格化输入（不规则点云）时表现不稳定

Functional Attention 的不同在于：它**显式地将输入建模为连续函数**，并通过函数空间中的线性算子来捕获全局依赖。

---

## Ch3：Functional Attention 核心方法

### 3.1 核心思想：从 n×n 到 k×k

Functional Attention 的关键洞察：

> *Attention 计算中真正重要的不是 n×n 的逐点亲和力矩阵本身，而是这个矩阵在函数空间上**诱导的线性算子**。*

因此，我们不直接计算 $A \in \mathbb{R}^{n \times n}$，而是估计一个紧凑的 $C \in \mathbb{R}^{k \times k}$（k ≪ n），使得它能够在函数空间中实现相同的变换效果。

### 3.2 数学公式

**输入**：
- Query, Key, Value 矩阵：$Q, K, V \in \mathbb{R}^{n \times d}$
- 可学习的基函数矩阵：$\Phi, \Psi \in \mathbb{R}^{n \times k}$（k 是基函数的数量，通常 k ∈ [32, 256]）

**Step 1：投影到谱系数（Spectral Projection）**

将 Q、K、V 从 token 空间投影到函数空间：

$$\tilde{Q} = \Psi^T Q \in \mathbb{R}^{k \times d}$$
$$\tilde{K} = \Psi^T K \in \mathbb{R}^{k \times d}$$
$$\tilde{V} = \Psi^T V \in \mathbb{R}^{k \times d}$$

> **类比理解**：
> 这就像傅里叶变换——把时域信号（n 个 token）转换到频域（k 个频率分量）。不过这里的「基」不是固定的正弦波，而是**可学习的自适应基函数**，由数据驱动。

**Step 2：估计 Functional Transport Operator**

在函数空间中求解最优的线性算子 $C$，使得 $C \tilde{K}$ 尽可能接近 $\tilde{Q}$：

$$C^* = \arg\min_C \|\tilde{Q} - C \tilde{K}\|_F^2 + \lambda \|C\|_F^2$$

这是带 Tikhonov 正则化的最小二乘问题。闭式解为：

$$C^* = \tilde{Q}\tilde{K}^T \left(\tilde{K}\tilde{K}^T + \lambda I_k\right)^{-1}$$

**Step 3：输出重构**

用 $\Phi$ 基将函数空间的变换结果映射回 token 空间：

$$\boxed{\text{FUNCATTN}(Q, K, V) = \Phi \; C^* \; \tilde{V}}$$

> **类比理解**：
> 整个过程就像翻译的三步：(1) 将原文（token 表示）编码为中间语义表示（k 维谱系数），(2) 在语义空间中做信息传递（k×k 算子），(3) 将结果解码回原文格式（token 空间）。

### 3.3 基函数的设计

基函数是 Functional Attention 的核心组件之一。论文选择了**可学习自适应基函数**：

$$B = \text{Softmax}\big(\text{Linear}(X)\big) \in \mathbb{R}^{n \times k}$$

其中 Linear 是从 $\mathbb{R}^d$ 到 $\mathbb{R}^k$ 的全连接层，Softmax 沿 k 维度应用。

这个设计有几个精妙之处：

1. **自适应性**：基函数不是预定义的（如径向基、傅里叶基），而是从数据中学习的，能够自动适配数据的语义/几何/物理结构
2. **Partition of Unity**：Softmax 保证了 $\sum_{j=1}^k \phi_j(x_i) = 1$，这是一个关键的规范化性质
3. **可解释的极限行为**：当 softmax 温度 $\tau \to 0$ 时，基函数退化为 P0 分段常数元（hard partition），每个 token 被分配到唯一的基函数

在实践中，$\Phi$ 和 $\Psi$ 使用相同的架构但不同的参数。

### 3.4 完整算法流程

```
Algorithm: Functional Attention Layer

Input:  X ∈ R^{n×d}          # n tokens, d dims
        k                     # number of basis functions
        λ                     # regularization strength

1. Linear Projections:
   Q = X @ W_Q               # R^{n×d}
   K = X @ W_K
   V = X @ W_V

2. Learn Basis Functions:
   Φ = Softmax(Linear_Φ(X))  # R^{n×k}
   Ψ = Softmax(Linear_Ψ(X))  # R^{n×k}

3. Spectral Projection:
   Q̃ = Ψ^T @ Q               # R^{k×d}
   K̃ = Ψ^T @ K               # R^{k×d}
   Ṽ = Ψ^T @ V               # R^{k×d}

4. Functional Transport Operator (Closed-form):
   M = K̃ @ K̃^T + λ·I_k       # R^{k×k}
   C = Q̃ @ K̃^T @ M^{-1}       # R^{k×k}  [或解线性系统 M·C^T = K̃ @ Q̃^T]

5. Output Reconstruction:
   Out = Φ @ C @ Ṽ            # R^{n×d}

Output: Out ∈ R^{n×d}
```

---

## Ch4：理论分析

### 4.1 基函数的性质（Proposition 4.3）

**命题 4.3（广义分段常数元）**：可学习基函数 $B = \text{Softmax}(\text{Linear}(X)/\tau)$ 满足：

1. **Partition of Unity**：$\sum_{j=1}^k B_{ij} = 1$ 对所有 $i$ 成立
2. **极限行为**：当 $\tau \to 0$，$B$ 收敛到 P0 分段常数元——每个输入点被硬分配到唯一的基函数

证明思路：Softmax 的定义直接保证了性质 1。性质 2 来自 softmax 在低温下的 argmax 行为。

**实际意义**：Partition of Unity 保证基函数的「覆盖」是完整的——没有输入点被遗漏。极限行为揭示了基函数在高置信度时如何进行「硬聚类」，为解释模型的内部表示提供了理论窗口。

### 4.2 稳定性分析（Proposition 4.5）

**命题 4.5（局部 Lipschitz 连续性）**：FUNCATTN 层是局部 Lipschitz 连续的，Lipschitz 常数由 $O(1/\lambda + 1/\lambda^2)$ 界定。

这个性质在实践中至关重要：
- 保证了小输入扰动不会导致输出剧烈变化
- 正则化参数 $\lambda$ 直接控制稳定性和灵活性之间的权衡
- 解释了为什么 FUNCATTN 对离散化变化具有鲁棒性

> **类比理解**：
> $\lambda$ 像汽车的减震器。$\lambda$ 太小，车遇到颠簸就剧烈摇晃（不稳定，但响应灵敏）；$\lambda$ 太大，车完全不晃但也失去了灵活性。论文建议 $\lambda \in [10^{-3}, 10^{-1}]$。

### 4.3 计算复杂度分析

论文附录 B.1 给出了完整的复杂度：$O(ndk + dk \cdot \min(k,d) + \min(k,d)^3)$。在典型的设定中 $k < d$（如 k=64, d=128-512），复杂度简化为：

| 操作 | 标准 Attention | Functional Attention |
|------|:---:|:---:|
| QK^T / 谱投影 | $O(n^2 d)$ | $O(n d k)$ |
| Softmax | $O(n^2)$ | — |
| 算子估计 | — | $O(d k^2 + k^3)$ |
| AV / 重构 | $O(n^2 d)$ | $O(n d k)$ |
| **总计** | $\mathbf{O(n^2 d)}$ | $\mathbf{O(n d k + d k^2 + k^3)}$（假设 $k < d$） |

由于 $k \ll n$（典型 $k=64$ vs $n=10000+$），Functional Attention 在大规模数据上的优势非常明显。

**内存复杂度**：Functional Attention 只需要 $O(k^2 + n k)$ 而非 $O(n^2)$，这意味着处理 $n=10^6$ 点时只需要约几百 MB 而非 TB 级别的内存。

### 4.4 与其他线性 Attention 的关系

| 方法 | 核心技巧 | 函数空间建模 | 分辨率不变性 |
|------|---------|:---:|:---:|
| Linear Transformer | $\phi(Q)\phi(K)^T$ 分解 | ✗ | ✗ |
| Performer | 随机特征近似 | ✗ | ✗ |
| Nyströmformer | Nyström 近似 | ✗ | ✗ |
| Linformer | 低秩投影 | ✗ | ✗ |
| **FUNCATTN** | **函数空间算子估计** | **✓** | **✓** |

Functional Attention 的根本不同在于：它不是在做数学近似（approximate attention），而是在做**语义重新定义**（redefine attention）。这使得它在物理和几何数据上天然优于所有基于矩阵近似的线性 attention。

---

## Ch5：实验设计与结果

### 5.1 Few-shot 正弦回归

**任务描述**：给定仅 4-5 个观测点 $(x_i, y_i)$，预测正弦函数的完整波形。

这是对模型**从极少样本中提取函数结构能力**的极限测试。

**结果**：
- FUNCATTN 的 MSE 比标准 softmax attention **低数个数量级**
- 比 Intention（之前最优方法）**好约 10 倍**
- 用仅 **5 个观测点** 的 FUNCATTN 误差 **低于** softmax attention 用 40 个观测点

> **类比理解**：
> 这就像给你一首歌的前 5 秒，让你续写出整首歌。Standard attention 需要听到 40 秒才能做得差不多好。FUNCATTN 只用了 5 秒就超越了它——因为它理解了「正弦」这个函数结构本身，而不只是记住了采样点的模式。

### 5.2 PDE 求解（6个标准基准）

| 方法 | Elasticity | Airfoil | Darcy | Pipe | Navier-Stokes | Plasticity |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Transolver (2024 SOTA) | baseline | baseline | baseline | baseline | baseline | baseline |
| **FUNCATTN** | **✓ 超越** | **✓ 超越** | **✓ 超越** | **⚠ 与 LNO 持平** | **✓ 超越** | **✓ 超越** |

FUNCATTN 在 **6 个 benchmark 中的 5 个** 上取得了最优结果（Pipe 基准上与 LNO 持平），全面超越 Transolver（2024 年的 SOTA）。论文原文声明："achieves the best performance on **five out of six** PDE benchmarks."

**Benchmark 说明**：
- **Darcy Flow**：多孔介质渗流，输入为渗透率场，输出为压力场
- **Navier-Stokes**：不可压缩流体，输入为初始涡量场，输出为后续时刻的涡量
- **Airfoil**：翼型绕流，输入为翼型几何 + 流动参数，输出为压力/速度场
- **Pipe**：管道湍流，输入为截面几何，输出为流场
- **Plasticity**：弹塑性变形，输入为材料参数场，输出为位移场
- **Elasticity**：线弹性变形，输入为材料参数场，输出为位移场

### 5.3 3D RNA 结构分割

**任务**：在 3D 点云上对 RNA 核苷酸进行语义分割。

**重要性**：RNA 结构预测是生物信息学的核心问题，点云是非结构化、不规则采样的——这正是 Functional Attention 擅长处理的场景。

FUNCATTN 在分割精度上达到 SOTA，且在处理不同采样密度的 RNA 结构时表现出高度一致性。

### 5.4 OOD 泛化：AirfRANS 翼型设计

**AirfRANS 数据集**：包含不同雷诺数（流动条件）下的翼型绕流数据。

**测试**：在高雷诺数上训练，在低雷诺数上测试（或反之）——测试模型的**物理外推**能力。

**结果**：FUNCATTN 在所有 OOD 设置下均表现最佳，证明了其捕获的「函数结构」在分布偏移时依然有效。

### 5.5 零样本超分辨率：1D Burgers 方程

**Burgers 方程**：$\partial_t u + u \partial_x u = \nu \partial_{xx} u$，是流体力学的基本模型方程。

**测试方案**：在粗网格（128 点）上训练，直接在细网格（512 点，4× 分辨率）上测试，**不做任何微调**。

**结果**：FUNCATTN 在零样本超分辨率任务上显著优于所有 baseline——这是分辨率不变性最直接的证明。

> **为什么这很重要**：传统方法需要为每种分辨率训练一个独立模型。FUNCATTN 可以**一次训练，多分辨率部署**——在工程上意味着巨大的成本节省。

---

## Ch6：代码实现详解

### 6.1 仓库结构

```
FUNCATTN/
├── Few-Shot-Regression/        # 少样本正弦回归实验
├── PDE-StandardBenchmark/      # 6 个 PDE 基准实验
├── RNA-Segmentation/           # 3D RNA 点云分割
├── Airfoil-Design-AirfRANS/    # OOD 泛化翼型设计
├── Burgers-Super-Res/          # 零样本超分辨率
├── requirements.txt            # 依赖：torch, numpy, scipy 等
└── README.md                   # 论文结果汇总
```

### 6.2 核心 FUNCATTN 概念实现

```python
# ⚠️ 基于论文 Algorithm 1 / Section 3 描述的概念实现
# 目的：帮助理解算法流程，非官方实现代码
# 来源：arxiv.org/abs/2605.31559 Section 3

import torch
import torch.nn as nn
import torch.nn.functional as F


class FunctionalAttention(nn.Module):
    """
    Functional Attention Layer
    
    将标准 softmax attention 替换为函数空间中的线性算子估计。
    
    参数形状：
      - n: 输入 token 数量
      - d: 特征维度
      - k: 基函数数量 (k << n, 典型值 64-128)
    """
    
    def __init__(self, d_model: int, k: int = 64, lambda_reg: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.lambda_reg = lambda_reg
        
        # Q, K, V 线性投影
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # 基函数生成器
        self.basis_phi = nn.Sequential(
            nn.Linear(d_model, k),
            nn.Softmax(dim=-1)  # Softmax 沿 k 维 → partition of unity
        )
        self.basis_psi = nn.Sequential(
            nn.Linear(d_model, k),
            nn.Softmax(dim=-1)
        )
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n, d) - batch of token sequences
        Returns:
            out: (B, n, d)
        """
        B, n, d = x.shape
        
        # Step 1: 线性投影
        Q = self.W_q(x)  # (B, n, d)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Step 2: 生成自适应基函数
        Phi = self.basis_phi(x)  # (B, n, k)
        Psi = self.basis_psi(x)  # (B, n, k)
        
        # Step 3: 谱投影 — 从 token 空间到函数空间
        Q_tilde = torch.bmm(Psi.transpose(1, 2), Q)  # (B, k, d)
        K_tilde = torch.bmm(Psi.transpose(1, 2), K)  # (B, k, d)
        V_tilde = torch.bmm(Psi.transpose(1, 2), V)  # (B, k, d)
        
        # Step 4: 估计 Functional Transport Operator C
        # C* = Q̃ K̃^T (K̃ K̃^T + λ I_k)^{-1}
        KKT = torch.bmm(K_tilde, K_tilde.transpose(1, 2))  # (B, k, k)
        # Tikhonov 正则化
        eye = torch.eye(self.k, device=x.device).unsqueeze(0)  # (1, k, k)
        M = KKT + self.lambda_reg * eye  # (B, k, k)
        
        # 解线性系统 M · C^T = K̃ Q̃^T
        QK = torch.bmm(Q_tilde, K_tilde.transpose(1, 2))  # (B, k, k)
        # C = QK @ M^{-1}  实际上解: M @ C^T = QK^T
        C = torch.linalg.solve(
            M, QK.transpose(1, 2)
        ).transpose(1, 2)  # (B, k, k)
        
        # Step 5: 输出重构
        # Out = Φ @ C @ Ṽ
        CV = torch.bmm(C, V_tilde)  # (B, k, d)
        out = torch.bmm(Phi, CV)  # (B, n, d)
        
        return self.W_o(out)


class FUNCATTNBlock(nn.Module):
    """完整的 FUNCATTN Transformer Block"""
    
    def __init__(self, d_model: int, k: int = 64, lambda_reg: float = 0.01,
                 expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = FunctionalAttention(d_model, k, lambda_reg)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 模拟 PDE 求解场景: n 个网格点, 每个点 d 维特征
    B, n, d = 4, 1024, 128
    k = 64  # 仅 64 个基函数
    
    model = FUNCATTNBlock(d_model=d, k=k).cuda()
    x = torch.randn(B, n, d).cuda()
    
    out = model(x)
    print(f"Input:  {x.shape}")   # (4, 1024, 128)
    print(f"Output: {out.shape}")  # (4, 1024, 128)
    print(f"Basis:  k={k}, compression ratio: {n/k:.0f}:1")
    print(f"Complexity: O(n·d·k)={1024*128*64:,} vs O(n²·d)={1024*1024*128:,}")
```

### 6.3 关键实现细节

1. **数值稳定性**：使用 `torch.linalg.solve` 而非显式求逆，避免数值精度问题
2. **基函数初始化**：论文使用可学习的 Linear 层生成基函数，权重按标准方式初始化（论文未指定具体初始化方案，实践中 Xavier/He 初始化均可使初始基函数近似均匀分布）
3. **正则化选择**：$\lambda \in [10^{-3}, 10^{-1}]$，数据集规模越小，$\lambda$ 应设得越大
4. **k 的选择**：$k=32$ 对简单任务足够，$k=128$ 对复杂 PDE 更有效，进一步增大 k 收益递减
5. **Woodbury 恒等式优化**：论文附录提到当 $k > d$ 时可用 Woodbury 恒等式转换为 $d \times d$ 矩阵求逆，始终在较小维度上计算

---

## Ch7：局限性与延伸阅读

### 7.1 已知局限性

**论文自述的局限性**（§6 Limitations & Future Works）：
1. 可学习基函数目前使用简单的 softmax 投影，探索更具表达力或结构化的基函数设计仍是一个开放方向
2. 缺乏严格的逼近保证或泛化界等理论分析
3. 压缩比 k/n 与逼近误差之间的形式化关系尚未建立
4. 附录 D.2 讨论了使用转置（Φ^T）而非伪逆（Φ†）的稳定性权衡

**报告的补充分析**：
5. **线性算子局限**：FUNCATTN 假设 Q-K 关系可以被**线性算子**良好近似。对于高度非线性的 Q-K 交互（如长程语义依赖），线性算子可能不足
6. **k 的选择缺乏自动化**：k 目前是超参数，无自动选择机制。太大浪费计算，太小丢失信息
7. **NLP 任务验证缺失**：论文主要在科学计算领域验证，在 NLP（长文本、机器翻译）上的表现未知
8. **FUNCATTN 是 Intention 的泛化**（论文附录 A.5）：报告未展开这一理论联系

### 7.2 潜在改进方向

| 方向 | 思路 |
|------|------|
| **非线性 FunctAttn** | 用神经网络参数化算子 C，而非线性最小二乘 |
| **多尺度基函数** | 层次化基函数，在不同尺度上学习不同的 k |
| **自适应 k** | 根据输入复杂度和任务难度动态调整 k |
| **与 Mamba/SSM 融合** | 用状态空间模型替代线性算子，获得长程建模能力 |
| **基函数正交化** | 强制基函数正交，可能提高稳定性和泛化能力 |

### 7.3 延伸阅读

| 论文 | 关联 |
|------|------|
| Ovsjanikov et al. (2012) *Functional Maps* | FUNCATTN 的理论灵感来源 |
| Vaswani et al. (2017) *Attention Is All You Need* | 标准 attention 的基础 |
| Transolver (Wu et al., 2024) | PDE 领域的前 SOTA，FUNCATTN 的对比基线 |
| Performer (Choromanski et al., 2021) | 核函数近似线性 attention 的代表 |
| Mamba (Gu & Dao, 2023) | 状态空间模型，另一种 O(n) 序列建模路径 |
| Fourier Neural Operator (Li et al., 2021) | 谱域算子学习，与 FUNCATTN 共享谱方法思路 |
| DeepONet (Lu et al., 2021) | 算子学习的另一主流范式 |