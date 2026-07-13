# KV-PRM: Efficient Process Reward Modeling via KV-Cache Transfer for Multi-Agent Test-Time Scaling

**arXiv:2607.09153** · Peng Kuang, Haibo Jin et al. · 2026

---

## 第一章 论文概述与核心贡献

### 1.1 问题

Process Reward Model (PRM) 引导 test-time scaling (TTS) 搜索时，传统 text-PRM 需重新编码整个推理轨迹以输出 step-level 评分。对于长度为 $L$ 的轨迹，text-PRM 的计算量为 $O(L^2)$，构成巨额计算瓶颈，严重限制 multi-agent system (MAS) 中多候选轨迹的并行验证效率。

### 1.2 方法

KV-PRM 复用生成阶段已产生的 KV cache，仅需在轨迹末尾追加单个 verify token 即可完成评分，计算量降为 $O(L)$。KV cache 作为生成的副产物，验证阶段无需重新编码任何前缀 token。

### 1.3 理论

论文证明 KV cache 的信息容量大于文本表示，且基于边际信息增益指数衰减的性质，理论上 $k=1$（单 verify token）已接近最优。

### 1.4 效率与性能

- FLOPs 减少 $5000\times$
- 延迟降低 $37\times$
- 内存减少 $34\times$

在 MATH、GSM8K、AIME 2024、AIME 2025 四个 benchmark 上，KV-PRM 性能持平或超越 text-PRM。

---

## 第二章 理论基础

### 2.1 线性表示假设

**Assumption 1 (Linear Representation Hypothesis, LRH)**：模型对隐藏状态空间的表示近似线性，隐藏向量可线性解码出对应的概念属性。该假设是后续信息密度与误差分解证明的基础。

### 2.2 KV cache 信息密度优势

**Proposition 1**：在 LRH 下，KV cache 的信息密度为文本 token 表示的 $\Omega\left(\frac{d}{\log|V|}\right)$ 倍，其中 $d$ 为隐藏维度，$|V|$ 为词表大小。

$$\rho_{\text{KV}} \geq \Omega\left(\frac{d}{\log|V|}\right) \cdot \rho_{\text{text}}$$

该结果表明 KV cache 以连续向量承载的信息远超离散文本 token。

### 2.3 验证误差分解

**Theorem 1**：评分器 $R_k$（使用 $k$ 个 verify token）的验证误差可分解为 Bayes 最优误差与表示能力间隙之和：

$$\epsilon(R_k) = \epsilon_{\text{Bayes}}(H) + \text{Gap}(F_k, H)$$

其中 $H$ 为真实假设，$F_k$ 为 $k$ 个 verify token 对应的函数族。误差下界由 Bayes 误差决定，KV-PRM 的差距来自 $F_k$ 对 $H$ 的逼近能力。

### 2.4 边际信息增益与 $k=1$ 近最优性

**Theorem 2**：每增加一个 verify token 带来的边际信息增益呈指数衰减：

$$\frac{\Delta I_k}{\Delta F} \propto \frac{e^{-\alpha \cdot n_h \cdot N_L \cdot (k-1)}}{d \cdot L}$$

其中 $\alpha$ 为衰减系数，$n_h$ 为注意力头数，$N_L$ 为层数，$L$ 为序列长度。由于 $k=1 \to k=2$ 的增益已随指数项急剧下降，单 verify token ($k=1$) 在信息-成本权衡下接近最优。

### 2.5 成本-信息权衡

将边际信息增益 $\Delta I_k$ 与边际计算成本 $\Delta F$ 结合，KV-PRM 选择 $k=1$ 在信息获取效率上达到近最优，额外 verify token 的收益无法补偿其计算开销。

---

## 第三章 KV-PRM 算法设计

### 3.1 架构

KV-PRM 由基础 LLM 与 LoRA 适配器组成。基础模型参数 $\theta$ 冻结，LoRA 参数 $\Delta\theta$ 可训练。verify token 设为 `"?"`，追加在候选轨迹末尾。

### 3.2 评分机制

给定轨迹的 KV cache $(K, V)$，评分函数计算 verify token 的 logit 并经 softmax 得出正向概率：

$$f_{\theta+\Delta\theta}(v \mid K, V) \xrightarrow{\text{softmax}} P(+)$$

$P(+)$ 即该步骤的正确性评分。

### 3.3 训练流程

- **标签生成**：通过 MCTS 搜索产生 step-level 正确性标签
- **损失函数**：MSE 损失
- **参数更新**：仅更新 LoRA 参数 $\Delta\theta$，基础模型 $\theta$ 保持冻结

### 3.4 计算成本分析

| 方法 | 计算复杂度 |
|------|-----------|
| Text-PRM | $O(d \cdot L^2)$ |
| KV-PRM | $O(d \cdot L)$ |

当 $L = 5000$ 时，KV-PRM 相对 text-PRM 的 FLOPs 减少约 $5000\times$。

### 3.5 推理开销

KV cache 是生成阶段的副产物，验证时直接复用，无需额外编码开销。

---

## 第四章 实验评估

### 4.1 实验设置

- **模型**：Qwen3-0.6B / 4B / 8B
- **Benchmark**：MATH、GSM8K、AIME 2024、AIME 2025
- **搜索算法**：Beam Search、MCTS、Weighted Voting
- **基线**：No PRM、Policy Log-prob、Text-PRM

### 4.2 主结果

Table 1（MATH + AIME 2024，sequential MAS）：

| Search | Config | Scorer | MATH-0.6B | MATH-4B | MATH-8B | AIME24-4B | AIME24-8B |
|--------|--------|--------|-----------|---------|---------|-----------|-----------|
| Random | n=1 | — | 28.00 | 54.15 | 56.70 | 20.00 | 20.00 |
| Majority | n=40 | — | 34.65 | 59.20 | 60.55 | 36.67 | 33.33 |
| Beam | W=1 | Text-PRM | 32.75 | 62.00 | 66.62 | 26.67 | 20.00 |
| Beam | W=1 | KV-PRM | 31.90 | 65.55 | 67.02 | 20.00 | 30.00 |
| MCTS | n=50 | Text-PRM | 35.55 | 65.95 | 68.92 | 30.00 | 36.67 |
| MCTS | n=50 | KV-PRM | 35.10 | 67.15 | 69.57 | 33.33 | 40.00 |

Table 2（GSM8K + AIME 2025，sequential MAS）：

| Search | Config | Scorer | GSM8K-0.6B | GSM8K-4B | GSM8K-8B | AIME25-4B | AIME25-8B |
|--------|--------|--------|-----------|---------|---------|-----------|-----------|
| Random | n=1 | — | 50.80 | 91.28 | 91.43 | 16.67 | 20.00 |
| Majority | n=40 | — | 62.77 | 93.10 | 93.03 | 20.00 | 16.67 |
| Beam | W=10 | Text-PRM | 58.68 | 93.78 | 95.00 | 26.67 | 26.67 |
| Beam | W=10 | KV-PRM | 58.00 | 92.80 | 95.07 | 26.67 | 30.00 |
| MCTS | n=30 | Text-PRM | 57.77 | 93.33 | 95.15 | 23.33 | 23.33 |
| MCTS | n=30 | KV-PRM | 54.74 | 92.87 | 94.62 | 23.33 | 26.67 |

### 4.3 效率对比

以 Qwen3-8B 为例，KV-PRM 相对 Text-PRM 的计算成本比：

| 配置 | KV-PRM / Text-PRM 成本比 |
|------|--------------------------|
| MATH + Beam Search | 1/2782 |
| MATH + MCTS | 1/4367 |
| AIME 2024 + MCTS | 1/3944 |

8B 模型上具体数值对比：

- MATH + Beam Search：KV-PRM 67.02 vs Text-PRM 66.62（+0.40），成本为 1/2782
- MATH + MCTS：KV-PRM 69.57 vs Text-PRM 68.92（+0.65），成本为 1/4367
- AIME 2024 + MCTS：KV-PRM 40.00 vs Text-PRM 36.67（+3.33），成本为 1/3944

### 4.4 分析

- 使用更小规模的 text-PRM 无法缩小与 KV-PRM 的效率差距
- $k > 1$ 时收益迅速饱和，印证 Theorem 2 的指数衰减结论

### 4.5 层次化 MAS

Table 3（Hierarchical MAS）：

| Dataset | Model | Text-PRM | KV-PRM | Δ |
|---------|-------|----------|--------|---|
| MATH | Qwen3-4B | 58.7 | 59.2 | +0.5 |
| MATH | Qwen3-8B | 60.2 | 61.0 | +0.8 |
| GSM8K | Qwen3-4B | 93.7 | 93.9 | +0.2 |
| GSM8K | Qwen3-8B | 94.2 | 94.5 | +0.3 |
| AIME 2024 | Qwen3-4B | 23.3 | 26.7 | +3.3 |
| AIME 2024 | Qwen3-8B | 23.3 | 30.0 | +6.7 |
| AIME 2025 | Qwen3-4B | 23.3 | 26.7 | +3.3 |
| AIME 2025 | Qwen3-8B | 16.7 | 20.0 | +3.3 |

层次化 MAS 拓扑下 KV-PRM 同样有效，AIME 2024 在 8B 模型上提升 +6.7pp。

---

## 第五章 KV Steering 与应用

### 5.1 KV Steering 方法

KV-PRM 的评分函数 $f_{\theta+\Delta\theta}$ 对 KV cache 可微，因此可通过梯度优化隐式消息（implicit message）的 KV 表示，直接 steering 生成行为，无需显式搜索。

### 5.2 实验结果

Table 4（KV Steering，无搜索）：

| Dataset | Model | No steering | KV Steering | Δ |
|---------|-------|------------|-------------|---|
| MATH | Qwen3-4B | 60.23 | 61.84 | +1.61 |
| MATH | Qwen3-8B | 60.79 | 62.00 | +1.21 |
| GSM8K | Qwen3-4B | 91.81 | 92.42 | +0.61 |
| GSM8K | Qwen3-8B | 93.86 | 94.09 | +0.23 |
| AIME 2024 | Qwen3-4B | 10.00 | 13.33 | +3.33 |
| AIME 2024 | Qwen3-8B | 13.33 | 16.67 | +3.33 |
| AIME 2025 | Qwen3-4B | 10.00 | 10.00 | +0.00 |
| AIME 2025 | Qwen3-8B | 13.33 | 16.67 | +3.33 |

KV Steering 在无搜索条件下：

- MATH：+1.6pp（4B 模型 +1.61）
- AIME 2024：+3.3pp（4B 与 8B 均 +3.33）

### 5.3 局限性

- **架构约束**：生成器与验证器需同架构，KV cache 方可直接复用
- **理论假设**：核心理论依赖 Linear Representation Hypothesis (Assumption 1)，若实际模型偏离 LRH，结论的适用性需进一步验证
