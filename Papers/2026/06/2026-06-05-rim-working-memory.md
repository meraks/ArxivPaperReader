# Reasoning in Memory (RiM): 解锁大语言模型的工作记忆实现潜在推理

**论文标题：** Reasoning in Memory (RiM): Unlocking the Working Memory of Large Language Models for Latent Reasoning  
**作者：** Lukas Aichberger, Sepp Hochreiter  
**机构：** ELLIS Unit Linz, LIT AI Lab, Johannes Kepler University Linz; NXAI GmbH  
**arXiv：** [2605.30343](https://arxiv.org/abs/2605.30343)  
**提交日期：** 2026-05-28 | **分类：** cs.CL / cs.AI  
**状态：** Preprint | **代码：** 暂无官方实现

---

## Ch1: 论文概述与核心贡献

### 1.1 一句话总结

RiM 用**固定的特殊 token 序列（memory blocks）**取代 chain-of-thought 的自回归推理步骤，让 LLM 在一个并行可处理的内在工作空间中完成"思考"——推理速度提升 ~27 倍，同时保持与 CoT 相当甚至更好的准确率。

### 1.2 问题本质

当前 LLM 推理的主流范式是 **Chain-of-Thought（CoT）**：模型必须把每一步推理"说"出来，逐 token 自回归生成。这个过程存在两个根本性低效：

1. **计算浪费在语言生成上**：模型需要把宝贵的计算资源花在语法正确性、句子流畅性、自然语言格式上，而非纯推理计算。
2. **顺序瓶颈**：每一步推理必须等待上一步完全生成后才能开始——100 个推理 token 意味着 100 次串行前向传播。

> **类比理解**：这就像要求一位数学家解方程时必须大声说出每一个心算步骤，包括"我把 3 移项到右边"、"现在两边同时除以 2"。真正高效的思考是安静的、内部的——这正是 RiM 的目标。

### 1.3 核心创新

RiM 提出的解决方案简洁而优雅：

- **Memory Blocks**：用 `[<b>, <m>, <m>, </b>]` 这样的固定特殊 token 序列作为模型的内部工作空间
- **单次前向传播**：所有 memory blocks 作为输入的一部分，在一次 forward pass 中并行处理
- **两阶段课程学习**：先教会模型在 blocks 中存储中间信息，再转为端到端的答案精炼

### 1.4 核心贡献

| # | 贡献 | 重要性 |
|---|------|--------|
| 1 | 提出 **RiM**——用固定 memory blocks 实现并行 latent reasoning | **核心方法创新** |
| 2 | 设计**两阶段课程学习**：监督 grounding → 答案迭代精炼 | **训练范式贡献** |
| 3 | 在 GPT-2 和 Llama-3.2 系列上匹配/超越 Coconut 等现有 latent 方法 | **实验验证** |
| 4 | 推理速度 ~27× CoT，~7× Coconut，接近直接回答的延迟 | **效率突破** |
| 5 | 通过 PCA 可视化和 linear probe 揭示 memory blocks 编码有意义信息 | **可解释性贡献** |

### 1.5 论文定位

RiM 不属于"让模型更好地说话"的 CoT 优化路线，也不完全是 Coconut 的 latent reasoning 延续。它属于 **"让模型学会思考而不是说话"** 的新路线——**Implicit Latent Reasoning with Filler Tokens**。与之前 filler token 工作（如 Pfau et al. 2024 需要特殊预训练，Goyal et al. 2024 需要密集监督）不同，RiM 仅用简单的两阶段 NTP 目标就实现了有效的 filler token 训练。

---

## Ch2: 研究背景与动机

### 2.1 Chain-of-Thought 的困境

自 Wei et al. (2022) 和 Kojima et al. (2022) 提出 CoT 以来，"思考出声"已成为 LLM 推理的标准范式。OpenAI o1/o3、DeepSeek-R1 等推理模型的核心机制都是在推理时生成大量中间 token。

**CoT 的效率代价：**

```
CoT: 问题 → "第一步..." → "第二步..." → ... → "最终答案"
       ↑ 每个箭头 = 一次串行自回归生成
```

对于一个需要 100 个推理 token 的问题，CoT 需要 100 次串行前向传播。Time-To-First-Token（TTFT）——从输入到第一个输出 token 的时间——可能达到数百毫秒。

### 2.2 Latent Reasoning 路线的演进

研究者们意识到不需要显式的自然语言推理步骤，可以用连续的潜在表示替代：

**Coconut (Cheng & Durme, 2024, COLM 2025)**：用连续表示 `z_1, z_2, ..., z_L` 替代文本推理步骤。但关键局限是：每个 `z_ℓ` 仍然需要自回归生成——前一个 `z` 的输出必须反馈为下一个 `z` 的输入。

```
Coconut: 问题 → z₁ → z₂ → ... → z_L → 答案
          ↑ 仍然是串行的！
```

**DART (Shen et al., 2025)**：使用"规划 token"的方案，但需要额外的多路径训练和密集监督信号。

### 2.3 Filler Token 的尝试与失败

更激进的思路是用**无意义的 token**（filler tokens）替代推理步骤：

- **Lanham et al. (2023)**：发现 filler tokens 可以提升性能，但难以训练
- **Pfau et al. (2024)**：需要在**预训练阶段**就引入 filler tokens，成本高昂
- **Goyal et al. (2024)**：需要**密集的中间监督**和多路径设计

这些方法共同的困境：filler tokens 初始时没有任何意义，模型很容易忽略它们。

### 2.4 认知科学启示

RiM 从人类认知中获得了关键启发：

> **Baddeley 工作记忆模型 (1992)**：人类有一个容量有限的中央执行系统，可以在不外部化的情况下保持和操作信息。

> **Vygotsky 发展理论 (1978)**：儿童最初通过"自言自语"（private speech）来思考，随着认知发展，这种外部言语逐渐内化为无声的内部思考。

这个类比精准对应了 RiM 的两阶段设计：
- **Stage 1（对应"自言自语"）**：通过明确的步骤监督教会模型使用 memory blocks
- **Stage 2（对应"内化"）**：撤去中间监督，模型在内部完成推理

> **类比理解**：想象教一个孩子做数学。一开始你要求他每一步都说出来（Stage 1）。熟练后你说"在心里算"（Stage 2）。RiM 做的就是同一件事——只不过学生是 LLM。

### 2.5 RiM 在这一版图中的位置

| 方法 | 中间表示 | 生成方式 | 训练难度 | 推理速度 |
|------|---------|---------|---------|---------|
| CoT | 自然语言 | 自回归 | 低 | 慢（1×） |
| Coconut | 连续向量 | 自回归 | 中 | 较快（~4×） |
| Pfau et al. | Filler tokens | 并行 | 极高（需预训练） | 快 |
| **RiM** | **Filler tokens** | **并行** | **中（两阶段）** | **极快（~27×）** |

RiM 的核心突破在于：**用最简单的两阶段 NTP 训练目标解决了 filler token 的训练难题**。

---

## Ch3: RiM 方法详解

### 3.1 Memory Blocks 的形式化定义

RiM 的核心数据结构是 **memory block**。设输入问题为 `x`，答案标签为 `y`。定义 `K` 个 memory blocks `m₁, m₂, ..., m_K`，每个 block 的结构为：

```
m_k = [<b>, <m>, <m>, ..., <m>, </b>]
      ↑                    ↑        ↑
    开始标记          M个填充token  结束标记
```

- **`<b>` 和 `</b>`**：block 边界标记（类似 HTML 标签）
- **`<m>`**：核心工作 token，默认 `M = 2`（即每个 block 有 2 个 `<m>` token）
- **所有特殊 token 的 embedding 可训练**，而预训练模型的其余 token embedding **保持冻结**
- **Memory blocks 附加在问题后面**：输入序列变为 `(x, m₁, m₂, ..., m_K)`

> **类比理解**：把 memory block 想象成**计算器的内存寄存器**。你输入问题，按了几个"思考键"（memory blocks），然后得到答案。你不需要看到中间的计算过程——寄存器在内部完成了所有工作。

### 3.2 为什么是并行处理？

这是 RiM 与所有自回归 latent 方法（Coconut 等）的本质区别：

```
Coconut（自回归 latents）：
  前向1: x → z₁
  前向2: x, z₁ → z₂  
  前向3: x, z₁, z₂ → z₃
  ...
  前向L: x, z₁..z_L-1 → z_L
  前向L+1: x, z₁..z_L → 答案
  → 共 L+1 次串行前向传播

RiM（并行 blocks）：
  单次前向: x, m₁, m₂, ..., m_K → 答案
  → 仅 1 次前向传播！
```

因为 memory blocks 是输入的一部分（不是生成的），所有 blocks 的 representations 在一次 forward pass 中通过 transformer 的 self-attention 并行计算。

**代价**：`K` 个 blocks 增加了输入序列长度，但现代 transformer 的 attention 对序列长度的计算是 O(n²) 的，而自回归生成是 O(L × n²) 的。当 L（推理步数）较大时，并行 memory blocks 的优势极为明显。

### 3.3 自定义因果注意力掩码

标准的 causal attention mask 会让每个 token 看到它之前的所有 token。这在 RiM 中会导致信息泄漏：后一个 readout 可以直接看到前一个 readout 预测的内容。

RiM 设计了**分段的因果掩码**（Figure 2）：

```
            Question  m₁  m₂  ...  m_K  Readout₁  Readout₂  ...  Readout_K
Question       ✓       ✗   ✗       ✗      ✗         ✗             ✗
m₁             ✓       ✓   ✗       ✗      ✗         ✗             ✗
m₂             ✓       ✓   ✓       ✗      ✗         ✗             ✗
...            ✓       ✓   ✓       ✓      ✗         ✗             ✗
m_K            ✓       ✓   ✓       ✓      ✗         ✗             ✗
Readout₁       ✓       ✓   ✗       ✗      ✗         ✗             ✗
Readout₂       ✓       ✓   ✓       ✗      ✗         ✗             ✗
Readout_K      ✓       ✓   ✓       ✓      ✗         ✗             ✗
```

关键规则：
- **Memory blocks 之间**：后面的 block 可以看到前面的 block（标准因果）
- **Readouts 不能互相看到**：每个 readout 只能看到问题 + 它前面的 memory blocks
- **Memory blocks 不能看到 readouts**：保证信息只能单向流动

> **类比理解**：这就像一个多层车库。每层（memory block）可以停不同的车（信息），但层与层之间有单向闸门——只能上去不能下来。每层的管理员（readout）只能看到自己这层以下停了什么车，看不到其他管理员在看什么。

### 3.4 Stage 1: 推理步骤监督（Grounding）

**目标**：教会 memory blocks 承载任务相关的中间信息。

**训练数据格式**：`(x, r₁, r₂, ..., r_T, y)` ——包含显式推理步骤的 CoT 数据。

**训练策略**：
- 为 `T` 个推理步骤分配 `T` 个 memory blocks
- 在 block `m_t` 之后附加一个 readout，训练它预测**下一个推理步骤** `r_{t+1}`
- `r_{T+1} = y`（最后一步是最终答案）

**损失函数：**

$$L_{S1}(w) = -\sum_{t=1}^{T} \lambda_t(s) \cdot \log p_w(r_{t+1} \mid x, m_{\leq t})$$

其中 $\lambda_t(s) \in [0, 1]$ 是一个**线性退火调度**：
- 从 1 逐渐衰减到 0
- 较早的推理步骤先"退火"（减少监督强度），较晚的步骤后"退火"
- 这意味着模型逐渐学会在更少的显式监督下工作

**为什么线性退火很重要？**

固定 $\lambda_t = 1$ 会让模型过度依赖显式步骤监督，无法过渡到 Stage 2。退火提供了一种"软着陆"：早期步骤学到强监督，后期逐步松开手，让模型适应独立的 latent reasoning。

> **类比理解**：教孩子骑自行车。Stage 1 就像辅助轮——每一步推理都有明确的指导（"现在踩左脚，现在踩右脚"）。线性退火则是逐渐松开辅助轮——你不再每一步都指导，但偶尔提醒一下。

### 3.5 Stage 2: 答案迭代精炼（Refinement）

**目标**：将 Stage 1 中 ground 好的 memory blocks 转化为专注于最终答案的固定计算序列。

**关键变化**：
- 使用**固定数量** `K` 个 memory blocks（通常 `K = 8`）——不再与推理步骤数一一对应
- **移除所有中间监督**——所有 readouts 都预测最终答案 `y`
- 不再需要 CoT 数据

**损失函数：**

$$L_{S2}(w) = -\sum_{k=1}^{K} \alpha_k \cdot \log p_w(y \mid x, m_{\leq k})$$

其中 $\alpha_k$ 是**线性递增权重**——后面的 block 对损失贡献更大：

$$\alpha_k \propto k$$

**为什么权重递增？** 后面的 memory blocks "看到"了更多信息（前面 blocks 的内容），理应有更好的答案预测能力。高权重鼓励模型在后面的 blocks 中更精准地预测答案。

**状态重置**：Stage 2 开始时，**optimizer state 和 learning rate scheduler 都被重置**。使用更低的学习率和更高的 dropout。这防止 Stage 1 的学习轨迹干扰 Stage 2 的优化目标。

### 3.6 训练的超参数决策

| 参数 | 值 | 理由 |
|------|-----|------|
| Stage 1 epochs | 6 | 给足够时间让 memory blocks 学会承载信息 |
| Stage 2 epochs | 2 | 精炼阶段，防止过拟合 |
| LoRA rank | 128 | 参数效率 + 足够表达能力 |
| M (`<m>` per block) | 2 | 最小有效配置，减少序列长度开销 |
| K (Stage 2 blocks) | 8 | 在计算量和推理能力间平衡 |
| Token embedding 策略 | 特殊 token 可训练，其余冻结 | 防止干扰预训练知识 |
| Stage 2 LR | 更低（相对 Stage 1） | 精细调节 |
| Stage 2 dropout | 更高 | 防止 Stage 2 过拟合小数据集 |

---

## Ch4: 实验设计与分析

### 4.1 实验设置总览

**训练数据**：GSM8K-Aug，386k 小学数学应用题，包含最多 13 个推理步骤。这是一个经过增强的 GSM8K 版本，为每道题提供了分步骤的 CoT 推理链。

**评估基准**：
- **GSM8K（分布内）**：标准小学数学推理基准
- **GSM-Hard（分布外）**：更难的小学数学问题，测试泛化能力

**验证策略**：16 折交叉验证（held-out GSM8K samples），选择最佳 checkpoint。

**模型家族**：
- GPT-2（小模型，验证方法可行性）
- Llama-3.2-1B（中等规模）
- Llama-3.2-3B（较大规模）

**基线方法**：
- SFT w/o CoT：直接预测答案（仅问题 → 答案）
- SFT w/ CoT：标准 CoT 监督微调（问题 → 推理步骤 → 答案）
- Coconut ± Stage 0：Coconut latent reasoning，有/无额外预训练阶段
- DART：filler token 基线

### 4.2 主要结果

#### GPT-2 在 GSM8K 上的结果

| 方法 | GSM8K (Greedy%) | GSM-Hard (Greedy%) | TTFT (ms) |
|------|-----------------|--------------------|------------|
| SFT w/o CoT | 15.4 ± 0.2 | 3.5 ± 0.1 | 7.6 |
| Coconut w/ S0 | 31.1 ± 0.2 | 7.1 ± 0.0 | 53.4 |
| **RiM (Final block)** | **33.6 ± 0.2** | **7.8 ± 0.1** | **7.6** |
| SFT w/ CoT | 39.8 ± 0.2 | 8.4 ± 0.0 | 213.7 |

**关键观察：**

1. **RiM vs Coconut**：在准确率上超越 Coconut（+2.5% 绝对值），而 TTFT 仅为其 1/7（7.6ms vs 53.4ms）
2. **RiM vs CoT**：准确率差距仅 6.2%，但速度提升 28.1 倍（213.7/7.6）
3. **RiM vs Direct**：准确率翻倍（33.6% vs 15.4%），TTFT 几乎相同
4. **GSM-Hard 上的泛化**：RiM 在 OOD 上保持了优势，说明不是简单的记忆

> **类比理解**：如果把 CoT 比作用打字机逐字输出推理过程，RiM 就像一个专用计算芯片——它不需要把过程"打"出来，需要的计算直接在硅片上完成。

#### Llama-3.2 系列结果

| 模型 | 方法 | GSM8K 准确率 |
|------|------|-------------|
| Llama-3.2-1B | Direct-answer SFT | 24.8% |
| Llama-3.2-1B | Coconut | 35.6% |
| Llama-3.2-1B | **RiM** | **43.1%** |

在更大的模型上，RiM 的优势更加显著：相对 Coconut 提升 7.5 个百分点。

### 4.3 Memory Block 数量的鲁棒性

论文的一个关键发现是：Stage 2 训练后，模型的推理性能对 memory block 数量 `K` 表现出显著的**鲁棒性**。

- **训练时**：使用固定 `K = 8`
- **推理时**：将 K 从 4 变化到 16
- **结果**：性能保持稳定，变化幅度 < 2%

**这意味着什么？** 模型学会的不是"记住第 k 步应该做什么"，而是一种**迭代精炼（iterative refinement）**的能力——每个额外的 memory block 都是进一步的"思考"，而不是预先规划好的步骤。类似人类解决难题时"再多想一会儿"。

> **类比理解**：训练时给模型 8 个"思考槽位"。推理时给它 12 个槽位，它不会说"我不知道怎么用多出来的 4 个"——它自然地把多余的槽位用来进一步精炼答案。这就像一个棋手，你让他多想 1 分钟，他不会困惑——他会用这段时间分析更多变化。

### 4.4 潜在轨迹分析（PCA）

论文通过 PCA 降维可视化了 memory blocks 的 hidden representations：

- **第 1 个 block**：所有问题的表示聚集在公共区域（尚未分化）
- **第 3-5 个 block**：不同问题的表示开始**分叉**——模型在针对每个问题进行特定计算
- **第 8 个 block**：表示高度分化，且正确和不正确的样本形成了**可区分的聚类**

结论：memory blocks 不是静态的"占位符"——它们的 hidden states 经历了系统的、问题特定的演化。

### 4.5 Linear Probe 实验

在每个 memory block 的位置训练一个简单的线性分类器，预测**最终答案是否正确**：

- 从 block 1 到 block 8，probe 的准确率**单调递增**
- 第 8 个 block 的 probe 可以达到 > 80% 的二分类准确率

这表明 memory blocks 编码了一种**内部"信心"信号**——模型在 latent space 中逐步建立了对答案正确性的评估。

> **类比理解**：这就像考试时你心里的"感觉"。做完一道题后，你可能隐隐感觉"这个答案应该对"或"这里可能有问题"。RiM 的 linear probe 实验表明，这种"感觉"是可以从 hidden states 中定量读出的——模型内部确实在评估自己的推理质量。

---

## Ch5: 代码实现详解

> ⚠️ **注意**：以下代码为基于论文描述的概念实现，非官方代码。论文作者（Lukas Aichberger, Sepp Hochreiter）尚未发布官方实现。目的为帮助理解 RiM 的核心机制，不可直接用于训练。

### 5.1 Memory Blocks 的 Embedding 层

```python
# ⚠️ 非官方概念实现 — 基于论文 Section 3 描述编写
# 目的：帮助理解 memory block 的 embedding 设计
# 与官方实现的主要差异：LoRA 注入位置、mask 实现细节可能不同

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class RiMModel(nn.Module):
    """
    Reasoning in Memory (RiM) 的核心模块。
    在预训练 LLM 基础上添加可训练的 memory block embeddings。
    """
    
    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-3.2-1B",
        num_memory_tokens: int = 2,  # M: <m> tokens per block
        num_blocks: int = 8,          # K: number of memory blocks
    ):
        super().__init__()
        # 加载预训练模型
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.config = self.base_model.config
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.hidden_size = self.config.hidden_size
        
        # 冻结预训练参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 添加特殊 tokens: <b>, <m>, </b>
        special_tokens = {'additional_special_tokens': ['<b>', '<m>', '</b>']}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        # 可训练的 memory token embeddings（只训练这三个特殊 token）
        self.b_token_id = self.tokenizer.convert_tokens_to_ids('<b>')
        self.m_token_id = self.tokenizer.convert_tokens_to_ids('<m>')
        self.eb_token_id = self.tokenizer.convert_tokens_to_ids('</b>')
        
        # 核心参数
        self.num_memory_tokens = num_memory_tokens  # M
        self.num_blocks = num_blocks                # K
    
    def build_memory_block(self, batch_size: int, device: torch.device):
        """构建单个 memory block 的 token IDs: [<b>, <m>(×M), </b>]"""
        block_len = 1 + self.num_memory_tokens + 1  # <b> + M×<m> + </b>
        block = torch.full((batch_size, block_len), self.m_token_id, device=device)
        block[:, 0] = self.b_token_id
        block[:, -1] = self.eb_token_id
        return block  # (B, 2+M)
    
    def build_memory_blocks(self, batch_size: int, device: torch.device):
        """构建 K 个 memory blocks，拼接为一个序列"""
        blocks = []
        for _ in range(self.num_blocks):
            blocks.append(self.build_memory_block(batch_size, device))
        return torch.cat(blocks, dim=1)  # (B, K*(2+M))
```

### 5.2 自定义注意力掩码

```python
def build_rim_attention_mask(
    input_ids: torch.Tensor,
    question_len: int,
    block_token_count: int,  # 每个 block 的总 token 数: 2+M
    num_blocks: int,
    num_readouts: int,
    readout_len: int = 1,  # 每个 readout 的 token 数
) -> torch.Tensor:
    """
    构建 RiM 的分段因果注意力掩码。
    
    规则：
    - Memory blocks 之间：标准因果（后续 block 可见前面的 block）
    - Readouts：只能看到问题 + 自己前面的 memory blocks
    - Readouts 之间互不可见
    """
    total_len = input_ids.shape[1]
    mask = torch.zeros(total_len, total_len, dtype=torch.bool)
    
    # 问题部分：所有 token 都可互见（双向注意力）
    mask[:question_len, :question_len] = True
    
    # Memory blocks 的起始位置
    block_start = question_len
    
    # 对每个 memory block（按顺序）
    for b in range(num_blocks):
        b_start = block_start + b * block_token_count
        b_end = b_start + block_token_count
        
        # Block 内所有 token 可互见
        mask[b_start:b_end, b_start:b_end] = True
        # 可见问题
        mask[b_start:b_end, :question_len] = True
        # 可见之前的 blocks
        mask[b_start:b_end, block_start:b_start] = True
    
    # Readouts 的起始位置
    readout_start = block_start + num_blocks * block_token_count
    
    # Readout 1 只看到问题 + block 1
    # Readout 2 只看到问题 + blocks 1-2
    # ...
    for r in range(num_readouts):
        r_start = readout_start + r * readout_len
        r_end = r_start + readout_len
        # 可见问题
        mask[r_start:r_end, :question_len] = True
        # 可见到 block_{r+1}
        mask[r_start:r_end, block_start:block_start + (r+1) * block_token_count] = True
    
    # 转换为因果掩码格式（False = 被遮盖）
    causal_mask = ~mask
    
    return causal_mask  # (total_len, total_len)
```

### 5.3 Stage 1: 推理步骤监督训练

```python
def stage1_forward(
    model: RiMModel,
    input_ids: torch.Tensor,
    reasoning_steps: list[torch.Tensor],  # T 个推理步骤
    answer_ids: torch.Tensor,
):
    """
    Stage 1: 每个 memory block 后预测下一个推理步骤。
    
    输入格式: [question] [block₁] [step₁] [block₂] [step₂] ... [block_T] [answer]
    其中 step_t 是推理步骤的 target，block_t 的 readout 需要预测 step_{t+1}
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device
    question_len = input_ids.shape[1]
    T = len(reasoning_steps)
    
    # 构建 memory blocks
    memory_blocks = model.build_memory_block(batch_size, device)
    
    # 组装序列: question + block₁ + step₁ + block₂ + step₂ + ... + block_T + answer
    sequence_parts = [input_ids]
    for t in range(T):
        sequence_parts.append(memory_blocks)
        sequence_parts.append(reasoning_steps[t])
    sequence_parts.append(memory_blocks)  # 最后一个 block
    sequence_parts.append(answer_ids)
    
    full_input = torch.cat(sequence_parts, dim=1)
    
    # 构建注意力掩码
    block_tokens = 1 + model.num_memory_tokens + 1  # <b> + M×<m> + </b>
    # 计算每个 step 的 token 长度
    step_lens = [step.shape[1] for step in reasoning_steps]
    answer_len = answer_ids.shape[1]
    
    mask = build_stage1_attention_mask(
        question_len, block_tokens, T + 1, step_lens, answer_len
    )
    
    # 前向传播
    outputs = model.base_model(
        inputs_embeds=model.base_model.get_input_embeddings()(full_input),
        attention_mask=mask,
    )
    logits = outputs.logits
    
    # 计算 Stage 1 损失
    loss = 0.0
    # 对每个 block，计算它对下一个推理步骤的预测损失
    current_pos = question_len  # 当前位置
    
    for t in range(T):
        # 跳过当前 block
        current_pos += block_tokens
        
        # 当前 block 的 readout 预测 reasoning_steps[t]
        step_len = step_lens[t]
        step_logits = logits[:, current_pos:current_pos + step_len, :]
        step_targets = reasoning_steps[t]
        
        # Stage 1 退火权重 λ_t
        lambda_t = compute_lambda(t, T, current_step=...)
        
        loss += lambda_t * nn.CrossEntropyLoss()(
            step_logits.reshape(-1, step_logits.size(-1)),
            step_targets.reshape(-1)
        )
        
        current_pos += step_len
    
    # 最后一个 block → answer
    current_pos += block_tokens
    answer_logits = logits[:, current_pos:current_pos + answer_len, :]
    lambda_T = 1.0  # answer 不退火
    loss += lambda_T * nn.CrossEntropyLoss()(
        answer_logits.reshape(-1, answer_logits.size(-1)),
        answer_ids.reshape(-1)
    )
    
    return loss / (T + 1)
```

### 5.4 Stage 2: 答案精炼训练

```python
def stage2_forward(
    model: RiMModel,
    input_ids: torch.Tensor,
    answer_ids: torch.Tensor,
):
    """
    Stage 2: K 个 memory blocks 后都预测最终答案。
    
    输入格式: [question] [block₁] [answer_copy₁] [block₂] [answer_copy₂] ... [block_K] [answer_copy_K]
    其中每个 answer_copy 都预测同一个 answer。
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device
    question_len = input_ids.shape[1]
    K = model.num_blocks
    block_tokens = 1 + model.num_memory_tokens + 1
    answer_len = answer_ids.shape[1]
    
    # 构建 memory blocks
    memory_block = model.build_memory_block(batch_size, device)
    
    # 组装序列: question + [block + answer] × K
    sequence_parts = [input_ids]
    for k in range(K):
        sequence_parts.append(memory_block)
        # 每个 block 后放置 answer（作为 readout）
        sequence_parts.append(answer_ids)
    
    full_input = torch.cat(sequence_parts, dim=1)
    
    # Stage 2 注意力掩码
    mask = build_stage2_attention_mask(question_len, block_tokens, K, answer_len)
    
    # 前向传播
    outputs = model.base_model(
        inputs_embeds=model.base_model.get_input_embeddings()(full_input),
        attention_mask=mask,
    )
    logits = outputs.logits
    
    # 计算 Stage 2 损失
    loss = 0.0
    current_pos = question_len
    
    for k in range(K):
        # 跳过 block
        current_pos += block_tokens
        
        # 当前 block 的 readout 预测 answer
        answer_logits = logits[:, current_pos:current_pos + answer_len, :]
        
        # Stage 2 权重 α_k：线性递增
        alpha_k = (k + 1) / K
        
        loss += alpha_k * nn.CrossEntropyLoss()(
            answer_logits.reshape(-1, answer_logits.size(-1)),
            answer_ids.reshape(-1)
        )
        
        current_pos += answer_len
    
    return loss / K
```

### 5.5 推理（Inference）

```python
@torch.no_grad()
def rim_inference(
    model: RiMModel,
    question: str,
    num_blocks: int = 8,
    max_new_tokens: int = 50,
):
    """RiM 推理：单次前向传播，仅解码答案。"""
    device = next(model.base_model.parameters()).device
    
    # Tokenize 问题
    question_ids = model.tokenizer.encode(question, return_tensors='pt').to(device)
    
    # 构建 K 个 memory blocks（拼接在问题后面）
    memory_blocks = model.build_memory_blocks(1, device)
    
    # 拼接输入（单次前向！）
    full_input = torch.cat([question_ids, memory_blocks], dim=1)
    
    # 前向传播获取所有 memory blocks 的 hidden states
    outputs = model.base_model(full_input, output_hidden_states=True)
    
    # 第 K 个 block 的最后一个 hidden state 作为"思考结果"
    # 论文中：使用最后一个 memory block 的输出作为答案解码的起点
    last_block_end = question_ids.shape[1] + num_blocks * (1 + model.num_memory_tokens + 1)
    thinking_output = outputs.hidden_states[-1][:, last_block_end - 1, :]  # (1, hidden)
    
    # 基于 thinking_output 解码最终答案
    # （实际实现中可能需要投影层或直接 decode）
    logits = model.base_model.lm_head(thinking_output)  # (1, vocab_size)
    
    # 采样/greedy 解码
    answer_tokens = torch.argmax(logits, dim=-1, keepdim=True)
    # ... 继续自回归解码直到 EOS 或 max_new_tokens
    
    return model.tokenizer.decode(answer_tokens[0])
```

### 5.6 与 Coconut 的核心实现差异

```python
# Coconut: 自回归生成 latent representations
def coconut_inference(model, question, num_latents=8):
    z_list = []
    current_input = question
    for _ in range(num_latents):
        # 每步都需要单独的前向传播！
        h = model(current_input)  # 第 i 次前向
        z = project_to_latent(h)  # 投影到 latent space
        z_list.append(z)
        current_input = concat(current_input, z)  # 反馈为下一步输入
    # 第 num_latents+1 次前向：解码答案
    return model.decode(current_input)

# RiM: 所有 memory blocks 一次前向处理
def rim_inference(model, question, num_blocks=8):
    memory_blocks = build_blocks(num_blocks)  # 固定 token IDs
    full_input = concat(question, memory_blocks)
    h = model(full_input)  # 仅 1 次前向传播！
    return model.decode(h)
```

**效率对比的数学本质**：

- Coconut: `O(num_latents × n²)` 次 attention 计算（n 随 latent 累积增长）
- RiM: `O(n²)` 次 attention 计算（n 为 question + blocks 的固定长度）

当 `num_latents = 8` 时，RiM 的单次前向传播大约节省 8× 的 attention 计算量。

---

## Ch6: 局限性与延伸阅读

### 6.1 方法局限性

**1. 数据集范围有限**

当前实验仅在小学数学推理（GSM8K/GMS-Hard）上验证。是否适用于更复杂的推理任务（数学竞赛、代码生成、科学推理）尚待验证。GSM8K 的推理步骤通常较浅（平均 ~5 步），问题结构相对固定。

**2. 有限的 CoT 数据依赖**

Stage 1 需要分步骤的 CoT 标注数据。对于很多领域（如代码调试、逻辑推理），高质量的步骤级标注可能稀缺或昂贵。

**3. 模型规模**

实验最大模型为 Llama-3.2-3B。对于 7B/13B/70B 级别模型，RiM 是否仍然有效、是否需要调整 memory block 设计、性能增益是否会随模型规模变化——这些关键问题尚未探索。

**4. 可解释性的丧失**

这是 RiM（以及所有 latent reasoning 方法）的根本性 trade-off。CoT 的优势在于每一步推理都是可阅读、可审计的。RiM 的 memory blocks 是黑箱——我们只能通过 PCA 和 probing 间接推断其内容，无法直接检查推理是否合理。

**5. 与 CoT 的精度差距**

在 GPT-2 上，RiM (33.6%) 仍明显低于 CoT (39.8%)——绝对差距 6.2%。虽然速度提升巨大，但绝对精度仍低于显式推理。在某些精度至上的场景中，CoT 可能仍然是首选。

**6. Filler Token 身份的选择**

论文固定使用了 3 种特殊 token（`<b>`, `<m>`, `</b>`）。是否有更优的 token 设计？不同的 token 数量、embedding 初始化策略如何影响性能？这些问题尚未系统研究。

### 6.2 未探索的研究方向

**1. RiM + 强化学习**

当前 RiM 使用监督微调（NTP loss）。结合 GRPO/RLHF 等 RL 方法，让模型在 latent space 中通过奖励信号自我优化推理策略，可能是巨大的提升方向。

**2. 混合 RiM-CoT**

最佳策略可能不是全有或全无。在简单步骤上使用 memory blocks（快速），在关键决策点"外化"为自然语言（可审计）。这种认知架构可能同时获得速度和可解释性。

**3. Curriculum 的可微分设计**

当前 Stage 1→2 的切换是硬编码的（6 epochs 后切换）。设计可微分的 curriculum 调度——让模型自动学习何时从"grounding"过渡到"精炼"——是一个有趣的理论方向。

**4. Memory Blocks 的动态分配**

当前 memory blocks 在所有问题上使用相同的固定数量 `K`。是否可以学习**动态分配**——简单问题用 2 个 blocks，难题用 16 个？这与人类的"思考努力调节"机制高度一致。

**5. 多模态 RiM**

将 memory blocks 扩展到视觉-语言模型：在图像理解中也使用 latent workspace 进行推理，而非生成冗长的视觉描述。

### 6.3 与相关工作的联系

| 相关工作 | 与 RiM 的关系 |
|---------|--------------|
| **Coconut** (Hao et al., 2024) | 最直接的 latent reasoning 对比基线；RiM 解决了其自回归瓶颈 |
| **DART** (Shen et al., 2025) | 同为 filler token 方案，但 RiM 的训练目标更简洁 |
| **COCONUT** (Cheng & Durme, 2024) | 连续 latent reasoning；RiM 用离散 filler tokens 替代连续表示 |
| **Let's Verify Step by Step** (Lightman et al., 2024) | PRM；可能与 RiM 结合——在 memory blocks 之间注入验证信号 |
| **Quiet-STaR** (Zelikman et al., 2024) | 在 token 之间插入"思考 token"；思想类似但训练方式完全不同 |
| **Buffer of Thoughts** (Yang et al., 2024) | 模板化思维；RiM 可以视为其 latent 版本 |
| **xLSTM** (Beck et al., 2024) | Hochreiter 团队的并列工作；xLSTM 的记忆机制与 RiM 的工作记忆概念互补 |

### 6.4 延伸阅读

1. **Hao et al. "Training Large Language Models to Reason in a Continuous Latent Space"** (arXiv:2412.06769) — Coconut 方法，RiM 的主要对比基线
2. **Pfau et al. "Let's Think Dot by Dot: Hidden Computation in Transformer Language Models"** (arXiv:2404.15758) — Filler token 的开创性工作
3. **Aichberger et al. "SDLG: Efficient Estimation of Aleatoric Semantic Uncertainty in LLMs"** — 同作者的前置工作
4. **Hochreiter & Schmidhuber. "Long Short-Term Memory"** (Neural Computation, 1997) — LSTM 的经典论文，理解 Hochreiter 对记忆机制的长期思考
5. **Baddeley. "Working Memory"** (Science, 1992) — 工作记忆的认知科学基础
6. **Vygotsky. "Mind in Society"** (1978) — 内部言语和认知发展理论
7. **DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"** (2025) — 当前最先进的 CoT 推理方法

---

## Ch7: 总结与展望

### 7.1 RiM 的方法论贡献

RiM 的核心贡献不在于"又一个比 Coconut 好的 latent reasoning 方法"，而在于它证明了三个重要的设计原则：

1. **并行 latent reasoning 是可行的**：filler tokens 不需要自回归生成，固定序列同样可以作为有效的工作空间
2. **简单的两阶段训练就够了**：不需要复杂的辅助损失、多路径设计或特殊预训练——标准 NTP 目标 + 智能的课程设计就足以训练 filler tokens
3. **工作记忆类比是有效的**：借鉴认知科学的工作记忆概念，为训练设计提供了直观且有效的框架

### 7.2 对 AI 推理研究的启示

RiM 指出了一个关键方向：**LLM 的推理能力不一定要绑定在语言生成上**。当前最先进的推理模型（o1/o3、R1）都在大量生成 CoT token，而 RiM 表明：大量的"思考"可以在 latent space 中完成——更快、更高效。

这可能代表了推理范式的一次重要转向：从 **"思考 = 说话"** 到 **"思考 ≠ 说话"**。

### 7.3 通往实用 RiM 的路线图

短期（1-6 个月）：
- 将 RiM 扩展到 7B+ 模型
- 在更多推理基准上评估（MATH、ARC、GPQA）
- 发布官方代码实现

中期（6-18 个月）：
- RiM + 强化学习（GRPO/RLHF）
- 混合 RiM-CoT 架构
- 动态 memory block 分配

长期（18+ 个月）：
- RiM 作为推理模型的默认 latent 推理引擎
- 与外部工具使用（tool use）集成
- 多模态 latent reasoning

---

> **论文评分**：Novelty 3 | Impact 2.5 | Technical 3 | Evidence 2.5 | Reusability 1.5 → **总分 12.5**，Deep Read 级别

> **评价**：这是一篇概念清晰、方法简洁、结果扎实的论文。来自 LSTM 发明人 Hochreiter 的团队，延续了其对记忆/推理机制的长期关注。最大的遗憾是暂无官方代码和更大规模模型的验证。作为 latent reasoning 方向的新 baseline，RiM 有潜力成为后续研究的重要参照点。