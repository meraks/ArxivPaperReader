# Unlocking the Working Memory of Large Language Models for Latent Reasoning

> **论文信息**  
> - **标题**：Unlocking the Working Memory of Large Language Models for Latent Reasoning  
> - **作者**：Lukas Aichberger¹, Sepp Hochreiter¹,²  
> - **机构**：¹ELLIS Unit Linz and LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria / ²NXAI GmbH, Linz, Austria  
> - **arXiv**：[2605.30343](https://arxiv.org/abs/2605.30343)  
> - **提交日期**：2026年5月28日  
> - **状态**：Preprint  
> - **官方代码**：暂无公开仓库

---

## Ch1. 论文概述与核心贡献

### 1.1 一句话总结

**RiM（Reasoning in Memory）** 提出用**固定的特殊token序列**（记忆块）替代LLM推理时自回归生成的中间步骤，将"思考"从串行的文本生成中解放出来，在单次forward pass中完成潜在推理——比显式CoT快~27倍，且性能匹配或超越。

### 1.2 问题定位

当前大模型推理能力的主流提升路径——Chain-of-Thought（CoT）——存在一个根本性矛盾：

> "This coupling forces an LLM to 'think out loud', anchoring its reasoning process to the syntax and structure of natural language. However, language is optimized for communication rather than for computation."

用人类认知类比：CoT相当于要求一个人解题时把**每一个脑内步骤都大声说出来**。语言是为交流优化的，不是为计算优化的。生成语法正确的中间步骤浪费了大量计算资源，更致命的是——**串行生成的latency瓶颈无法规避**。

### 1.3 核心创新

RiM做出了一个范式转变——**不再生成任何中间文本**。它引入：

1. **记忆块（Memory Blocks）**：固定序列的特殊token `[<b>, <m>, ..., <m>, </b>]`，作为LLM内部的"工作记忆"空间
2. **两阶段课程学习**：Stage 1用推理步骤监督建立grounding → Stage 2精炼最终答案
3. **并行处理**：所有记忆块作为输入token在单次forward pass中处理，彻底消除自回归瓶颈

### 1.4 核心贡献

| # | 贡献 | 意义 |
|---|------|------|
| 1 | 提出RiM框架——用固定记忆块替代自回归思想生成 | 计算与通信解耦的推理新范式 |
| 2 | 设计两阶段课程学习（grounding→refinement） | 解决"无中生有"的训练信号问题 |
| 3 | 定制注意力掩码防止信息泄露 | 确保计算真正发生在记忆块内 |
| 4 | 实验验证：~27x加速，GSM8K 43.1%准确率 | 效率+效果双重证明 |
| 5 | 推断时可变内存预算 | 无需重训即可调整"思考深度" |

---

## Ch2. 研究背景与动机

### 2.1 CoT的隐性成本

Chain-of-Thought prompting的成功让人们看到了LLM推理能力的巨大潜力。但其背后的成本——每个中间步骤都需一次forward pass——在长推理链面前急剧膨胀。

```
SFT w/ CoT: Question → [Step1 → Step2 → ... → StepT] → Answer
             ↑ 每次生成一个token需要一次forward pass
             ↑ 100个推理token ≈ 100次forward pass
             ↑ TTFT从7.6ms膨胀到~200ms (Llama-3.2-1B)
```

> **类比理解**：CoT像口算考试——你必须把每一步"念念有词"地写下来。这不仅浪费时间（每个字都要写），还可能因为遣词造句的负担干扰计算本身。真正的心算高手是在脑中直接操作数字，只说出最终答案。

### 2.2 已有潜在推理方法的局限

近年出现了多种"潜在推理"方法（如Coconut），它们将离散的文本token替换为连续向量表示，在"连续空间"中思考。

**核心问题**：Coconut等方法虽然消除了文本生成开销，但仍保留了**自回归生成范式**——它们逐个生成连续thought向量，依然是"在连续空间中自言自语"。

| 方法 | 中间表示 | 生成方式 | 主要瓶颈 |
|------|---------|---------|---------|
| CoT | 文本token | 自回归逐token | 文本生成开销 + 串行latency |
| Coconut | 连续向量 | 自回归逐向量 | 串行latency |
| **RiM** | 固定记忆块 | **单次forward pass** | 仅需设计训练信号 |

### 2.3 认知科学启发：人类工作记忆

Baddeley的工作记忆模型（1974）指出，人脑有一个**容量有限的短期存储系统**，用于临时保持和操作任务相关信息。关键特征：

- **内部操作**：不需要外化（说出来/写下来）
- **容量固定**：通常4±1个chunk
- **主动维护**：需要持续注意/复述

RiM的记忆块设计直接受到此启发：每个记忆块是固定容量的"chunk"，多个记忆块串联提供迭代精炼能力。

---

## Ch3. RiM方法详解：记忆块与Stage 1训练

### 3.1 记忆块结构

记忆块是一个**预定义的token序列**，而非生成的token：

```
mk = [<b>, <m>, <m>, ..., <m>, </b>]
     ↑               ↑ M个<m>         ↑
   开始标记        潜在工作空间     结束标记
```

**关键设计决策**：

| 属性 | 设计 | 原因 |
|------|------|------|
| Token身份 | 固定（所有问题共享相同token） | 无需生成，可并行处理 |
| Token位置 | 固定（每个block位置预设） | 位置编码一致，便于训练 |
| 上下文表征 | **动态、任务依赖** | 通过attention从问题+前置block聚合信息 |
| Embedding | 仅训练新token的嵌入 | 冻结已有词汇嵌入，防止干扰预训练知识 |

> **类比理解**：记忆块像一个"便签本"——纸是固定的（固定token），但你在上面写的字（上下文表征）取决于你在想什么。所有便签同时铺开在桌上，你可以一眼扫过（并行attention），不需要一张一张翻（自回归生成）。

### 3.2 为什么需要两阶段训练？

如果直接将记忆块插入输入并期望LLM"自动"学会在其中存储信息，**模型不会获得任何训练信号**。记忆块本身没有语义——它们不会"自动"变成工作记忆。

> 核心挑战：如何让模型学会在"空白的"特殊token中编码任务相关信息？

答案：**用推理步骤作为监督信号**（Stage 1），然后**过渡到最终答案精炼**（Stage 2）。

### 3.3 Stage 1：推理步骤监督（Grounding Phase）

**训练数据格式**：
```
输入：Question x + memory blocks [m₁, m₂, ..., mT]
目标：每个memory block m_t 后预测下一个推理步骤 r_{t+1}
```

**训练细节**：

1. 对于有T个推理步骤的问题，附加T个记忆块
2. 每个记忆块后追加一个readout，监督其预测**下一个**推理步骤
3. **关键**：定制注意力掩码确保每个readout只能attend到：
   - 问题x
   - 当前及之前的记忆块
   - **不能**attend到任何已写的推理步骤文本
4. 所有T个readout在**单次forward pass**中训练

**注意力掩码示意图**（简化表示）：

```
           Q   m₁  m₂  m₃  r₁  r₂  r₃
readout₁:  ✓   ✓   ✗   ✗   ✗   ✗   ✗   ← 只能看问题+当前block
readout₂:  ✓   ✓   ✓   ✗   ✗   ✗   ✗   ← 看不到r₁（防止泄露）
readout₃:  ✓   ✓   ✓   ✓   ✗   ✗   ✗   ← 看不到r₁,r₂
```

**Stage 1损失函数**：

$$L_{S1}(w) = -\sum_{t=1}^T \lambda_t(s) \log p_w(r_{t+1} \mid x, m_{\leq t})$$

其中 $\lambda_t(s)$ 是一个**退火系数**：在训练过程中从1线性退火到0，且**早期block的监督先消失**。这实现了"soft multi-stage curriculum"——模型逐渐从依赖显式监督过渡到独立使用记忆空间。

> **类比理解**：Stage 1像教练手把手教学生解题——"第一步这样做，第二步那样做"。教练在每一步后检查学生是否正确。但教练逐渐撤掉对早期步骤的检查（退火），迫使学生自己记住前面的信息并传递到后续步骤。

### 3.4 设计精妙之处

1. **信息隔离**：注意力掩码确保推理步骤不可见——记忆块**必须**自己编码所需信息，无法"偷看"已写步骤
2. **密集监督**：每个block都有loss信号，防止某些block成为"dead tokens"
3. **单次并行训练**：所有T个readout共享同一次forward，训练效率高
4. **软退火**：逐渐减少对早期block的监督，平滑过渡到自主编码

---

## Ch4. Stage 2训练与推断策略

### 4.1 Stage 2：最终答案精炼（Refinement Phase）

Stage 1给记忆块注入了"编码推理信息"的能力。Stage 2的目标是将这个能力**重新定向**到最终答案预测。

**核心变化**：

| | Stage 1 | Stage 2 |
|---|---------|---------|
| 目标 | 预测下一个推理步骤 $r_{t+1}$ | 预测最终答案 $y$ |
| Block数量 | 等于推理步骤数T | 固定K个（超参数） |
| 初期监督 | 每个block独立损失 | 每个block预测答案 |
| 后期权重 | 均匀→退火 | 线性递增（后期block权重更高） |
| 训练超参 | 标准 | 重置optimizer state + 更低学习率 + 更高dropout |

**Stage 2损失函数**：

$$L_{S2}(w) = -\sum_{k=1}^K \alpha_k \log p_w(y \mid x, m_{\leq k})$$

其中 $\alpha_k$ 是线性递增的权重：**越后面的block对最终答案的影响越大**。这鼓励模型将前期block用于编码和分析，后期block用于精炼和最终决策。

**为什么重置optimizer/降低学习率/增加dropout？**
- Stage 1的模型已经学会在记忆块中编码信息，但过度拟合了"预测推理步骤"这一辅助任务
- 重置optimizer state + 降低LR = 小心微调，不破坏已学到的编码能力
- 增加dropout = 鼓励更鲁棒的潜在表征

> **类比理解**：Stage 1是"学会了用草稿纸"，Stage 2是"只交答案卷"。你用同一本草稿纸（记忆块），但目标从"写出完整步骤"变成"算出最终答案"。教师不再检查中间步骤，只看最终结果——并且越到后来的草稿（后期block）越重要。

### 4.2 推断时的内存预算灵活性

RiM的一个关键优势：**推断时可以改变记忆块数量而不需重新训练**。

```
训练时：K=6 blocks
推断时：可以使用 K=1,2,3,...,8,... 个blocks
模型仍能正常工作，只是"思考深度"不同
```

**为什么会work？** Stage 2的训练让每个block预测**同一个最终答案**，而不是顺序依赖特定的block数量。模型学到的是"用任意数量的内存迭代精炼答案"，而非"恰好6步的程序"。

这一特性在实践中有重要意义：
- **简单问题**：K=2-3，更快
- **复杂问题**：K=8-10，更深思考
- **时延敏感**：K=1（退化到direct answer），最低延迟

### 4.3 注意力掩码变体

论文探索了两种记忆块内注意力模式：

- **因果（default）**：标准自回归mask，$m_i$ 可attend到 $m_{\le i}$（包括同block内的前序token）
- **双向（bidirectional）**：记忆块内所有token互相attend，更丰富的内部交互

双向变体可能提供更强的块内计算能力，但论文未报告具体的性能对比数字。

---

## Ch5. 实验结果与分析

### 5.1 实验设置

| 维度 | 配置 |
|------|------|
| 模型 | GPT-2 (124M), Llama-3.2-1B, Llama-3.2-3B |
| 训练数据 | GSM8K-Aug（~386K道小学数学题 + 推理trace） |
| 评估（ID） | GSM8K测试集 |
| 评估（OOD） | GSM-Hard（更难的变体） |
| 基线 | SFT w/o CoT, SFT w/ CoT, Coconut w/ Stage 0, DART |

### 5.2 核心性能结果

#### GSM8K（分布内）

| 方法 | Model | 准确率 (greedy) |
|------|-------|----------------|
| SFT w/o CoT | Llama-3.2-1B | 24.8% |
| Coconut w/ Stage 0 | Llama-3.2-1B | 35.6% |
| SFT w/ CoT | Llama-3.2-1B | ~44% (approximate, from paper) |
| **RiM (final block)** | Llama-3.2-1B | **43.1%** |
| **RiM (any block)** | Llama-3.2-1B | **48.8%** (best across all blocks) |

**关键观察**：
- RiM的最终block准确率（43.1%）匹配或接近显式CoT（~44%），但**TTFT快了~27倍**
- "any block"的高准确率表明模型在多个block都能正确回答——内在的ensemble效应

#### GSM-Hard（分布外泛化）

| 方法 | 模型范围 | 准确率范围 |
|------|---------|-----------|
| SFT w/o CoT | GPT-2, Llama-1B, Llama-3B | 3.5-8.5% |
| Coconut w/ Stage 0 | GPT-2, Llama-1B, Llama-3B | 7.1-10.2% |
| **RiM (final block)** | GPT-2, Llama-1B, Llama-3B | **7.8-12.0%** |

RiM在OOD上也持续超越所有基线，表明记忆块学到的是**可泛化的推理能力**，而非对GSM8K格式的死记硬背。

### 5.3 效率分析

这是RiM最引人注目的优势：

| 方法 | TTFT (ms) | 相对速度 | GSM8K准确率 |
|------|-----------|---------|-------------|
| SFT w/ CoT | ~200ms | 1× (基线) | ~44% |
| Coconut w/ Stage 0 | 53.4-188.8ms | ~3-4× faster | 35.6% |
| **RiM** | **7.6-27.9ms** | **~27× faster** | **43.1%** |
| SFT w/o CoT | 7.6-27.9ms | ~27× faster | 24.8% |

**RiM的TTFT = SFT w/o CoT的TTFT**。这意味着记忆块作为固定输入token几乎不增加任何推理开销——模型只需要做一次forward pass，就像回答一个不思考的直接问题一样快。

> **类比理解**：CoT像你一边想一边说（~27秒），Coconut像你一边想一边默默比划（~7秒），RiM像你盯着问题沉默2秒然后直接说出答案——但答案的正确率和边说边想几乎一样好。

### 5.4 内部表征分析

论文通过PCA和线性探针揭示了记忆块内部的计算动态：

**PCA轨迹分析**：
- 将不同记忆块的隐藏状态投影到PCA前两个主成分
- 发现：**表征沿平滑、系统性的轨迹演变**
- 不同问题的初始状态相似，但随着通过更多block，**轨迹显著分化** → 问题特定的潜在计算

**线性探针**：
- 从记忆块的隐藏状态训练一个简单线性分类器，预测最终答案是否正确
- 高准确率 → 记忆块的内部状态直接编码了"推理进展"和"答案置信度"

> 这些发现表明记忆块不仅仅是"额外的容量"，而是**结构化的计算原语**——每个block对问题进行一步操作，其状态自然地编码了"解决问题到了哪一步"。

### 5.5 消融：推断时内存预算

| 推断Block数 | GSM8K准确率（近似） |
|------------|-------------------|
| K=1 | ~34% (约) |
| K=3 | ~40% (约) |
| K=6 (训练设置) | 43.1% |
| K=8 | ~42% (约，未下降) |

增加block数带来适度的准确率提升，减少block数仍有合理性能。这说明模型学到了"迭代精炼"而非"恰好6步"。

---

## Ch6. 概念代码实现

> ⚠️ **非官方概念实现** — 基于论文Section 2-3描述和公式编写。  
> 目的：帮助理解RiM的核心训练流程和注意力掩码设计。  
> 不可直接用于训练——与官方实现的差异未知。  
> 官方代码暂未公开（截至2026年6月，NX-AI/JKU均未发布RiM仓库）。

### 6.1 记忆块嵌入初始化

```python
# ⚠️ 非官方概念实现 — 基于论文 Section 2.1 描述
import torch
import torch.nn as nn

def add_memory_tokens_to_tokenizer(tokenizer, num_m_tokens=4):
    """
    将记忆块特殊token添加到tokenizer。
    论文使用：<b>, <m>, </b> 三种特殊token。
    """
    special_tokens = {'additional_special_tokens': ['<b>', '</b>']}
    # 添加M个<m> tokens（论文未指定M的具体值，通常4-16）
    for i in range(num_m_tokens):
        special_tokens['additional_special_tokens'].append(f'<m_{i}>')
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def build_memory_block(tokenizer, num_m=4):
    """
    构建一个记忆块： [<b>, <m_0>, ..., <m_{M-1}>, </b>]
    返回token IDs。
    """
    tokens = ['<b>']
    for i in range(num_m):
        tokens.append(f'<m_{i}>')
    tokens.append('</b>')
    return tokenizer.convert_tokens_to_ids(tokens)
```

### 6.2 定制注意力掩码（Stage 1核心）

```python
# ⚠️ 非官方概念实现 — 基于论文 Section 2.2 和图2的注意力掩码设计

def build_stage1_attention_mask(
    q_len: int,        # 问题token数
    block_len: int,    # 每个记忆块token数（含<b>,</b>）
    num_blocks: int,   # 记忆块数量
    step_lens: list,   # 每个推理步骤的token数
):
    """
    构建Stage 1的注意力掩码。
    
    规则：
    1. 问题token可以attend到所有token
    2. 记忆块m_t可以attend到：问题 + m_{≤t}（不能看到已写推理步骤）
    3. 推理步骤的readout可以attend到：问题 + m_{≤t}
       （不能attend到任何其他推理步骤）
    4. 因果mask在记忆块内和推理步骤内各自保持
    """
    total_len = q_len + num_blocks * block_len + sum(step_lens)
    mask = torch.zeros(total_len, total_len)
    
    # Causal baseline（下三角=1表示可见）
    for i in range(total_len):
        mask[i, :i+1] = 1
    
    # 推理步骤之间互相屏蔽
    # 定位每个推理步骤的起始位置
    step_starts = []
    offset = q_len + num_blocks * block_len
    for sl in step_lens:
        step_starts.append(offset)
        offset += sl
    
    # 推理步骤t不能attend到推理步骤s（s≠t）
    for t in range(num_blocks):
        t_start = step_starts[t]
        t_end = t_start + step_lens[t]
        for s in range(num_blocks):
            if s != t:
                s_start = step_starts[s]
                s_end = s_start + step_lens[s]
                mask[t_start:t_end, s_start:s_end] = 0
    
    return mask
```

### 6.3 Stage 1训练循环

```python
# ⚠️ 非官方概念实现 — 基于论文公式 (1) 和训练描述

def stage1_loss(model, input_ids, attention_mask, 
                question_len, block_len, num_blocks, step_labels, step):
    """
    计算Stage 1的损失：每个记忆块后预测下一个推理步骤。
    
    参数：
    - step: 当前训练步数（用于退火计算）
    - step_labels: list of [token_ids_for_step_1, ..., token_ids_for_step_T]
    """
    # Forward pass（所有readout共享同一次forward）
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch, seq_len, vocab)
    
    total_loss = 0.0
    # 定位每个记忆块后的readout位置
    readout_start = question_len + num_blocks * block_len
    
    for t in range(num_blocks):
        step_start = readout_start + sum(len(sl) for sl in step_labels[:t])
        step_end = step_start + len(step_labels[t])
        
        # 获取该推理步骤位置的logits
        step_logits = logits[:, step_start:step_end, :]
        step_targets = step_labels[t].unsqueeze(0)
        
        # 交叉熵损失
        loss_t = nn.functional.cross_entropy(
            step_logits.reshape(-1, step_logits.size(-1)),
            step_targets.reshape(-1),
            ignore_index=-100
        )
        
        # 退火权重 λ_t(s) — 线性从1退火到0
        # 早期block先退火（t越大，λ保持得越久）
        total_steps = 10000  # 假设的总训练步数
        lambda_t = max(0.0, 1.0 - (step / total_steps) * (num_blocks / (num_blocks - t)))
        total_loss += lambda_t * loss_t
    
    return total_loss
```

### 6.4 Stage 2训练循环

```python
# ⚠️ 非官方概念实现 — 基于论文公式 (2) 和Stage 2描述

def stage2_loss(model, input_ids, attention_mask,
                question_len, block_len, num_blocks, answer_labels):
    """
    计算Stage 2的损失：每个记忆块后预测最终答案。
    
    权重α_k线性递增——越后期的block对最终答案影响越大。
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    total_loss = 0.0
    total_weight = 0.0
    
    for k in range(num_blocks):
        # 每个block后都有一个readout位置
        readout_pos = question_len + (k + 1) * block_len
        
        # 获取该readout位置的logits
        # 实际实现中，readout可能是单个token或多token生成
        readout_logits = logits[:, readout_pos:readout_pos + len(answer_labels), :]
        
        loss_k = nn.functional.cross_entropy(
            readout_logits.reshape(-1, readout_logits.size(-1)),
            answer_labels.unsqueeze(0).reshape(-1),
            ignore_index=-100
        )
        
        # 线性递增权重：α_k = (k+1) / sum(1..K)
        alpha_k = (k + 1) / (num_blocks * (num_blocks + 1) / 2)
        total_loss += alpha_k * loss_k
        total_weight += alpha_k
    
    return total_loss
```

### 6.5 推断时使用

```python
# ⚠️ 非官方概念实现 — 推断时构建输入，单次forward pass完成

@torch.no_grad()
def rim_inference(model, tokenizer, question: str, num_blocks: int = 6, num_m: int = 4):
    """
    RiM推断：将问题+记忆块拼接后，单次forward pass获取答案。
    
    推断时的num_blocks可以与训练时不同——论文验证了这一灵活性。
    """
    # Tokenize question
    q_ids = tokenizer.encode(question, add_special_tokens=True)
    
    # 构建记忆块
    block_ids = build_memory_block(tokenizer, num_m=num_m)
    
    # 拼接：Question + [block_1] + [block_2] + ... + [block_K]
    input_ids = q_ids + block_ids * num_blocks
    input_tensor = torch.tensor([input_ids])
    
    # 单次forward pass（这就是为什么TTFT ≈ SFT w/o CoT）
    outputs = model(input_tensor)
    
    # 从最后一个block后生成答案（或从任意block取readout）
    answer_start = len(input_ids)
    
    # 使用最后一个记忆块的信息预测答案
    # 实际实现中可能在最后block后附加一个答案readout
    generated = model.generate(
        input_tensor,
        max_new_tokens=128,
        do_sample=False,  # greedy
    )
    
    answer = tokenizer.decode(generated[0][len(input_ids):], skip_special_tokens=True)
    return answer
```

### 6.6 关键实现注意事项

1. **Embedding初始化**：新特殊token的embedding从随机初始化开始；所有现有词汇的embedding在整个训练中冻结
2. **位置编码**：记忆块的位置编码是固定的（与输入序列中的绝对位置对应），不是学出来的
3. **Block内注意力**：默认为因果（causal），论文也提到了双向（bidirectional）变体
4. **Readout方式**：论文中"readout"似乎是通过在记忆块后附加额外输出位置实现的，具体架构细节需参考官方实现

---

## Ch7. 深层解读与批判性分析

### 7.1 RiM与CoT的本质区别

| 维度 | CoT | RiM |
|------|-----|-----|
| 中间表示 | 自然语言token | 连续向量（通过特殊token的上下文表征） |
| 生成方式 | 自回归逐token | 单次forward pass |
| 信息带宽 | 受限于离散token的熵 | 每个特殊token可编码d_model维连续信息 |
| 计算效率 | O(T·L²) (T个forward pass) | O(L²) (1个forward pass) |
| 可解释性 | 高（人可读推理trace） | 低（隐空间向量） |
| 训练信号 | 强（逐token监督） | 弱（仅block后监督） |

> **类比理解**：CoT是"说出来的思考"（explicit thought），RiM是"脑内的思考"（implicit thought）。前者可以被老师检查每一步，后者只能通过最终答案评价质量。RiM的训练难点在于——你怎么教一个人在脑内"正确地思考"，而不是偷懒跳过步骤？

### 7.2 RiM的成功条件

论文的成功实验背后有几个微妙但关键的条件：

1. **密集监督 + 逐步退火**：如果直接用Stage 2（仅答案监督），模型会坍缩到"不思考直接猜答案"
2. **注意力隔离**：如果不屏蔽推理步骤，记忆块会退化——模型直接偷看已写步骤，不在记忆块中编码信息
3. **嵌入冻结**：只更新记忆token的embedding，保护预训练的语义空间
4. **足够的M（块内token数）**：内部维度d_model很大，但需要多个token来构造结构化的潜在表示

### 7.3 潜在局限与未解决问题

#### 1. 信息容量的硬上限

每个记忆块有M个token × d_model维的容量。对于需要非常长推理链的问题（如数学竞赛题），固定数量的记忆块可能不够。

**对策讨论**：论文展示了推断时可增加K（block数），但K的增加受KV cache增长的约束——每个额外block增加O(KL)的attention计算。对于极端复杂的推理，可能需要动态调整K或混合CoT。

#### 2. 数学推理之外的可迁移性

当前实验仅在GSM8K/GSM-Hard（小学数学）上验证。以下领域的效果未知：
- 代码生成（结构化输出 vs 答案预测）
- 多步逻辑推理（更高抽象层级）
- 开放式生成（没有"标准答案"的任务）

> **为什么可能work**：记忆块作为通用的"计算原语"，理论上可适应任何需要中间推理的任务。Stage 1可以适应任何有"推理trace"的数据格式。
> **为什么可能不work**：某些任务（如创意写作）的"推理"与"输出"不可分割——思考过程本身就是产品的一部分。

#### 3. 需要推理trace数据

Stage 1的训练依赖高质量的推理trace（$r_{1:T}$）。这意味着：
- 无法在无标注数据上训练
- trace的质量直接影响记忆块编码的质量
- 对于不同domain需要不同的trace数据

**缓解方案**：可以用强LLM（如GPT-4）蒸馏出合成推理trace，类似于现有CoT蒸馏工作。

#### 4. 无代码开源

截至2026年6月，论文未提供公开代码。作为Hochreiter组（xLSTM的作者）的产品，这不太寻常——他们通常会在NX-AI GitHub上发布代码。

可能原因：
- 论文仍在投稿/审稿中
- 实现与xLSTM内部代码base耦合，需要清理
- 计划发布更完善的版本

#### 5. 与xLSTM的潜在关联

值得注意的是，Hochreiter组此前的工作xLSTM（Extended LSTM）也强调了**内部记忆**和**并行化**。RiM的记忆块概念与xLSTM的matrix memory有哲学上的连续性——都是"在模型内部维护可操作的状态表示"。

可能的future direction：将RiM的记忆块与xLSTM的mLSTM结合，获得更强大的内部计算能力。

#### 6. 安全性与对齐

潜在推理（latent reasoning）的一个被低估的问题是**不可审计性**。CoT产生人可读的推理trace，可以被检查是否有害推理。RiM的记忆块是完全不透明的——你无法知道模型是否在"思考"有害内容。

这对AI safety有深远影响：在获得效率的同时，我们失去了对推理过程的可见性。

### 7.4 与其他潜在推理方法的比较

| 方法 | 年份 | 核心机制 | 效率 | 训练复杂度 |
|------|------|---------|------|-----------|
| Coconut | 2024 | 连续thought token + 自回归 | 中 | 中（需多阶段） |
| COCONUT-P | 2025 | 计划层+执行层分离 | 中 | 高 |
| DART | 2025 | 蒸馏CoT到潜在空间 | 高 | 低（单阶段蒸馏） |
| **RiM** | **2026** | **固定记忆块 + 并行处理** | **极高** | **中（两阶段）** |

RiM在效率维度上有明显优势，但训练需要推理trace数据（与DART类似，但DART是纯蒸馏）。

---

## Ch8. 影响与未来方向

### 8.1 对LLM推理研究的启示

1. **推理 ≠ 文本生成**：RiM干净地证明了推理的计算过程可以与文本生成解耦。这对于未来LLM架构设计有深远影响——是否需要将"推理引擎"和"语言引擎"显式分离？

2. **训练信号设计是核心**：RiM的成功很大程度上来自两阶段课程的设计，而非架构创新。这暗示在现有Transformer架构下，"教模型如何思考"比"给模型更好的思考工具"更重要。

3. **隐空间计算的可行性**：记忆块的PCA分析表明，Transformer可以在没有显式token生成的情况下进行有意义的内部计算。这为更多"无声推理"方法打开了大门。

### 8.2 实践意义

- **低延迟推理**：对实时应用（对话、游戏AI、自动驾驶）有直接价值
- **成本优化**：减少推理所需的GPU时间，尤其对大规模部署
- **API设计**：可能催生"thinking budget"参数——用户指定"思考深度"而非token数

### 8.3 论文中提出的未来工作

1. **RL for Stage 2**：用最终答案的奖励信号替代监督信号，可能发现更好的潜在计算策略
2. **混合RiM+CoT**：对简单子问题用RiM，对需要可解释性的关键步骤用CoT
3. **更复杂的推理域**：扩展到数学竞赛、代码调试、科学推理

### 8.4 我们的扩展思考

1. **记忆块的可解释性**：是否可以训练一个"翻译器"，将记忆块的内容解码为自然语言解释？这将结合RiM的效率和CoT的可解释性
2. **动态内存分配**：是否可以让模型自己决定需要多少记忆块？（类似于adaptive computation time）
3. **跨任务迁移**：在数学上训练的RiM记忆块是否能迁移到代码生成？
4. **与MoE的结合**：不同专家处理不同记忆块？这可以增加计算容量而不增加latency

---

## Ch9. 总结

### 9.1 论文核心公式速查

| 公式 | 用途 |
|------|------|
| $m_k = [\texttt{<b>}, \texttt{<m>}, ..., \texttt{<m>}, \texttt{</b>}]$ | 记忆块token序列 |
| $\mathcal{L}_{S1} = -\sum \lambda_t(s) \log p_w(r_{t+1} \mid x, m_{\leq t})$ | Stage 1损失（推理步骤监督+退火） |
| $\mathcal{L}_{S2} = -\sum \alpha_k \log p_w(y \mid x, m_{\leq k})$ | Stage 2损失（答案精炼+递增权重） |

### 9.2 一句话总结

RiM证明了LLM可以在**不生成任何中间文本**的情况下进行有效的多步推理——通过固定记忆块+两阶段课程学习，在GSM8K上达到CoT级别的准确率，同时将推理延迟降低了~27倍。

### 9.3 推荐阅读

- **Coconut** (Hao et al., 2024)：连续空间中的潜在推理 — RiM的"前身"
- **xLSTM** (Beck et al., 2024)：同组的扩展LSTM工作 — 理解Hochreiter组的工作记忆研究脉络
- **Let's Verify Step by Step** (Lightman et al., 2023)：过程监督 vs 结果监督 — 与RiM两阶段训练相关
- **Quiet-STaR** (Zelikman et al., 2024)：在推理时生成内部thought — 另一种"无声推理"思路
- **DART** (2025)：蒸馏CoT到潜在空间 — 与RiM的Stage 2类似但不同路径

### 9.4 延伸追踪

- **作者**：Lukas Aichberger ([LinkedIn](https://www.linkedin.com/in/lukas-aichberger/))、Sepp Hochreiter
- **机构**：[JKU ML Institute](https://www.jku.at/en/institute-for-machine-learning/)、[NXAI](https://www.nx-ai.com/)
- **arXiv页面**：[2605.30343](https://arxiv.org/abs/2605.30343)
- **alphaXiv讨论**：[alphaxiv.org/overview/2605.30343](https://www.alphaxiv.org/overview/2605.30343)
- **代码**：暂未公开 — 关注 [NX-AI GitHub](https://github.com/NX-AI) 获取更新
- **HuggingFace**：暂无相关模型 — 关注 [NX-AI](https://huggingface.co/NX-AI) 和 [JKU ICG](https://huggingface.co/JKU-ICG)

---

*报告生成日期：2026年6月8日*