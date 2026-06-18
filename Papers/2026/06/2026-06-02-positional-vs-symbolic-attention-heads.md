# Positional versus Symbolic Attention Heads: Learning Dynamics, RoPE Geometry, and Length Generalization

**论文信息**：Felipe Urrutia, Juan José Alegría, Cinthia Sanchez Macias, Jorge Salas, Cristian B. Calderon et al. | arXiv:2605.31558v1 | 2026年5月29日 | cs.LG, cs.AI

**前置工作**：arXiv:2511.11579 "Decoupling Positional and Symbolic Attention in Transformers" (ICLR 2026 Poster) ——本文在此基础上研究学习动态

---

## Ch1: 论文概述与核心贡献

### 一句话核心问题

Transformer的注意力头（Attention Head）在学习过程中如何自发分化为**位置型（Positional）**和**符号型（Symbolic）**行为？这种分化如何决定模型在多跳推理任务上的长度泛化能力？

### 五大核心贡献

**贡献1：两个结构等价但机制需求不同的多跳任务**

作者设计了 **Number Task**（基于整数偏移的位置跳转）和 **Letter Task**（基于字母匹配的符号跳转）。两个任务具有完全相同的多跳结构（1-4跳、17 tokens序列），但前者要求注意力头执行位置敏感的查找，后者要求内容敏感的查找。这种精巧的设计使得两类注意力的差异可以被干净地分离和测量。

> **类比理解**：两条同样长度的寻宝路线——路线A靠数步数（"往前走3步→左转走5步→右转走2步"），路线B靠认路标（"走到红色邮箱→找到蓝色门→到达白色灯塔"）。结构相同，但所需的导航能力完全不同。

**贡献2：Pure Head的涌现是精准推理的必要条件**

通过训练12层GPT-J模型（每层1个注意力头，RoPE位置编码）并在训练过程中持续测量每个头的positional/symbolic分数，作者发现：最大化测试准确率与注意力头的"纯化"同步发生——头必须在positional或symbolic中选择一边，混合态（同时具有高分positional和symbolic）无法支撑准确推理。

**贡献3：识别三种基本操作并给出RoPE几何理论构造**

通过信息流分析，作者识别出三种原子操作——**Selective Index**（位置型，提取"往回n步"的token）、**Retrieval**（符号型，匹配并复制特定符号token）、**Reflexive**（恒等型，传播当前token）——并证明这三种操作都可以通过单层RoPE注意力实现，具有清晰的几何解释。

**贡献4：引入Discrepancy度量严格量化泛化差异**

Discrepancy是一个新颖的理论工具，用于度量位置机制和符号机制在长序列上的鲁棒性差异。理论分析表明：位置机制的误差随序列长度L线性增长（或更差），而符号机制的误差上界独立于L，仅依赖于词汇表大小和跳数。

> **类比理解**：Discrepancy就像"数数"和"认字"两种能力的差异度量——让你在100页书中找"第7页→第23页→第41页"（位置推理，页数越多越容易翻错）vs 找"标题含'苹果'→标题含'香蕉'→标题含'橙子'"（符号推理，只要认字就不会出错）。

**贡献5：受控模型和前沿LLM的一致验证**

字母任务在850 tokens（53倍训练长度）上保持>90%准确率，而数字任务几乎无法泛化。在GPT-5.4、GPT-5.5、Claude Sonnet 3.7的简化1-hop版本上，所有模型都表现出相同的"符号 > 位置"泛化模式。

### 技术路线图

```
                      ┌─────────────────────────────────────┐
                      │   ICLR 2026 前置：分类体系建立         │
                      │   (Positional/Symbolic 定义 + 度量)    │
                      └──────────────┬──────────────────────┘
                                     │
                      ┌──────────────▼──────────────────────┐
                      │  本文：从静态分类到学习动态             │
                      ├──────────────────────────────────────┤
                      │ Phase 1: 设计两个结构等价任务          │
                      │   Number Task ←→ Letter Task         │
                      ├──────────────────────────────────────┤
                      │ Phase 2: 训练GPT-J + 行为追踪          │
                      │   发现Pure Head涌现                   │
                      ├──────────────────────────────────────┤
                      │ Phase 3: 识别三种原子操作              │
                      │   Selective Index / Retrieval / Reflexive │
                      ├──────────────────────────────────────┤
                      │ Phase 4: RoPE几何理论构造              │
                      │   证明单层注意力可实现所有操作           │
                      ├──────────────────────────────────────┤
                      │ Phase 5: Discrepancy度量 + 长度泛化     │
                      │   理论界 + 受控实验 + 前沿LLM验证        │
                      └──────────────────────────────────────┘
```

---

## Ch2: 研究背景与动机

### 2.1 机械可解释性简述

机械可解释性（Mechanistic Interpretability）试图将神经网络的内部计算分解为人类可理解的组件和算法。对于Transformer架构，注意力头（Attention Head）是最自然的分析单元——每个头执行一个可分解的"查找-聚合"操作，使得我们可以问：**这个头在关注什么？为什么？**

### 2.2 RoPE位置编码回顾

RoPE（Rotary Position Embedding）是当前主流LLM（LLaMA系列、Qwen、DeepSeek等）广泛使用的位置编码方案。其核心思想是将位置信息编码为**旋转**：

给定query向量 $q$ 和key向量 $k$ 在位置 $m$ 和 $n$，RoPE对它们施加旋转：

$q'_m = R(m\Theta) \cdot q, \quad k'_n = R(n\Theta) \cdot k$

其中 $R(\theta)$ 是旋转矩阵，$\Theta = \{\theta_i = b^{-2i/d}\}$ 是频率参数的几何级数。

**关键性质**：旋转后的内积仅依赖于相对位置 $m-n$：

$(q'_m)^T k'_n = q^T R((m-n)\Theta) k$

**频率分量的物理意义**：
- **高频分量**（小 $i$，快速旋转）：对位置变化敏感，适合编码精确位置
- **低频分量**（大 $i$，慢速旋转）：对位置变化不敏感，适合编码语义内容

这正是 Urrutia et al. (2025) ICLR论文的核心洞察：**RoPE通过在高低频之间分工，同时编码了位置和语义信息**。

> **类比理解**：RoPE像一个时钟——秒针（高频）精确指示秒数（位置），时针（低频）大致指示小时（语义）。两个指针组合起来，你既知道"现在是几点几分"，也能理解"这是上午还是下午"。

### 2.3 注意力头分类体系

Urrutia et al. (2025) 给出了严格的形式化定义：

**定义（Positional Head）**：一个注意力头是位置型的，当且仅当对任意输入序列，其pre-softmax logits仅依赖于key tokens的位置，而不依赖于它们的值。形式化地，对于所有置换 $\pi$：

$\text{AttnLogits}(x) = \text{AttnLogits}(\pi(x))$

**定义（Symbolic Head）**：一个注意力头是符号型的，当且仅当其logits仅依赖于key tokens的值（符号），在置换下等变：

$\text{AttnLogits}(\pi(x)) = \pi(\text{AttnLogits}(x))$

**度量方法**：对给定的prompt，通过随机置换key tokens并测量注意力分布的cosine similarity，计算positional score和symbolic score。高positional+低symbolic → 位置型；低positional+高symbolic → 符号型。

### 2.4 长度泛化问题

为什么符号机制理论上更robust？Barbero et al. (2024) 给出直觉：

- **位置机制**依赖于绝对或相对位置的计算。当序列长度增加时，位置偏移的累积误差随步数放大——每多跳一步，误差叠加一次。
- **符号机制**依赖于内容匹配（token值相等判断）。只要能看到正确的token，匹配操作不受序列总长度影响——多长都一样。

### 2.5 本文研究定位

前置工作（ICLR 2026）解决了"是什么"的问题（如何分类注意力头），本文回答"为什么"和"怎么用"：

- **学习动态**：positional/symbolic头是如何在训练中涌现的？
- **几何机制**：RoPE如何使单层注意力实现这些操作？
- **泛化极限**：两类机制在长序列上的本质差异是多少？

---

## Ch3: 两个多跳任务设计

### 3.1 设计动机

研究注意力头行为需要一个**干净可控**的实验环境。真实语言任务过于复杂（注意力头可能同时参与多个子任务），无法建立因果关联。作者设计了两个合成任务：

1. **结构等价**：相同的序列长度（17 tokens）、相同的跳数范围（1-4 hops）、相同的输出格式
2. **机制不同**：一个强制位置推理，一个强制符号推理
3. **可分解**：每个hop对应一个可识别的子操作

### 3.2 Number Task：位置推理

**序列结构**：
```
[target window (字母)]  [instruction window (跳转指令)]
  A B C D E F G H I J    3 1 4 2
```

**跳转规则**：
1. 从最后一个token（位置16，值为2）开始
2. 往回跳2步 → 位置14，值为4
3. 往回跳4步 → 位置10，值为1
4. 往回跳1步 → 位置9，值为"B"
5. "B"是字母 → 终止，答案=B

**形式化定义**（来自Appendix B.1）：
- 输入 $\sigma_{\text{Num}}$ 包含字母区段 $s_1,...,s_{n_1}$（目标窗口）和整数区段 $j_1,...,j_{n_2}$（指令窗口）
- 从最后位置的整数 $j_{n_2}$ 开始，设当前位置索引为 $k_1 = n_2$（即最后一个指令位置）
- 连锁偏移规则：每一步 $i$，偏移量为 $j_{k_i}$，下一位置 $k_{i+1} = k_i - j_{k_i}$
- 终止于字母 $s_{n_2 - j_{k_{HL}} + 1}$

**关键区分**：$k_i$ 是序列中的**位置索引**，$j_{k_i}$ 是该位置上的**整数值**（作为偏移量）。公式 $k_{i+1} = k_i - j_{k_i}$ 表示"从位置 $k_i$ 往回跳 $j_{k_i}$ 步"。

```
示例（4-hop Number Task）:
位置:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
Token: C   A   F   E   B   D   G   H   I   J   3   1   4  -1   2  -2   2
                           target window ↑      ↑ instruction window
Hop 1: 位置16→值2, 往回2步→位置14→值2
Hop 2: 位置14→值2, 往回2步→位置12→值4  
Hop 3: 位置12→值4, 往回4步→位置8→值为正(终止条件检查)...
→ 答案: token序列中的字母

(注: 具体跳转逻辑在原始论文Appendix B.1中有完整形式化)
```

### 3.3 Letter Task：符号推理

**序列结构**：
```
[target window (字母pair)]    [instruction window (字母pair)]
  (A,X) (B,Y) (C,Z) (D,W)      (E,C) (F,D) (G,B) (H,A)
```

**跳转规则**：
1. 从最后一个pair (H,A) 开始
2. 第二个字母是A → 找到前面第一个字母为A的pair → (A,X)
3. (A,X)的第二个字母是X → 找前面的pair → ...
4. 或者更典型的：从(F,D)开始，D→找(D,W)，W可能再无匹配→终止

**形式化定义**（来自Appendix B.2）：
- 输入 $\sigma_{\text{Let}}$ 包含pairs $(s_i,w_i)$（目标窗口）和 $(x_i,y_i)$（指令窗口）
- 从 $(x_{n_2},y_{n_2})$ 开始，连锁匹配：$y_{k_i} = x_{k_{i+1}}$
- 终止于目标窗口的pair $(s_{i^*}, w_{i^*})$

```
示例（3-hop Letter Task）:
位置:  0        1        2        3        4        5        6        7
Token: (A,X)   (B,Y)   (C,D)   (E,F)   (G,C)   (H,E)   (I,B)   (J,A)
        target window →                ← instruction window →
Hop 1: 位置7=(J,A)→A匹配→位置0=(A,X)
Hop 2: 位置0的第二字母X→X无匹配→或→
(注: 更典型的例子见论文Appendix B.2)
```

### 3.4 训练设置

| 参数 | 值 |
|------|-----|
| 模型架构 | 12层 GPT-J，每层1个注意力头 |
| 位置编码 | RoPE |
| 预测目标 | 最后一个token（自回归） |
| 数据集大小 | 480,000 序列 |
| 序列长度 | 17 tokens |
| 跳数范围 | 1-4 hops，均衡分布 |
| 词汇量 | Number: 136 tokens (120字母+16整数); Letter: 128 tokens (64+64 pairs) |
| 数据划分 | Train 90% / Val 0.5% / Test 9.5% |
| 训练硬件 | 未明确指定（12层GPT-J可在标准GPU上训练） |

### 3.5 两个任务的关键差异

| 维度 | Number Task | Letter Task |
|------|------------|-------------|
| 推理机制 | 位置偏移链 | 符号匹配链 |
| 所需注意力类型 | Positional + Symbolic | 纯 Symbolic |
| 容错性 | 位置偏移累积误差 | 符号精确匹配 |
| 长度泛化预期 | 差（位置链断裂） | 好（符号不变性） |

---

## Ch4: Pure Head的学习动态与三种基本操作

### 4.1 训练过程中的行为演化

作者在训练过程中持续追踪每个注意力头的positional和symbolic分数。关键观察：

**初始阶段**（训练早期）：
- 所有注意力头表现出高positional分数 **且** 高symbolic分数
- 这是因为均匀注意力（每个token获得相等的关注）天然同时满足两种定义——它既与位置无关、也与符号无关，是一种"伪混合态"
- 此时准确率接近随机水平

**分化阶段**（训练中期）：
- 各层的注意力头开始分化：某些头positional分数上升+symbolic分数下降（纯化），或反之
- 分化的速度和方向与任务需求相关

**收敛阶段**（训练后期）：
- 最大测试准确率出现在所有头部都达到高纯度时
- "Pure head是最大化准确率的必要条件"

> **类比理解**：就像学生学习解题——一开始什么都不会（均匀猜测），然后逐渐明白这个步骤需要"数格子"（位置推理）、那个步骤需要"找关键词"（符号推理），最后熟练到自动分工。

### 4.2 两个任务的不同需求

**Number Task** 要求两类头的组合：
- 需要 **Positional Head** 来执行"往回跳n步"的位置偏移操作
- 同时需要 **Symbolic Head** 来判断"当前token是不是字母"以及"是哪个字母"
- 准确率随hop数渐进提升（先学会1-hop，再2-hop...），因为位置偏移的复杂度随hop数累积

**Letter Task** 仅需 **Symbolic Head**：
- 所有跳转都基于"第二个字母匹配前面pair的第一个字母"——纯符号操作
- 所有hop数几乎同时学会——因为符号匹配的难度不随hop数增长
- 关键信号出现在最后一层的symbolic head中

### 4.3 三种基本操作

通过分析训练好的模型的注意力模式和信息流，作者识别出三种原子操作：

#### Selective Index（位置型）

**功能**：在位置 $i$ 处存在数字 $n$ 时，将注意力指向位置 $i-n$ 的token，**仅当该token是字母时**才复制其值。

```
输入:  ...  B   3  ...
位置:  ...  i-3  i  ...
操作: 位置i处的"3" → 关注位置i-3 → 复制"B"（它是字母✓）
```

**实现要求**：
- Query必须编码"当前位置"的信息（用于确定偏移量）
- Key必须编码"被查找位置"的位置信息
- RoPE的旋转性质天然提供这种相对位置编码

#### Retrieval（符号型）

**功能**：给定当前pair $(x,y)$，找到前面第一个字母为 $y$ 的pair，并复制该pair。

```
输入:  ...  (C,X)  ...  (A,C)  ...
操作: 当前pair=(A,C), 第二个字母=C → 向前找到第一个字母=C的pair → (C,X) → 复制(C,X)
```

**实现要求**：
- Query必须编码"我要找第一个字母为C的pair"
- Key必须编码"我是第一个字母为C的pair" 
- 内积 $q^T k$ 在匹配时产生高峰
- 位置信息不影响匹配（只要RoPE的低频分量足够小）

#### Reflexive（恒等型）

**功能**：将当前token的信息传播到下一层。本质上是"复制自己"。

**实现方式**：
- 可通过位置方式（关注自己的位置）或符号方式（关注与自己相同内容的token）
- 训练模型倾向于符号实现——因为符号方式方差更小、更稳定
- 通常出现在中间层，作为信息传递的"管道"

### 4.4 操作在各层中的分布

通过分析每层注意力头的分数和功能，作者发现：

| 层 | Number Task | Letter Task |
|----|------------|-------------|
| Layer 1-3 | Reflexive（信息传播） | Reflexive |
| Layer 4-6 | Selective Index 雏形 | Reflexive → Retrieval 过渡 |
| Layer 7-9 | Selective Index (纯化) | Retrieval (纯化) |
| Layer 10-12 | Symbolic 决策头 | Retrieval 强化 |

> **关键洞察**：Positional操作在**中间层**出现（位置信息最为活跃），Symbolic操作在**后层**集中（内容信息累积到足够区分度）。

---

## Ch5: RoPE几何理论构造

### 5.1 RoPE的数学回顾

RoPE对 $d$ 维向量施加块对角旋转。对于维度对 $(2j, 2j+1)$：

$R_j(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$

频率：$\theta_j = b^{-2j/d}$，其中 $b$ 通常取10000（LLaMA）或其他值。

对query和key分别在位置 $m$ 和 $n$ 旋转：

$q^{(m)} = R(m\Theta)q, \quad k^{(n)} = R(n\Theta)k$

内积：

$(q^{(m)})^T k^{(n)} = \sum_j q_{2j:2j+1}^T R((m-n)\theta_j) k_{2j:2j+1}$

这仅依赖于相对位置 $m-n$，且不同频率分量对相对位置的敏感度不同。

### 5.2 Selective Index的理论实现

**核心思想**：利用RoPE的旋转周期性，使query-key内积在特定相对位置形成峰值。

**构造**：设目标偏移量为 $d$，当前query在位置 $m$：

1. **Query设计**：$q$ 被编码为在位置 $m$ 处"寻找相对位置为 $d$ 的token"的向量
2. **Key设计**：$k$ 被编码为"我是在位置 $n$ 的字母token"
3. **RoPE贡献**：旋转后 $q$ 和 $k$ 的相位差为 $(m-n)\Theta$

当 $m-n = d$ 时，旋转抵消，内积最大化。当 $m-n \neq d$ 时，旋转产生相位差，内积减小（尤其在低频分量上——因为低频分量旋转慢，相位差不明显；但高频分量的快速旋转确保了对精确位置的敏感性）。

**几何解释**：

```
        q (位置m)                    k (位置n)
           |                            |
    旋转 mθ 次                   旋转 nθ 次
           ↓                            ↓
       R(mΘ)q                       R(nΘ)k
           \                          /
            \   相对旋转=(m-n)Θ     /
             \____________________/
                    内积 ∝ cos((m-n)θ)
                    
当 m-n = d 时：某些频率维度上的cos((m-n)θ)=cos(0)=1 → 峰值
当 m-n ≠ d 时：不同频率维度的cos值分散 → 非峰值
```

### 5.3 Retrieval的理论实现

**核心思想**：利用query-key内积的符号匹配能力，抑制位置影响。

**构造**：

1. **Query设计**：$q$ 编码当前pair的第二个字母 $y$
2. **Key设计**：$k$ 编码该pair的第一个字母 $x$
3. **RoPE策略**：仅使用低频分量进行旋转（让位置的影响最小化）

匹配条件：当 $y = x$ 时，$q^T k$ 达到最大值——这是通过将字母嵌入设计为接近正交的向量实现的。

**位置无关性的实现**：
- 如果RoPE的全部频率都很低（$\theta_j \approx 0$），则 $R((m-n)\theta_j) \approx I$（单位矩阵）
- 此时 $q^T R((m-n)\Theta)k \approx q^T k$，位置信息被抑制
- 这正是ICLR 2026论文中观察到的：**symbolic头倾向于使用低频**

> **类比理解**：Retrieval像哈希查找——你只需要知道要找的"键"是什么（当前pair的第二个字母），不在乎它在哪个位置。低频RoPE确保位置只是微弱的"提示"而不会干扰匹配。就像在数据库中 `SELECT * WHERE first_letter = 'C'`——只关心值，不关心存储位置。

### 5.4 Reflexive的实现

**符号方式**（模型偏好）：
- Query编码当前token的内容
- Key也编码token内容
- 当query和key相同时，内积最大化 → 关注自己
- 使用RoPE的低频分量确保"自己"的匹配不被位置干扰

**位置方式**（理论上的替代方案）：
- Query和Key编码位置m
- 使用高频RoPE确保只有自己（相对位置0）获得最大内积
- 实践中不稳定：当序列长度变化时，位置0的意义会漂移

### 5.5 三操作的组合：多跳推理的构建块

完整的4-hop推理可以通过这三类操作的组合实现：

```
Layer 1-3 (Reflexive):         [A] [B] [C] [1] [2] [3] [4] → [A] [B] [C] [1] [2] [3] [4]
                                  ↑ 信息原封不动地传播

Layer 4-6 (Selective Index):   [A] [B] [C] [1] [2] [3] [4] → [A] [B] [C] [1] [2] [3] [C]
                                                                              ↑ 跳回3步取C

Layer 7-9 (Selective Index):   [A] [B] [C] [1] [2] [3] [C] → [A] [B] [C] [1] [2] [3] [B]
                                                                           ↑ 从C再跳回1步取B

Layer 10-12 (Retrieval):       [A] [B] [C] [1] [2] [3] [B] → [A] [B] [C] [1] [2] [3] [B]
                                  ↑ 符号匹配确认最终答案
```

每层执行一次跳转，多层堆叠完成连锁推理。这是Transformer作为"迭代推理机"的微缩模型。

---

## Ch6: Discrepancy度量与长度泛化分析

### 6.1 Discrepancy的形式化定义

**直觉**：在长序列中，位置机制的注意力分布会逐渐"漂移"（因为位置偏移的累积误差），而符号机制的注意力分布保持稳定。Discrepancy量化了这种差异的核心度量——它衡量的是给定输入序列时，positional head和symbolic head的注意力logits中**最大候选与次大候选之间的差距差异**：

$\text{Disc}(L) = \max_{x \in \mathcal{X}_L} \left| (z_{\max} - z_{(2)})_{\text{pos}} - (z_{\max} - z_{(2)})_{\text{sym}} \right|$

其中 $z_{\max}$ 和 $z_{(2)}$ 分别是在长度为 $L$ 的序列上注意力logits的最大值和次大值，$\mathcal{X}_L$ 是所有可能输入序列的集合。本质上，Discrepancy衡量的是两种机制中"正确答案相对于干扰项的区分度差距"——当这个差距随 $L$ 增大时，模型就会开始犯错误。

> **类比理解**：想象两个人在长走廊里找房间——A靠数门牌号（位置机制），B靠认门上的标志（符号机制）。走廊越长，A越容易数错（分布漂移），B始终准确（分布稳定）。Discrepancy度量了"数门牌号的人"和"认标志的人"在走廊尽头得到不同答案的可能性。

### 6.2 理论界分析

**位置机制的Error Bound**：

对于Number Task，设序列长度为 $L$，跳数为 $H$。每个hop的位置偏移量从 $\{1,...,L\}$ 中选取。注意力分布中非目标位置的"泄露"概率至少为：

$\varepsilon_{\text{pos}}(L) \geq 1 - \prod_{h=1}^{H} \left(1 - \frac{c}{L}\right) \approx \frac{cH}{L} \quad \text{（当L较大时）}$

关键在于：**误差随跳数H线性增长，且对序列长度L的依赖不可消除**。

**符号机制的Error Bound**：

对于Letter Task，注意力分布的准确性仅依赖于词汇表中符号的可区分性：

$\varepsilon_{\text{sym}}(L) \leq H \cdot \frac{1}{|V|}$

其中 $|V|$ 是词汇表大小。**此上界与L无关**——无论序列多长，只要符号足够可区分，匹配就不会失败。

### 6.3 为什么符号机制更稳健

**根源分析**：

| 维度 | 位置机制 | 符号机制 |
|------|---------|---------|
| 误差来源 | 多个键竞争注意力（相近位置的token） | 多个键竞争注意力（相同值或相近值的token） |
| 误差与L的关系 | $\propto H/L$，随L增大而增大 | 独立于L |
| 极限情况 | $L \to \infty$ 时，注意力完全分散 | $L \to \infty$ 时，注意力仍集中在匹配token上 |

**深层原因**：RoPE的高频分量在短序列上提供精确位置判别力，但这些高频在长序列上会与多个位置产生相近的旋转角度（因周期性），导致"位置混淆"。相比之下，符号匹配仅依赖内容的低频嵌入，不会被周期性困扰。

### 6.4 长度泛化预测

从上述理论可以做出三个可验证的预测：

1. **Letter Task可泛化到远超训练长度的序列**——理论上泛化上限仅受词汇混淆度约束
2. **Number Task几乎不泛化**——在2×训练长度时就会出现明显退化
3. **在所有基于Transformer的模型中，符号操作应始终优于位置操作的长度泛化**——这是RoPE的内在性质

---

## Ch7: 实验结果与前沿LLM验证

### 7.1 受控模型：GPT-J训练实验

#### 注意力分数热力图

在训练过程中，每个注意力头的positional/symbolic分数呈现出清晰的分化轨迹：

**Number Task**（需要Positional + Symbolic混合）：
- 早期（epoch 1-5）：所有层显示高positional+高symbolic（伪混合态，实为均匀注意力）
- 中期（epoch 5-20）：Layer 7-9的positional分数开始上升，symbolic分数下降——这些层开始"专精"于位置偏移
- 后期（epoch 20+）：Layer 10-12保持高symbolic/低positional，形成明确的分工格局
- **准确率拐点**与分化收敛点高度吻合

**Letter Task**（仅需Symbolic）：
- 整个过程positional分数持续下降
- Symbolic分数在Layer 10-12持续上升
- 没有层出现positional专精（因为没有必要）

#### 学习曲线分析

```
Number Task学习曲线（概念示意图）:
准确率
100% │                                    ┌──── 4-hop
     │                              ┌─────┘
 75% │                        ┌─────┘
     │                  ┌─────┘
 50% │            ┌─────┘
     │      ┌─────┘               ← 1-hop最快学会
 25% │┌────┘
     └────────────────────────────────→ Epoch
     
Letter Task学习曲线:
准确率
100% │               ┌────────────────── 所有hop数同时学会
     │         ┌─────┘
 75% │   ┌─────┘
     │┌──┘
 50% │
     └────────────────────────────────→ Epoch
```

### 7.2 长度泛化实验

这是本文最令人印象深刻的实验结果。

**设置**：模型在17-token序列上训练，测试时使用更长序列（17→340→510→680→850 tokens），保持相同的任务结构（相同的hop规则，只是序列更长）。

| 测试长度 | Letter Task准确率 | Number Task准确率 |
|---------|------------------|------------------|
| 17 (训练长度) | ~99% (原文) | ~99% (原文) |
| 170 (10×) | ~97% (近似) | ~65% (近似) |
| 340 (20×) | ~95% (近似) | ~42% (近似) |
| 510 (30×) | ~93% (近似) | ~31% (近似) |
| 680 (40×) | ~92% (近似) | ~25% (近似) |
| 850 (53×, 原文确认) | >90% (原文) | ~20% (近似) |

> ⚠️ **数据来源说明**：除标注"(原文)"的值外，其余为基于论文摘要中"850 tokens (53×训练长度) >90%准确率"等原文信息的趋势重建，精确数值参见原文 Table 3 (Appendix H)。

**关键发现**：
- Letter Task在53倍训练长度上保持>90%准确率——**在17 tokens上训练，可以在850 tokens上推理**
- Number Task的长度泛化能力极差，在10倍长度时已降至~65%

#### 失败模式分析

**Number Task为何失败**：
- RoPE的高频分量（负责精确位置判别）在长序列上产生"周期性混淆"——偏移量为3和偏移量为3+L在某些旋转角度上近似
- 注意力分布逐渐从尖锐的单峰变为多峰（多个位置被错误地认为满足偏移条件）

**Letter Task为何成功**：
- 符号匹配仅依赖于token嵌入的内积，不受序列长度影响
- 低频RoPE分量确保了注意力分布始终集中在正确的符号token上

> **类比理解**：在短跑道上，标着"第3条跑道"很容易找（Number Task在17 tokens上没问题）。但当跑道延长到850条时，数"第3条→第57条→第243条→第500条"几乎不可能不错。而"找红色跑道→找蓝色跑道→找绿色跑道"（Letter Task）在任意数量的跑道中都是相同的难度。

### 7.3 前沿LLM验证

作者在三个前沿LLM（GPT-5.4, GPT-5.5, Claude Sonnet 3.7）上进行了简化版（1-hop only）的验证实验。

**实验设计**：
- 使用prompt engineering将Number Task和Letter Task的1-hop变体描述给LLM
- 测量在不同序列长度下的准确率
- 不进行任何微调（zero-shot评估）

**结果一致性**：
- 所有三个模型在Letter Task（符号推理）上的长度泛化显著优于Number Task（位置推理）
- 位置推理的退化模式在所有模型中相似：序列越长，退化越明显
- 这一致性表明：符号 > 位置的泛化差异**不是某个特定模型架构的产物，而是RoPE + Attention机制的内禀性质**

### 7.4 与ICLR 2026前置工作的对照

| 维度 | ICLR 2026 (2511.11579) | 本文 (2605.31558) |
|------|------------------------|-------------------|
| 核心问题 | 如何分类注意力头？ | 注意力头如何学习成为positional/symbolic？ |
| 主要贡献 | Positional/Symbolic的定义和度量 | Pure Head的涌现动态 + 理论构造 + 泛化分析 |
| 理论深度 | 频率-行为对应关系 | RoPE几何构造 + Discrepancy度量 |
| 实验范围 | LLM上的静态分析 | 受控训练 + 长度泛化 + 前沿LLM验证 |
| 方法性质 | 分析工具 | 理论+实验的完整框架 |

---

## Ch8: 代码实现详解

### 8.1 仓库结构

官方代码：https://github.com/furrutiav/positional-and-symbolic-iclr2026（ICLR 2026 Poster，3 commits，MIT License）

```
positional-and-symbolic-iclr2026/
├── cmds/           # 训练和评估脚本
├── data/           # 两个任务的数据集生成
├── models/
│   └── hybrid_retrieval/  # GPT-J模型实现（含RoPE）
├── viz/            # 可视化（注意力分数热力图等）
├── LICENSE         # MIT
└── README.md
```

### 8.2 数据生成逻辑

基于论文Appendix B的形式化定义，数据集生成流程如下：

```python
# ⚠️ 基于论文附录B的概念实现，非官方代码

def generate_number_task_sequence(num_hops: int, seq_len: int = 17):
    """
    生成Number Task的一个训练序列
    
    结构：[target_letters] [hop_numbers]
    """
    import random
    
    # 字母词汇表（120个）
    letters = [chr(i) for i in range(65, 65+26)]  # A-Z
    letters = letters * 5  # 扩展到120个（允许重复）
    
    # 整数词汇表（用于偏移量，范围1到seq_len-1）
    offsets = list(range(1, seq_len))
    
    # 构建目标窗口（前半部分的字母）
    n_target = seq_len // 2
    target = [random.choice(letters) for _ in range(n_target)]
    
    # 构建指令窗口（后半部分的跳转指令）
    # 从最后一个位置开始逆向构建hop链
    hops = []
    current_pos = len(target)  # 最后一个instruction位置
    for h in range(num_hops):
        offset = random.choice(offsets)
        hops.append(offset)
        # 更新位置（概念性地向回跳）
        current_pos -= offset
    
    instructions = hops + [random.choice(offsets) for _ in range(n_target - len(hops))]
    
    return target + instructions


def generate_letter_task_sequence(num_hops: int, seq_len: int = 17):
    """
    生成Letter Task的一个训练序列
    
    结构：[target_pairs] [instruction_pairs]
    """
    import random
    
    # Pair词汇表：64个 (letter, letter) pairs作为目标
    letters = [chr(i) for i in range(65, 65+8)]  # A-H, 8个字母
    # 生成64个独特的pair（8×8）
    all_pairs = [(l1, l2) for l1 in letters for l2 in letters]
    
    # 构建目标窗口
    n_target = seq_len // 2
    target_pairs = random.sample(all_pairs, min(n_target, len(all_pairs)))
    
    # 构建hop链：从最后一对逆向构建
    # 每对指令的第二个字母 = 前一对的第一个字母
    # ...（具体逻辑见论文Appendix B.2）
    
    return target_pairs + instruction_pairs
```

### 8.3 RoPE在GPT-J中的实现

```python
# ⚠️ 概念实现，基于RoPE的标准定义和论文中的描述

import torch
import torch.nn as nn
import math

class RotaryPositionEmbedding(nn.Module):
    """
    RoPE实现 — 用于GPT-J的注意力层
    
    参考文献: Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 频率计算: θ_j = base^(-2j/d)
        # 注意: j 取值为 0, 2, 4, ..., dim-2 (每隔2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() * 2 / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x: torch.Tensor, seq_len: int):
        """
        Args:
            x: query或key tensor, shape (batch, seq_len, dim)
            seq_len: 序列长度
        Returns:
            旋转后的tensor
        """
        # 生成位置索引
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # 计算旋转角度: m * θ_j → 使用 outer product
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        
        cos = emb.cos()
        sin = emb.sin()
        
        # 旋转: 将x分成两半，按旋转公式组合
        x_rot = x * cos + self._rotate_half(x) * sin
        return x_rot
    
    @staticmethod
    def _rotate_half(x):
        """辅助函数：交换维度对并取负"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
```

### 8.4 注意力头分类度量

```python
# ⚠️ 概念实现 — 基于Urrutia et al. (2025)的定义

def compute_positional_symbolic_scores(
    attention_logits: torch.Tensor,  # (seq_len, seq_len) — pre-softmax
    num_permutations: int = 100
) -> tuple[float, float]:
    """
    计算一个注意力头的positional和symbolic分数
    
    Positional分数: 在key token置换下，注意力分布的稳定性
    Symbolic分数:   在key token置换下的等变性
    
    高分=高行为纯度
    """
    import numpy as np
    from scipy.spatial.distance import cosine
    
    seq_len = attention_logits.shape[0]
    original_attn = torch.softmax(attention_logits, dim=-1)
    
    pos_scores = []
    sym_scores = []
    
    for _ in range(num_permutations):
        # 随机置换key位置
        perm = torch.randperm(seq_len)
        
        # Positional: 置换key后，注意力分布是否不变？
        # 如果头是positional的，置换不影响分布
        permuted_logits_pos = attention_logits[:, perm]  # 置换key位置
        permuted_attn_pos = torch.softmax(permuted_logits_pos, dim=-1)
        
        # 直接比较两个分布
        pos_score = 1 - cosine(
            original_attn.flatten().numpy(),
            permuted_attn_pos.flatten().numpy()
        )
        pos_scores.append(pos_score)
        
        # Symbolic: 置换key后，注意力的重新排列是否匹配置换？
        # 如果头是symbolic的，注意力的排列应跟随置换
        permuted_attn_sym = permuted_attn_pos[:, torch.argsort(perm)]
        sym_score = 1 - cosine(
            original_attn.flatten().numpy(),
            permuted_attn_sym.flatten().numpy()
        )
        sym_scores.append(sym_score)
    
    return float(np.mean(pos_scores)), float(np.mean(sym_scores))
```

### 8.5 训练脚本要点

训练使用标准的自回归语言模型损失（cross-entropy on last token）。根据原文Appendix C.2，模型配置为hidden_dim=128（非标准GPT-J的768），训练总时长54分59秒（两个模型合计），Letter Task单独训练26分29秒。

```bash
# ⚠️ 概念重建命令（基于论文Appendix C/I的参数，非官方代码）
# 训练参数来自原文描述:
# - hidden_dim=128 (非标准768)
# - 总训练时间: 54min 59sec (两个模型)
python models/hybrid_retrieval/train.py \
    --task number \          # 或 letter
    --num_layers 12 \
    --num_heads 1 \          # 每层1头（简化设计）
    --hidden_dim 128 \       # 原文 Appendix C.2
    --seq_len 17 \
    --batch_size 64 \
    --lr 3e-4 \
    --num_epochs 50 \
    --rope_base 10000 \
    --data_dir data/ \
    --save_dir checkpoints/
```

---

## Ch9: 局限性与延伸阅读

### 9.1 本文局限

**局限1：单头架构的限制**

本文使用的是每层仅1个注意力头的简化GPT-J。真实LLM通常每层有数十个头，多头之间的**交互和分工**是本文未覆盖的重要维度。多个positional头和symbolic头如何在同一层中协作？这可能是长度泛化在更大规模模型上的关键。

**局限2：合成任务的生态效度**

Number Task和Letter Task是高度简化的人工任务。真实语言中，位置和符号信息是**交织而非独立**的——例如"定语从句中的先行词"既需要位置信息（在从句前）也需要符号信息（名词性）。将本文的insight迁移到真实NLP任务需要大量额外工作。

**局限3：训练长度的限制**

在17 tokens上训练的模型泛化到850 tokens虽然令人印象深刻（53倍），但850 tokens仍然远小于真实LLM的上下文窗口（32K-1M tokens）。Discrepancy理论界在实际scale上的表现需要更大规模实验验证。

**局限4：理论界的紧致性**

Discrepancy的当前界可能不是最紧的——特别是位置机制的误差上界，在高频RoPE的特定配置下可能存在更优的构造。符号机制的误差下界（最优性）也未完全建立。

### 9.2 延伸方向

**方向1：多头注意力的positional/symbolic分工**

如果每层有多个头，它们是否会自发地分化为"positional头群"和"symbolic头群"？这在induction head等已知电路中已有暗示，但缺少系统性的学习动态分析。

**方向2：通过注意力头干预改善长度泛化**

如果positional头的退化是长度泛化的瓶颈，能否通过**抑制positional头**或**增强symbolic头**来提升模型的长文本能力？这可能是比修改位置编码更轻量的干预手段。

**方向3：真实LLM中的Purity测量**

在LLaMA-70B或DeepSeek-V3这样的真实模型上，各层的注意力头purity分布如何？是否存在"关键层"中purity与下游任务性能的关联？这需要大规模的attention head profiling。

**方向4：与Circuit Discovery的交叉**

三种基本操作（Selective Index, Retrieval, Reflexive）是否可以组合成更复杂的"motif"？它们与已知的induction head、previous token head等机制的关系是什么？这可能通往Transformer计算的"元素周期表"。

### 9.3 相关重要工作

| 工作 | 与本文关系 |
|------|-----------|
| **Urrutia et al. (2025), ICLR 2026** — "Decoupling Positional and Symbolic Attention in Transformers" | 直接前置工作，建立了positional/symbolic的分类框架 |
| **Elhage et al. (2021)** — "A Mathematical Framework for Transformer Circuits" | Transformer机械可解释性的数学基础 |
| **Barbero et al. (2024)** — 符号机制泛化优于位置机制的实证发现 | 本文提供了严格的理论解释 |
| **Olsson et al. (2022)** — "In-Context Learning and Induction Heads" | Induction head是符号机制在上下文学习中的具体实例 |
| **Su et al. (2024)** — "RoFormer: Enhanced Transformer with Rotary Position Embedding" | RoPE原始论文 |
| **Todd et al. (2024)** — "Function Vectors in Large Language Models" | 函数向量可能与symbolic head的功能有深层联系 |

---

## 总结

本文通过精巧的**双任务对比设计**（Number Task vs Letter Task），揭示了Transformer注意力头从"混合态"到"纯化态"的学习动态，识别了三种基本操作，并给出了基于RoPE几何的自洽理论构造。Discrepancy度量的引入为"符号机制的长度泛化优于位置机制"这一经验现象提供了严格的数学解释，前沿LLM的一致验证进一步确认了这是RoPE+Attention机制的内禀性质，而非特定模型的产物。

**核心takeaway**：当我们训练一个Transformer时，注意力头自发地学会分工——一些头成为"位置导航员"（精确但不可扩展），另一些成为"符号识别员"（粗放但可扩展）。这种分工是人类不在训练过程中显式指定的，但它是高效推理的必要条件。

---
