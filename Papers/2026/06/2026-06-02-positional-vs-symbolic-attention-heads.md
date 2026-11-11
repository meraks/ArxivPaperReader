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

Discrepancy是一个新颖的理论工具，用于度量位置机制和符号机制在长序列上的鲁棒性差异。理论分析（Theorem 3.2）给出：Index 的 Discrepancy 上界为 $\min\{1-\cos\theta,\ 2\pi^2/n^2\}$（随长度 $n$ 增大而收紧），Retrieval 的上界为 $1-\cos(\omega-n\theta)$（$\theta=0$ 即 NoPE 时与长度无关）；论文据此论证符号机制对长序列更稳健。

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
  C A F E B D G H I J    3 1 4 1 2 3 2
```

**跳转规则**：
1. 从最后一个token（位置16，值为2）开始
2. 往回跳2步 → 位置14，值为2
3. 往回跳2步 → 位置12，值为4
4. 往回跳4步 → 位置8，值为"I"
5. "I"是字母 → 终止，答案=I

**形式化定义**（来自Appendix B.1）：
- 输入 $\sigma_{\text{Num}}$ 包含字母区段 $s_1,...,s_{n_1}$（目标窗口）和整数区段 $j_1,...,j_{n_2}$（指令窗口）
- 从最后位置的整数 $j_{n_2}$ 开始，设当前位置索引为 $k_1 = n_2$（即最后一个指令位置）
- 连锁偏移规则：每一步 $i$，偏移量为 $j_{k_i}$，下一位置 $k_{i+1} = k_i - j_{k_i}$
- 终止于字母 $s_{n_2 - j_{k_{HL}} + 1}$

**关键区分**：$k_i$ 是**指令窗口（整数子序列 $j_1,...,j_{n_2}$）中的索引**，完整序列中的位置为 $n_1+k_i$；$j_{k_i}$ 是该位置上的**整数值**（作为偏移量）。公式 $k_{i+1} = k_i - j_{k_i}$ 表示"在整数子序列中往回跳 $j_{k_i}$ 步"。

```
示例（3-hop Number Task）:
位置:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
Token: C   A   F   E   B   D   G   H   I   J   3   1   4   1   2   3   2
                           target window ↑      ↑ instruction window
Hop 1: 位置16→值2, 往回2步→位置14→值2
Hop 2: 位置14→值2, 往回2步→位置12→值4
Hop 3: 位置12→值4, 往回4步→位置8→字母I（终止）
→ 答案: I

(注: 上例按 Figure 1 的直观跳转方式（在当前/落点 token 上直接回跳）；Appendix B.1 的形式化用 $k_i$ 索引整数子序列。)
```

### 3.3 Letter Task：符号推理

**序列结构**：
```
[target window (字母-数字pair)]    [instruction window (字母-字母pair)]
  (A,3) (B,1) (C,2) (D,4)           (F,A) (E,C) (D,E) (G,D)
```

**跳转规则**：
1. 从最后一个pair (G,D) 开始
2. 第二个字母是D → 向前找到第一个字母为D的pair → (D,E)
3. (D,E)的第二个字母是E → 向前找到第一个字母为E的pair → (E,C)
4. (E,C)的第二个字母是C → 向前找到第一个字母为C的pair → 目标窗口的 (C,2)（字母-数字pair）→ 终止

**形式化定义**（来自Appendix B.2）：
- 输入 $\sigma_{\text{Let}}$ 包含pairs $(s_i,w_i)$（目标窗口）和 $(x_i,y_i)$（指令窗口）
- 从 $(x_{n_2},y_{n_2})$ 开始，连锁匹配：$y_{k_i} = x_{k_{i+1}}$
- 终止于目标窗口的pair $(s_{i^*}, w_{i^*})$

```
示例（3-hop Letter Task）:
位置:  0        1        2        3        4        5        6        7
Token: (A,3)   (B,1)   (C,2)   (D,4)   (F,A)   (E,C)   (D,E)   (G,D)
        target window →                ← instruction window →
Hop 1: 位置7=(G,D)→D匹配→位置6=(D,E)
Hop 2: 位置6=(D,E)→E匹配→位置5=(E,C)
Hop 3: 位置5=(E,C)→C匹配→位置2=(C,2)（字母-数字pair）→终止，答案=(C,2)
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
| 训练硬件 | AMD EPYC 7282 16核 + 3× NVIDIA RTX A5000 24GB（Appendix I） |

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
- 各层的注意力头开始分化：某些头保持高positional并使symbolic分数下降（纯化为位置型），或反之（纯化为符号型）
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
- 训练模型经验上选择符号实现——论文仅指出这是由 positional-symbolic 分数显示的非平凡经验观察，并未给出方差/稳定性解释
- 通常出现在 Index/Retrieval 之后的后续层，作为信息传递的"管道"

### 4.4 操作在各层中的分布

通过信息流分析，作者发现两个任务分别由以下操作序列构成：

| 阶段 | Number Task | Letter Task |
|------|------------|-------------|
| Index/Retrieval 阶段 | Selective Index（最多 h 次；示例中位于 Layer 0、1、3） | Retrieval（最多 h 次） |
| Reflexive 阶段 | Reflexive（传播已检索的 token） | Reflexive（传播已检索的 token） |

> **关键洞察**：论文的信息流分析给出的是每个任务内部的操作序列，而非论文未逐层验证的固定层号表：Number Task 为 $h$ 次 Selective Index 后接 Reflexive 传播，Letter Task 为 $h$ 次 Retrieval 后接 Reflexive 传播。Retrieval 属于 Letter Task，不出现在 Number Task 的层表中。

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
3. **RoPE贡献**：定理构造使用二维、**单一频率** $\theta$ 的 RoPE（所有 key 取同一向量 $v$，query 按当前偏移旋转），使 logit 形如 $\cos((m-n-d)\theta)$。

当 $m-n = d$ 时，$\cos(0)=1$，内积最大化；当 $m-n \neq d$ 时，内积下降。一般多头 RoPE 才有多个频率分量，本文的可行性构造不使用多频率分散。

**几何解释**：

```
        q (位置m)                    k (位置n)
           |                            |
    旋转 mθ 次                   旋转 nθ 次
           ↓                            ↓
       R(mΘ)q                       R(nΘ)k
           \                          /
            \   相对旋转=(m-n-d)θ   /
             \____________________/
                    内积 ∝ cos((m-n-d)θ)
                    
当 m-n = d 时：cos(0)=1 → 峰值
当 m-n ≠ d 时：内积的 cos 值下降 → 非峰值
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
Layer 0-3 (Selective Index):    [A] [B] [C] [1] [2] [3] [4] → [A] [B] [C] [1] [2] [3] [C]
                                                                              ↑ 跳回3步取C
                                → [A] [B] [C] [1] [2] [3] [B]
                                                                           ↑ 从C再跳回1步取B

Layer 4+  (Reflexive):          [A] [B] [C] [1] [2] [3] [B] → [A] [B] [C] [1] [2] [3] [B]
                                  ↑ 已检索到的答案在后继层中传播
```

每层可执行一次跳转，前 h 层完成连锁索引，之后各层以 Reflexive 传播结果。这是Transformer作为"迭代推理机"的微缩模型。

---

## Ch6: Discrepancy度量与长度泛化分析

### 6.1 Discrepancy的形式化定义

**直觉**：对单个注意力头，Discrepancy 衡量其 logits 能否把正确 token 与干扰项分开；当它随输入长度增大而变小时，模型就会开始犯错误。论文将注意力头的 Discrepancy 定义为：

$\Delta(L) = \max_{x \in \mathcal{X}_L} \left( z_{\max}(x) - z_{(2)}(x) \right)$

其中 $z_{\max}$ 和 $z_{(2)}$ 分别是在长度为 $L$ 的输入上该头注意力logits的最大值和次大值，$\mathcal{X}_L$ 是所有长度为 $L$ 的输入集合。Theorem 3.2 分别给出 Index 与 Retrieval 构造的 Discrepancy 上界，从而量化两种机制对长序列的鲁棒性差异。

> **类比理解**：想象两个人在长走廊里找房间——A靠数门牌号（位置机制），B靠认门上的标志（符号机制）。走廊越长，A越容易数错（分布漂移），B始终准确（分布稳定）。Discrepancy度量的是单个头在走廊尽头还能否把正确房间与干扰房间分开——对位置机制，走廊越长这个差距越小。

### 6.2 理论界分析

**位置机制（Index）的界**：

Theorem 3.2 给出，对长度 $n>1$ 的输入，Index 构造的 Discrepancy 满足

$\Delta_{\text{Index}}(n) \leq \min\left\{1-\cos\theta,\ \frac{2\pi^2}{n^2}\right\}$

其中 $\theta$ 是构造中 RoPE 使用的单一角度。随着序列长度 $n$ 增大，$2\pi^2/n^2$ 项收紧，Discrepancy 被迫变小——正确 token 与次大 logit 的差距变小，模型因此更容易出错。注意该界不含跳数 $H$ 项。

**符号机制（Retrieval）的界**：

对 Retrieval，令 $\omega = 2\pi/|\text{ALPH}|$ 且 $\theta < \omega/(2n_0)$，则

$\Delta_{\text{Retrieval}}(n) \leq 1-\cos(\omega-n\theta)$

在 NoPE 特殊情形 $\theta=0$ 下，上界退化为常数 $1-\cos\omega$，与 $n$ 无关——只要符号可区分，长序列下匹配不会额外退化。（$\theta>0$ 时界仍含 $n\theta$。）

### 6.3 为什么符号机制更稳健

**根源分析**：

| 维度 | 位置机制 | 符号机制 |
|------|---------|---------|
| 误差来源 | 多个键竞争注意力（相近位置的token） | 多个键竞争注意力（相同值或相近值的token） |
| Discrepancy 与L的关系 | $\min\{1-\cos\theta, 2\pi^2/L^2\}$，随 L 增大而收紧 | $1-\cos(\omega-L\theta)$；$\theta=0$ 时与 L 无关 |
| 极限情况 | $L \to \infty$ 时，注意力完全分散 | $L \to \infty$ 时，注意力仍集中在匹配token上 |

**深层原因**：RoPE的高频分量在短序列上提供精确位置判别力，但这些高频在长序列上会与多个位置产生相近的旋转角度（因周期性），导致"位置混淆"。相比之下，符号匹配仅依赖内容的低频嵌入，不会被周期性困扰。

### 6.4 长度泛化预测

从上述理论可以做出三个可验证的预测：

1. **Letter Task可泛化到远超训练长度的序列**——理论上泛化上限仅受词汇混淆度约束
2. **Number Task几乎不泛化**——在2×训练长度时就会出现明显退化
3. **在受控单头模型与所测前沿 LLM 中，符号操作的长度泛化优于位置操作**——论文将其作为 RoPE 机制的强证据，但未证明适用于所有 Transformer/RoPE 模型（单头设置与有限词表是明确局限）

---

## Ch7: 实验结果与前沿LLM验证

### 7.1 受控模型：GPT-J训练实验

#### 注意力分数热力图

在训练过程中，每个注意力头的positional/symbolic分数呈现出清晰的分化轨迹：

**Number Task**（需要Positional + Symbolic混合）：
- 训练开始时所有层显示高positional+高symbolic（伪混合态，实为均匀注意力；Figure 2B 中表现为高熵）
- 随训练步数增加（Figure 2B/6 横轴为 training steps，约 0-6000），各层分数开始分化：部分头纯化为 positional，部分纯化为 symbolic
- 论文报告准确率收敛时两类头都达到纯化；Figure 2B 只展示 Layer 1、5、11，Appendix E/Figure 6 给出所有层
- **准确率拐点**与分化收敛点高度吻合

**Letter Task**（仅需Symbolic）：
- 训练中 positional 分数下降；在较深层（如 Layer 11）symbolic 分数上升并纯化
- 最后一层（Layer 11）出现每个 hop 条件都依赖的关键 symbolic head
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
     └────────────────────────────────→ Training steps
     
Letter Task学习曲线:
准确率
100% │               ┌────────────────── 所有hop数同时学会
     │         ┌─────┘
 75% │   ┌─────┘
     │┌──┘
 50% │
     └────────────────────────────────→ Training steps
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

> ⚠️ **数据来源说明**：除标注"(原文)"的值外，其余为基于论文中"850 tokens (53×训练长度) >90%准确率"等原文信息的趋势重建；受控模型的长度泛化曲线见 Figure 5B（测试长度延伸到 5000）。Table 3 是前沿 LLM 的 1-hop 实验（长度 12-300），不含 17→850 的受控模型数值。

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
- 不进行微调，但每题 prompt 含自然语言说明与 few-shot 示例（并非 zero-shot）

**结果一致性**：
- 所有三个模型在Letter Task（符号推理）上的长度泛化显著优于Number Task（位置推理）
- 位置推理的退化模式在所有模型中相似：序列越长，退化越明显
- 这一致性表明：符号 > 位置的泛化差异在受控模型与所测前沿 LLM 上一致出现；论文据此认为它反映了 RoPE 机制的普遍倾向，但受限于单头设置与有限词表，尚不能断言适用于所有 Transformer/RoPE 模型

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

本文实验代码：论文脚注给出 `number-and-letter-neurips26-C754`；下文结构参考前置工作仓库 https://github.com/furrutiav/positional-and-symbolic-iclr2026（ICLR 2026 Poster）。

```
positional-and-symbolic-iclr2026/
├── cmds/           # 训练和评估脚本
├── data/           # 两个任务的数据集生成
├── models/
│   └── hybrid_retrieval/  # GPT-J模型实现（含RoPE）
├── viz/            # 可视化（注意力分数热力图等）
├── LICENSE
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
    
    # 字母词汇表（120个）：26 个单字母 + 94 个 bigram（aa…dp）
    letters = [chr(i) for i in range(ord('a'), ord('z')+1)]  # 26 个单字母 a-z
    letters += [chr(a)+chr(b) for a in range(ord('a'), ord('e'))
                for b in range(ord('a'), ord('z')+1)][:94]  # 94 个 bigram（aa…dp）
    
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
    
    # Pair词汇表：目标为 64 个 (letter, integer) pairs；指令为 64 个 (letter, letter) pairs
    letters = [chr(i) for i in range(65, 65+8)]  # A-H, 8个字母
    numbers = list(range(1, 9))  # INT = {1,...,8}
    target_vocab = [(l, n) for l in letters for n in numbers]       # ALPH × INT
    instr_vocab = [(l1, l2) for l1 in letters for l2 in letters]    # ALPH × ALPH
    
    # 构建目标窗口
    n_target = seq_len // 2
    target_pairs = random.sample(target_vocab, min(n_target, len(target_vocab)))
    
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

训练使用标准的自回归语言模型损失（cross-entropy on last token）。根据原文 Appendix C.2，模型配置为 hidden_dim=128（非标准 GPT-J 的 768）；训练时长见 Appendix I（Table 4）：总时长 54 分 59 秒（两个模型合计），Letter Task 单独训练 26 分 29 秒。

```bash
# ⚠️ 概念重建命令（基于论文Appendix C/I的参数，非官方代码）
# 原文给出的配置（Appendix C.2）:
# - hidden_dim=128 (非标准768)
# - 训练时长: 54min 59sec (两个模型, Appendix I)
# - batch_size/lr/num_epochs/rope_base 原文未报告，以下为示例默认值
python models/hybrid_retrieval/train.py \
    --task number \          # 或 letter
    --num_layers 12 \
    --num_heads 1 \          # 每层1头（简化设计）
    --hidden_dim 128 \       # 原文 Appendix C.2
    --seq_len 17 \
    --batch_size 64 \        # 假设默认值（原文未报告）
    --lr 3e-4 \              # 假设默认值（原文未报告）
    --num_epochs 50 \        # 假设默认值（原文未报告）
    --rope_base 10000 \      # 假设默认值（原文未报告）
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

本文通过精巧的**双任务对比设计**（Number Task vs Letter Task），揭示了Transformer注意力头从"混合态"到"纯化态"的学习动态，识别了三种基本操作，并给出了基于RoPE几何的自洽理论构造。Discrepancy度量的引入为"符号机制的长度泛化优于位置机制"这一经验现象提供了严格的数学解释，前沿LLM的一致验证进一步支持这反映了RoPE+Attention机制的普遍倾向，而非特定模型的产物；但论文的单头设置与有限词表仍是明确局限。

**核心takeaway**：当我们训练一个Transformer时，注意力头自发地学会分工——一些头成为"位置导航员"（精确但不可扩展），另一些成为"符号识别员"（粗放但可扩展）。这种分工是人类不在训练过程中显式指定的，但它是高效推理的必要条件。

---
