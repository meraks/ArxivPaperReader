# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

## 论文元数据
- **标题**：Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- **作者**：Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, Denny Zhou (Google Research, Brain Team)
- **arXiv ID**：[2201.11903](https://arxiv.org/abs/2201.11903)
- **发表**：NeurIPS 2022
- **提交日期**：2022-01-28
- **官方代码**：无（prompt 方法，论文附录提供完整示例）
- **引用量**：10,000+（截至 2025，AI 领域引用量最高的论文之一）

---

## 第1章：论文概述与核心贡献

### 1.1 一句话总结

Chain-of-Thought (CoT) Prompting 通过在 few-shot prompt 中加入中间自然语言推理步骤，让 ~100B+ 参数的大型语言模型在复杂推理任务上的能力显著提升——这是一种**模型规模的涌现属性**，不需要额外的训练数据或模型微调。

### 1.2 核心贡献

| # | 贡献 | 意义 |
|---|------|------|
| 1 | **提出 Chain-of-Thought Prompting 方法** | 在 few-shot 示例中加入 `<input, 推理链, output>` 三元组，让模型展示中间推理过程 |
| 2 | **发现推理能力的涌现性** | 推理能力的提升仅在 ~100B+ 参数规模的模型上出现，小模型 (<100B) 无增益甚至受损 |
| 3 | **覆盖多领域推理任务** | 在算术 (GSM8K: 17.9%→56.9%)、常识 (Sports: 80.5%→95.4%)、符号 (Last Letter: 0%→63%) 三类推理上均有效 |
| 4 | **消融验证自然语言中间步骤的不可替代性** | 纯方程、纯计算量（点号）、answer-after-CoT 均无法复现增益 |
| 5 | **验证鲁棒性** | 不同标注者、不同 prompt 风格、不同示例数量均稳定优于标准 prompting |

### 1.3 关键结果速览

| 任务 | 数据集 | Standard | CoT | 增益 |
|------|--------|:--------:|:---:|:----:|
| 算术推理 | GSM8K | 17.9% | **56.9%** | +39.0% |
| 算术推理 | MAWPS | 79.2% | **93.3%** | +14.1% |
| 常识推理 | Sports Understanding | 80.5% | **95.4%** | +14.9% |
| 常识推理 | StrategyQA | 68.6% | **77.8%** | +9.2% |
| 符号推理 | Last Letter (OOD) | 0.0% | **63.0%** | +63.0% |
| 符号推理 | Coin Flip (OOD) | 54.8% | **90.2%** | +35.4% |

---

## 第2章：研究背景与动机

### 2.1 论文发表时的技术状态（2022年1月）

2022 年初，LLM 领域正处于 GPT-3 (175B, 2020) 和 PaLM (540B, 2022) 之间的过渡期：

**标准 prompting 的局限**：Brown et al. (2020) 展示了 GPT-3 的 few-shot 学习能力，但在需要多步推理的任务上表现糟糕。模型可以流畅生成语言，却无法执行简单的算术推理。例如，GSM8K 基准上的最佳标准 prompting 仅有 17.9% 的准确率。

**已有推理方法的痛点**：
- **Finetuning 方法**（如 Cobbe et al. 2021 的 verifier 方法）需要大量标注数据（~7500 GSM8K 训练样本），成本高、泛化能力有限
- **Natural Language Rationales**（Ling et al. 2017）需要在特定数据集上 finetune
- **Program Synthesis** 需要执行环境（代码解释器等），且适用于特定问题类型

**可扩展性困境**：模型规模增长确实提升了许多 NLP 任务性能，但对于需要多步逻辑推理的任务，规模扩展并不直接转化为推理能力提升。这提示需要**方法的突破而非规模的扩展**。

### 2.2 CoT Prompting 的动机

论文的核心洞察来自两个方向的交叉：

1. **算术推理可以从自然语言中间步骤受益**（Cobbe et al. 2021 在 finetuning 场景下验证）
2. **LLM 通过 few-shot prompting 可以学习上下文模式**（Brown et al. 2020）

将两者结合：**在 few-shot prompt 中加入自然语言推理链**，让模型在生成答案前先展示推理过程。这个组合不需要 finetuning，不需要额外数据，仅需手动编写几个推理链示例。

### 2.3 与已有工作的关键区别

| 维度 | 已有工作 | CoT Prompting |
|------|---------|---------------|
| 训练要求 | 需要 finetuning 数据集 | 0 额外训练数据 |
| 模型兼容性 | 特定模型 finetune | 通用，适用于所有 LLM |
| 推理格式 | 输出答案 | 输出推理链 + 答案 |
| 适用任务 | 通常单领域 | 算术/常识/符号任务通用 |
| 可解释性 | 无中间步骤 | 自然语言推理过程 |

---

## 第3章：Chain-of-Thought Prompting 方法详解

### 3.1 方法定义

Chain-of-Thought Prompting 是 few-shot prompting 的一种扩展。标准的 few-shot prompt 提供 `⟨input, output⟩` 对；CoT prompting 提供 `⟨input, chain of thought, output⟩` 三元组。

**关键设计决策**：使用和标准 prompting 相同的输出格式前缀 `"A:"`，模型在推理时先自然生成中间的思考链，然后输出最终答案。这保证兼容性——同一模型可以同时处理标准 prompting 和 CoT prompting，只需改变提供示例的格式。

### 3.2 示例对比

**标准 Prompting：**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: The answer is 11.
```

**Chain-of-Thought Prompting：**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
```

### 3.3 四大关键属性

**① 分解能力 (Decomposition)**：CoT 让模型将多步问题分解为可管理的中间步骤。这对于需要多个运算步骤的问题（如 GSM8K）尤其重要——每一步的计算相对简单，但步骤数量决定了难度。

**② 可解释性 (Interpretability)**：推理链为模型行为提供窗口。可以检查模型"哪里错了"——是计算错误、步骤遗漏还是语义理解偏差。论文对 LaMDA 137B 的错误分析发现：50% 的错误推理链"几乎正确"（仅小误差），54% 存在语义错误。

**③ 广泛适用性 (Broad Applicability)**：因为 CoT 使用自然语言作为推理媒介，它天然适用于任何可以用语言描述推理过程的任务。论文验证了三个完全不同的推理领域：算术（数学题）、常识（需要世界知识）、符号（抽象规则操作）。

**④ 简便性 (Ease)**：只需在 few-shot prompt 中写出几个推理链示例即可。不需要特殊训练、不需要修改模型架构、不需要外部工具或执行环境。

### 3.4 推理链的本质分析

CoT 与以下已有概念的关键区别：

| 概念 | 与 CoT 的关系 | 区别 |
|------|--------------|------|
| **Prompting** (Brown 2020) | 基础框架 | CoT 在 prompt 中加入推理链 |
| **Natural Language Explanations** (Camburu 2018) | 类似概念 | CoT 使用 prior 而非 post-hoc 推理 |
| **Program Synthesis** | 都是中间步骤 | CoT 使用自然语言而非形式化代码 |
| **Scratchpad** | 并发工作 | Nye et al. (2021) 使用 finetuning，CoT 使用 prompting |

---

## 第4章：实验设计与结果分析

### 4.1 实验设计

**模型规模覆盖**：论文覆盖了三个模型族在多个参数量级上的表现：
- **GPT-3**: 350M, 1.3B, 6.7B, 175B (text-davinci-002)
- **PaLM**: 8B, 62B, 540B
- **LaMDA**: 422M, 2B, 8B, 68B, 137B
- **UL2**: 20B
- **Codex**: code-davinci-002

**Prompt 设计**：手动编写 8 个 CoT 示例用于算术推理（除 AQuA 用 4 个）。所有算术基准使用**同一套 8 个示例**——这很重要，它说明 CoT 学习的是推理模式而非特定数据集的答案分布。

**解码策略**：greedy decoding（论文发表后，Wang et al. 2022 的 Self-Consistency 证明了 majority voting 可以进一步大幅提升）。

### 4.2 算术推理结果

这个部分是最具冲击力的。

**GSM8K**（小学数字词问题，最难的算术基准）：
| 方法 | 准确率 |
|------|:------:|
| 标准 prompting (PaLM 540B) | ~18% |
| **CoT prompting (PaLM 540B)** | **~57%** |
| Finetuned GPT-3 + verifier (SOTA) | ~55% |
| Finetuned GPT-3 + verifier + 训练集 | ~55% |

**关键结论**：PaLM 540B 仅用 8 个 CoT 示例就超过了 finetuned GPT-3 的最佳系统——这是 zero-finetuning 对 finetuning 的重大胜利。

**涌现性**：这是论文最重要的发现之一。

| 模型规模 | ~8B | ~62B | ~175B/137B | ~540B |
|----------|:---:|:----:|:----------:|:-----:|
| GPT-3 标准 | ~5% | ~10% | ~18% | — |
| GPT-3 CoT | ~5% | ~15% | **~37%** | — |
| PaLM 标准 | ~4% | ~8% | — | ~18% |
| PaLM CoT | ~4% | ~12% | — | **~57%** |

对于 <100B 的模型，CoT 几乎没有效果甚至略微降低性能。从 62B→175B（GPT-3）和 62B→540B（PaLM）的跃迁中，CoT 的增益才**涌现**出来。这成为 LLM 涌现能力研究的一个标志性案例。

### 4.3 常识推理结果

CoT 的语言属性使其天然适用于常识推理。

| 数据集 | 任务描述 | Standard | CoT | 关键观察 |
|--------|---------|:--------:|:---:|---------|
| **CSQA** | 常识问答（5选1） | 78.1% | 79.9% | 选择范式限制了 CoT 的增益 |
| **StrategyQA** | 是/否策略推理 | 68.6% | **77.8%** | 需要多步推理的数据集获益最大 |
| **Date Understanding** | 自然语言日期计算 | 58.1% | **71.1%** | 强烈的多步计算需求 |
| **Sports Understanding** | 运动场景推理 | 80.5% | **95.4%** | 超过人类表现 (84%) |
| **SayCan** | 机器人指令排序 | 82.5% | 83.5% | 相对简单，天花板效应 |

**Sports Understanding 的超人表现**（95.4% vs 人类 84%）是一个令人惊讶的结果——模型通过 CoT 的逐步推理在此任务上超越了人类被试。但这也提示 CoT 在相对简单的任务上可能过度计算。

### 4.4 符号推理与长度泛化

最具启发性的部分：

**Last Letter Concatenation**（取姓名的最后一个字母并拼接）：
- 域内 (In-Domain, 2个词)：标准 prompting 约 40%，CoT **99%+**
- **域外泛化 (OOD, 4个词)**：标准 prompting **0%**，CoT **63%**
- **域外泛化 (OOD, 更长)**：CoT 仍表现出有意义的外推能力

**Coin Flip**（判断硬币是否朝上）：
- 域内：标准约 90%，CoT **99%+**
- **域外 (4步)**：标准 54.8%，CoT **90.2%**

**这些结果的关键意义**：CoT 本质上让模型**按步骤计算**，而非记忆训练集中出现的模式。当模型可以一步步操作符号时，它对未见过的更长序列也能泛化——这是单纯记忆无法做到的。

### 4.5 错误分析

论文对 LaMDA 137B 在 GSM8K 上的输出做了人工分析。50 个正确答案中，50/50 的推理链逻辑正确（仅 2 个偶然正确）。

50 个错误答案中：
- **46% "几乎正确"**：小误差导致答案错误
  - 计算错误（算术失误）
  - 符号映射错误（错误地选择了哪个数对应哪个变量）
  - 缺少一个中间步骤
- **54% 语义/一致性问题**
  - 推理链中的语义矛盾
  - 错误的推理方向
  - 完全无关的推理

在更大模型（PaLM 540B）上，许多"几乎正确"的错误被修正——说明**规模扩展主要修复了小误差类错误**，而语义类错误需要更大的模型或更好的 prompt。

---

## 第5章：消融实验与鲁棒性分析

### 5.1 消融实验设计

||GSM8K|MAWPS|Sports|Coin Flip|
|---|:---:|:---:|:---:|:---:|
|Standard|6.5%|43.2%|59.5%|49.0%|
|**CoT**|**14.3%**|**57.9%**|**85.8%**|**99.6%**|
|Equation only|5.4%|50.1%|–|–|
|Variable compute (dots)|6.4%|41.3%|61.6%|50.7%|
|CoT after answer|6.1%|43.6%|63.0%|50.2%|

### 5.2 核心消融发现

**方程 only**：去掉自然语言，仅用数学方程作为中间步骤。在简单问题上（MAWPS +6.9%）有帮助，但在复杂问题上（GSM8K -0.9%）无效。说明自然语言提供了方程无法捕捉的问题理解能力。

**Variable compute (dots)**：用与方程长度等量的"."替代推理链。这个条件控制"计算量"的变量——结果：**无改善**。这证明 CoT 不是通过"给模型更多计算步骤"起作用的，而是自然语言推理步骤的语义内容本身是关键。

**CoT after answer**：先生成答案（后接标准格式），再在答案后附上推理链。这个条件控制"推理格式的顺序"——结果：**无效**。这证明推理链必须在答案之前，顺序推理的过程是必需的。模型并不是通过预先知道答案再编推理链来工作的——它是真正通过推理得到答案。

### 5.3 鲁棒性分析

论文测试了多种 prompt 变体：

**不同标注者**：三位独立标注者编写 CoT 示例（A: 作者/论文版本；B: 简洁风格；C: 详细风格）。

**结果**：**所有风格的 CoT 都优于标准 prompting**。但有趣的是，B 和 C 在某些任务上优于 A（GSM8K 上 A 14.3% vs B 27.4% vs C 25.5%），这可能是由于过拟合实验者偏差的消除。

**不同示例数量**：从 2 个到 8 个示例，CoT 一致优于标准 prompting。更多示例 → 更好性能，但边际收益递减。

**不同示例顺序**：多次随机重排示例顺序，结果稳定。

**使用 GSM8K 训练集随机选示例**（非手工编写）：仍然优于标准 prompting。

### 5.4 外部验证器

与 Finetuned GPT-3 Verifier 组合（论文的 Figure 4 补充分析）：

论文还测试了在 CoT 生成的推理链之后，使用外部计算器验证算术步骤（post-hoc arithmetic）：
- GSM8K: 56.9% → 58.6% (+1.7%)
- 增益有限：说明 CoT 的主要价值在推理过程和方程构造，而非基本的算术运算精度

---

## 第6章：Prompt 代码与使用示例

### 6.1 GSM8K 完整 8-shot CoT Prompt

论文附录 A.1 提供了完整的 CoT prompt 示例。以下是其核心结构：

```
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then they planted some more. The total is 21. So the number they planted is 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars initially. Then 2 more arrive. So the total is 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have total?
A: Leah had 32 and her sister had 42. So they started with 32 + 42 = 74. They ate 35, so they have 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20. He now has 12. The number he gave to Denny is 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: He started with 5. He got 2 from mom and 2 from dad, so he got 4 more. So he has 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were 9 computers to start. Each day 5 more were added for 4 days. So they added 5 * 4 = 20. So total is 9 + 20 = 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58. He lost 23 on Tuesday, so he has 58 - 23 = 35. On Wednesday he lost 2 more, so 35 - 2 = 33. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She had 23. Each bagel costs 3. She bought 5 bagels, so 5 bagels cost 5 * 3 = 15. She has 23 - 15 = 8 left. The answer is 8.
```

### 6.2 实际推理时的模型输出

测试输入：
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
```

**期望的 CoT 输出**：
```
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
```

### 6.3 跨数据集统一 Prompt

论文的一个重要设计是：**8 个算术 CoT 示例在所有算术基准上统一使用**（GSM8K, SVAMP, ASDiv, MAWPS）。唯一例外是 AQuA（逻辑推理选择题，使用 4 个领域适配示例）。这显著增强了方法的说服力——CoT 学习的是通用的推理模式，而非特定数据集的特化。

### 6.4 常识推理 Prompt 示例

常识推理使用不同的 CoT 示例。以 Sports Understanding 为例：

```
Q: Is it true that the Super Bowl is a type of basketball championship game?
A: The Super Bowl is the annual championship game of the National Football League (NFL). Basketball has different championships like the NBA Finals. So the answer is no.

Q: Can a gorilla win a chess tournament against a professional chess player?
A: A gorilla is not capable of playing chess at a professional level because they lack the cognitive abilities for strategic board games. So the answer is no.

[4 examples total]
```

### 6.5 Python 调用示例

用 Python 调用 OpenAI API 执行 CoT Prompting 的参考实现：

```python
import openai

cot_prompt = """Q: There are 15 trees in the grove...
A: There are 15 trees originally...
...
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A:"""

response = openai.Completion.create(
    model="text-davinci-002",
    prompt=cot_prompt,
    max_tokens=256,
    temperature=0,          # greedy decoding (论文使用)
    stop=["\n\n"]
)

print(response.choices[0].text)
# Expected: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
```

### 6.6 推理成本分析

| 维度 | Standard Prompting | CoT Prompting |
|------|:------------------:|:-------------:|
| 每次推理 token 数 | ~50-100 | ~300-1000 |
| 推理时间 | 快 | 慢 3-10x |
| Prompt 构建成本 | 低（仅答案） | 中（需写推理链） |
| 可解释性 | 无 | 高 |

---

## 第7章：局限性、影响与延伸阅读

### 7.1 论文自身讨论的局限性

**计算成本**：CoT 生成推理链需要 3-10x 更多的生成 token，增加了推理延迟和计算开销。这个问题在后续工作中通过蒸馏（distillation）、更好解码策略等方式部分缓解。

**不可靠的推理链**：模型可能生成流畅但逻辑错误的推理链，同时仍输出正确答案。这降低了 CoT 作为可解释性工具时的可靠性。后续工作发现，Verifier 和 Self-Consistency 可以缓解此问题。

**有限的数据适用性**：对于其他领域（如代码生成、数学证明、逻辑规划），需要领域专家人工构建 CoT 示例，成本较高。

**涌现性的双刃剑**：小模型无法受益于 CoT，将其限制在 ~100B+ 模型的范围内。

### 7.2 方法层面的局限

**Prompt 敏感性**：虽然论文证明了鲁棒性，但后续工作（如 Kojima et al. 2022, Zhang et al. 2022）发现 CoT 对 prompt 的具体措辞、示例选择策略高度敏感。

**错误的传播**：多步推理中，早期步骤中的任何微小误差都会沿推理链传播和放大。这在长推理链（>5步）的问题上尤其突出。

**无法自我纠正**：模型不检查推理链中步骤间的一致性。如果第 2 步出错，第 3 步可能基于错误前提继续推理。

### 7.3 后续影响力与延伸工作

CoT Prompting 催生了整个 reasoning 研究范式。以下是直接衍生的关键工作：

**推理增强**
| 工作 | 年份 | 核心思想 |
|------|:----:|---------|
| **Self-Consistency** (Wang et al.) | 2022 | 多次采样 + majority voting |
| **Least-to-Most Prompting** (Zhou et al.) | 2022 | 将问题分解为子问题，依次解决 |
| **Tree-of-Thoughts** (Yao et al.) | 2023 | 推理链树状搜索 + 回溯 |
| **Graph-of-Thoughts** (Besta et al.) | 2023 | 推理步骤图结构，支持合并和分支 |
| **Chain-of-Thought with Self-Correction** | 2023+ | 模型自我验证和纠正推理链 |
| **Chain-of-Thought with Program Execution** | 2023 | CoT + 代码解释器的混合方法 |
| **KV Cache + CoT** | 2024 | 长推理链的缓存优化 |
| **o1 / o3 系列推理模型** | 2024-2025 | 将 CoT 作为推理计算的核心机制 |

**零样本 CoT**
- **"Let's Think Step by Step"** (Kojima et al. 2022)：不需要 few-shot 示例，只需在 prompt 末尾添加 "Let's think step by step"，零样本下即可触发 CoT 推理。虽不如 few-shot 版本，但提供了从 few-shot 到 zero-shot 的桥接。

**CoT 分析与理论**
- Wang et al. 2023 对 CoT 的理论分析（CoT 本质上等同于一种"概率推理链"）
- Feng et al. 2024 从计算复杂度角度证明 CoT 扩展了 Transformer 的表达能力（可计算更复杂的问题）
- Merrill & Sabharwal 2023 证明 CoT 可以将 Transformer 的"模拟计算"能力从恒定深度扩展到可变深度

### 7.4 对后续 LLM 推理研究的启示

**推理计算量分配**：CoT 开创了"让模型在推理时花更多 token 思考"的范式。o1 系列模型将这一概念推向极致——在内部执行数万 token 的隐式推理链。

**涌现性作为模型能力的边界**：CoT 的涌现性特征提示我们，某些能力是规模的闸门（scale threshold）——低于阈值则完全不存在，高于阈值则自然显现。这影响了后续对 Scaling Law 的讨论。

**Prompt Engineering 作为科学研究**：CoT 证明精心设计的 prompt 可以达到甚至超过 finetuning 的效果。这在当时是革命性的——它开辟了 prompt engineering 作为一个严肃的研究子领域。

**可解释 AI 的新方向**：CoT 的自然语言推理链成为 LLM 可解释性的主要范式。

### 7.5 代表性延伸阅读

| 论文 | 链接 |
|------|:----:|
| Self-Consistency Improves CoT Reasoning | [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) |
| Large Language Models are Zero-Shot Reasoners | [arXiv:2205.11916](https://arxiv.org/abs/2205.11916) |
| Tree of Thoughts: Deliberate Problem Solving | [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) |
| Least-to-Most Prompting Enables Complex Reasoning | [arXiv:2205.10625](https://arxiv.org/abs/2205.10625) |
| Automatic Chain of Thought Prompting | [arXiv:2210.03493](https://arxiv.org/abs/2210.03493) |
| Towards Reasoning in Large Language Models: A Survey | [arXiv:2212.10403](https://arxiv.org/abs/2212.10403) |

---

> *"No language models were finetuned in the process of writing this paper."* — 论文的引言幽默结尾，准确捕捉了 CoT 方法的优雅和简洁
