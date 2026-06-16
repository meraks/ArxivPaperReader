# Emergent Abilities of Large Language Models — 论文研读报告

## 论文元数据

| 项目 | 内容 |
|------|------|
| 标题 | Emergent Abilities of Large Language Models |
| 作者 | Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, William Fedus |
| 机构 | Google Research, Stanford University, UNC Chapel Hill, DeepMind |
| arXiv | 2206.07682 |
| 发表 | Transactions on Machine Learning Research (08/2022) |
| 类型 | Survey / 综述（无官方代码仓库） |

---

## 第1章　概述与论文定位

Scaling laws 表明，放大语言模型（训练算力、参数量）可在下游任务上**可预测地**提升性能与样本效率；其中 cross-entropy loss 的 scaling curve 经实证跨越超过 7 个数量级（Kaplan et al., 2020; Hoffmann et al., 2022）。但部分下游任务的表现**并不**随规模连续改善，且无法在事前预测（Ganguli et al., 2022）。

核心问题由此提出：某些能力在小模型上完全不存在，却在模型放大到某一临界点后**突然出现**。论文将这类能力定义为 *emergent abilities*——不可通过外推小模型的性能改进来预测的能力。

论文定位为系统性综述，沿四条主线展开：

1. **定义与判据**（§2）：给出可操作化的涌现定义。
2. **证据汇总**（§3 few-shot prompting、§4 augmented prompting）：梳理前人工作中的涌现实例与各自涌现门槛。
3. **解释与争议**（§5）：讨论评估指标、cross-entropy 行为与机制推测。
4. **降低门槛与未来方向**（§5.2、§5.6）：超越单纯缩放的途径。

与两项 scaling 工作的关系：Kaplan et al. (2020) 提供了"loss 随规模平滑可预测"的基础；Hoffmann et al. (2022) 的 Chinchilla 指出此前工作**低估**了训练数据量，主张 compute-optimal 训练——Chinchilla 参数量仅为 Gopher 的四分之一，却使用相近的训练算力。这表明"规模"是多变量复合体，涌现不应被归因于单一维度。

---

## 第2章　涌现能力的定义与判据

### 2.1 定义

论文采用 adapted 自 Steinhardt (2022)、根植于诺贝尔奖物理学家 Philip Anderson 1972 年文集 "More Is Different" 的一般性表述：

> Emergence is when quantitative changes in a system result in qualitative changes in behavior.

聚焦到模型规模（训练算力与参数量）上，给出可操作化判据：

> An ability is emergent if it is not present in smaller models but is present in larger models.

其等价推论：涌现能力**无法**通过对小模型 scaling law（一致的性能提升）的外推来直接预测。

### 2.2 phase transition 模式

在 scaling curve（x 轴：模型规模，y 轴：性能）上，涌现呈现清晰模式——性能在达到某个**临界阈值**前接近随机，越过阈值后急剧攀升至远高于随机水平。这种行为的整体剧变即 *phase transition*（Huberman & Hogg, 1987），无法通过考察更小规模的系统来预见。

### 2.3 x 轴的选择

当今语言模型主要沿三因素缩放：算力、参数量、训练数据量（Kaplan et al., 2020; Hoffmann et al., 2022）。论文的处理：

- **正文**以训练 FLOPs 为 x 轴（沿用 Hoffmann et al., 2022 的度量）。
- **附录 D** 给出以参数量为 x 轴的对应图（Figure 11、12，正文 Figure 4、10）。
- **§5.3** 进一步以 WikiText103 perplexity 为 x 轴分析（Merity et al., 2016）。

由于大多数 dense Transformer 模型族的训练算力与参数量大致按比例同步缩放，FLOPs 与参数量两条曲线形状相似（Kaplan et al., 2020）。训练数据量同样是重要因素，但许多模型族对所有规模使用固定训练样本数（Brown et al., 2020; Rae et al., 2021; Chowdhery et al., 2022），故未将其作为独立 x 轴。

### 2.4 涌现门槛不是固定属性

某一能力首次被观测到的规模取决于多重因素，并非该能力的固有属性。例如，用更高质量数据训练的模型可能在更少算力或更少参数下即出现涌现；反之，涌现也取决于不被数据量、数据质量或参数量所限制。当前语言模型大概率未被最优训练（Hoffmann et al., 2022）。论文目标不是宣称某一具体规模为涌现的必要条件，而是汇总前人工作中涌现行为的实例。

---

## 第3章　Few-Shot Prompting 中的涌现能力

### 3.1 范式

Few-shot prompting 由 GPT-3（Brown et al., 2020）推广：给预训练模型一个 prompt（自然语言指令 + 若干 input-output 示例作为前导），让其在**不进行任何进一步训练或参数梯度更新**的情况下完成未见样本（Figure 1）。

> 注：prompting 任务设置早于 GPT-3（Trinh & Le, 2018; McCann et al., 2018; Radford et al., 2019; Raffel et al., 2020）。

当模型在某任务上"随机水平持续到某一规模，之后跃升至远高于随机"时，该 few-shot 能力即为涌现。Figure 2 展示了横跨**五个语言模型族**（LaMDA、GPT-3、Gopher、Chinchilla、PaLM）的八个涌现实例。

### 3.2 八个案例（Figure 2）

**（A）Modular arithmetic**——3-digit 加减法 + 2-digit 乘法（BIG-Bench, 2022，2-shot）。GPT-3 与 LaMDA 在数个数量级的算力上接近零性能，之后分别在 $2.3\times10^{22}$ FLOPs（13B 参数，GPT-3）与 $10^{23}$ FLOPs（68B 参数，LaMDA）处急剧跃升。

**（B）IPA transliterate**——国际音标转写（BIG-Bench）。与算术在相近规模出现类似涌现。

**（C）Word unscramble**——从打乱字母恢复原词（BIG-Bench）。

**（D）Persian QA**——波斯语问答（BIG-Bench）。多语言涌现同时依赖模型规模与训练数据：PaLM 需兼具其训练数据集**并**缩放到 62B 参数才能在该任务上表现高于随机。

**（E）TruthfulQA**——衡量如实回答能力的对抗性基准（Lin et al., 2021），专门针对 GPT-3 策展，GPT-3 即使放到最大也**不**超过随机。小 Gopher 同样不超过随机，直到放大到最大模型 $5.0\times10^{23}$ FLOPs（280B 参数），性能跃升至高出随机 **20% 以上**（Rae et al., 2021）。

**（F）Grounded conceptual mappings**——将概念域（如基本方位）映射到文本网格世界（Patel & Pavlick, 2022）。仅在使用最大 GPT-3 模型（$3.1\times10^{23}$ FLOPs，175B 参数）时才跃升高于随机。

**（G）MMLU**——57 项测试、覆盖数学/历史/法律等的大规模多任务理解基准（Hendrycks et al., 2021a）。对 GPT-3、Gopher、Chinchilla，约 $10^{22}$ FLOPs（约 10B 参数）及以下在所有主题平均上不优于猜测；放大到 $3$–$5\times10^{23}$ FLOPs（70B–280B 参数）才显著超过随机。这意味着在不带检索/外部记忆的 dense 模型上，解决跨大量主题的知识性问答可能需要越过该阈值。

**（H）Word in Context (WiC)**——语义理解基准（Pilehvar & Camacho-Collados, 2019）。这是论文反复援引的关键案例：

- GPT-3 与 Chinchilla 即使放到各自最大规模（约 $5\times10^{23}$ FLOPs）也**无法**达到优于随机的 one-shot 表现。
- Brown et al. (2020) 曾将此负面结果归因于 GPT-3 的自回归语言建模目标（而非去噪目标）或架构，建议训练同等规模的 bidirectional 模型作为补救。
- 但后续工作表明，**继续放大** decoder-only 模型即已足够：将 PaLM 从 $3\times10^{23}$ FLOPs（62B 参数）放大到 $2.5\times10^{24}$ FLOPs（540B 参数），性能显著跃升——无需 Brown et al. 所建议的重大架构改动。

> 补充：PaLM 540B 的 $2.5\times10^{24}$ FLOPs 约为 GPT-3 算力预算的 **8 倍**（附录 C Table 2）。
> 脚注：GPT-3 在 dev set 上用 few-shot（而非 one-shot）可达约 55% 的略高于随机表现，但该表现并非规模所致，且未在 test set server 上成立。

### 3.3 案例小结

下表汇总 Figure 2 八案例涉及的模型族与涌现门槛（数值取自 Table 1 与正文）：

| 案例 | 基准 | 涌现模型 | 涌现门槛 (FLOPs) | 参数量 |
|------|------|----------|------------------|--------|
| A | 算术 (3-digit) | GPT-3 | 2.3E+22 | 13B |
| B | IPA 转写 | LaMDA | ~10²³ | ~68B |
| C | 单词重组 | LaMDA | 见附录 E | — |
| D | 波斯语 QA | PaLM | 需 62B + 数据 | 62B |
| E | TruthfulQA | Gopher | 5.0E+23 | 280B |
| F | 概念映射 | GPT-3 | 3.1E+23 | 175B |
| G | MMLU (57 主题) | GPT-3 | 3.1E+23 | 175B |
| H | WiC | PaLM | 2.5E+24 | 540B |

WiC 案例的核心启示：当某能力在现有最大模型上仍失败时，**进一步缩放**仍可能是解锁该能力的有效路径，而非必然需要架构革新。

---

## 第4章　增强提示策略中的涌现

### 4.1 定义

除 few-shot prompting 外，近期工作提出多种 prompting / finetuning 策略来增强模型能力。若某项技术在与"不使用该技术"的基线相比时，在应用到足够大的模型之前**无改进甚至有害**，则该技术本身也被视为一种涌现能力。

### 4.2 四个案例（Figure 3）

**（A）Chain-of-Thought（CoT）多步推理**——引导模型在给出最终答案前产出一串中间步骤（Wei et al., 2022b）。如 Figure 3A，CoT 仅在缩放到 $10^{23}$ FLOPs（约 100B 参数）时才超过不含中间步骤的标准 prompting（Table 1 精确值：LaMDA 数学应用题 $1.3\times10^{23}$ FLOPs，68B 参数）。在答案**之后**附加解释的增强 few-shot 也有类似涌现（Lampinen et al., 2022）。

**（B）Instruction following（指令微调）**——在以指令表述的任务混合体上微调，使模型能仅凭指令（无 few-shot 示例）执行未见任务（Ouyang et al., 2022; Wei et al., 2022a; Sanh et al., 2022; Chung et al., 2022）。如 Figure 3B，Wei et al. (2022a) 发现该技术对 $7\times10^{21}$ FLOPs（8B 参数）及更小的模型**反而有害**，仅在放大到 $1.3\times10^{23}$ FLOPs（68B 参数，FLAN）时才改善性能。

**（C）Scratchpad（程序执行）**——对涉及多步的计算任务（如大数相加、执行程序），微调模型预测中间输出（Nye et al., 2021）。如 Figure 3C，在 8-digit 加法上，scratchpad 仅对约 $8.9\times10^{19}$ FLOPs（40M 参数）及更大的模型有益。

**（D）True/False 校准**——衡量模型能否预测自己哪些问题能答对。Kadavath et al. (2022) 比较两种校准方式：模型先给出答案再评估"P(True)"的 True/False 法，与比较正确答案相对其他选项概率的标准法。如 Figure 3D，True/False 法的优越性仅在最大规模 $2.6\times10^{23}$ FLOPs（52B 参数，Anthropic 模型）时才涌现。

### 4.3 Table 1 完整清单

Table 1 列出论文汇总的全部涌现能力及各自涌现门槛，共 **23 行**（few-shot prompting 10 项、augmented prompting 13 项；其中 "Many BIG-Bench tasks" 为汇总项）。

**Few-shot prompting abilities**

| 涌现能力 | Train. FLOPs | Params | 模型 | 参考 |
|----------|-------------|--------|------|------|
| Addition/subtraction (3 digit) | 2.3E+22 | 13B | GPT-3 | Brown et al. (2020) |
| Addition/subtraction (4-5 digit) | 3.1E+23 | 175B | GPT-3 | Brown et al. (2020) |
| MMLU (57 topic avg.) | 3.1E+23 | 175B | GPT-3 | Hendrycks et al. (2021a) |
| Toxicity classification (CivilComments) | 1.3E+22 | 7.1B | Gopher | Rae et al. (2021) |
| Truthfulness (TruthfulQA) | 5.0E+23 | 280B | Gopher | Rae et al. (2021) |
| MMLU (26 topics) | 5.0E+23 | 280B | Gopher | Rae et al. (2021) |
| Grounded conceptual mappings | 3.1E+23 | 175B | GPT-3 | Patel & Pavlick (2022) |
| MMLU (30 topics) | 5.0E+23 | 70B | Chinchilla | Hoffmann et al. (2022) |
| Word in Context (WiC) | 2.5E+24 | 540B | PaLM | Chowdhery et al. (2022) |
| Many BIG-Bench tasks | Many | Many | Many | BIG-Bench (2022) |

**Augmented prompting abilities**

| 涌现能力 | Train. FLOPs | Params | 模型 | 参考 |
|----------|-------------|--------|------|------|
| Instruction following (finetuning) | 1.3E+23 | 68B | FLAN | Wei et al. (2022a) |
| Scratchpad: 8-digit addition | 8.9E+19 | 40M | LaMDA | Nye et al. (2021) |
| Open-book knowledge for fact checking | 1.3E+22 | 7.1B | Gopher | Rae et al. (2021) |
| Chain-of-thought: Math word problems | 1.3E+23 | 68B | LaMDA | Wei et al. (2022b) |
| Chain-of-thought: StrategyQA | 2.9E+23 | 62B | PaLM | Chowdhery et al. (2022) |
| Differentiable search index | 3.3E+22 | 11B | T5 | Tay et al. (2022b) |
| Self-consistency decoding | 1.3E+23 | 68B | LaMDA | Wang et al. (2022b) |
| Leveraging explanations in prompting | 5.0E+23 | 280B | Gopher | Lampinen et al. (2022) |
| Least-to-most prompting | 3.1E+23 | 175B | GPT-3 | Zhou et al. (2022) |
| Zero-shot chain-of-thought reasoning | 3.1E+23 | 175B | GPT-3 | Kojima et al. (2022) |
| Calibration via P(True) | 2.6E+23 | 52B | Anthropic | Kadavath et al. (2022) |
| Multilingual chain-of-thought reasoning | 2.9E+23 | 62B | PaLM | Shi et al. (2022) |
| Ask me anything prompting | 1.4E+22 | 6B | EleutherAI | Arora et al. (2022) |

门槛横跨极大区间：从 LaMDA scratchpad 的 $8.9\times10^{19}$ FLOPs（40M 参数），到 PaLM WiC 的 $2.5\times10^{24}$ FLOPs（540B 参数），跨越约 5 个数量级。

---

## 第5章　涌现的解释与争议

### 5.1 评估指标争议

涌现虽实例众多，但缺乏令人信服的解释。论文首先审视评估指标（BIG-Bench, 2022）：

- 对长序列目标使用 **exact string match**，可能把"持续累积的渐进改进"伪装成涌现。
- 多步/算术推理若只对最终答案计分、不给部分正确解任何分数，同理。
- 但这只能解释最终答案准确率的跃升，**不能**解释中间步骤质量为何突然升至高于随机；且涌现同样出现在许多分类任务上（如 Figure 2D–H），故"不给部分分"至多是**不完整**的解释。

### 5.2 Cross-entropy 分析（附录 A）

论文对六个 LaMDA 涌现 BIG-Bench 任务测量 cross-entropy loss（预训练 scaling law 所用指标）：三个生成式任务（modified arithmetic、IPA transliterate、word unscramble，指标 EM/BLEU）与三个分类任务（logical arguments、sports understanding、figure of speech，指标 accuracy）。

关键发现：即使下游指标（EM/BLEU/accuracy）接近随机且不改善，**cross-entropy loss 仍在持续改善**——即对数似然的改进被下游指标掩盖。六项任务全部落入此情形（论文称为 Outcome 2）。在涌现点处，cross-entropy loss 也出现 "elbow"（拐点）。但这并未解释下游指标为何涌现，也无法预测涌现发生的规模。

对多选题的进一步分析：正确响应与错误响应的对数概率都随规模下降（因更大模型产生更不极端的概率），但二者最终在某一规模处分叉，对应任务性能的大幅上升。

### 5.3 部分分指标分析（附录 A.2）

对三个生成式任务用 BIG-Bench 提供的全部指标（BLEU、ROUGE、BLEURT 等会给部分分）重新评估：涌现行为**独立于**所选评估指标。故"用 exact match 而非给部分分的指标"**不是**生成式任务涌现的完整解释。word unscramble 与 repeat copy logic 因仅 exact match 为合理指标而排除。

### 5.4 机制性推测

对特定任务有自然直觉：若多步推理需 $l$ 步顺序计算，可能要求模型深度至少为 $O(l)$ 层；更多参数与训练带来更好的记忆，有利于需要世界知识的任务（如 closed-book QA 可能需要足够参数来容纳压缩后的知识库）。但这些仅为推测，尚无完整理论。

### 5.5 多变量视角（§5.3）

规模不必是唯一视角。以 WikiText103 perplexity 为 x 轴看 MMLU 涌现，曲线与用 FLOPs/参数量时相似——因对所考虑的 Gopher/Chinchilla，perplexity 与算力高度相关。但该相关性未来未必成立（如检索增强模型可能以更少算力/参数获得强 perplexity）。总体而言，涌现应被视为**多个相关变量的函数**。

---

## 第6章　降低涌现门槛的途径

### 6.1 规模并非唯一因素

某能力虽在某一规模涌现，但日后可能在更小规模上实现——模型规模不是解锁涌现的单一因素。随训练科学发展，新架构、更高质量数据或改进训练流程可使小模型也获得该能力。

### 6.2 PaLM 62B 反例

有 **14 个** BIG-Bench 任务：LaMDA 137B 与 GPT-3 175B 表现接近随机，而参数与算力都更少的 PaLM 62B 却达到高于随机的表现（附录 F 列出全部 14 项：anachronisms、ascii word recognition、conceptual combinations、cryptonite、disambiguationqa、emojimovie、goalstepwikihow、grereadingcomprehension、linguisticspuzzles、logicgridpuzzle、metaphor boolean、metaphor understanding、odd one out、parsinlu qa）。虽无逐一消融，但 PaLM 的潜在优势包括更高质量训练数据（如比 LaMDA 更多的多语言与代码数据）与架构差异（如 split digit-encodings）。

### 6.3 预训练目标与继续预训练

另一解锁途径是不同预训练目标：Tay et al. (2022c) 表明，在 mixture-of-denoisers 目标（Tay et al., 2022a）上进行计算高效的**继续预训练**阶段，可在若干 BIG-Bench 任务上启用涌现表现。

### 6.4 把已发现能力下放到更小模型

一旦某能力被发现，后续研究可使其适用于更小模型：

- **Sanh et al. (2022)**：用 encoder-decoder 架构在 **11B** 模型上诱导出指令跟随行为（encoder-decoder 微调后通常优于 decoder-only；Wang et al., 2022a），尽管 Wei et al. (2022a) 最初发现指令微调仅对 68B 及以上的 decoder-only 模型有效。
- **InstructGPT / Ouyang et al. (2022)**：用 finetuning + RLHF（人类反馈强化学习），使 **1.3B** 模型在人类评估中**优于**远大于它的模型。

### 6.5 数据分布的作用

预训练数据的某些特征（长程连贯性、大量稀有类别）与涌现式 few-shot prompting 相关，可能在小模型上启用它（Xie et al., 2022; Chan et al., 2022）。计算语言学工作表明，在参数量与算力恒定时，训练数据的 **threshold frequencies** 可激活涌现式句法规则学习（Wei et al., 2021b），甚至出现类似心理语言学文献中的 striking "aha" 时刻（Abend et al., 2017; Zhang et al., 2021）。

### 6.6 缩放的固有局限

仅靠增大规模存在局限：硬件约束可能成为瓶颈；训练数据分布之外的远 OOD 任务可能永远达不到显著性能；能力可能涌现后即 plateau（无保证达到期望水平）。

---

## 第7章　涌现风险与社会学影响

### 7.1 Emergent risks

类似涌现能力在未被显式纳入预训练时出现，风险也可能涌现（Bommasani et al., 2021; Steinhardt, 2021; Ganguli et al., 2022）。论文汇总若干关于特定社会风险与规模的关系：

- **Bias**：在 WinoGender（Rudinger et al., 2017）上规模迄今改善了表现（Du et al., 2021; Chowdhery et al., 2022）；但 BIG-Bench (2022) 在 BBQ 基准（Parrish et al., 2022）上发现，对**模糊语境**，bias 可随规模上升。
- **Toxicity**：Askell et al. (2021) 发现更大模型可从 RealToxicityPrompts（Gehman et al., 2020）产生更多有毒响应，但可通过给予 "helpful, harmless, honest" 示例的 prompt 缓解。
- **Memorization**：更大模型更可能记住训练数据（Carlini et al., 2021; 2022），但去重方法可同时降低记忆并改善性能（Kandpal et al., 2022; Lee et al., 2022a）。
- **Truthfulness**：TruthfulQA（Lin et al., 2021）显示 GPT-3 越大越易模仿人类 falsehood；但 Rae et al. (2021) 在多选题版本上显示 Gopher 放大到 280B 可启用**高于**随机的涌现表现。

未来/尚未刻画的风险（Hendrycks et al., 2021b）：backdoor 漏洞、无意欺骗、有害内容合成。已提出数据过滤、预测、治理、自动发现有害行为等应对手段（Bender et al., 2021; Weidinger et al., 2021; Steinhardt, 2021; Ganguli et al., 2022; Perez et al., 2022）。

### 7.2 Sociological shift

另一类质变是社会学的：规模增长改变了社区看待与使用语言模型的方式。NLP 历史上以**任务特定**模型为主（Jurafsky & Martin, 2009）；近期规模带来了"**通用目的**"模型的爆发——单一模型旨在执行训练数据中未显式编码的一系列任务（如 GPT-3、Chinchilla、PaLM；Manning, 2022）。

这一转变的关键证据：few-shot prompted 的通用模型**超越**此前由微调任务特定模型保持的 SOTA：

- GPT-3 175B 在 TriviaQA 与 PiQA 上取得新 SOTA（Brown et al., 2020）。
- PaLM 540B 在三项算术推理基准上取得新 SOTA（Chowdhery et al., 2022）。
- 多模态 Flamingo 80B 在六项视觉问答基准上取得新 SOTA（Alayrac et al., 2022）。

这些能力的 scaling curve 平滑可预测，严格说未必"涌现"，但印证了向通用模型的社会学转变。

### 7.3 实际应用

通用模型仅凭少量示例执行未见任务的能力，催生了 NLP 社区之外的大量应用：将自然语言指令翻译为机器人可执行动作（Ahn et al., 2022; Huang et al., 2022）、与用户交互（Coenen et al., 2021; Wu et al., 2021; 2022a; Lee et al., 2022b）、多模态推理（Zeng et al., 2022; Alayrac et al., 2022）。LLM 已部署于真实产品（如 GitHub Copilot）及作为服务本身（如 OpenAI 的 GPT-3 API）。

---

## 第8章　未来方向与总结

论文给出六条未来方向：

1. **Further model scaling**——继续放大至今仍能增加能力，但计算昂贵且需解决硬件挑战。
2. **Improved architectures & training**——sparse mixture-of-experts（Lepikhin et al., 2021; Fedus et al., 2021; Artetxe et al., 2021; Zoph et al., 2022）在恒定单输入算力下放大参数；可变计算量（Graves, 2016; Dehghani et al., 2018）、局部化学习（Jaderberg et al., 2017）、外部记忆（Guu et al., 2020; Borgeaud et al., 2021; Wu et al., 2022b）等。
3. **Data scaling**——Hoffmann et al. (2022) 指出 Kaplan et al. (2020) 低估了 compute-optimal 模型所需数据量；在固定模型规模约束下，收集大数据集训练更久可解锁更广涌现。
4. **Better prompting techniques & understanding**——输出概率校准（Zhao et al., 2021; Holtzman et al., 2021）、noisy channel（Min et al., 2022a）、中间步骤增强（Reynolds & McDonell, 2021; Nye et al., 2021; Wei et al., 2022b）；对 prompting 成功原因的理解（Wei et al., 2021a; Xie et al., 2022; Min et al., 2022b; Olsson et al., 2022）可助在小模型上激发涌现。
5. **Frontier tasks**——即使最大模型仍无法以高于随机精度完成的任务。BIG-Bench 数十个此类任务见附录 E.4，常涉及抽象推理（如弈棋、挑战性数学）。多语言涌现（模型规模与训练数据共同作用）、多模态 prompting（Alayrac et al., 2022; Ramesh et al., 2022）也是增长方向。
6. **Understanding emergence**——理解涌现**为何**及**如何**发生。论文做了初步分析（cross-entropy 缩放、生成式任务多指标、任务类型分析，见附录 A/B），但未给出完整答案。未来可分析涌现任务与训练中相似数据的关系、构造需多个组合子任务的合成任务等。

**BIG-Bench 任务类型分析（附录 A.3）**：论文将 210 个 BIG-Bench 任务手工分类。涌现占比最高的五个关键词为 analogical reasoning、word sense disambiguation、truthfulness、social reasoning、emotional understanding；arithmetic 与 mathematics 的涌现占比反而较低。flat（所有模型都不优于随机）比例最高的是 visual reasoning（8/13）。40 个任务未归入任何类别。

**前沿现状**：BIG-Bench 上仍有数十个任务，即使最大的 GPT-3 与 PaLM 模型也达不到高于随机的表现（附录 E.4 列出 flat 任务，如 checkmate in one、sudoku、mathematical induction、program synthesis 等）。

**总结**：涌现能力是放大语言模型的一个新近被发现的结果，跨越多种模型、任务类型与实验场景。其"如何涌现"与"更多缩放是否会带来进一步涌现"是 NLP 领域重要的未来研究方向。理解涌现之所以重要，在于它可能让我们**预测未来模型将具备的能力**，并为训练更强模型提供新洞见。
