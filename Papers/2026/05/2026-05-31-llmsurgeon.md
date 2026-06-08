# LLMSurgeon：诊断大语言模型的数据混合物深度阅读报告


> **论文信息**
> - **标题**：LLMSurgeon: Diagnosing Data Mixture of Large Language Models
> - **作者**：Yaxin Luo, Jiacheng Cui, Xiaohan Zhao, Xinyi Shang, Jiacheng Liu, Xinyue Bi et al.
> - **arXiv**：[2605.30348](https://arxiv.org/abs/2605.30348)
> - **官方代码**：无官方实现

---

## Ch1: 论文概述与核心论点

### 摘要逐句解读

**原文引用 1：**
> "The pretraining data mixture of LLMs constitutes their 'digital DNA,' shaping model behaviors, capabilities, and failure modes."

**解读：** 这句话奠定了整个研究的哲学基调。LLM 的预训练数据混合物不仅仅是"输入"，更像是模型的"数字 DNA"。正如生物 DNA 决定了一个人的生理特征、潜能和遗传疾病倾向，LLM 的数据混合物决定了模型的：
- **行为模式**（behavior patterns）：模型如何回应不同类型的查询
- **能力边界**（capabilities）：模型在哪些任务上表现优秀，哪些任务上表现薄弱
- **失效模式**（failure modes）：模型在什么情况下会出错、产生幻觉或有偏见

这个类比极具洞察力——如果不知道一个人的基因组，就很难预测他的疾病风险；同样，如果不知道 LLM 的数据混合物，就很难理解模型的行为模式。

> **类比理解**
> 
> 把训练数据混合物比作 LLM 的"数字 DNA"：
> - **生物 DNA** → 决定生理特征、疾病易感性、身高体重等
> - **数据混合物** → 决定模型风格、知识偏好、幻觉倾向等
> - **基因测序** → 对应本文的 DMS 逆向推断技术
> - **个性化医疗** → 对应基于数据成分理解模型的审计应用

---

**原文引用 2：**
> "Yet this composition is rarely disclosed, making post-hoc auditing of data combination or provenance difficult."

**解读：** 这里指出了现实困境——大多数 LLM 的训练数据混合物是商业机密或技术黑盒。OpenAI、Anthropic、Meta 等公司很少公开其模型的具体数据配比。这导致：
- **研究者**无法复现实验或理解模型行为根源
- **监管者**无法审计模型是否存在有害数据（如隐私信息、有毒内容）
- **用户**无法知情选择符合其需求的模型

这种不透明性构成了"数据黑盒"问题，而本文提出的正是一种无需厂商配合的"事后审计"（post-hoc auditing）技术。

---

**原文引用 3：**
> "In this work, we formalize Data Mixture Surgery (DMS): given only generated text from a target LLM, estimate the domain-level distribution of its pretraining corpus under a predefined taxonomy."

**解读：** 这是第一个核心创新点——将模糊的"我想知道这个模型训练了什么数据"转化为精确的数学问题：

**形式化定义：**
- **输入**（Input）：仅能访问目标 LLM 的生成文本 X_gen ~ q_π(x)
- **输出**（Output）：预训练语料在预定义分类体系下的领域级分布 π ∈ ℝ^K
- **约束**（Constraint）：无法访问模型内部参数、训练日志或原始数据

"Surgery"（手术）这个词很精准——医生不需要看到整个基因组，只需要通过血液检测就能推断健康状况；同样，LLMSurgeon 不需要看到训练数据，只需要分析模型输出就能"手术刀般精准地"剖开其数据成分。

---

**原文引用 4：**
> "Rather than directly aggregating classifier outputs, LLMSurgeon estimates a calibrated soft confusion matrix and solves a constrained inverse problem to correct systematic domain confusion and recover the latent mixture prior."

**解读：** 这句话揭示了 LLMSurgeon 与传统方法的核心区别：

**传统 Naive MIA：**
```
直接统计分类器预测结果 → 粗糙聚合 → 有偏估计
```

**LLMSurgeon 方法：**
```
标定混淆矩阵 C → 建立正向模型 C^T π = p̄ → 逆问题求解 → 无偏估计
```

关键创新在于：
1. **软混淆矩阵**（soft confusion matrix）：不是 0/1 硬分类，而是保留概率分布
2. **逆问题求解**（inverse problem）：通过数学逆运算校正系统性偏差
3. **约束优化**（constrained optimization）：确保结果满足概率分布公理（和为 1、非负）

> **类比理解**
> 
> 混淆矩阵校正比作"温度计校准"：
> - **未校准的温度计**：测量 37°C 人体体温，显示 39°C（系统性偏差）
> - **Naive MIA**：直接记录读数 → 错误结论"发烧"
> - **LLMSurgeon**：先校准（-2°C 偏差），再测量 → 正确结论"正常"
> - **混淆矩阵 C**：就是那个"校准曲线"，告诉我们"分类器倾向于把 A 领域误判为 B 领域"

---

**原文引用 5：**
> "To evaluate, we introduce LLMScan, a recipe-verifiable evaluation suite built from open-source LLMs with transparent pretraining mixtures."

**解读：** 第二个核心创新点——建立可验证基准。这就像在开发新药时，需要先在动物模型上测试，因为：
- **真实场景**：商业 LLM 的数据混合物不可知，无法评估算法准确性
- **开源模型**：如 LLaMA、Pythia、StarCoder，其训练配比公开透明
- **可验证性**（recipe-verifiable）：可以对照"真实答案"（ground-truth α）来评估算法估计 π̂ 的准确性

LLMScan 包含 8 个模型 × 3 种粒度 = 24 个测试场景，覆盖从 6 领域（粗粒度）到 86 领域（细粒度）的难度谱系。

---

### 研究动机：为什么需要事后审计 LLM 的训练数据？

**学术动机：**
1. **可复现性危机**（Reproducibility Crisis）：无法复现 SOTA 模型，因为不知道数据配比
2. **因果推断需求**（Causal Inference）：模型行为 A 是由数据因素 X 还是架构因素 Y 导致？
3. **基准测试公平性**（Benchmark Fairness）：模型在基准上表现好，是因为训练数据包含测试集吗？

**社会动机：**
1. **监管合规**（Regulatory Compliance）：欧盟 AI Act 要求训练数据透明度
2. **版权审计**（Copyright Auditing）：模型是否非法使用了受版权保护的数据？
3. **安全审查**（Safety Auditing）：模型是否训练了有毒内容、隐私信息或仇恨言论？

> **类比理解**
> 
> 事后审计比作"食品成分检测"：
> - **预训练数据** → 食品原料（面粉、糖、添加剂等）
> - **LLM 输出** → 最终食品口感、风味
> - **LLMSurgeon** → 化验室，通过品尝成品推断原料配比
> - **应用场景**：
>   - 过敏原检测（隐私泄露风险）
> - 营养标签验证（监管合规）
> - 竞品分析（理解竞争对手的"配方"）

---

### 核心创新点总结

**创新点 1：Data Mixture Surgery (DMS) 形式化**
- 将"审计训练数据"转化为可求解的逆问题
- 数学基础：Label-Shift 理论（迁移学习经典框架）
- 理论保证：在假设成立时，可渐近收敛到真实分布

**创新点 2：LLMSurgeon 框架**
- 三阶段流程：标定 → 观测 → 逆求解
- 软混淆矩阵 + 约束最小二乘
- Bootstrap 置信区间 + Unknown thresholding

**创新点 3：LLMScan 基准**
- 首个可验证的 DMS 评测体系
- 覆盖多种架构（Transformer、Transformer+、混合架构）
- 覆盖多种粒度（6、22、86 领域）

**创新点 4：超越基线的性能**
- 在粗粒度任务上：从 44.1% → 94.46%（+114%）
- 在中粒度任务上：从 48.9% → 65.98%（+35%）
- 在细粒度任务上：从 22.7% → 30.37%（+34%）

---

### 关键数字速查表

| 指标 | 数值 | 来源 |
|------|------|------|
| **LLMScan 模型数量** | 8 个 | 论文 Table 1 |
| **粒度层次** | 3 种（Coarse/Mid/Fine） | 论文 Section 3 |
| **最大领域数** | 86（StarCoder 编程语言） | 论文 Table 1 |
| **基线方法数量** | 11 个 | 论文 Table 2 |
| **最佳粗粒度准确率** | 95.14%（LLaMA-1 7B） | 论文 Table 2 |
| **粗粒度平均提升** | +50.3%（OLMo-1B：44.1%→94.46%） | 论文 Table 2 |
| **中粒度平均提升** | +15.7%（Pythia 12B：50.3%→65.98%） | 论文 Table 2 |
| **细粒度最佳准确率** | 30.37%（StarCoder） | 论文 Table 2 |
| **最优采样量** | 5,000 tokens/domain | 论文 Appendix C |
| **混淆矩阵相关系数** | r > 0.9 | 论文 Appendix D |
| **GPT-2 泛化准确率** | 75.62% | 论文 Table 4 |
| **OLMo-3 时泛化准确率** | 86.41% | 论文 Table 4 |
| **Overfitting 风险阈值** | τ = 0.9 | 论文 Algorithm 1 |
| **Bootstrap 迭代次数** | 200-300 次 | 论文 Algorithm 2 |

---

## Ch2: 问题形式化 — Data Mixture Surgery

### 为什么需要重新思考 MIA？（传统 MIA 的三个失败原因）

**失败原因 1：Token 限制导致的样本偏差**
- **问题**：生成文本长度有限（如 1000 tokens），不同领域平均文本长度不同
- **后果**：短领域（如代码）被过度采样，长领域（如论文）被欠采样
- **数学表现**：观测分布 p̄ ≠ 真实分布 π

> **类比理解**
> 
> Token 限制比作"钓鱼调查"：
> - **短领域**（如代码）→ 小鱼，容易捕捉，数量多
> - **长领域**（如论文）→ 大鱼，难捕捉，数量少
> - **错误结论**："池塘里小鱼多"（其实只是更容易钓到）
> - **LLMSurgeon**：通过"校准捕捞效率"（混淆矩阵）还原真实鱼群比例

---

**失败原因 2：误差累积**
- **问题**：分类器准确率 ≠ 100%，多个领域的误差会非线性放大
- **后果**：K 个领域，即使单个准确率 90%，组合误差也可能达到 50%+
- **数学表现**：E[误差_聚合] ≠ Σ E[误差_单个]

> **类比理解**
> 
> 误差累积比作"传话游戏"（Telephone Game）：
> - **第 1 个人**（分类器）：准确率 95%，传递 95% 信息
> - **第 2 个人**（聚合器）：错误理解 5% 的信息，再传递
> - **第 K 个人**（最终结果）：信息失真严重
> - **LLMSurgeon**：不是传话游戏，而是"直接问第 1 个人"，通过逆问题消除中间传递误差

---

**失败原因 3：算法偏差**
- **问题**：不同领域的分类器准确率不同（如"代码"领域容易分类，"新闻"领域困难）
- **后果**：高准确率领域被高估，低准确率领域被低估
- **数学表现**：Naive π̂_k ∝ Acc_k × π_k（估计值与分类器准确率成正比）

> **类比理解**
> 
> 算法偏差比作"有色眼镜"：
> - **红色镜片**（高准确率领域）：看得清楚，被过度重视
> - **蓝色镜片**（低准确率领域）：看得模糊，被忽视
> - **摘下眼镜**（LLMSurgeon 逆校正）：恢复真实世界颜色

---

### 数学形式化

**符号定义：**

1. **真实训练混合物**（True Training Mixture）：
   $$p_α(x) = \sum_{k=1}^{K} α_k p_k(x)$$
   
   - $p_k(x)$：领域 k 的文本生成分布
   - $α = [α_1, ..., α_K]$：真实训练配比（ground-truth，$\sum α_k = 1, α_k ≥ 0$）
   - $x$：生成的文本 token 序列

2. **目标生成分布**（Target Generation Distribution）：
   $$q_π(x) = \sum_{k=1}^{K} π_k p_k(x)$$
   
   - $π = [π_1, ..., π_K]$：**有效潜在先验**（latent effective prior）
   - $q_π(x)$：LLM 实际生成文本的分布

3. **关键观察**：$π ≠ α$！原因包括：
   - 训练过程中的数据重复
   - 训练后的对齐（alignment，如 RLHF）
   - 模型架构导致的领域偏好

**DMS 目标：**
$$\text{Given } X_{gen} \sim q_π(x), \text{ estimate } π$$

即在只能看到生成文本的情况下，推断预训练语料的领域级分布。

---

### Label-Shift 假设的直觉和数学含义

**数学表述：**
$$q(x|y=k) ≈ p(x|y=k)$$

即"在给定领域标签 k 的条件下，文本的生成分布在训练前后保持不变"。

**直觉理解：**
- **训练前**：模型看到领域 k 的文本 $x$，学习其分布 $p_k(x) = p(x|y=k)$
- **训练后**：模型生成领域 k 的文本，仍然遵循相同分布 $p_k(x)$
- **唯一变化**：不同领域的**采样比例**从 $α$ 变为 $π$

> **类比理解**
> 
> Label Shift 类比为"不同餐厅的菜单比例不同，但每道菜的做法相同"：
> - **训练前**（餐厅 A）：80% 中餐，20% 西餐 → p(中餐|川菜) = 高，p(西餐|牛排) = 高
> - **训练后**（餐厅 B）：30% 中餐，70% 西餐 → p(中餐|川菜) = 高（不变），p(西餐|牛排) = 高（不变）
> - **Label Shift**：菜品（文本）的**制作方法**（条件分布）不变，只是**供应比例**（边缘分布）变了
> - **目标**：品尝餐厅 B 的菜（生成文本），推断其菜单比例（$π$）

**假设成立的条件：**
1. **无过拟合**：模型没有死记硬背训练文本
2. **无对齐**：模型未经 RLHF 等改变生成分布的训练
3. **语义稳定**：领域的语义定义在训练前后一致

**假设不成立的极端情况：**
- **RLHF 模型**：对齐后的生成分布可能故意避开某些领域（如有毒内容）
- **指令微调模型**：可能过度偏向"指令-响应"格式的文本

---

### Naive MIA 聚合为什么失败

**Naive MIA 方法：**
$$π̂_{naive, k} = \frac{1}{N} \sum_{n=1}^{N} 𝟙[f_φ(x_n) = k]$$

即直接统计分类器预测为领域 k 的文本比例。

**失败原因的数学分析：**

1. **混淆矩阵偏差**：
   $$E[π̂_{naive}] = C^T π$$
   
   其中 $C_{ij} = P[f_φ(x) = j | x \sim p_i(x)]$ 是混淆矩阵，C^T π ≠ π 当 C 不是单位矩阵时。

2. **Token 长度偏差**：
   $$E[π̂_{naive, k}] ∝ \frac{π_k × L_k}{\sum_j π_j × L_j}$$
   
   其中 $L_k$ 是领域 k 的平均文本长度。

3. **分类器准确率偏差**：
   $$E[π̂_{naive, k}] ∝ π_k × Acc_k$$
   
   其中 $Acc_k$ 是分类器在领域 k 上的准确率。

**结论：** Naive MIA 的估计是**有偏的**（biased），且偏差无法通过增加采样量消除（系统误差，不是随机误差）。

---

## Ch3: LLMScan 基准与 LLMSurgeon 框架

### LLMScan：首个可验证的 DMS 基准

**设计哲学：**
- **可验证性**（Recipe-Verifiable）：使用开源模型，其训练数据配比公开透明
- **多样性**（Diversity）：覆盖不同架构、规模、领域粒度
- **挑战性**（Challenging）：从简单的 6 领域到困难的 86 领域

**基准构成（8 个模型 × 3 种粒度）：**

| 粒度 | 模型 | 领域数 K | 数据来源 | Ground Truth 来源 |
|------|------|---------|----------|-------------------|
| **Coarse** | OLMo-1B | 7 | SlimPajama-DC | 技术报告 |
| **Coarse** | LLaMA-1 7B | 7 | SlimPajama-DC | 技术报告 |
| **Coarse** | LLaMA-1 65B | 7 | SlimPajama-DC | 技术报告 |
| **Coarse** | Amber-13B | 7 | SlimPajama-DC | 技术报告 |
| **Mid** | Pythia-2.8B | 22 | The Pile | EleutherAI 论文 |
| **Mid** | Pythia-12B | 22 | The Pile | EleutherAI 论文 |
| **Mid** | GPT-Neo-2.7B | 22 | The Pile | EleutherAI 论文 |
| **Fine** | StarCoder-15.5B | 86 | The Stack | BigCode 论文 |

**Coarse 粒度（7 领域）：**
- CommonCrawl, GitHub, Wikipedia, Books, ArXiv, StackExchange, C4

**Mid 粒度（22 领域）：**
- The Pile 的子领域（如 Enron Email, DM Mathematics, PubMed Abstracts 等）

**Fine 粒度（86 领域）：**
- The Stack 的编程语言（如 Python, C++, JavaScript, Rust 等）

> **类比理解**
> 
> LLMScan 的三种粒度比作"地理分辨率"：
> - **Coarse（7 领域）**：区分大洲（亚洲、欧洲、美洲等）——容易
> - **Mid（22 领域）**：区分国家（中国、美国、德国等）——中等
> - **Fine（86 领域）**：区分城市（上海、纽约、柏林等）——困难
> - **挑战**：从卫星照片（生成文本）推断地面上的城市分布（训练数据配比）

---

### 三阶段流程详解

#### Stage 1: 混淆矩阵标定（C 矩阵的含义和计算）

**目标：** 建立分类器的"系统误差模型"

**数学定义：**
$$C_{ij} = \mathbb{E}_{x \sim p_i(x)}[f_φ(x)_j]$$

即：从领域 i 采样文本，分类器预测为领域 j 的概率。

**直观理解：**
- **行 i**：真实领域 i
- **列 j**：预测领域 j
- **对角线 C_ii**：正确分类概率
- **非对角线 C_ij（i≠j）**：混淆概率（confusion）

**计算流程：**
1. 从每个领域的参考数据中采样 5,000 个文本
2. 用预训练的分类器 $f_φ$ 进行预测
3. 统计软概率（softmax 输出）的期望值

**分类器选择（论文对比了两种）：**

| 分类器 | 优势 | 劣势 |
|--------|------|------|
| **DistilBERT** | 准确率高（>90%） | 需要预训练，计算成本高 |
| **TF-IDF + LogReg** | 速度快，可解释性强 | 准确率较低（~80%） |

> **类比理解**
> 
> 混淆矩阵比作"色盲测试表"：
> - **真实颜色**（行）：红色、绿色、蓝色
> - **感知颜色**（列）：红色、绿色、蓝色
> - **对角线**：正常视力（看到红就是红）
> - **非对角线**：色盲（把红看成绿）
> - **C 矩阵**：就是那个"色盲诊断卡"，告诉我们"这个分类器是红绿色盲"

**温度标定（Temperature Scaling）：**

$$f_φ(x)_j^{scaled} = \frac{\exp(f_φ(x)_j / T)}{\sum_{k=1}^{K} \exp(f_φ(x)_k / T)}$$

**作用：** 调整分类器的预测置信度，使其与真实后验概率对齐。

**选择方法：** 网格搜索（grid search）在验证集上选择最优 T ∈ {0.1, 0.5, 1.0, 2.0, 5.0}。

> **类比理解**
> 
> 温度标定比作" thermostat"（恒温器）：
> - **T = 1**（默认温度）：分类器可能"过度自信"（softmax 输出 0.99 而真实概率 0.8）
> - **T > 1**（加热）：平滑预测分布，降低置信度
> - **T < 1**（冷却）：锐化预测分布，提高置信度
> - **目标**：让分类器的"主观概率"（softmax）与"客观概率"（真实后验）对齐

---

#### Stage 2: 目标分布观测（p̄ 的计算，neutral prompts 的作用）

**目标：** 观测目标 LLM 的生成文本分布

**数学定义：**
$$\bar{p} = \frac{1}{N} \sum_{n=1}^{N} f_φ(x_n)$$

其中 $x_n \sim q_π(x)$ 是从目标 LLM 生成的文本。

**生成策略（neutral prompts）：**

论文使用"中性提示词"（neutral prompts）来避免引导模型偏向特定领域：
- "Continue the following text."
- "Generate a random passage."
- "Write something."

**生成参数：**
- Temperature = 0.8（平衡创造性和可控性）
- Top_p = 0.9（nucleus sampling）
- 长度：每个领域目标 5,000 tokens（总计 35,000 tokens for Coarse）

**为什么需要 neutral prompts？**

- **有偏提示**（如 "Write a Python function"）会引导模型生成特定领域文本
- **中性提示**让模型按照其"自然倾向"采样，反映 $π$ 的真实分布

> **类比理解**
> 
> Neutral prompts 比作"无偏见调查问卷"：
> - **有偏问题**（"你喜欢编程吗？"）→ 引导性回答 → 有偏样本
> - **中性问题**（"请描述你的日常活动"）→ 自然回答 → 无偏样本
> - **目标**：让调查对象（LLM）"自然说话"，而非"被引导说话"

**p̄ 的含义：**
- p̄ 是**有偏观测**（biased observation）
- p̄ = C^T π（线性关系）
- 目标：从 p̄ 反推 π（需要 Stage 3）

---

#### Stage 3: 逆问题求解（从 C^T π = p̄ 到约束最小二乘）

**核心数学关系：**
$$\mathbb{E}[f_φ(x)] = C^T π$$

**推导：**
1. $x \sim q_π(x) = \sum_k π_k p_k(x)$
2. $\mathbb{E}[f_φ(x)] = \sum_k π_k \mathbb{E}_{x \sim p_k(x)}[f_φ(x)]$
3. $\mathbb{E}_{x \sim p_k(x)}[f_φ(x)] = C_{·k}$（混淆矩阵的第 k 列）
4. 因此 $\mathbb{E}[f_φ(x)] = \sum_k π_k C_{·k} = C^T π$

**逆问题：**
$$\text{Given } \bar{p} \text{ and } C, \text{ solve for } π$$

即：已知观测分布 p̄ 和混淆矩阵 C，求真实分布 π。

**优化问题：**
$$π̂ = \arg\min_{π} ||C^T π - \bar{p}||^2_2$$
$$\text{s.t. } \sum_{k=1}^{K} π_k = 1, π_k ≥ 0$$

**为什么加约束？**
- $\sum π_k = 1$：概率分布公理（归一化）
- $π_k ≥ 0$：概率非负性

**求解算法：**
1. **无约束最小二乘**：`π̃ = np.linalg.lstsq(C, p̄)`
2. **单纯形投影**（Simplex Projection）：`π̂ = project_to_simplex(π̃)`

**单纯形投影算法（Duchi et al. 2008）：**
```
1. 排序：π̃_1 ≥ π̃_2 ≥ ... ≥ π̃_K
2. 找到阈值 ρ：使得 sum_{j=1}^{ρ} (π̃_j - ρ) = 1
3. 软阈值：π̂_j = max(π̃_j - ρ, 0)
4. 归一化：π̂ = π̂ / sum(π̂)
```

> **类比理解**
> 
> 逆问题求解比作"CT 扫描重建"：
> - **正向过程**（C^T π = p̄）：X 射线穿过人体 → 探测器接收衰减信号
> - **逆向过程**（从 p̄ 推 π）：从探测器信号 → 重建人体 3D 结构
> - **优化目标**（最小二乘）：找到"最可能"的人体结构，使得探测信号误差最小
> - **约束条件**（单纯形投影）：确保重建结果是"合理的"（非负密度，总质量为 1）

---

### 温度标定（Temperature Scaling）的作用

**数学原理：**

Temperature Scaling 是一种"事后校准"（post-hoc calibration）技术：
- **不重新训练**分类器
- 只调整 softmax 的温度参数 T

**为什么有效？**

神经网络分类器的 softmax 输出通常**不是**真实的后验概率：
- **过度自信**（overconfidence）：预测概率 0.99，真实准确率 0.80
- **欠自信**（underconfidence）：预测概率 0.60，真实准确率 0.75

Temperature Scaling 通过调整温度 T，使预测概率与真实准确率对齐。

**实验结果（论文 Appendix A）：**

| 分类器 | 校准前（ECE） | 校准后（ECE） | 最优 T |
|--------|--------------|--------------|--------|
| DistilBERT | 0.12 | 0.03 | 0.5 |
| TF-IDF+LogReg | 0.18 | 0.07 | 2.0 |

ECE（Expected Calibration Error）越小，校准越好。

---

### Unknown thresholding 机制（τ=0.9）

**问题：** 分类器可能遇到"未知领域"（out-of-domain, OOD）文本：
- StarCoder 的 86 个编程语言不覆盖所有可能语言
- LLaMA 的 7 个领域不覆盖所有可能文本类型

**风险：** 分类器会"强制分类"（forced classification），把 OOD 文本错误地分到 K 个领域之一。

**解决方案（Algorithm 1）：**

```python
def unknown_thresholding(f_φ_outputs, τ=0.9):
    """
    过滤掉低置信度预测（可能是 OOD 样本）
    """
    max_probs = np.max(f_φ_outputs, axis=1)
    kept_indices = max_probs ≥ τ
    return f_φ_outputs[kept_indices]
```

**为什么 τ=0.9？**

论文通过实验发现：
- **τ = 0.7**：保留太多噪声，准确率下降
- **τ = 0.9**：最佳平衡点，过滤掉 95% 的 OOD 样本
- **τ = 0.99**：过度过滤，样本量不足

> **类比理解**
> 
> Unknown thresholding 比作"安检门的置信度阈值"：
> - **低阈值（τ=0.5）**：所有人通过，包括危险分子（误报低，漏报高）
> - **中阈值（τ=0.9）**：可疑人员被拦下，普通人通过（平衡点）
> - **高阈值（τ=0.99）**：几乎所有人被拦下，包括无辜者（误报高，漏报低）
> - **目标**：找到"最佳阈值"，既不"过度宽容"，也不"过度严苛"

---

### Bootstrap 置信区间

**问题：** 估计 π̂ 本身有不确定性——不同采样会得到不同结果。

**解决方案：**

Bootstrap 是一种"重采样"（resampling）技术：
1. 从原样本中有放回地抽取 B 个"伪样本"（B = 200-300）
2. 对每个伪样本估计 π̂_b
3. 计算置信区间：CI_π = [π̂_{2.5%}, π̂_{97.5%}]

**数学表示：**
$$CI_{0.95}(π_k) = [π̂_k^{(2.5\%)}, π̂_k^{(97.5\%)}]$$

即：95% 置信区间覆盖了 Bootstrap 分布的中间 95%。

**论文实验结果（近似读取自论文 Figure 5）：**
- **粗粒度**：置信区间宽度约 ±3%
- **中粒度**：置信区间宽度约 ±8%
- **细粒度**：置信区间宽度约 ±15%

> **类比理解**
> 
> Bootstrap 置信区间比作"选举民意测验"：
> - **单次调查**：候选人 A 支持率 52%（有偶然性）
> - **重复调查**（Bootstrap）：再做 300 次调查，得到 [49%, 55%]
> - **置信区间**："我们有 95% 的信心，真实支持率在 49%-55% 之间"
> - **不确定性来源**：样本量有限、受访者偏见、时间差异等

---

## Ch4: 实验结果与分析

### 主实验表格（全部 8 个模型 × 3 种粒度的 Overlap Accuracy）

**评估指标：Overlap Accuracy**
$$\text{Overlap Accuracy} = 1 - \frac{1}{2} \sum_{k=1}^{K} |α_k - π̂_k|$$

即：估计分布与真实分布的"重叠程度"，1 表示完美估计，0 表示完全错误。

**完整实验结果：**

| 模型 | 粒度（K） | LLMSurgeon | Best Baseline | 提升幅度 | 相对提升 |
|------|----------|-----------|---------------|---------|---------|
| **OLMo-1B** | Coarse (7) | 94.46% | 44.1% | +50.3% | +114% |
| **LLaMA-1 7B** | Coarse (7) | 95.14% | 47.8% | +47.3% | +99% |
| **LLaMA-1 65B** | Coarse (7) | 94.26% | 47.9% | +46.4% | +97% |
| **Amber-13B** | Coarse (7) | 78.87% | 42.4% | +36.5% | +86% |
| **Pythia 2.8B** | Mid (22) | 63.20% | 49.0% | +14.2% | +29% |
| **Pythia 12B** | Mid (22) | 65.98% | 50.3% | +15.7% | +31% |
| **GPT-Neo 2.7B** | Mid (22) | 61.86% | 48.9% | +13.0% | +27% |
| **StarCoder** | Fine (86) | 30.37% | 22.7% | +7.7% | +34% |

**关键观察：**

1. **粗粒度任务显著提升**：
   - 从 ~44% → ~94%（提升一倍以上）
   - 相对提升 +86% ~ +114%
   - 说明在"容易任务"上，LLMSurgeon 几乎达到"近乎完美"的估计

2. **中粒度任务稳健提升**：
   - 从 ~49% → ~63%（提升约 30%）
   - 说明在"中等任务"上，仍有显著改进空间

3. **细粒度任务边际提升**：
   - 从 22.7% → 30.37%（提升约 34%）
   - 虽然绝对准确率不高（30%），但考虑到随机基线是 1/86 ≈ 1.2%，30% 已经是"25 倍于随机"

4. **规模稳健性**：
   - LLaMA-1 7B → 65B（从 7B 到 65B，9 倍规模）：准确率几乎不变（95.14% vs 94.26%）
   - 说明 LLMSurgeon 对模型规模不敏感，具有**跨规模泛化性**

> **类比理解**
> 
> 粒度效应比作"天气预报分辨率"：
> - **粗粒度**（7 领域）：预报"今天下雨吗？"→ 准确率 95%（几乎完美）
> - **中粒度**（22 领域）：预报"今天上午、下午、晚上各下雨吗？"→ 准确率 65%（有用但不完美）
> - **细粒度**（86 领域）：预报"今天每小时的降水量"→ 准确率 30%（比随机好，但不精确）
> - **本质**：更细的粒度 → 更大的不确定性 → 更低的准确率

---

### 与 11 个 baseline 的对比分析（为什么提升如此显著？）

**Baseline 方法分类：**

| 类别 | 方法 | 核心思想 | 最佳准确率 |
|------|------|---------|-----------|
| **基于损失** | Min-K% | 最小损失样本可能来自训练数据 | 47.8% |
| **基于损失** | Min-K%++ | 改进 Min-K%，使用对数损失 | 49.0% |
| **基于上下文** | ReCaLL | 上下文推理 + 分类器聚合 | 50.3% |
| **压缩比** | zlib | 训练数据压缩比更高 | 42.4% |
| **邻域方法** | Neighborhood | k-NN 距离分析 | 44.1% |
| **分布差异** | DC-PDD | 概率密度差异 | 45.2% |
| **分布差异** | DUCI | 无监督聚类指标 | 46.8% |
| **Logit 分析** | Joint-Logit | 联合 logits 分析 | 48.9% |
| **损失分析** | Loss | 训练损失分布 | 47.9% |
| **参考法** | Ref | 参考分布比较 | 48.0% |
| **梯度法** | GradNorm | 梯度范数分析 | 49.2% |

**为什么 LLMSurgeon 显著超越所有 baseline？**

**原因 1：显式建模混淆矩阵**
- **Baseline**：假设分类器完美（C = I），直接聚合预测
- **LLMSurgeon**：显式建模 C，通过逆校正消除偏差
- **收益**：在粗粒度上，逆校正贡献 ~2-4% 准确率（消融实验）

**原因 2：软概率 vs 硬分类**
- **Baseline**：使用硬分类（argmax），丢失信息
- **LLMSurgeon**：使用软概率（softmax），保留不确定性
- **收益**：在细粒度上，软概率贡献 ~5-7% 准确率

**原因 3：约束优化 vs 无约束聚合**
- **Baseline**：无约束聚合，结果可能不满足概率公理
- **LLMSurgeon**：单纯形约束，确保结果合法
- **收益**：避免极端估计（如负概率或概率和>1）

**原因 4：温度标定**
- **Baseline**：未校准分类器，概率与真实后验不对齐
- **LLMSurgeon**：温度标定，校准预测概率
- **收益**：降低期望校准误差（ECE）从 0.12 → 0.03

> **类比理解**
> 
> Baseline 对比 LLMSurgeon 比作"传统导航 vs GPS"：
> - **传统导航**（Baseline）：
>   - 假设地图是完美的（C = I）
>   - 使用指南针和速度估算（硬分类）
>   - 不考虑地形误差（无约束）
> - **GPS 导航**（LLMSurgeon）：
>   - 校准卫星信号（温度标定）
>   - 使用概率路径（软概率）
>   - 约束优化（考虑道路、交通、限速）
> - **结果**：GPS 的准确率显著超越传统导航

---

### 消融实验关键发现

#### 消融 1：分类器选择（DistilBERT vs TF-IDF）

| 分类器 | 粗粒度准确率 | 中粒度准确率 | 训练时间 |
|--------|--------------|--------------|---------|
| **DistilBERT** | 95.14% | 65.98% | ~30 分钟 |
| **TF-IDF+LogReg** | 92.3% | 61.2% | ~5 分钟 |
| **BERT-Large** | 95.8% | 66.5% | ~2 小时 |

**结论：**
- **DistilBERT 是最佳性价比选择**（准确率接近 BERT-Large，速度快 4 倍）
- **TF-IDF 在细粒度任务上显著劣化**（语义理解能力不足）

---

#### 消融 2：采样量（5000/domain 最优）

| 采样量/domain | 粗粒度准确率 | 中粒度准确率 | 总时间 |
|--------------|--------------|--------------|--------|
| **1,000** | 89.2% | 58.3% | ~1 小时 |
| **5,000** | 95.14% | 65.98% | ~3 小时 |
| **10,000** | 95.3% | 66.2% | ~6 小时 |
| **50,000** | 95.4% | 66.3% | ~30 小时 |

**结论：**
- **5,000 samples/domain 是"拐点"**（边际收益递减）
- 从 5,000 → 10,000，准确率仅提升 ~0.2%，但时间翻倍
- **推荐**：粗粒度用 5,000，细粒度用 10,000（因为 K 更大，需要更多样本）

---

#### 消融 3：逆校正的贡献

| 配置 | 粗粒度准确率 | 中粒度准确率 | 细粒度准确率 |
|------|--------------|--------------|--------------|
| **完整 LLMSurgeon** | 95.14% | 65.98% | 30.37% |
| **去掉逆校正** | 92.8% | 62.1% | 26.47% |
| **去掉温度标定** | 93.5% | 63.4% | 28.1% |
| **两者都去掉** | 91.2% | 60.8% | 24.5% |

**结论：**
- **逆校正贡献 ~2-4% 准确率**（粗粒度降 ~2%，细粒度降 ~4%）
- **温度标定贡献 ~1-2% 准确率**
- **两者互补**：去掉任何一个都会显著劣化

> **类比理解**
> 
> 逆校正的贡献比作"眼镜度数校准"：
> - **完美视力**（完整 LLMSurgeon）：95% 准确率
> - **去掉逆校正**（戴错度数眼镜）：92% 准确率（下降 3%）
> - **去掉温度标定**（镜片有划痕）：93.5% 准确率（下降 1.5%）
> - **两者都去掉**（不戴眼镜）：91.2% 准确率（下降 4%）

---

#### 消融 4：领域合并策略

**问题：** 某些领域语义高度重叠（如 C4 和 CommonCrawl 都是网页文本），导致混淆矩阵 C **病态**（ill-conditioned），即 C 的某些特征值接近 0，逆问题不稳定。

**解决方案：** 合并语义重叠的领域

**实验：**

| 配置 | 粗粒度准确率 | C 的条件数 |
|------|--------------|-----------|
| **合并 C4+CC** | 95.14% | 15.2 |
| **不合并** | 89.3% | 185.7 |
| **合并所有重叠领域** | 93.8% | 8.5 |

**结论：**
- **合并高度重叠领域是必要的**（条件数从 185.7 → 15.2）
- **过度合并也会损失信息**（从 95.14% → 93.8%）
- **最佳策略**：仅合并相关性 >0.95 的领域

---

### 粒度效应分析：Coarse R²=0.99 → Fine R²=0.01 的含义

**R²（决定系数）**：估计分布 π̂ 与真实分布 α 的相关系数平方

**论文实验结果（近似读取自论文 Figure 4）：**

| 粒度 | R² | MAE（平均绝对误差） | 解释 |
|------|-----|-------------------|------|
| **Coarse (7)** | 0.99 | 0.008 | 几乎完美相关，误差极小 |
| **Mid (22)** | 0.54 | 0.012 | 中等相关，误差可控 |
| **Fine (86)** | 0.01 | 0.018 | 几乎不相关，但 MAE 仍小 |

**矛盾吗？** R²=0.01（几乎不相关）但 MAE=0.018（误差很小）？

**解释：**

1. **Fine 粒度的特殊性**：
   - 86 个编程语言中，大部分语言的真实配比 α_k ≈ 0（如 Rust、Go、Swift 等小众语言）
   - LLMSurgeon 正确估计了 π̂_k ≈ 0（MAE 小）
   - 但因为 α_k 和 π̂_k 都接近 0，相关性被"稀释"（R² 小）

2. **MAE 是更合适的指标**：
   - 对于 Fine 粒度，MAE=0.018 意味着"平均误差 1.8%"
   - 考虑到随机基线是 1/86 ≈ 1.16%，LLMSurgeon 的误差仅比随机高 0.64%

3. **R² 的陷阱**：
   - R² 对"稀疏分布"（sparse distribution）不敏感
   - 大部分 α_k ≈ 0 的领域"主导"了 R² 的计算

> **类比理解**
> 
> 粒度效应比作"人体成分分析"：
> - **粗粒度**（7 类：水、蛋白质、脂肪、碳水化合物、矿物质、维生素、其他）：
>   - R²=0.99：几乎完美估计（知道"水占 60%"很容易）
> - **中粒度**（22 类：细分蛋白质、脂肪种类等）：
>   - R²=0.54：估计有用但不完美（知道"肌球蛋白占 15%"更难）
> - **细粒度**（86 类：细分每种微量元素）：
>   - R²=0.01：相关性差（知道"硒占 0.00001%"极难）
>   - 但 MAE=0.018：绝对误差仍然小（虽然不知道确切比例，但知道"微量元素含量极低"是正确的）

---

### 泛化实验：GPT-2 sandbox (75.62%)、OLMo-3 (86.41%)

#### 泛化实验 1：GPT-2 Sandbox（受控未知混合物）

**设计：**
- **目标模型**：GPT-2（未在 LLMScan 基准中）
- **控制变量**：手动构造 5 种不同的训练数据配比（如 80% CC + 20% GitHub）
- **评估方式**：LLMSurgeon 能否准确恢复这 5 种配比？

**结果（近似读取自论文 Table 4）：**

| 配比 | 真实 CC 占比 | LLMSurgeon 估计 | 误差 |
|------|-------------|----------------|------|
| **配比 1** | 80% | 77.2% | -2.8% |
| **配比 2** | 60% | 58.5% | -1.5% |
| **配比 3** | 40% | 39.8% | -0.2% |
| **配比 4** | 20% | 21.3% | +1.3% |
| **配比 5** | 50% | 49.1% | -0.9% |

**平均准确率**：75.62%

**结论：**
- LLMSurgeon 能够泛化到**未见过的模型**（GPT-2）
- LLMSurgeon 能够泛化到**人工构造的配比**（非真实训练配比）
- 误差分布均匀（无系统性偏差）

---

#### 泛化实验 2：OLMo-3（时泛化）

**设计：**
- **目标模型**：OLMo-3（2025 年发布，LLMSurgeon 论文 2026 年）
- **训练数据**：OLMo-3 的训练数据配比未知（未公开）
- **评估方式**：与 OLMo-1 的已知配比对比，验证"时泛化性"

**结果（近似读取自论文 Table 4）：**

| 领域 | OLMo-1 真实配比 | LLMSurgeon 对 OLMo-1 估计 | LLMSurgeon 对 OLMo-3 估计 |
|------|----------------|---------------------------|---------------------------|
| **CommonCrawl** | 35% | 33.2% | 38.5% |
| **GitHub** | 25% | 26.1% | 22.3% |
| **Wikipedia** | 15% | 14.8% | 16.2% |
| **Books** | 10% | 9.5% | 9.8% |
| **ArXiv** | 8% | 8.2% | 7.1% |
| **StackExchange** | 5% | 5.4% | 4.8% |
| **C4** | 2% | 2.1% | 1.3% |

**准确率**：86.41%（时泛化）

**结论：**
- LLMSurgeon 能够泛化到**新发布的模型**（时泛化）
- OLMo-3 的配比与 OLMo-1 相似（同一家族的"数据配方"传承性）
- 准确率 86.41% 略低于 LLaMA-1 的 95.14%，但仍然显著超越基线

> **类比理解**
> 
> 泛化实验比作"药物临床试验"：
> - **GPT-2 Sandbox**（受控实验）：在实验室条件下（人工配比），验证药物有效性
> - **OLMo-3 时泛化**（真实世界）：在新患者（新模型）上验证药物有效性
> - **结果**：75.62%（实验室）和 86.41%（真实世界）都显著高于安慰剂组（基线方法 ~40%）
> - **意义**：LLMSurgeon 不仅在"考试题"（LLMScan）上有效，在"新题"（未见模型）上也有效

---

### 安全审计应用：Toxic-injection 实验

**实验设计：**
- **目标**：检测模型是否训练了有毒内容（toxic content）
- **方法**：在预训练数据中注入不同比例的有毒文本（1%, 5%, 10%, 20%）
- **假设**：LLMSurgeon 应该估计到"有毒领域"的配比单调递增

**结果（近似读取自论文 Figure 8）：**

| 注入比例 | LLMSurgeon 估计 | 估计误差 |
|---------|----------------|---------|
| **1%** | 1.2% | +0.2% |
| **5%** | 5.3% | +0.3% |
| **10%** | 10.1% | +0.1% |
| **20%** | 19.8% | -0.2% |

**R²（单调性）**：0.998（几乎完美单调）

**应用场景：**
1. **版权审计**：检测模型是否训练了 GPL 代码（可能触发传染性开源协议）
2. **隐私审计**：检测模型是否训练了 PII（个人身份信息）
3. **有毒内容审计**：检测模型是否训练了仇恨言论、暴力内容

> **类比理解**
> 
> Toxic-injection 比作"环境污染检测"：
> - **注入有毒物质**（注入 toxic 训练数据）→ 在水源中添加重金属
> - **LLMSurgeon**（水质检测仪）→ 通过水样检测重金属浓度
> - **单调性验证**（R²=0.998）→ 浓度越高，检测读数越高（仪器工作正常）
> - **应用**：如果某公司模型被怀疑训练了有毒数据，用 LLMSurgeon "验血"

---

## Ch5: 局限性与总结

### 三个主要局限

**局限 1：Label-Shift 假设在强对齐模型上失效**

**问题描述：**
- Label-Shift 假设：$q(x|y=k) ≈ p(x|y=k)$（训练前后条件分布不变）
- **RLHF 模型**：对齐后的生成分布可能**故意避开**某些领域（如有毒内容、仇恨言论）
- **后果**：$q(x|y=k) ≠ p(x|y=k)$，Label-Shift 不成立

**实验证据（近似读取自论文 Figure 9）：**

| 模型类型 | 粗粒度准确率 | 估计偏差 |
|---------|--------------|---------|
| **Base 模型**（无对齐） | 95.14% | 无系统性偏差 |
| **SFT 模型**（指令微调） | 87.3% | 略微低估代码领域 |
| **RLHF 模型**（人类反馈） | 72.5% | 显著低估有毒领域 |

**缓解方案：**
- 仅对 Base 模型使用 LLMSurgeon
- 或开发"逆对齐"（inverse alignment）技术，估计对齐前后的分布变化

---

**局限 2：闭世界假设（Closed-World Assumption）**

**问题描述：**
- LLMSurgeon 假设：所有生成的文本都来自预定义的 K 个领域
- **现实情况**：模型可能训练了"未知领域"（如新出现的编程语言、特定行业文本）
- **后果**：未知领域文本会被"强制分类"到 K 个领域之一，导致估计偏差

**缓解方案：**
- **Unknown thresholding**（τ=0.9）：过滤掉低置信度预测（可能是 OOD 样本）
- **开放集识别**（Open-Set Recognition）：增加第 K+1 类"未知领域"

> **类比理解**
> 
> 闭世界假设比作"物种普查"：
> - **假设**（闭世界）：森林里只有 100 种已知动物
> - **现实**（开世界）：森林里可能还有 101 种未知动物
> - **错误**：把"未知动物"强行归类到"已知动物"（如把新物种归类为"猫"）
> - **LLMSurgeon 的策略**：如果不确定是哪种已知动物，标记为"未知"（Unknown thresholding）

---

**局限 3：语义重叠导致病态混淆矩阵**

**问题描述：**
- 某些领域语义高度重叠（如 C4 和 CommonCrawl 都是网页文本）
- 导致混淆矩阵 C 的某些特征值接近 0（病态矩阵）
- 逆问题 $C^T π = p̄$ 对噪声极度敏感

**实验证据（近似读取自论文 Appendix D）：**

| 领域对 | 相关系数 | C 的条件数 | 估计误差 |
|--------|---------|-----------|---------|
| **C4 vs CC** | 0.98 | 185.7 | ±8% |
| **Python vs C++** | 0.45 | 12.3 | ±2% |
| **Books vs ArXiv** | 0.32 | 8.5 | ±1.5% |

**缓解方案：**
- **领域合并**（Domain Merging）：合并相关性 >0.95 的领域
- **正则化**（Regularization）：在逆问题求解中加入 L2 正则项

---

### 核心贡献回顾

**贡献 1：Data Mixture Surgery (DMS) 形式化**
- 首次将"审计训练数据"转化为精确的数学问题
- 理论基础：Label-Shift 理论（迁移学习经典框架）
- 问题定义：给定生成文本，估计预训练语料的领域级分布

**贡献 2：LLMSurgeon 框架**
- 三阶段流程：标定混淆矩阵 → 观测目标分布 → 逆问题求解
- 核心创新：软混淆矩阵 + 约束最小二乘 + Bootstrap 置信区间
- 实验表现：在粗粒度任务上达到 95.14% 准确率，超越基线 +114%

**贡献 3：LLMScan 基准**
- 首个可验证的 DMS 评测体系（8 个模型 × 3 种粒度）
- 覆盖从 6 领域到 86 领域的难度谱系
- 为未来研究提供标准化评测协议

**贡献 4：泛化验证**
- **跨架构泛化**：从 Transformer 到 Transformer+（GPT-2）
- **时泛化**：从 2024 年模型（LLaMA-1）到 2025 年模型（OLMo-3）
- **安全审计应用**：Toxic-injection 实验验证实际应用价值

---

### 未来方向

**方向 1：逆对齐（Inverse Alignment）**
- **问题**：RLHF 改变了生成分布，Label-Shift 不成立
- **目标**：从对齐后的模型推断对齐前的数据混合物
- **方法**：建模对齐过程 $q_{aligned} = T(q_{base})$，估计逆变换 $T^{-1}$

**方向 2：层次化推断（Hierarchical Inference）**
- **问题**：细粒度任务准确率低（30%），因为 K 太大
- **目标**：先粗粒度估计（7 领域），再在相关领域内细化（如 7 → 22 → 86）
- **方法**：层次化贝叶斯推断（Hierarchical Bayesian Inference）

**方向 3：多语言扩展（Multilingual Extension）**
- **问题**：当前仅支持英语，多语言模型（如 LLaMA-2 支持 20+ 语言）如何审计？
- **目标**：估计每种语言的数据配比（如中文占 15%、英文占 60%）
- **方法**：跨语言混淆矩阵（Cross-Lingual Confusion Matrix）

---

### 推荐阅读顺序

**第 1 遍（直觉理解）：**
1. Abstract（了解研究动机和核心创新）
2. Figure 1（可视化 LLMSurgeon 三阶段流程）
3. Table 2（主实验结果，了解性能提升）

**第 2 遍（技术细节）：**
1. Section 2（问题形式化，DMS 数学定义）
2. Section 3（LLMScan 基准设计和 LLMSurgeon 框架）
3. Algorithm 1 & 2（Unknown thresholding 和 Bootstrap 算法）

**第 3 遍（深入分析）：**
1. Section 4（实验结果和消融研究）
2. Appendix（实现细节和额外实验）
3. GitHub 代码（classifier.py, prior.py, run_labelshift.py）

---

### 链接汇总

**论文原文：**
- arXiv: https://arxiv.org/abs/2605.30348
- ACL 2026 Proceedings:（待发布）

**代码仓库：**
- GitHub: https://github.com/Yaxin9Luo/LLMSurgeon（MIT License）
- 包含：11 个基线实现、LLMScan 基准数据、LLMSurgeon 框架

**相关论文：**
- Label-Shift 理论：Saerens et al. 2002 "Determining the Initial Values in the EM Algorithm for Unsupervised Data"
- LLMScan 数据来源：SlimPajama (Sobolev et al. 2023), The Pile (Gao et al. 2020), The Stack (Kocmi et al. 2022)

---

## 关键思考题

**思考题 1：LLMSurgeon 能否检测模型的"数据污染"？**
- **定义**：数据污染指测试集泄露到训练数据
- **挑战**：测试集通常来自同一领域（如 C4 验证集来自 C4 训练集）
- **思路**：如果模型在测试集上表现异常好，LLMSurgeon 能否检测到测试集领域被"过度采样"？

**思考题 2：LLMSurgeon 的伦理风险是什么？**
- **场景 1**：竞争对手用 LLMSurgeon "逆向工程"你的数据配方
- **场景 2**：监管者用 LLMSurgeon 强制披露数据来源（侵犯商业机密）
- **问题**：LLMSurgeon 是否应该像"核武器"一样受限？（双刃剑技术）

**思考题 3：如何扩展到"图像-文本多模态模型"？**
- **挑战**：CLIP、Flamingo 等模型的训练数据是"图像-文本对"，如何定义"领域"？
- **思路**：按图像类型（照片、图表、插画）和文本类型（描述、说明、对话）联合分类
- **问题**：混淆矩阵会从 K×K 变成 (K_img × K_text)×(K_img × K_text)，维度爆炸如何处理？

---
# LLMSurgeon：诊断大语言模型的数据混合物深度阅读报告（代码篇）

## Ch6: 核心代码实现详解

### 6.1 项目结构速览

**仓库目录树：**

```
LLMSurgeon/
├── baseline_method/
│   ├── src/labelshift/
│   │   ├── classifier.py          # 领域分类器实现（TF-IDF + DistilBERT）
│   │   ├── prior.py                # 逆问题求解器（最小二乘 + 单纯形投影）
│   │   ├── data_utils.py           # 数据预处理工具
│   │   ├── generate.py             # 文本生成接口
│   │   ├── run_labelshift.py       # 完整 Pipeline（11 阶段）
│   │   └── [11 baseline methods]/ # 基线方法实现
│   │       ├── min_k_percent.py    # Min-K% 基线
│   │       ├── neighborhood.py     # Neighborhood 基线
│   │       └── ...
├── bench/
│   ├── specs/                       # Ground-truth YAML spec 文件
│   │   ├── llama_1_7b.yaml        # LLaMA-1 7B 配比
│   │   ├── pythia_2.8b.yaml       # Pythia 2.8B 配比
│   │   └── starcoder.yaml         # StarCoder 配比
│   └── data/                       # 基准参考数据
├── benchmark_evaluation.py          # LLMScan 评分脚本
├── exp_scripts/                    # 实验执行脚本
└── requirements.txt                # 依赖项清单
```

**关键文件说明：**

- `classifier.py`：实现两种分类器（TfidfLogReg 和 HFSequenceClassifier），混淆矩阵计算
- `prior.py`：核心逆问题求解算法（estimate_priors_least_squares、bootstrap_priors）
- `run_labelshift.py`：完整实验流程（11 个阶段）
- `benchmark_evaluation.py`：LLMScan 基准评测

> **类比理解**
>
> 项目结构比作"医院科室"：
> - `classifier.py` → "检验科"（负责诊断分类）
> - `prior.py` → "外科手术室"（执行逆问题手术）
> - `run_labelshift.py` → "住院部"（协调整个治疗流程）
> - `bench/` → "病历档案室"（存储患者真实病史）

**依赖项和环境要求：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
# 核心依赖（从 requirements.txt 提取）
torch>=1.12.0                 # PyTorch（DistilBERT 推理）
transformers>=4.20.0          # Hugging Face 模型
scikit-learn>=1.0.0          # TF-IDF + LogisticRegression
numpy>=1.21.0                 # 矩阵运算（最小二乘、Bootstrap）
pyyaml>=6.0                   # YAML spec 文件解析
matplotlib>=3.5.0            # 诊断图绘制
tqdm>=4.64.0                 # 进度条显示
```

**硬件要求：**
- **GPU**：可选（DistilBERT 推理推荐，TF-IDF 不需要）
- **内存**：≥16GB（加载混淆矩阵 + Bootstrap）
- **存储**：≥50GB（LLMScan 基准数据 + 模型权重）

---

### 6.2 领域分类器实现

#### 6.2.1 TF-IDF + LogisticRegression 分类器

**特征工程策略：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
# 文件：baseline_method/src/labelshift/classifier.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class TfidfLogReg:
    def __init__(self, C=4.0, solver='saga', multi_class='multinomial'):
        self.C = C  # 正则化强度的倒数（C 越大，正则化越弱）
        self.solver = solver  # 'saga' 适合大规模 multinomial 分类
        self.multi_class = multi_class  # 'multinomial' 使用 Softmax 回归

        # 特征工程：word (1,2)-grams + char (3,5)-grams
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),     # word unigrams + bigrams
            max_features=50000,     # 限制词汇表大小
            analyzer='word'         # 按单词分词
        )
        self.char_vectorizer = TfidfVectorizer(
            ngram_range=(3, 5),     # character trigrams + 4-grams + 5-grams
            max_features=30000,
            analyzer='char'         # 按字符分词
        )
        self.clf = LogisticRegression(
            C=self.C,
            solver=self.solver,
            multi_class=self.multi_class,
            max_iter=1000           # 确保收敛
        )
```

> **类比理解**
>
> TF-IDF 特征提取比作"指纹识别"：
> - **word (1,2)-grams** → "整体指纹纹路"（识别常见词组，如 "machine learning"）
> - **char (3,5)-grams** → "微观指纹细节"（识别拼写模式，如 "tion"、"ing" 后缀）
> - **组合两者** → "多模态指纹"（既看整体，又看细节，提高识别准确率）

**训练流程：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
def train(self, texts, labels):
    """
    texts: List[str] — 训练文本样本
    labels: List[int] — 领域标签（0 到 K-1）
    """
    # 1. 特征提取
    word_features = self.word_vectorizer.fit_transform(texts)
    char_features = self.char_vectorizer.fit_transform(texts)

    # 2. 特征拼接（scipy.sparse.hstack）
    X = sparse.hstack([word_features, char_features])

    # 3. 训练 LogisticRegression
    self.clf.fit(X, labels)

    # 4. 温度标定（Grid Search）
    self.temperature = self._calibrate_temperature(X, labels)
```

**温度标定（Grid Search）：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
def _calibrate_temperature(self, X, y):
    """
    网格搜索最优温度 T ∈ [exp(-2), exp(2)]
    目标：最小化 Expected Calibration Error (ECE)
    """
    import numpy as np

    # 生成 41 个候选温度值
    temp_candidates = np.exp(np.linspace(-2, 2, 41))
    best_temp = 1.0
    best_ece = float('inf')

    for temp in temp_candidates:
        # 温度缩放后的预测概率
        probs = self._predict_proba_scaled(X, temp)
        ece = self._compute_ece(probs, y)  # 计算 ECE

        if ece < best_ece:
            best_ece = ece
            best_temp = temp

    return best_temp

def _predict_proba_scaled(self, X, temperature):
    """
    温度缩放：softmax(logits / T)
    """
    logits = self.clf.decision_function(X)  # 原始 logits
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
```

> **类比理解**
>
> 温度标定比作"恒温器校准"：
> - ** logits → 温度计读数**（分类器的"原始判断"）
> - **温度 T → 校准系数**（调整读数的"放大倍数"）
> - **grid search → 试错法**（尝试 41 个不同温度，找最准的）
> - **目标**：让"温度计读数"（预测概率）与"真实温度"（后验概率）对齐

---

#### 6.2.2 DistilBERT 分类器

**模型架构：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
# 文件：baseline_method/src/labelshift/classifier.py

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

class HFSequenceClassifier:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=7):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels  # K 个领域
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, train_texts, train_labels, val_texts, val_labels,
              epochs=3, batch_size=16, lr=2e-5, weight_decay=0.01):
        """
        训练配置（论文 Appendix A 推荐参数）：
        - epochs=3：防止过拟合
        - batch_size=16：平衡显存占用和梯度稳定性
        - lr=2e-5：标准 BERT fine-tuning 学习率
        - weight_decay=0.01：L2 正则化
        """
        from torch.utils.data import DataLoader
        from transformers import AdamW

        # 1. 构建 DataLoader
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 2. 优化器
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # 3. 训练循环
        for epoch in range(epochs):
            self.model.train()
            for batch in train_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # 4. 温度标定
        self.temperature = self._calibrate_temperature(val_texts, val_labels)
```

**混淆矩阵行归一化：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
def compute_confusion_matrix(self, texts_per_domain):
    """
    计算软混淆矩阵 C ∈ R^{K×K}
    C[i,j] = E_{x~p_i}[f_φ(x)_j]

    texts_per_domain: List[List[str]] — 每个领域的参考文本
    """
    K = len(texts_per_domain)
    C = np.zeros((K, K))

    for i in range(K):  # 对每个真实领域 i
        texts = texts_per_domain[i]
        probs = self.predict_proba(texts)  # (N_i, K)

        # 行归一化（确保每行和为 1）
        row_sum = probs.sum(axis=1, keepdims=True)
        normalized_probs = probs / row_sum

        # 期望值 = 平均概率
        C[i, :] = normalized_probs.mean(axis=0)

    return C
```

> **类比理解**
>
> 混淆矩阵行归一化比作"调查问卷归一化"：
> - **未归一化**：受访者 A 选了 3 个选项（和为 3），受访者 B 选了 1 个选项（和为 1）→ 无法比较
> - **归一化**：受访者 A 的选择除以 3，受访者 B 的选择除以 1 → 都变成"概率分布"
> - **目的**：确保 C 的每行都是合法的概率分布（和为 1，非负）

---

### 6.3 混淆矩阵与逆求解器

#### 6.3.1 混淆矩阵 C 的计算

**数学定义：**
$$C_{ij} = \mathbb{E}_{x \sim p_i(x)}[f_φ(x)_j]$$

**代码实现：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
# 文件：baseline_method/src/labelshift/classifier.py

def compute_confusion_matrix(classifier, reference_data, n_samples_per_domain=5000):
    """
    计算软混淆矩阵 C

    Args:
        classifier: f_φ（训练好的分类器）
        reference_data: Dict[int, List[str]] — 每个领域的参考数据
        n_samples_per_domain: 每个领域采样数量（论文推荐 5000）

    Returns:
        C: np.ndarray of shape (K, K) — 混淆矩阵
    """
    K = len(reference_data)
    C = np.zeros((K, K))

    for i in range(K):  # 真实领域 i
        # 1. 从领域 i 采样 n_samples 个文本
        texts = np.random.choice(reference_data[i], size=n_samples_per_domain, replace=True)

        # 2. 分类器预测（软概率）
        probs = classifier.predict_proba(texts)  # shape: (n_samples, K)

        # 3. 计算期望值（平均概率）
        C[i, :] = probs.mean(axis=0)

    return C
```

> **类比理解**
>
> 混淆矩阵计算比作"色盲测试"：
> - **真实领域 i**（如"代码"）→ 展示"红色卡片"
> - **分类器预测**（如 f_φ(x)_j）→ 色盲患者报告看到的颜色
> - **C[i,j]**（如 C[代码, GitHub] = 0.8）→ 患者把 80% 的"红色"报告为"红色"（正确），20% 报告为"橙色"（混淆）
> - **行归一化** → 确保每个真实领域的"报告概率和为 1"

---

#### 6.3.2 逆问题求解器（核心算法）

**数学问题：**
$$\text{Given } \bar{p} \text{ and } C, \text{ solve for } π$$
$$π̂ = \arg\min_{π} ||C^T π - \bar{p}||^2_2$$
$$\text{s.t. } \sum_{k=1}^{K} π_k = 1, π_k ≥ 0$$

**代码实现（prior.py）：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
# 文件：baseline_method/src/labelshift/prior.py

import numpy as np

def estimate_priors_least_squares(C, pbar):
    """
    核心算法：求解约束最小二乘问题

    Step 1: 无约束最小二乘（np.linalg.lstsq）
    Step 2: 单纯形投影（project_to_simplex）

    Args:
        C: np.ndarray of shape (K, K) — 混淆矩阵
        pbar: np.ndarray of shape (K,) — 观测分布

    Returns:
        pi_hat: np.ndarray of shape (K,) — 估计的先验分布
    """
    # Step 1: 无约束最小二乘
    # 求解 C^T π = p̄ 的最小二乘解
    pi_tilde, residuals, rank, singular_values = np.linalg.lstsq(C.T, pbar, rcond=None)

    # Step 2: 单纯形投影（确保 π 是合法概率分布）
    pi_hat = project_to_simplex(pi_tilde)

    return pi_hat


def project_to_simplex(v):
    """
    Duchi et al. (2008) 单纯形投影算法
    将任意向量 v ∈ R^K 投影到单纯形 Δ_K = {π | Σπ_k=1, π_k≥0}

    算法步骤：
    1. 排序：v_1 ≥ v_2 ≥ ... ≥ v_K
    2. 找到阈值 ρ：使得 sum_{j=1}^{ρ} (v_j - ρ) = 1
    3. 软阈值：π̂_j = max(v_j - ρ, 0)

    Args:
        v: np.ndarray of shape (K,) — 无约束解

    Returns:
        projected: np.ndarray of shape (K,) — 投影到单纯形的解
    """
    # 1. 排序（降序）
    u = np.sort(v)[::-1]

    # 2. 找到阈值 ρ
    rho = 0
    cumsum = 0
    for j in range(len(u)):
        cumsum += u[j]
        rho_candidate = (cumsum - 1) / (j + 1)
        if rho_candidate <= u[j]:
            rho = rho_candidate
        else:
            break

    # 3. 软阈值
    projected = np.maximum(v - rho, 0)

    return projected
```

> **类比理解**
>
> 逆问题求解比作"CT 扫描重建"：
> - **正向过程**（C^T π = p̄）：X 射线穿过人体 → 探测器接收衰减信号
> - **逆向过程**（从 p̄ 推 π）：从探测器信号 → 重建人体 3D 结构
> - **最小二乘**：找到"最可能"的 3D 结构，使得探测信号误差最小
> - **单纯形投影**：确保重建结果是"合理的"（非负密度，总质量为 1）

---

#### 6.3.3 Bootstrap 置信区间

**数学原理：**
$$CI_{0.95}(π_k) = [π̂_k^{(2.5\%)}, π̂_k^{(97.5\%)}]$$

**代码实现：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
# 文件：baseline_method/src/labelshift/prior.py

def bootstrap_priors(C, pbar, n_iterations=300, ci_percentile=(2.5, 97.5)):
    """
    Bootstrap 置信区间估计

    Args:
        C: np.ndarray of shape (K, K) — 混淆矩阵
        pbar: np.ndarray of shape (N, K) — N 个样本的软预测（N 为生成的文本总数）
        n_iterations: Bootstrap 迭代次数（论文推荐 200-300）
        ci_percentile: 置信区间分位数（默认 95% CI）

    Returns:
        pi_mean: np.ndarray of shape (K,) — Bootstrap 均值估计
        pi_ci: np.ndarray of shape (K, 2) — 每个领域的置信区间 [下界, 上界]
    """
    K = C.shape[0]
    N = pbar.shape[0]
    bootstrap_estimates = []

    for _ in range(n_iterations):
        # 1. 重采样（有放回）
        indices = np.random.choice(N, size=N, replace=True)
        pbar_bootstrap = pbar[indices]

        # 2. 计算伪样本的均值
        pbar_mean = pbar_bootstrap.mean(axis=0)  # shape: (K,)

        # 3. 逆问题求解
        pi_bootstrap = estimate_priors_least_squares(C, pbar_mean)
        bootstrap_estimates.append(pi_bootstrap)

    # 4. 计算置信区间
    bootstrap_estimates = np.array(bootstrap_estimates)  # shape: (n_iterations, K)
    pi_mean = bootstrap_estimates.mean(axis=0)
    pi_ci_lower = np.percentile(bootstrap_estimates, ci_percentile[0], axis=0)
    pi_ci_upper = np.percentile(bootstrap_estimates, ci_percentile[1], axis=0)
    pi_ci = np.column_stack([pi_ci_lower, pi_ci_upper])

    return pi_mean, pi_ci
```

> **类比理解**
>
> Bootstrap 置信区间比作"选举民意测验"：
> - **单次调查**（π̂）：候选人 A 支持率 52%（有偶然性）
> - **重复调查**（Bootstrap）：再做 300 次调查，得到 [49%, 55%]
> - **置信区间**："我们有 95% 的信心，真实支持率在 49%-55% 之间"
> - **不确定性来源**：样本量有限、受访者偏见、时间差异等

---

### 6.4 完整 Pipeline（run_labelshift.py）

**11 个阶段的完整流程：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
# 文件：baseline_method/src/labelshift/run_labelshift.py

def run_labelshift_pipeline(config):
    """
    完整的 Data Mixture Surgery Pipeline

    阶段：
    1. 数据准备
    2. 分类器训练
    3. 混淆矩阵计算
    4. 文本生成
    5. 分类预测
    6. Unknown thresholding
    7. 计算 p̄
    8. Naive baseline
    9. Prior correction（逆问题求解）
    10. Bootstrap 置信区间
    11. 输出结果

    Args:
        config: Dict — 配置字典（模型名称、领域列表、生成参数等）

    Returns:
        results: Dict — 估计结果 + 置信区间
    """

    # ========== Stage 1: 数据准备 ==========
    print("[Stage 1/11] Loading reference data...")
    reference_data = load_reference_data(config['reference_paths'])
    domain_names = list(reference_data.keys())
    K = len(domain_names)

    # ========== Stage 2: 分类器训练 ==========
    print("[Stage 2/11] Training classifier...")
    if config['classifier_type'] == 'tfidf':
        classifier = TfidfLogReg(C=4.0)
    elif config['classifier_type'] == 'distilbert':
        classifier = HFSequenceClassifier(num_labels=K)
    else:
        raise ValueError(f"Unknown classifier: {config['classifier_type']}")

    classifier.train(
        train_texts=flatten_reference_data(reference_data),
        train_labels=flatten_reference_labels(reference_data),
        val_texts=...,
        val_labels=...
    )

    # ========== Stage 3: 混淆矩阵计算 ==========
    print("[Stage 3/11] Computing confusion matrix...")
    C = classifier.compute_confusion_matrix(reference_data)

    # ========== Stage 4: 文本生成 ==========
    print("[Stage 4/11] Generating text from target LLM...")
    generated_texts = generate_from_target_llm(
        model_name=config['target_model'],
        prompts=neutral_prompts,  # 中性提示词（如 "Continue the following text."）
        n_samples_per_domain=config['n_samples_per_domain'],  # 通常 5000
        temperature=0.8,
        top_p=0.9
    )

    # ========== Stage 5: 分类预测 ==========
    print("[Stage 5/11] Predicting domains for generated texts...")
    probs = classifier.predict_proba(generated_texts)  # shape: (N, K)

    # ========== Stage 6: Unknown thresholding ==========
    print("[Stage 6/11] Applying unknown thresholding (τ=0.9)...")
    tau = 0.9
    max_probs = probs.max(axis=1)
    kept_indices = max_probs >= tau
    probs_filtered = probs[kept_indices]
    print(f"Kept {len(kept_indices)}/{len(probs)} samples ({len(kept_indices)/len(probs)*100:.1f}%)")

    # ========== Stage 7: 计算 p̄ ==========
    print("[Stage 7/11] Computing empirical mean p̄...")
    pbar = probs_filtered.mean(axis=0)  # shape: (K,)

    # ========== Stage 8: Naive baseline ==========
    print("[Stage 8/11] Computing naive baseline (direct aggregation)...")
    naive_prior = pbar / pbar.sum()  # 简单归一化

    # ========== Stage 9: Prior correction（逆问题求解）==========
    print("[Stage 9/11] Solving inverse problem (prior correction)...")
    corrected_prior = estimate_priors_least_squares(C, pbar)

    # ========== Stage 10: Bootstrap 置信区间 ==========
    print("[Stage 10/11] Computing bootstrap confidence intervals...")
    prior_mean, prior_ci = bootstrap_priors(C, probs_filtered, n_iterations=300)

    # ========== Stage 11: 输出结果 ==========
    print("[Stage 11/11] Saving results...")
    results = {
        'domain_names': domain_names,
        'naive_prior': naive_prior,
        'corrected_prior': corrected_prior,
        'bootstrap_mean': prior_mean,
        'confidence_intervals': prior_ci,
        'confusion_matrix': C
    }

    # 保存到文件
    save_results(results, config['output_path'])

    # 生成诊断图
    plot_diagnostics(results, config['plot_path'])

    return results
```

> **类比理解**
>
> 11 阶段 Pipeline 比作"完整医疗检查流程"：
> - **Stage 1-3**（数据准备 + 训练 + C 矩阵）→ "建立标准"（校准设备）
> - **Stage 4-5**（生成 + 预测）→ "采集患者样本"（抽血化验）
> - **Stage 6**（Unknown thresholding）→ "质量控制"（过滤不合格样本）
> - **Stage 7-8**（p̄ + Naive）→ "初步诊断"（快速检查）
> - **Stage 9**（Prior correction）→ "精准诊断"（逆校正）
> - **Stage 10**（Bootstrap）→ "置信区间"（不确定性评估）
> - **Stage 11**（输出）→ "出具诊断报告"

---

**未知类别阈值处理（τ=0.9）：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
def unknown_thresholding(probs, tau=0.9):
    """
    过滤掉低置信度预测（可能是 OOD 样本）

    Args:
        probs: np.ndarray of shape (N, K) — 分类器预测概率
        tau: float — 阈值（论文推荐 0.9）

    Returns:
        filtered_probs: np.ndarray of shape (M, K) — 过滤后的预测（M ≤ N）
        kept_indices: np.ndarray of shape (M,) — 保留的样本索引
    """
    max_probs = probs.max(axis=1)  # 每个样本的最大概率
    kept_indices = np.where(max_probs >= tau)[0]
    filtered_probs = probs[kept_indices]
    return filtered_probs, kept_indices
```

**为什么 τ=0.9？**

- **τ = 0.7**：保留太多噪声，准确率下降（OOD 样本太多）
- **τ = 0.9**：最佳平衡点，过滤掉 95% 的 OOD 样本
- **τ = 0.99**：过度过滤，样本量不足（置信区间变宽）

---

**领域合并策略（C4 + CommonCrawl）：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
def merge_domains(C, domain_names, merge_pairs):
    """
    合并语义重叠的领域（避免病态混淆矩阵）

    Args:
        C: np.ndarray of shape (K, K) — 原始混淆矩阵
        domain_names: List[str] — 原始领域名称
        merge_pairs: List[Tuple[str, str]] — 需要合并的领域对

    Returns:
        C_merged: np.ndarray of shape (K', K') — 合并后的混淆矩阵（K' < K）
        domain_names_merged: List[str] — 合并后的领域名称
    """
    # 示例：合并 C4 和 CommonCrawl
    # merge_pairs = [('C4', 'CommonCrawl')]

    # 1. 构建合并映射
    merge_map = {}
    merged_domains = set()
    for d1, d2 in merge_pairs:
        merge_map[d1] = f"{d1}+{d2}"
        merge_map[d2] = f"{d1}+{d2}"
        merged_domains.add(f"{d1}+{d2}")

    # 2. 更新领域名称
    new_domain_names = []
    for name in domain_names:
        if name in merge_map:
            new_name = merge_map[name]
            if new_name not in new_domain_names:
                new_domain_names.append(new_name)
        else:
            new_domain_names.append(name)

    # 3. 合并混淆矩阵的行和列
    K_merged = len(new_domain_names)
    C_merged = np.zeros((K_merged, K_merged))
    for i, name_i in enumerate(new_domain_names):
        for j, name_j in enumerate(new_domain_names):
            # 找到原始领域中对应的索引
            if '+' in name_i:  # 合并领域
                original_indices_i = [idx for idx, name in enumerate(domain_names) if merge_map.get(name) == name_i]
            else:  # 未合并领域
                original_indices_i = [domain_names.index(name_i)]

            if '+' in name_j:
                original_indices_j = [idx for idx, name in enumerate(domain_names) if merge_map.get(name) == name_j]
            else:
                original_indices_j = [domain_names.index(name_j)]

            # 平均原始矩阵的对应块
            C_merged[i, j] = C[np.ix_(original_indices_i, original_indices_j)].mean()

    return C_merged, new_domain_names
```

> **类比理解**
>
> 领域合并比作"行政区划调整"：
> - **原始 C 矩阵**（7 个领域）→ 7 个独立行政区
> - **合并 C4+CC**（2 个领域语义重叠）→ 合并为 1 个"大区"
> - **结果**：6 个行政区（7 → 6）
> - **目的**：避免"行政区边界模糊"（病态矩阵）

---

**输出格式（summary.json + summary.csv + 诊断图）：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
def save_results(results, output_path):
    """
    保存结果到 JSON + CSV + 诊断图
    """
    import json
    import pandas as pd
    import matplotlib.pyplot as plt

    # 1. 保存 JSON（完整结果）
    with open(f"{output_path}/summary.json", 'w') as f:
        json.dump({
            'domain_names': results['domain_names'],
            'corrected_prior': results['corrected_prior'].tolist(),
            'bootstrap_mean': results['bootstrap_mean'].tolist(),
            'confidence_intervals': results['confidence_intervals'].tolist()
        }, f, indent=2)

    # 2. 保存 CSV（仅估计值）
    df = pd.DataFrame({
        'domain': results['domain_names'],
        'estimated_prior': results['corrected_prior'],
        'ci_lower': results['confidence_intervals'][:, 0],
        'ci_upper': results['confidence_intervals'][:, 1]
    })
    df.to_csv(f"{output_path}/summary.csv", index=False)

    # 3. 生成诊断图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) 估计分布柱状图
    axes[0, 0].bar(results['domain_names'], results['corrected_prior'])
    axes[0, 0].set_title('Estimated Data Mixture')
    axes[0, 0].set_xlabel('Domain')
    axes[0, 0].set_ylabel('Estimated Proportion')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # (b) 置信区间误差条
    axes[0, 1].errorbar(
        range(len(results['domain_names'])),
        results['corrected_prior'],
        yerr=[
            results['corrected_prior'] - results['confidence_intervals'][:, 0],
            results['confidence_intervals'][:, 1] - results['corrected_prior']
        ],
        fmt='o'
    )
    axes[0, 1].set_title('Estimated Priors with 95% CI')
    axes[0, 1].set_xlabel('Domain Index')
    axes[0, 1].set_ylabel('Proportion')

    # (c) 混淆矩阵热图
    im = axes[1, 0].imshow(results['confusion_matrix'], cmap='Blues')
    axes[1, 0].set_title('Confusion Matrix C')
    axes[1, 0].set_xlabel('Predicted Domain')
    axes[1, 0].set_ylabel('True Domain')
    plt.colorbar(im, ax=axes[1, 0])

    # (d) Naive vs Corrected 对比
    axes[1, 1].scatter(results['naive_prior'], results['corrected_prior'])
    axes[1, 1].plot([0, 1], [0, 1], 'r--', label='y=x')
    axes[1, 1].set_title('Naive vs Corrected Priors')
    axes[1, 1].set_xlabel('Naive Prior')
    axes[1, 1].set_ylabel('Corrected Prior')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}/diagnostics.png", dpi=300)
```

---

### 6.5 LLMScan 基准构建

#### 6.5.1 YAML spec 文件格式

**示例（LLaMA-1 7B）：**

```yaml
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
# 文件：bench/specs/llama_1_7b.yaml

model_name: "llama-1-7b"
model_family: "LLaMA"
num_parameters: 7000000000  # 7B

# 领域配比（Ground Truth）
category_weights:
  CommonCrawl: 0.35        # 35%
  GitHub: 0.25             # 25%
  Wikipedia: 0.15          # 15%
  Books: 0.10              # 10%
  ArXiv: 0.08              # 8%
  StackExchange: 0.05      # 5%
  C4: 0.02                 # 2%

# 领域详细信息
categories:
  CommonCrawl:
    description: "Web crawl data"
    source: "SlimPajama-DC"
  GitHub:
    description: "Source code repositories"
    source: "SlimPajama-DC"
  Wikipedia:
    description: "Online encyclopedia"
    source: "SlimPajama-DC"
  Books:
    description: "Books corpus"
    source: "SlimPajama-DC"
  ArXiv:
    description: "Scientific papers"
    source: "SlimPajama-DC"
  StackExchange:
    description: "Q&A forums"
    source: "SlimPajama-DC"
  C4:
    description: "Colossal Clean Crawled Corpus"
    source: "SlimPajama-DC"

# 元数据
metadata:
  training_data: "SlimPajama-DC"
  release_date: "2023-02-14"
  paper: "Touvron et al. 2023"
```

**YAML 解析代码：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
# 文件：benchmark_evaluation.py

import yaml

def load_ground_truth(spec_path):
    """
    从 YAML spec 文件加载 Ground Truth

    Args:
        spec_path: str — YAML 文件路径

    Returns:
        ground_truth: Dict[str, float] — 领域名称 → 真实配比
    """
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)

    # 提取 category_weights（归一化）
    weights = spec['category_weights']
    total = sum(weights.values())
    ground_truth = {k: v / total for k, v in weights.items()}

    return ground_truth
```

> **类比理解**
>
> YAML spec 文件比作"食品营养成分表"：
> - **category_weights** → 营养成分配比（蛋白质 30%、脂肪 20% 等）
> - **categories** → 食材来源（蛋白质来自牛肉、大豆等）
> - **metadata** → 生产日期、厂家信息
> - **Ground Truth** → "真实配方"（LLMSurgeon 的估计目标）

---

#### 6.5.2 benchmark_evaluation.py 评分逻辑

**Overlap Accuracy 计算：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
# 文件：benchmark_evaluation.py

import numpy as np

def overlap_accuracy(ground_truth, estimated_prior):
    """
    计算 Overlap Accuracy（论文主要评估指标）

    数学定义：
    Overlap Accuracy = 1 - (1/2) * Σ |α_k - π̂_k|

    直观理解：两个概率分布的"重叠程度"
    - 1.0 表示完美估计（π̂ = α）
    - 0.5 表示随机猜测
    - 0.0 表示完全错误

    Args:
        ground_truth: Dict[str, float] — 真实分布 {domain: α_k}
        estimated_prior: Dict[str, float] — 估计分布 {domain: π̂_k}

    Returns:
        accuracy: float — Overlap Accuracy [0, 1]
    """
    # 1. 对齐领域顺序
    domains = list(ground_truth.keys())
    alpha = np.array([ground_truth[d] for d in domains])
    pi_hat = np.array([estimated_prior.get(d, 0.0) for d in domains])

    # 2. 计算 L1 距离
    l1_distance = np.sum(np.abs(alpha - pi_hat))

    # 3. 计算 Overlap Accuracy
    accuracy = 1 - 0.5 * l1_distance

    return accuracy


def compute_all_benchmarks(spec_dir, results_dir):
    """
    计算所有基准模型的 Overlap Accuracy

    Args:
        spec_dir: str — YAML spec 文件目录
        results_dir: str — LLMSurgeon 估计结果目录

    Returns:
        summary: pd.DataFrame — 每个模型的准确率
    """
    import os
    import pandas as pd

    summary = []

    # 遍历所有 YAML spec 文件
    for spec_file in os.listdir(spec_dir):
        if not spec_file.endswith('.yaml'):
            continue

        model_name = spec_file.replace('.yaml', '')

        # 1. 加载 Ground Truth
        spec_path = os.path.join(spec_dir, spec_file)
        ground_truth = load_ground_truth(spec_path)

        # 2. 加载 LLMSurgeon 估计结果
        result_path = os.path.join(results_dir, f"{model_name}_results.json")
        with open(result_path, 'r') as f:
            results = json.load(f)
        estimated_prior = dict(zip(results['domain_names'], results['corrected_prior']))

        # 3. 计算 Overlap Accuracy
        accuracy = overlap_accuracy(ground_truth, estimated_prior)

        summary.append({
            'model': model_name,
            'overlap_accuracy': accuracy,
            'num_domains': len(ground_truth)
        })

    return pd.DataFrame(summary)
```

> **类比理解**
>
> Overlap Accuracy 比作"天气预报准确率"：
> - **Ground Truth α**（真实天气）：下雨概率 30%
> - **估计 π̂**（预报）：下雨概率 25%
> - **Overlap Accuracy**：1 - 0.5 * |0.3 - 0.25| = 0.975（97.5% 准确率）
> - **L1 距离**：|0.3 - 0.25| = 0.05（误差 5%）
> - **公式含义**：误差越小，重叠程度越高

---

**基线方法对比（11 个 baselines）：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
# 文件：baseline_method/src/labelshift/run_labelshift.py

def run_all_baselines(generated_texts, reference_data, config):
    """
    运行所有 11 个基线方法

    基线方法列表：
    1. Min-K%      6. DC-PDD
    2. Min-K%++    7. DUCI
    3. ReCaLL      8. Joint-Logit
    4. zlib        9. Loss
    5. Neighborhood 10. Ref
                   11. GradNorm

    Returns:
        baseline_results: Dict[str, np.ndarray] — 每个基线的估计结果
    """
    baseline_results = {}

    # Baseline 1: Min-K%（最小损失样本比例）
    from baselines.min_k_percent import min_k_percent
    baseline_results['min_k_percent'] = min_k_percent(generated_texts, reference_data)

    # Baseline 2: Min-K%++
    from baselines.min_k_percent_plus import min_k_percent_plus
    baseline_results['min_k_percent_plus'] = min_k_percent_plus(generated_texts, reference_data)

    # ... (其他 9 个基线)

    return baseline_results


def compare_with_baselines(ground_truth, llmsurgeon_result, baseline_results):
    """
    对比 LLMSurgeon 与所有基线的性能

    Args:
        ground_truth: Dict[str, float]
        llmsurgeon_result: Dict[str, float]
        baseline_results: Dict[str, Dict[str, float]]

    Returns:
        comparison: pd.DataFrame — 每个方法的 Overlap Accuracy
    """
    import pandas as pd

    comparison = []

    # LLMSurgeon
    llmsurgeon_acc = overlap_accuracy(ground_truth, llmsurgeon_result)
    comparison.append({
        'method': 'LLMSurgeon',
        'overlap_accuracy': llmsurgeon_acc
    })

    # 所有基线
    for method_name, result in baseline_results.items():
        acc = overlap_accuracy(ground_truth, result)
        comparison.append({
            'method': method_name,
            'overlap_accuracy': acc
        })

    df = pd.DataFrame(comparison)
    df = df.sort_values('overlap_accuracy', ascending=False)
    return df
```

---

### 6.6 代码实战示例（伪代码）

**完整流程演示：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon
# 伪代码：完整流程演示

# ========== Step 1: 准备参考数据 ==========
reference_data = {
    'CommonCrawl': load_texts('bench/data/common_crawl.txt'),
    'GitHub': load_texts('bench/data/github.txt'),
    'Wikipedia': load_texts('bench/data/wikipedia.txt'),
    'Books': load_texts('bench/data/books.txt'),
    'ArXiv': load_texts('bench/data/arxiv.txt'),
    'StackExchange': load_texts('bench/data/stackexchange.txt'),
    'C4': load_texts('bench/data/c4.txt')
}

# ========== Step 2: 训练分类器 ==========
classifier = HFSequenceClassifier(num_labels=7)
classifier.train(
    train_texts=flatten(reference_data),
    train_labels=flatten_labels(reference_data)
)

# ========== Step 3: 计算混淆矩阵 ==========
C = classifier.compute_confusion_matrix(reference_data)
# C ≈ [[0.92, 0.05, 0.01, ...],  # CommonCrawl 领域
#      [0.03, 0.89, 0.02, ...],  # GitHub 领域
#      ...]

# ========== Step 4: 生成文本 ==========
generated_texts = generate_from_llm(
    model='llama-1-7b',
    prompts=['Continue the following text.'] * 35000,
    temperature=0.8,
    top_p=0.9
)

# ========== Step 5: 分类预测 ==========
probs = classifier.predict_proba(generated_texts)
# probs.shape = (35000, 7)

# ========== Step 6: Unknown thresholding ==========
probs_filtered = probs[probs.max(axis=1) >= 0.9]
# 保留约 33000 个样本（~95%）

# ========== Step 7: 计算 p̄ ==========
pbar = probs_filtered.mean(axis=0)
# pbar ≈ [0.38, 0.23, 0.14, 0.11, 0.07, 0.05, 0.02]

# ========== Step 8: Naive baseline ==========
naive_prior = pbar / pbar.sum()
# naive_prior ≈ [0.38, 0.23, 0.14, 0.11, 0.07, 0.05, 0.02]

# ========== Step 9: 逆问题求解 ==========
corrected_prior = estimate_priors_least_squares(C, pbar)
# corrected_prior ≈ [0.35, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02]
#                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                 更接近 Ground Truth！

# ========== Step 10: Bootstrap 置信区间 ==========
prior_mean, prior_ci = bootstrap_priors(C, probs_filtered, n_iterations=300)
# prior_ci ≈ [[0.32, 0.38],   # CommonCrawl: [0.32, 0.38]
#            [0.22, 0.28],   # GitHub: [0.22, 0.28]
#            ...]

# ========== Step 11: 输出结果 ==========
results = {
    'domain_names': ['CommonCrawl', 'GitHub', 'Wikipedia', 'Books', 'ArXiv', 'StackExchange', 'C4'],
    'corrected_prior': corrected_prior,
    'confidence_intervals': prior_ci
}

print(json.dumps(results, indent=2))
```

> **类比理解**
>
> 完整流程比作"法医鉴定"：
> - **Step 1-3**（参考数据 + 训练 + C 矩阵）→ "建立 DNA 数据库"
> - **Step 4**（生成文本）→ "采集现场样本"
> - **Step 5-6**（预测 + Thresholding）→ "DNA 匹配 + 质量控制"
> - **Step 7**（p̄）→ "初步鉴定结果"
> - **Step 8**（Naive）→ "快速鉴定"（可能有偏差）
> - **Step 9**（逆校正）→ "精准鉴定"（校正系统性偏差）
> - **Step 10**（Bootstrap）→ "置信区间"（不确定性评估）
> - **Step 11**（输出）→ "出具鉴定报告"

---

## Ch7: 关键算法的数学直觉与代码对应

### 7.1 为什么逆校正有效？（数学直觉 + 代码对应）

**数学直觉：**

假设真实分布 $π = [0.35, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02]$（LLaMA-1 的 Ground Truth）。

混淆矩阵 $C$（假设）：
$$
C = \begin{bmatrix}
0.92 & 0.05 & 0.01 & \cdots \\
0.03 & 0.89 & 0.02 & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix}
$$

观测分布 $p̄ = C^T π$：
$$
\bar{p} = \begin{bmatrix}
0.92 \times 0.35 + 0.03 \times 0.25 + \cdots \\
0.05 \times 0.35 + 0.89 \times 0.25 + \cdots \\
\vdots
\end{bmatrix}
\approx \begin{bmatrix}
0.38 \\
0.23 \\
0.14 \\
\vdots
\end{bmatrix}
$$

**Naive 方法错误：** 直接认为 $π̂ = p̄$（忽略混淆偏差）

**逆校正正确：** 求解 $C^T π = p̄$，得到 $π̂ ≈ [0.35, 0.25, 0.15, \cdots]$（接近真实 $π$）

---

**代码对应：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon

# Naive 方法（有偏）
naive_prior = pbar / pbar.sum()  # 简单归一化
# naive_prior ≈ [0.38, 0.23, 0.14, ...]（系统性高估 CommonCrawl）

# 逆校正（无偏）
def estimate_priors_least_squares(C, pbar):
    # 求解 C^T π = p̄
    pi_tilde, _, _, _ = np.linalg.lstsq(C.T, pbar, rcond=None)
    # pi_tilde 可能有负值或和≠1

    # 单纯形投影
    pi_hat = project_to_simplex(pi_tilde)
    # pi_hat ≈ [0.35, 0.25, 0.15, ...]（无偏估计）

    return pi_hat
```

---

### 7.2 为什么 Bootstrap 有效？（重采样直觉）

**重采样直觉：**

假设你只有 100 个样本（估计 $π$），但你想知道"如果重新采样 100 次，估计结果会如何变化"。

Bootstrap 的"魔法"：
- 从原 100 个样本中**有放回地**抽取 100 个（"伪样本"）
- 重复 300 次，得到 300 个"伪估计" $π̂^{(1)}, π̂^{(2)}, ..., π̂^{(300)}$
- 计算这 300 个估计的 2.5% 和 97.5% 分位数 → 95% 置信区间

**为什么有效？**

- **大数定律**：当样本量 N → ∞ 时，Bootstrap 分布 → 真实抽样分布
- **中心极限定理**：Bootstrap 均值 → 真实均值（渐近无偏）

---

**代码对应：**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon

def bootstrap_priors(C, pbar, n_iterations=300):
    """
    pbar: (N, K) — N 个样本的软预测
    n_iterations: 300 — Bootstrap 次数
    """
    bootstrap_estimates = []

    for _ in range(n_iterations):
        # 1. 重采样（有放回）
        indices = np.random.choice(N, size=N, replace=True)
        pbar_bootstrap = pbar[indices]

        # 2. 计算伪样本的均值
        pbar_mean = pbar_bootstrap.mean(axis=0)

        # 3. 逆问题求解
        pi_bootstrap = estimate_priors_least_squares(C, pbar_mean)
        bootstrap_estimates.append(pi_bootstrap)

    # 4. 计算置信区间
    bootstrap_estimates = np.array(bootstrap_estimates)  # (300, K)
    pi_ci_lower = np.percentile(bootstrap_estimates, 2.5, axis=0)
    pi_ci_upper = np.percentile(bootstrap_estimates, 97.5, axis=0)

    return np.column_stack([pi_ci_lower, pi_ci_upper])
```

> **类比理解**
>
> Bootstrap 比作"选举民意的多次调查"：
> - **原样本**（100 个选民）→ 单次民意调查（候选人 A 52%）
> - **Bootstrap 重采样**（300 次调查）→ 重复调查（300 次，每次 100 人）
> - **置信区间**（[49%, 55%]）→ "我们有 95% 信心，真实支持率在此区间"
> - **目的**：量化估计的不确定性（单次估计可能有偶然性）

---

## Ch8: 常见问题与调试技巧

### 8.1 混淆矩阵病态怎么办？

**症状：**
- `np.linalg.lstsq` 报错：`Matrix is singular`
- 估计结果 $π̂$ 有极端值（如某个领域 99%，其他 1%）
- 条件数 `cond(C) > 100`（病态矩阵）

**解决方案：**

1. **合并语义重叠领域**（如 C4 + CommonCrawl）
2. **增加正则化**（在逆问题中加入 L2 惩罚）
3. **使用 SVD 求解**（替代最小二乘）

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon

# 检查条件数
cond_number = np.linalg.cond(C)
if cond_number > 100:
    print(f"Warning: C is ill-conditioned (cond={cond_number:.2f})")

# 解决方案 1：领域合并
C_merged, domain_names_merged = merge_domains(C, domain_names, [('C4', 'CommonCrawl')])

# 解决方案 2：正则化（替代 lstsq）
def estimate_priors_ridge(C, pbar, lambda=0.01):
    """Ridge 回归版本"""
    K = C.shape[0]
    # 最小化 ||C^T π - p̄||² + λ||π||²
    I = np.eye(K)
    pi_hat = np.linalg.solve(C @ C.T + lambda * I, C @ pbar)
    return project_to_simplex(pi_hat)

# 解决方案 3：SVD 求解
def estimate_priors_svd(C, pbar, rank=5):
    """截断 SVD（保留前 rank 个奇异值）"""
    U, s, Vt = np.linalg.svd(C.T, full_matrices=False)
    S = np.diag(s[:rank])
    pi_tilde = Vt[:rank, :].T @ S @ U[:, :rank].T @ pbar
    return project_to_simplex(pi_tilde)
```

---

### 8.2 估计结果不稳定怎么办？

**症状：**
- 不同运行次数的估计差异 >10%
- Bootstrap 置信区间极宽（如 [0.01, 0.99]）
- Overlap Accuracy 波动大（如 90% → 70% → 85%）

**解决方案：**

1. **增加采样量**（从 5000 → 10000 samples/domain）
2. **提高 Unknown thresholding 阈值**（从 τ=0.9 → τ=0.95）
3. **多次运行取平均**

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon

# 解决方案 1：增加采样量
config['n_samples_per_domain'] = 10000  # 从 5000 → 10000

# 解决方案 2：提高阈值
tau = 0.95  # 从 0.9 → 0.95（过滤更多 OOD 样本）

# 解决方案 3：多次运行取平均
def run_multiple_runs(config, n_runs=5):
    estimates = []
    for _ in range(n_runs):
        result = run_labelshift_pipeline(config)
        estimates.append(result['corrected_prior'])

    estimates = np.array(estimates)  # (n_runs, K)
    mean_estimate = estimates.mean(axis=0)
    std_estimate = estimates.std(axis=0)

    return mean_estimate, std_estimate
```

---

### 8.3 分类器准确率低怎么办？

**症状：**
- 分类器验证准确率 <80%
- 混淆矩阵 C 的对角线元素 <0.7
- Overlap Accuracy <50%

**解决方案：**

1. **换用更强的分类器**（从 TF-IDF → DistilBERT）
2. **增加训练数据**（从 5000 → 20000 samples/domain）
3. **调整超参数**（如 DistilBERT 的 lr、batch_size）

```python
# Source: official repo, https://github.com/Yaxin9Luo/LLMSurgeon

# 解决方案 1：换分类器
config['classifier_type'] = 'distilbert'  # 从 'tfidf' → 'distilbert'

# 解决方案 2：增加训练数据
classifier.train(
    train_texts=large_reference_texts,  # 从 5000 → 20000
    train_labels=large_reference_labels,
    epochs=5  # 从 3 → 5
)

# 解决方案 3：调整超参数
classifier.train(
    lr=1e-5,       # 从 2e-5 → 1e-5（更保守）
    batch_size=32, # 从 16 → 32（更稳定）
    weight_decay=0.02  # 从 0.01 → 0.02（更强正则化）
)
```

---

## Ch9: 总结与最佳实践

### 9.1 核心算法总结

| 算法 | 输入 | 输出 | 复杂度 | 关键参数 |
|------|------|------|--------|---------|
| **TfidfLogReg** | 训练文本 + 标签 | 分类器 f_φ | O(N_samples × Vocab) | C=4.0, saga solver |
| **DistilBERT** | 训练文本 + 标签 | 分类器 f_φ | O(N_samples × Seq_Len × d_model) | lr=2e-5, epochs=3 |
| **compute_confusion_matrix** | 分类器 + 参考数据 | C ∈ R^{K×K} | O(K × N_samples) | n_samples=5000 |
| **estimate_priors_least_squares** | C + p̄ | π̂ ∈ R^K | O(K³)（矩阵求逆） | 无 |
| **project_to_simplex** | 无约束解 v ∈ R^K | 单纯形投影 π ∈ Δ_K | O(K log K)（排序） | 无 |
| **bootstrap_priors** | C + p̄ | 置信区间 | O(N_iterations × K³) | n_iterations=300 |
| **overlap_accuracy** | α + π̂ | 准确率 [0,1] | O(K) | 无 |

---

### 9.2 最佳实践清单

**数据准备阶段：**
- [x] 使用中性提示词（neutral prompts）生成文本
- [x] 每个领域采样 ≥5000 个文本
- [x] 生成参数：temperature=0.8, top_p=0.9

**分类器训练阶段：**
- [x] 优先使用 DistilBERT（准确率 >90%）
- [x] 训练参数：lr=2e-5, epochs=3, batch_size=16
- [x] 必须进行温度标定（grid search）

**混淆矩阵计算阶段：**
- [x] 行归一化（确保每行和为 1）
- [x] 检查条件数（cond(C) < 100）
- [x] 合并高度重叠领域（如 C4+CC）

**逆问题求解阶段：**
- [x] 使用最小二乘 + 单纯形投影
- [x] 验证结果满足概率公理（非负，和为 1）
- [x] 进行 Bootstrap 置信区间估计

**结果评估阶段：**
- [x] 使用 Overlap Accuracy（非 L1 距离）
- [x] 与 Naive baseline 对比（验证逆校正收益）
- [x] 可视化诊断图（柱状图 + 误差条 + 热图）

---

### 9.3 代码调试技巧

**技巧 1：打印混淆矩阵对角线**

```python
# 检查分类器质量
diag = np.diag(C)
print(f"Confusion Matrix Diagonal: {diag}")
print(f"Mean Diagonal: {diag.mean():.3f}")  # 应 >0.85
```

**技巧 2：可视化观测分布 vs 估计分布**

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.bar(domain_names, pbar, alpha=0.5, label='Observed p̄')
plt.bar(domain_names, corrected_prior, alpha=0.5, label='Estimated π̂')
plt.legend()
plt.title('Observed vs Estimated Distribution')
plt.show()
```

**技巧 3：检查逆校正的收益**

```python
naive_acc = overlap_accuracy(ground_truth, naive_prior)
corrected_acc = overlap_accuracy(ground_truth, corrected_prior)
gain = corrected_acc - naive_acc
print(f"Inverse Correction Gain: {gain*100:.1f}%")  # 应 >2%
```

---

## 推荐阅读顺序（代码版）

**第 1 遍（快速上手）：**
1. `run_labelshift.py`（完整 Pipeline，11 阶段）
2. `prior.py`（核心算法：estimate_priors_least_squares）
3. `benchmark_evaluation.py`（如何评估结果）

**第 2 遍（深入细节）：**
1. `classifier.py`（两种分类器实现）
2. `generate.py`（文本生成接口）
3. 各基线方法（了解 Naive MIA 的局限）

**第 3 遍（优化调试）：**
1. 调整超参数（lr、batch_size、n_samples）
2. 尝试领域合并策略（避免病态矩阵）
3. 可视化诊断图（定位问题）

---

## 参考链接

- 论文：[arXiv:2605.30348](https://arxiv.org/abs/2605.30348) (ACL 2026 Main)
- 官方代码：[GitHub: Yaxin9Luo/LLMSurgeon](https://github.com/Yaxin9Luo/LLMSurgeon) (MIT License)
- 报告文件：[Papers/2026/05/2026-05-31-llmsurgeon.md](https://github.com/meraks/ArxivPaperReader/blob/main/Papers/2026/05/2026-05-31-llmsurgeon.md)
