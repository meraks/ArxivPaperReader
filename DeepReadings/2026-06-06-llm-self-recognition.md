# LLM Self-Recognition: Steering and Retrieving Activation Signatures

**论文精读报告**

- **论文标题**: LLM Self-Recognition: Steering and Retrieving Activation Signatures
- **作者**: Thibaud Ardoin, Jonas Schäfer, Gerhard Wunder (Freie Universität Berlin)
- **arXiv**: [2606.06315](https://arxiv.org/abs/2606.06315)
- **会议**: ICML 2026 (PMLR 306)
- **代码**: [Thibaud-Ardoin/LLM-Self-Recognition](https://github.com/Thibaud-Ardoin/LLM-Self-Recognition)

---

## Ch1: 论文概述与核心贡献

### 1.1 一句话总结

本文发现 **LLM 的内部激活空间中天然存在自识别信号**（模型能"认出"自己的输出），并通过**稀疏随机向量转向残差流**在生成时注入可检测指纹，实现了 >98% 准确率的文本归因，且不影响生成质量。

### 1.2 核心贡献

| # | 贡献 | 技术意义 |
|---|------|---------|
| 1 | 建立 LLM 自识别能力的可靠性证据 | LDA 线性探针在无 prompt 条件下以 >98% AUROC 区分人类/模型文本，远超 perplexity 基线 |
| 2 | 稀疏转向水印机制 | 99.7% 稀疏随机向量注入残差流，通过 cosine similarity 实现零样本归因 |
| 3 | 信号跨离散化传播 | 转向信号能通过 token 采样和重新嵌入的离散化步骤，检测阶段用**无转向的基模型**即可恢复信号 |
| 4 | 质量无损 | 稀疏转向对 perplexity 和 MMLU 的影响可忽略，且对 Dipper-XXL 强改写具有鲁棒性 |

### 1.3 方法总览

```
生成阶段 (嵌入指纹):
  Prompt → LLM (Layer 1..l) → Al(xi) + α·v → LLM (Layer l+1..N) → 输出文本 t_v
                                        ↑
                                  稀疏随机向量 v
                                  (99.7% 零元素)

检测阶段 (恢复指纹):
  文本 t_v → LLM (Layer 1..l) → Al(t_v) → {cosine similarity with v} 或 {MLP 分类器}
```

### 1.4 为什么重要

> **类比理解**: 想象你在打字时，每次敲键盘的力度轻重形成了独特的"击键节奏"。即使旁人看到的是相同的文字输出，但键盘内部的压力传感器数据能精确识别是谁在打字。这篇论文的本质发现是：LLM 生成文本时的**内部激活模式**就像这个"击键节奏"——即使用户看到的文字完全自然，模型内部仍保留了可检测的"作者签名"。

当前 AI 生成文本的归因面临三个困境：(1) 文本水印修改 token 概率分布，可能降低生成质量；(2) 后验分类器对分布偏移敏感；(3) 无法区分同一基模型的不同微调/转向变体。本文用**激活空间工程**同时解决了这三个问题。

---

## Ch2: 研究背景与动机

### 2.1 AI 生成文本的归因困境

AI 生成内容的爆发式增长带来了严重的归因需求：

- **信息溯源**: 一段文本到底是 GPT-4、Claude 还是 Llama 生成的？
- **滥用检测**: 如何追溯恶意生成内容的来源模型？
- **版权保护**: 如何证明某个模型的输出被未授权使用？

传统方法有三类，各有致命缺陷：

| 方法 | 原理 | 缺陷 |
|------|------|------|
| **文本水印** (KGW, 2023) | 修改 token 概率分布，在词表中嵌入统计信号 | 改变输出分布，可能降低文本质量；改写后信号丢失 |
| **后验分类器** | 在生成文本上训练二分类器 | 对分布偏移敏感；无法区分多个模型 |
| **检索比对** | 将生成文本与已知模型输出库比对 | 需要维护庞大的输出数据库；隐私问题 |

### 2.2 两条关键研究线索的交汇

本文的方法建立在两条独立发展的研究线索之上：

**线索一：LLM 自识别 (Self-Recognition)**

Ackerman et al. (2025) 和 Bowman et al. (2024) 发现了一个引人深思的现象：**LLM 能够以高于随机的准确率区分自己的输出和其他模型的输出**。这种能力并非通过文本表面的风格特征（如用词偏好），而是编码在模型的内部激活中。然而，此前的研究未系统量化这种能力的可靠性和边界条件。

**线索二：激活工程 (Activation Engineering)**

Panickssery et al. (2023) 和 Liu et al. (2023) 展示了通过修改模型中间层的激活值可以精确控制模型行为，且不产生明显的副作用。关键洞察是：**激活空间中的某些方向对应于高级语义概念，沿这些方向的线性干预不会破坏其他语义维度**。

本文的交汇点在于：如果 LLM 的激活空间同时具备 (a) 天然的自识别信号和 (b) 可操控的语义方向，那么能否利用这些性质构建一个**基于激活空间的文本归因系统**？

> **类比理解**: 想象每个人类写作者都有独特的笔迹。即使刻意模仿他人字体，纸张背面的"压痕深度模式"仍然能暴露真实作者。LLM 的输出文本就像墨水字迹（所有人都能看到的内容），而内部激活就是"压痕"——只有拥有原始模型的人能读取这个隐藏签名。

### 2.3 威胁模型

本文假设**白盒检测**：检测者拥有模型的完整内部访问权限（能读取中间层激活值）。这不是一个不合理的假设——模型提供商（如 OpenAI、Meta）在检测自身模型输出时天然具备白盒条件。论文明确声明 **不针对对抗性攻击者**（即主动尝试移除水印的恶意用户），尽管实证结果显示方法对改写攻击具有鲁棒性。

检测场景分两种：
1. **Prompt 条件检测**: 检测时已知原始 prompt（更准确，因为有 prompt 激活作为参照）
2. **Prompt 无关检测**: 仅基于生成文本本身的激活（更实用，但更难）

---

## Ch3: 方法一 — 自识别：激活空间的隐式签名

### 3.1 激活提取

对于具有 $L$ 层的 LLM $M$，选择一个中间层 $l$ 作为检测层。对于文本 $t = (x_0, x_1, ..., x_{n-1})$，提取该层每个 token 的残差流激活：

$$A_l(x_i) \in \mathbb{R}^d \quad (d \text{ 为模型隐藏维度})$$

对文本中的所有 token 求均值，获得固定维度的文本表示：

$$\mathbf{r} = \frac{1}{n} \sum_{i=0}^{n-1} A_l(x_i) \in \mathbb{R}^d$$

**层选择**: 实验表明中间层（约总层数的 40-60%）效果最佳。激活信号在各层之间分布相对平坦（Appendix E），说明自识别信息并非集中在某一层，而是分散在多个层的表示中。

### 3.2 LDA 分类器

使用 **线性判别分析 (LDA)** 配合 **Ledoit-Wolf 协方差收缩**作为分类器：

- **为什么 LDA？** 论文发现激活空间中的自识别信号是**线性可分的**——人类文本和模型文本的激活表示分别聚集在不同的线性子空间中。LDA 恰好能捕捉这种结构，且计算效率极高。
- **Ledoit-Wolf 收缩**: 在小样本场景下稳定协方差矩阵估计，防止过拟合。
- **阈值控制**: 通过调整决策阈值 $\tau$，可以精确控制假阳性率 (FPR)，满足不同应用的安全需求。

### 3.3 实验结果：自识别

实验设置：XL-Sum 英文新闻摘要数据集，LLM 生成 1-2 句摘要 vs. 人类撰写的参考摘要（≤2048 字符）。80/20 训练/测试分割。

| 模型 | 带 Prompt AUROC | 无 Prompt AUROC | PPL 基线 (带 Prompt) | PPL 基线 (无 Prompt) |
|------|:---:|:---:|:---:|:---:|
| Ministral-3-8B (MoE) | **100** | **99.99** | 99.71 | 32.33 |
| Llama-3.1-8B | 99.99 | 99.16 | 99.19 | 47.86 |
| Llama-3.2-3B | 99.96 | 99.03 | 99.43 | 47.49 |
| Llama-3.2-1B | 99.82 | 98.58 | 97.07 | 52.27 |

**关键发现**：

1. **无 Prompt 条件下激活法碾压 perplexity**: perplexity 在无 prompt 条件下几乎失效（AUROC 降至 32-52%，接近随机），而激活法保持在 98.58-99.99%。这是因为 perplexity 需要 prompt 作为条件概率的上下文，而激活中的自识别信号仅依赖于生成文本本身的内部表示。

2. **Ministral 的 perplexity 反转**：对于 Ministral-3-8B，人类撰写的摘要反而具有**更低**的 perplexity（模型认为人类的摘要"更符合自己的语言习惯"），但激活法仍然 100% 准确。这说明自识别信号与语言模型的质量评判（perplexity）是**正交**的。

3. **模型规模效应微弱**：从 1B 到 8B，自识别 AUROC 仅从 98.58% 升至 99.99%，说明即使是小模型也具备相当的自识别能力。

### 3.4 混淆因素控制

为确保检测信号不是"作弊"（如利用文本长度、大小写、标点），论文进行了严格的消融实验：
- **小写化**所有文本
- **去除标点符号**
- **长度匹配**（截断至相同 token 数）

→ AUROC 下降 < 0.5 个百分点，证明信号来自更深层的语义表示而非表面文本特征。

### 3.5 跨领域泛化

在某个新闻类别（如政治）训练的探针，在其他类别（体育、科技）上测试，AUROC > 99.7%。这说明自识别信号是**领域无关的**——模型对其输出文本的"认知"不依赖于内容领域。

> **类比理解**: 自识别信号就像一个工厂在产品上嵌入的隐形水印。不管产品（文本）被用在什么场景，水印本身与使用场景无关。更妙的是，这个水印不是刻意嵌入的，而是制造过程（模型推理）自然留下的"工艺痕迹"。

---

## Ch4: 方法二 — 稀疏转向水印：可控的指纹注入

### 4.1 动机：从检测到归因

自识别只能回答"这段文本是模型 $M$ 生成的吗？"（是/否）。当有多个不同配置的 $M$ 变体时，需要回答"**哪个**变体生成了这段文本？"。这要求**主动注入**可区分的信号。

### 4.2 稀疏随机转向向量

在每个 token 生成步骤中，向残差流中间层注入一个固定向量：

$$A_l(x_i) \leftarrow A_l(x_i) + \alpha \cdot v$$

其中：
- $v \sim U([-1, 1]^d)$：随机采样（均匀分布）
- **99.7% 稀疏**：绝大多数维度为零，仅约 0.3% 的维度非零（对于 Llama-8B 的 d=4096，约 12 个非零元素）
- $\alpha = 5$：缩放系数（在所有实验中固定）

**为什么稀疏？**

论文对比了密集向量（所有维度非零）和稀疏向量的效果：

| 属性 | 密集向量 | 稀疏向量 (99.7%) |
|------|---------|-----------------|
| 可检测性 | 高 | 高（几乎无差异） |
| 生成质量影响 | 明显下降 | **可忽略** |
| MMLU 影响 | 有负面影响 | **~0%** |

直觉：密集向量在所有 4096 个维度上同时扰动，扰乱了大量语义特征；稀疏向量仅在少数维度上操作，而这些维度可能与语义空间近乎正交。

> **类比理解**: 在电影胶片上加一个肉眼不可见的标记点——密集水印相当于在每一帧上都画一个小标记（观众会注意到画质下降），稀疏水印相当于只在几帧的角落嵌入一个像素级标记（完全不可见，但放映设备能检测到）。

### 4.3 两种检测方法

#### 方法 A：MLP 分类器（有监督）

收集 $K$ 个转向向量的标注数据集 $D = \{(A_l(t_{v_k, p}), k) \mid k \in \{1, ..., K\}, p \in \mathcal{P}\}$。

训练一个小型 MLP：
- 结构：2 个隐藏层，每层 32 个神经元
- 训练轮数：**仅 1 个 epoch**
- 文本级预测：对每个 token 独立预测，**多数投票**决定最终类别

关键设计选择：
- **按 prompt 分割数据**（70/10/20），确保没有 prompt 同时出现在训练和测试集中——这保证了模型学到的是**转向信号**而非 prompt 相关特征
- 极小网络+单 epoch 训练避免了过拟合和分布偏移

#### 方法 B：Cosine Similarity（零样本）

这是本文最优雅的发现：**随机转向向量与语义流形近似正交**，因此可以直接通过余弦相似度恢复：

$$\text{sim}(t, v_k) = \frac{A_l(t) \cdot v_k}{\|A_l(t)\| \cdot \|v_k\|}$$

对候选向量 $\{v_1, ..., v_K\}$ 分别计算相似度，最大值对应最可能的来源。**不需要任何训练数据**。

### 4.4 实验结果：多模型归因

**二分类（两个转向向量）：**

| 模型 | ELI5 Token/Text Acc% | Fresh News Token/Text Acc% |
|------|:---:|:---:|
| Ministral-3-8B | 99.3 / **100** | 94.6 / **100** |
| Llama-3.1-8B | 94.0 / 99.1 | 90.5 / 99.1 |
| Llama-3.2-3B | 83.3 / 95.5 | 75.5 / 88.3 |
| Llama-3.2-1B | 72.0 / 85.3 | 69.5 / 83.8 |

**关键发现**：
1. **大模型天然更适合**：8B 参数的模型在文本级几乎完美（99-100%），1B 模型降至 83-85%
2. **高熵更难**：Fresh News（开放生成）比 ELI5（事实问答）更难，可能因为指令调优模型在开放生成时激活分布更分散
3. **扩展到 8 个向量**：仍远超随机基线 (12.5%)，暗示可构建**组合水印**（N 个向量 → 2^N 种可区分状态）

---

## Ch5: 实验结果与分析

### 5.1 实验设置一览

| 组件 | 详情 |
|------|------|
| **模型** | Llama-3.1-8B-Instruct, Ministral-3-8B-Instruct (MoE), Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct |
| **数据集** | XL-Sum（低熵摘要）、ELI5（中熵问答）、Fresh News（高熵开放生成，Guardian 2025年11月-2026年1月） |
| **检测层** | 各模型中间层（按模型大小优化） |
| **向量稀疏度** | 99.7% |
| **转向系数 α** | 5（所有实验统一） |
| **MLP 结构** | 2×32 隐藏层，1 epoch |
| **文本级决策** | 多数投票 |

### 5.2 质量影响分析

论文从两个维度评估转向对生成质量的影响：

**维度一：DeBERTa 质量分类器**

使用预训练的文本质量评估模型 (DeBERTa-based) 对转向生成文本和基模型生成文本进行评分比较。结果显示密集转向有统计显著的质量下降，而**稀疏转向的质量差异在统计噪声范围内**。

**维度二：MMLU 基准**

在标准 MMLU 基准上测试转向后的模型能力：

| 指标 | 无转向 (基线) | 密集转向 | 稀疏转向 (99.7%) |
|------|:---:|:---:|:---:|
| MMLU 准确率 | 基准 | 下降 | **~0% 变化** |

稀疏转向对模型的知识和推理能力几乎没有影响，这意味着该方法可以实际部署而不会牺牲模型的实用价值。

### 5.3 改写鲁棒性

改写攻击是传统水印方法的"阿喀琉斯之踵"——KGW 水印在强改写（如 Dipper-XXL paraphrase 模型）后信号几乎完全丢失。

本文的方法展示了截然不同的鲁棒性：

- 用 **Dipper-XXL**（强改写模型）对水印文本进行改写
- 改写后通过 **Cosine Similarity** 仍能恢复原始转向向量
- 原因：改写改变了文本的**表层 token 序列**，但模型在处理改写文本时，其内部激活中仍保留了与原始转向向量的相似性——说明信号编码在比 token 更深层的表示中

> **类比理解**: 传统水印像是在纸上用水性笔写了一个"此文件由张三打印"的水印——用湿布一擦就没了。本文的方法像是在造纸时就在纸浆中混入了微量荧光纤维——即使纸被揉皱、重新展平，用紫外灯一照，纤维仍在。

### 5.4 跨领域性能

| 转移方向 | AUROC |
|---------|:---:|
| 同一新闻类别 | 99.9%+ |
| 跨新闻类别 | >99.7% |
| 摘要→问答 (零样本) | 有下降但可用 |

自识别探针的跨领域泛化能力很强，但转向水印的跨领域检测准确率会下降。论文将此列为一项局限性，需要针对目标领域做微调。

### 5.5 与现有方法对比

| 方法 | 可检测性 | 质量影响 | 改写鲁棒性 | 多模型归因 | 黑盒可用 |
|------|:---:|:---:|:---:|:---:|:---:|
| KGW 水印 | 高 | 中等 | ❌ | ❌ | ✅ |
| 后验分类器 | 中 | N/A | 低 | ❌ | ✅ |
| **本文方法** | **极高** | **可忽略** | ✅ | ✅ | ❌ |

显然，本文方法在除了"黑盒可用性"之外的所有维度上都有显著优势。对于模型提供商而言，白盒访问是天然的，因此这一限制在实际部署中不构成障碍。

---

## Ch6: 代码实现详解

> ⚠️ 以下代码从官方仓库 [Thibaud-Ardoin/LLM-Self-Recognition](https://github.com/Thibaud-Ardoin/LLM-Self-Recognition) 中提取，按论文方法重构为概念实现。仓库结构包含两个模块：`self_recognition/` (自识别)和 `steering_watermark/` (转向水印)。

### 6.1 激活提取器

```python
# 来源: self_recognition/ (概念实现)
# 作用: 从LLM中间层提取token级激活
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class ActivationExtractor:
    """提取指定层的残差流激活"""
    
    def __init__(self, model_name: str, layer_idx: int):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.layer_idx = layer_idx
        self.activations = None
        
        # 注册 forward hook 到指定层
        target_layer = self.model.model.layers[layer_idx]
        target_layer.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        """捕获残差流输出"""
        # output[0] 是残差流激活: (batch, seq_len, hidden_dim)
        self.activations = output[0].detach()
    
    @torch.no_grad()
    def extract(self, text: str) -> torch.Tensor:
        """提取文本的token级激活"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        _ = self.model(**inputs, output_hidden_states=False)
        # activations: (1, seq_len, hidden_dim)
        return self.activations.squeeze(0)  # (seq_len, hidden_dim)
    
    def aggregate(self, text: str) -> torch.Tensor:
        """均值池化 → 固定维度文本表示"""
        acts = self.extract(text)  # (n, d)
        return acts.mean(dim=0)    # (d,)
```

### 6.2 自识别 LDA 分类器

```python
# 来源: self_recognition/LLM_SelfRecognition_via_Activations-public/ (概念实现)
# 作用: 用LDA+均值激活区分人类/模型文本
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import LedoitWolf

class SelfRecognitionDetector:
    """基于激活的自识别检测器"""
    
    def __init__(self, extractor: ActivationExtractor):
        self.extractor = extractor
        self.lda = LinearDiscriminantAnalysis(solver='lsqr')
        self.threshold = 0.5
    
    def fit(self, human_texts: list[str], model_texts: list[str]):
        """训练LDA分类器"""
        # 提取所有文本的均值激活
        X_human = np.stack([self.extractor.aggregate(t).cpu().numpy() 
                            for t in human_texts])
        X_model = np.stack([self.extractor.aggregate(t).cpu().numpy() 
                            for t in model_texts])
        
        X = np.concatenate([X_human, X_model])
        y = np.concatenate([np.zeros(len(human_texts)), 
                           np.ones(len(model_texts))])
        
        self.lda.fit(X, y)
        return self
    
    def predict(self, text: str) -> tuple[bool, float]:
        """预测文本是否为模型生成，返回(判定, 置信度)"""
        r = self.extractor.aggregate(text).cpu().numpy().reshape(1, -1)
        proba = self.lda.predict_proba(r)[0, 1]  # 模型生成的概率
        return proba > self.threshold, proba
```

### 6.3 稀疏转向生成器

```python
# 来源: steering_watermark/ (概念实现)
# 作用: 生成时注入稀疏随机向量 → 水印文本
import torch
import torch.nn.functional as F

class SteeringWatermarkGenerator:
    """稀疏转向水印生成器"""
    
    def __init__(self, model, tokenizer, layer_idx: int, 
                 hidden_dim: int, sparsity: float = 0.997, alpha: float = 5.0):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.sparsity = sparsity
        self.alpha = alpha
        
        # 生成稀疏随机转向向量
        self.steering_vector = self._generate_sparse_vector(hidden_dim)
        self._hook_handle = None
    
    def _generate_sparse_vector(self, d: int) -> torch.Tensor:
        """生成99.7%稀疏的随机向量"""
        v = torch.zeros(d)
        n_nonzero = int(d * (1 - self.sparsity))  # ~12 for d=4096
        indices = torch.randperm(d)[:n_nonzero]
        v[indices] = torch.rand(n_nonzero) * 2 - 1  # U([-1, 1])
        return v
    
    def _steering_hook(self, module, input, output):
        """Forward hook: 向残差流注入转向向量"""
        # output[0]: (batch, seq_len, hidden_dim)
        modified = output[0] + self.alpha * self.steering_vector.to(output[0].device)
        return (modified,) + output[1:] if len(output) > 1 else (modified,)
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """生成带水印的文本"""
        # 注册转向 hook
        target_layer = self.model.model.layers[self.layer_idx]
        self._hook_handle = target_layer.register_forward_hook(self._steering_hook)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, 
                do_sample=True, temperature=0.7
            )
            generated = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            return generated
        finally:
            self._hook_handle.remove()
```

### 6.4 零样本 Cosine Similarity 检测

```python
# 来源: steering_watermark/ (概念实现)
# 作用: 无需训练，直接用cosine similarity恢复转向向量
class ZeroShotAttributor:
    """零样本水印归因：通过cosine similarity匹配转向向量"""
    
    def __init__(self, extractor: ActivationExtractor, 
                 candidate_vectors: list[torch.Tensor]):
        """
        Args:
            extractor: 激活提取器
            candidate_vectors: 候选转向向量列表 [v1, v2, ..., vK]
        """
        self.extractor = extractor
        self.candidate_vectors = candidate_vectors
    
    def attribute(self, text: str) -> tuple[int, list[float]]:
        """
        归因文本到最可能的转向向量
        Returns: (最佳索引, [各向量相似度])
        """
        acts = self.extractor.aggregate(text)  # (d,)
        
        similarities = []
        for v in self.candidate_vectors:
            sim = torch.nn.functional.cosine_similarity(
                acts.unsqueeze(0), v.unsqueeze(0), dim=1
            ).item()
            similarities.append(sim)
        
        best_idx = int(np.argmax(similarities))
        return best_idx, similarities
```

### 6.5 MLP 多模型归因

```python
# 来源: steering_watermark/ (概念实现)
# 作用: 训练小型MLP进行K分类水印归因
class MultiModelAttributor:
    """基于MLP的多模型水印归因器"""
    
    def __init__(self, input_dim: int, num_vectors: int):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_vectors)
        )
        self.num_vectors = num_vectors
    
    def fit(self, extractor, texts_by_vector: dict[int, list[str]], 
            epochs: int = 1, lr: float = 0.001):
        """单epoch训练"""
        # 构建数据集
        X, y = [], []
        for vec_id, texts in texts_by_vector.items():
            for t in texts:
                X.append(extractor.aggregate(t).cpu().numpy())
                y.append(vec_id)
        
        X = torch.tensor(np.stack(X), dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        
        self.mlp.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.mlp(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        
        return self
    
    @torch.no_grad()
    def predict_text(self, extractor, text: str) -> int:
        """文本级预测：token级多数投票"""
        acts = extractor.extract(text)  # (n, d)
        self.mlp.eval()
        logits = self.mlp(acts)  # (n, num_vectors)
        token_preds = logits.argmax(dim=1)  # (n,)
        
        # 多数投票
        return torch.mode(token_preds).values.item()
```

---

## Ch7: 局限性与延伸阅读

### 7.1 论文明确指出的局限

1. **白盒检测要求**: 检测者必须拥有模型的完整内部访问权限。这排除了第三方检测场景（如社交媒体平台希望检测用户帖子是否由 ChatGPT 生成）。但对于模型提供商（OpenAI 的 GPT-4 检测 GPT-4 输出），这完全不是问题。

2. **跨领域性能下降**: 转向水印的检测准确率在跨领域时下降。论文指出需要针对目标领域进行微调，相当于需要少量目标领域的标注数据。

3. **不能区分不同模型族**: 当前方法仅支持同一基模型的不同转向变体，不能回答"这段文本是 Llama 还是 GPT 生成的"。这是方法设计的固有限制——信号依赖于特定模型的激活空间结构。

4. **小模型性能明显低于大模型**: Llama-3.2-1B 在多项任务上的归因准确率仅为 70-85%，远低于 8B 模型的 99%+。

5. **未针对对抗攻击优化**: 论文未测试针对性的激活扰动攻击。理论上，攻击者如果能访问模型内部，可以尝试通过激活扰动来破坏信号。

### 7.2 值得关注的潜在方向

- **组合水印编码**: N 个独立转向向量可编码 2^N 位信息（如版本号、生成时间戳、用户 ID），论文已展示 8 向量可区分的结果
- **跨模型迁移**: 如果能找到不同模型激活空间之间的映射，可能实现跨模型族的归因
- **黑盒扩展**: 类似于知识蒸馏，能否训练一个"代理检测器"模型来模拟原始模型的激活特征？
- **对抗鲁棒性**: 结合对抗训练或差分隐私，可能增强对主动攻击的防御

### 7.3 延伸阅读

- Kirchenbauer et al. (2023) — *A Watermark for Large Language Models* (KGW 水印)
- Panickssery et al. (2023) — *Steering Llama 2 via Contrastive Activation Addition*
- Ackerman et al. (2025) — LLM self-recognition pioneering work
- Bowman et al. (2024) — LLM self-recognition benchmarks
- Zou et al. (2023) — *Representation Engineering: A Top-Down Approach to AI Transparency*

---

## 附录: 技术细节补充

### A. 为什么稀疏向量优于密集向量？

论文对比实验中，密集向量（所有维度非零）虽然也可检测，但会产生副作用：
- 生成文本的流畅度下降
- MMLU 得分降低
- 输出分布偏移

直觉解释：LLM 的激活空间是高度结构化的，大多数维度参与语义编码。密集向量同时扰动所有维度，相当于在所有语义特征上叠加噪声。稀疏向量仅在极少数维度上操作，这些维度可能有较大的"操作余量"（类似于主成分分析中的低方差维度）。

### B. 信号通过离散化的机制

这是论文最令人惊讶的发现：注入连续激活空间的信号能通过离散 token 采样（argmax 或温度采样）和重新嵌入传播。一个可能的解释是：转向向量创造了一个**偏置**，使得后续层的注意力机制和 FFN 倾向于选择与转向方向一致的 token 分布，而这种偏置在 token 嵌入重新进入网络后又能被无转向的基模型"检测到"。

> **类比理解**: 你在河流上游滴入一滴染料。即使下游的水被舀起来倒进另一个杯子（离散化），杯子里的水仍然带色——因为染料改变了水分子的吸收光谱，而不仅仅是改变了"哪勺水被舀起来"。