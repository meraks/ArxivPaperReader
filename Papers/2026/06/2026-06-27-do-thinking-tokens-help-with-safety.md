# Do Thinking Tokens Help with Safety?

## 论文元数据
- 标题：Do Thinking Tokens Help with Safety?
- 作者：Narutatsu Ri, Abhishek Panigrahi, Sanjeev Arora (Princeton University)
- arXiv ID：2606.25013
- 发表/提交日期：2026-06-23
- 发表/会议：ICML 2026 AI4GOOD Workshop (Oral)
- 官方代码：github.com/narutatsuri/lrm_safety_deliberation
- 代码发现方式：PDF原文

---

## Ch1: 论文概述与核心贡献

大型推理模型（Large Reasoning Models, LRMs）在回答前生成 thinking tokens（思考链），在数学与代码基准上超越指令微调版本。一个普遍信念随之产生：这种"更审慎"的模式应当改善对齐与安全，因为模型获得了安全空间来斟酌计划中的回复是否违反安全原则。

本文针对这一信念提供反面证据。研究对象覆盖四个前沿开源权重推理模型家族——GPT-OSS、Qwen、Olmo、Phi。核心论点由三个层层递进的发现支撑。

**核心贡献速览：**

1. **安全决策在思考开始前就已高度可读。** 仅用第一个 thinking token 的 hidden representation 训练一个线性 head，即可在 0.84–0.95 AUROC 下预测最终 refusal/compliance 结局（Table 1）。这一可读性早于任何可见思考内容。

2. **Thinking 过程更像 prefix completion 而非 deliberative revision。** 在 thinking trace 完成约 20% 之后，最终结局极少改变（continuation variance <0.2，Figure 3）；文本层面看似审慎的"立场震荡"绝大多数（72.8–76.6%）发生在结局已被锁定之后。

3. **现有安全防御方法效果有限，且抑制了本就稀疏的 deliberation 信号。** 无论 inference-time 还是 training-based 方法，都主要沿 ASR–ORR tradeoff 移动模型行为，无法同时改善两个指标；它们在动机上声称"诱导审慎"，实际效果却是降低震荡计数最高达 95%（Table 3）。

论文由此得出结论：当前推理模型的安全行为远比普遍假设的"非审慎"。这指向一个开放需求——设计能真正诱发安全 deliberation 的方法。

---

## Ch2: 研究背景与动机

### 2.1 LRM 与 thinking tokens

指令微调模型（instruction-tuned models）直接输出回答。推理模型则在回答前先生成一段 thinking trace，该 trace 通常以 `<think>`/`</think>` 标签包裹，对用户可见或部分可见。这段额外的 token 序列被期望承担"安全缓冲"角色：模型在承诺有害内容前有机会自我审查。

### 2.2 两个安全指标

论文用两个互补指标刻画安全行为：

- **ASR（Attack Success Rate，攻击成功率）↓越低越好**：在 harmful prompts 上模型实际给出有害回复的比例。衡量 under-refusal（对有害请求过度配合）。
- **ORR（Over-Refusal Rate，过度拒绝率）↓越低越好**：在 benign prompts 上模型错误拒绝的比例。衡量 over-refusal（对无害请求过度拒绝）。

理想防御应当**同时降低** ASR 与 ORR。若一个方法降低 ASR 却抬高 ORR，只是把"不够安全"换成了"过度保守"，并未真正改善安全行为质量。

数据集规模（§2.1）：2,500 条 harmful prompts 用于 ASR，2,885–6,750 条 benign prompts 用于 ORR（含扩展池）。覆盖四个模型：Qwen3-8B、Olmo-3-7B-Think、Phi-4-Reasoning、GPT-OSS-20B。标签由四个 guardrail 分类器集成判定：WildGuard、Qwen3Guard、Granite Guardian、OSS-Safeguard。

### 2.3 被挑战的信念

主流叙事认为：thinking 提供了 deliberation 机会 → 模型能识别有害请求 → 安全性提升。这一直觉若成立，应表现为（a）结局在思考过程中可被改变，（b）思考文本真正承载权衡，（c）基于"诱导审慎"设计的防御方法能同时改善 ASR 与 ORR。

本文的三项发现分别否定这三点：结局在思考前已锁定（Ch3/Ch4）、文本层面的审慎多为 performative（Ch4）、现有防御沿 tradeoff 移动而非双赢（Ch5）。

---

## Ch3: 核心发现一：安全决策在可见思考前已高度可读

### 3.1 Fisher 判别 J(t) 的 U 形（refusal valley）

论文沿 thinking 位置 $t$ 计算拒绝类与配合类 hidden state 的 Fisher 判别度 $J(t)$，即两类均值差的平方除以两类方差之和：

$$J(t) = \frac{\lVert \mu_{\text{refuse}}(t) - \mu_{\text{comply}}(t) \rVert^2}{\sigma^2_{\text{refuse}}(t) + \sigma^2_{\text{comply}}(t)}$$

$J(t)$ 呈 U 形：在 $t=0$（第一个 thinking token）已处于高位，随思考推进下沉，临近 prefill 末端再度抬升。这一形态有两层含义：

- **第一 token 处的高 $J$**：结局在思考一开始就被 hidden representation 区分开了。
- **中段的"refusal valley"**：真正的思考过程反而让两类表示更难分离——即 deliberation 的表象并未对应可分离度的提升。

prefill 阶段 readout 同步显示：信号在 prompt 处理过程中一直很弱，**临近 prefill 末端才出现尖峰**。这说明判别能力并非逐步累积，而是在进入 thinking 前的最后一刻集中涌现。

### 3.2 第一 token probe 的可预测性

在第一个 thinking token 的 hidden representation 上训练线性 probe，预测最终 refusal/compliance。Table 1 给出精确结果：

| 模型 | 数据集 | AUROC | BAcc |
|------|:-----:|:----:|:----:|
| Qwen3-8B | Harmful | 0.948±0.010 | 0.878±0.019 |
| Qwen3-8B | Benign | 0.888±0.025 | 0.804±0.031 |
| Qwen3-8B | Pooled | 0.971±0.004 | 0.919±0.008 |
| Olmo-3-7B-Think | Harmful | 0.881±0.016 | 0.794±0.017 |
| Olmo-3-7B-Think | Benign | 0.891±0.012 | 0.810±0.014 |
| Olmo-3-7B-Think | Pooled | 0.949±0.005 | 0.872±0.008 |
| Phi-4-Reasoning | Harmful | 0.875±0.014 | 0.792±0.016 |
| Phi-4-Reasoning | Benign | 0.840±0.016 | 0.763±0.018 |
| Phi-4-Reasoning | Pooled | 0.897±0.006 | 0.816±0.008 |
| GPT-OSS-20B | Harmful | 0.916±0.013 | 0.829±0.020 |
| GPT-OSS-20B | Benign | 0.878±0.012 | 0.796±0.013 |
| GPT-OSS-20B | Pooled | 0.939±0.004 | 0.861±0.007 |

全部模型在 pooled 数据上 AUROC 达 0.897–0.971。摘要给出的总体量级为 0.84–0.95 AUROC 与 ~88% BAcc——即便只用 Harmful 子集，最低值也达 0.875（Phi-4）。结局在思考尚未开始时就已经近乎可解。

### 3.3 文本基线无效：信号藏在 hidden representation 中

为排除"可读性来自 token 本身"的解释，论文构造文本基线：对第一个 token 的可见字符串做 TF-IDF 特征，再训练相同 probe。该基线在所有模型与所有数据集上均接近 chance 水平（AUROC ~0.50，BAcc ~0.50–0.59）。

判别能力既不在 token 文本里，也不依赖思考内容，而完整地编码在 prefill 末端形成的 hidden representation 中。这是后续两项发现的基础：既然结局已被"写入"表示，思考阶段对结局的改变空间被严重压缩。

---

## Ch4: 核心发现二：Thinking 是 Prefix Completion 而非 Deliberative Revision

### 4.1 Continuation variance 在 B=20% 已坍缩

衡量"思考能否改变结局"的直接检验是 continuation variance：取一条参考 thinking trace，在长度占比 $B$ 处截断，从截断点多次采样续写，统计最终 refusal/compliance 结局相对于参考结局的翻转率。

Figure 3 显示该翻转率在 $B=20\%$ 处已降至 ~0.2 以下（近似，Figure 3）。配合 majority-label flips 70–97% 保持不变（Figure 4），结论一致：**前 20% 的思考前缀之后，结局基本锁定**。后续 80% 的思考 token 并未行使"重新决策"的功能，而是在补全一个已经确定的结局。

### 4.2 Thinking 不改善 ASR–ORR tradeoff

若 thinking 真正承载审慎，应能同时压低 ASR 与 ORR。实际观察不到任何模型实现双赢：

- **Qwen3-8B、Olmo-3-7B-Think、Phi-4-Reasoning 倾向 over-comply**：思考把它们推向更配合的方向（更易满足请求，ASR 端承压）。
- **GPT-OSS-20B 呈相反 tradeoff**：思考把它推向更拒绝的方向（ORR 端承压）。

四种模型分布在 tradeoff 的两端，没有一种通过思考同时改善两个指标。这与"thinking = deliberation"的直觉直接冲突。

### 4.3 句子级分析：大多数震荡是 performative

论文对 thinking trace 做句子级立场标注，识别"立场震荡"（一段 trace 内 stance 从拒绝翻转到配合或反之）。统计结果（Figure 5、§3.2）：

- 仅 **15–34%** 的 rollout 包含立场震荡——震荡本身就不常见。
- 在这些震荡中，**71–92% 是 performative**：文本上立场变了，但结局在统计上不变。
- **72.8–76.6%** 的震荡发生在结局已被锁定之后——它们是事后叙述，而非决策动因。
- 当震荡确实"有意义"（meaningful segment）时，**87–98%** 朝 stance 所暗示的正确方向移动（例如模型写"实际上这有害"后确实更趋拒绝）。

真正起作用的 deliberation 极其稀少且方向正确，但它太罕见，不足以让 thinking 整体承担安全审慎的角色。绝大部分文本层面的"思考"是在为一个已锁定的结局做 prefix completion，并附带 performative 的立场表演。

四项数字共同支持核心论断：thinking 不是 deliberative revision，而是 prefix completion。

---

## Ch5: 现有防御方法评估

论文评估两大类、共九种防御方法，结论是它们都无法同时改善 ASR 与 ORR，且普遍抑制 deliberation。

### 5.1 Base 行为基线

四个模型的裸模型（Base）安全画像差异巨大（Table 2），这是理解后续变化的参照：

| 模型 | ASR↓ | ORR↓ |
|------|:---:|:---:|
| Qwen3-8B | 86.9 | 6.0 |
| Olmo-3-7B-Think | 70.6 | 21.2 |
| Phi-4-Reasoning | 47.3 | 38.5 |
| GPT-OSS-20B | 26.5 | 44.9 |

Qwen3-8B 极度 over-comply（ASR 86.9、ORR 仅 6.0），GPT-OSS-20B 极度 over-refuse（ORR 44.9、ASR 仅 26.5）。这构成了 tradeoff 的两个极端。

### 5.2 Inference-time 防御：降 ASR、抬 ORR，并抑制震荡

三种 inference-time 方法（SafePath-ZS、PSR、SafeRemind）的行为高度一致——以 ORR 上升为代价换取 ASR 下降。完整数据见 Table 2：

| 方法 | Qwen3-8B ASR↓ | Qwen3-8B ORR↓ | Olmo-3-7B-Think ASR↓ | Olmo-3-7B-Think ORR↓ | Phi-4-Reasoning ASR↓ | Phi-4-Reasoning ORR↓ | GPT-OSS-20B ASR↓ | GPT-OSS-20B ORR↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Base | 86.9 | 6.0 | 70.6 | 21.2 | 47.3 | 38.5 | 26.5 | 44.9 |
| PSR | 86.9↑0.0 | 6.1↑0.1 | 38.0↓32.6 | 43.5↑22.3 | 32.7↓14.6 | 70.6↑32.1 | 29.8↑3.3 | 42.7↓2.2 |
| SafeRemind | 80.8↓6.1 | 11.0↑5.0 | 32.5↓38.1 | 44.9↑23.7 | 33.3↓14.0 | 70.4↑31.9 | 21.3↓5.2 | 49.2↑4.3 |
| SafePath-ZS | 68.2↓18.7 | 13.5↑7.5 | 30.7↓39.9 | 49.1↑27.9 | 31.9↓15.4 | 71.9↑33.4 | 22.5↓4.0 | 48.8↑3.9 |

关键模式：

- 在 Olmo-3-7B-Think 上，三种方法都大幅降 ASR（SafePath-ZS 70.6→30.7，↓39.9 最强），但 ORR 同步暴涨（49.1，↑27.9）——纯 tradeoff 移动，无净收益。
- PSR 对 Qwen3-8B 几乎无效（ASR 86.9→86.9，↓0.0；ORR 6.0→6.1，↑0.1），方法效果随模型剧烈波动。
- 在 Phi-4-Reasoning 上，ORR 被推到 70.4–71.9，模型几乎拒绝一切。

Table 3 的震荡分析揭示了真正机制——这些方法声称"诱导审慎"，实则**抑制**审慎：

| 方法 | Qwen3-8B Osc. | Qwen3-8B Mean. | Olmo-3-7B-Think Osc. | Olmo-3-7B-Think Mean. | Phi-4-Reasoning Osc. | Phi-4-Reasoning Mean. | GPT-OSS-20B Osc. | GPT-OSS-20B Mean. |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Base | 1.20 | 0.10 | 1.43 | 0.19 | 0.66 | 0.14 | 0.32 | 0.09 |
| PSR | 0.64↓47% | 0.07↓27% | 0.48↓66% | 0.02↓87% | 0.03↓95% | 0.01↓93% | 0.25↓23% | 0.09↓5% |
| SafeRemind | 1.13↓5% | 0.10↑6% | 0.30↓79% | 0.03↓85% | 0.14↓78% | 0.02↓87% | 0.25↓22% | 0.13↑38% |
| SafePath-ZS | 0.72↓40% | 0.07↓29% | 0.56↓61% | 0.03↓85% | 0.12↓82% | 0.05↓64% | 0.17↓47% | 0.05↓41% |

PSR 在 Phi-4-Reasoning 上把震荡计数从 0.66 砍到 0.03（↓95%），meaningful 计数从 0.14 降到 0.01（↓93%）。SafeRemind 与 SafePath-ZS 在 Olmo/Phi 上同样把 meaningful 计数压到接近零。这些防御不是在增加 deliberation，而是在消除它——通过让模型更早、更刚性地下决心（通常是拒绝）来降 ASR，代价是 ORR 与真正的审慎一并丧失。

### 5.3 Training-based 防御：沿 tradeoff 散布，部分方法使 ASR 更糟

六种 training-based 方法（STAR-1、SafeKey、R1-ACT、ThinkSafe、STAIR、RAPO）同样无法双赢，且效应在不同模型上方向相反（Table 2）：

| 方法 | Qwen3-8B ASR↓ | Qwen3-8B ORR↓ | Olmo-3-7B-Think ASR↓ | Olmo-3-7B-Think ORR↓ | Phi-4-Reasoning ASR↓ | Phi-4-Reasoning ORR↓ | GPT-OSS-20B ASR↓ | GPT-OSS-20B ORR↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| STAR-1 | 41.0↓45.9 | 45.5↑39.5 | 39.2↓31.4 | 49.8↑28.6 | 35.8↓11.5 | 45.3↑6.8 | 71.7↑45.2 | 14.8↓30.1 |
| SafeKey | 56.3↓30.6 | 28.0↑22.0 | 48.1↓22.5 | 36.9↑15.7 | 52.7↑5.4 | 33.5↓5.0 | 71.0↑44.5 | 19.4↓25.5 |
| R1-ACT | 57.0↓29.9 | 37.8↑31.8 | 85.4↑14.8 | 27.7↑6.5 | 53.5↑6.2 | 39.2↑0.7 | 54.6↑28.1 | 46.7↑1.8 |
| ThinkSafe | 30.9↓56.0 | 44.6↑38.6 | 46.7↓23.9 | 38.4↑17.2 | 40.8↓6.5 | 40.8↑2.3 | 18.8↓7.7 | 52.3↑7.4 |
| RAPO | 34.7↓52.2 | 35.0↑29.0 | 17.7↓52.9 | 49.9↑28.7 | 55.4↑8.1 | 53.4↑14.9 | 44.0↑17.5 | 67.5↑22.6 |
| STAIR | 53.8↓33.1 | 45.3↑39.3 | 54.6↓16.0 | 38.3↑17.1 | 47.5↑0.2 | 38.8↑0.3 | 26.4↓0.1 | 68.6↑23.7 |

几个反直觉但关键的观察：

- **部分方法使安全性显著恶化**。STAR-1 在 GPT-OSS-20B 上把 ASR 从 26.5 推到 71.7（↑45.2），SafeKey 同向（71.0，↑44.5）——两个本意提升安全的方法把模型变成过度配合。R1-ACT 在 Olmo-3-7B-Think 上 ASR 升至 85.4（↑14.8）。RAPO 在 Phi-4-Reasoning 上 ASR 与 ORR 同时恶化（55.4/53.4）。这些方法沿 tradeoff 漂移，漂移方向还与目标模型的安全画像耦合。
- **即使 ASR 大降，ORR 同步飙升**。ThinkSafe 在 Qwen3-8B 上 ASR 降到 30.9（↓56.0，全场最强降 ASR），但 ORR 飙至 44.6（↑38.6）——又是一个纯 tradeoff 移动。
- **on-policy 与 off-policy 对早期信号的影响相反**。on-policy 类方法（ThinkSafe、STAIR）保留 first-token readability（结局仍早早可读）但减少震荡；off-policy 类方法（STAR-1、SafeKey）则**主动压制早期 refusal 信号**（first-token probe AUROC 下降约 0.11）并减少震荡。off-policy 方法通过抹平早期表示来"藏住"决策，这恰恰是在削弱 Ch3 揭示的可读性，而非建立真正的审慎。

### 5.4 小结

九种方法没有一种同时降低 ASR 与 ORR。inference-time 方法的共性是"抑制 deliberation 以换取刚性拒绝"，training-based 方法的共性是"沿 ASR–ORR tradeoff 漂移，方向取决于模型初始画像"。两类方法都未能创造论文所呼吁的那种真正可改变结局的安全 deliberation。

---

## Ch6: 研究意义与开源实现

### 6.1 研究意义

论文把"指令微调模型的对齐很浅"这一既有结论推广到推理模型：thinking tokens 没有提供新的安全 deliberation 能力，安全决策在 prefill 末端就已固化。这与 chain-of-thought faithfulness 文献形成呼应——可见的推理过程并不总是因果地决定结论。对实践者的含义是：不能用"模型有思考"作为安全保证，评估安全必须直接检验结局可读性与震荡有效性，而非看思考文本是否显得审慎。

### 6.2 代码仓库

官方实现位于 `github.com/narutatsuri/lrm_safety_deliberation`（来源：PDF 原文）。该仓库承载论文四个分析支柱的复现代码。⚠️ 以下模块描述依据论文方法论（§2–§4）整理，仓库内具体文件组织未在研究材料中列出，引用前请以仓库实际内容为准。

**关键模块（按论文方法论对应）：**

1. **Representation extraction**：在指定位置（prefill 各步、每个 thinking token）抽取 hidden state，按最终 refusal/compliance 分组。
2. **Probes**：在 hidden state 上训练线性 head，输出 Table 1 的 AUROC/BAcc；含 TF-IDF 文本基线对照。
3. **Fisher discriminant**：沿位置 $t$ 计算 $J(t)$，绘制 U 形曲线（Figure 2）。
4. **Guardrails**：集成 WildGuard、Qwen3Guard、Granite Guardian、OSS-Safeguard 四个分类器，对每个回复投票/聚合作出 refuse/comply 标签，用于 ASR/ORR 计算（Table 2）。
5. **Oscillation analysis**：句子级 stance 标注，统计 total / meaningful 震荡计数（Table 3、Figure 5）。

### 6.3 如何复现关键图表

- **Table 1（first-token probe）**：抽取第一个 thinking token 的 hidden state → 划分 train/test → 训练线性 probe → 报告 AUROC/BAcc（Harmful / Benign / Pooled 三种设定）。
- **Figure 2（$J(t)$ U 形）**：在每个 thinking 位置 $t$ 上按结局分组，计算 §3.1 的 Fisher 判别度并绘图。
- **Figure 3（continuation variance）**：截断 thinking trace 于占比 $B$ → 多次续写 → 统计结局翻转率对 $B$ 作图。
- **Table 2（ASR/ORR）**：对每种 defense 在四个模型上跑 harmful/benign prompt → guardrail 集成判标签 → 计算 ASR 与 ORR。
- **Table 3（震荡计数）**：对 defense 输出复用震荡分析 pipeline → 报告 total/meaningful 每 rollout 均值。

Fisher 判别度的标准实现（用于理解 $J(t)$，非官方代码，未经仓库验证）：

```python
# ⚠️ 非官方概念实现，未经验证
import torch

def fisher_discriminant(h_refuse, h_comply):
    """h_refuse, h_comply: [n_i, d] 两组样本的 hidden states at position t"""
    mu_r = h_refuse.mean(dim=0)          # [d]
    mu_c = h_comply.mean(dim=0)          # [d]
    var_r = h_refuse.var(dim=0).sum()    # scalar, trace of covariance
    var_c = h_comply.var(dim=0).sum()
    return ((mu_r - mu_c).pow(2).sum()) / (var_r + var_c + 1e-8)
```

线性 probe 的标准实现（用于复现 Table 1 可读性，非官方代码，未经仓库验证）：

```python
# ⚠️ 非官方概念实现，未经验证
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# h_first: 第一个 thinking token 的 hidden state [n, d]
# y: 最终 refuse(1)/comply(0) 标签
clf = LogisticRegression(max_iter=1000).fit(h_train, y_train)
prob = clf.predict_proba(h_test)[:, 1]
auroc = roc_auc_score(y_test, prob)
bacc = balanced_accuracy_score(y_test, prob > 0.5)
```

---

## Ch7: 局限性与延伸阅读

### 7.1 局限性

- **模型范围**：仅评估 open-weight、中等规模的推理模型（7B–20B 级）。闭源超大模型（其 thinking 可能由不同训练范式产生）是否同样表现为 prefix completion，本文未覆盖。
- **安全维度**：只评估 refusal/compliance 这一轴，不涉及事实性（hallucination）、欺骗（deception）、隐私泄露等其他安全维度。"安全决策早固化"是否扩展到这些维度，尚属开放问题。
- **早期固化的来源未解释**：论文证明了结局在 prefill 末端可读，但没有追溯这一固化来自训练的哪个环节（RLHF？safety SFT？reasoning post-training？）。

### 7.2 未来工作

论文在讨论中提出两个明确方向：

1. **理解早期固化的来源**——溯源到具体的训练阶段，解释为何安全决策在思考前就被"写死"。
2. **设计新的训练目标以诱发真正的 deliberation**——现有防御（Ch5）都未能让 thinking 因果地改变结局，需要一种能奖励"在思考中真正重新权衡并修正结局"的训练信号，而非奖励"更早、更刚性的拒绝"。

### 7.3 延伸阅读线索

论文将自身置于两条文献脉络的交汇处：一是"指令微调模型对齐浅层化"的研究（本文将其推广到 reasoning model）；二是 chain-of-thought faithfulness 文献（关注可见推理是否忠实反映真实计算）。理解本文结论时，可从这两条脉络切入：前者解释"为何安全决策早早固化"，后者提供"为何可见 deliberation 多为 performative"的理论框架。

---

*报告基于 arXiv:2606.25013《Do Thinking Tokens Help with Safety?》(Ri, Panigrahi, Arora; Princeton University, 2026)。所有量化数据标注精确来源；图表目测值以 ~ 与（近似，Figure X）标示。*
