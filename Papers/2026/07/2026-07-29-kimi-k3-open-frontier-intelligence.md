> **论文**：Kimi K3: Open Frontier Intelligence
> **作者**：Kimi Team (Moonshot AI)
> **arXiv ID**：2607.24653
> **发表时间**：2026-07-27
> **许可协议**：Kimi K3 License
> **代码仓库**：https://github.com/MoonshotAI/Kimi-K3

## 第 1 章 概述

### 1.1 一句话定位

Kimi K3 是 Moonshot AI 发布的 2.8T 参数开源 MoE 模型（104B 激活参数），原生支持多模态与 100 万 token 上下文窗口，同时推进预训练规模和测试时计算两条 scaling 轴，在编码、Agent、推理和视觉任务上达到前沿水平，是迄今为止最大的开源模型。

### 论文图表总览

论文包含多个 Figure 和 Table，主要数据来自论文全文及 GitHub README 中的完整 benchmark 对比表。由于论文为技术报告形式，图表编号以论文实际标注为准。

| 编号 | 内容 | 章节 |
|------|------|------|
| **Figure 1** | 主结果雷达图/柱状图 | 第 1 章 |
| **Figure 2** | Kimi K3 架构总览（KDA、Gated MLA、Stable LatentMoE） | 第 2 章 |
| **Table 1**（README） | 模型配置汇总（参数、层数、专家数等） | 第 2 章 |
| **Table 2**（README/论文） | 推理与知识 benchmark 对比 | 第 5 章 |
| **Table 3**（README/论文） | 编码 benchmark 对比 | 第 5 章 |
| **Table 4**（README/论文） | Agentic benchmark 对比 | 第 5 章 |
| **Table 5**（README/论文） | 视觉 benchmark 对比 | 第 5 章 |

### 1.2 核心贡献

1. **开源前沿规模**：训练了 2.8T 参数原生多模态 MoE 模型（104B 激活参数、100 万 token 上下文窗口），是首个 3T 级开源模型
2. **架构创新**：KDA（Kimi Delta Attention）高效长序列混合 + AttnRes（Attention Residuals）跨层选择性信息流 + Stable LatentMoE（896 专家，16 激活/token），综合提升约 2.5× scaling 效率
3. **多域后训练**：在编码、通用 Agent、推理任务等多个域上执行不同推理强度的 RL，通过多教师 on-policy 蒸馏合并为统一模型
4. **基础设施突破**：KDA 系统协同设计、MoonEP 完美平衡专家并行训练、百万 token Agentic RL 系统（含可恢复沙箱状态）
5. **完全开源**：完整模型权重在 Kimi K3 License 下开放

### 1.3 关键结果速览

- **编码**：DeepSWE 67.5%（仅次于 GPT-5.6 Sol 73.0% 和 Claude Fable 5 70.0%），FrontierSWE 81.2%（超越 GPT-5.6 Sol 71.3%），SWE-Marathon 42.0%（最高），Kimi Code Bench 2.0 72.9%
- **Agent**：BrowseComp 91.2%（最高）、MCPMark-Verified 94.5%（最高）、ResearchRubrics 76.2%（最高）、AutomationBench 30.8%（最高）
- **推理与知识**：GPQA Diamond 93.5%（接近 GPT-5.6 Sol 94.1%）、HLE-Full 43.5/56.0%、AA-LCR 74.7%（最高）
- **视觉**：OmniDocBench 91.1%（最高）、Video-MME 90.0%（最高）、MMVU 82.1%（最高）、CharXiv (RQ) 84.8/91.3%
- 整体性能略逊于最强闭源模型（Claude Fable 5、GPT-5.6 Sol），但优于所有其他开源和闭源对比模型

## 第 2 章 Kimi K3 架构设计

Kimi K3 的架构设计围绕三个维度展开：**序列长度维度**的信息混合（KDA）、**模型深度维度**的信息流动（AttnRes）、以及**模型宽度维度**的稀疏扩展（Stable LatentMoE）。此外，模型配备原生视觉编码通路 MoonViT-V2。

### 2.1 Kimi Delta Attention（KDA）

KDA 是 Kimi K3 的核心注意力机制，旨在为百万 token 级长上下文提供高效的长序列混合。KDA 的设计理念源于 DeltaNet 和 Gated Linear Attention（GLA）等线性注意力工作，但在序列混合的效率上更进一步。KDA 的思路是将传统的单一注意力操作分解为更细粒度的混合操作，通过周期性插入 Gated MLA（Multi-head Latent Attention）层来保持全局交互能力。

KDA 的核心计算单元可以表示为：

$$o_t = \sigma(q_t)^\top \sum_{i=1}^{t} \left( \alpha^{t-i} \cdot \sigma(k_i) \odot v_i \right)$$

其中 $\sigma(\cdot)$ 为非线性门控函数，$\alpha$ 为可学习的衰减因子，控制长程信息的遗忘速率。与传统 softmax 注意力不同，KDA 使用线性复杂度的递推形式，使长序列处理的计算量从 $O(L^2)$ 降至 $O(L)$。

具体地，Kimi K3 的 93 层 Transformer 中，包含 **69 层 KDA** 和 **24 层 Gated MLA**。KDA 每 3 层为一组，然后插入 1 层 Gated MLA，交替排列。这种「3:1」的 KDA 与 Gated MLA 配比在设计上取得了效率与能力的平衡：KDA 负责高效的局部/长程序列混合，Gated MLA 负责全局信息的精炼和交互。

KDA 的设计还包含了以下关键组件：
- **卷积模块**（Conv）：在线性投影前后引入深度可分离卷积，增强局部上下文建模能力
- **双线性门控**：使用 $\sigma(q)$ 和 $\sigma(k)$ 双曲正切门控，控制信息流的写入与读取
- **LayerNorm 集成**：在 KDA 模块内部嵌入 LayerNorm，确保递推计算过程中的数值稳定性

这种架构组合使得模型能够在保持长上下文能力的同时，显著降低计算开销。KDA 的实现还包括以下系统协同设计：
- **融合 Kernel**：将 KDA 的卷积、线性投影、门控等操作合并为单个 GPU kernel
- **KDA Context Parallelism**：跨设备并行化 KDA 的长序列计算，将百万 token 上下文均匀分布在多个 GPU 上
- **状态感知 Prefix Caching**：跨请求缓存 KDA 的隐状态，在复用上下文的场景下避免重复计算

### 2.2 Attention Residuals（AttnRes）

AttnRes 允许每一层**有选择性地关注所有前层的表征**，而非仅关注上一层的输出。这是通过**可学习的伪查询向量（pseudo-queries，记为 $w$）** 实现的。对于第 $n$ 个 Block，AttnRes 的计算过程为：

1. 对 Embedding $h_0$ 和之前所有 Block 的输出 $h_1, h_2, \ldots, h_{n-1}$，分别计算与伪查询 $w_n$ 的注意力分数
2. 通过 softmax 归一化得到权重系数 $\alpha_{n,i}$：
   $$\alpha_{n,i} = \frac{\exp(w_n^\top h_i)}{\sum_{j=0}^{n-1} \exp(w_n^\top h_j)}, \quad \sum_{i=0}^{n-1} \alpha_{n,i} = 1$$
3. 加权聚合得到当前 Block 的输入：
   $$\tilde{h}_n = \sum_{i=0}^{n-1} \alpha_{n,i} \cdot h_i$$

这种机制使模型能够在网络深度方向上实现**选择性信息流**：每一层可以动态决定是更多地借鉴浅层 Embedding（保留原始语义）、中层抽象特征，还是前一层的最新输出。这有效缓解了深层网络中的信息衰减问题（vanishing gradient 的一种变体）。

AttnRes 与残差连接（Residual Connection）的关键区别在于：
- **残差连接**：每层固定地接收上层的输出 $x_{n} = f(x_{n-1}) + x_{n-1}$，信息流路径固定
- **AttnRes**：每层可选择性地关注所有前层，信息流路径随输入和训练动态调整，更加灵活

Kimi K3 的 Figure 2 直观展示了这一机制：每个 Block 上方的 $w$ 符号代表伪查询，$\alpha$ 表示计算出的注意力权重分布。

### 2.3 Stable LatentMoE

Kimi K3 的 MoE 架构在稀疏性方面取得了显著扩展，是支撑 2.8T 参数量的关键设计：

- **Routed Experts**：896 个
- **每 token 激活**：16 个 routed experts
- **Shared Experts**：2 个（所有 token 共享）
- **MoE 隐藏维度**：每个 expert 3072
- **Latent MoE 维度**（路由空间）：3584
- **总参数量**：2.8T / **激活参数量**：104B（约 2.7% 激活率）

Kimi K3 的 MoE 计算路径为：每个 token 先经过 Shared Experts 的固定处理，然后通过 Router 选择 16/896 个 routed experts 进行稀疏计算。Route 过程使用 Latent MoE 维度（3584）作为路由空间，将 token 表征投影到 896 维的专家选择向量上。

为确保在极高稀疏度下的训练稳定性，Stable LatentMoE 引入了以下技术：

1. **SiTU-GLU 激活函数**：Stable LatentMoE 中的每个 FFN expert 使用 SiTU-GLU 而非标准的 SwiGLU 或 ReLU。SiTU-GLU 结合了 SwiGLU 的平滑门控特性和 Tanh Unit（TU）的数值稳定性，在 2.8T 规模下保持梯度健康。其定义为：
   $$\text{SiTU-GLU}(x) = \text{SiLU}(x W_1) \odot \text{Tanh}(x W_2)$$

2. **Quantile Balancing**：用于控制 896 个 routed experts 间负载均衡的新型方法。传统 MoE 使用辅助负载均衡损失（auxiliary loss），如 switch loss 或 z-loss，但这些方法存在损失权重调参困难和与主损失竞争的问题。Quantile Balancing 基于专家被选中的概率的分位数统计，使高负载专家的选择概率自动衰减、低负载专家的选择概率自动增加，无需辅助损失项。

3. **Layer Normalization 适配**：在 MoE 层的前后分别插入 RMSNorm，确保经过稀疏计算后分布的稳定性。96.5% 的 MoE 层路由到的专家分布相对均匀，Quantile Balancing 将路由不平衡控制在 0.05 以内。

4. **专家粒度分析**：896 个 routed experts 中，部分专家在训练中形成了隐式的领域专业化趋势——某些专家在数学推理 token 上更频繁被选中，另一些则在代码 token 上被激活更多。这种自组织的专业化是 MoE 架构的核心优势，Stable LatentMoE 通过 Quantile Balancing 确保这种专业化不会退化为专家退化（dead experts）。

### 2.4 MoonViT-V2 视觉编码器

Kimi K3 原生支持视觉输入，其视觉编码器 MoonViT-V2 拥有 **4.01 亿参数**，嵌入在模型输入层，使文本和视觉信息在 Transformer 的早期阶段即可融合。

### 2.5 模型配置汇总

| 配置项 | 数值 |
|:------|:----:|
| 架构 | MoE |
| 总参数量 | 2.8T |
| 激活参数量 | 104B |
| 层数 | 93（1 个 Dense 层 + 92 个 MoE 层） |
| Attention 组成 | 69 KDA + 24 Gated MLA |
| Attention 隐藏维度 | 7168 |
| Attention Head 数 | 96 |
| Latent MoE 维度 | 3584 |
| MoE 隐藏维度（每 expert） | 3072 |
| Routed Experts | 896 |
| 每 token 激活 Experts | 16 |
| Shared Experts | 2 |
| 词表大小 | 160K |
| 上下文长度 | 1,048,576 |
| 视觉编码器参数 | 401M |
| 量化方式 | MXFP4 权重 / MXFP8 激活（量化感知训练） |
| 模态 | 文本、图像 |

### 2.6 MXFP4 量化

Kimi K3 从 SFT 阶段开始应用量化感知训练，使用 **MXFP4 权重**搭配 **MXFP8 激活**。这种量化方案在保证推理精度的同时大幅降低内存占用和推理成本，并且具有广泛的硬件兼容性。

## 第 3 章 预训练与数据策略

### 3.1 Scaling 效率提升

Kimi K3 在预训练阶段实现了约 **2.5× 的 scaling 效率提升**（相对于 Kimi K2）。这意味着在相同的计算预算下，Kimi K3 能达到比 Kimi K2 高出 2.5 倍的有效模型能力。这一提升来源于三个方面的综合贡献：

1. **KDA 序列混合效率**：KDA 以 $O(L)$ 而非 $O(L^2)$ 的复杂度处理长序列，使得在固定计算预算下可以将更多资源分配给模型宽度（更多专家）和深度（更多层），而非花费在注意力矩阵计算上。
2. **Stable LatentMoE 激活效率**：896 个 routed experts 中仅激活 16 个（约 1.8% 激活率），使得 2.8T 总参数对应的计算成本仅相当于 104B 密集模型。与传统 MoE（如 Mixtral 8×7B 的 2/8、25% 激活率）相比，Kimi K3 的激活率更低，稀疏度更高，但通过 Quantile Balancing 避免了专家退化。
3. **精细化的数据配方**：训练数据的配比（各领域、各语言、各难度级别）经过系统化优化，确保每个训练 token 的信息密度最大化。

### 3.2 训练数据策略

虽然论文未公开具体的训练数据配比细节，但从模型的能力特点可以推断出数据策略的关键方向：

- **多语言覆盖**：160K 词表支持高压缩率的多语言文本处理
- **代码数据高占比**：模型在编码领域的突出表现（SWE-Marathon 42.0%、ProgramBench 77.8%）暗示代码数据在训练中占据显著比例
- **多模态对齐数据**：MoonViT-V2 的 4.01 亿参数需要通过大量图文对数据训练，以实现文本空间与视觉空间的深度对齐
- **长上下文数据**：100 万 token 上下文窗口的实现依赖包含超长文档、代码仓库和多轮对话数据的长序列训练

### 3.3 训练基础设施

2.8T 参数量的 MoE 预训练对计算基础设施提出了极高要求。MoonEP 系统确保了在超大规模下训练的有效性和效率（详见第 6 章）。训练过程中的关键难题包括：

1. **模型并行**：模型权重分布在数百个 GPU 上，每个 GPU 承载部分 expert 和 attention 层的计算。
2. **数据并行**：训练数据在数据并行维度上独立采样，但梯度在 expert parallel 维度上聚合。
3. **长序列训练稳定性**：百万 token 序列的梯度方差远大于短序列，需要定制化的学习率调度和梯度裁剪策略。
4. **多模态训练协调**：视觉编码器和语言主干的训练步调不同，需要异步或交替更新策略。

KDA 的系统协同设计（融合 kernel、上下文并行、prefix caching）直接将算法效率转化为训练吞吐量的提升，据论文报告，KDA 在长序列场景下的吞吐量相比标准注意力实现有数倍提升。


## 第 4 章 后训练


### 4.1 多域多强度 RL 训练体系

#### 4.1.1 多推理强度（Multi-Effort）训练

Kimi K3 的后训练在 RL 过程中引入了**推理努力级别**（thinking effort levels）的概念。模型在训练时就已经覆盖了完整的推理强度谱系，支持 low、medium、high、max 四个级别（Kimi K3 支持子集）。训练时，每个训练样本关联一个目标 effort 级别，模型在该级别下生成思考过程的 token 预算相应变化。

这种 effort-conditioned 训练的关键优势在于：**训练时覆盖的 effort 谱系使推理时可以选择最合适的强度**。对于简单任务使用 low effort 可节省推理成本，对于复杂推理任务使用 max effort 可获得最佳性能。模型在低 effort 下学会快速、直接的回答模式，在 max effort 下学会深度、迭代的推理模式。

Effort 级别在推理时通过上下文中的 thinking-effort option message（而非修改生成前缀或 token 预算）指定。模型读取该消息后自动调整推理深度，与训练时的 effort-conditioned 行为保持一致。

#### 4.1.2 多教师 On-Policy 蒸馏

在各领域和各 effort 级别上独立训练会得到多个专用策略模型。将这些能力合并为统一模型的方法是**多教师 on-policy 蒸馏**（multi-teacher on-policy distillation）：

1. 各领域 RL 训练产出领域专家模型（编码专家、Agent 专家、推理知识专家等）
2. 每个领域内按 effort 级别产出 effort 专家（low-effort 编码专家、max-effort 编码专家等）
3. 使用这些专家模型作为教师，在目标模型（学生）的 on-policy 采样数据上进行蒸馏训练
4. 蒸馏目标是最小化学生模型输出与教师模型输出的 KL 散度，同时保持学生模型的自主推理能力

这种方式使得单个 Kimi K3 模型能够在多个维度上达到平衡的能力，无需为每个领域部署独立的模型实例。

### 4.2 Agentic RL 环境

Kimi K3 的后训练环境覆盖了广泛的任务类型，形成了通用的「推理 → 行动 → 观察 → 验证 → 适应」（reason-act-observe-verify-adapt）循环：

- **可验证搜索与专业知识工作**：基于检索的知识密集型任务，模型需搜索、阅读、综合多个信息源
- **软件工程与内核优化**：GPU kernel 开发、编译器开发、芯片设计、操作系统调试等涉及数百次编译运行的场景
- **多模态视觉推理**：结合视觉-工具循环推理，模型在观察图像后调用工具获取更多信息或执行操作
- **持续助手工作流**：长期陪伴式对话，保持上下文一致性和用户偏好适配
- **自主 Web 开发**：从需求分析到代码生成、部署、测试的完整 Web 应用开发生命周期

这些任务的共同特点是长期期性（long-horizon），通常涉及数百或数千次工具调用，累计上下文超过百万 token。

### 4.3 Agentic RL 的奖励设计

Kimi K3 的后训练 RL 奖励信号设计遵循**可验证奖励**（verifiable rewards）原则：

- **编码任务**：通过编译成功、测试通过率、代码效率指标等客观可验证信号
- **Agent 任务**：通过任务完成度、工具调用正确性、最终结果质量等自动化评估
- **推理任务**：通过答案正确性（数学）、逻辑自洽性、引用完整性等

奖励信号按 effort 级别差异化：max effort 下对推理过程的完整性有更高奖励，low effort 下对响应速度和简洁性给予更高权重。

### 4.4 推理 Effort 与思考模式的设计细节

Kimi K3 的消息结构采用**通道分离**设计（受 Harmony response format 启发）：

- **think 通道**：承载推理过程（reasoning trace）
- **response 通道**：承载用户可见的回答
- **tools 通道**：承载工具调用

两个生成模式通过生成前缀区分——`[open]think[sep]` 启动思考模式、`[open]response[sep]` 启动指令模式——而非通过分离的模板。Kimi K3 仅支持**保留思考模式**（preserved thinking mode）：在思考模式下，think 通道始终保留在历史中，即使内容为空也保留，使模型在多轮对话中观察到一致的消息结构。

上下文消息还进一步分为**输入消息**（用户/系统/助手/工具角色）、**选项消息**（option messages，如工具声明、effort 设置）。选项消息在上下文中的位置反映其作用域：全局选项（工具声明和 effort 设置）出现在所有输入消息之前，一次性选项（tool_choice、response_format）在输入消息之后。

## 第 5 章 实验评估

Kimi K3 在多个 benchmark 上与最强闭源模型（Claude Fable 5、GPT-5.6 Sol）及开源/其他闭源模型（Claude Opus 4.8、GPT-5.5、GLM-5.2）进行了全面对比。以下结果均设置为最高推理强度（max / xhigh）。

### 5.1 推理与知识

| Benchmark | Kimi K3 (max) | Claude Fable 5 (max) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) | GLM-5.2 (max) |
|:----------|:------------:|:------------------:|:-----------------:|:--------------------:|:---------------:|:------------:|
| GPQA Diamond | **93.5%** | 92.6% | 94.1% | 91.0% | 93.5% | 91.2% |
| CritPt | 23.4% | 28.6% | **32.3%** | 20.9% | 27.1% | 20.9% |
| AA-LCR | **74.7%** | 70.0% | 73.7% | 67.7% | 74.3% | 71.3% |
| HLE-Full | 43.5/56.0% | **53.3/63.0%** | 44.5/58.0% | 49.8/57.9% | 41.4/52.2% | — |

Kimi K3 在 GPQA Diamond 和 AA-LCR 上达到最高或接近最高水平，但在 CritPt 和 HLE-Full 上落后于 Claude Fable 5。HLE-Full 分数为「无工具/有工具」两种设置的结果。

### 5.2 编码

| Benchmark | Kimi K3 (max) | Claude Fable 5 (max) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) | GLM-5.2 (max) |
|:----------|:------------:|:------------------:|:-----------------:|:--------------------:|:---------------:|:------------:|
| DeepSWE | 67.5% | 70.0% | **73.0%** | 59.0% | 67.0% | 46.2% |
| ProgramBench | **77.8%** | 76.8% | 77.6% | 71.9% | 70.8% | 63.7% |
| Terminal-Bench 2.1 | 88.3% | 88.0% | **88.8%** | 84.6% | 83.4% | 82.7% |
| FrontierSWE | 81.2% | **86.6%** | 71.3% | 66.7% | 64.9% | 67.3% |
| SWE-Marathon | **42.0%** | 35.0% | 39.0% | 40.0% | 14.0% | 13.0% |
| PostTrainBench | 36.6% | **41.4%** | 34.6% | 34.1% | 28.4% | 34.3% |
| MLS-Bench-Lite | 48.3% | **49.9%** | 46.2% | 42.8% | 35.5% | 40.4% |
| SciCode | 58.7% | **60.2%** | 56.1% | 53.5% | 56.1% | 50.5% |
| Kimi Code Bench 2.0 | 72.9% | **76.9%** | 64.8% | 71.7% | 69.0% | 64.2% |

Kimi K3 在编码领域表现强劲：在 ProgramBench、FrontierSWE 和 SWE-Marathon 上取得最高分，尤其在 SWE-Marathon 上大幅领先（42.0% vs 第二名 40.0%）。FrontierSWE 上 Kimi K3 的 81.2% 远超 GPT-5.6 Sol 的 71.3%。Claude Fable 5 在 DeepSWE、MLS-Bench-Lite 和 SciCode 上仍然领先。

### 5.3 Agent 能力

| Benchmark | Kimi K3 (max) | Claude Fable 5 (max) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) | GLM-5.2 (max) |
|:----------|:------------:|:------------------:|:-----------------:|:--------------------:|:---------------:|:------------:|
| BrowseComp | **91.2%** | 88.0% | 90.4% | 84.3% | 84.4% | — |
| DeepSearchQA (F1) | **95.0%** | 94.2% | — | 93.1% | — | — |
| ResearchRubrics | **76.2%** | — | 73.8% | 73.5% | 64.0% | 71.1% |
| GDPval-AA v2 (Elo) | 1686 | **1747** | 1736 | 1593 | 1491 | 1510 |
| Toolathlon-Verified | 76.5% | **77.9%** | 74.9% | 76.2% | 73.5% | 59.9% |
| MCPMark-Verified | **94.5%** | 87.4% | 92.9% | 76.4% | 92.9% | — |
| MCP-Atlas | 84.2% | **84.7%** | 83.6% | 83.6% | 82.8% | 82.6% |
| AutomationBench | **30.8%** | 29.1% | 29.7% | 27.2% | 22.7% | 12.9% |
| JobBench | 54.3% | **57.4%** | 45.4% | 48.4% | 38.3% | 43.4% |
| AA-Briefcase (Elo) | 1548 | **1583** | 1495 | 1354 | 1158 | 1260 |
| Agents' Last Exam | 28.3% | 25.7%† | **29.6%** | 27.0% | 26.6% | 20.4% |
| APEX-Agents | 41.0% | **43.3%** | 39.9% | 39.4% | 38.5% | 35.6% |
| OfficeQA Pro | 63.3% | **69.9%** | 63.2% | 63.9% | 60.9% | 41.4% |
| SpreadsheetBench 2 | **34.8%** | 34.7% | 32.4% | 31.6% | 29.1% | 28.1% |
| OSWorld-Verified | 84.8% | **85.0%** | 83.0% | 83.4% | 79.0% | — |
| OSWorld 2.0 | 58.3% | **66.1%** | 62.6% | 55.7% | 49.5% | — |
| SaaS-Bench | 60.1% | — | **61.4%** | 56.1% | 43.8% | — |
| τ³-Banking | **33.4%** | 26.8% | 33.0% | 27.6% | 31.3% | 26.8% |
| Harvey Lab-AA | **94.6%** | 93.6% | 87.2% | 91.1% | 86.3% | 91.0% |
| CorpFin v2 | 71.6% | **71.8%** | 64.4% | 66.7% | 68.4% | 66.1% |
| Finance Agent v2 | 54.4% | **56.3%** | 53.8% | 53.9% | 51.8% | 49.7% |
| Legal Research Bench | 44.2% | **49.5%** | 48.1% | 43.8% | 40.4% | 31.3% |

Kimi K3 在 Agent 领域表现突出：在 BrowseComp（91.2%）、DeepSearchQA（95.0% F1）、ResearchRubrics（76.2%）、MCPMark-Verified（94.5%）、AutomationBench（30.8%）、SpreadsheetBench 2（34.8%）、τ³-Banking（33.4%）、Harvey Lab-AA（94.6%）等多个基准上取得最高分。Claude Fable 5 在 GDPval、Toolathlon、JobBench、AA-Briefcase、OSWorld 2.0、CorpFin v2 等方面仍保持领先。

### 5.4 视觉

| Benchmark | Kimi K3 (max) | Claude Fable 5 (max) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) |
|:----------|:------------:|:------------------:|:-----------------:|:--------------------:|:---------------:|
| WorldVQA ForceAnswer | 51.0% | **56.7%** | 41.8% | 39.1% | 38.5% |
| OmniDocBench | **91.1%** | 89.8% | 85.8% | 87.9% | 89.4% |
| PerceptionBench | **58.5%** | 57.2% | 59.7% | 47.2% | 55.8% |
| Video-MME (w/ sub) | **90.0%** | — | 89.5% | 86.0% | 89.3% |
| MMVU | **82.1%** | — | 81.2% | 79.2% | 81.7% |
| BabyVision w/ python | 85.7% | **90.5%** | 88.9% | 81.2% | 83.6% |
| MMMU-Pro | 81.6/83.4% | 81.2/86.5% | **83.0/84.6%** | 78.9/82.7% | 81.2/83.2% |
| CharXiv (RQ) | 84.8/91.3% | **88.9/93.5%** | 84.6/89.1% | 80.5/89.9% | 84.1/89.0% |
| MathVision | 94.3/97.8% | 94.8/98.6% | **95.8/97.8%** | 86.7/97.1% | 92.2/96.8% |
| ZeroBench (pass@5) | 23.0/41.0% | 23.0/**46.0%** | 17.0/35.0% | 17.0/34.0% | 22.0/41.0% |

在视觉领域，Kimi K3 在 OmniDocBench（91.1%）、Video-MME（90.0%）、MMVU（82.1%）上取得最高分。多模态基准（MMMU-Pro、CharXiv、MathVision、ZeroBench）的每格报告为「无工具/有工具（Python）」两种设置的结果。Claude Fable 5 在 WorldVQA、BabyVision、CharXiv 和 ZeroBench 上领先。

### 5.5 综合评估结论

Kimi K3 在多个领域展现了前沿水平的表现。其优势领域包括：

1. **Agent 任务**：在 BrowseComp、DeepSearchQA、ResearchRubrics、MCPMark、AutomationBench 等 10+ 个 Agent 基准上取得最高分，展现了强大的工具使用和长程任务执行能力
2. **编码任务**：在 ProgramBench、FrontierSWE、SWE-Marathon 上取得最高分，尤其在长程软件工程任务上实力突出
3. **视觉理解**：在文档理解（OmniDocBench）、视频理解（Video-MME、MMVU）上领先
4. **推理与知识**：在 GPQA Diamond 和 AA-LCR 上与最强模型持平或领先

Kimi K3 的性能仅略逊于最强的闭源系统——Claude Fable 5 和 GPT-5.6 Sol——但一致地优于评测套件中的所有其他开源和闭源模型。作为开源模型，Kimi K3 建立的性能基线标志着开源社区首次突破 3T 级参数规模，并使开源前沿与闭源前沿的差距显著缩小。

## 第 6 章 基础设施与部署

Kimi K3 的 2.8T 参数规模对基础设施提出了极高的要求。Moonshot AI 在多个层面进行了系统性创新。

### 6.1 KDA System Co-Design

KDA 的算法创新需要配套的系统支持才能在 2.8T 规模上高效运行。Moonshot AI 在系统层面实现了三项核心协同设计：

- **融合 Kernel**（Fused Kernels）：将 KDA 的卷积、线性投影、门控、递推更新等操作合并为单一 GPU kernel，减少 kernel launch 开销和全局内存读写。每个 KDA layer 的计算在单个融合 kernel 中完成，避免中间结果的显存读写。
- **KDA Context Parallelism**：跨设备的注意力上下文并行化策略。与传统序列并行不同，KDA Context Parallelism 将百万 token 上下文均匀分布在多个 GPU 上，每个 GPU 只计算自己负责的片段，通过高效的集合通信在设备间同步必要的隐状态。这使长序列训练可以在有限显存预算下进行。
- **状态感知 Prefix Caching**（State-Aware Prefix Caching）：KDA 的递推计算本质上是状态机——给定前缀状态 $s_{t-1}$ 和新 token $x_t$，新状态 $s_t = f(s_{t-1}, x_t)$。状态感知 Prefix Caching 缓存每个前缀的最终状态，在复用上下文的场景下（如多轮对话中历史消息不变），直接从缓存恢复状态而无需重新计算整个前缀。

### 6.2 MoonEP：专家并行训练

MoonEP（Moonshot Expert Parallelism）是 Moonshot AI 为 2.8T MoE 预训练开发的关键基础设施系统：

- **完美平衡的专家执行**（Perfectly Balanced Expert Execution）：传统 MoE 训练中的 token 路由不均匀导致不同设备上计算量差异显著，造成「straggler」效应拖慢整体训练。MoonEP 通过**静态计算形状**（Static Computation Shapes）确保每个设备上执行相同数量的 expert 计算，无论路由分布如何。具体方法是将每个 expert 视为固定大小的计算单元，在设备间均匀分配。
- **零拷贝通信**（Zero-Copy Communication）：Expert 间数据传输是 MoE 训练的主要通信瓶颈。MoonEP 使用 GPU 直接通信（如 NVLink/NVSwitch）的零拷贝机制，避免中间 CPU 缓冲和额外的显存拷贝。
- **高效内存管理**：在 2.8T 规模下，模型权重、优化器状态和激活值的显存需求远超单 GPU 容量。MoonEP 的显存管理系统将权重和优化器状态分布在设备内存中，仅在需要时交换到计算设备。同时，多模态编码器的专门优化（如图像特征的显存释放策略）进一步降低了训练显存压力。

### 6.3 百万 Token Agentic RL 系统

Kimi K3 的后训练涉及超长上下文（百万 token）的强化学习，为此设计了专门的协定位系统，核心挑战在于 agentic RL 轨迹的长期期性、环境交互的不可重入性和训练资源的连续可用性：

- **部分 Rollout**（Partial Rollouts）：将数百万 token 的完整轨迹拆分为可管理的片段（segments），每个片段独立计算奖励和梯度。如果某个片段失败（如环境错误、超时），仅回滚该片段而非整个轨迹，大幅降低重计算开销。
- **外部 KV-Cache 保留**（External KV-Cache Retention）：在 rollout 片段之间，将模型的 KV-cache 状态保存到 CPU 内存或 NVMe 存储，而非丢弃后重新计算。这使得长达百万 token 的推理轨迹可以在多个训练步骤间保持一致性。
- **自适应节流**（Adaptive Throttling）：根据当前 GPU 集群负载、API 延迟和奖励计算速度动态调整 rollout 的批次大小和并发度。在高峰期降低并发以维持稳定性，在低谷期增加并发以提高吞吐量。
- **可恢复微 VM 沙箱**（Resumable MicroVM Sandboxes）：为每个 agent 实例维护一个轻量级虚拟机的运行时环境。如果训练任务被抢占或中断，可以从最近的 checkpoint 恢复 agent 的执行环境状态（文件系统、进程状态、环境变量），无需从头启动。这对需要长时间运行（如代码编译和测试）的编码任务至关重要。

### 6.4 推理与部署

Kimi K3 的推理部署经过全面优化：

- **MXFP4 量化推理**：MXFP4 权重联合 MXFP8 激活，实现约 4 倍权重压缩。量化感知训练从 SFT 阶段开始应用，确保量化精度损失最小。这种方案在保持推理精度的同时显著降低显存需求，使 2.8T 模型在 8×H20（或类似配置）上可实际部署。
- **Cache 与 Budget 感知的集群调度**：根据各推理节点的 KV-cache 状态、当前负载和剩余计算预算，智能路由请求到最优节点。对于已知前缀的请求（如复用上下文的对话），优先路由到已有缓存状态的节点。
- **支持主流推理引擎**：vLLM（社区已提供 recipes）、SGLang（含 cookbook）、TokenSpeed（Lightseek 提供 recipes）
- **API 服务**：部署于 https://platform.kimi.ai，提供 OpenAI/Anthropic 兼容接口。支持 reasoning_effort 参数（low/high/max）、流式推理、tool calling、structured output 等高级功能。
- **Kimi Code CLI 集成**：Kimi K3 与 Kimi Code CLI 深度集成，用户可在终端中使用 `/model` 命令切换至 Kimi K3。Kimi Code 是 Kimi K3 的官方 Agent 框架，在多个编码 benchmark 的评测中作为默认 harness 使用。

### 6.5 模型使用示例

Kimi K3 的使用接口设计简洁。以下是一个使用 preserved thinking history 的 Python 调用示例：

```python
import openai

client = openai.OpenAI(
    base_url="https://platform.kimi.ai",
    api_key="YOUR_API_KEY"
)

messages = [
    {"role": "user", "content": "Explain the key advantage of KDA over standard attention."}
]

response = client.chat.completions.create(
    model="kimi-k3",
    messages=messages,
    reasoning_effort="max",
    max_tokens=4096
)

# In preserved thinking mode, the response includes:
print(response.choices[0].message.reasoning_content)  # Think channel
print(response.choices[0].message.content)            # Response channel
```

关键要点：多轮对话中必须将 `reasoning_content` 和 `tool_calls` 同时传回，而不仅仅是 `content`。

## 第 7 章 局限性与延伸阅读

### 7.1 局限性

1. **与最强闭源模型的差距**：虽然 Kimi K3 在多个基准上超越 GPT-5.6 Sol 或 Claude Fable 5，但总体性能仍略逊于 Claude Fable 5 和 GPT-5.6 Sol。在 CritPt、HLE-Full 等需要极端推理能力的任务上差距较大。
2. **视觉能力有待提升**：在 WorldVQA ForceAnswer、BabyVision、ZeroBench 等需要精细化视觉理解的基准上，Kimi K3 落后于 Claude Fable 5，暗示 MoonViT-V2 与原生融合视觉编码器尚有改进空间。
3. **长上下文实用评估有限**：虽然模型支持 100 万 token 上下文窗口，但公开的长上下文基准评测有限，实际百万 token 使用场景中的检索精度和事实一致性有待进一步验证。
4. **部署成本**：2.8T 总参数（104B 激活）的 MoE 模型即使在 MXFP4 量化下，部署成本仍然显著高于小模型，可能限制其在边缘设备等资源受限场景的使用。
5. **评估 Harness 差异**：Kimi K3、Claude Fable 5 和 GPT-5.6 Sol 使用不同的评估框架（Kimi Code、Claude Code、Codex），在某些基准上计算配置差异影响可比性。

### 7.2 延伸阅读

Kimi K3 建立在 Moonshot AI 系列技术积累之上，相关论文包括：

- **Kimi K2**（[58]）：Kimi K3 的前代模型，为本次 2.5× scaling 效率提升提供了对比基线
- **Kimi Delta Attention (KDA)**（[63]）：K3 的核心注意力机制论文
- **Attention Residuals (AttnRes)**（[57]）：跨层信息流机制的原始论文
- **Kimi K2.5 Agent Swarm**（[59]）：从顺序推理扩展到并行 Agent 协调的测试时计算
- **DeepSeek-R1**（[40]）：大规模 RL 激发推理能力的开创性工作
- **Kimi K1.5**（[118]）：Moonshot AI 自身在 RL 激发推理方面的工作

模型权重和代码：
- **GitHub**：https://github.com/MoonshotAI/Kimi-K3
- **HuggingFace**：https://huggingface.co/moonshotai/Kimi-K3
- **vLLM Recipes**：https://recipes.vllm.ai/moonshotai/Kimi-K3
- **SGLang Cookbook**：https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3
