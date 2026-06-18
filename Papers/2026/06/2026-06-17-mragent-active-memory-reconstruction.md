# Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents

## 论文元数据
- 标题：Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents
- 作者：Shuo Ji, Yibo Li, Bryan Hooi (National University of Singapore)
- 会议：ICML 2026
- arXiv ID：2606.06036
- 发布日期：2026-06-04
- 官方代码：[GitHub - Ji-shuo/MRAgent](https://github.com/Ji-shuo/MRAgent)
- 关键词：LLM Agent, Memory, Active Reconstruction, Graph Memory, ICML 2026


---


## Ch1: 论文概述与核心贡献

### 核心问题

LLM Agent 在长期交互历史中面临推理困难。尽管大型语言模型在数学计算和逻辑推理方面表现优异，但在需要跨长时间跨度整合信息的任务中存在显著短板——这一现象被称为"锯齿形"认知轮廓（jagged cognitive profile）。现有记忆增强型agent普遍采用静态的"检索-推理"（retrieve-then-reason）流水线：系统基于初始查询一次性完成记忆检索，再在固定检索结果上进行推理。这种刚性的管道设计使得agent无法根据推理过程中发现的中间证据动态调整记忆访问策略。

### 核心观察：记忆是主动重建

认知神经科学研究表明，人类记忆并非被动读取，而是"主动重建"（active reconstruction）。记忆提取过程遵循以下机制：contextual cues（上下文线索）激活 engrams（记忆痕迹）→ 渐进式重建（progressive reconstruction）→ 完整记忆内容。关键在于，记忆重建是一个迭代过程，每一步都依赖当前累积的证据来决定下一步搜索方向，而非基于初始查询一次性完成所有检索决策。

### 三大贡献

#### 1. 主动记忆重建新范式

MRAgent 将记忆访问深度集成到推理循环中。agent在推理的每个步骤根据中间证据动态选择下一步记忆访问动作（forward traversal沿Cue→Tag→Content方向扩展，reverse traversal反向追溯），形成"推理-访问-推理"的闭环。这种设计使记忆检索策略能够随推理状态演化，而非被初始查询锁死。

形式化表达：
- Passive retrieval: $$\pi_{\mathrm{p}}(x)$$ — 仅基于初始查询x选择记忆单元
- Active reconstruction: $$\pi_{\mathrm{a}}^{(t)}(x, \mathcal{S}^{(t-1)})$$ — 基于累积证据$$\mathcal{S}^{(t-1)}$$动态选择第t步动作

论文提供理论证明（Theorem 4.1）：对于任意检索预算T≥2，主动检索策略的假设类严格包含被动检索策略的假设类，即 $$\mathcal{H}^{\mathrm{LM}}_{\mathrm{passive}}(T) \subsetneq \mathcal{H}^{\mathrm{LM}}_{\mathrm{active}}(T)$$。

#### 2. Cue-Tag-Content异构图记忆

记忆被组织为异构图 $$\mathcal{M} = (\mathcal{C}, \mathcal{V}, \mathcal{R})$$：
- **Cues ($$\mathcal{C}$$)**: 细粒度关键词（实体、属性、时间戳等）
- **Contents ($$\mathcal{V}$$)**: 具体记忆单元（对话片段、事件记录）
- **Tags (ℛ ⊆ 𝒞 × 𝒢 × 𝒱)**: 带类型的关联边，将cues连接到contents

Tags作为语义桥梁，实现两阶段检索：
1. 基于当前cue激活相关tags: $$\phi_{c \to g}$$
2. 基于tag+cue检索content: $$\phi_{(c,g) \to v}$$

这种结构的核心优势在于将"相似度匹配"转化为"关联遍历"——agent不必预知目标内容的精确表示，只需沿着语义关联逐步探索，即可在推理过程中重建相关记忆路径。

#### 3. 多粒度记忆层次

受认知神经科学启发，记忆组织为三个互补层次：
- **Episodic Layer**: 事件特定记忆（对话片段、交互记录），带时间戳
- **Semantic Layer**: 稳定知识（用户偏好、个人事实、长期属性）
- **Abstraction Layer (Topics)**: 跨episodes的重复模式（高频话题、主题簇）

这种分层结构使得agent既能快速访问稳定事实（语义层），又能追踪对话上下文的时序演变（情节层），还能识别跨会话的主题规律（抽象层）。

### 关键结果速览

LOCOMO benchmark（50个长期对话，~35个会话，~300轮对话，~200个QA问题）：
- MRAgent (Gemini-2.5-Flash): J score 84.21
- Mem0 (最优baseline): J score 68.31
- 相对提升：23.3%

LOCOMO (Claude-Sonnet-4.5):
- MRAgent: J score 88.32
- LangMem: J score 78.61
- 相对提升：12.4%

LongMemEval benchmark（~500个问题，~115K tokens历史）：
- MRAgent (Gemini): J score 72.95
- MRAgent* (Claude): J score 86.76
- 相对baseline提升：32%

效率优势：
- Token消耗: MRAgent 118k vs A-Mem 632k（降低81.3%）
- 运行时: MRAgent 586.11s vs Mem0 533.29s vs LangMem 1209.57s

### 创新与传统方法对比

| 维度 | 传统方法（RAG/Mem0/A-Mem） | MRAgent |
|------|---------------------------|---------|
| 检索范式 | 被动检索：基于初始查询一次性选择top-k | 主动重建：基于中间证据迭代调整检索策略 |
| 记忆结构 | 扁平向量库 或 预定义图结构 | Cue-Tag-Content异构图，tags作为语义桥梁 |
| 推理集成 | 检索与推理分离，固定检索结果 | 记忆访问融入推理循环，每步检索依赖当前推理状态 |
| 多跳能力 | N-hop邻居扩展（预定义，固定路径） | 动态路径探索，根据需要终止或深入 |
| 理论保证 | 无 | Theorem 4.1: 主动检索严格强于被动检索 |
| Token效率 | 高（A-Mem 632k） | 低（118k） |


---


## Ch2: 研究背景与动机

### LLM Agent在长期交互中的记忆瓶颈

#### Context Window限制

LLM的上下文窗口有限（主流模型在32K-128K tokens），长期对话历史必然超出单次推理容量。即使使用长context模型（如200K+），将全部历史载入也会导致：
- 注意力分散在无关信息上，降低推理质量
- Token成本线性增长
- 推理延迟增加

#### 历史信息丢失

现有对话系统仅保留最近N轮对话，更早的交互内容被丢弃。这在需要跨时间跨度推理的任务中造成灾难性遗忘：
- 用户在前10轮对话中表达偏好，在第50轮被问及相同话题时，agent无法召回该偏好
- 跨会话的长期记忆（用户属性、历史事件）完全丢失

#### "锯齿形"认知轮廓

LLM在记忆密集型任务上表现远弱于逻辑推理任务。这一"锯齿形"能力曲线表明：单纯提升模型推理能力无法解决长期记忆问题——问题不在推理本身，而在记忆访问机制。

### 现有记忆系统的三类范式及局限性

#### 1. 基于相似度的检索（RAG, MemoryBank, Mem0）

**机制**：将记忆单元embedding为向量，基于查询与记忆的相似度（cosine similarity）选择top-k。

**局限**：
- 静态top-k选择：无法根据中间证据调整检索策略
- 语义漂移：查询的embedding表示可能与目标记忆的表示差异较大（如query用"夏天"检索，记忆中用"July"记录）
- 一次性决策：无法多步适应，必须一次性决定所有检索项

#### 2. 基于图的记忆检索（A-Mem, Zep, MemoryOS）

**机制**：预构建记忆图（节点=记忆单元，边=语义/时间关联），通过N-hop邻居扩展检索。

**局限**：
- 预定义扩展策略：N固定（如2-hop或3-hop），无法根据推理需要动态调整深度
- 噪声累积：每跳扩展都会引入无关节点，固定聚合方式无法区分噪声与信号
- 缺乏灵活性：过度依赖预构建的图结构，无法处理图构建时未预见的新查询模式

#### 3. 共性：被动检索范式

以上方法均为**被动检索**（passive retrieval）：检索策略$$\pi_{\mathrm{p}}(x)$$仅基于初始查询x选择记忆单元，一次决定所有检索项，后续推理过程无法影响记忆访问路径。形式化表达：

$$\pi_{\mathrm{p}}(x) = \{v_1, v_2, \ldots, v_T\}$$

其中T为检索预算，所有$$v_i$$的选择不依赖推理过程中发现的中间证据。

### 被动检索的三项根本缺陷

#### 缺陷(i)：无法基于中间状态调整策略

假设agent在推理第3步时发现关键词"July"，但历史记录中该事件用"2024-07-15"标注，初始查询"夏天活动"无法直接匹配。被动检索系统无法根据"July"这一中间线索重新定向检索，只能在初始top-k结果中搜索（很可能为空），导致推理失败。

主动检索则可以：
- 第1步：基于"夏天活动"检索 → 发现"July"线索
- 第2步：基于"July"激活时间相关tags → 检索"2024-07-15"事件
- 第3步：基于"2024-07-15"检索具体对话片段

#### 缺陷(ii)：固定聚合带来的噪声累积

被动检索将top-k结果一次性聚合（concatenation或summarization），无法区分相关与无关记忆。随着k增加：
- 相关信息占比下降，噪声主导
- LLM注意力分散在无关内容上，降低推理质量
- 即使增加检索预算（k从5增到50），性能反而下降

主动检索通过迭代剪枝解决：每步检索后由LLM判断相关性，仅保留相关分支，无关路径提前终止。

#### 缺陷(iii)：过度依赖预构建结构

基于图的方法假设图构建时已编码所有可能的关联模式，但：
- 图构建规则（如相似度阈值、N-hop策略）需人工预设
- 无法处理未见过的查询类型
- 图更新成本高（需重新embedding/重建索引）

MRAgent的主动遍历不依赖预定义路径：tags作为语义桥梁，LLM根据当前状态动态选择遍历方向（forward或reverse），适应新查询模式。

### 形式化定义：Memory Retrieval as Sequential Decision Process

定义累积证据集：

$$S^{(t)} = \{v^{(1)}, v^{(2)}, \ldots, v^{(t)}\}$$

其中$$v^{(i)}$$为第i步检索的记忆单元。

**被动检索策略**：

$$\pi_{\mathrm{p}}(x) = \{v^{(1)}, v^{(2)}, \ldots, v^{(T)}\}$$

所有$$v^{(i)}$$的选择基于初始查询x，与中间证据无关。

**主动重建策略**：

$$\pi_{\mathrm{a}}^{(t)}(x, S^{(t-1)})$$

第t步检索决策基于初始查询x和累积证据$$S^{(t-1)}$$，每一步都依赖前序步骤发现的线索。

关键区别：被动检索是无状态的（stateless），主动重建是有状态的（stateful）。主动重建的决策空间包含所有被动检索策略，且严格更大（论文Theorem 4.1证明）。

### 认知神经科学启发的记忆模型

人类记忆提取的"标准模型"（standard model of memory consolidation）：

1. **Encoding**: 信息输入 hippocampus，形成短期记忆
2. **Consolidation**: 通过replay和整合，记忆转移到 neocortex，形成长期记忆
3. **Retrieval**: contextual cues激活engrams → 渐进式重建 → 完整记忆内容

关键特征：
- **Associative nature**: 记忆通过关联网络访问，cues激活相关engrams，而非直接"读取"记忆内容
- **Reconstructive process**: 记忆内容在提取时重建，每次提取可能略有不同（受当前context影响）
- **Progressive refinement**: 从模糊线索到具体内容，逐步细化

这一机制启示：
- 记忆不应是"存储-读取"系统，而是"关联-重建"系统
- 访问策略应依赖当前context和已激活线索，而非固定查询
- 需要中间关联结构（类比engrams）连接cues与content

### Cue-Tag-Content架构的必要性

基于上述认知模型，MRAgent提出Cue-Tag-Content（CTC）异构图：
- **Cues**: 对应contextual cues，细粒度可观测线索
- **Contents**: 对应记忆内容，具体记忆单元
- **Tags**: 对应engrams，作为中间关联结构

Tags的核心作用：
1. **语义桥梁**：cues与contents的粒度不同，直接匹配困难（如cue"夏天"与content"2024-07-15下午三点在公园跑步"）。Tags作为中间层，提供语义关联（"时间标签"、"活动标签"）。
2. **引导遍历**：LLM基于当前状态选择遍历方向（激活哪些tags，沿哪些tags访问contents），实现主动重建。
3. **避免组合爆炸**：若直接构建cue→content二部图，边数会随cue和content数量乘积增长。Tags作为聚合节点，降低边数，控制图规模。

CTC架构使MRAgent能够实现认知神经科学中的记忆重建机制：从cues出发，沿tags逐步激活相关contents，形成动态的、状态依赖的记忆访问路径。


---


## 第3章：Cue–Tag–Content 图记忆架构

## 3.1 Cue–Tag–Content 关联记忆

MRAgent 的核心记忆结构是一个异构图 (heterogeneous graph)，形式化为 $$\mathcal{M} = (\mathcal{C}, \mathcal{V}, \mathcal{R})$$，其中：

- **Cues ($$\mathcal{C}$$)**：细粒度的关键词索引单元，包括实体、属性、时间戳、描述符等。例如在对话历史中，cues 可能是 "咖啡"、"周五"、"下午3点"、"喜欢" 等词汇。这些 cues 作为记忆检索的入口点，它们本身不包含完整的语义信息，而是指向特定内容的索引标记。

- **Contents ($$\mathcal{V}$$)**：具体的记忆项目，即实际需要存储和检索的信息单元。在对话场景中，content 可能是完整的对话段落 "用户说周五下午3点想喝咖啡"。Contents 是记忆系统的原子存储单元，在检索时被整体访问。

- **Tags ($$\mathcal{R} \subseteq \mathcal{C} \times \mathcal{G} \times \mathcal{V}$$)**：类型化的三元组关系，编码 cues 与 contents 之间的关联。每个 tag 表示一个 cue 与一个 content 通过特定类型 g 建立的关联。类型 g 可以是 "实体关系"、"时间属性"、"偏好表达" 等语义标签。例如，tag ("咖啡", "prefer Friday afternoon3pm", "用户说周五下午3点想喝咖啡") 将 cue "咖啡" 通过时间偏好类型连接到具体内容。

这种三元组结构的关键创新在于引入 **Tags 作为可选的中间语义桥梁**。传统记忆系统通常采用直接的 cue→content 映射（如键值对或向量相似度），当 cues 和 contents 数量很大时，所有可能的关联会呈指数级增长。CTC 架构通过插入 tags 层，将原本的全连接问题分解为两个受控的子问题：(1) 哪些 cues 与哪些 tags 相关；(2) 哪些 tag-content 对是有效的。这种分解避免了组合爆炸，同时保留了细粒度的关联能力。

### 3.1.1 映射算子

CTC 架构定义两种核心映射算子来支持记忆访问：

**$$\phi_{c \to g}(c) \triangleq \{g \mid (c, g, \cdot) \in \mathcal{R}\}$$**

从 cue 激活关联 tags 的算子。给定一个 cue（例如 "咖啡"），该算子返回所有通过该 cue 可访问的 tag 类型（例如 "时间偏好"、"消费记录"、"口味偏好"）。这允许 agent 在不知道具体内容的情况下，先探索与当前 cue 相关的所有关联维度。

**$$\phi_{(c,g) \to v}(c, g) \triangleq \{v \mid (c, g, v) \in \mathcal{R}\}$$**

从 cue+tag 组合检索具体 contents 的算子。给定 cue 和选定的 tag 类型（例如 ("咖啡", "时间偏好")），该算子返回所有匹配该三元组的 content 节点。这实现了精确定向检索，避免了从单个 cue 泛滥式地返回所有相关内容。

### 3.1.2 两阶段检索流程

基于上述算子，CTC 架构实现两阶段检索流程：

**阶段 1：Tag 选择**

LLM 分析当前查询和推理上下文，从激活的 cues 出发，利用 φ_{c→g} 算子获得所有候选 tags。然后 LLM 根据语义相关性选择其中最相关的 tag 子集。例如，对于查询 "用户通常在什么时间喝咖啡？"，从 cue "咖啡" 激活的 tags 可能包括 "时间偏好"、"消费记录"、"口味偏好"，LLM 会优先选择 "时间偏好" tag。

**阶段 2：Content 访问**

基于选定的 tags，LLM 调用 φ_{(c,g)→v} 算子检索具体的内容节点。由于 tag 已经过第一轮筛选，返回的 contents 具有更高的语义针对性。继续上述例子，通过 ("咖啡", "时间偏好") 检索会返回所有与咖啡消费时间相关的对话段落，而排除口味、价格等其他维度的信息。

这种两阶段设计的核心优势在于 **Tags 作为可选的语义路由器**。当记忆图规模较小时，agent 可以跳过 tag 层直接从 cue 到 content；当记忆图扩展到大规模时，tags 提供必要的中间抽象层，避免每对 cue-content 都需要显式连接。这种可扩展性使 CTC 架构能够同时处理少量短对话和大规模长对话历史。

## 3.2 多粒度记忆层

CTC 图记忆架构进一步组织为三个互补的记忆层，灵感来自认知神经科学中人类记忆系统的分层结构：

### 3.2.1 Episodic Layer（事件层）

事件层存储具体的、按时间线组织的对话事件记忆。每个事件节点 e_i ∈ V_e 表示一个离散的对话单元，例如 "用户在第三轮对话中询问了航班延误政策"。事件层具有以下特性：

- **时序组织**：事件按时间戳排序，支持时间约束的查询（例如 "用户上周二说了什么？"）。时序信息通过时间戳 cues 编码在图中，agent 可以利用 φ 算子执行时间范围的过滤。

- **细粒度内容**：事件节点保留原始对话的详细内容，包括用户陈述、系统响应、上下文信息等。这确保了记忆检索能够恢复足够的信息来回答具体问题。

- **多路索引**：每个事件通过多个 cues 连接到图的不同部分。例如同一航班咨询事件可能通过 "航班"、"延误"、"政策"、"退款" 等多个 cues 索引，支持从不同角度的访问路径。

事件层对应认知神经科学中的情节记忆 (episodic memory)，负责存储自传式事件和经历。在 MRAgent 中，这是记忆系统的基础层，所有其他层次的信息都从事件层抽象或衍生。

### 3.2.2 Semantic Layer（语义层）

语义层存储稳定的、去上下文化的知识单元。每个语义节点 s_i ∈ V_s 表示从多次对话中提炼出的通用知识，例如 "用户的家乡是上海"、"用户偏好素食"、"用户是高级会员"。语义层的特性包括：

- **Aspect-Level Tags**：语义知识通过 aspect-level 类型的 tags 锚定到实体。与事件层的时间戳 cues 不同，语义层的 tags 表示属性维度（例如 "地理信息"、"饮食偏好"、"会员等级"）。这些 tags 使 agent 能够按属性维度检索相关知识，而不受时间限制。

- **跨事件聚合**：语义节点不是单次对话的记录，而是从多个相关事件中提炼的稳定模式。例如，用户在三次不同对话中提到 "我不吃肉"，系统会在语义层创建 "用户偏好素食" 节点，并将其关联到 "饮食偏好" tag。

- **事实性陈述**：语义层存储明确可验证的事实，而非主观体验。这区分了语义记忆（客观知识）和情节记忆（主观经历），符合认知科学的理论框架。

语义层对应认知神经科学中的语义记忆 (semantic memory)，负责存储关于世界的通用知识和个人属性。在 MRAgent 中，语义层提供了高效的查询捷径：当问题询问用户属性时，agent 可以直接从语义层检索，而不需要遍历所有历史事件。

### 3.2.3 Abstraction Layer（抽象层/主题层）

抽象层（论文中也称为 Topics 层）存储跨事件的高层次主题和重复模式。每个主题节点 τ_i ∈ V_τ 总结了多个事件共享的主题，例如 "用户频繁询问国际货运流程"、"用户关心环保包装选项"。抽象层的特性包括：

- **模式归纳**：主题节点不是原始对话的复制，而是从相关事件簇中归纳的共同主题。例如，当 agent 检测到用户在多次对话中询问退货流程时，会创建 "退货流程咨询" 主题节点。

- **连接相关事件**：主题节点作为 hub，连接所有属于该主题的事件节点。这支持 top-down 推理：当查询与某主题相关时，agent 可以先定位主题节点，然后通过主题连接访问所有相关事件，而不需要从 cues 开始逐层检索。

- **动态更新**：抽象层随着新事件的加入而动态演进。当新模式出现或旧模式不再相关时，主题节点会被创建、合并或删除，保持抽象层的时效性。

抽象层在认知神经科学中没有直接对应物，但类似于图式 (schema) 或脚本 (script) 概念，即组织化的事件知识结构。在 MRAgent 中，抽象层提供了高层索引，加速对大规模对话历史的检索。

### 3.2.4 三层协作机制

三个记忆层通过专门的 φ 算子连接，形成层次化的检索网络：

**$$\phi_{\tau \to e}(\tau)$$**：从主题节点下沉到相关事件节点。给定一个主题（例如 "货运咨询"），该算子返回所有属于该主题的事件。这支持 top-down 推理：agent 先识别问题主题，再访问相关具体事件。

**$$\phi_{e \to \tau}(e)$$**：从事件节点上升到相关主题节点。给定一个事件，该算子返回该事件所属的所有主题。这支持 bottom-up 抽象：当新事件出现时，agent 可以识别其主题并将其连接到现有主题节点。

**$$\phi_{e \to s}(e)$$**：从事件节点提炼到语义节点。给定一个事件，该算子检查是否包含稳定的个人属性或偏好，如果发现则更新或创建对应的语义节点。例如，从事件 "用户说 '我下周要去上海'" 中提炼 "用户所在地：上海" 语义知识。

**$$\phi_{s \to e}(s)$$**：从语义节点定位到支持性事件。给定一个语义知识（例如 "用户偏好素食"），该算子返回所有支持该知识的事件证据。这支持知识溯源：当 agent 检索语义知识时，可以同时访问原始事件以提供上下文或验证。

通过这些跨层算子，agent 可以在不同粒度之间灵活切换。对于查询 "用户为什么认为航班延误政策不公平？"，agent 可以：从 "航班" cue 激活相关 tags → 检索包含该 cue 的所有事件 → 识别这些事件的主题（如 "延误投诉"）→ 从语义层检索用户的情绪模式 → 最终给出综合回答。这种多粒度访问使 MRAgent 能够在保持效率的同时，回答需要跨时间、跨主题整合的问题。

## 3.3 基于LLM蒸馏的记忆填充

CTC 图记忆架构的构建不依赖人工标注，而是通过自动化流水线从对话历史中蒸馏信息。该流水线利用 LLM 的理解能力，将非结构化对话转化为结构化的三元组关系。

### 3.3.1 记忆填充流水线

记忆填充过程包含以下步骤：

**步骤 1：输入流分段**

连续的对话流被分割为离散的事件单元 e_i。分割可以基于对话轮次、时间间隔或主题变化。每个事件单元包含该时间窗口内的所有用户陈述、系统响应和上下文信息。在 MRAgent 的实现中，一个事件单元通常对应一次完整的用户会话或一个对话段落。

**步骤 2：LLM 提取 Tags 和 Cues**

对每个事件单元 e_i，LLM 执行结构化提取：
- **Tags 提取**：g_i = F_tag(e_i) 提取事件的语义类型标签。例如，从事件 "用户询问航班延误后的改签费用" 中，LLM 提取 tag "费用咨询"、"航班服务"。
- **Cues 提取**：C_i = F_cue(e_i) 提取细粒度的关键词索引。例如，从同一事件中提取 cues "改签"、"费用"、"延误"、"航班"。

这种提取利用 LLM 的零样本或少样本能力，通过 prompt 指定输出格式（JSON 或列表）来实现结构化提取。提取的 tags 和 cues 遵循统一的 schema，确保跨事件的一致性。

**步骤 3：语义单元提取**

对于需要语义层知识的事件，LLM 进一步执行稳定性判断和属性提炼：
- **稳定性判断**：LLM 评估事件中包含的信息是否为稳定事实（如用户家乡）或临时状态（如用户当前位置）。只有稳定事实被提取到语义层。
- **属性提炼**：对于稳定事实，LLM 提取属性-值对，例如 ("籍贯", "上海")、("会员等级", "Gold")。

这些语义单元 s_i 被关联到 aspect-level tags（如 "地理信息"、"会员状态"），与事件层区分开来。

**步骤 4：主题节点生成**

主题节点通过总结共享主题的事件簇生成：
- **相似事件聚类**：系统周期性分析最近添加的事件，通过 tags 和 cues 的相似度识别事件簇。
- **主题归纳**：对于每个事件簇，LLM 生成一个主题描述 τ_i，总结该簇的共享主题。例如，从多次航班咨询事件中归纳主题 "国际货运流程咨询"。
- **主题连接**：新创建的主题节点被连接到所有支持它的事件节点，通过 $$\phi_{\tau \to e}$$ 和 $$\phi_{e \to \tau}$$ 算子实现双向访问。

主题生成可以是增量式的（每次新事件后检查是否需要创建新主题）或批处理式的（定期分析积累的事件）。

### 3.3.2 Cue-Tag-Event 关系构建

通过上述流水线，系统最终构建出记忆图的 episodic 层，其核心是 Cue-Tag-Event 三元组关系：

- 每个 Event e_i 通过多个 Tags g_i 关联到多个 Cues C_i
- 例如：Event "用户询问改签费用" 通过 Tag "费用咨询" 关联到 Cue "改签"、"费用"、"航班"
- 这些三元组 (c, g, v) 构成图的边集 R

这种结构使记忆图既保留了细粒度的索引能力（通过 cues），又提供了语义组织（通过 tags），同时避免了全连接的组合爆炸。每个事件只需要关联到少量相关的 cues 和 tags，而不是所有可能的索引词。

### 3.3.3 记忆更新策略

记忆图不是静态构建后即固定，而是随着新对话的进行持续更新：
- **增量添加**：新事件通过相同的蒸馏流水线添加到图中，生成新的三元组关系
- **语义层更新**：当新事件支持或修正现有语义知识时，对应的语义节点被更新
- **主题演进**：当新模式出现时创建新主题，当旧主题不再相关时标记为过时
- **冲突解决**：当新信息与现有语义知识冲突时（例如用户改变了偏好），系统通过时间戳保留历史版本，确保语义知识反映最新状态

这种动态更新机制使记忆图能够适应长期对话中的信息变化，保持记忆的时效性和准确性。


---


## 第4章：MRAgent 主动记忆重建机制

## 4.1 重建框架定义

MRAgent 的核心创新在于将记忆检索从静态的 "retrieve-then-reason" 转变为动态的 "active reconstruction"。该过程被形式化为一个受控的马尔可夫决策过程，其中 agent 通过多轮迭代逐步重建与查询相关的记忆内容。

### 4.1.1 重建状态

重建过程在时刻 t 的状态定义为 $$\mathcal{S}^{(t)} = (\mathcal{Z}^{(t)}, \mathcal{H}^{(t)})$$：

- **$$\mathcal{Z}^{(t)}$$**：当前激活的记忆元素集合，包括所有已被访问并保留的 cues、tags 和 content 节点。$$\mathcal{Z}^{(t)}$$ 是记忆图的子图，表示 agent 在时刻 t 的 "工作记忆"。初始状态 $$\mathcal{Z}^{(0)}$$ 包含从查询直接提取的 cues 匹配的存储 cues。

- **$$\mathcal{H}^{(t)}$$**：已累积的推理上下文 (evidence)，即从已访问内容中提取的与查询相关的信息片段。$$\mathcal{H}^{(t)}$$ 是一个文本序列，由 LLM 从 $$\mathcal{Z}^{(t)}$$ 中的 contents 中提炼出来，用于回答查询。初始状态 $$\mathcal{H}^{(0)}$$ 通常为空或仅包含查询本身。

状态 $$\mathcal{S}^{(t)}$$ 的二元结构区分了 "原始记忆数据" ($$\mathcal{Z}^{(t)}$$) 和 "推理结果" ($$\mathcal{H}^{(t)}$$)。这种区分使 agent 能够在推理过程中灵活选择：是直接从原始内容中提取新证据，还是基于已有推理结果进行下一步动作。

### 4.1.2 遍历动作集

agent 可以执行的记忆遍历动作集合定义为 $$\mathcal{A} = \{\Pi_1, \ldots, \Pi_m\}$$。这些动作分为前向遍历和反向遍历两大类：

**前向遍历动作**（从索引到内容）：
- **$$\Pi_{c \to g}$$**：从 cues 激活 tags。给定 $$\mathcal{C}^{(t)}$$ 中的 cues，应用 $$\phi_{c \to g}$$ 算子获取所有关联的 tags。例如，从 cue "咖啡" 激活 tags "时间偏好"、"消费记录" 等。
- **$$\Pi_{(c,g) \to v}$$**：从 cue+tag 检索 contents。给定 cue 和选定的 tag，应用 $$\phi_{(c,g) \to v}$$ 算子获取具体的内容节点。例如，从 ("咖啡", "时间偏好") 检索所有与咖啡消费时间相关的对话段落。

前向遍历对应传统的记忆检索流程，从索引逐渐定位到具体内容。

**反向遍历动作**（从内容发现新索引）：
- **$$\Pi_{v \to (c,g)}$$**：从已检索的 content 反向提取新的 cues 和 tags。给定 $$\mathcal{V}^{(t)}$$ 中的 contents，分析其文本内容，提取其中包含的 cues 和 tags，并将它们加入 $$\mathcal{Z}^{(t+1)}$$。例如，从检索到的对话段落中发现新的实体、时间或属性描述符。

反向遍历是 MRAgent 的关键创新，它使 agent 能够在检索过程中发现新的查询线索，扩展检索路径。这模拟了人类记忆检索中的联想过程：访问一个记忆会触发对相关记忆的联想。

### 4.1.3 组合动作空间

agent 在每个决策步骤可以选择执行动作的任意子集 $$\mathcal{A}^{(t)} \subseteq \mathcal{A}$$。这种组合动作空间使 agent 能够：

- **并行探索**：同时执行多个前向遍历，从多个 cues 激活多条路径。例如，同时从 "航班"、"延误"、"改签" 三个 cues 激活 tags，并在后续步骤中合并结果。
- **前向-反向交替**：在执行前向检索获得内容后，立即执行反向遍历从内容中提取新线索，形成迭代深化。
- **多粒度切换**：在不同记忆层之间切换访问。例如，先在事件层检索具体事件，然后上升到主题层识别相关主题，再从主题层下沉到其他相关事件。

这种灵活的动作空间使 agent 能够根据当前推理状态动态调整遍历策略，而不是固定遵循预定义的检索路径。

## 4.2 记忆重建过程

基于上述状态和动作定义，MRAgent 的记忆重建过程是一个迭代循环，包含初始化、LLM 推理与动作选择、受控记忆遍历、LLM 路由与状态更新、终止判定五个阶段。

### 4.2.1 初始化阶段

重建过程从查询 q 开始，系统执行以下初始化步骤：

**步骤 1：Query Cue 提取**

LLM 从查询中提取关键 cues C_q。例如，对于查询 "用户通常在什么时间喝咖啡？"，LLM 提取 cues {"咖啡", "时间", "通常"}。这些 cues 作为检索的起点。

**步骤 2：Cue 匹配**

系统将提取的 cues 与记忆图中已存储的 cues 进行匹配。匹配可以是精确匹配（相同字符串）或语义匹配（通过嵌入相似度）。匹配结果形成初始激活集合 Z(0)，包含所有匹配的存储 cues 和直接连接的 tags。

**步骤 3：初始状态构建**

初始重建状态 S(0) = (Z(0), H(0))，其中 H(0) 通常为空或仅包含查询文本 q。此时 agent 尚未检索任何内容，但已识别了记忆图的入口点。

### 4.2.2 迭代循环

对于 t = 0, 1, 2, ...，系统执行以下循环直到满足终止条件：

**步骤 1：LLM 推理与动作选择**

给定当前状态 $$\mathcal{S}^{(t)} = (\mathcal{Z}^{(t)}, \mathcal{H}^{(t)})$$，LLM 执行推理以选择下一步动作集合 $$\mathcal{A}^{(t)}$$：

- **输入**：LLM 接收三个输入：(1) 原始查询 q，(2) 已累积的推理上下文 $$\mathcal{H}^{(t)}$$，(3) 当前激活记忆元素 $$\mathcal{Z}^{(t)}$$ 的描述（包括激活的 cues、tags 和已访问内容的摘要）。
  
- **推理目标**：LLM 判断当前 $$\mathcal{H}^{(t)}$$ 是否足以回答 q。如果不足，则决定需要检索哪些新信息。这种决策基于当前的推理缺口：例如，如果 $$\mathcal{H}^{(t)}$$ 包含用户的咖啡消费时间，但缺少频率信息，LLM 会决定检索与 "频率" 相关的内容。
  
- **动作选择**：LLM 从动作集 $$\mathcal{A}$$ 中选择子集 $$\mathcal{A}^{(t)}$$。选择可以是显式的（通过工具调用 API）或隐式的（通过解析 LLM 输出中的动作指令）。例如，LLM 可能决定同时执行 $$\Pi_{c \to g}$$（从 cue "咖啡" 激活新 tags）和 $$\Pi_{(c,g) \to v}$$（从现有 cue-tag 对检索内容）。

这一步骤将 LLM 的推理能力直接整合到记忆访问决策中，使检索策略能够根据推理上下文动态调整。

**步骤 2：受控记忆遍历**

对每个选中的动作 $$\Pi_a \in \mathcal{A}^{(t)}$$，系统执行对应的遍历算子，生成候选节点集合：

- **前向遍历**：如果 $$\mathcal{A}^{(t)}$$ 包含 $$\Pi_{c \to g}$$，系统对 $$\mathcal{Z}^{(t)}$$ 中的每个 cue c 应用 $$\phi_{c \to g}(c)$$，获取所有关联 tags。如果包含 $$\Pi_{(c,g) \to v}$$，系统对每个 (c, g) 对应用 $$\phi_{(c,g) \to v}(c, g)$$，获取候选 contents。
  
- **反向遍历**：如果 $$\mathcal{A}^{(t)}$$ 包含 $$\Pi_{v \to (c,g)}$$，系统对 $$\mathcal{Z}^{(t)}$$ 中的每个 content v 分析其文本，提取其中包含的 cues 和 tags（通过 LLM 或规则提取器），并将这些新索引加入候选集合。
  
- **候选集合生成**：所有执行结果合并为 $$\widetilde{\mathcal{Z}}^{(t+1)} = \bigcup_{a \in \mathcal{A}^{(t)}} \Pi_a(\mathcal{Z}^{(t)})$$。这是中间结果，尚未经过 LLM 路由。

这一步骤是记忆系统的机械操作，不涉及语义理解，确保了遍历的效率和可扩展性。

**步骤 3：LLM 路由与状态更新**

LLM 评估候选集合 $$\widetilde{\mathcal{Z}}^{(t+1)}$$ 中每个新节点的语义相关性，决定保留或丢弃：

- **相关性评估**：LLM 分析每个候选 content 与当前查询 q 和推理上下文 $$\mathcal{H}^{(t)}$$ 的语义相关性。对于 content，LLM 判断其是否包含有助于回答 q 的新信息。对于新 cues 和 tags，LLM 判断它们是否可能导向相关内容。
  
- **路由决策**：基于相关性评估，LLM 生成过滤后的激活集合 $$\mathcal{Z}^{(t+1)} \subseteq \widetilde{\mathcal{Z}}^{(t+1)} \cup \mathcal{Z}^{(t)}$$，保留相关节点并丢弃无关节点。同时，LLM 从保留的 contents 中提取新的证据文本，更新推理上下文为 $$\mathcal{H}^{(t+1)}$$。
  
- **状态更新**：新状态 $$\mathcal{S}^{(t+1)} = (\mathcal{Z}^{(t+1)}, \mathcal{H}^{(t+1)})$$ 成为下一轮迭代的输入。

这一步骤将记忆遍历的 "广度" 转化为推理所需的 "深度"，通过语义路由避免无关信息的干扰。

**步骤 4：终止判定**

系统评估是否满足终止条件：

- **充分性判定**：LLM 判断 $$\mathcal{H}^{(t+1)}$$ 是否包含足够信息回答 q。如果是，则终止重建过程，输出 $$\mathcal{H}^{(t+1)}$$ 作为最终记忆证据。
  
- **预算约束**：如果迭代次数达到预设上限 T_max（例如 8 轮），则强制终止，返回当前 $$\mathcal{H}^{(t)}$$。
  
- **收敛检测**：如果 $$\mathcal{Z}^{(t+1)}$$ 相比 $$\mathcal{Z}^{(t)}$$ 没有实质性变化（没有新的 relevant contents 被检索），则判定检索已收敛，终止过程。

如果未满足终止条件，则回到步骤 1 继续下一轮迭代。

### 4.2.3 过程示例

以查询 "用户为什么认为航班延误政策不公平？" 为例，展示完整的重建过程：

**初始轮 (t=0)**：
- Query cues: {"航班", "延误", "政策", "不公平"}
- $$\mathcal{Z}^{(0)}$$: 匹配的存储 cues + 直接连接的 tags
- $$\mathcal{H}^{(0)}$$: 查询文本

**第一轮 (t=1)**：
- LLM 推理: 需要检索用户关于延误政策的具体陈述和情绪表达
- 动作选择: $$\mathcal{A}^{(1)} = \{\Pi_{c \to g}, \Pi_{(c,g) \to v}\}$$，从 cues 激活 tags 并检索内容
- 遍历结果: $$\widetilde{\mathcal{Z}}^{(1)}$$ 包含 10 个相关对话段落
- LLM 路由: 保留 3 个最相关的段落，提取用户抱怨改签费用高、赔偿不明确
- $$\mathcal{H}^{(1)}$$: "用户在三次对话中提到改签费用过高，赔偿标准不透明"

**第二轮 (t=2)**：
- LLM 推理: 需要了解用户的具体经历（是否有实际延误经历）
- 动作选择: $$\mathcal{A}^{(2)} = \{\Pi_{v \to (c,g)}\}$$，从已检索内容中发现新 cues
- 反向遍历: 从段落中发现新 cues {"改签", "费用", "赔偿", "4月3日航班"}
- LLM 路由: 激活新 cue "4月3日航班"，检索该具体航班事件
- $$\mathcal{H}^{(2)}$$: 补充 "用户在4月3日航班延误后，被要求支付改签费用 500 元，未收到赔偿"

**第三轮 (t=3)**：
- LLM 推理: 需要确认用户的会员等级（是否影响赔偿政策）
- 动作选择: $$\mathcal{A}^{(3)} = \{\Pi_{s \to e}\}$$，从语义层检索用户属性
- 遍历结果: 获取语义信息 "用户是 Gold 会员"
- LLM 路由: 检索 Gold 会员的赔偿政策条款
- $$\mathcal{H}^{(3)}$$: 补充 "Gold 会员应享受免费改签和优先赔偿"

**终止 (t=4)**：
- LLM 判定: $$\mathcal{H}^{(3)}$$ 包含完整信息（用户经历 + 政策标准 + 用户期望），可以回答查询
- 最终输出: $$\mathcal{H}^{(3)}$$ 作为记忆证据传递给回答生成模块

该示例展示了 MRAgent 如何通过多轮迭代，从初始 cues 出发，逐步发现新线索，检索不同记忆层的信息，最终重建出完整的上下文。每一步都由 LLM 推理驱动，确保检索与推理深度耦合。

## 4.3 理论分析：主动 vs 被动检索

MRAgent 的主动记忆重建机制不仅在经验上有效，而且在理论上严格优于被动检索范式。论文通过形式化分析和定理证明，建立了主动检索的表达优势。

### 4.3.1 被动检索的形式化定义

被动检索策略 $$\pi_{\mathrm{p}}: \mathcal{Q} \times \mathcal{M} \to \mathcal{V}^T$$ 将查询 q 和记忆图 $$\mathcal{M}$$ 映射到固定大小 T 的内容集合，且该选择仅依赖于 q，不依赖于任何中间状态。形式上：

$$\pi_{\mathrm{p}}(q, \mathcal{M}) = \{v_1, \ldots, v_T\}$$

其中 $$v_i \in \mathcal{V}$$ 是记忆图中的内容节点，T 是检索预算（允许访问的最大内容数）。被动检索的关键特性是 **one-shot 决策**：所有 T 个内容在观察到任何内容之前就已经被选定。

被动检索的典型实例包括：
- **Top-K 向量检索**：基于查询与内容的嵌入相似度，选择 top-K 个内容
- **预定义图遍历**：按照固定的路径规则（如 "从 cue 激活所有 tags，再检索所有连接的内容"）访问内容

这些方法的共同点是检索策略与检索结果解耦：策略不根据已访问内容调整。

### 4.3.2 主动检索的形式化定义

主动检索策略 $$\pi_{\mathrm{a}}: \mathcal{Q} \times \mathcal{M} \times \mathcal{S}(t) \to \mathcal{A}(t)$$ 在每一步 t 基于当前状态 $$\mathcal{S}(t)$$ 选择动作 $$\mathcal{A}(t)$$。形式上，主动检索生成一个访问序列：

$$v^{(1)}, v^{(2)}, \ldots, v^{(T)}$$

其中 $$v^{(t)}$$ 的选择依赖于已访问内容 $$\{v^{(1)}, \ldots, v^{(t-1)}\}$$ 和累积的推理上下文 $$\mathcal{H}^{(t-1)}$$。主动检索的关键特性是 **sequential decision**：每一步的选择都可以利用之前步骤获得的信息。

主动检索策略类定义为：

$$\mathcal{H}^{\mathrm{LM}}_{\mathrm{active}}(T) = \{f: \mathcal{Q} \times \mathcal{M} \to \mathrm{Answer} \mid \exists \pi_a \text{ s.t. 在 } T \text{ 步内以 } f \text{ 方式访问 } \mathcal{M}\}$$

即所有可以通过 T 步主动检索策略计算的函数集合。

### 4.3.3 定理 4.1：主动检索严格强于被动检索

**定理陈述**：

对任意检索预算 T ≥ 2，被动检索假设类被严格包含于主动检索假设类：

$$\mathcal{H}^{\mathrm{LM}}_{\mathrm{passive}}(T) \subsetneq \mathcal{H}^{\mathrm{LM}}_{\mathrm{active}}(T)$$

**证明思路**：

论文通过构造一个二元树搜索问题 D_{n,d} 来证明该定理。问题定义如下：
- 记忆结构为深度 d 的完全二元树，每个叶节点对应一个内容
- 目标叶节点通过 d 个二元决策确定（每个内部节点代表一个 yes/no 问题）
- 查询 q 要求找到目标叶节点

对于该问题：
- **主动策略**可以在 d+1 步内精确找到目标叶节点：从根节点开始，每步根据当前节点的标签选择下一个子节点，经过 d 步到达叶节点，最后 1 步访问叶节点内容。总步骤 = d+1。
- **被动策略**必须在第一步就选择所有要访问的节点。由于被动策略无法根据中间结果调整，它必须预先选择足够的叶节点以确保覆盖目标。对于深度 d 的二元树，需要选择至少 2^{d-1} 个叶节点才能保证 50% 的命中率（随机猜测）。当 T < 2^{d-1} 时，被动策略的错误率呈指数级增长。

当 T ≥ 2 时，存在 d 使得 d+1 ≤ T < 2^{d-1}（例如 d = 5, T = 6）。此时主动策略可以在预算内精确求解，而被动策略必然失败。因此存在函数 f ∈ H^{active}_{LM}(T) 但 f ∉ H^{passive}_{LM}(T)，证明严格包含关系。

**直觉解释**：

主动检索的优势在于 **信息利用效率**。被动检索必须在看到任何内容之前就做出所有决策，相当于在信息匮乏的状态下就锁定了策略。主动检索可以在每一步利用已获得的信息来调整下一步，这种反馈机制使其能够用更少的访问次数达到更高的精度。

类比于人类问题解决：被动检索像是一次性列出所有可能查阅的书籍（不管书里写什么），主动检索是根据已读章节的内容决定下一本读什么。后者显然更高效。

### 4.3.4 理论结果的实践意义

定理 4.1 不仅具有理论意义，也解释了 MRAgent 在实验中的性能优势：

- **检索效率**：对于相同的检索预算 T（例如允许访问的内容数），主动策略可以表达更复杂的检索函数。这意味着在有限资源下，MRAgent 可以回答需要多跳推理的问题，而被动系统会因预算耗尽而失败。
  
- **错误率降低**：被动策略在早期错误选择后无法纠正，错误会累积。主动策略通过反馈机制可以在后续步骤中修正方向，降低整体错误率。这解释了 MRAgent 在多跳推理任务上的显著提升（LOCOMO 上相对提升 23.3%）。
  
- **成本节约**：由于主动策略可以用更少的访问达到相同精度，MRAgent 在实践中显著降低了 token 消耗（实验中 118k vs A-Mem 的 632k）。这与理论预测一致：主动检索通过 "按需访问" 避免了被动检索的 "过度检索"。

该定理为 MRAgent 的设计提供了坚实的理论基础，说明 "memory as reconstruction" 不仅是认知启发的类比，而是在计算能力上严格优于传统范式的架构选择。


---


## 第5章：实验结果与分析

## 5.1 实验设置

### 5.1.1 数据集

实验在两个长程记忆benchmark上评估：

**LOCOMO**：包含50个对话，平均每个对话约35个session，约300个turns，每个对话约200个QA pairs。该benchmark专注于多轮对话中的长程依赖问题。

**LongMemEval-S**：包含约500个问题，历史记录约115K tokens。该benchmark评估超长上下文中的记忆检索能力。

### 5.1.2 Baseline方法

实验对比了五种强baseline：
- **RAG**：基础检索增强生成，使用vector store进行one-shot top-k检索
- **A-Mem**：结构化笔记 + 图记忆系统
- **MemoryOS**：三层记忆架构（episodic, semantic, abstraction）
- **LangMem**：向量数据库 + 动态记忆更新
- **Mem0**：增量事实提取系统

### 5.1.3 实现细节

- **Backbone模型**：Gemini-2.5-Flash 和 Claude-Sonnet-4.5
- **评估指标**：F1分数，LLM-Judge（使用GPT-4o-mini，temperature=0.0），Evidence Recall
- **实验配置**：3次独立运行，报告均值±标准差
- **资源约束**：每个查询最多8轮推理，每轮最多10次工具调用


---


## 5.2 主实验结果

### 5.2.1 LOCOMO benchmark

**Table 1: LOCOMO benchmark结果（J score）**

Gemini backbone：
- RAG: 61.30
- A-Mem: 55.97
- MemoryOS: 63.35
- LangMem: 62.86
- Mem0: 68.31
- **MRAgent: 84.21** （相对Mem0提升23.3%）

Claude backbone：
- RAG: 61.10
- A-Mem: 68.45
- MemoryOS: 61.18
- LangMem: 78.61
- Mem0: 69.02
- **MRAgent: 88.32** （相对LangMem提升12.4%）

MRAgent在两个backbone上均取得最优性能，证明该方法的有效性不依赖于特定LLM。

### 5.2.2 LongMemEval benchmark

**Table 2: LongMemEval结果（J score）**

- MRAgent (Gemini): 72.95
- MRAgent* (Claude检索): 86.76

MRAgent相对最优baseline提升32%，证明在超长上下文场景中的有效性。

### 5.2.3 按问题类型分析

在四种问题类型上均取得提升：
- **Multi-hop**：需要跨多个事件推理
- **Temporal**：需要理解时间顺序
- **Open-domain**：需要利用外部知识
- **Single-hop**：简单事实检索

主动重建机制在复杂推理任务（multi-hop）上优势最明显，因为多次迭代能够逐步积累证据。


---


## 5.3 成本分析

**Table 3: Token消耗和运行时间**

| Method | Token Consumption | Runtime (s) |
|--------|:-:|:-:|
| A-Mem | 632k | 1,122.23 |
| MemoryOS | 273k | 3,135.54 |
| LangMem | 3,268k | 1,209.57 |
| Mem0 | 245k | 533.29 |
| MRAgent | **118k** | 586.11 |

MRAgent的token消耗最低（118k），远低于A-Mem（632k）和LangMem（3,268k）。这种效率来自于：

1. **选择性on-demand访问**：只在需要时访问记忆，避免批量检索
2. **轻量构建阶段**：复杂关系推理延迟到检索阶段，以query-specific方式执行
3. **语义引导剪枝**：通过associative tag引导检索方向，在访问昂贵episodic内容前过滤无关路径

运行时间方面，MRAgent（586.11s）与最快baseline（Mem0: 533.29s）相当，但准确率显著更高（+23.3%）。


---


## 5.4 消融研究

**Figure 5: 结构变体消融实验**

实验对比了三种记忆结构：
- **CE** (Cue→Episode)：直接关联，无中间语义层
- **CTE** (Cue–Tag–Episode)：引入tag作为语义桥梁
- **CTC** (Cue–Tag–Content)：完整三层结构

每种变体在两种设定下评估：
- 无推理（绿色）：被动one-shot检索
- 有推理（蓝色）：主动多步重建

**关键发现**：

1. **主动多步推理是性能提升的主要因素**：有推理的变体始终优于对应的无推理变体，证明迭代式证据积累的有效性。

2. **关联tag提供有效语义引导**：在无推理设定下，性能从CE→CTE→CTC单调提升，证明tag作为语义桥梁的价值。

3. **事件层和语义层互补**：移除语义层（CE vs CTE）导致清晰性能下降，证明两层结构的必要性。


---


## 5.5 多轮推理分析

**Figure 6: 累计证据召回随推理轮次变化**

- **Single-hop查询**：约3轮达到近乎完美召回
- **Temporal查询**：约3轮达到稳定召回
- **Multi-hop查询**：持续受益，逐步召回提升>30%

**终止效率**：MaxValidTurns ≈ AverageTurns，说明LLM能够有效判断何时继续搜索、何时停止，避免不必要的计算。

**深度 vs 广度**（Figure 9, Appendix D.6）：增加并行检索预算K不能替代重建深度T。证明主动推理的价值在于探索深度，而非并行度。


---


## 第6章：代码实现详解

基于官方GitHub仓库 https://github.com/Ji-shuo/MRAgent

## 6.1 总体流水线

MRAgent实现为两阶段流水线：

**Phase 1: 图记忆构建**
- 从对话历史中提取episodes
- LLM蒸馏提取tags和cues
- 构建Cue-Tag-Content异构图

**Phase 2: 基于记忆回答查询**
- LLM选择遍历动作
- 图记忆执行结构化检索
- 迭代直到收集足够证据

入口点：`run.py`，支持LoCoMo和LongMemEval两个benchmark。

## 6.2 仓库结构

```
run.py                   # 主入口
agent/
   agent.py              # 流水线编排
   tools.py              # 工具schema + 调度
memory/
   system.py             # 内存图存储
   controller.py         # 图查询工具
llm/
   controller.py         # LLM tool-calling封装
   embeddings.py         # text-embedding客户端
prompts/
   prompts.py            # 所有LLM prompt
   schema.py             # JSON输出验证器
```

## 6.3 核心实现细节

### 6.3.1 图记忆存储（memory/system.py）

```python
## ⚠️ 非官方概念实现，未经验证
class CueTagContentGraph:
    def __init__(self):
        self.cues = {}           # cue_id -> Cue
        self.tags = {}           # tag_id -> Tag
        self.contents = {}       # content_id -> Content
        self.cue_to_tags = {}    # φ_{c→g} mapping
        self.tag_to_contents = {}  # φ_{(c,g)→v} mapping

    def activate_tags(self, cue):
        """φ_{c→g}: 从cue激活关联tags"""
        return self.cue_to_tags.get(cue, [])

    def retrieve_contents(self, cue, tag):
        """φ_{(c,g)→v}: 从cue+tag检索contents"""
        return self.tag_to_contents.get((cue, tag), [])
```

该结构实现in-memory异构图，支持论文中的两个核心算子：
- `φ_{c→g}`：从cue激活关联tags
- `φ_{(c,g)→v}`：从cue+tag检索contents

### 6.3.2 主动重建循环（agent/agent.py）

```python
## ⚠️ 非官方概念实现，未经验证
class MRAgent:
    def reconstruct(self, query, max_turns=8):
        state = ReconstructionState(query)
        
        for turn in range(max_turns):
            # LLM选择动作
            action = self.llm_select_action(state)
            
            # 执行图遍历
            observations = self.memory.traverse(action)
            
            # LLM评估并路由
            should_continue = self.llm_route(state, observations)
            
            if not should_continue:
                break
        
        return state.evidence
```

核心循环实现论文Algorithm 1：
1. LLM从当前状态选择遍历动作
2. 图记忆执行结构化检索
3. LLM基于新观测判断是否继续
4. 迭代直到终止条件满足

### 6.3.3 工具定义（agent/tools.py）

定义遍历动作的tool schemas：

```python
## ⚠️ 非官方概念实现，未经验证
TOOLS = [
    {
        "name": "query_tag_events",
        "description": "查询与特定tag关联的所有events",
        "parameters": {
            "tag": "string",
            "time_range": "optional [start, end]"
        }
    },
    {
        "name": "query_conversation_time",
        "description": "查询特定时间段的conversations",
        "parameters": {
            "start_time": "string",
            "end_time": "string"
        }
    },
    {
        "name": "query_event_context",
        "description": "查询event的上下文内容",
        "parameters": {
            "event_id": "string"
        }
    }
]
```

这些工具映射到论文中的三种遍历动作：
- Forward (Cue→Tag, (Cue,Tag)→Content)
- Reverse (Content→(Cue,Tag))
- Temporal (time-range query)

### 6.3.4 LLM封装（llm/controller.py）

实现tool-calling接口，支持多模型backbone：

```python
## ⚠️ 非官方概念实现，未经验证
class LLMController:
    def __init__(self, model_name, api_key):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model_name
    
    def tool_call(self, prompt, tools):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="auto"
        )
        return response.choices[0].message.tool_calls
```

支持OpenAI-compatible API，可接入Gemini、Claude等多种backbone。

### 6.3.5 Prompt设计（prompts/prompts.py）

核心prompt模板：

```python
## ⚠️ 非官方概念实现，未经验证
RECONSTRUCTION_PROMPT = """
You are an agent with access to a graph-structured memory.
Your task is to answer the query by actively exploring the memory.

Query: {query}

Current evidence:
{accumulated_evidence}

Available tools:
{tools}

Think step-by-step:
1. What information do I still need?
2. Which tool can provide that information?
3. Call the tool and observe the results.
4. Decide if I have enough evidence to answer.

If you have enough evidence, output your final answer.
Otherwise, continue exploring.
"""
```

该prompt实现论文中的主动重建循环，引导LLM进行迭代式证据收集。


---


## 第7章：局限性与延伸阅读

## 7.1 局限性

1. **场景局限**：当前仅在对话记忆场景验证，尚未在更广泛的agent任务（网页导航、工具使用、多agent协作）上测试。主动重建机制在其他任务类型中的泛化性需进一步验证。

2. **可扩展性**：记忆构建依赖LLM蒸馏，对于极长历史（>1000 sessions）的计算开销和存储需求未充分评估。tag提取的质量和成本随历史长度增长的影响需更系统分析。

3. **标签噪声**：当前实现假设LLM能够准确提取tags和cues，但标签噪声对下游重建性能的影响尚未量化。错误传播机制（低质量tag→误导检索→错误累积）需更深入研究。

4. **理论假设**：理论分析假设完美终止判断，但实际LLM的停止策略可能存在偏差。过度搜索（浪费计算）和提前停止（证据不足）的权衡需更精细建模。

## 7.2 未来方向

1. **多agent记忆共享**：扩展MRAgent到多agent设置，研究agent间记忆共享、冲突解决、协同推理机制。

2. **动态记忆更新**：当前实现假设离线构建记忆，未来需支持在线增量更新，处理新信息与旧记忆的整合、遗忘、冲突解决。

3. **与其他推理范式融合**：探索主动记忆重建与Chain-of-Thought、Tree-of-Thought、ReAct等推理范式的协同效应。

4. **跨模态记忆**：扩展到多模态场景（图像、音频、视频），研究跨模态关联tag的设计和检索策略。

## 7.3 延伸阅读

**记忆增强生成基础**：
- RAG (Lewis et al., 2020): "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" — 提出检索增强生成范式，使用vector store进行one-shot top-k检索
- GraphRAG (Han et al., 2025): "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" — 使用图结构化检索语料库，通过社区摘要提供多粒度信息

**LLM Agent记忆系统**：
- A-Mem (Xu et al., 2025): "A-Mem: Autonomous Memory for Agents" — 结构化笔记 + 图记忆系统，支持长期记忆管理
- MemoryOS (Kang et al., 2025): "MemoryOS: Optimizing Memory for Language Model Agents" — 三层记忆架构（episodic, semantic, abstraction）
- Mem0 (Chhikara et al., 2025): "Mem0: Building and Managing Memory for AI Agents" — 增量事实提取系统，支持动态记忆更新

**认知科学启发**：
- Tolman (1948): "Cognitive Maps in Rats and Men" — 提出认知地图理论，强调记忆的主动重构性质
- Bartlett (1932): "Remembering: A Study in Experimental and Social Psychology" — 提出图式理论，记忆是重构而非再现

**图神经网络检索**：
- "Graph Neural Networks for Memory Reasoning" (2024): 使用GNN在记忆图上进行推理
- "Hierarchical Graph Memory for LLM Agents" (2025): 层次化图记忆结构


---


**参考文献**：

论文原文：Shuo Ji, Yibo Li, Bryan Hooi. "Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents." ICML 2026. arXiv:2606.06036

代码仓库：https://github.com/Ji-shuo/MRAgent
