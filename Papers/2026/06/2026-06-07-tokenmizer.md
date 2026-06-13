# TokenMizer: 图结构会话记忆 — 为长周期LLM会话管理上下文



---

> **论文**: TokenMizer: Graph-Structured Session Memory for Long-Horizon LLM Context Management  
> **作者**: Shweta Mishra (Independent Researcher)  
> **arXiv**: [2606.06337](https://arxiv.org/abs/2606.06337) · 2026-06-04  
> **代码**: [github.com/Shweta-Mishra-ai/tokenmizer](https://github.com/Shweta-Mishra-ai/tokenmizer) (MIT License)  
> **评分**: Novelty 3 · Impact 3 · Technical 3 · Evidence 3 · Reusability 2 → **14/15 (Deep Read)**

---

## Ch1: 论文概述与核心贡献

### 一句话总结

TokenMizer 将 LLM 会话历史建模为**类型化知识图谱**（14 种节点 × 7 种边），通过混合提取管线增量填充图谱，在上下文窗口溢出前将其序列化为紧凑的「恢复块」（平均仅 **78 tokens** —— 比基线小 2×），同时保持**更高的决策回忆率**（+9–17 pp），使长周期 LLM 任务可以在不修改应用代码的情况下无限延续。

### 问题背景

大语言模型部署长周期任务时面临一个根本性约束：**上下文窗口有限，但生产性工作会话并非有限**。

- **MECW（Maximum Effective Context Window）**：Paulsen (2025) 定义的任务精度低于可接受阈值时的上下文长度。对于复杂任务，MECW 约为 **~16k tokens** —— 远低于广告中的 128k。
- 以 **~950 tokens/turn** 的典型消耗速率，一次会话在仅 **16 轮**后就溢出上下文窗口。
- 溢出的瞬间，关键结构化信息 —— **架构决策、任务状态转换、文件修改历史、错误/解决方案配对** —— 被静默丢弃。

> **类比理解**：传统的上下文管理就像一本没有目录、没有章节标题、没有索引的小说。当你读到第 20 章时，第 3 章做的人物关系判断已经被撕掉了。TokenMizer 相当于在阅读过程中自动构建一个结构化的「人物关系图 + 情节时间线」，在第 16 章时把它塞进一张便利贴 —— 78 个 token 足以告诉你「谁和谁是什么关系，现在剧情到哪了，哪些伏笔还没回收」。

### 7 项核心贡献

| # | 贡献 | 说明 |
|---|------|------|
| 1 | **类型化图谱 Schema** | 14 种节点类型 + 7 种边类型 + 状态生命周期，正式定义了会话状态的语义结构 |
| 2 | **混合提取管线** | V1（34 条正则，0.5ms，$0）+ V2 改进（扩展触发词/模糊匹配/复合拆分），可选 LLM 升级路径 |
| 3 | **三级检查点系统** | Critical(≤100 tok) / Standard(≤300 tok) / Full(≤600 tok)，带图谱 diff |
| 4 | **8 层压缩管线** | 47.3% token 缩减，零外部依赖（仅填充移除即贡献 31.2%） |
| 5 | **语义嵌入缓存** | 受控 10 查询工作负载上 70% 命中率（初步结果） |
| 6 | **21 会话开源 Benchmark** | 5 领域含 ground-truth 标注，定义 token 效率指标 η |
| 7 | **会话特征相关性分析** | 更长会话从结构化检查点获益比例更大 |

### 方法速览

```
常规流程（无 TokenMizer）：
  Turn 1 → Turn 2 → ... → Turn 16 → 上下文溢出 → 会话状态丢失 ✗

TokenMizer 流程：
  Turn 1-14：透明代理，增量填充知识图谱
  Turn 14：累计 tokens ≥ 85% MECW → 触发检查点
           → 图谱序列化为 78-token 恢复块
           → 恢复块 + 最近 2 条消息 → 替换旧上下文
  Turn 15+：在恢复的图结构基础上继续会话 ✓
```

### 关键结果

| 指标 | TokenMizer V2 | 最佳基线 | 优势 |
|------|--------------|---------|------|
| 恢复块大小 (avg) | **78 tokens** | 159 tokens (Sliding Window) | **2.1× 更小** |
| 决策回忆率 | **46.6%** | 38% (Naive Summary) | **+8.6 pp** |
| 任务回忆率 | 51.0% | 50% (Sliding Window) | +1.0 pp |
| 文件回忆率 | 58.7% | 60% (Sliding Window) | -1.3 pp |

**核心结论**：没有一个基线保留了决策的「为什么」。滑动窗口可能保留技术的名称，但丢失了选择它而不是替代方案的**理由**。TokenMizer 用一半的 token 成本做到了这一点。

---

## Ch2: 研究背景与问题定义

### 2.1 上下文窗口失效的三种模式

LLM 上下文窗口的失效不是瞬间发生的，而是呈现三种退化模式：

**(1) 截断（Truncation）**：最简单也最常见 —— 保留最近的 N 个 token，丢弃早期的所有内容。
- **问题**：早期内容往往包含最重要的信息 —— 目标定义、架构决策、技术选型。
- 在第 16 轮被截断的可能是第 3 轮做出的「用 Redis 而非 PostgreSQL」的决策及其理由。

**(2) 摘要（Summarization）**：通过 LLM 自身压缩历史。ACC (Anthropic) 达到 22.7% 的压缩率。
- **问题**：LLM 摘要无法可靠区分已完成和进行中的任务，也无法保留决策的「为什么」。
- 摘要的输出是平面自然语言文本，丢失了类型化关系和状态生命周期。

**(3) 检索增强（Retrieval Augmentation）**：将历史向量化，按相关性检索。
- **问题**：「Lost in the Middle」—— U 形注意力模式。语义上相似但上下文无关的内容可能被错误检索；语义上远但结构上关键的信息可能被遗漏。

### 2.2 TokenMizer vs MemGPT

这是理解本文定位最关键的区别：

| 维度 | MemGPT | TokenMizer |
|------|--------|------------|
| **目标** | 跨会话的长期事实回忆 | **单会话内的结构连续性** |
| **架构** | OS 启发的内存分页系统，LLM 自主管理 | **透明 HTTP 反向代理**，零应用代码修改 |
| **触发** | LLM 判断何时换页 | 确定性规则：累计 tokens ≥ 85% MECW |
| **记忆形式** | 自然语言记忆块 | **类型化知识图谱**（14 节点 × 7 边） |
| **状态跟踪** | 无 | 有（状态生命周期：pending→in_progress→completed/failed） |
| **检索** | LLM 调用函数检索 | 确定性图遍历 + 语义嵌入缓存 |

MemGPT 解决的问题是「我三个月前和你说过什么？」；TokenMizer 解决的问题是「在当前这个持续进行的数据科学项目中，哪些特征已工程化、哪些待处理、为什么选 XGBoost 而不是 LightGBM？」

### 2.3 形式化定义

论文为问题提供了精确的形式化：

**会话溢出（Session Overflow）**：第一个满足 `cumulative_tokens > MECW` 的轮次。

**结构化恢复块（Structured Resume Block）**：
`C' = R ∪ {最后两条消息}`，其中 `|C'| ≤ MECW`

`R` 是将当前会话图谱序列化为紧凑文本的产物，包含：
- 已完成/进行中/待处理的任务（带状态）
- 决策及理由
- 文件修改历史
- 环境和项目事实

**信息损失（Information Loss）**：
$$L = 1 - \frac{1}{3}(R_{\text{task}} + R_{\text{dec}} + R_{\text{file}})$$

其中回忆率使用模糊匹配计算（Definition 5/6，详见 Ch4）。

**Token 效率（Token Efficiency）**：
$$\eta = \frac{\text{mean recall}}{|R|/100}$$

这个指标归一化了回忆率与恢复块 token 成本，使不同会话和领域的方法可以公平比较。**调试会话**被发现是最高效率的领域 —— 其结构化状态（错误→修复→验证）天然适合图表示。

---

## Ch3: 知识图谱 Schema 设计

这是 TokenMizer 最核心的设计决策：**用什么结构来编码会话状态？**

### 3.1 14 种节点类型 × 3 大类别

```python
class NodeType(str, Enum):
    # Action nodes — 带状态生命周期
    TASK        = "task"        # 任务（可完成/失败）
    FILE        = "file"        # 文件（可修改）
    ERROR       = "error"       # 错误（可修复）
    TEST        = "test"        # 测试（可运行）
    SCHEMA      = "schema"      # 数据模式（可演化）
    METRIC      = "metric"      # 指标（可追踪）
    # Decision nodes — 记录选择
    DECISION    = "decision"    # 技术决策（如 "Redis over PostgreSQL"）
    DEPENDENCY  = "dependency"  # 依赖关系
    API         = "api"         # API 端点
    # Context nodes — 无状态背景
    GOAL        = "goal"        # 会话目标
    ENVIRONMENT = "environment" # 运行环境
    PROJECT     = "project"     # 项目信息
    CONCEPT     = "concept"     # 领域概念
    AGENT       = "agent"       # Agent 身份
```

### 3.2 状态生命周期（仅 Action nodes）

```
         ┌─────────┐
         │ PENDING │ ← 初次提取
         └────┬────┘
              │
         ┌────▼────────┐
         │ IN_PROGRESS │ ← 检测到进行中信号
         └────┬────────┘
              │
     ┌────────┼────────┐
     ▼        ▼        ▼
┌─────────┐ ┌──────┐ ┌──────────┐
│COMPLETED│ │FAILED│ │ MODIFIED  │
└─────────┘ └──────┘ └──────────┘
```

状态转换是单调的 —— 降级被拒绝（`STATUS_ORDER` 字典确保 `COMPLETED` 永远不会回退到 `PENDING`）。

> **类比理解**：状态生命周期相当于给每个会话元素装了一个「红绿灯」。绿灯=COMPLETED（可以通过了），黄灯=IN_PROGRESS（正在处理），红灯=FAILED（需要关注）。传统平面文本方法无法区分这三种状态 —— 它们只看到文字，看不到信号灯。

### 3.3 7 种边类型

```python
class EdgeType(str, Enum):
    DEPENDS_ON  = "depends_on"   # 任务 A 依赖任务 B
    RELATED_TO  = "related_to"   # 概念关联
    IMPLEMENTS  = "implements"   # 文件实现某功能
    FIXES       = "fixes"        # 提交/修复解决某错误
    BLOCKS      = "blocks"       # 阻塞关系
    PART_OF     = "part_of"      # 部分-整体
    SUPERSEDES  = "supersedes"   # 替代关系
```

边连接节点形成结构化关系网络。例如：
- `TASK:"setup Redis"` —[`DEPENDS_ON`]→ `DECISION:"Redis over PostgreSQL"`
- `FILE:"src/cache.py"` —[`IMPLEMENTS`]→ `TASK:"setup Redis"`
- `ERROR:"connection timeout"` —[`BLOCKS`]→ `TASK:"setup Redis"`
- `COMMIT:"a3f2b1"` —[`FIXES`]→ `ERROR:"connection timeout"`

### 3.4 去重与修剪策略

**去重**：每个节点 ID 由 `sha1(type:normalized_label)[:12]` 生成，相同语义的节点即使措辞不同也会获得相同 ID。

**修剪**：当节点数超过 500 时，驱逐 `importance < 0.1` 的节点。但 `PINNED_TYPES = {DECISION, ENVIRONMENT, GOAL, SCHEMA}` 始终保留 —— 这些是高价值会话结构信息，丢失它们意味着会话无法恢复。

### 3.5 GraphMemory 核心实现

```python
# Source: github.com/Shweta-Mishra-ai/tokenmizer/blob/main/tokenmizer/graph_memory/graph.py

class GraphMemory:
    """持久化会话知识图谱，SQLite 后端存储"""
    
    def __init__(self, session_id: str, storage_dir: str = ".tokenmizer"):
        self.session_id = session_id
        self._db_path = Path(storage_dir) / f"{session_id}.db"
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._init_db()
        self._load()

    def extract_from_messages(self, messages: list[dict], 
                              incremental: bool = True) -> None:
        """从对话消息中提取图节点"""
        from .extractor_v2 import heuristic_extract_v2
        from .validator import GraphValidator
        validator = GraphValidator()
        extracted = heuristic_extract_v2(messages)
        for task in extracted.get("tasks", []):
            node = GraphNode(
                id=GraphNode.make_id(NodeType.TASK, task["label"]),
                type=NodeType.TASK,
                label=task["label"][:120],
                status=NodeStatus(task.get("status", "pending")),
            )
            self._upsert_node(node, validator)
        # ... 类似处理 decisions, files, errors 等

    def _upsert_node(self, node: GraphNode, validator) -> None:
        """插入或更新节点，带置信度过滤"""
        node.confidence = validator.score(node)
        if node.confidence < validator.min_confidence:
            return  # 低置信度候选被丢弃
        if node.id in self._nodes:
            existing = self._nodes[node.id]
            existing.importance = min(1.0, existing.importance + 0.1)
            # 重复出现提升重要性（最多到 1.0）
            if node.status and existing.status:
                if STATUS_ORDER[node.status] > STATUS_ORDER[existing.status]:
                    existing.status = node.status  # 单调状态升级
        else:
            self._nodes[node.id] = node
```

**设计亮点**：
- `importance` 随重复出现递增（上限 1.0）—— 频繁出现的节点天然获得更高保留优先级
- `STATUS_ORDER` 确保状态单调升级 —— COMPLETED 不会回退到 PENDING
- SQLite 持久化使会话可以跨进程重启恢复

---

## Ch4: 混合提取管线

### 4.1 设计哲学：从文本中提取结构化事实

提取管线的输入是 LLM 响应的纯文本，输出是类型化的图节点。这本质上是**从非结构化文本中提取结构化事实**的问题。

TokenMizer 采用**双轨策略**：
- **V1（启发式）**：34 条正则表达式，延迟 0.5ms，成本 $0
- **V2（改进启发式）**：四项优化，延迟不变，召回率显著提升
- **未来路径（LLM）**：将最后 3 条消息发送给小型模型进行提取（未在当前版本评估）

### 4.2 V1 基线：34 条正则

```python
_COMPLETED = re.compile(
    r'(?:Completed|Implemented|Fixed|Added)[:\s]+(.+?)(?:\.|$)', re.I)
_IN_PROGRESS = re.compile(
    r'(?:Working on|Adding|Building)[:\s]+(.+?)(?:\.|$)', re.I)
_PENDING = re.compile(
    r'(?:Need to|TODO|Will|Next)[:\s]+(.+?)(?:\.|$)', re.I)
_DECISION = re.compile(
    r'(?:Decided|Using|Running|Chose|Selected)[:\s]+(.+?)(?:\.|$)', re.I)
_FILE = re.compile(
    r'([\w./][\w./\-]*\.(?:py|js|ts|jsx|tsx|yaml|yml|json|md|go|rs|tf|toml|sh))', re.I)
```

V1 的 34 条正则覆盖了最常见的文本模式。它是**零成本**的基线，但受限于两个固有问题：
1. **触发词覆盖不足**：只能识别精确匹配的动词（如 "Completed"），无法识别变体
2. **无模糊匹配**：提取出的标签（如 "set up Redis cache for session management"）无法与 ground-truth 标注中的简短表述（如 "redis setup"）匹配

### 4.3 V2 四项改进

**改进 1：扩展触发词表（+9 动词，0pp task recall）**

新增：`Deployed, Resolved, Migrated, Launched, Shipped, Published, Done, Finished, Updated`

这一改进本身**未产生直接的任务回忆率提升**，但它**使模糊匹配成为可能** —— 更多提取出的候选标签为模糊匹配提供了原料。

**改进 2：模糊标签匹配（+33pp task recall）—— 主导改进**

这是论文最关键的技术贡献。将提取的冗长标签与 ground-truth 中的简洁标注匹配。

```python
# Source: github.com/Shweta-Mishra-ai/tokenmizer/blob/main/tokenmizer/graph_memory/extractor_v2.py

def fuzzy_match(a: str, b: str) -> bool:
    """
    匹配冗长的提取标签与简洁的 ground-truth 标注。
    论文 Definition 5: 子串包含 或 ≥50% token 重叠。
    ≥3 字符过滤器防止常见短词（"a", "in", "the"）膨胀重叠分数。
    """
    a, b = a.lower(), b.lower()
    if a in b or b in a:          # 情况 1: 子串包含
        return True
    wa = set(re.findall(r'\w{3,}', a))  # 仅 ≥3 字符 token
    wb = set(re.findall(r'\w{3,}', b))
    if not wa or not wb:
        return False
    shorter = wa if len(wa) <= len(wb) else wb
    return len(wa & wb) / len(shorter) >= 0.5  # 情况 2: 50% 重叠
```

**消融结果**显示了模糊匹配的关键性：

| 配置 | Task Recall |
|------|-----------|
| V2 完整 | 51.0% |
| V2 无模糊匹配 | 18.0% |
| **Δ（模糊匹配贡献）** | **+33.0 pp** |

> **类比理解**：模糊匹配相当于给手工标注和自动提取之间架了一座桥。手工标注者写下 "redis setup"，LLM 助手说 "I've completed setting up a Redis cache for session management" —— 传统精确匹配判定为不匹配（0%），模糊匹配看到 2/2 tokens 重叠 → 判定匹配（100%）。没有这座桥，大部分提取的标签都是「精确标记，但被判定为未命中」。

**改进 3：复合句拆分（0pp task recall）**

在句号后拆分复合标签（`re.split(r'.\s+(?=[A-Z])', label)`），使 "Setup Redis. Configure Nginx" 被识别为两个独立任务。

**改进 4：CSV 决策拆分（轻微回归）**

对逗号和 `and` 分隔的决策进行拆分。作者指出这导致决策回忆率的**轻微回归** —— 拆分可能将 "Python and FastAPI" （一个完整的技术栈决策）错误地拆分为两个独立决策。

### 4.4 GraphValidator：置信度评分

```python
# Source: github.com/Shweta-Mishra-ai/tokenmizer/blob/main/tokenmizer/graph_memory/validator.py

class GraphValidator:
    """
    对提取的候选节点进行置信度评分。
    基础分 0.50 + 奖金 - 惩罚：
      + 文件路径模式在 label 中
      + label 长度适中（不是极短/极长）
      + 包含版本号
      - 短 label（<10 字符）
      - 单一通用动词
    拒绝分数 < 0.50。
    """
    def score(self, node: GraphNode) -> float:
        score = 0.50
        # +0.10: 包含文件路径模式
        # +0.05: label 长度 15-80 字符
        # +0.05: 包含版本号 (v\d+)
        # -0.10: label < 10 字符
        # -0.15: 单一通用动词 (如 "Fixed", "Added" 后无内容)
        return max(0.0, min(1.0, score))
```

GraphValidator 是**成本控制机制** —— 每条正则匹配都产生候选，但只有通过置信度阈值的候选才会被插入图谱。这防止了噪声积累。

### 4.5 提取管线延迟分析

- V1/V2 启发式提取：**0.5ms** 延迟
- GraphValidator 评分：**内存计算，可忽略**
- LLM 提取器（未来）：预计 **100–500ms**，需额外推理成本

---

## Ch5: 三级检查点与压缩系统

### 5.1 检查点触发机制

```
触发条件：cumulative_tokens ≥ 85% × MECW
```

此时图谱被序列化为恢复块 `R`，新上下文变为：
```
C' = R + {最后 2 条消息}
```

`|C'| ≤ MECW` 确保新上下文在窗口内。

### 5.2 Algorithm 1：预算约束下的序列化

```python
# Source: github.com/Shweta-Mishra-ai/tokenmizer/blob/main/tokenmizer/graph_memory/graph.py

def to_resume_block(self, budget: int = 300) -> str:
    """在 token 预算内将图谱序列化为紧凑恢复块"""
    from ..core.tokenizer import count_tokens
    lines = [f"[CHECKPOINT: {self.session_id}]"]
    token_budget = budget - count_tokens(lines[0])
    
    def add(line: str) -> bool:
        nonlocal token_budget
        cost = count_tokens(line)
        if cost > token_budget:
            return False
        lines.append(line)
        token_budget -= cost
        return True
    
    # 按 importance 降序排列
    sorted_nodes = sorted(self._nodes.values(), 
                         key=lambda n: n.importance, reverse=True)
    
    # 优先级: completed → pending → decisions → files → env
    completed = [n for n in sorted_nodes if n.status == COMPLETED]
    pending   = [n for n in sorted_nodes if n.status in (PENDING, IN_PROGRESS)]
    decisions = [n for n in sorted_nodes if n.type == DECISION]
    files_    = [n for n in sorted_nodes if n.type == FILE]
    
    if completed:   add("DONE: " + ", ".join(n.label for n in completed[:8]))
    if pending:     add("PENDING: " + ", ".join(n.label for n in pending[:5]))
    if decisions:   add("DECIDED: " + ", ".join(n.label for n in decisions[:8]))
    if files_:      add("FILES: " + ", ".join(n.label for n in files_[:6]))
    
    return "\n".join(lines)
```

**序列化示例输出（78 tokens）**：
```
[CHECKPOINT: data-science-project]
DONE: explore dataset shape, handle null values
PENDING: feature engineering for text columns, train baseline XGBoost
DECIDED: XGBoost over LightGBM (better handling of categorical features)
FILES: src/eda.py, src/preprocess.py, data/raw/transactions.csv
```

### 5.3 三级检查点预算

| 级别 | Token 预算 | 内容 | 适用场景 |
|------|-----------|------|---------|
| **Critical** | ≤100 | GOAL + 进行中任务 + 最高 importance 决策 | 极长会话，上下文极度紧张 |
| **Standard** | ≤300 | 所有任务 + 决策 + 文件 + 环境 | 常规会话（默认） |
| **Full** | ≤600 | 完整图谱 | 高价值会话归档 |

```python
# Source: github.com/Shweta-Mishra-ai/tokenmizer/blob/main/tokenmizer/checkpoints/manager.py

class CheckpointManager:
    TIER_BUDGETS = {"critical": 100, "standard": 300, "full": 600}
    
    def create(self, session_id: str, graph, tier: str = "standard"):
        budget = TIER_BUDGETS.get(tier, 300)
        block = graph.to_resume_block(budget=budget)
        ckpt = Checkpoint(
            id=f"ckpt_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            tier=tier,
            resume_block=block,
            resume_tokens=count_tokens(block),
            graph_diff={},
            created_at=time.time(),
        )
        self._persist(ckpt)
        return ckpt
```

### 5.4 8 层压缩管线

TokenMizer 的压缩管线在序列化**之后**运行，进一步缩减恢复块的 token 数。

| 层 | 类型 | 描述 | 贡献 |
|----|------|------|------|
| 1 | 启发式 | 填充词移除 (filler removal) | **31.2%** |
| 2 | 启发式 | 冗余空格压缩 | 3.1% |
| 3 | 启发式 | 短标签合并 | 2.8% |
| 4 | 启发式 | 已完成任务批量折叠 | 4.5% |
| 5 | 启发式 | 环境事实去重 | 3.0% |
| 6 | 启发式 | 文件路径前缀压缩 | 2.7% |
| 7 | 可选神经 | LLMLingua-2 token 重要性评分 | 未独立测量 |
| 8 | 可选神经 | LongLLMLingua 选择性保留 | 未独立测量 |

**启发式总计**：47.3% token 缩减（零外部依赖，零推理成本）

---

## Ch6: 实验评估与消融分析

### 6.1 实验设置

**Benchmark**: 21 个会话，横跨 5 个领域：

| 领域 | 会话数 | 特点 |
|------|--------|------|
| Software Engineering | 6 | 显式命令式措辞，清晰的文件路径 |
| Data Science | 5 | 探索性分析，迭代实验 |
| DevOps | 4 | 基础设施配置，依赖关系密集 |
| Research | 3 | 隐式推理，长段落，少显式动作标记 |
| Debugging | 3 | 错误-修复-验证循环，高度结构化 |

每个会话有 ground-truth 标注（任务、决策、文件），由单一标注者（作者本人）提供 —— 这是论文承认的限制之一。

### 6.2 对比基线

| 方法 | Task Recall | Decision Recall | File Recall | Avg Tokens |
|------|------------|-----------------|-------------|------------|
| Naive Truncation | 45% | 35% | 55% | 165 |
| Sliding Window | 50% | 30% | **60%** | 159 |
| Naive Summary | 42% | 38% | 48% | 170 |
| **TokenMizer V2** | **51%** | **47%** | 59% | **78** |

**关键观察**：

1. **TokenMizer 用一半的 token 成本达到最高决策回忆率**。这是最重要的结果 —— 决策回忆率是「结构化记忆质量」的核心指标，因为它衡量的是「是否保留了**为什么**」而非「**什么**」。

2. **文件回忆率的比较持平**（59% vs 60%）。文件路径天然可以通过滑动窗口保留（文件名出现在最近的对话中），这不意外。

3. **所有基线的决策回忆率都远低于任务回忆率** —— 因为决策信号往往出现在会话早期，先被截断或摘要化丢失。

### 6.3 Token 效率指标 η

$$\eta = \frac{\text{mean recall}}{|R|/100}$$

| 领域 | η (TokenMizer) | 说明 |
|------|---------------|------|
| Debugging | 最高 | 错误-修复循环天然适合图表示 |
| Software Engineering | 高 | 显式命令式措辞 → 高提取质量 |
| DevOps | 中 | 依赖关系密集，图结构获益 |
| Data Science | 中低 | 探索性分析，迭代概念 |
| Research | 最低 | 隐式推理，无显式动作标记 |

### 6.4 消融实验

| 消融配置 | Task Recall | Δ |
|----------|------------|-----|
| V2 完整 | 51.0% | — |
| V2 - 模糊匹配 | 18.0% | **-33.0 pp** |
| V2 - 扩展触发词 | 51.0% | 0.0 pp |
| V2 - 化合物拆分 | 51.0% | 0.0 pp |
| V2 - CSV 拆分 | ~50% | -1.0 pp |

**核心发现**：模糊标签匹配是**唯一显著改进**（+33 pp task recall）。其他三项改进为模糊匹配提供了更多「弹药」（候选标签），但匹配算法本身才是决定性因素。

### 6.5 领域变化分析

**高表现领域（Software Engineering）**：
- 显式的命令动词（"Implement the user authentication module"）
- 清晰的文件路径模式
- 决策表述明确（"Let's use PostgreSQL for this"）

**低表现领域（Research）**：
- 隐式推理（"The attention pattern distribution suggests..."）
- 无命令式标记
- 概念随时间演化，不是完成/未完成的二元状态

### 6.6 语义缓存结果（初步）

在受控 10 查询工作负载上：
- **Hit rate**: 70%
- 实现：sentence-transformer embeddings + cosine similarity threshold
- 评价：初步结果，未在大规模工作负载上验证

---

## Ch7: 局限性、部署指南与延伸阅读

### 7.1 已知局限

1. **Synthetic benchmark**：21 个会话由作者构建和标注，未覆盖真实生产环境的多样性和噪声。
2. **Single annotator**：ground-truth 标注由单一标注者完成，可能存在标注一致性偏差。
3. **LLM 提取器未评估**：已实现但未评估，对隐式推理场景可能显著改善。
4. **Decision CSV 拆分回归**：~1pp 的决策回忆率下降。

### 7.2 部署配置

```yaml
# tokenmizer.yaml
providers:
  default: openai
checkpoint:
  tier: standard
  trigger_ratio: 0.85
  mecw: 16000
extraction:
  engine: v2
  validator_threshold: 0.50
compression:
  heuristic: true
  neural: false
cache:
  enabled: true
  backend: sentence-transformers
```

### 7.3 延伸阅读

- **MECW**: Paulsen, 2025
- **MemGPT**: Packer et al., 2024
- **LLMLingua-2**: Pan et al., 2024
- **GraphRAG**: Edge et al., 2024

---

*报告完成于 2026-06-07 | Deep Read*