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

TokenMizer 将 LLM 会话历史建模为**类型化知识图谱**（14 种节点 × 7 种边），通过混合提取管线增量填充图谱，在上下文窗口溢出前将其序列化为紧凑的「恢复块」（标准层 **201–302 tokens，均值 254.3**），同时在任务回忆率上与基线持平（75.6%），并在**决策回忆率**（85.0% vs 70.0%，+15 pp）与文件回忆率（100% vs 91.7%）上更高，使长周期 LLM 任务可以在不修改应用代码的情况下延续。

### 问题背景

大语言模型部署长周期任务时面临一个根本性约束：**上下文窗口有限，但生产性工作会话并非有限**。

- **MECW（Maximum Effective Context Window）**：Paulsen (2025) 定义的任务精度低于可接受阈值时的上下文长度。对于复杂任务，MECW 约为 **~16k tokens** —— 远低于广告中的 128k。
- 以 **~950 tokens/turn** 的典型消耗速率，一次会话在仅 **16 轮**后就溢出上下文窗口。
- 溢出的瞬间，关键结构化信息 —— **架构决策、任务状态转换、文件修改历史、错误/解决方案配对** —— 被静默丢弃。

> **类比理解**：传统的上下文管理就像一本没有目录、没有章节标题、没有索引的小说。当你读到第 20 章时，第 3 章做的人物关系判断已经被撕掉了。TokenMizer 相当于在阅读过程中自动构建一个结构化的「人物关系图 + 情节时间线」，在第 16 章时把它塞进一张便利贴 —— 约 254 个 token（实测均值）足以告诉你「谁和谁是什么关系，现在剧情到哪了，哪些伏笔还没回收」。

### 核心贡献

| # | 贡献 | 说明 |
|---|------|------|
| 1 | **类型化图谱 Schema** | 14 种节点类型 + 7 种边类型 + 8 状态生命周期，正式定义了会话状态的语义结构 |
| 2 | **混合提取管线** | 确定性正则启发式默认启用（含复合句与 CSV 决策拆分），可选后台 LLM 提取（未评估） |
| 3 | **三级检查点系统** | Critical(≤100 tok) / Standard(≤300 tok) / Full(≤600 tok)，带图谱 diff |
| 4 | **生产形态服务层** | 透明 OpenAI 兼容代理、SSE 流式、安全中间件、9 个 provider 适配器、压缩与语义缓存（压缩/缓存未测量） |
| 5 | **决策转换记录** | trigger / reason / evidence / confidence delta，独立持久化，形成决策审计轨迹 |
| 6 | **3 会话开源 Benchmark** | 3 个合成会话含 ground-truth 标注，全部数值可追溯到单一版本化结果文件，定义 token 效率指标 η |
| 7 | **检查与互操作接口** | 监控面板、D3/HTML/Obsidian 导出、MCP 检查点/恢复工具 |

### 方法速览

```
常规流程（无 TokenMizer）：
  Turn 1 → Turn 2 → ... → Turn 16 → 上下文溢出 → 会话状态丢失 ✗

TokenMizer 流程：
  Turn 1-14：透明代理，增量填充知识图谱
  Turn 14：累计 tokens ≥ 85% MECW → 触发检查点
           → 图谱序列化为标准层恢复块（实测均值约 254 tokens）
           → 恢复块 + 最近 2 条消息 → 替换旧上下文
  Turn 15+：在恢复的图结构基础上继续会话 ✓
```

### 关键结果

| 指标 | TokenMizer v0.3.1 | Plain-summary 基线 | 差异 |
|------|-------------------|--------------------|------|
| 任务回忆率 | 75.6% | 75.6% | 持平 |
| 决策回忆率 | **85.0%** | 70.0% | **+15.0 pp** |
| 文件回忆率 | **100%** | 91.7% | **+8.3 pp** |
| 恢复块大小 (mean) | **254.3 tokens**（201–302） | 未记录 | 未作 token 经济性比较 |

**核心结论**：基线（全转录关键词扫描）在任务上与图提取持平，但在决策上落后 —— 它可能保留技术的名称，却丢失了选择它而不是替代方案的**理由**；文件回忆率因存在天花板效应而难以区分。作者明确说明 n=3，结果仅具方向性。

---

## Ch2: 研究背景与问题定义

### 2.1 上下文窗口失效的三种模式

LLM 上下文窗口的失效不是瞬间发生的，而是呈现三种退化模式：

**(1) 截断（Truncation）**：最简单也最常见 —— 保留最近的 N 个 token，丢弃早期的所有内容。
- **问题**：早期内容往往包含最重要的信息 —— 目标定义、架构决策、技术选型。
- 在第 16 轮被截断的可能是第 3 轮做出的「用 Redis 而非 PostgreSQL」的决策及其理由。

**(2) 摘要（Summarization）**：通过 LLM 自身压缩历史。ACC（Smith & Park, 2026）用 LLM 调用压缩会话内上下文，论文未报告压缩率。
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
$L = 1 - \frac{1}{3}(R_{\text{task}} + R_{\text{dec}} + R_{\text{file}})$

其中回忆率使用模糊匹配计算（Definition 5/6，详见 Ch4）。

**Token 效率（Token Efficiency）**：
$\eta = \frac{\text{mean recall}}{|R|/100}$

这个指标归一化了回忆率与恢复块 token 成本，使不同会话的方法可以公平比较。在 3 个基准会话上，η 分别为 **0.29、0.32、0.46（均值 0.35）**。

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
    # Decision nodes — 记录选择
    DECISION    = "decision"    # 技术决策（如 "Redis over PostgreSQL"）
    DEPENDENCY  = "dependency"  # 依赖关系
    API         = "api"         # API 端点
    ENDPOINT    = "endpoint"    # 服务端点
    # Context nodes — 无状态背景
    GOAL        = "goal"        # 会话目标
    ENVIRONMENT = "environment" # 运行环境
    PROJECT     = "project"     # 项目信息
    CONCEPT     = "concept"     # 领域概念
    AGENT       = "agent"       # Agent 身份
```

### 3.2 状态生命周期（8 状态）

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

修订状态（决策可被修订）：
SUPERSEDED  被更新决策取代，保留在历史中
INVALIDATED 被明确否定/取消，恢复时作为警告提示
ARCHIVED    仍有效但不再相关，从恢复块隐藏
```

核心转换是单调的 —— 降级被拒绝（`STATUS_ORDER` 字典确保 `COMPLETED` 永远不会回退到 `PENDING`）；`MODIFIED` 作为向后兼容别名保留。

> **类比理解**：状态生命周期相当于给每个会话元素装了一个「红绿灯」。绿灯=COMPLETED（可以通过了），黄灯=IN_PROGRESS（正在处理），红灯=FAILED（需要关注）。传统平面文本方法无法区分这三种状态 —— 它们只看到文字，看不到信号灯。

### 3.3 7 种边类型

```python
class EdgeType(str, Enum):
    DEPENDS_ON  = "depends_on"   # 任务 A 依赖任务 B
    RELATED_TO  = "related_to"   # 概念关联
    IMPLEMENTS  = "implements"   # 文件实现某功能
    FIXES       = "fixes"        # 文件修复解决某错误
    BLOCKS      = "blocks"       # 阻塞关系
    PART_OF     = "part_of"      # 部分-整体
    SUPERSEDES  = "supersedes"   # 替代关系
```

边连接节点形成结构化关系网络。例如：
- `TASK:"setup Redis"` —[`DEPENDS_ON`]→ `DECISION:"Redis over PostgreSQL"`
- `FILE:"src/cache.py"` —[`IMPLEMENTS`]→ `TASK:"setup Redis"`
- `ERROR:"connection timeout"` —[`BLOCKS`]→ `TASK:"setup Redis"`
- `FILE:"src/auth.py"` —[`FIXES`]→ `ERROR:"422 response on login"`

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

TokenMizer 的默认路径是**确定性启发式**（`graph_memory/hybrid_extractor.py`）：针对 task、decision、file、error、environment、endpoint 触发模式的编译正则管线，含复合句拆分与 CSV 决策拆分。它不做任何推理调用，成本为 $0。
- **默认（启发式）**：编译正则管线，8.1–529.9 ms/会话
- **未来路径（LLM）**：将最近消息发送给小型模型以结构化 JSON 提示提取（未在本文评估）

### 4.2 正则提取示例

```python
# 触发模式类别（task / decision / file / error / environment / endpoint）
# 以下为常见模式示意，非论文中的完整模式表
_COMPLETED = re.compile(
    r'(?:Completed|Implemented|Fixed|Added)[:\s]+(.+?)(?:\.|$)', re.I)
_IN_PROGRESS = re.compile(
    r'(?:Working on|Adding|Building)[:\s]+(.+?)(?:\.|$)', re.I)
_PENDING = re.compile(
    r'(?:Need to|TODO|Will|Next)[:\s]+(.+?)(?:\.|$)', re.I)
_DECISION = re.compile(
    r'(?:Decided|Using|Chose|Selected)[:\s]+(.+?)(?:\.|$)', re.I)
_FILE = re.compile(
    r'([\w./][\w./\-]*\.(?:py|js|ts|jsx|tsx|yaml|yml|json|md|go|rs|tf|toml|sh))', re.I)
```

该管线是**零推理成本**的默认路径，但受限于显式措辞：
1. **只能识别显式触发词**：如 "Decided: ..."、"Completed: ..."，无法覆盖隐式表达（"let's go with Redis"）
2. **基准对启发式有利**：3 个会话全部使用显式命令式措辞，论文明确警告不要将结果外推到隐式表述的会话（L2）

### 4.3 复合句与 CSV 决策拆分

论文的默认提取器包含两项结构性处理（与「模糊匹配」无关）：
- **复合句拆分**：在句号后拆分复合标签，使 "Setup Redis. Configure Nginx" 被识别为两个独立任务。
- **CSV 决策拆分**：对逗号和 `and` 分隔的决策进行拆分。

论文**没有报告提取器消融**：既没有 V1/V2 对比，也没有对触发词、拆分等单项的消融数字。**模糊匹配不是提取器的改进项**，而是评测的评分协议（Definition 5）：标签 a、b 匹配当且仅当 a ⊆ b、b ⊆ a，或 ≥3 字符 token 集合的重叠比例 ≥50%。论文明确说明该协议对图方法和基线**同样适用**（"The identical protocol scores both methods"）。

```python
# 评分协议（Definition 5 / Appendix），非提取器代码
def _fuzzy_match(a: str, b: str) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    if a in b or b in a:
        return True
    wa = set(re.findall(r'\w{3,}', a))
    wb = set(re.findall(r'\w{3,}', b))
    if not wa or not wb:
        return False
    shorter = wa if len(wa) <= len(wb) else wb
    return len(wa & wb) / len(shorter) >= 0.50
```

> **类比理解**：模糊匹配相当于给手工标注和自动提取之间架了一座桥。手工标注者写下 "redis setup"，LLM 助手说 "I've completed setting up a Redis cache for session management" —— 传统精确匹配判定为不匹配（0%），模糊匹配看到 2/2 tokens 重叠 → 判定匹配（100%）。但这是**评测口径**，两种方法都享受同样的匹配宽松度，因此并不构成 TokenMizer 相对基线的优势。

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

- 启发式提取：**8.1–529.9 ms/会话**（3 会话基准，均值 187.2 ms，中位数 23.6 ms）
- 18 轮会话为 529.9 ms，是 10 轮会话中位数的 22×，论文明确表示原因待 profiling（L7）
- LLM 提取器（未来路径）：未在本文评估

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

**序列化示例输出（示意，非实测规模；标准层实测均值约 254 tokens）**：
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

### 5.4 压缩管线（未测量）

TokenMizer 的压缩引擎对**大输入**应用分级启发式（AI 填充词移除、保序去重、空白归一化、语言感知注释剥离、带保护近期窗口的历史修剪、文件类型感知截断），并可在语义相似度质量门后可选启用 LLMLingua 系列神经阶段；输出修剪器在非流式路径上压缩冗长响应。

**论文明确的范围说明**：v0.3.1 结果文件**不含任何压缩或缓存测量**，因此本文不对这两个子系统作定量声明（Section V-C；L6）。

---

## Ch6: 实验评估与结果分析

### 6.1 实验设置

**Benchmark**: 3 个合成会话（heuristic-only 提取，单一 plain-summary 基线）：

| 会话 | 领域 | 轮数 | GT 任务/决策/文件 |
|------|------|------|-------------------|
| fastapi_auth | Backend auth service | 18 | 5/5/5 |
| react_dashboard | Frontend dashboard | 10 | 3/5/4 |
| ml_pipeline | ML training pipeline | 10 | 3/4/4 |

所有会话使用显式命令式措辞（"Decided: ..."），对启发式提取有利，由此产生的天花板效应在论文 Section XI 讨论。每个会话由作者标注（任务、决策、文件）；作者自建会话、ground truth 与提取器构成论文承认的三重冲突（L4）。

### 6.2 对比基线

单一基线是 **plain-summary**：对拼接后的完整转录做关键词扫描 —— 决策用技术名词匹配、文件用路径模式、任务用完成标记片段，近似「总结这段对话」会浮现的内容。作者明确给出两个**偏向基线**的说明：它读取完整转录、没有 token 预算；其技术关键词表与 ground-truth 决策词表高度重叠。LLM 摘要、MemGPT 式记忆、检索等更强基线留作未来工作。

| 方法 | Task Recall | Decision Recall | File Recall |
|------|------------|-----------------|-------------|
| Plain-summary 基线 | 75.6% | 70.0% | 91.7% |
| **TokenMizer v0.3.1** | 75.6% | **85.0%** | **100%** |

**关键观察**：

1. **任务回忆率完全持平**（75.6%）：在显式完成标记下，关键词扫描与结构化提取同样能恢复任务。

2. **决策回忆率分离两种方法**（85.0% vs 70.0%，均值 +15.0 pp；两个有差异的会话分别 +20 与 +25 pp）。决策回忆率是「结构化记忆质量」的核心指标，因为它衡量的是「是否保留了**为什么**」而非「**什么**」。

3. **文件回忆率图为 100%、基线 91.7%**，但论文指出此处很可能存在天花板效应，需要更难会话来可靠区分。按 Eq. (1)，平均信息保留率为 86.9%（图）vs 79.1%（基线）。

### 6.3 Token 效率指标 η

$\eta = \frac{1-L}{|R|/100} = \frac{\text{mean recall}}{|R|/100}$

论文只报告了 3 个会话的 η，**没有按领域分解**：

| 会话 | RT (tokens) | η |
|------|------------|---|
| fastapi_auth | 302 | 0.29 |
| react_dashboard | 260 | 0.32 |
| ml_pipeline | 201 | 0.46 |
| **均值** | **254.3** | **0.35** |

### 6.4 消融与归因

论文**没有报告提取器消融**（无 V1/V2 对比，也无模糊匹配、触发词、拆分等单项消融）。模糊匹配是评测的评分协议（Definition 5），而非提取器的改进项，且对图方法和基线同样适用。论文对结果的归因是定性的：在显式完成标记下任务回忆率持平，差异集中在决策回忆率。

### 6.5 论文承认的局限

- **n=3 合成会话**：冒烟级基准，均值仅作方向参考（L1）。
- **有利措辞与天花板效应**：所有会话使用启发式所针对的显式命令式风格；文件回忆率 100%、两个决策回忆率接近 100%，基准无法在顶部区分（L2）。
- **单一弱基线**：读取完整转录、关键词表与 ground-truth 重叠（L3）。
- **作者自建一切**：会话、ground truth、提取器同一人完成，可能因共享词汇抬高回忆率（L4）。
- **精度未报告**：结果文件只持久化 recall（L5）。
- **延迟扩展未 profiling**：18 轮会话的 529.9 ms 原因未定（L7）。
- **无端到端恢复指标**：实体回忆只是代理（L8）。

### 6.6 语义缓存（未测量）

实现为 sentence-transformer embeddings + cosine 相似度阈值（默认 0.92），带 LRU 驱逐与 TTL；但 v0.3.1 结果文件**不含缓存测量**，论文不对命中率作任何声明（Section V-C；L6）。

---

## Ch7: 局限性、部署指南与延伸阅读

### 7.1 已知局限

1. **Synthetic benchmark**：仅 3 个合成会话，由作者构建和标注，未覆盖真实生产环境的多样性和噪声。
2. **Single annotator**：ground-truth 标注由单一标注者完成，可能存在标注一致性偏差。
3. **LLM 提取器未评估**：已实现但未评估，对隐式推理场景可能显著改善。
4. **压缩/缓存未测量**：v0.3.1 结果文件无压缩率或缓存命中率，论文不作定量声明。
5. **精度未报告**：运行器计算 precision，但结果文件只持久化 recall。

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
- **MemGPT**: Packer et al., 2023
- **LLMLingua / LongLLMLingua**: Jiang et al., 2023
- **GraphRAG**: Edge et al., 2024

---

*报告完成于 2026-06-07 | Deep Read*