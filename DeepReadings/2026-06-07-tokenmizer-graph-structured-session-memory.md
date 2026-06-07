# TokenMizer: 图结构会话记忆 — 为长周期LLM会话管理上下文

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

### 关键结果

| 指标 | TokenMizer V2 | 最佳基线 | 优势 |
|------|--------------|---------|------|
| 恢复块大小 (avg) | **78 tokens** | 159 tokens (Sliding Window) | **2.1× 更小** |
| 决策回忆率 | **46.6%** | 38% (Naive Summary) | **+8.6 pp** |
| 任务回忆率 | 51.0% | 50% (Sliding Window) | +1.0 pp |
| 文件回忆率 | 58.7% | 60% (Sliding Window) | -1.3 pp |

**核心结论**：没有一个基线保留了决策的「为什么」。滑动窗口可能保留技术的名称，但丢失了选择它而不是替代方案的**理由**。TokenMizer 用一半的 token 成本做到了这一点。