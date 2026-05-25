# Arxiv LLM Daily Papers \- 每日大模型论文抓取仓库

**每日自动抓取 Arxiv 最新 LLM 相关论文 \| 大模型智能筛选、摘要提炼、结构化归档**



## 项目介绍

本仓库是一款**基于大模型驱动的 Arxiv LLM 领域论文自动抓取工具**，依托 GitHub Actions 实现全天候自动化运行，每日定时爬取 Arxiv 平台最新发布的大语言模型（LLM）相关论文。

区别于传统纯爬虫项目，本项目通过大模型对抓取的原始论文数据进行**智能筛选、内容清洗、摘要精炼、分类归档**，过滤水文、重复及无关论文，精准留存高质量LLM前沿研究成果，帮助开发者、科研人员快速追踪大模型领域最新学术动态，无需每日手动翻阅Arxiv。

## 核心功能

- **每日定时自动抓取**：依托 GitHub Actions 定时任务，每日固定时间扫描 Arxiv 最新收录的 LLM 领域论文，全程无人值守自动化运行。

- **大模型智能筛选**：通过大模型语义理解能力，精准识别纯LLM相关论文，过滤多模态、CV、NLP通用无关论文，剔除低质、灌水、重复稿件。

- **AI 内容精炼处理**：大模型自动翻译英文标题、提炼论文核心摘要、总结创新点、标注研究方向（微调、对齐、推理、Agent、多模态LLM等）。

- **结构化归档存储**：每日论文自动生成独立归档文档，按日期分类存储，包含论文标题、作者、发布时间、Arxiv链接、AI精炼摘要、核心创新点等完整信息。

- **历史数据汇总统计**：自动汇总每日更新内容，生成月度、季度论文汇总清单，方便批量查阅和复盘领域研究趋势。

- **轻量化部署**：无需本地服务器，依托GitHub免费算力运行，配置简单、零运维成本、稳定可靠。

## 技术栈

- **核心语言**：Python 3\.9\+

- **数据来源**：Arxiv Official API

- **智能处理**：通用大模型API（支持OpenAI、通义千问、讯飞星火等主流模型）

- **自动化调度**：GitHub Actions

- **数据存储**：Markdown 结构化文档（易读、易检索、易分享）

## 论文输出格式

每日生成的论文文档包含以下标准化内容：

- 论文英文标题 \+ 中文翻译

- 作者信息 \& 所属机构

- 发布时间 \& Arxiv 编号

- 原文链接 \& PDF下载链接

- LLM 精炼核心摘要（去除冗余学术套话）

- 核心创新点 \& 研究价值总结

- 研究方向分类标签

## 项目优势

- **AI 赋能，告别无效信息**：区别于普通爬虫只爬取原始数据，大模型智能筛选提纯，只保留高价值LLM前沿论文。

- **零成本全自动**：依托GitHub免费算力，无需服务器、无需付费资源，长期稳定更新。

- **轻量化易拓展**：代码结构清晰，支持自定义筛选规则、替换任意大模型、新增统计/分类功能。

- **阅读体验极佳**：结构化Markdown归档，排版整洁，支持检索、回溯、批量阅读。

## 更新计划

- 支持论文热度、引用趋势智能分析

- 新增领域细分分类（LLM推理、微调、Agent、对齐、轻量化等）

- 支持每周/每月领域研究趋势总结报告

- 新增论文关键词检索功能

## 📚 论文索引（每日精读）

| 日期 | 论文标题 | 作者 | 链接 |
|------|---------|------|------|
| 2026-05-25 | Advancing Mathematics Research with AI-Driven Formal Proof Search | George Tsoukalos et al. (Google DeepMind) | [arXiv](https://arxiv.org/abs/2605.22763) \| [精读](Papers/2026/05/2026-05-25-advancing-mathematics-research-ai-formal-proof-search.md) |
| 2026-05-24 | Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention | Ali Hatamizadeh et al. | [arXiv](https://arxiv.org/abs/2605.22791) \| [精读](Papers/2026/05/2026-05-24-gated-deltanet-2-linear-attention.md) |
| 2026-05-23 | Vector Policy Optimization: Training for Diversity Improves Test-Time Search | Ryan Bahlous-Boldi et al. | [arXiv](https://arxiv.org/abs/2605.22817) \| [精读](Papers/2026/05/2026-05-23-vector-policy-optimization.md) |
| 2026-05-22 | MOSS: Self-Evolution through Source-Level Rewriting in Autonomous Agent Systems | Qianshu Cai et al. | [arXiv](https://arxiv.org/abs/2605.22794) \| [精读](Papers/2026/05/2026-05-22-moss-self-evolution-source-level-rewriting.md) |

## 📖 深度阅读（Deep Readings）

> 交互式深度论文精读，包含完整的论文阅读报告 + 代码逐行解析，由 Claude Code 生成 + Review 循环优化。

| 日期 | 论文标题 | 作者 | 链接 |
|------|---------|------|------|
| 2023-12-01 | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Albert Gu, Tri Dao | [arXiv](https://arxiv.org/abs/2312.00752) \| [深度阅读](DeepReadings/2023-12-01-mamba-linear-time-sequence-modeling-selective-state-spaces.md) |

---

## 支持一下

如果本项目对你有帮助，欢迎 **Star**、**Fork** 支持！持续更新LLM前沿学术动态，助力科研与学习。

## 许可证

本项目基于 **MIT License** 开源，可自由用于学习、科研、二次开发。



> （注：文档部分内容可能由 AI 生成）
