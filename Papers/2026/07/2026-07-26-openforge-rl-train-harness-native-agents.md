
> **论文**：OpenForge RL: Train Harness-native Agents in Any Environment
> **作者**：Paul Furgale, Severin Klingler, James Nolan, Matt Staats, Gaia Di Lorenzo, Elisa Martinez Abad, Christian Schüller, Razvan Dinu, Alessio Devoto, Pascal Berard, Gal Kaplun, Elad Sarafian, Riccardo Roveri, Leon Derczynski, Ricardo Silveira Cabral (Microsoft)
> **arXiv ID**：2607.21557
> **发表时间**：2026-07-23
> **代码仓库**：将开源（论文发布时尚未公开）

## 第 1 章 概述

### 1.1 一句话定位

OpenForge RL 是一个开源框架，通过轻量级代理（Proxy）和 Kubernetes 编排器，将任意推理 Harness（Claude Code、Codex、OpenClaw 等）与标准 RL 训练栈（veRL 等）解耦，首次实现 Harness 原生 Agent 在多样环境中的端到端强化学习训练。

### 论文图表总览

| 编号 | 内容描述 | 所在章节 |
|------|---------|---------|
| **Figure 1** | 系统架构概览（Agent inside Harness） | 第 3 章 |
| **Figure 2** | OpenForge RL 训练流程——Proxy 将 Harness Model Call 转发至训练栈 | 第 3 章 |
| **Figure 3** | 数据合成管线：Propose → Prune → Build → Test → Refine | 第 3 章 |
| **Figure 4** | 训练任务分类分布 | 第 5 章 |
| **Figure 5** | 工具使用分布（左）与 Agent 能力雷达图（右） | 第 6 章 |
| **Table 1** | 训练数据统计：各领域 SFT 与 RL 任务数 | 第 3 章 |
| **Table 2** | Claw Agents 主结果（ClawEval / QwenClawBench / MCPAtlas） | 第 5 章 |
| **Table 3** | GUI Agents 主结果（OSWorld / Mind2Web / WebVoyager） | 第 5 章 |
| **Table 4** | 跨 Harness 评估（ReACT / ZeroClaw / OpenClaw / Codex） | 第 6 章 |
| **Table 5** | 未见 Harness 泛化实验 | 第 6 章 |
| **Table A1** | 训练超参数汇总 | 第 7 章 |

### 1.2 核心贡献

1. **OpenForge RL 框架**：提出 Proxy 解耦 + Kubernetes 远程容器编排的通用训练基础设施，将任意 Harness × 任意环境的组合接入标准 RL 代码库（veRL），无需为每个 Harness 定制训练循环。

2. **广泛实证研究**：覆盖文本型工具使用/Claw Agents（4 种 Harness × 3 个 Benchmark）和多模态 GUI Agents（浏览器使用 + 桌面使用共 3 个 Benchmark），在几乎全部 Benchmark 上超越同尺寸开源模型，部分场景持平或超越数倍大的模型。

3. **Harness 对 Agent 行为影响分析**：借助端到端训练能力，首次系统分析 Harness 选择如何影响 Agent 学习难度——更简单、设计更好的 Harness 更易学习；RL 训练提升 Agent 可靠性（自验证、工具覆盖、多步规划），但错误恢复能力仍然薄弱。

### 1.3 关键结果速览

| 场景 | 模型 | Benchmark | 指标 | OpenForge | 基线（同尺寸） | 最佳闭源 |
|------|------|----------|------|-----------|-------------|---------|
| Claw | 30B-A3B MoE | ClawEval | pass^3 | **31.7** | 14.3（Qwen3 基座） | 70.8（Claude Opus 4.6） |
| Claw | 30B-A3B MoE | ClawEval | pass@3 | **55.9** | 39.8（Qwen3 基座） | 80.8（Claude Opus 4.6） |
| Claw | 30B-A3B MoE | QwenClawBench | pass@1 | **33.7** | 21.8（Qwen3 基座） | 59.5（Claude Opus 4.6） |
| Claw | 30B-A3B MoE | MCPAtlas | pass@1 | **28.1** | 12.4（Qwen3 基座） | 76.4（Claude Opus 4.6） |
| GUI(Computer) | 8B VLM | OSWorld-Verified | 成功率 | **37.7** | 29.4（Qwen3-VL-8B） | 76.2（Gemini 3.1 Pro） |
| GUI(Browser) | 8B VLM | Online-Mind2Web | 成功率 | **63.0** | 38.7（Qwen3-VL-8B） | 92.8（GPT 5.4） |
| GUI(Browser) | 8B VLM | WebVoyager | 成功率 | **72.3** | 49.2（Qwen3-VL-8B） | — |

## 第 2 章 研究背景与动机

### 2.1 Harness 原生的 Agent 生态

现代 AI Agent 已不再是裸 LLM/VLM，而是被越来越复杂的**推理 Harness** 包裹——这些编排框架管理多轮交互、工具调用、上下文，并连接 MCP 服务器、浏览器和 GUI。代表性的 Harness 包括 Claude Code、Codex 和 OpenClaw。Harness 层已成为 Agent 能力的核心：近期 Agent Benchmark 的进步有相当比例来自 Harness 设计本身，而非更强的基座模型。

### 2.2 端到端训练的两大障碍

尽管 Harness 装备的 Agent 能力出色，但端到端改进它们对开源社区仍然遥不可及，原因有二：

1. **训练栈无法表达 Stateful Harness 推理**：Harness 将推理转变为包含嵌套工具调用、子 Agent 和长程上下文管理的 stateful 多进程过程。现有开源 RL 框架（veRL、Slime、SkyRL）假设简单 rollout（单轮生成或带轻量工具调用的多轮交互），无法原生表达这种复杂度。开源努力往往需要为训练重写简化版 Harness，造成训练-部署鸿沟（train-deploy mismatch）。

2. **规模化 Harness 需要容器化环境**：Harness rollout 需要专用的容器化环境（CPU 和内存），不能与训练节点同置——但大多数开源 RL 框架假设 rollout 在训练器本地运行。

### 2.3 相关工作

**Agent Inference Harnesses**：早期 SWE-Agent 展示了精心设计的 Agent-Computer Interface 能显著提升任务成功率。后续 Claude Code 和 Codex 将其精炼为软件工程专用，开源 OpenClaw 则扩展到了日常任务和 GUI 任务。

**训练 Harness 原生 Agent**：现有开源 RL 框架（veRL、Slime、OpenRLHF、SkyRL、AReaL）支持 PPO 和 GRPO 等高级算法，但假设简单 rollout——无法处理复杂 Harness 的内部控制流和上下文管理。早期尝试（DigiRL、Search-R1、WebRL）需要为每个任务大幅修改训练循环，代码难以扩展和维护。

## 第 3 章 OpenForge RL 系统架构

### 3.1 问题形式化

在复杂、长程环境中完成任务可建模为 MDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle$。Agent $\pi_\theta$ 接收观察 $s_t \sim \mathcal{S}$，生成动作 $a_t \sim \pi(\cdot|s_t)$，转移至下一观察 $s_{t+1}$。

对于复杂任务（编码、GUI 控制），Agent 被包裹在 Harness $\mathcal{H}(\pi)$ 中，提供子 Agent、技能、规划模式等内部工具和控制流。Harness 中的每个 prompt-response pair 表示为 $(\mathcal{H}(s_t), a_t)$，完整轨迹为 $\tau = \langle (\mathcal{H}(s_0), a_0), (\mathcal{H}(s_1), a_1), \ldots, (\mathcal{H}(s_T), a_T) \rangle$。

### 3.2 核心架构

OpenForge RL 包含两个核心组件：

**轻量级 Proxy**：包装推理服务器（如 vLLM），拦截 Harness 容器发出的所有生成请求。当 rollout 完成时，Proxy 收集最终奖励 $r_T$ 和 prompt-response pairs，重建训练样本：

$$
\tau = (s_0^{\mathcal{H}}, a_0, r_0), (s_1^{\mathcal{H}}, a_1, r_1), \ldots, (s_T^{\mathcal{H}}, a_T, r_T); \quad r_t = \gamma^{T-t} \cdot r_T
$$

通常 $\gamma = 1.0$。使用 GRPO 时，按照 Feng et al. 的方法计算优势——比较同一组内不同轨迹的平均样本奖励。

**Kubernetes 编排器**：基于 Orchard 实现，在云提供商（如 Microsoft Azure）上创建、管理和删除 rollout 容器 Pod。支持弹性扩展，不超载训练节点。

### 3.3 关键工程方案

**异步 Rollout 与超时**：因每个 rollout 远程运行且不受训练器控制，单个无响应的 rollout 会阻塞整个训练批次的收集。论文采用墙上时钟超时——当 job 超过超时时间，终止该 job 并返回错误信号，训练继续收集其余 rollout 的数据。

**错误处理**：环境中的 rollout 可能因网络问题、Harness 崩溃或超时而失败。论文遵循 DAPO 策略——丢弃因错误终止的轨迹的所有样本，因为部分 rollout 可能注入误导性训练信号（如正确的前缀收到负奖励）。

### 3.4 数据合成

由于 Claw 和计算机使用等领域缺乏大规模公开训练任务，论文构建了一条数据合成管线，设计五个阶段：Propose（从资产池生成候选任务）→ Prune（去重和筛选）→ Build（构建可执行环境 + Verifier）→ Test（用独立 LLM/VLM 尝试任务）→ Refine（修补缺陷）。该管线支持 SFT 和 RL 任务，可扩展至 Linux/CLI、GUI 和计算机使用。

| 领域 | Harness 类型 | SFT 轨迹数 | RL 任务数 |
|------|------------|-----------|---------|
| Claw（日常工具使用） | ReACT, ZeroClaw, OpenClaw, Codex | 892 | 343 |
| GUI（计算机使用） | Kimi-Agent（修改版） | 795 | 252 |
| GUI（浏览器使用） | MolmoWeb（修改版） | 1,496 | 900 |

单个 RL 任务合成平均耗时 16.1 分钟/4.36 美元（Claw），21.3 分钟/6.12 美元（计算机使用）。

## 第 4 章 实验结果

### 4.1 Claw Agents

**设置**：使用 Qwen3-30B-A3B-Thinking 作为骨干模型。SFT 从更强的教师模型（MiniMax-M2.5）蒸馏成功轨迹（N=3），RL 继续训练使用 GRPO。训练后端为 veRL，云提供商为 Microsoft Azure，8×B200 GPU，batch size=8，group size=8。

**评价 Benchmark**：ClawEval（pass^3 和 pass@3）、QwenClawBench（pass@1）、MCPAtlas（pass@1）。

**主要结果**：

| System | ClawEval(pass^3) | ClawEval(pass@3) | QwenClawBench(pass@1) | MCPAtlas(pass@1) |
|--------|:---------------:|:---------------:|:--------------------:|:---------------:|
| **闭源大模型** | | | | |
| Claude Opus 4.6 | 70.8 | 80.8 | 59.5 | 76.4 |
| GPT 5.4 | 60.2 | 75.8 | 56.7 | 68.5 |
| Gemini 3.1 Pro | 55.9 | 80.8 | — | — |
| **同尺寸模型（~3B 激活参数）** | | | | |
| Qwen3-30B-A3B-Thinking(基座) | 14.3 | 39.8 | 21.8 | 12.4 |
| Qwen3-Coder-30B-A3B-Instruct | 30.4 | 49.7 | 24.3 | 19.1 |
| **OpenForge-Claw** | | | | |
| OpenForge-Claw(SFT) | 21.7 | 52.1 | 32.1 | 23.6 |
| OpenForge-Claw(SFT+RL) | **31.7** | **55.9** | **33.7** | **28.1** |

OpenForge-Claw(SFT+RL) 在全部三个 Benchmark 上超越同尺寸模型，并在 QwenClawBench 上相对基座提升 11.9pp（21.8→33.7）。RL 训练在 pass^3（鲁棒性）和 pass@1（平均成功率）上均带来显著改善。

### 4.2 GUI Agents

**设置**：使用 Qwen3-VL-8B-Thinking 为骨干，蒸馏 Kimi-K2.5 的成功轨迹。计算机使用环境使用 Kimi-Agent（修改版，增加 bash 和 str_replace_editor 工具），浏览器使用使用 MolmoWeb 修改版。均采用截图模式（仅视觉输入，无 DOM 或辅助数据）。

**主要结果**：

| System | OSWorld-Verified | Online-Mind2Web | WebVoyager |
|--------|:---------------:|:---------------:|:----------:|
| **闭源 VLM** | | | |
| GPT 5.4 | 75.0 | 92.8 | — |
| Claude Opus 4.6 | 72.7 | 84.0 | — |
| Gemini 3.1 Pro | 76.2 | — | — |
| **同尺寸模型（~8B）** | | | |
| Qwen3-VL-8B | 29.4 | 38.7 | 49.2 |
| UI-TARS-1.5-7B | 27.4 | 31.3 | 66.4 |
| OpenForge-GUI(SFT) | 34.4 | 57.4 | 61.5 |
| OpenForge-GUI(SFT+RL) | **37.7** | **63.0** | **72.3** |

OpenForge-GUI(SFT+RL) 在 OSWorld-Verified 上达 37.7%，超越所有 8B 级别开源模型；在 Online-Mind2Web 上达 63.0%（超越 MolmoWeb-8B 的 35.3%）。值得注意的是，MolmoWeb 使用 20 万+ 任务训练，而 OpenForge-GUI 仅使用 2,500 个 SFT 任务 + 900 个 RL 任务。

## 第 5 章 训练分析与讨论

### 5.1 跨 Harness 评估

将 OpenForge-Claw 模型在 ClawEval 上用四种不同 Harness 评估：

| Model | ReACT*(p@1) | ZeroClaw(p@1) | OpenClaw(p@1) | Codex(p@1) |
|--------|:----------:|:------------:|:------------:|:---------:|
| Qwen3-30B-A3B-Thinking | 26.1 | 32.5 | 11.4 | 12.2 |
| OpenForge-Claw(SFT) | 36.2 | 44.5 | 16.7 | 21.1 |
| OpenForge-Claw(SFT+RL) | **45.1** | **48.5** | **20.9** | **32.5** |

两个趋势清晰可见：
1. **Harness 对齐度决定性能**：支持直接注册自定义工具的 Harness（ReACT、ZeroClaw）达到最高性能。OpenClaw 和 Codex 虽功能丰富，但不支持自定义工具（需通过 SKILL.md 文件暴露），性能较低。
2. **RL 训练带来全面收益**：SFT+RL 在除 OpenClaw 外的所有 Harness 上带来大幅提升，但 OpenClaw 增益相对有限——其更长的 prompt 和上下文可能稀释了 RL 的学习信号。

### 5.2 未见 Harness 泛化

| 训练方式(SFT+RL) | ZeroClaw(p@1) | OpenClaw(p@1) | Codex(p@1) |
|-----------------|:------------:|:------------:|:---------:|
| 无训练（基座） | 32.5 | 11.4 | 12.2 |
| 仅 ZeroClaw 训练 | 46.0 (+13.5) | 14.7 (+3.3) | 16.8 (+4.6) |
| ZeroClaw+OpenClaw+Codex | **48.5 (+16.0)** | **20.9 (+9.5)** | **32.5 (+20.3)** |

训练单一 Harness（ZeroClaw）已能泛化到未见过的 Harness：OpenClaw +3.3pp，Codex +4.6pp。而同时训练三个 Harness 在所有情况下最优——在复杂 Harness（Codex）上带来最大的绝对提升（+20.3pp），并反哺 ZeroClaw 自身（48.5 vs 46.0）。这表明多样性训练使模型在面对不同工具和场景时更加鲁棒。

### 5.3 RL 学到的能力

对 SFT 和 SFT+RL 检查点的对比分析（各 100 条轨迹）揭示了 RL 带来的行为变化：

**工具使用模式变化**（ZeroClaw Harness）：
- 通用 shell 调用从 22.6% 降至 13.9%
- 使用模式向专用服务工具重新分配
- 轨迹长度略微缩短

**高级 Agent 能力变化**（Codex Harness）：
- **格式鲁棒性**：格式错误导致的中断率降低
- **错误恢复**：遇到失败命令后仍能完成任务的比例提升
- **自验证**：写操作后回读确认的比例提高
- **工具覆盖**：多步骤任务中调用所有必需工具的比例提高
- **步骤效率**：接近最短路径完成任务的能力提升

**错误恢复**仍然是所有能力中最薄弱的环节，即使经过 RL 训练。这表明某些能力可能需要专门的数据或训练方法才能进一步增强。

## 第 6 章 局限性与未来方向

### 6.1 局限性

1. **数据合成的成本与规模**：单个 RL 任务合成需要 16.1 分钟和 4.36 美元（Claw 任务），计算机使用任务更高（21.3 分钟/6.12 美元）。当前生成的任务池（数百到数千）相较闭源系统的训练规模仍然有限。

2. **错误恢复能力薄弱**：RL 训练后错误恢复仍是 Agent 最弱的能力。论文推测这需要专门的数据或训练方法——纯粹的 RL 信号可能不足以让模型学会从错误中恢复。

3. **对 Harness 本身的依赖**：简单、对齐良好的 Harness（ReACT、ZeroClaw）学习效率远高于复杂 Harness（OpenClaw、Codex）。后者的长 prompt 和丰富的控制流可能稀释 RL 学习信号。如何改进复杂 Harness 的可训练性仍是开放问题。

4. **训练资源的 GPU 需求**：当前实验使用 8×B200 GPU，对于独立研究者而言门槛不低。

### 6.2 未来方向

- 改进部分 Rollout 的信度分配——当前策略丢弃错误轨迹的所有样本，可能浪费有价值的训练信号
- 开发针对错误恢复的专门训练方法
- 扩展到更多类型的环境（移动设备、嵌入式系统等）
- 降低数据合成成本，提升任务覆盖的多样性
