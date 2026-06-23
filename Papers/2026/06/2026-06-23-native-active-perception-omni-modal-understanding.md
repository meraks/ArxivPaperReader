# Native Active Perception as Reasoning for Omni-Modal Understanding

## 论文元数据
- 标题：Native Active Perception as Reasoning for Omni-Modal Understanding
- 作者：Zhenghao Xing, Ruiyang Xu, Yuxuan Wang, Jinzheng He, Ziyang Ma, Qize Yang, Yunfei Chu, Jin Xu, Junyang Lin, Chi-Wing Fu, Pheng-Ann Heng（CUHK, SJTU, Qwen 团队/阿里巴巴, NTU）
- arXiv ID：2606.19341
- 发表/提交日期：2026-06（ICML 2026）
- 官方代码：[OmniAgent](https://github.com/harryhsing/OmniAgent)
- 模型权重：[OmniAgent-SFT-7B](https://huggingface.co/harryhsing/OmniAgent-SFT-7B), [OmniAgent-RL-7B](https://huggingface.co/harryhsing/OmniAgent-RL-7B)

---

## Ch1: 论文概述与核心贡献

### 问题背景

传统长视频理解模型采用"watch-it-all"被动范式，对所有帧进行统一处理而不考虑查询难度，导致计算成本随视频时长线性增长。尽管已出现一些交互式框架，但它们通常依赖全局预扫描，上下文成本仍随视频长度扩展。

### 核心方案

OmniAgent 是首个原生全模态（omni-modal）智能体，将视频理解形式化为 POMDP（部分可观测马尔可夫决策过程）上的迭代 Observation-Thought-Action（OTA）循环。该智能体通过按需行动将音视觉线索蒸馏到持久化文本记忆中，有效解耦推理复杂度与原始视频时长。

### 三大技术创新

1. **Agentic SFT（Agentic Supervised Fine-Tuning）**：基于 best-of-N 轨迹合成与双阶段质量控制的冷启动训练方法。使用 58K 轨迹，覆盖 5 个数据集，采用结果验证（精确匹配/IoU/MRA）+ 理性审计（GPT-4o 5点Likert ≥3/5）两级质量过滤。

2. **TAURA（Turn-aware Adaptive Uncertainty Rescaled Advantage）**：熵引导的强化学习目标函数，解决多轮 GRPO（Group Relative Policy Optimization）中的 Advantage Homogenization 问题。通过轮级熵加权重新调整 advantage，79.2% 的关键分叉轮获得更高奖励权重。

3. **原生全模态架构**：同一模型处理语义理解、推理和行动选择，无外部感知模块，支持文本、图像、音频、视频的联合理解。

### 关键实验结果

**性能提升**：7B 模型在 10 个 benchmark 上全面超越 Qwen2.5-Omni-7B baseline，最大提升在时间定位任务（LongVALE IoU 从 5.7%→39.1%，+33.4 percentage points；VUE-TR Vision 从 8.0%→46.1%，+38.1 percentage points）。

**超越更大模型**：LVBench 上 OmniAgent-7B 达到 50.5%，超过 Qwen2.5-VL-72B 的 47.3%（+3.2 percentage points）。

**效率提升**：平均每视频处理帧数从 768 帧降至 203 帧，减少 73%。

**测试时扩展**：VideoMME-Long 上将轮数从 6 增至 52 时性能提升 6.2 percentage points，在约 11.7 轮时达到饱和。

---

## Ch2: 研究背景与动机

### 多模态基础模型的进展

近年来多模态基础模型快速发展，包括 Qwen2.5-Omni、GPT-4o、Gemini 等。这些模型在图文理解上取得显著进展，但视频理解仍面临高维时空数据带来的计算开销挑战。

### 视频理解的特殊挑战

视频数据不仅包含空间维度（图像帧），还包含时间维度（帧序列）。对于时长数分钟甚至数小时的视频，传统"统一采样+全局编码"范式导致：

- 计算成本随视频时长线性增长
- 简单查询与复杂查询消耗相同资源
- 长视频中的关键信息被大量无关帧稀释

### 现有主动/交互式方法的局限

VideoAgent、VideoAgent2、VriptAgent 等交互式方法尝试通过分步采样减少计算量，但仍存在以下问题：

1. **全局预扫描依赖**：需要先对全视频进行初步扫描，无法完全避免与视频长度相关的计算
2. **上下文成本扩展**：即使进行选择性采样，上下文窗口仍需容纳大量中间表示
3. **缺乏原生多模态支持**：多数方法仅处理视觉模态，未能充分利用音频信息

### 核心洞察

感知应当是迭代的、查询驱动的信息蒸馏过程，而非穷举预处理。智能体应根据当前问题状态主动选择感知行动，从原始多模态媒体中提取最相关的结构化信息，并在信息充足时及时终止感知。

### 全模态理解的必要性

真实世界视频包含丰富的多模态信息：视觉场景、物体运动、语音对话、环境音效等。完整理解视频内容需要联合建模：

- **文本**：问题、对话、字幕
- **图像**：关键帧视觉内容
- **音频**：语音、音效、背景音乐
- **视频**：时空动态过程

OmniAgent 首次在主动感知框架中实现全模态联合理解。

---

## Ch3: OmniAgent 架构——POMDP 与 OTA 循环

### POMDP 形式化

环境定义为 $\Omega = \{V, F, A, CL\}$：

- $V$：原始视频流
- $F$：视频帧集合（通过索引范围获取）
- $A$：音频流
- $CL$：视频片段（clip，包含帧和音频）

智能体通过观察 $O_k$、推理 $T_k$、行动 $A_k$ 与环境交互，最多进行 $K$ 轮循环。

### Observation-Thought-Action 循环

**Observation（观察）**：结构化文本摘要，将高维多模态感知压缩为可读文本。例如：

```
Frame 120-150: A person is cooking in a kitchen, holding a knife.
Audio 45-60s: "[Sound of chopping vegetables, background music playing]"
```

**Thought（推理）**：内部推理过程，连接当前观察与下一步行动。例如：

```
The question asks about the final dish. I need to see the cooking result.
I should get frames from the end of the video.
```

**Action（行动）**：符号化操作，精确指定待获取的信息：

- `get_frames(start, end)`：获取指定时间范围的帧
- `get_audio(start, end)`：获取指定时间范围的音频转录
- `get_clip(start, end)`：获取包含帧和音频的片段
- `answer(final_response)`：提交最终答案并终止

### 记忆管理机制

**持久化文本记忆**：所有 Observation 和 Thought 以文本形式累积在上下文窗口中，形成可追溯的推理链。

**原始媒体丢弃**：帧和音视频在使用后立即丢弃，不保留原始像素或音频数据。这使得上下文成本仅取决于推理复杂度，而非视频时长。

### OTA 算法伪代码（Algorithm 1）

```
Input: Question Q, Video V
Memory M ← {Q}
For k = 1 to K:
    O_k ← Observe(M, Action_History)  # 生成结构化观察
    T_k ← Think(M, O_k, Action_History)  # 生成推理思考
    A_k ← Act(T_k)  # 选择行动
    If A_k.type == "answer":
        Return A_k.response
    Else:
        Execute A_k on V  # 获取指定内容
        Add (O_k, T_k, A_k) to M
```

### 与现有方法的核心区别

1. **统一模型处理全流程**：同一模型负责语义理解、推理生成和行动选择，无需外部感知模块或策略网络
2. **原生全模态支持**：自然处理文本、图像、音频、视频，无需为不同模态设计独立编码器
3. **真正解耦时长与成本**：上下文仅包含文本记忆，原始媒体在使用后丢弃

---

## Ch4: 训练方法——Agentic SFT 与 TAURA

### Agentic SFT：冷启动训练

**数据规模**：58K 轨迹，来自 5 个数据集：

- LongVideo-Reason
- Video-Holmes
- VSI-Train-10k
- LongVALE
- MultiHop-EgoQA

**任务类型**：覆盖 3 类任务

1. **多选题（MCQ）**：从选项中选择正确答案
2. **数值推理（NUM/SIZE）**：输出数值答案
3. **时间定位（TR）**：输出时间范围区间

**Best-of-N 轨迹合成**：教师模型（Qwen2.5-Omni-7B）进行多次探索，允许自纠错。例如，首次探索选择了错误帧，后续探索可以纠正该选择，形成"错误→恢复→正确"的轨迹。

**双阶段质量控制**：

阶段 1 - 结果验证：
- 多选题：精确匹配选项字母
- 时间定位：IoU ≥ 0.5
- 大小估计：MRA ≥ 0.5

阶段 2 - 理性审计：
- 使用 GPT-4o 评估逻辑推理性（5点Likert量表）
- 过滤低于 3/5 分的轨迹（排除"运气猜测"）

**训练配置**：
- 2 epochs
- 学习率 1e-5
- Batch size 64
- 16×A100 80GB+

### Advantage Homogenization 问题

Vanilla GRPO 在多轮场景下存在的问题：

**问题定义**：多轮 GRPO 中，单个轨迹级 advantage 混合了关键发现轮和无关操作。例如，一个轨迹在第 3 轮做出了关键正确行动，但在第 1、2、5 轮的行动质量一般，vanilla GRPO 会给予整个轨迹相同的 advantage 权重，掩盖了关键轮次的重要性。

**实验证据**：79.2% 的关键分叉轮（critical bifurcation turns）的熵显著高于轨迹均值，说明这些轮次在策略决策中包含更多不确定性。

**影响**：RL 训练无法有效学习"在关键时刻做出正确决策"的策略，导致优化效率下降。

### TAURA：熵引导 Advantage 重调

**核心思想**：对每个轮次 k，根据该轮的熵 $H_{i,k}$ 重新调整 advantage：

$$Â_{i,k} = A_i · \frac{H_{i,k}}{\frac{1}{N_G} \sum_{j=1}^{N_G} \sum_{m=1}^{M} H_{j,m}}$$

其中：
- $A_i$：轨迹级原始 advantage
- $H_{i,k}$：轨迹 i 在轮次 k 的策略熵
- $N_G$：group size（GRPO 中的采样组大小）
- $M$：轨迹长度（轮次数）

**效果机制**：

1. **正确轨迹中的高熵轮**：获得更大的正 advantage（奖励探索性正确决策）
2. **错误轨迹中的高熵轮**：获得更大的负 advantage（惩罚混乱的猜测）
3. **低熵轮**：保持原始 advantage（对于确定性操作不进行调整）

**损失函数**：带 clipping 的逐 token 重要性采样，使用 $Â_{i,turn(t)}$ 替代传统 advantage：

$$L_{RL} = -\mathbb{E}_{\pi_\theta} \left[ \frac{\pi_\theta(a_t|o_t)}{\pi_{old}(a_t|o_t)} Â_{i,turn(t)} \right]$$

### TAURA vs Vanilla GRPO

实验结果表明，TAURA 在全部 10 个 benchmark 上优于 vanilla GRPO，特别是在时间定位任务上优势明显。

---

## Ch5: 实验结果与分析

### 实验设置

**基础模型**：Qwen2.5-Omni-7B

**训练配置**：
- SFT：16×A100 80GB+，2 epochs，lr 1e-5，batch 64
- RL：64×A100 80GB+，150 steps，group size 8，lr 1e-6

**评测基准**：10 个 benchmark，覆盖三类任务

| 类型 | Benchmark | 说明 |
|------|-----------|------|
| 视频理解 | VideoMME, VSI-Bench, MLVU, Minerva, LVBench | 通用视频问答、长视频推理 |
| 音视频 | DailyOmni, WorldSense, OmniVideoBench | 音频-视觉联合推理 |
| 时间定位 | LongVALE, VUE-TR | 时间范围定位、关键事件检测 |

视频时长从数秒到 2+ 小时。

### 主要结果（Table 1）

**对比 Qwen2.5-Omni-7B baseline**：

| Benchmark | OmniAgent-7B | Qwen2.5-Omni-7B | 提升幅度 |
|-----------|--------------|-----------------|----------|
| LVBench | 50.5% | 43.0% | +7.5 |
| VideoMME Overall | 67.8% | 64.8% | +3.0 |
| VideoMME Long | 59.6% | 54.8% | +4.8 |
| VSI-Bench | 48.4% | 35.5% | +12.9 |
| MLVU | 71.1% | 65.2% | +5.9 |
| Minerva | 41.4% | 33.4% | +8.0 |
| DailyOmni | 64.8% | 60.1% | +4.7 |
| WorldSense | 47.2% | 45.4% | +1.8 |
| OmniVideoBench | 37.1% | 29.3% | +7.8 |
| LongVALE (IoU) | 39.1% | 5.7% | +33.4 |
| VUE-TR (Vision+Audio) | 36.5% | 3.5% | +33.0 |
| VUE-TR (Vision) | 46.1% | 8.0% | +38.1 |

**关键发现**：
- 全部 10 个 benchmark 提升
- 时间定位任务提升最显著（+24~38 percentage points）
- 通用视频理解任务提升 2.3~7.5 percentage points

### 超越更大模型（Table 2）

LVBench 上对比更大规模模型：

| 模型 | 参数 | LVBench |
|------|------|---------|
| OmniAgent-7B | 7B | 50.5% |
| Qwen2.5-VL-72B | 72B | 47.3% |
| Qwen2.5-Omni-7B (baseline) | 7B | 43.0% |

OmniAgent-7B 超越 10× 参数的 Qwen2.5-VL-72B（+3.2 percentage points）。

### 消融研究

**双阶段质量控制效果**：
- 结果验证 + 理性审计 > 仅结果验证 > 无质量控制
- 理性审计有效过滤了"运气正确"的轨迹

**TAURA vs Vanilla GRPO**：
- TAURA 在全部 10 个 benchmark 上优于 vanilla GRPO
- 时间定位任务优势最明显

**SFT 必要性**：
- Agentic SFT > Vanilla SFT（无轨迹合成）> No SFT

### 效率分析

**帧数减少**：
- 平均每视频处理帧数：OmniAgent 203 vs Qwen2.5-VL-72B 768
- 减少 73%（565 帧）

**测试时扩展（Test-Time Scaling）**：
- VideoMME-Long：将最大轮数从 6 增至 52，性能提升 6.2 percentage points
- 饱和点：约 11.7 轮后性能不再显著提升
- 正向扩展：更多轮次带来持续性能提升，未出现退化

---

## Ch6: 代码实现详解

### 仓库结构

```
OmniAgent/
├── agent_system/          # OTA agent 基础设施
├── data/                  # 数据集配置与处理
├── demo/                  # 演示脚本
│   ├── inference_one.py      # 单样本推理入口
│   ├── batch_eval.sh         # 批量评估脚本
│   └── web_demo.sh           # Web demo
├── inference/             # 推理相关代码
├── recipe/                # 训练配置
│   └── sft_agent_final.yaml  # SFT 配置文件
└── verl/                  # GRPO/TAURA 实现
```

### OTA 系统核心组件

**推理入口（demo/inference_one.py）**：
```python
# 伪代码示例
def infer_single_video(question, video_path, max_turns=11):
    memory = {"question": question}
    for turn in range(max_turns):
        obs = agent.observe(memory)
        thought = agent.think(memory, obs)
        action = agent.act(thought)
        if action.type == "answer":
            return action.response
        else:
            result = execute_action(action, video_path)
            memory.update(obs, thought, action, result)
    return agent.generate_final_answer(memory)
```

**批量评估（demo/batch_eval.sh）**：
- 支持多 GPU 并行评估
- 自动处理不同数据集格式
- 输出结果为 JSON 格式，包含指标分解

### 数据格式

训练/评估数据使用 JSONL 格式：

```json
{
  "video": "/path/to/video.mp4",
  "question": "What is the final dish shown in the video?",
  "answer": "A",
  "type": "MCQ"
}
```

**answer 类型**：
- `MCQ`：多选题，单选字母（A/B/C/D）
- `TR`：时间定位，格式 `[[start_time, end_time]]`（如 `[[42.5, 47.8]]`）
- `FF`：自由格式文本
- `NUM/SIZE`：数值答案

### 依赖与环境

**Python 版本**：3.11+

**CUDA 版本**：12.6

**外部依赖**：
- ffmpeg（视频/音频处理）
- ms-swift（SFT 训练框架）
- verl（GRPO/TAURA 实现）

**硬件需求**：
- 推理：1×A100 80GB（单样本）
- 批量评估：8×A100 80GB（并行）
- 训练：64×A100 80GB+（RL）

### 训练配置

**SFT（recipe/sft_agent_final.yaml）**：
```yaml
model: Qwen2.5-Omni-7B
data: 58K trajectories from 5 datasets
epochs: 2
learning_rate: 1e-5
batch_size: 64
gradient_accumulation: 4
```

**RL（verl 框架）**：
- 算法：GRPO with TAURA
- Steps：150
- Group size：8
- Learning rate：1e-6

### Web Demo

运行 `demo/web_demo.sh` 启动交互式 Web 界面，支持：
- 上传本地视频
- 输入自然语言问题
- 实时显示 OTA 循环过程
- 展示中间观察和推理步骤

---

## Ch7: 局限性与未来方向

### 当前局限

**实时性未充分探索**：
- 当前实现未针对实时场景优化
- 每轮行动仍需等待模型推理完成
- 未来可探索流式处理与异步行动

**超长视频性能**：
- 当前评测最长视频约 2 小时
- 更长视频（>2h）的极限性能尚未验证
- 记忆容量与信息检索效率需进一步优化

**多语言泛化能力**：
- 当前训练数据主要为英文
- 中文、其他语言的理解能力有待验证
- 多语言场景下的文化理解是未来方向

**训练成本**：
- SFT 需 16×A100，RL 需 64×A100
- 对于资源受限团队，复现门槛较高
- 可探索参数高效微调方法降低成本

### 未来研究方向

**基础模型迁移性**：
- 当前仅在 Qwen2.5-Omni-7B 上验证
- 方法在其他基础模型（如 LLaMA、Gemini）上的泛化性未探索
- 可研究跨模型架构的 OTA 训练方法

**多智能体协作**：
- 当前为单智能体系统
- 多智能体协作（如分工处理不同模态）可能进一步提升效率
- 需设计智能体间的通信与协调机制

**与外部工具结合**：
- 当前行动仅限于视频内部感知
- 结合外部工具（如搜索引擎、知识库）可扩展问答能力
- 需设计工具调用 API 与安全策略

**理论分析**：
- POMDP 形式化提供了理论框架
- 可进一步研究最优策略的收敛性、样本复杂度等理论问题
- 探索 TAURA 的熵加权机制的理论保证

### 应用前景

**视频问答与检索**：
- 大规模视频库的高效问答
- 基于语义的视频片段检索

**视频摘要与监控**：
- 长视频自动摘要
- 监控视频事件检测与报警

**教育与分析**：
- 教学视频知识点提取
- 体育视频战术分析
