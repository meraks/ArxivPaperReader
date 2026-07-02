# Orca: The World is in Your Mind — 世界基础模型

## 论文信息

- **标题**: Orca: The World is in Your Mind
- **作者**: Yihao Wang, Yuheng Ji, Mingyu Cao, Yanqing Shen, Runze Xiao et al. (Orca Team, BAAI — 北京智源人工智能研究院)
- **发表形式**: arXiv Preprint, 2026年7月 (v2, 2026-06-30 提交)
- **链接**: [arXiv:2606.30534](https://arxiv.org/abs/2606.30534) | [项目主页](https://orca-wm.github.io/)
- **HuggingFace Daily Papers**: #1 热门论文 (2026-07-03)

## 核心要点

### 一句话总结
Orca 提出**Next-State-Prediction**（下一状态预测）建模范式，通过「无意识学习」（连续视频中的密集自然状态转换）和「有意识学习」（语言描述事件引导的稀疏语义状态转换）两种互补范式，学习统一的世界潜在空间（world latent space），并可通过轻量级解码器读出文本、图像和动作。

### 核心贡献
1. **新范式**: 从 Next-Token/Frame/Action-Prediction 转向 Next-State-Prediction，以世界潜在状态而非特定模态输出为建模核心
2. **双学习范式**: 无意识学习（Unconscious Learning）捕获密集自然状态转换 + 有意识学习（Conscious Learning）捕获稀疏语义状态转换
3. **大规模数据**: 构建 125K 小时视频数据 + 160M 事件标注 + 11.5M VQA 数据（当前仅用 1/10）
4. **三模态验证**: 固定 Orca backbone，仅训练轻量级解码器，在文本生成/图像预测/具身动作上均超越同等规模专用基线

## 1. 方法详解

### 1.1 宏观建模

Orca 将世界学习形式化为潜在世界状态建模（latent world-state modeling）：

$$S = f_\theta(X)$$

其中 $X = \{X_m\}_{m \in \mathcal{M}}$ 是多模态世界信号集合。状态演化建模为：

$$S_{t+\Delta} \sim p_\Theta(S_{t+\Delta} | S_t, z_t, c_t), \quad \Delta \in \mathbb{Z}_{\neq 0}$$

其中 $z_t$ 捕获不可见动力学（物理定律、对象属性等），$c_t$ 表示显式条件（指令等）。$\Delta > 0$ 预测未来状态，$\Delta < 0$ 回溯过去状态。

### 1.2 双学习范式

#### 无意识学习（Unconscious Learning）
- 从连续视频中学习自然、密集的状态转换，无需标注
- 目标：通过预测下一帧的潜在表示来学习自然演化
- 等价于 $c_t = \emptyset$，即 $S_{t+\Delta} \sim p_\Theta^u(S_{t+\Delta} | S_t, z_t)$

#### 有意识学习（Conscious Learning）
- 在语言指令约束下学习有意义、稀疏的状态转换
- 语言作为显式条件，指定事件 $c_t = e_{t+\Delta}$、任务意图或因果前提
- 包含两个子目标：
  - **事件条件状态转换**（Event-conditioned state transition）：根据指令预测目标事件下的潜在状态
  - **VQA 回答生成**（VQA response generation）：基于视频和问题生成文本回答

### 1.3 模型架构

Orca 采用 Encoder-Decoder 架构：

**Encoder**: 基于 Qwen3.5 VLM（4B 或 0.8B），接收视觉和语言信号，通过可学习 Query 向量实现两种状态转换预测：
- <Query 1> 用于无意识学习（预测下一帧潜在）
- <Query 2> 用于有意识学习（预测指令指定事件的潜在）
- 预测潜在通过两层 MLP 映射，与冻结视觉编码器提取的真实潜在进行匹配

**Decoder**: 冻结 Encoder，训练轻量级模态专用解码器：
- **文本读出**: 复用 VLM 的 LM Head，无需额外模块
- **图像读出**: MLP 适配器 + SD3.5 MMDiT (LoRA)，参数量 556.9M
- **动作读出**: DiT-based Action Expert（8 个 Transformer 块，流匹配损失）

### 1.4 损失函数

总体预训练损失：

$$\mathcal{L} = \lambda_{\text{obs}}\mathcal{L}_{\text{obs}} + \lambda_{\text{evt}}\mathcal{L}_{\text{evt}} + \lambda_{\text{vqa}}\mathcal{L}_{\text{vqa}}$$

权重配置：$\lambda_{\text{obs}}=0.1$, $\lambda_{\text{evt}}=0.5$, $\lambda_{\text{vqa}}=0.4$。

潜在空间匹配损失：

$$\ell_{\text{lat}}(\hat{v}^l, v^l) = 0.1\|\hat{v}^l - v^l\|_2^2 + 0.9\left(1 - \frac{\langle\hat{v}^l, v^l\rangle}{\|\hat{v}^l\|_2\|v^l\|_2}\right)$$

## 2. 实验与结果

### 2.1 数据规模与 Scaling

| 数据类型 | 规模 | 用途 |
|---------|------|------|
| 视频数据 | 125K 小时（当前使用 12.5K 小时） | 观察状态转换 + 事件条件状态转换 |
| 事件标注 | 160M | 事件条件状态转换 |
| VQA 数据 | 11.5M | VQA 回答生成 |

视频来源覆盖四种场景：自我中心交互、外部视角操作、无动作机器人执行、自然动力学。

### 2.2 文本生成结果

**基准**: MVBench, TemporalBench, 3DSRBench, SWITCH

| 模型 | 参数量(B) | MVBench↑ | TemporalBench↑ | 3DSRBench↑ | SWITCH↑ | Avg↑ |
|------|----------|----------|---------------|-----------|---------|------|
| **世界模型** | | | | | | |
| V-JEPA 2.1(+LLaMA3-8B) | 10 | 75.4 | 28.5 | / | / | / |
| Emu3 | 8 | 35.2 | 9.5 | 39.1 | 38.0 | 30.4 |
| Emu3.5 | 34 | 39.5 | 9.5 | 31.3 | 38.9 | 29.8 |
| **VLM (小规模)** | | | | | | |
| Qwen3.5 | 4 | 67.1 | 25.2 | 48.1 | 42.8 | 46.7 |
| DeepSeek-VL2 | 3 | 40.5 | 21.0 | 32.1 | 35.5 | 32.3 |
| Gemma 4 | 4 | 45.6 | 20.2 | 44.8 | 52.4 | 40.8 |
| **Orca** | **4** | **65.3** | **34.2** | **52.1** | **55.6** | **51.8** |

Orca-4B 以 51.8 的平均分超越所有同等规模 VLM，尤其在 TemporalBench（+9.0）和 SWITCH（+12.8）上优势显著。

**跨基准能力维度分析**:

| 能力维度 | Qwen3.5-4B | Orca-4B | 提升 |
|---------|-----------|---------|------|
| 状态转换 | 51.86 | 64.13 | +12.27% |
| 常识推理 | 57.76 | 62.95 | +5.19% |
| 空间关系 | 54.68 | 55.25 | +0.57% |
| 动态运动 | 57.03 | 65.55 | +8.52% |

### 2.3 图像预测结果

**基准**: PRICE-V0.1（真实世界交互状态预测）

| 模型 | 参数量(B) | Gemini 3.1 Pro↑ | GPT 5.4↑ | Doubao-Seed↑ | Gemma 4-31B↑ | Avg↑ |
|------|----------|-----------------|----------|-------------|-------------|------|
| OmniGen2 | 3+4 | 24.6 | 46.8 | 41.4 | 45.5 | 39.6±10.2 |
| FLUX.1-Kontext | 12 | 21.6 | 46.9 | 42.7 | 52.5 | 40.9±13.5 |
| FLUX.2 [klein] | 4+4 | 29.7 | 64.6 | 60.0 | 70.2 | 56.1±18.1 |
| **Orca** | **4+2** | **44.0** | **67.9** | **61.0** | **66.3** | **59.8±10.9** |

Orca 平均分 59.8 为最佳，且标准差 10.9 最低（稳定性最好）。

### 2.4 具身动作生成结果

5 个真实机器人操作任务（Take Book, Stacked Bowls, Pull Out Tissue, Stamp, Scoop Sugar），每种 200 条训练轨迹。

| 模型 | 环境OOD(Rule-based↑) | 对象OOD(Rule-based↑) | 总体(Rule-based↑) |
|------|---------------------|--------------------|------------------|
| V-JEPA 2.1 w/ AE | 15.2 | 18.8 | 17.0 |
| Qwen3.5 w/ AE | 12.4 | 8.6 | 10.5 |
| π0.5 | 27.6 | 31.2 | 29.4 |
| **Orca w/ AE** | **36.6** | **28.2** | **32.4** |

Orca 在环境 OOD 上大幅领先（36.6 vs π0.5 的 27.6），总体表现最佳（32.4）。关键指标：
- **DRR (Drawdown Recovery Ratio)**: Orca 30.3 vs π0.5 26.7 — 更好的错误恢复能力
- **FNS (Failure Near-Success)**: Orca 15.1 vs π0.5 15.3 — 失败前能推进到更后阶段
- **SR (Success Rate)**: Orca 6% vs π0.5 5%

### 2.5 消融实验

| 配置 | 文本生成 | 图像预测 | 动作生成 | 平均 |
|------|---------|---------|---------|------|
| Only $\mathcal{L}_{\text{obs}}$ | 48.4 | - | 10.2 | 29.3 |
| $\mathcal{L}_{\text{obs}} + \mathcal{L}_{\text{evt}}$ | - | 58.2 | 30.9 | 44.6 |
| $\mathcal{L}_{\text{obs}} + \mathcal{L}_{\text{vqa}}$ | 50.5 | - | 32.6 | 41.6 |
| $\mathcal{L}_{\text{evt}} + \mathcal{L}_{\text{vqa}}$ | 50.1 | 54.7 | 23.0 | 42.6 |
| **三者全用** | **51.8** | **59.8** | **32.4** | **48.0** |

消融结论：
1. 三者联合提供最均衡的下游性能
2. $\mathcal{L}_{\text{obs}}$（无意识学习）对动作生成至关重要
3. $\mathcal{L}_{\text{evt}}$（事件条件转换）是图像预测的关键
4. $\mathcal{L}_{\text{vqa}}$（VQA）保持语言接口并增强语义基础

### 2.6 Scaling 行为

- 随着预训练数据从 2K 小时增加到 10K 小时，总损失持续下降，未见收敛趋势
- 4B 模型在各模态读出的下游性能均随预训练数据量增加而提升
- **涌现能力**: 预训练中未使用任何动作标签，但动作读出性能仍随视频数据量增加而提升

### 2.7 基础设施

基于 FlagScale 框架 + FSDP2 分布式训练，优化管线实现：

| 优化 | Samples/Sec/GPU |
|------|----------------|
| StarVLA 基线 | 0.66 |
| FSDP2 基线 | 0.97 |
| + Chunked Cross-Entropy Loss | 1.35 |
| + Activation Recompute | 2.86 |
| + Forward/Backward Pre-fetching | **2.91** (4.4×加速) |

训练硬件：32 节点 × 256 H100 GPU。

## 3. 讨论与局限

### 主要优势
1. **范式创新**: Next-State-Prediction 统一了文本/图像/动作三种模态的学习目标
2. **双学习范式设计**: 无意识学习（被动观察）和有意识学习（主动理解）互补
3. **零样本迁移**: 预训练无需动作标签，但学到的时间动态知识可有效迁移到机器人操控
4. **良好的 Scaling**: 损失曲线持续下降，未出现饱和

### 局限性
1. **模态有限**: 当前仅覆盖视觉和语言，缺乏音频、触觉、力觉、光照等物理信号
2. **ViT 空间约束**: 状态预测在预训练 VLM 的冻结视觉编码器空间中进行，未学习原生世界状态表示
3. **模型规模限制**: 最大仅 4B，受资源限制。4B 模型已出现语言/图像/动作之间的权衡
4. **数据使用比例低**: 125K 小时视频仅用 1/10（12.5K 小时）
5. **短视距状态转换**: 事件标注主要描述分钟级局部转换，缺乏对小时/天级别的长期建模
6. **动作任务较简单**: 当前机器人任务仍然较短和简单

### 未来方向
1. 更丰富的模态输入（音频、触觉、力觉等）并统一到同一个状态空间
2. 原生世界状态建模 —— 从多源信号直接学习统一的潜在空间
3. 建立世界模型状态转换评估体系
4. 模型-数据-评估自我进化闭环
5. 扩展到 AI for Science、量子力学、宏观宇宙等领域

## 4. 对社区的影响与启发

Orca 作为 BAAI 发布的世界基础模型早期探索，其核心价值在于：

1. **统一框架**: 提供了一个将 Next-Token/Frame/Action-Prediction 统一到 Next-State-Prediction 下的理论框架
2. **双学习范式**: 无意识学习与有意识学习的区分具有认知科学启发性
3. **涌现能力证明**: 仅通过视频预训练就能提升动作生成能力，表明世界模型可以通过观察学习具身技能
4. **开源生态友好**: 基于 Qwen3.5、SD3.5、FlagScale 等开源组件构建

**开源/代码可用性**: 项目主页 [orca-wm.github.io](https://orca-wm.github.io/) 已上线，论文提及使用 FlagScale 和多项开源组件，但完整权重尚未明确发布。

## 参考文献

- Mur-Labadia et al. V-JEPA 2.1, arXiv 2026
- Wang et al. Emu3 (Multimodal Learning with Next-Token Prediction), Nature 2026
- Qwen Team. Qwen3.5, 2026
- DeepMind. Gemma 4, 2026
- Physical Intelligence. π0.5, arXiv 2025
- Stability AI. Stable Diffusion 3.5, 2024
- FlagOS. FlagScale, 2026
- Ji et al. PRM-as-a-Judge, arXiv 2026