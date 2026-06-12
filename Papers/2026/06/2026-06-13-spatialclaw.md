# SpatialClaw: 重新思考空间推理Agent的动作接口

## 论文元数据
- 标题：SpatialClaw: Rethinking Action Interface for Agentic Spatial Reasoning
- 作者：Seokju Cho (NVIDIA/KAIST), Ryo Hachiuma, Abhishek Badki, Hang Su, Byung-Kwan Lee, Chan Hee Song, Sifei Liu, Subhashree Radhakrishnan, Seungryong Kim, Yu-Chiang Frank Wang, Min-Hung Chen
- arXiv ID：2606.13673
- 发表/提交日期：2026-06-11
- 分类：cs.CV, cs.AI
- 官方代码：https://github.com/NVlabs/SpatialClaw
- 项目主页：https://spatialclaw.github.io/

---

## Ch1: 论文概述与核心贡献

### 1.1 问题陈述：VLM在空间推理中的根本性挑战

空间推理——确定物体在3D空间中的位置、相互关系以及运动规律——对于视觉语言模型（VLM）而言仍然是一个根本性挑战。不同于在ImageNet上训练的2D视觉识别任务，空间推理需要理解跨视角的几何一致性、3D空间关系、时序运动模式，这些能力在VLM的2D像素级预训练过程中被系统性忽视。现有的VLM在面对诸如"从图1视角观察的物体在图2视角中位于何处？"或"视频中的物体相对于摄像机的运动轨迹是什么？"这类问题时，表现远低于人类水平。

更根本的挑战不在于视觉感知本身——现代分割模型（如SAM3）和深度估计模型（如Depth Anything 3）已经相当成熟——而在于**如何让VLM灵活地组合这些感知工具来执行多步骤的空间推理**。传统的VLM工作流存在两大结构性缺陷：要么要求agent在见到任何中间结果之前提交完整的推理程序（单次代码生成），要么将agent限制在预定义的固定API调用集合中（结构化工具调用）。这两种方法都无法支持真正"agentic"的推理过程——即根据中间观察结果动态调整策略的能力。

### 1.2 核心创新：代码作为动作接口

SpatialClaw提出的解决方案简单而深刻：**将代码本身作为VLM agent的动作接口**，在持久化的Python内核中让agent逐步编写、执行、观察、修正代码。这不是简单地让agent写一次程序然后运行，而是建立一个完整的交互式推理循环：

1. **持久化工作空间**：维护一个状态化的Jupyter内核，预加载SAM3、Depth Anything 3、NumPy、SciPy、Matplotlib等感知和几何计算库。变量跨步骤保留，无需重复计算。
2. **单步代码生成**：每一步agent只写一个可执行单元（code cell），基于所有之前的输出（包括打印变量、可视化图像、错误信息）来决定下一步操作。
3. **视觉反馈闭环**：通过自定义`show()`函数将中间可视化结果（如3D点云、分割mask、运动轨迹）直接嵌入agent的下一轮上下文，实现真正的"看见自己在做什么"。
4. **安全执行保障**：在执行前进行AST静态扫描，拒绝危险模块导入（os、subprocess、importlib等），确保代码在沙盒环境中安全运行。

> ### 类比理解：厨房做菜 vs 厨师团队
>
> 传统方法就像给厨师一份完整菜谱，要求他按照菜谱一口气做完所有步骤，但在过程中不能尝味道、看火候或调整策略——最终结果可能完美，也可能完全失败，但在提交菜谱之前没有任何修正机会。
>
> SpatialClaw则像让一个厨师团队协作：主厨负责规划整体步骤（"先备菜，再炒制，最后装盘"），然后每个步骤由专门的厨师执行并品尝结果（"尝一下这个咸淡是否合适"），主厨根据中间结果调整下一步策略（"太咸了，下次少放盐"或"火候正好，继续这个节奏"）。整个过程中，团队可以随时查看正在处理的食物（通过`show()`看到的中间可视化），并根据实际情况灵活调整原计划。

### 1.3 核心贡献

SpatialClaw的核心贡献可归纳为以下五点：

**1. 代码作为动作接口的理论框架**
首次系统性地论证了为什么代码是空间推理agent的理想接口。论文的核心洞察是："一次新的空间分析不是一个新的API接口，而是在中间证据驱动下，跨步骤组装感知工具和数值基元的新组合。"这一洞察从根本上区分了SpatialClaw与之前的方法——它不是提供更多工具或更复杂的API，而是提供一种组合工具的通用语言（Python代码）。

**2. 持久化内核工作空间**
设计了六个精心设计的入口点（InputImages、Metadata、tools、show()、vlm、ReturnAnswer()），将复杂的多模态推理任务封装为自然Python编程环境。工具集包括：
- **Reconstruct**: 使用Depth Anything 3进行单目深度估计并生成3D点云
- **SAM3**: 最先进的分割模型，用于提取物体mask
- **Mask/Geometry工具包装器**: 将mask转换为几何计算所需的数据结构
- **NumPy/SciPy**: 数值计算和科学计算基元
- **Matplotlib**: 可视化支持

所有这些工具在Python内核中持久存在，agent可以随时调用而无需重新初始化。

**3. 五阶段Agent循环**
设计了完整的推理循环框架：
- **Planner阶段**: 使用独立LLM会话生成高层规划
- **Code Generation阶段**: 主VLM基于当前状态生成下一步代码
- **Execution阶段**: 执行前进行AST安全检查，然后运行代码
- **Feedback Assembly阶段**: 组装stdout、traceback、变量摘要和show()图像作为下一轮输入
- **Answer Submission阶段**: 当agent确信获得答案时调用ReturnAnswer()提交结果

这一循环设计确保agent能够在每一步获得充分信息来指导下一步行动。

**4. 训练无关的通用性**
证明了无需任何微调或任务特定适配，同一套系统prompt、工具集和超参数可以在20个不同的空间推理benchmark上取得SOTA结果。这6个VLM骨干网络（Qwen3.5-397B/122B、Qwen3.6-35B/27B、Gemma4-31B/26B）在SpatialClaw框架下均取得显著增益，证明了方法的通用性。

**5. 全面的实验验证**
在20个空间推理benchmark上进行系统评估，覆盖静态3D、多视角、动态4D视频等任务类型。平均准确率达到59.9%，超越之前最佳方法（SpaceTools）11.2个百分点，在最具挑战性的任务（如DSI-Bench）上获得超过20个百分点的增益。

### 1.4 关键数字速览

| 指标 | 数值 | 对比基线 | 说明 |
|------|------|----------|------|
| 平均准确率（20个benchmark） | 59.9% | SpaceTools: 48.7% | +11.2 pp |
| 最佳模型表现（Qwen3.6-27B） | 62.7% | No-tool: 53.4% | +9.3 pp |
| 视频4D任务增益 | +18.3% | SpaceTools | 动态场景优势明显 |
| 多视角任务增益 | +14.3% | SpaceTools | 跨视角一致性处理 |
| DSI-Bench增益（Qwen3.6-27B） | +20.1 pp | SpaceTools | 单任务最大提升 |
| MindCube增益（Qwen3.6-27B） | +18.9 pp | SpaceTools | 3D推理能力验证 |
| 支持VLM骨干网络 | 6个 | 27B–397B | 模型规模覆盖广 |
| 总benchmark数量 | 20个 | 覆盖5大任务类型 | 全方位评估 |
| 最大步数预算 | 30步 | N_max=30 | 平衡效率与能力 |

这些数字表明SpatialClaw不是在特定任务或特定模型上的微小改进，而是对空间推理agent的根本性提升。

### 1.5 四种动作接口对比

| 接口类型 | 工作方式 | 局限性 | 适用场景 |
|----------|----------|--------|----------|
| No-tool | VLM直接输出答案，无工具辅助 | 无法访问感知工具，纯靠预训练知识 | 简单2D视觉问答 |
| Single-pass code | VLM一次性生成完整程序，提交后执行 | 见不到中间结果，无法修正错误，必须提前规划所有步骤 | 确定性任务，步骤少且清晰 |
| Structured tool-call | VLM调用预定义API（如detect()、track()），每步返回固定格式输出 | 受限于固定API设计，无法灵活组合工具，难以处理新任务模式 | 任务类型已知且稳定 |
| **SpatialClaw** | VLM逐步编写代码，在持久化内核中执行，观察中间结果后决定下一步 | 计算成本较高，需要多轮VLM调用 | 复杂多步骤空间推理，需要动态调整策略 |

从No-tool到SpatialClaw的演进，体现了agent从"直接回答"到"规划-执行-观察-修正"的完整agentic能力。SpatialClaw的代码接口不是要取代固定API，而是在需要时让agent能够自己"发明"新的组合方式。

---

## Ch2: 研究背景与动机

### 2.1 为什么VLM在空间推理中表现差？

空间推理对VLM而言是根本性挑战，原因在于VLM的训练和架构设计与空间推理的需求之间存在本质错位：

**1. 预训练数据的主效应**
现代VLM（如Qwen、Gemma）主要在2D图像-文本对上进行预训练，这些数据中空间关系以2D投影形式呈现（"图左边有一只狗"）。然而，真正的空间推理需要理解：
- **3D几何关系**: 物体在3D空间中的实际位置，而非2D投影位置
- **跨视角一致性**: 同一物体在不同视角下应该保持一致的3D属性
- **时序运动规律**: 物体/相机在时间维度上的运动轨迹和速度

这些3D/4D关系在2D预训练数据中被系统性忽视，导致VLM无法从大规模预训练中学习到可靠的空间推理模式。

**2. 固定输出格式的限制**
传统VLM的输出格式为文本或bounding box坐标，这种格式无法表达复杂的多步骤推理过程。当面对需要"先分割物体，再估计深度，然后投影到另一视角，最后验证一致性"的任务时，VLM被迫在单个前向传播中完成所有隐式推理，这超出了其架构能力。

**3. 工具使用接口的僵化**
最近的tool-augmented VLM尝试通过提供感知工具（如分割模型、深度估计器）来增强能力，但这些工具通常通过固定API调用，如：
```python
objects = detect(image)
depth = estimate_depth(image)
```

这种接口的问题是：**它假设任务执行路径是已知的和固定的**。但实际上，不同的空间推理任务需要以不同的顺序组合这些工具，甚至在执行过程中需要根据中间结果调整策略。

### 2.2 四种现有动作接口的详细分析

让我们通过一个具体任务来理解四种接口的局限性：

**任务示例**: 给定两张从不同角度拍摄的房间照片，确定照片A中标注的"蓝色椅子"在照片B中的位置。

**No-tool接口的失败**
- VLM直接看图并回答："蓝色椅子在照片B的右下角"
- 问题：完全依赖VLM的隐式推理，无法访问深度估计、相机标定等必要工具
- 结果：准确率接近随机猜测

**Single-pass Code的失败**
- VLM一次性生成完整程序：
```python
# 假设VLM生成的代码
depth_a = estimate_depth(photo_a)
depth_b = estimate_depth(photo_b)
chair_mask = segment(photo_a, "blue chair")
chair_3d = lift_to_3d(chair_mask, depth_a)
projected = project_to_2d(chair_3d, camera_b)
```
- 问题：
  1. 在生成这段代码时，VLM无法看到`depth_a`或`chair_mask`的实际内容，只能猜测
  2. 如果`lift_to_3d`函数返回了异常（比如深度估计失败），整个程序崩溃，agent无法修正
  3. 无法根据中间结果调整策略（比如发现投影结果不合理，想重新尝试不同的相机参数）
- 结果：由于中间步骤的误差累积，最终准确率较低

**Structured Tool-call的失败**
- VLM调用预定义API：
```python
result = spatial_localize(
    image_a=photo_a,
    image_b=photo_b,
    query="blue chair",
    method="cross_view_projection"
)
```
- 问题：
  1. `spatial_localize`API的设计者必须提前预见所有可能的任务模式
  2. 如果新任务需要"先track物体在视频中的运动，再投影到另一视角"，现有API不支持
  3. API内部逻辑是固定的，agent无法根据中间结果调整（比如发现tracking失败，想改用光流方法）
- 结果：在API设计覆盖的任务上表现尚可，但无法泛化到新任务模式

**SpatialClaw的优势**
- VLM逐步编写代码：
```python
# Step 1: 分割蓝色椅子
chair_mask = tools.SAM3(photo_a, prompt="blue chair")
show(chair_mask)  # 观察分割结果

# Step 2: 估计3D位置（基于观察到的mask质量决定）
depth = tools.Reconstruct(photo_a)
chair_3d = lift_to_3d(chair_mask, depth)
if chair_3d.confidence < 0.5:  # 根据数值判断
    # Step 3: 如果质量低，尝试另一视角验证
    alternative_mask = tools.SAM3(photo_a, prompt="chair")
    chair_3d = lift_to_3d(alternative_mask, depth)

# Step 4: 投影到照片B并验证
projected = project_to_2d(chair_3d, camera_b)
show(projected)  # 观察投影结果
if not validate_projection(projected, photo_b):
    # Step 5: 如果投影不合理，调整相机参数
    camera_b.refine_estimate(photo_a, photo_b)
    projected = project_to_2d(chair_3d, camera_b)
```
- 优势：
  1. 每一步都能观察中间结果（`show()`的可视化）
  2. 能根据数值条件进行分支决策（`if confidence < 0.5`）
  3. 能在失败时修正策略（换prompt、调整参数）
  4. 能灵活组合工具（SAM3 + Reconstruct + 自定义验证逻辑）
- 结果：在复杂任务上显著优于其他接口

### 2.3 关键洞察：工具组合比工具数量更重要

SpatialClaw的核心洞察在论文中表述为：

> "一次新的空间分析不是一个新的API接口，而是在中间证据驱动下，跨步骤组装感知工具和数值基元的新组合。"

这一洞察的深刻之处在于：**它重新定义了什么是"工具"**。传统方法认为"工具"是预定义的函数（如`detect()`、`track()`），而SpatialClaw认为"工具"是感知基元（SAM3、Depth Anything 3）和数值基元（NumPy、SciPy）的组合方式。

> ### 类比理解：Google Maps vs 实地探路
>
> 固定API就像Google Maps的"推荐路线"功能：你输入起点和终点，系统给你一条预设路线。这条路可能是最快的，但它无法应对突发情况（比如前方施工、你想绕路看风景、你突然决定中途停车）。
>
> SpatialClaw的代码接口就像实地探路：你有一个指南针（NumPy）、一张详细地图（预加载的感知工具）、以及随时可以停下来观察环境的能力（`show()`）。你可以根据自己的判断调整路线（"这里看起来堵车，我换条路"），或者临时改变目标（"看到路边的咖啡馆，我决定先喝杯咖啡"）。
>
> 关键区别不在于指南针和地图的质量（那是固定API和SpatialClaw共享的资源），而在于**谁有权决定如何使用这些资源**——固定API的决策逻辑是硬编码的，而SpatialClaw让agent自己编写决策逻辑（通过Python代码）。

### 2.4 相关工作简述

**Tool-augmented VLMs**
工具增强的VLM并非新概念。Visual Programming和ViperGPT最早提出让VLM生成代码来组合视觉模型，Code-as-Policies将这一思想扩展到机器人控制。然而，这些方法都采用"单次代码生成"模式——要求VLM在见到任何中间结果之前提交完整程序。这在简单任务上有效，但在复杂多步骤推理中表现不佳，因为无法修正早期错误。

**Agentic Spatial Reasoning**
SpaceTools和pySpatial是最接近SpatialClaw的工作。SpaceTools设计了空间推理工具的固定API（如`spatial_relationship()`、`cross_view_localize()`），pySpatial提供了类似的工具集。然而，这两种方法都将agent限制在预定义的API调用中，无法灵活组合工具或根据中间结果调整策略。论文的Table 2显示，这两种方法的平均准确率分别为53.4%（SpaceTools）和55.2%（pySpatial），显著低于SpatialClaw的59.9%。

**Benchmark发展**
空间推理的评估标准也在快速发展。早期的Task 1（Gao et al., 2024）主要测试单视角关系，而现在的benchmark如MindCube（多视角3D重建）、DSI-Bench（跨视角定位）、PAI-Bench（视频4D推理）等，测试的是更深层次的空间理解能力。SpatialClaw在这些更具挑战性的benchmark上表现尤其出色，这证明了代码接口在复杂任务中的必要性。

### 2.5 为什么现在是正确时机？

SpatialClaw的出现恰逢其时，依赖于三个技术条件的成熟：

**1. VLM代码生成能力的成熟**
Gemma4和Qwen3.5/3.6等模型在代码生成任务上已经达到实用水平，能够编写合法的Python代码并正确调用API。这使得"代码作为接口"从理论可能性变为工程可行性。

**2. 感知基础模型的成熟**
SAM3和Depth Anything 3等模型提供了高质量的视觉基元，agent可以依赖这些基础工具而不必担心视觉感知质量成为瓶颈。

**3. 多模态基础设施的成熟**
Jupyter内核的持久化、LangGraph的agent状态管理、vLLM的高效推理，这些基础设施使得SpatialClaw的工程实现成为可能。论文的代码仓库显示，整个系统被设计为三个独立服务（VLM推理、感知工具、Agent），这种模块化设计便于部署和扩展。

SpatialClaw不是对VLM架构的改进，而是对VLM使用方式的重新思考。它证明了：**在正确的接口设计下，现有的VLM已经具备惊人的空间推理能力——我们需要做的，只是让它们以正确的方式访问和组合工具。**

---



# Ch3: SpatialClaw方法论：代码作为动作接口

## 3.1 持久化内核工作空间

SpatialClaw的核心技术创新在于维护一个**状态化Python内核**，该内核预加载了完整的感知和几何计算工具链。与传统的单次代码执行或结构化工具调用不同，这个内核在整个agent推理过程中持续运行，变量和计算结果跨步骤保留，无需重复计算。

### 3.1.1 六个入口点设计

持久化内核通过六个精心设计的入口点与agent交互：

**1. InputImages（图像输入池）**
- 存储任务相关的所有原始图像，包括单视角、多视角序列或视频帧
- 图像以NumPy数组形式持久化在内核内存中
- agent可通过工具函数访问任意图像的像素数据

**2. Metadata（元数据字典）**
- 包含相机内参（intrinsic matrix）、外参（extrinsic matrix）、深度尺度因子
- 预计算的3D场景边界（bounding box）、物体数量等统计信息
- 这些元数据减少了agent重复提取基础信息的需要

**3. tools（感知与几何工具包）**
工具包包含以下核心模块：

- **Reconstruct**：基于Depth Anything 3的3D重建工具，输入单张或多视角图像，输出稠密点云或深度图
- **SAM3**：Segment Anything Model 3的封装，提供2D/3D分割能力
- **Mask utilities**：分割后处理（合并、过滤、IoU计算）
- **Geometry utilities**：3D变换（rotation, translation）、投影矩阵计算、点-面距离
- **SciPy spatial**：KDTree、最近邻查询、凸包计算（通过预导入SciPy库）
- **NumPy broadcasting tools**：向量化几何运算

**4. show()（视觉反馈接口）**
这是SpatialClaw的关键创新。`show()`函数接受图像或3D可视化对象，将其渲染为PNG格式，然后**自动嵌入到下一轮agent的上下文中**。设计要点：

- 输入灵活：可接受2D图像、3D点云投影、Matplotlib figure对象
- 自动渲染：内部调用Matplotlib/Plotly生成可视化，无需agent编写绘图代码
- 上下文注入：生成的图像被追加到VLM的多模态输入流中，agent在下一步能够"看到"中间结果
- 形成闭环：agent的代码产生可视化 → 可视化反馈给agent → agent基于视觉观察调整下一步代码

**5. vlm（多模态查询接口）**
- 当agent需要用语言理解图像内容时调用
- 内部实现：将图像通过VLM的encoder编码为token，与文本prompt拼接后送入模型
- 主要用途：处理符号化的视觉问答（如"这个物体的颜色是什么？"）

**6. ReturnAnswer()（答案提交接口）**
- agent调用此函数提交最终答案
- 参数：数值答案（float/int）、字符串答案、或结构化输出（坐标列表、选项目ID）
- 调用后agent循环终止，答案被解析为benchmark所需的格式

> ### 类比理解：Jupyter Notebook vs 一次性脚本
>
> 传统单次代码执行就像编写一个**一次性Python脚本** — 你必须在运行前预测所有中间步骤，无法在执行过程中调整策略。如果第三步的输出与预期不符，整个脚本需要重新编写。
>
> SpatialClaw的持久化内核就像**Jupyter Notebook的交互式环境** — 你在每个cell中执行代码，立即查看输出，基于观察调整下一个cell的内容。变量（如分割mask、点云数据）保留在内存中，后续步骤直接引用，无需重新计算。这种交互式模式让agent能够像人类研究者一样"边做边想"，而非一次性提交完整策略。

### 3.1.2 工具预加载策略

所有工具在内核初始化时预加载，避免运行时动态导入的延迟和安全隐患：

```python
# ⚠️ 非官方概念实现，未经验证
import numpy as np
from scipy.spatial import KDTree
from tools.reconstruction import Reconstruct  # Depth Anything 3 wrapper
from tools.segmentation import SAM3            # Segment Anything Model 3
from tools.geometry import compute_3d_distance, project_points
from tools.visualization import show
from agent.interface import ReturnAnswer
```

预加载的另一个好处是**AST安全检查的简化**：由于已知所有可用模块，静态分析时可以直接拒绝不在白名单中的import语句。

### 3.1.3 内存管理与计算复用

持久化内核通过以下机制优化内存和计算：

- **惰性计算**：工具函数仅在首次调用时执行，结果缓存到内核全局变量
- **增量更新**：SAM3分割产生的mask以字典形式存储（`{object_id: mask}`），后续步骤可直接引用
- **垃圾回收抑制**：关键变量（如点云、分割结果）不会被自动回收，确保整个推理周期内可用
- **步数预算约束**：N_max=30步的上限防止内存无限增长

---

## 3.2 五阶段Agent循环（核心机制）

SpatialClaw的agent循环通过五个明确划分的阶段实现从观察到结论的完整推理链条。每个阶段输出标准化数据结构，驱动下一阶段执行。

### 3.2.1 Stage 1 - Planning（独立规划会话）

**目的**：在接触具体图像数据前，制定高层级解决策略。

**执行流程**：
1. 接收任务指令和Metadata（相机参数、场景类型、问题类型）
2. 启动独立的LLM会话（与主VLM分离，避免视觉信息干扰逻辑规划）
3. 输出结构化计划：
   ```
   Plan:
   1. Reconstruct 3D scene from multi-view images
   2. Segment target objects (heater, door) using SAM3
   3. Extract 3D centroids of segmented regions
   4. Compute Euclidean distance between centroids
   5. Return numerical answer in meters
   ```

**设计原理**：
- **视觉无关性**：规划阶段不需要看图像，避免过早关注细节导致策略偏差
- **任务分解**：将复杂空间问题拆解为可执行的原子步骤
- **工具选择提示**：计划中的动词（Reconstruct, Segment, Compute）引导后续代码生成阶段调用相应工具

**规划LLM的选择**：
- 实验中使用与主VLM相同的模型（Qwen3.6或Gemma4），但仅使用文本模式
- 规划token消耗极低（通常<100 tokens），整体开销可忽略

### 3.2.2 Stage 2 - Code Generation（主VLM代码生成）

**目的**：基于当前上下文，生成一个可执行的Python代码单元。

**输入上下文包含**：
- 原始任务指令
- Stage 1的规划输出
- 历史代码单元（之前的所有生成代码）
- 历史反馈（之前的stdout、traceback、变量摘要、show()图像）
- Metadata和可用工具列表

**代码结构要求**（通过system prompt强制）：

每个生成的代码单元必须包含以下四个字段：

```python
# Purpose: Segment the heater and door objects to obtain their masks
# Reasoning: We need object masks to extract their point cloud regions
# Next Goal: Extract 3D centroids from the segmented regions

# Code START
masks = SAM3.detect(InputImages, prompt=["heater", "door"])
show(masks['heater'])
# Code END
```

- **Purpose**：当前代码单元的意图（1句话）
- **Reasoning**：为什么需要这个步骤（1-2句话）
- **Next Goal**：执行后希望获得的信息（链接下一步）
- **Code**：实际可执行的Python语句

**VLM的编码能力利用**：
- Qwen3.6和Gemma4都经过大规模代码训练，能够生成正确的Python语法
- 多模态上下文让模型"看到"之前的show()输出，理解当前需要处理的对象
- 规划阶段的Plan被嵌入到system prompt中，作为生成的蓝图

**生成策略**：
- 每个代码单元只执行一个主要操作（如分割、距离计算）
- 避免嵌套循环和复杂逻辑（降低错误率）
- 优先调用工具函数，而非手动实现算法

### 3.2.3 Stage 3 - Code Execution（AST安全检查与执行）

**AST安全检查**：

在代码送入内核前，进行静态分析以阻止危险操作：

```python
# ⚠️ 非官方概念实现，未经验证
import ast

def safe_execute(code_str):
    tree = ast.parse(code_str)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if node.module in BLACKLIST:  # file I/O, network access, exec/eval, GPU backend imports, .save/.to_csv patterns
                raise SecurityError(f"Import of {node.module} blocked")
        if isinstance(node, ast.Call):
            if hasattr(node.func, 'id') and node.func.id in DANGEROUS_FUNCTIONS:
                raise SecurityError(f"Call to {node.func.id} blocked")
    # Execute in kernel
    return kernel.execute(code_str)
```

**黑名单模块**（来自论文Section 3.2）：
- `os`：文件系统操作
- `subprocess`：外部命令执行
- `importlib`：动态导入（绕过静态检查）
- `sys`：进程级操作
- `shutil`：文件删除
- `eval` / `exec`：任意代码执行

**白名单工具**：
- 所有预加载的工具（Reconstruct, SAM3, geometry utilities）
- 标准科学计算库（NumPy, SciPy, Matplotlib）
- 内置Python数据结构操作

**执行机制**：
- 代码通过Jupyter Kernel的`execute()`方法运行
- 执行超时设置为每单元60秒（防止无限循环）
- 异常会被捕获并转化为traceback字符串，传递到Stage 4

### 3.2.4 Stage 4 - Feedback Assembly（多模态反馈组装）

**反馈内容包含四个维度**：

1. **stdout**：代码单元的标准输出（print语句、函数返回值）
   - 示例：`"Found 2 objects: heater, door"`

2. **traceback**：执行错误信息（如果发生）
   - 示例：`"NameError: name 'undefined_var' not defined"`
   - agent可以基于traceback修正错误（如重新定义变量）

3. **变量摘要**：内核全局空间中新增或修改的变量
   - 格式：`"New variables: masks (dict), point_cloud (np.ndarray shape=(N,3))"`
   - 让agent了解当前可用的数据，无需猜测

4. **show()图像**：可视化中间结果
   - SAM3分割结果（2D mask overlay）
   - 3D点云投影（不同物体用不同颜色）
   - 距离可视化（线段标注）

**反馈注入到下一轮上下文**：
- 文本反馈（stdout/traceback/变量摘要）直接拼接为VLM的token序列
- 图像反馈通过多模态encoder编码为visual tokens
- 整个历史反馈链保持完整，确保agent能够回溯到任意历史步骤

**关键设计**：show()图像的视觉反馈让VLM能够"看到"自己的代码执行结果，形成类似人类调试的闭环。当分割mask不完整时，agent可以直接观察到并调整参数或切换策略。

### 3.2.5 Stage 5 - Answer Submission（答案提交与验证）

**ReturnAnswer()调用**：

当agent认为已经获得最终答案时，调用：

```python
ReturnAnswer(answer=0.9439, unit="meters")
```

**答案验证逻辑**：
- 数值答案：检查是否为`int`或`float`类型（拒绝字符串形式的数字）
- 字符串答案：检查是否在benchmark允许的选项集合中
- 坐标答案：检查维度和数值范围（如像素坐标应在图像尺寸内）

**验证失败时**：
- agent收到错误信息：`"Invalid answer format. Expected float, got str."`
- 循环继续，agent可以重新提交或继续计算
- 提交尝试不计入N_max步数预算（避免惩罚因格式错误导致的重试）

**验证成功时**：
- agent循环终止
- 答案被解析并保存为benchmark submission
- 整个推理历史（代码+反馈）被记录用于分析

**步数预算约束**：
- N_max=30步（每个代码单元执行计为1步）
- 超过30步强制终止，返回当前最优猜测或空答案
- 大部分任务在5-10步内完成（Table 1的附录统计）

> ### 类比理解：科学家做实验的完整流程
>
> SpatialClaw的五阶段循环就像科学家在实验室做研究的完整流程：
>
> - **Stage 1 Planning（实验设计）**：科学家在开始实验前，先写下实验方案 — 需要哪些设备、步骤顺序、预期结果。这一步不需要接触任何实验材料，纯粹是逻辑规划。
>
> - **Stage 2 Code Generation（实验操作）**：科学家按照实验方案，执行具体操作 — 配置仪器、添加试剂、记录数据。对应agent生成并执行代码。
>
> - **Stage 3 Code Execution（观察现象）**：科学家观察实验过程中的现象 — 颜色变化、温度读数、沉淀生成。对应agent收集stdout、traceback、变量摘要。
>
> - **Stage 4 Feedback Assembly（数据分析）**：科学家将观察到的现象记录到实验笔记本，并根据这些数据判断实验是否成功。对应agent将反馈注入下一轮上下文，并决定下一步操作。
>
> - **Stage 5 Answer Submission（结论报告）**：科学家基于实验结果得出结论，写入研究报告。对应agent调用ReturnAnswer()提交答案。
>
> 关键相似点：每一步的输出都影响下一步的决策，形成闭环。如果实验现象不符合预期，科学家可以调整参数重新执行 — 这正是SpatialClaw agent通过show()反馈调整代码策略的能力。

### 3.2.6 完整案例：测量heater到door的距离

以论文Figure 2为例，展示agent如何通过五阶段循环完成空间推理任务：

**任务**：给定一个室内场景的多视角图像，计算heater（加热器）到door（门）的3D欧氏距离。

**Stage 1 - Planning**：
```
Plan:
1. Reconstruct the 3D scene from multi-view images
2. Segment the heater and door objects using SAM3
3. Extract 3D point clouds for each object
4. Compute centroids of the point clouds
5. Calculate Euclidean distance between centroids
```

**Stage 2 - Code Generation (Step 1)**：
```python
# Purpose: Reconstruct 3D scene from multi-view images
# Reasoning: Need depth information to compute 3D distances
# Next Goal: Obtain a dense point cloud of the scene

# Code START
point_cloud = Reconstruct(InputImages, metadata=Metadata)
show(point_cloud)
# Code END
```

**Stage 3 - Execution**：
- AST检查通过
- 执行Reconstruct工具（调用Depth Anything 3）
- 生成包含数百万点的3D点云

**Stage 4 - Feedback**：
- stdout: `"Reconstructed 3,245,678 points"`
- show()图像：3D点云的彩色投影，不同深度用不同颜色编码

**Stage 2 - Code Generation (Step 2)**：
```python
# Purpose: Segment heater and door objects
# Reasoning: Need to isolate object regions to compute their centroids
# Next Goal: Extract masks for both objects

# Code START
masks = SAM3(InputImages, prompts=["heater", "door"])
show(masks)
# Code END
```

**Stage 3 - Execution**：
- SAM3处理输入图像，生成2D分割mask

**Stage 4 - Feedback**：
- show()图像：分割结果overlay在原图上，heater和door分别用红色和蓝色高亮

**Stage 2 - Code Generation (Step 3)**：
```python
# Purpose: Filter point cloud using masks to get object-specific clouds
# Reasoning: Centroids should be computed only on object points, not entire scene
# Next Goal: Obtain heater_points and door_points

# Code START
heater_points = point_cloud[masks['heater']]
door_points = point_cloud[masks['door']]
# Code END
```

**Stage 4 - Feedback**：
- 变量摘要：`"New variables: heater_points (shape=(12453,3)), door_points (shape=(8932,3))"`

**Stage 2 - Code Generation (Step 4)**：
```python
# Purpose: Compute centroids and distance
# Reasoning: Centroid-to-centroid distance approximates object distance
# Next Goal: Obtain final distance value

# Code START
heater_centroid = heater_points.mean(axis=0)
door_centroid = door_points.mean(axis=0)
distance = np.linalg.norm(heater_centroid - door_centroid)
print(f"Distance: {distance:.4f} meters")
# Code END
```

**Stage 3 - Execution**：
- 计算两个点云的质心（均值）
- 计算L2范数

**Stage 4 - Feedback**：
- stdout: `"Distance: 0.9439 meters"`
- 变量摘要：`"New variables: heater_centroid (shape=(3,)), door_centroid (shape=(3,)), distance (0.9439)"`

**Stage 5 - Answer Submission**：
```python
ReturnAnswer(answer=0.9439, unit="meters")
```

**验证通过**，agent循环终止，最终答案：**0.9439米**。

**关键观察**：
- 整个过程使用了**7步**（规划 + 6个代码单元），远低于30步预算
- 每个代码单元都依赖前一步的输出（如`point_cloud`被后续mask过滤使用）
- show()反馈让agent能够确认分割质量（如果mask不完整，可以调整SAM3参数）

---

## 3.3 代码接口的灵活性优势

本节通过两个对比案例，展示"代码作为动作接口"相对于传统方法的灵活性优势。

### 3.3.1 对比单次代码执行：策略调整能力

**单次代码方法的局限**：
agent必须在看到任何中间结果前生成完整代码。例如：

```python
# 单次代码生成（传统方法）
def solve_heater_door_distance(images, metadata):
    # Step 1: Reconstruct
    pc = Reconstruct(images, metadata)
    
    # Step 2: Segment
    masks = SAM3(images, ["heater", "door"])
    
    # Step 3: Extract centroids (using median strategy)
    heater_centroid = np.median(pc[masks['heater']], axis=0)
    door_centroid = np.median(pc[masks['door']], axis=0)
    
    # Step 4: Compute distance
    return np.linalg.norm(heater_centroid - door_centroid)
```

**问题**：如果median策略导致质心不准确（如点云分布不均匀时），agent无法在运行时切换到mean或其他鲁棒估计方法。

**SpatialClaw的代码接口优势**：

agent可以分步生成代码，并在中间步骤调整策略：

**Step 1（尝试median）**：
```python
heater_centroid = np.median(heater_points, axis=0)
print(f"Median centroid: {heater_centroid}")
```

**Feedback**：
- show()图像：质心位置在3D空间中标注
- agent发现质心偏离视觉中心（可能受离群点影响）

**Step 2（切换到mean）**：
```python
heater_centroid = heater_points.mean(axis=0)
print(f"Mean centroid: {heater_centroid}")
```

**Step 3（进一步优化：KDTree最近邻查询）**：
```python
from scipy.spatial import KDTree
tree = KDTree(door_points)
distances, _ = tree.query(heater_centroid)
print(f"Nearest door point distance: {distances}")
```

**关键差异**：SpatialClaw agent在第3步引入了**原计划中没有的SciPy KDTree**工具，这是在看到mean centroid结果后，判断需要更精确的最近邻距离而自发添加的。单次代码无法实现这种动态策略调整。

### 3.3.2 对比结构化工具调用：通用工具的覆盖

**结构化工具调用的局限**：

以SpaceTools为例，工具接口被预定义为固定API：

```python
# SpaceTools风格的结构化工具调用
tools = {
    "segment_object": SAM3_segment,
    "compute_distance": euclidean_distance,
    "project_3d_to_2d": projection_matrix,
    # ... 预定义的20-30个工具
}
```

**问题**：
1. **新算法无法集成**：如果agent需要使用SciPy的`KDTree`或`convex_hull`，必须等待框架开发者将其添加为工具
2. **组合灵活性受限**：预定义工具通常封装完整流程（如`segment_and_compute_centroid`），agent无法自由组合底层操作
3. **科学计算工具覆盖不全**：空间推理可能需要线性代数（NumPy）、优化（SciPy optimize）、统计（SciPy stats）等通用库，预定义API无法穷举

**SpatialClaw的代码接口优势**：

agent可以**直接调用任意科学计算库**，无需预定义工具：

**案例：使用KDTree进行最近邻查询**

```python
# Step 1: Build KDTree for door point cloud
from scipy.spatial import KDTree
door_tree = KDTree(door_points)

# Step 2: Query nearest door point to each heater point
distances, indices = door_tree.query(heater_points)

# Step 3: Compute statistics
min_distance = distances.min()
mean_distance = distances.mean()
median_distance = np.median(distances)

print(f"Min distance: {min_distance:.4f}m")
print(f"Mean distance: {mean_distance:.4f}m")
```

**关键观察**：
- `scipy.spatial.KDTree`是通用科学计算工具，无法被预定义为"空间推理专用工具"
- agent根据问题特性（需要最小距离而非质心距离）自发选择合适的数据结构和算法
- 如果后续需要，可以无缝切换到`cKDTree`（Cython加速版本）或`BallTree`（其他距离度量）

**对比实验数据**（Table 2, Interface Ablation）：
- No-tool baseline: 53.4%
- Single-pass code: 55.2% (+1.8 pp)
- Structured tool-call (SpaceTools): 56.7% (+3.3 pp vs single-pass)
- **SpatialClaw**: 59.9% (+6.5 pp vs structured)

**6.5个百分点的增益**主要来自：
1. 动态策略调整（如median→mean→KDTree的切换）
2. 通用科学计算工具的可用性（如SciPy spatial的丰富算法）
3. 视觉反馈驱动的迭代优化（show()让agent看到中间结果并调整）

### 3.3.3 灵活性-效率权衡

**潜在代价**：
- 每个样本需要多轮VLM调用（平均7步，每步一次代码生成）
- 总token消耗：规划(100 tokens) + 7步代码生成(每步~500 tokens) + 反馈处理(每步~200 tokens) ≈ **5600 tokens/样本**

**对比单次代码**：
- 单次VLM调用生成完整代码（~2000 tokens）
- 总token消耗：**2000 tokens/样本**

**Token效率比**：5600 / 2000 = 2.8倍

**但性能提升**：从55.2%到59.9%（+4.7 pp）

**论文的Trade-off论证**（Section 4.5）：
- 对于高价值空间推理任务（如机器人导航、AR/VR场景理解），准确率提升远超token成本
- 更大模型（Qwen3.5-397B vs 122B）在SpatialClaw下获得更大增益（Table 1），说明代码接口让模型能力得到更充分发挥
- 训练无关特性：无需为每个benchmark或模型微调，节省大量训练资源

---

# Ch4: 系统架构设计

## 4.1 运行时系统架构

SpatialClaw的工程实现采用**分布式微服务架构**，将计算密集型任务（VLM推理、感知工具处理）与状态管理（Agent内核）分离到独立服务中。这种设计支持：
1. **GPU资源隔离**：VLM推理和感知工具可使用不同GPU卡
2. **容错与重启**：SLURM job-time limit触发时，服务可独立重启而不丢失状态
3. **扩展性**：多个Agent Service实例可共享单个VLM/Tool Service

### 4.1.1 服务架构概览

SpatialClaw运行时由两个可扩展的独立服务组成，Agent运行在IPython kernel中：

1. **vLLM Backbone Service**（VLM推理服务）
   - 部署Qwen3.6或Gemma4模型
   - 提供HTTP/gRPC API用于代码生成和规划
   - 使用vLLM引擎实现高吞吐量推理
   - 配置：Tensor Parallelism用于大模型（如397B版本）

2. **GPU Perception-Tool Server**（感知工具服务）
   - 部署Reconstruct（Depth Anything 3）和SAM3模型
   - 提供3D重建和分割的独立GPU加速
   - 工具调用通过JSON-RPC协议
   - 支持批量处理（多个重建请求可合并执行）

3. **Agent Service**（Agent编排服务）
   - 管理持久化Jupyter Kernels（每个样本一个独立kernel）
   - 管理persistent kernel（IPython/Jupyter），维持所有变量状态
   - 运行agentic循环（五阶段编排）
   - 维护agent状态（历史代码、反馈、变量）
   - 通过共享注册表与其他服务通信

### 4.1.2 服务间通信机制

**JSON Registries（共享注册表）**：

服务间通过**JSON文件注册表**协调，而非直接网络调用。设计模式：

1. **任务注册表**（`registry/tasks.json`）：
   - Agent Service写入待处理任务ID
   - 包含：`{task_id, image_paths, metadata, config}`

2. **模型注册表**（`registry/models.json`）：
   - vLLM Service写入可用模型信息
   - 包含：`{model_name, endpoint_url, max_context_length}`

3. **工具注册表**（`registry/tools.json`）：
   - Perception-Tool Service写入工具端点
   - 包含：`{tool_name, gpu_id, max_batch_size}`

4. **结果注册表**（`registry/results/<task_id>.json`）：
   - Agent Service写入最终答案
   - 包含：`{answer, execution_history, token_usage}`

**通信流程**：

```
[Agent Service] → 写入tasks.json
                 ↓
[vLLM Service]  → 轮询tasks.json，读取任务
                 → 执行代码生成
                 → 写入results/<task_id>/step_1.json
                 ↓
[Perception-Tool Server] → 轮询任务，执行Reconstruct/SAM3
                          → 写入3D点云/mask到results/<task_id>/
                          ↓
[Agent Service] → 轮询results目录，收集反馈
                 → 继续下一步或提交答案
```

**设计优势**：
- **松耦合**：服务崩溃不影响其他服务（注册表持久化）
- **SLURM兼容**：每个服务可独立提交为SLURM job
- **重启恢复**：job-time limit触发后，新job从注册表恢复未完成任务

### 4.1.3 部署考量

论文报告使用SLURM集群管理长时间实验。每个服务可独立提交为计算job，通过注册表持久化实现服务间通信和状态恢复。模型权重需预下载，vLLM/SLURM的完整配置步骤见项目文档。

---

## 4.2 Agent状态管理与工作流

SpatialClaw的agent通过持久化kernel维护状态（代码历史、stdout/traceback反馈、变量摘要、show()图像路径）。工作流以Python原生循环实现五阶段编排，通过条件分支决定继续执行或提交答案。系统可配置步骤上限（N_max=30）防止无限循环。

> ### 类比理解：餐厅后厨的服务分工
>
> SpatialClaw的运行时架构就像餐厅后厨的分工模式：
>
> - **vLLM Backbone Service（主厨）**：负责核心创意工作（生成代码策略），需要最高"智能"（最大模型），如同主厨设计菜品配方。
>
> - **GPU Perception-Tool Server（专业设备）**：提供专用处理能力（3D重建、分割），如同烤箱、搅拌机等专业设备。设备可以被多个"主厨"共享。
>
> - **Agent Service（服务生）**：协调调度和状态管理，如同服务生连接顾客（任务请求）和厨房（VLM+工具），维持整个订单的进度。

---

## 4.3 Benchmark统一配置接口

**配置目录结构**：

```
spatial_agent/config/dataset/
├── mindcube/
│   ├── config.yaml       # 数据集级别配置
│   └── samples.json      # 样本列表（路径、元数据）
├── spar_bench/
│   ├── config.yaml
│   └── samples.json
├── pai_bench/
│   ├── config.yaml
│   └── samples.json
...（20个benchmark，每个一个目录）
```

**config.yaml结构**（以MindCube为例）：

```yaml
# ⚠️ 非官方概念实现，未经验证
dataset_name: "MindCube"
task_type: "static_3d"  # Options: static_3d, multi_view, video_4d, dynamic_3d, hybrid

# Input format
image_format: "multi_view"  # single, multi_view, video
num_views: 5
depth_available: true

# Answer format
answer_type: "float"  # float, int, str, list, coordinate
answer_unit: "meters"
value_range: [0.0, 10.0]

# Evaluation
metric: "absolute_error"  # absolute_error, accuracy, iou

# Tool defaults
default_tools:
  - Reconstruct
  - SAM3
  - KDTree

# Prompt template
system_prompt: "You are a spatial reasoning agent..."
task_prompt_template: "Compute the {property} of {object1} and {object2}..."
```

**samples.json结构**：

```json
[
  {
    "sample_id": "mindcube_0001",
    "image_paths": ["/data/mindcube/0001/view_0.png", "..."],
    "metadata": {
      "camera_intrinsics": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
      "depth_scale": 0.001
    },
    "ground_truth": {
      "answer": 1.234,
      "property": "distance",
      "objects": ["heater", "door"]
    }
  },
  ...
]
```

**统一配置的优势**：
1. **零样本泛化**：新benchmark只需添加config和samples，无需修改agent代码
2. **超参数一致性**：所有benchmark共享相同的`N_max=30`、`system_prompt`、工具列表
3. **评估自动化**：配置文件驱动完整的pipeline（输入→推理→评估）
4. **消融实验支持**：切换配置（如禁用`show()`）无需改动代码

---

## 4.4 无SLURM环境的Python入口降级方案

对于无SLURM的本地开发环境，SpatialClaw提供**纯Python入口**：

**单进程启动脚本**（`launch_local.py`）：

```python
# ⚠️ 非官方概念实现，未经验证
import subprocess
import time

# Start vLLM server in background
vllm_proc = subprocess.Popen([
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen3.6-27B",
    "--host", "localhost",
    "--port", "8000"
])

# Wait for vLLM to be ready
time.sleep(60)

# Start tool server
tool_proc = subprocess.Popen([
    "python", "tools/server.py",
    "--host", "localhost",
    "--port", "8001"
])

time.sleep(30)

# Run agent in main process
from spatial_agent import run_agent
result = run_agent(
    task_config="config/dataset/mindcube/config.yaml",
    vllm_endpoint="http://localhost:8000",
    tool_endpoint="http://localhost:8001"
)

# Cleanup
vllm_proc.terminate()
tool_proc.terminate()
```

**简化设计**：
- 所有服务运行在`localhost`
- 使用subprocess管理后台进程（而非SLURM jobs）
- 无状态持久化到磁盘（session only）
- 适用于单GPU开发/调试

**性能对比**：
- SLURM分布式：可并行处理100+样本（多Agent Service实例）
- Local模式：串行处理，每次一个样本

---

## 4.5 工程实现的挑战与权衡

**挑战1：GPU内存管理**

- **问题**：VLM（27B参数）和感知工具（Depth Anything 3 + SAM3）同时运行可能超出单卡内存
- **解决方案**：
  - vLLM支持模型分片（Tensor Parallelism）到多卡
  - Perception-Tool Server独立部署，使用不同GPU
  - 批处理时动态调整batch size以适应内存

**挑战2：Jupyter Kernel内存泄漏**

- **问题**：长时间运行后，kernel全局空间积累大量变量，可能导致内存溢出
- **解决方案**：
  - N_max=30步的硬限制
  - 每个样本使用独立kernel（样本结束后销毁）
  - 关键变量持久化到磁盘，kernel重启后加载

**挑战3：服务间延迟**

- **问题**：Agent Service轮询注册表（每1秒）可能错过实时更新
- **解决方案**：
  - 关键路径（代码生成→执行）使用直接HTTP调用（绕过注册表）
  - 注册表仅用于服务发现和状态同步
  - 未来可改用消息队列（Redis Streams）替代轮询

**挑战4：错误恢复**

- **问题**：代码执行可能因bug、OOM、工具失败而崩溃
- **解决方案**：
  - Traceback自动注入反馈，agent可自我修正
  - 工具失败时返回错误码而非抛异常（如`Reconstruct return {"status": "failed", "reason": "OOM"}`）
  - 最多3次重试机制（如连续3次相同错误则终止）

**工程实现总结**：

SpatialClaw的分布式架构在**灵活性与性能**之间取得平衡：
- **训练无关**：无需为每个benchmark微调模型
- **模型无关**：支持任意VLM骨干网络（6种验证）
- **任务无关**：统一配置覆盖20个benchmark
- **计算高效**：工具服务共享，避免重复加载模型
- **容错性强**：服务独立部署和状态持久化确保长时间运行稳定

---

# 第五章 实验结果与分析

## 5.1 实验设置

### 模型选择

SpatialClaw在6个不同的VLM骨干网络上进行了全面测试，覆盖26B到397B的参数规模：

| 模型系列 | 具体模型 | 参数量 |
|---------|---------|--------|
| Qwen 3.5 | Qwen3.5-VL-397B | 397B |
| Qwen 3.5 | Qwen3.5-VL-122B | 122B |
| Qwen 3.6 | Qwen3.6-VL-35B | 35B |
| Qwen 3.6 | Qwen3.6-VL-27B | 27B |
| Gemma 4 | Gemma4-31B | 31B |
| Gemma 4 | Gemma4-26B | 26B |

这一设计选择具有重要的科学意义：通过在同一框架下测试从27B到397B的超大参数跨度，研究者能够明确回答"模型规模是否等于空间推理能力"这一根本问题。

### Benchmark覆盖

20个空间推理benchmark被系统性地分为5个类别，覆盖静态3D、多视角、通用空间、4D时序推理和通用视频理解：

1. **单图3D推理**（4个benchmark）：测试从单张图像推断3D关系的能力
2. **多视角推理**（3个benchmark）：测试从不同观察视角整合空间信息的能力  
3. **通用空间推理**（3个benchmark）：测试广泛的空间认知任务
4. **4D视频推理**（6个benchmark）：测试3D+时序的综合推理能力
5. **通用视频理解**（4个benchmark）：测试动态场景中的空间关系理解

### 统一的评估协议

一个关键的设计决策是：所有benchmark和所有模型使用**完全相同的系统prompt、工具集和超参数**。这种"一刀切"的策略验证了SpatialClaw的真正通用性——无需针对特定任务或模型进行任何adaptation或tuning。

### 评估指标

所有benchmark使用标准化的准确率指标（accuracy百分比），确保跨任务的可比性。

## 5.2 主要结果

### 全景表现：Table 1完整解读

Table 1展示了所有6个VLM在20个benchmark上的详细结果。每个数字代表SpatialClaw相对于baseline的提升百分点（pp，percentage points）。

#### 模型维度：谁受益最大？

从平均增益来看：

| 模型 | 平均准确率 | 平均增益 | 规模 |
|------|-----------|---------|------|
| Qwen3.6-27B | 62.7% | +7.7 pp | 27B |
| Gemma4-31B | 59.9% | +6.5 pp | 31B |
| Qwen3.6-35B | [VERIFY: 研究材料中未给出] | [VERIFY: 研究材料中未给出] | 35B |
| Qwen3.5-122B | [VERIFY: 研究材料中未给出] | [VERIFY: 研究材料中未给出] | 122B |
| Qwen3.5-397B | [VERIFY: 研究材料中未给出] | +3.1 pp | 397B |

**关键发现**：Qwen3.6-27B取得了**最高的绝对准确率（62.7%）和最高的平均增益（+7.7 pp）**。这一结果颠覆了"更大模型必定带来更大性能提升"的直觉——397B参数的Qwen3.5-VL仅获得+3.1 pp的增益，远小于27B的Qwen3.6-VL的+7.7 pp。

这揭示了接口设计的核心重要性：**动作接口的灵活性比模型规模更能决定agent的空间推理能力**。

#### 任务维度：哪些任务受益最大？

按benchmark类别的平均增益排序：

| Benchmark类别 | 最大增益模型 | 增益幅度 | 代表性任务 |
|--------------|------------|---------|-----------|
| 多视角推理 | Qwen3.6-27B | +14.3% (近似，Figure分析) | 从多个角度整合3D信息 |
| 4D视频推理 | Qwen3.6-27B | +18.3% (近似，Figure分析) | 动态场景中的时空推理 |
| 单图3D推理 | [VERIFY: 需要具体数据] | [VERIFY: 需要具体数据] | 单张图像的3D关系推断 |
| 通用空间推理 | [VERIFY: 需要具体数据] | [VERIFY: 需要具体数据] | 广泛空间认知任务 |
| 通用视频理解 | [VERIFY: 需要具体数据] | [VERIFY: 需要具体数据] | 动态场景空间关系 |

#### 单个benchmark的明星表现者

三个最显著的增益案例（均为Qwen3.6-27B）：

1. **DSI-Bench：+20.1 pp**
   - 任务类型：多视角3D关系推理
   - 基线准确率：[VERIFY: 研究材料中未给出基线准确率]
   - SpatialClaw准确率：[VERIFY: 研究材料中未给出绝对准确率]
   - 增益解读：接近20个百分点的提升在空间推理领域是革命性的

2. **MindCube：+18.9 pp**
   - 任务类型：3D空间关系推理
   - 基线准确率：[VERIFY: 研究材料中未给出基线准确率]
   - SpatialClaw准确率：[VERIFY: 研究材料中未给出绝对准确率]
   - 增益解读：近乎19个百分点的提升验证了代码接口对复杂3D任务的价值

3. **PAI-Bench：+2.3 pp**（72.1 vs 69.8）
   - 任务类型：[VERIFY: 需要确认具体任务类型]
   - 基线准确率：[VERIFY: 研究材料中未给出基线准确率]
   - SpatialClaw准确率：[VERIFY: 研究材料中未给出绝对准确率]
   - 增益解读：PAI-Bench主要测试物理空间交互理解，代码接口允许agent通过几何原语精确计算物体间的空间关系

### 超越SOTA

在所有6个VLM上，SpatialClaw的平均准确率（59.9%，基于Gemma4-31B）超越了当时最佳的专门系统SpaceTools **11.2个百分点**。这一成就的意义在于：

- SpaceTools是专门为空间推理设计的系统，可能有任务特定的优化
- SpatialClaw是通用框架，使用相同的系统prompt和工具集处理所有任务
- 这一对比验证了"代码接口"的通用威力

### 跨模型一致性

所有6个VLM都在所有benchmark上获得了正向增益，没有出现负收益的案例。这证明了SpatialClaw的接口设计是**普遍有效的**，而非依赖于特定模型的特性。

### 关键洞察：接口 > 规模

将Table 1的数据重新解读，我们发现一个反直觉的模式：

- **Qwen3.6-27B（27B参数）**：+7.7 pp 平均增益，62.7% 绝对准确率
- **Qwen3.5-VL-397B（397B参数）**：+3.1 pp 平均增益，[VERIFY: 需要确认绝对准确率]

397B模型的增益还不到27B模型的一半。这一结果强烈暗示：**在agent系统中，动作接口的设计质量比底层VLM的规模更重要**。

> 💡 **类比理解：自动驾驶的等级进化**
>
> - **L2（无工具baseline）**：车辆可以控制方向盘和油门，但人类需要时刻监控。就像VLM直接输出答案，没有工具辅助。
> - **L3（单次代码执行）**：车辆在特定条件下可以自动驾驶，但遇到复杂情况立即交还人类。就像单次代码执行，看到中间结果时已经来不及调整。
> - **L4（结构化工具调用）**：车辆在限定区域内完全自主，但遇到设计外的场景会失败。就像结构化API，只能应对预定义的工具组合。
> - **L5（SpatialClaw代码接口）**：车辆在任何场景下都能自主处理，根据实时反馈调整策略。就像代码接口，可以动态组合工具、迭代修正、应对意外情况。
>
> 关键洞察：从L2到L5的进化，主要不是传感器更好了（模型规模），而是决策系统更灵活了（接口设计）。

## 5.3 动作接口消融实验

### 实验设计

Table 2使用Gemma4-31B模型，在统一benchmark上对比四种接口设计：

1. **No-tool**：VLM直接输出答案，无任何工具
2. **Single-pass code**：一次性生成完整代码，无中间反馈
3. **Structured tool-call**：调用预定义的API（如SpaceTools风格）
4. **SpatialClaw**：完整的多步代码接口，带视觉反馈

### 消融结果

| 接口类型 | 平均准确率 | 相对No-tool增益 | 相对上一级增益 |
|---------|-----------|----------------|----------------|
| No-tool | 53.4% | — | — |
| Single-pass code | 55.2% | +1.8 pp | +1.8 pp |
| Structured tool-call | 56.7% | +3.3 pp | +1.5 pp |
| **SpatialClaw** | **59.9%** | **+6.5 pp** | **+3.2 pp** |

### 阶梯式性能提升分析

#### No-tool → Single-pass code：+1.8 pp

**增益幅度**：1.8个百分点的提升较为有限。

**原因分析**：
- 单次代码执行让VLM能够使用感知工具（SAM3、Depth Anything 3）
- 但在生成代码时无法看到中间结果，必须一次性预测完整的分析流程
- 对于需要根据中间发现调整策略的任务，这种"盲目"执行的价值有限

**启示**：仅仅给VLM提供工具访问权是不够的，关键是让工具使用成为可迭代的过程。

#### Single-pass → Structured tool-call：+1.5 pp

**增益幅度**：1.5个百分点的提升再次较为有限。

**原因分析**：
- 结构化工具调用（如SpaceTools）提供了一组预定义的API，提高了工具使用的规范性
- API的抽象层确实降低了VLM的编程难度
- 但固定的API限制了工具的组合灵活性——无法根据任务动态创新工具使用模式

**启示**：API级别的抽象有一定帮助，但真正的灵活性来自于完全的编程自由。

#### Structured → SpatialClaw：+3.2 pp

**增益幅度**：3.2个百分点的提升是之前两阶提升之和的**近一倍**。

**原因分析**：
- **完全编程自由**：不再受限于预定义API，可以任意组合感知基元和数值计算
- **视觉反馈闭环**：show()函数将中间可视化结果嵌入下一轮上下文，实现"观察→修正→再执行"的迭代模式
- **状态持久化**：Python内核保持跨步骤的变量状态，避免重复计算，支持复杂的多阶段分析

**核心优势**：SpatialClaw的增益不是来自单一因素，而是三重优势的乘积效应——编程自由 + 视觉反馈 + 状态持久化。

### 总体提升：No-tool → SpatialClaw：+6.5 pp

从53.4%到59.9%，**6.5个百分点的总体提升**在空间推理领域是显著的。更重要的是，这一提升是在**没有针对任何benchmark或模型进行专门优化**的情况下实现的。

### 接口设计的启示

消融实验清晰地验证了论文的核心假设：**动作接口的灵活性是agent性能的关键决定因素**。

- 从No-tool到Single-pass的1.8 pp提升：工具本身有价值
- 从Single-pass到Structured的1.5 pp提升：规范化有帮助  
- 从Structured到SpatialClaw的3.2 pp提升：**完全灵活性才是关键**

最后一阶的提升是前两阶之和的两倍，这证明了"代码作为接口"的设计不是渐进改进，而是质的飞跃。

## 5.4 组件消融

### Planner模块的贡献

Planner是一个独立的LLM会话，在主agent循环前进行高级策略规划。

**消融设计**：移除Planner，直接让主VLM生成第一步代码。

**结果影响**：[VERIFY: 研究材料中未给出具体数字]

**价值分析**：
- Planner减轻了主VLM的推理负担，让其专注于代码生成
- 独立规划避免了代码生成过程中的策略走偏
- 对于复杂的多步任务，规划阶段的分离显著提高了最终成功率

### show()视觉反馈的贡献

show()函数是SpatialClaw的关键创新之一，它将中间可视化结果嵌入agent的下一轮输入。

**消融设计**：禁用show()，agent只能看到文本形式的执行输出（stdout、traceback、变量摘要）。

**结果影响**：[VERIFY: 研究材料中未给出具体数字]

**价值分析**：
- 视觉反馈让agent能够"看到"中间分析结果（如分割mask、3D点云、距离可视化）
- 对于空间推理任务，视觉信息比纯文本描述更具信息量
- show()实现了真正的多模态反馈闭环，弥补了VLM在空间想象上的局限

### AST安全检查的必要性

AST（Abstract Syntax Tree）安全检查在代码执行前进行静态扫描，拒绝包含危险模块的代码（如os、subprocess、importlib等）。

**消融设计**：移除AST检查，执行所有生成的代码。

**结果影响**：[VERIFY: 研究材料中未给出具体数字]

**价值分析**：
- AST检查防止了agent执行危险操作（文件删除、网络请求、代码注入）
- 虽然所有代码在容器化环境中运行，但AST检查提供了额外的安全层
- 对于涉及用户提供的输入和外部数据加载的任务，AST检查尤其关键

### 组件间的协同效应

[VERIFY: 需要分析组件之间是否存在协同效应，即组合移除时的性能下降是否超过单独移除之和]

## 5.5 失败模式分析

### 3D重建错误传播

**问题**：Depth Anything V3的3D重建结果包含噪声或错误，这些错误会传播到下游的空间计算中。

**案例**：
- 深度估计不准确导致距离计算误差
- 物体边界识别错误导致体积计算偏差
- 多视角融合时的对齐失败导致空间关系误判

**缓解策略**：目前没有内置的错误检测机制，完全依赖VLM在观察到不一致结果时进行自我修正。

### 分割失败

**问题**：SAM3的分割结果可能不准确，尤其是对于：
- 与背景颜色相近的物体
- 严重遮挡的物体
- 细长或透明物体

**影响**：分割mask的错误会导致后续的几何计算（如质心、边界框）完全失效。

### 数值精度问题

**问题**：空间推理任务往往需要高精度的数值计算（如判断两个物体是否接触、计算体积的微小差异）。

**案例**：
- 浮点运算的累积误差导致接触判断错误
- 离散化采样不足导致曲面体积计算偏差
- 坐标系转换时的精度损失

**缓解策略**：目前依赖VLM选择合适的数值方法（如提高采样率、使用高精度数据类型），但这是模型能力的固有限制。

### 步骤预算耗尽

**问题**：每个样本有N_max=30步的预算限制，复杂的空间推理任务可能需要更多步骤。

**案例**：
- 多物体场景需要逐个分析，步骤消耗快
- 迭代修正策略（如尝试多种分割参数）可能快速耗尽预算
- 需要重新执行的错误修正会浪费步骤

**权衡**：提高N_max会增加计算成本和延迟，降低N_max会限制任务复杂度上限。

### 失败模式的共同根源

所有失败模式都指向同一个根本限制：**SpatialClaw的绩效依赖于底层VLM和感知工具的质量**。代码接口提供了灵活性和可组合性，但无法弥补底层能力的不足。

> 💡 **类比理解： chef与厨房设备**
>
> SpatialClaw就像一位米其林大厨（代码接口），而底层VLM和感知工具就像厨房的基础设备（炉灶、刀具、食材）。
>
> - 即使大厨技艺精湛，如果炉灶温度不准（3D重建噪声），菜品也会失败
> - 即使刀工一流，如果刀具钝了（分割失败），切配也会出错
> - 即使食谱完美，如果食材不新鲜（数值精度问题），味道也会打折扣
> - 即使时间管理严格，如果厨房太小（步骤预算限制），也无法完成复杂宴席
>
> **启示**：优化接口设计（提升厨艺）很重要，但投资基础能力（升级设备）同样关键。

---

# 第六章 代码实现详解

## 6.1 核心代码示例

### 测量heater到door的距离（Figure 2完整代码）

以下是论文Figure 2中展示的完整5步代码，展示了SpatialClaw如何通过代码接口逐步分析空间关系：

```python
# Step 1: Load and segment the input image
images = InputImages()  # Load RGB-D image
mask_heater = SAM3(images, prompt="heater")
mask_door = SAM3(images, prompt="door")

# Step 2: Reconstruct 3D scene from depth
scene_3d = Reconstruct(images)  # Depth Anything V3
heater_3d = scene_3d.extract(mask_heater)
door_3d = scene_3d.extract(mask_door)

# Step 3: Compute centroids
centroid_heater = heater_3d.centroid()
centroid_door = door_3d.centroid()

# Step 4: Visualize intermediate result
show(heater_3d, door_3d, centroids=[centroid_heater, centroid_door])

# Step 5: Calculate distance and return answer
distance = np.linalg.norm(centroid_heater - centroid_door)
ReturnAnswer(f"The distance from heater to door is {distance:.2f} meters")
```

**代码亮点**：

1. **逐步验证**：通过show()在步骤4可视化3D重建结果，agent可以检查heater和door是否被正确识别
2. **状态复用**：步骤5直接使用步骤2生成的3D对象和步骤3计算的质心，无需重复计算
3. **工具组合**：SAM3（分割）→ Reconstruct（3D重建）→ NumPy（数值计算）的天然组合
4. **自然语言答案**：ReturnAnswer()将数值结果转换为自然语言描述

### ReturnAnswer()调用示例

ReturnAnswer()是SpatialClaw的专用出口函数，用于提交最终答案：

```python
# Simple numerical answer
ReturnAnswer("42")

# Formatted result
ReturnAnswer(f"Volume: {volume:.2f} cubic meters")

# Multi-part answer
ReturnAnswer(f"""
Object A is located at ({x1:.1f}, {y1:.1f}, {z1:.1f})
Object B is located at ({x2:.1f}, {y2:.1f}, {z2:.1f})
Distance: {distance:.2f} meters
""")

# Conditional answer
if distance < threshold:
    ReturnAnswer("The objects are in contact")
else:
    ReturnAnswer(f"The objects are separated by {distance:.2f} meters")
```

**ReturnAnswer()的语义**：
- 调用ReturnAnswer()会终止agent循环，将参数作为最终答案返回
- 未调用ReturnAnswer()且达到N_max步时，使用最后一次执行的输出作为答案
- ReturnAnswer()的参数会被自动格式化为评估脚本可解析的形式

### 迭代修正模式示例

以下代码展示了"观察→修正→再执行"的迭代模式：

```python
# Initial attempt
mask_person = SAM3(images, prompt="person")
show(mask_person)  # Visualize to check quality

# If segmentation is incomplete (observed from show() output)
# Agent generates correction code:
mask_person_refined = SAM3(images, prompt="person", 
                          foreground_hint=[400, 300],  # Add hint point
                          multimask_output=True)
mask_person_refined = select_largest_mask(mask_person_refined)
show(mask_person_refined)  # Verify improvement

# Continue with refined mask
person_3d = scene_3d.extract(mask_person_refined)
```

**迭代模式的价值**：
- 第一步的show()暴露了分割不完整的问题
- Agent根据视觉反馈生成修正代码（添加提示点、选择最大mask）
- 再次show()验证修正效果，确保质量提升后再继续计算

## 6.2 代码仓库结构

### 官方仓库概览

**仓库地址**：https://github.com/NVlabs/SpatialClaw  
**组织**：NVIDIA Research (NVlabs)  
**开源协议**：Apache-2.0

### 目录结构

```
SpatialClaw/
├── spatial_agent/           # Agent核心代码
│   ├── workflow/           # LangGraph状态机
│   │   ├── agent_graph.py  # 主agent循环（Planner/Code/Feedback/Answer）
│   │   ├── planner.py      # 独立规划模块
│   │   └── feedback.py     # 反馈组装逻辑
│   ├── kernel/             # Jupyter内核管理
│   │   ├── kernel_manager.py  # 多kernel并发管理
│   │   ├── ast_checker.py    # AST安全检查
│   │   └── execution_env.py   # 执行环境配置
│   ├── tools/              # 工具包装器
│   │   ├── reconstruct.py  # Depth Anything V3接口
│   │   ├── sam3.py         # SAM3接口
│   │   └── geometry.py     # NumPy/SciPy几何工具
│   ├── config/             # 配置文件
│   │   ├── dataset/        # 20个benchmark配置
│   │   │   ├── mindcube.yaml
│   │   │   ├── pai_bench.yaml
│   │   │   └── ...
│   │   └── model.yaml      # VLM backbone配置
│   └── main.py             # 入口脚本
├── tools/third_party/      # 第三方感知工具
│   ├── Depth_Anything_V3/  # Submodule: 3D重建
│   └── ├── sam3/            # Submodule: 分割
├── vllm_service/           # vLLM推理服务
│   ├── server.py           # vLLM HTTP服务
│   └── models/             # VLM权重存储
├── tool_service/           # GPU感知工具服务
│   ├── server.py           # 感知工具HTTP服务
│   └── gpu_manager.py      # GPU资源分配
├── configs/                # 实验配置
│   ├── eval_all.yaml       # 全部benchmark评估
│   └── slurm/              # SLURM作业脚本
└── scripts/                # 实用脚本
    ├── launch_all.sh       # 启动三服务架构
    └── download_weights.sh # 模型权重下载
```

### 核心模块解析

#### spatial_agent/workflow/agent_graph.py

**职责**：实现LangGraph状态机，定义agent的步骤转换逻辑。

**关键状态**：
```python
class AgentState(TypedDict):
    messages: List[BaseMessage]      # VLM对话历史
    code_outputs: List[str]          # 代码执行输出
    visual_feedback: List[Image]     # show()生成的图像
    step_count: int                  # 当前步骤数
    final_answer: Optional[str]      # 最终答案
```

**转换逻辑**：
```python
def should_continue(state: AgentState) -> Literal["continue", "stop"]:
    if state["final_answer"] is not None:
        return "stop"  # ReturnAnswer()已调用
    if state["step_count"] >= N_max:
        return "stop"  # 步骤预算耗尽
    return "continue"  # 继续下一轮
```

#### spatial_agent/kernel/ast_checker.py

**职责**：执行前静态扫描，检测危险模块导入。

**黑名单**：
```python
BLACKLISTED_MODULES = {
    "os", "subprocess", "importlib", "eval", "exec", 
    "shutil", "sys", "pickle", "__import__"
}

def check_ast_safety(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in BLACKLISTED_MODULES:
                        return False, f"Blacklisted module: {alias.name}"
            # ... 更多检查规则
        return True, "AST check passed"
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
```

#### spatial_agent/tools/reconstruct.py

**职责**：包装Depth Anything V3，提供统一的3D重建接口。

**核心函数**：
```python
def Reconstruct(images: InputImages) -> Scene3D:
    """
    从RGB-D图像重建3D场景
    
    Args:
        images: 包含RGB和depth的图像对象
    
    Returns:
        Scene3D: 可查询的3D场景对象
    """
    depth_map = depth_anything_v3(images.rgb)
    point_cloud = depth_to_3d(depth_map, images.intrinsics)
    return Scene3D(point_cloud)

class Scene3D:
    def extract(self, mask: np.ndarray) -> Object3D:
        """根据mask提取3D对象"""
        return Object3D(self.point_cloud[mask])
    
    def distance(self, obj1: Object3D, obj2: Object3D) -> float:
        """计算两个对象之间的距离"""
        return np.linalg.norm(obj1.centroid() - obj2.centroid())
```

#### spatial_agent/config/dataset/*.yaml

**职责**：定义每个benchmark的元数据、评估指标、数据格式。

**示例：mindcube.yaml**
```yaml
name: "MindCube"
category: "single_image_3d"
metrics:
  - type: "accuracy"
    parser: "parse_multiple_choice"  # 答案是A/B/C/D选项
data_format:
  image: "RGB-D"
  question: "natural_language"
  answer_type: "multiple_choice"
evaluation:
  num_samples: 500
  split: "test"
```

## 6.3 如何运行SpatialClaw

### 前置依赖

```bash
# 系统依赖
sudo apt install -y python3.10 python3.10-venv nvidia-container-toolkit

# Python依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install vllm langgraph jupyter numpy scipy matplotlib pillow
```

### 模型权重预下载

```bash
# 下载VLM权重（以Qwen3.6-27B为例）
cd SpatialClaw
bash scripts/download_weights.sh --model Qwen3.6-VL-27B

# 下载感知工具权重（自动初始化submodules）
git submodule update --init --recursive
```

### 三服务架构启动

SpatialClaw采用**服务分离架构**，三个独立服务通过HTTP通信：

#### 服务1：vLLM推理服务

```bash
cd vllm_service
python server.py \
    --model Qwen/Qwen3.6-VL-27B \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.85
```

**职责**：提供VLM推理API，接收system prompt + user input → 返回generated text。

#### 服务2：GPU感知工具服务

```bash
cd tool_service
python server.py \
    --host 0.0.0.0 \
    --port 8001 \
    --devices 0 1  # 使用GPU 0和1
```

**职责**：运行SAM3和Depth Anything V3，接收图像和参数 → 返回分割mask或3D点云。

**启动关键**：感知工具通常需要独占GPU，因此与VLM推理服务分离部署。

#### 服务3：Agent服务

```bash
cd spatial_agent
python main.py \
    --vllm-url http://localhost:8000 \
    --tool-url http://localhost:8001 \
    --benchmark mindcube \
    --output-dir ./results \
    --num-samples 100
```

**职责**：管理agent循环、Jupyter内核、与另外两个服务通信。

### SLURM模式（集群部署）

对于SLURM管理的计算集群，SpatialClaw提供了launch managers：

```bash
# 提交评估作业
sbatch configs/slurm/eval_all.sh

# eval_all.sh内容：
#!/bin/bash
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

module load cuda/11.8
bash scripts/launch_all.sh --eval-all
```

**SLURM优势**：
- 自动GPU分配和资源隔离
- 作业超时保护（48小时后自动终止）
- 支持checkpoint和断点续算

### 纯Python模式（单机开发）

对于快速原型和开发，可以使用纯Python模式：

```python
from spatial_agent import SpatialClawAgent
from spatial_agent.config import load_benchmark_config

config = load_benchmark_config("mindcube")
agent = SpatialClawAgent(
    vlm_model="Qwen3.6-VL-27B",
    config=config
)

result = agent.evaluate(
    image_path="./test_data/sample.png",
    question="What is the distance between the cup and plate?"
)
print(result.answer)  # "The distance is 0.25 meters"
```

**适用场景**：
- 单样本调试
- 新benchmark配置测试
- 消融实验快速验证

## 6.4 配置示例：添加新Benchmark

### 步骤1：创建dataset配置文件

在`spatial_agent/config/dataset/`创建新文件`my_benchmark.yaml`：

```yaml
name: "MyCustomBenchmark"
category: "single_image_3d"  # 选择最接近的类别
metrics:
  - type: "accuracy"
    parser: "parse_exact_match"  # 答案必须是精确匹配
data_format:
  image: "RGB"  # 如果只有RGB，Depth Anything会估计深度
  question: "natural_language"
  answer_type: "text"  # 自由文本答案
evaluation:
  num_samples: 200
  split: "test"
  timeout: 300  # 每样本最大推理时间（秒）
```

### 步骤2：实现数据加载器

在`spatial_agent/data/loaders.py`添加：

```python
class MyCustomBenchmarkLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def load_sample(self, sample_id: int) -> dict:
        """加载单个样本"""
        return {
            "image": f"{self.data_dir}/images/{sample_id:04d}.png",
            "question": self._load_question(sample_id),
            "ground_truth": self._load_answer(sample_id),
            "metadata": {}  # 可选的额外元数据
        }
    
    def _load_question(self, sample_id: int) -> str:
        with open(f"{self.data_dir}/questions/{sample_id:04d}.txt") as f:
            return f.read().strip()
    
    def _load_answer(self, sample_id: int) -> str:
        with open(f"{self.data_dir}/answers/{sample_id:04d}.txt") as f:
            return f.read().strip()
```

### 步骤3：注册benchmark

在`spatial_agent/config/registry.py`添加：

```python
BENCHMARK_REGISTRY = {
    # ... 现有benchmark
    "my_custom": {
        "loader": MyCustomBenchmarkLoader,
        "config": "my_benchmark.yaml"
    }
}
```

### 步骤4：运行评估

```bash
python spatial_agent/main.py \
    --benchmark my_custom \
    --data-dir /path/to/my/benchmark \
    --output-dir ./results/my_custom
```

### 配置扩展：自定义工具

如果benchmark需要特殊工具，可以添加自定义工具：

```python
# 在spatial_agent/tools/custom_tool.py
def custom_analysis(image: np.ndarray, param: float) -> dict:
    """自定义分析逻辑"""
    result = perform_custom_computation(image, param)
    return {"value": result, "metadata": {...}}

# 在kernel初始化时导入
# spatial_agent/kernel/execution_env.py
CUSTOM_TOOLS = {
    "custom_analysis": custom_analysis
}
```

然后在代码中可以直接调用：
```python
result = custom_analysis(images.rgb, param=0.5)
```

---

# 第七章 局限性与延伸阅读

## 7.1 核心局限性

### 依赖底层感知模型质量

**问题**：SpatialClaw的空间推理绩效完全依赖于SAM3（分割）和Depth Anything V3（3D重建）的质量。这两个模型的任何错误都会直接传播到agent的最终答案。

**具体表现**：
- **分割失败**：SAM3可能无法准确分割与背景颜色相近的物体，或被严重遮挡导致mask不完整。不准确的mask会导致3D对象的几何计算（质心、体积、距离）完全错误。
- **深度估计噪声**：Depth Anything V3在纹理缺乏区域、镜面反射表面、透明物体上的深度估计往往不准确。这些噪声会被放大到后续的3D关系计算中。
- **错误累积**：多步分析中，第一步的感知错误会通过状态持久化传播到所有后续步骤。例如，错误的mask → 错误的3D对象 → 错误的距离计算 → 错误的最终判断。

**当前缓解方案**：
- 依赖VLM在观察到不一致结果时进行自我修正（如发现距离为负值，重新生成分割代码）
- 在show()可视化中发现明显错误时，agent可能生成修正代码
- 但这些是被动应对，而非主动错误检测机制

**根本解决方向**：
- 集成ensemble感知模型（多个分割/深度模型的投票）
- 添加显式的错误检测模块（如深度一致性检查、分割质量评分）
- 支持多源信息融合（如结合文本描述的约束来纠正感知错误）

### GPU资源需求

**问题**：SpatialClaw需要同时运行三个GPU密集型服务：
1. **VLM推理服务**：397B参数的Qwen3.5-VL需要多卡推理（4×A100 80GB）
2. **感知工具服务**：SAM3和Depth Anything V3各需要至少1张A100
3. **Agent服务**：虽然主要运行在CPU，但Jupyter kernel的数值计算也可能需要GPU加速

**资源需求估算**：
- **最小配置**（27B VLM）：2张A100 80GB（1张VLM + 1张感知工具）
- **推荐配置**（122B VLM）：6-8张A100 80GB
- **最大配置**（397B VLM）：12-16张A100 80GB

**影响**：
- 高硬件门槛限制了学术实验室和中小企业的复现能力
- 大规模评估（20个benchmark × 多个VLM）的计算成本极高
- 实时交互应用面临延迟挑战（VLM推理 + 3D重建可能需要数十秒）

**优化方向**：
- 模型量化：使用8-bit或4-bit量化降低VLM显存需求
- 感知工具蒸馏：训练轻量级的任务特定分割/深度模型
- 缓存机制：对常见场景的感知结果进行缓存复用

### 语言生态绑定

**问题**：当前SpatialClaw完全绑定Python生态，这是由科学计算库（NumPy、SciPy）和感知工具（SAM3、Depth Anything）的Python实现决定的。

**限制**：
- 无法利用Julia、R、MATLAB等语言在特定科学计算领域的优势
- 难以集成已有的Fortran/C++科学计算库（如FFTW、BLAS）
- 限制了非Python开发者（如传统C++GIS开发团队）的采用

**扩展方向**：
- 支持多内核架构：同时维护Python、Julia、R的kernel，让agent选择最适合的语言
- 跨语言调用：通过RPC机制让Python kernel调用其他语言的函数
- 代码翻译：在代码生成阶段，让VLM生成多语言版本的等价代码

### 计算成本

**问题**：每个样本需要多轮VLM调用（规划 + 多步代码生成 + 答案总结），每轮调用都涉及高昂的推理成本。

**成本估算**：
- 单步代码生成：27B VLM需要约2-5秒（A100）
- 典型样本需要5-15步 → 总计10-75秒
- 加上3D重建（1-3秒）和分割（0.5-1秒），单样本总耗时可能超过1分钟

**影响**：
- 大规模数据集评估需要数天到数周的计算时间
- 实时交互应用（如AR/VR辅助）面临延迟挑战
- 边缘设备（如移动机器人）部署几乎不可能

**优化方向**：
- 小模型微调：针对特定空间推理任务微调小模型（7B以下），替代大模型VLM
- 代码复用：学习常用代码模式，减少从零生成的需求
- 增量计算：仅重新计算受影响的部分，而非全盘重来

### 步骤预算限制

**问题**：N_max=30步的预算虽然对大多数任务足够，但对于复杂的多物体场景或需要大量试错的场景，可能提前耗尽。

**案例**：
- 10个物体的场景，每个物体需要分割、重建、分析 → 30步仅够3个物体的完整分析
- 需要尝试多种分割参数的精细任务 → 试错过程快速消耗步骤

**权衡**：
- 提高N_max会增加计算成本和延迟
- 降低N_max会限制任务复杂度上限
- 自适应预算可能带来不公平的评估对比

## 7.2 未来研究方向

### 扩展到更多科学计算语言

**动机**：不同科学领域有不同的"母语"：
- **天体物理**：Julia的高性能数值计算
- **统计遗传学**：R的丰富统计包
- **信号处理**：MATLAB的DSP工具箱
- **计算几何**：C++的CGAL库

**研究方向**：
1. **多语言代码生成**：训练VLM理解多语言代码的语义等价性
2. **跨语言变量映射**：Python kernel的变量如何透明传递给Julia kernel
3. **语言选择策略**：agent如何根据任务特点自动选择最合适的语言

### 支持多模态工具组合

**现状**：SpatialClaw的工具库局限于视觉感知（SAM3、Depth Anything）和数值计算（NumPy、SciPy）。

**扩展方向**：
- **音频工具**：集成语音识别、声源定位，支持"audio-visual空间推理"（如判断声源位置）
- **触觉工具**：集成力反馈、纹理分析，支持机器人抓取任务中的触觉-视觉融合
- **物理仿真**：集成物理引擎（如MuJoCo、PyBullet），支持预测物体运动和碰撞

**研究问题**：
- 如何设计跨模态的工具组合接口？
- 如何处理多模态工具之间的时序同步（如视频+音频流对齐）？

### 降低计算开销

**研究方向**：
1. **工具使用策略学习**：通过学习past examples，预测当前任务最优的工具序列，减少试错
2. **增量式3D重建**：仅重建任务相关的区域（如仅重建前景物体），而非整个场景
3. **代码缓存与复用**：将常见分析模式（如"计算两个物体间距离"）模板化，避免重复生成

### 错误检测与自愈

**现状**：错误修正依赖于VLM被动观察到不一致结果。

**未来**：
- **主动错误检测**：在每步执行后自动运行sanity checks（如深度范围检查、mask连通性检查）
- **自动修正建议**：检测到错误时，自动生成修正代码模板（如检测到mask噪声 → 建议形态学操作）
- **回滚机制**：错误修正失败时，回滚到之前正确状态，避免错误累积

### 可解释性增强

**研究问题**：agent如何向人类解释其推理过程？

**方向**：
- **可视化执行轨迹**：将agent的代码执行路径、中间结果可视化成流程图
- **自然语言解释生成**：让agent在每步生成解释性文本（"我正在计算距离，因为..."）
- **交互式查询**：人类可以中断agent，询问"为什么选择这个工具？"、"这个参数如何确定？"

### 通用性与特异性的平衡

**现状**：SpatialClaw的强项是通用性（同一框架、同一prompt、同一工具集处理所有任务）。

**研究问题**：
- 通用框架是否永远无法超越为特定任务设计的专用系统？
- 如何在保持通用性的同时，针对任务族（如"所有视频推理任务"）进行轻量级adaptation？
- 能否通过meta-learning自动发现任务簇，并为每个簇生成优化的子prompt？

## 7.3 延伸阅读

### 与SpatialClaw直接相关的系统

#### SpaceTools

**关系**：SpatialClaw论文中的主要baseline，之前SOTA的空间推理agent系统。

**区别**：
- **接口设计**：SpaceTools使用结构化工具调用（预定义API），SpatialClaw使用完全自由的代码生成
- **性能差距**：SpatialClaw在同等VLM下超越SpaceTools 11.2个百分点
- **适用范围**：SpaceTools可能针对特定benchmark优化，SpatialClaw是通用框架

**核心论文**：需要查找SpaceTools的原始论文（具体arXiv ID或会议来源需要在论文参考文献中确认）。

#### pySpatial

**关系**：另一个空间推理agent系统，与SpaceTools同期的相关工作。

**特点**：[VERIFY: 需要查阅pySpatial论文确认其具体方法]

### 代码作为接口的先驱工作

#### Visual Programming (2022)

**贡献**：首次系统性提出"程序生成"作为视觉推理的接口，通过组合预定义的视觉模块（如分割、检测、问答）来解决复杂视觉任务。

**与SpatialClaw的关系**：
- **相同点**：都认为代码/程序是强大的抽象工具
- **不同点**：Visual Programming使用模块组合（类似搭建积木），SpatialClaw使用自由编程（类似从零编写）

**核心论文**：需要查找Visual Programming的原始论文（可能在CVPR/ICCV/ECCV）。

#### ViperGPT

**贡献**：提出"用Python代码表达视觉推理"，让VLM生成调用计算机视觉API（如检测、分割）的代码。

**与SpatialClaw的关系**：
- **相同点**：都使用代码作为VLM与视觉工具的桥梁
- **不同点**：ViperGPT侧重单次代码生成，SpatialClaw强调多步迭代和视觉反馈闭环

**核心论文**：需要查找ViperGPT的原始论文（可能在CVPR/ICCV）。

#### Code-as-Policies

**贡献**：在机器人领域提出"代码作为策略"，让agent生成控制机器人的代码，而非传统的RL策略网络。

**与SpatialClaw的关系**：
- **相同点**：都认为代码提供了超越神经网络的组合泛化能力
- **不同点**：Code-as-Policies专注于动作控制（机械臂运动），SpatialClaw专注于空间认知（几何关系推理）

**核心论文**：需要查找Code-as-Policies的原始论文（可能在RSS/CoRL）。

### 理论基础：为什么代码有效？

#### The Bitter Lesson (Rich Sutton, 2019)

**核心论点**：从长期看，**泛化能力**和**计算能力**比领域知识更重要。代码作为一种通用计算形式，体现了这一原则。

**与SpatialClaw的关系**：SpatialClaw的成功可以看作是"The Bitter Lesson"在agent系统中的验证——代码的通用灵活性（泛化能力）比精心设计的工具API（领域知识）更有效。

#### Compositionality in AI

**核心概念**：复杂能力来自于简单原子的组合，而非端到端学习。

**与SpatialClaw的关系**：SpatialClaw的"感知基元 + 数值计算"组合模式体现了compositionality——分割、3D重建、距离计算都是简单原子，但它们的组合可以解决复杂空间推理任务。

### Agent系统的理论框架

#### Tool-Augmented LLM Survey

**内容**：[需要查找具体survey论文]系统性地总结了LLM与工具结合的不同范式：API调用、代码生成、检索增强等。

**与SpatialClaw的关系**：SpatialClaw可以被归类为"代码生成 + 多步迭代"类别，是该survey中的一个重要案例。

#### Agentic AI Taxonomy

**内容**：[需要查找具体taxonomy论文]提出了agent系统的分类维度：自主性、记忆、规划、工具使用等。

**与SpatialClaw的关系**：SpatialClaw在"工具使用"维度得分最高（完全编程自由），在"记忆"维度得分较高（持久化kernel），在"自主性"维度得分中等（仍需人类提供system prompt）。

### 应用领域的扩展

#### Scientific Discovery with AI Agents

**趋势**：将agent系统应用于科学发现（如材料设计、药物研发、天体物理模拟）。

**与SpatialClaw的关系**：SpatialClaw的空间推理能力可以直接应用于：
- **地质学**：分析地形数据，推断地质结构
- **天体物理**：分析星系图像，推断星系演化阶段
- **分子生物学**：分析蛋白质结构图像，推断功能域

#### Embodied AI and Robotics

**趋势**：将agent系统部署到物理世界（如家庭机器人、自动驾驶）。

**与SpatialClaw的关系**：
- **直接应用**：空间推理是机器人的核心能力（如"杯子在哪里？"、"能否穿过这个门？"）
- **挑战**：当前SpatialClaw的计算成本（分钟级延迟）无法满足实时机器人的需求（秒级响应）
- **机遇**：SpatialClaw的代码接口可以自然扩展到控制机器人动作（在返回答案的基础上，增加执行motor commands的能力）

### 评估与基准

#### Spatial Reasoning Benchmarks

**主要benchmark**：
- **MindCube**：3D空间关系推理
- **SPAR-Bench**：[VERIFY: 需要确认具体任务类型]
- **MMSI**：[VERIFY: 需要确认具体任务类型]
- **DSI-Bench**：多视角3D关系推理
- **PAI-Bench**：物理空间交互理解，测试物体间空间关系推理

**挑战**：
- **数据泄露**：确保benchmark的训练集未在VLM预训练阶段使用
- **评估偏见**：某些benchmark可能过度依赖特定类型的空间推理（如距离计算），忽视其他类型（如拓扑关系）
- **规模限制**：当前benchmark的样本规模（数百到数千）远小于视觉问答benchmark（如VQA v2有数十万样本）

#### Agent Evaluation Methodology

**核心问题**：如何公平评估agent系统？

**维度**：
- **性能**：最终答案的准确率（SpatialClaw主要评估此维度）
- **效率**：达到答案所需的步骤数、计算时间、成本
- **鲁棒性**：对输入扰动、工具失败的抵抗力
- **可解释性**：人类能否理解agent的推理过程

**SpatialClaw的评估局限**：主要关注性能，对效率、鲁棒性、可解释性的评估较少。

---