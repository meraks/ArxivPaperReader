
> **论文**：Generative Compilation: On-the-Fly Compiler Feedback as AI Generates Code
> **作者**：Niels Mündler-Sasahara (ETH Zurich), Hristo Venev (INSAIT), Dawn Song (UC Berkeley), Martin Vechev (ETH Zurich), Jingxuan He (UC Berkeley)
> **arXiv ID**：2607.13921
> **发表时间**：2026-07-15
> **分类**：cs.PL, cs.AI, cs.LG
> **代码仓库**：https://github.com/eth-sri/generative-compilation

## 第 1 章 概述

### 1.1 一句话定位

本文提出 **Generative Compilation**，一种在 LLM 逐 token 代码生成过程中实时获取编译器反馈的方法，首次实现利用现有编译器（无需重新实现语言语义）检查不完整程序，同时支持黑盒模型 API。

### 论文图表总览

| 编号 | 内容 | 章节 |
|------|------|------|
| Figure 1 | 非编译通过 LLM 生成 Rust 代码示例 | 第 2 章 |
| Figure 2 | Rust 编译器错误信息 (E0502) | 第 2 章 |
| Figure 3 | 修复后的 Rust 代码 | 第 2 章 |
| Figure 4 | Constrained decoding 拒绝 ' . ' 时刻 | 第 2 章 |
| Figure 5 | Sealor 生成的完整 Rust 程序 | 第 3 章 |
| Figure 6 | Generative Compilation + LLM 系统架构图 | 第 3 章 |
| Figure 7 | FR 核心演算语法定义 | 第 4 章 |
| Figure 8 | FR 类型规则（含 Lean 机械化修正） | 第 4 章 |
| Figure 9 | FR 部分语法定义 | 第 4 章 |
| Figure 10 | 实现关系 (Realization) 定义 | 第 4 章 |
| Figure 11 | SFR Sealor 定义（13 种 case） | 第 4 章 |
| Figure 13 | 错误种类分布直方图 | 第 6 章 |
| Figure 14 | Restart Budget 分析 | 第 6 章 |
| Figure 15 | 端到端运行示例 (rustls-webpki) | 附录 |

### 1.2 核心贡献

1. **Generative Compilation 框架**：首个在代码生成过程中获得编译器反馈的方法，介于 post-generation feedback（延迟反馈）和 constrained decoding（需白盒+重实现）之间
2. **Sealor 概念**：一种轻量级、语法导向的变换，将不完整程序转为可被现有编译器检查的完整程序；提出了 completeness（不错误拒绝可完成部分程序）和 soundness（不错误接受死胡同部分程序）的形式化定义
3. **FR 核心演算 + Lean 机械化**：在 Featherweight Rust（FR）上构建 sealor，用 Lean 完成完整机械化证明（包括类型安全性、借用安全性），过程中修正了原 FR 形式化的多个错误
4. **首个 Rust 部分程序检查器**：将 sealor 方法论扩展到真实 Rust，委托 rustc 处理类型检查、借用检查和生命周期推理
5. **多模型实验验证**：在 7 个前沿 LLM（GPT 5.3 Codex、Claude Opus 4.8、Gemini 3.5 Flash、Qwen 3.5 397B、Qwen 3.5 9B、GLM 5.2、Kimi K2.7 Code）上评估，两项仓库级 Rust 任务（C-to-Rust 翻译 20 个 + UpdatedAPI 30 个）验证有效性

### 1.3 关键结果速览

- **编译通过率提升**：Generative Compilation (GC) 达到 84.9–87.0%，超过 Post-generation Compilation (PC) 的 79.2–80.5%（聚合所有模型）
- **重启效率**：在 7 次重启处 GC 超越 PC 4.4 个百分点；将重启预算从 15 增至 20 次时，GC 通过率从 84.9% 提升至 87.0%（+2.1 pp），PC 从 79.2% 提升至 80.5%（+1.3 pp）
- **反馈提前量**：在 32 行函数的第 4 行即可捕获错误（vs PC 需等完整生成）
- **最频繁错误**：E0308（类型不匹配）占 38.2%（翻译）和 37.4%（UpdatedAPI）
- **覆盖错误类型**：语法错误、类型错误、借用检查 / 生命周期错误、其他

## 第 2 章 研究背景与动机

### 2.1 AI 代码生成的挑战

LLM 生成的代码无法保证遵守目标语言的静态语义规则。对于 Rust 这类具有严格所有权和借用检查机制的语言，有效代码的生成比 C 等更宽松的语言更困难。现有两种主流策略各有局限。

### 2.2 Post-Generation Compiler Feedback

标准做法是等待 LLM 完整生成文件后，再运行编译器获取诊断信息并反馈给模型。其优点是与黑盒模型兼容、直接复用编译器基础设施。但存在两个关键问题：

- **反馈延迟**：每次错误出现在第 N 行，但编译器诊断只在完整生成第 N+K 行后才返回。早期错误会累积影响后续生成（雪球效应）
- **批量错误混乱**：模型收到的是多个错误打包在一起的诊断报告，难以定位和修复

考虑以下代码生成场景——LLM 被要求实现一个从消息中剥离固定大小头部的函数：

```rust
fn fmt_msg(msg: &mut String) -> String {
    let tag = &msg[..HEADER_SIZE];
    msg.drain(..HEADER_SIZE);
    let prio = tag.starts_with("HIGH");
    let body = msg;
    let n = body.lines().count();
    format!("{} {}: {}", prio, n, body)
}
```

Rust 编译器报告 E0502 错误：`tag` 在 Line 2 不可变借用 `msg`，而 Line 3 尝试可变借用 `msg`，Line 4 又使用了 `tag`。此错误在 Line 4 就已不可挽回，但编译器反馈只在生成全部 8 行后才给出。

### 2.3 Constrained Decoding

另一种方案是在解码过程中通过前缀检查（prefix checker P: Σ* → B）过滤 token，确保生成的每个前缀都存在有效完成路径。其关键局限：

- **语言语义需重新实现**：Rust 编译器相关代码超过 60 万行，不可能重新实现。现有系统只能覆盖语法 [39, 53] 或极小子集 [31, 34]
- **静默过滤**：模型被悄悄导向低概率的延续路径（如使用未定义标识符来避免借用冲突），但从未被告知原因，也无法修改已生成的前缀
- **白盒需求**：需要访问模型的下一个 token 分布和采样过程，无法用于 OpenAI/Anthropic 等闭源 API

### 2.4 Generative Compilation 的定位

**Generative Compilation** 介于两者之间——像 post-generation feedback 一样产生文本诊断，像 constrained decoding 一样处理部分程序，但：
- 复用现有编译器基础设施（无需重新实现）
- 支持黑盒模型（通过流式 API 获取部分输出）
- 在错误的第一时间（而非生成完成后）提供聚焦的诊断

## 第 3 章 Generative Compilation 框架

### 3.1 形式定义

**Generative Compiler** 是一个函数 G: Σ* → B × Σ*，与常规编译器接口相同。它对部分程序执行前缀检查，即判断当前输出 c 是否存在有效完成 w 使得 c ◦ w ∈ L。与 constrained decoding 不同，它保留编译器诊断信息：ok=false 时 err 提供文本解释。

**Completeness**（完备性）：若 c 存在有效完成（∃w. c ◦ w ∈ L），则 G 接受 c。G 在 X ⊆ Σ* 上是完备的，若对所有 c ∈ X，当存在有效完成时 G 必接受。

**Soundness**（可靠性）：G 接受 c 时，c 确实存在有效完成。G 在 X 上是可靠的，若对所有 c ∈ X，G 接受则 c 必存在有效完成。

**Sealor** S: Σ* → Σ*：一种轻量级、语言特定的变换，将部分程序转化为常规编译器可处理的完整程序。给定编译器 C 和 sealor S，诱导的 generative compiler 为 G_{C,S}(c) = C(S(c))。

**定理 3.2**（Sealor 性质提升）：若 C 是 exact（即同时完备且可靠），且 S 在 X 上完备，则 G_{C,S} 在 X 上完备。同理，若 S 在 X 上可靠，则 G_{C,S} 在 X 上可靠。

**目标设定**：Global completeness（全局完备性）+ selective soundness（选择性可靠性）。不完整性是更有害的失败模式——错误拒绝可完成的程序会误导模型放弃有效的前缀。可靠性可以放松，因为拒绝会触发修订（模型可以在反馈后修改已生成的 token）。

### 3.2 系统架构

Generative Compilation 与 LLM 代码生成作为两个并发模块运行：

1. **LLM 模块**：执行标准自回归解码，将每个已生成的前缀 c 流式发送给 generative compiler。无需访问 token 分布，仅需流式 API 输出
2. **GC 模块**：接收前缀后使用 sealor 转换为完整程序，调用现有编译器检查。使用 latest-wins 策略（不中断正在进行的验证，缓存最新前缀）
3. 当编译器拒绝时，GC 模块将诊断信息附加到 prompt 中重新生成
4. 当编译器接受且生成已结束（EOS），返回 c 为最终程序

两个模块仅通过纯文本通信，支持黑盒 LLM API。

### 3.3 Sealor 的完备性与可靠性权衡

Constrained decoding 需要全局可靠性（global soundness）来确保逐 token 归纳总是处于有效路径上，这迫使其逼近 exact prefix checking。而 GC 改变了优先级：

- **Completeness 优先**：拒绝有有效完成的程序是最有害的失败模式
- **Soundness 可放松**：接受死胡同前缀是容忍的——错误可在后续检查或最终编译阶段捕获
- 目标定为 **global completeness + X 上的 selective soundness**，X 是一个精心选择的程序类别

## 第 4 章 FR 核心演算与 Sealor

### 4.1 为什么选 FR

Featherweight Rust (FR) [41] 遵循 Featherweight Java 的设计哲学：在保持类型安全性证明精髓的同时提供简洁的最小演算。FR 的优势在于：

- 接近 Rust 表面语法（直接操作源代码级结构）
- 捕获关键语义：copy/move、可变/不可变借用、词法生命周期
- 紧凑（语法规则约 2 页）
- 已有类型和借用安全性的形式化证明

### 4.2 FR 语法与类型系统

FR 语法包括两大类值类型：拥有引用 ℓ•（drop 时递归释放）和借用引用 ℓ◦（drop 时不释放）。类型系统采用流敏感的 input/output 环境传递方式（Γ1 ⊢ ⟨t: T⟩_l σ ⊣ Γ2），允许在移动语义后标记槽为已移出。

关键类型规则：
- **T-Copy vs T-Move**：分别对应 Rust 中实现了 Copy trait 和未实现的类型
- **T-ImmBorrow / T-MutBorrow**：模拟 Rust 的共享与可变借用互斥
- **T-Block**：引入词法生命周期，防止块局部存储的引用逃逸
- **T-Declare / T-Assign**：变量声明和赋值，带环境更新

### 4.3 部分语法与实现关系

FR 的部分语法（partial syntax）定义了自回归生成中可能出现的部分程序形式。核心特点是允许一个子成分处于"未完成"状态（b ṭ表示部分项）：

```plaintext
b ṭ ::= b             还没有任何内容
       | ṭ            完整项
       | b v           部分值
       | box b ṭ        部分堆分配
       | copy b ẇ      部分复制
       | ...           （共约 13 种情况）
```

**实现关系**（Realization ⇝）：b ṭ ⇝ ṭ 表示 ṭ 通过扩展 b ṭ 得到（即自回归生成的建模）。这不同于 sealing——sealing 可能丢弃或修改未完成片段。

### 4.4 SFR Sealor

SFR 是一个语法导向的 sealor，按 13 种 case 定义。其设计核心是：保留生成的结构以暴露类型检查义务，抽象掉尚未生成信息的部分。

关键设计选择：
- 完全生成的项（ṭ）保持原样不变
- copy b ẇ、b ẇ、& [mut] b ẇ、let mut b x 直接封为 ε（它们的未完成子成分在类型规则中的前提尚不可用）
- 未闭合块 {l ṭ; b ṭ 递归密封尾部后追加 ε 和闭合括号
- let mut x = b ṭ 和 ẇ = b ṭ 递归密封右侧，丢弃声明/赋值自身

**全局完备性**（Theorem 5.3）：SFR 在任意字符串上全局完备。证明分三层：
1. Theorem 5.1：若 b ṭ 可实现一良类型项，则 SFR(b ṭ) 也良类型
2. Corollary 5.2：顶层的完备性
3. Theorem 5.3：提升到任意字符串

**选择性可靠性**（Theorem 5.5）：SFR 在 statement boundaries（{l ṭ; b，即完整语句后）上可靠。这是一个在实践中频繁出现的、有用的局部可靠性保证。

### 4.5 Lean 机械化

FR 的语法、类型规则、操作语义、类型安全性证明和借用安全性证明全部在 Lean 中机械化实现。过程中修正了原 FR 论文的多处类型规则错误（T-Block、T-Declare、T-Assign 等规则的修正），以及单元值规则、运行时语义对齐、progress invariant 的借用安全性和线性化性质。

## 第 5 章 真实 Rust 实现

### 5.1 Sealor 方法论迁移

从 FR 到真实 Rust 的核心方法论有三要素：

1. **语法保留**：生成的 Rust 源代码结构原样保留（语法可被 rustc 解析）
2. **占位符插入**：对未完成位置插入 `holeval()`（返回类型兼容的泛型占位值），不引入类型错误
3. **委托检查**：类型、借用、生命周期推理委托给 rustc（基于 Rust 1.95.0）

### 5.2 部分程序检查器

首个 Rust 部分程序检查器的工作流程：
- 从流式 API 接收部分生成的 Rust 代码前缀
- Sealor 将其转换为语法完整的 Rust 程序（插入占位符、闭合未闭合的结构）
- rustc 增量编译检查密封后的程序
- 诊断信息中的源码位置映射回部分程序中的对应位置
- 将映射后的诊断返回给 LLM

对 rustc 的最小补丁：访问类型推理结果以识别已定义的标识符和可用函数/方法及预期参数数量。补丁随实现开源。

### 5.3 增量编译优化

现代编译器支持增量编译（如 rustc 的 incremental compilation），可显著加速多次编译。Generative Compilation 利用此特性：每次前缀检查只需重新编译变化的部分，而非从头编译整个密封程序。

### 5.4 代码仓库结构

官方实现仓库（eth-sri/generative-compilation）包含 6 个 commits 的结构：
- `bindings/`：语言绑定
- `crates/`：Rust crate 实现
- `llm/`：LLM 推理代码和评估结果
- `mechanization/`：Lean 机械化证明
- `patches/`：Rust 编译器补丁

## 第 6 章 实验评估

### 6.1 实验设置

**模型**：7 个前沿 LLM，涵盖闭源（GPT 5.3 Codex、Claude Opus 4.8、Gemini 3.5 Flash）和开源（Qwen 3.5 397B、Qwen 3.5 9B、GLM 5.2、Kimi K2.7 Code）。温度统一设为 0.6（Opus 的 API 不支持设置）。

**任务**：

| 任务 | 来源 | 数量 | 平均函数体数 | 平均字符数 | 测试用例 |
|------|------|:----:|:----------:|:---------:|:-------:|
| C-to-Rust 翻译 | CRUST-Bench [20] 筛选 | 20 个困难任务 | — | — | — |
| UpdatedAPI | 精选 100 个热门 crate | 30 个单文件任务 | 5.9 | 3,142 | 3.23 |

C-to-Rust 翻译任务从 CRUST-Bench 的 100 个翻译任务中筛选出 GPT 5.3 和 Opus 4.8 零样本无法通过的 20 个困难任务。

### 6.2 主实验结果

**编译通过率**（聚合所有模型）：

| 方法 | 通过率范围 | 提升 |
|------|:---------:|:----:|
| Post-generation Compilation (PC) | 79.2–80.5% | 基线 |
| Generative Compilation (GC) | **84.9–87.0%** | +5.7–6.5 pp |

**重启效率分析**：两种方法在前几次重启内快速提升，之后趋于平稳。从第 4 次重启开始 GC 超越 PC。在 7 次重启处，GC 比 PC 多解决 4.4 个百分点的生成。将 restart budget 从 15 增至 20 次，GC 提高至多 2.1 pp（84.9% → 87.0%），PC 提高至多 1.3 pp（79.2% → 80.5%）。

### 6.3 错误种类分析

GC 在生成过程中检测到的错误种类分布：

| 错误类型 | Translation | UpdatedAPI |
|---------|:----------:|:----------:|
| E0308 (类型不匹配) | 38.2% | 37.4% |
| E0609 (未知字段) | 9.5% | 5.4% |
| E0061 (参数数量错误) | — | 18.1% |
| E0107 (泛型参数数量错误) | — | 7.2% |
| E0559 (不存在的初始化字段) | — | 7.2% |
| E0502 (借用冲突) | 包含在 Borrow & Lifetime 类 | — |

Translation 任务中最频繁的错误是类型不匹配（E0308, 38.2%）和访问未知字段（E0609, 9.5%），反映了将 C 数据布局移植到 Rust 骨架中的困难。UpdatedAPI 任务中除了 E0308（37.4%）外，参数数量和泛型参数数量的错误（E0061 18.1%、E0107 7.2%）也频繁出现，这是 API 更新后的典型模式。

### 6.4 端到端案例分析

在 rustls-webpki 的 API 更新任务中（Claude Opus 4.8），GC 在生成第 4 行时就捕获了 E0716（临时值在借用期间被 drop）错误。模型收到诊断后，将临时的 CertificateDer 绑定到局部变量，延长了其生命周期。反馈在 32 行函数的第 4 行即提供，防止了后续大量潜在浪费性生成。

## 第 7 章 局限性与延伸阅读

### 7.1 局限性

- **仅支持 Rust**：当前实现限于 Rust。方法论通用，但迁移到其他语言需要相应语言的 partial syntax 定义和 sealor 实现
- **Soundness 覆盖有限**：选择性可靠性仅在 statement boundaries 处保证。对于非边界位置的死胡同前缀，sealor 可能产生虚假的编译通过
- **额外编译开销**：尽管增量编译缓解了问题，每个前缀检查仍需要一次编译。对于极大型生成，编译时间可能成为瓶颈
- **最小编译器补丁依赖**：访问类型推理结果需要对 rustc 做微小修改，虽然补丁极小但维护成本存在

### 7.2 相关工作

- **Constrained Decoding** [23, 31, 39, 43, 53]：GC 与之互补——GC 不试图替代 constrained decoding，而是提供一种黑盒兼容的替代方案。两者可组合使用
- **Compiler Feedback Loops** [5, 9, 56]：post-generation feedback 是当前主流实践，GC 将其扩展到生成过程中
- **Program Synthesis** [44]：形式化合成方法提供更强的正确性保证，但需要完整的规范输入
- **Type Error Localization** [48, 55]：专注于诊断质量的提升，GC 利用现有编译器而非重新实现诊断

### 7.3 未来方向

- 将 sealor 方法论扩展到 Rust 标准库和 unsafe 代码
- 探索 GC 与 constrained decoding 的组合使用
- 支持更多具有丰富静态语义的语言（Haskell、OCaml、Scala 等）
- 利用增量编译和 lazy checking 进一步降低编译开销
- 改进 sealor 的 soundness 覆盖范围（超越 statement boundaries）

### 7.4 延伸阅读

- **FR（Featherweight Rust）**[41]：原文形式化，提供了本工作的理论基础
- **RustBelt**[19]：Rust 更全面的语义基础
- **Synchromesh**[43]：代表性 constrained decoding 工作
- **TreeCoder**[45]：2026 年最新 constrained decoding 进展
- **CRUST-Bench**[20]：C-to-Rust 翻译基准，本文实验的基础
- **BaxBench**[54]：LLM 代码生成安全性和正确性基准