# Auto Research / AI Scientist / Research Agent / RSI 及相关 Benchmark 调研

> 面向项目：科学计算领域（以计算生物学问题为载体）的 Auto Research Benchmark 设计
> 调研日期：2026-10-20。文中数字均来自论文摘要、正文或官方页面；无法核实的数字不写。

---

## 1. 概述与分类

近两年，“让 AI 自主做研究”从概念演示变成了可以量化评测的方向。相关工作可按**自主程度**和**评测对象**粗略分为四类：

| 类别 | 关注点 | 典型代表 |
|---|---|---|
| **Research Agent（研究智能体）** | 在给定研究子任务（数据分析、复现、写代码）中能否正确完成 | ScienceAgentBench, DiscoveryBench, CORE-Bench, BixBench, MLAgentBench |
| **Auto Research（自动化研究工程）** | 端到端完成开放式研究或工程项目，并以量化指标衡量“研究效果” | MLE-bench, MLGym, RE-Bench, ASI-Bench, Terminal-Bench Science |
| **AI Scientist（AI 科学家系统）** | 从 idea 到实验、写论文、评审的完整闭环系统 | Sakana *The AI Scientist* (v1/v2) |
| **RSI（递归自我改进）相关评测** | AI 能否加速 AI 研发本身，属于前沿安全评估中的“AI R&D 能力”维度 | RE-Bench (METR), MLE-bench, PaperBench |

另有一类**知识/技能型科学基准**（SciCode, LAB-Bench）不要求长程自主执行，但常作为科学能力的底座评测，本文一并对比。

**本项目最相关的两项参考：**

- **ASI-Bench**（arXiv 2608.17271，2026-08）：60 个项目级研究任务，覆盖 11 个科学领域，每个任务在相同目标、数据和评价标准下逐级撤除人类方法指导（B1→B4），用于衡量“自主研究”能力。
- **Terminal-Bench Science**（Harbor 任务格式）：把真实科研工作流打包成可在容器中执行、由确定性程序验证的终端任务。v0.1.0 共 70 题，发布时目标是前沿模型只能解决约 10–20%。

---

## 2. 对比总表

| 名称 | 年份 | 领域 | 任务规模与形式 | 实验环境 | 工具调用 | 评价指标 | 自动评分机制 |
|---|---|---|---|---|---|---|---|
| **ASI-Bench** | 2026 | 11 个科学领域（数学、物理、化学、生物、天文、材料、地学、医学与生物统计、CS、机器人、电气） | 60 个项目级研究任务；给定目标、数据和必需产出（预测场、谱、图、分析结果、代码）；B1–B4 四级指导 | 隔离沙箱，端到端执行；主实验不开放外部工具 | 以通用 coding agent harness 为载体（Codex、Claude Code、OpenHands 等） | 各任务加权得分的宏平均（0–100），按指导等级分别报告 | 用 `generate_gt.py` 生成参考结果，再经有效性检查、评测门槛（gate）和加权 scorer 打分；做了 scorer 重放与稳定性验证 |
| **Terminal-Bench 2.0** | 2025 | 通用终端任务（软件、系统、安全、科学计算等） | 89 个人工审核的难题，每题包含独立环境、参考解和测试 | Docker 容器（Harbor） | 终端 shell，可完全自由操作 | 任务解决率（pass rate） | `tests/test.sh` 写出 reward |
| **Terminal-Bench Science** | 2026 | 生命 19、物理 17、数学 17、工程 9、地球 8（v0.1.0） | 70 个科研工作流任务（目标 100+）；产出须是数值结果、文件、统计拟合等可检查结果 | Harbor：agent 容器与 verifier 容器分离，verifier 不联网 | 终端 shell；作者上限 5 h，发布前统一提高到 8 h | Pass@1 解决率，二元 0/1，无部分分 | `tests/test.sh` 写 `/logs/verifier/reward.txt`（或 `reward.json`），值为 0/1；诊断信息另写 `metrics.json` |
| **MLE-bench** (OpenAI) | 2024 | ML 工程 | 75 个 Kaggle 竞赛 | 容器内运行，提供 GPU | 自由的代码执行（AIDE 等脚手架） | 获得奖牌（铜牌及以上）的竞赛比例；o1-preview + AIDE 达到 16.9% | 按各竞赛原始指标在本地测试集上打分，再与 Kaggle 公开排行榜比较换算奖牌 |
| **RE-Bench** (METR) | 2024 | AI 研发工程 | 7 个开放式 ML 研究工程环境；61 名专家共 71 次 8 小时尝试作为人类基线 | 带 GPU 的虚拟机 | 自由的 shell 和代码执行 | 归一化得分（起始方案为 0，参考方案为 1）；比较 2 h、8 h、32 h 预算下的结果 | 每个环境有确定性评分函数（如 loss、运行时间） |
| **PaperBench** (OpenAI) | 2025 | ML 论文复现 | 20 篇 ICML 2024 Spotlight/Oral 论文，层级 rubric 共 8,316 个可评分叶节点 | 带 GPU 的容器，agent 从零写代码并提交仓库 | 自由的 shell、代码执行和联网检索（禁止访问原作者代码） | 复现得分（rubric 加权）；最佳 Claude 3.5 Sonnet (New) 为 21.0%，未超过 ML 博士 | LLM judge 按 rubric 评分；另建 JudgeEval 评测 judge 本身 |
| **MLAgentBench** | 2023 | ML 实验 | 13 个任务（从 CIFAR-10 到 BabyLM） | 本地工作目录 + starter code | 读写文件、执行脚本、检查输出等动作 | 成功率：相对 starter baseline 提升 ≥10% 即算成功；最佳 Claude 3 Opus 平均 37.5% | 运行任务自带的评测脚本 |
| **MLGym** (Meta) | 2025 | CV、NLP、RL、博弈论 | 13 个开放式 AI 研究任务 | Gym 式环境，可用于训练 RL agent | shell、文件编辑、`validate` 命令 | Performance profile 和 AUP；分别报告 Best Submission@4 和 Best Attempt@4 | 各任务指标自动计算，再跨任务聚合 |
| **ScienceAgentBench** | 2024 | 生物信息、计算化学、GIS、心理与认知神经科学（4 个学科） | 102 个任务，来自 44 篇论文；要求输出一个独立 Python 程序 | conda/Docker 执行环境 | 生成代码（可配合 self-debug 或 OpenHands） | 最佳 o1-preview 为 42.2%；同时统计执行结果和成本 | 运行程序后用任务专属脚本检查输出；图表类任务用 GPT-4o 判断 |
| **DiscoveryBench** | 2024 | 6 个领域（社会学、工程等），数据驱动发现 | 264 个真实任务和 903 个合成任务；输入数据集与目标，输出假设 | 数据集 + 代码执行 | 用代码分析数据 | 假设匹配度；最佳系统约 25% | 按 facet（上下文、变量、关系）由 LLM 判定匹配 |
| **CORE-Bench** | 2024 | CS、社会科学、医学 | 270 个任务，来自 90 篇论文（CodeOcean 胶囊）；Easy/Medium/Hard 三级 | 虚拟机运行作者代码 | shell 和代码执行 | 准确率；Hard 级最佳约 21% | 按问题比较复现出的数值与参考答案（带容差） |
| **SciCode** | 2024 | 16 个自然科学子领域 | 80 个主问题拆为 338 个子问题，函数级代码生成 | 无交互环境（离线执行测试） | 不涉及（纯生成） | 主问题和子问题通过率；最佳 Claude 3.5 Sonnet 4.6% | 科学家标注的测试用例做数值比对 |
| **LAB-Bench** (FutureHouse) | 2024 | 生物学研究 | 2,400 余道选择题：文献、图表、数据库、序列操作等 | 无（部分题需要检索或序列工具） | 可选（数据库、序列工具） | Accuracy、Precision、Coverage（允许回答“信息不足”）；与人类专家比较 | 选择题精确匹配 |
| **BixBench** (FutureHouse) | 2025 | 生物信息学 | 50 余个真实分析场景（capsule）及配套开放问答；论文不同版本的题数不同 | 预构建 Docker 内的 Jupyter，包含 Python/R/Bash 生信包 | `edit_cell`、`list_workdir`、`submit_answer` | 开放题准确率（论文报告前沿模型约 17%）；选择题分“允许拒答/不允许拒答”两种设置 | 开放题由 LLM judge（Claude 3.5 Sonnet）对照参考答案给 0/1 分 |
| **The AI Scientist** (Sakana) | 2024/2025 | ML（扩散模型、语言模型、学习动力学） | 系统，非 benchmark：idea、代码、实验、作图、写论文、模拟评审 | 基于模板代码的本地实验 | 代码执行、文献检索、LaTeX | 自动 reviewer 评分（据称接近人类）；每篇论文成本低于 15 美元 | LLM reviewer（参照 NeurIPS 评审标准）；v2 有一篇论文通过了 ICLR 2025 workshop 同行评审 |

---

## 3. 分维度分析

### 3.1 任务设计

- **粒度谱系**：从函数级（SciCode）、问答级（LAB-Bench、BixBench）、单程序级（ScienceAgentBench）、复现级（CORE-Bench、PaperBench），到项目级（MLE-bench、RE-Bench、ASI-Bench）。越往项目级，越接近真实研究，评分也越难。
- **任务来源**：多来自已发表论文或竞赛（Kaggle、ICML 论文、CodeOcean），优点是有可信参考答案，缺点是有**训练数据污染**风险。
- **指导分级**：ASI-Bench 在**同一项目**内逐级撤掉方法指导（B1 给出完整方法 → B3 只给目标）。18 个配置的平均分从 50.91 降到 26.62，最大降幅出现在 B1→B2。可见“自己实现方法”比“自己选方法”更难。B4 加入无关干扰信息，用来测试鲁棒性。这一设计可以直接借鉴。
- **难度校准**：Terminal-Bench Science 明确要求发布时前沿模型解决率约 10–20%，难度要来自真实复杂度（多阶段流程、专家知识），不能对着模型弱点做对抗式筛选。
- **领域覆盖**：主流 Auto Research 基准集中在 ML 本身（MLE-bench、RE-Bench、MLGym、PaperBench）。生物方向以问答或单步分析为主（LAB-Bench、BixBench），缺少“计算生物 + 数值计算 + 项目级 + 确定性评分”的组合。

### 3.2 实验环境

- **容器化成为主流**：Terminal-Bench/Harbor、MLE-bench、BixBench 均使用 Docker 镜像固定依赖。Harbor 的任务目录是自包含的，不依赖具体框架：`task.toml`、`instruction.md`、`environment/Dockerfile`、`solution/solve.sh`、`tests/test.sh`。
- **验证环境隔离**：Terminal-Bench Science 要求 `environment_mode = "separate"`：agent 容器先销毁，verifier 再在只装有声明产物和测试依赖的干净容器里运行，并且不联网。这样从机制上避免 agent 读到或篡改隐藏测试。
- **算力需求**：RE-Bench、MLE-bench、PaperBench 依赖 GPU 和长时运行（数小时到 24 h），ASI-Bench 平均每个任务累计超过 35 小时 agent 执行。成本限制了复现和多次采样。

### 3.3 工具调用

- 主流做法是给**通用 shell 和代码执行**能力（Terminal-Bench、RE-Bench、MLE-bench、ASI-Bench），由 harness（Codex、Claude Code、OpenHands、mini-SWE-agent 等）决定具体工具。评测的是“模型 + harness”组合，而不只是模型。
- 少数基准给**受限工具集**（BixBench 的三个 notebook 工具、MLAgentBench 的动作空间、MLGym 的 `validate`），便于分析行为，也降低了作弊空间。
- 联网策略各不相同：PaperBench 允许联网但屏蔽原作者仓库；ASI-Bench 主实验不开放外部工具；Terminal-Bench Science 的 verifier 禁止联网。

### 3.4 评价指标

| 指标类型 | 例子 | 优点 | 缺点 |
|---|---|---|---|
| 二元通过率 | Terminal-Bench 系列、SciCode、CORE-Bench | 清晰，适合排行榜 | 难度高时弱模型全为 0，区分度差 |
| 归一化连续分 | RE-Bench（0=起始方案，1=参考方案）、ASI-Bench 加权分 | 能反映“做得多好”，体现研究效果 | 需要设计归一化锚点 |
| 相对人类/排行榜 | MLE-bench 奖牌、RE-Bench 人类专家对比 | 有现实意义 | 依赖外部基线，基线可能过时 |
| 阈值成功率 | MLAgentBench（≥10% 提升） | 简单 | 阈值选择任意 |
| 聚合曲线 | MLGym AUP、Best Attempt vs Best Submission | 跨任务可比，区分“探索到的上限”和“最终交付” | 解释成本高 |
| 效率与成本 | ScienceAgentBench 成本、RE-Bench 时间预算曲线、ASI-Bench 每次运行成本 | 衡量研究效率 | 常被忽略或只作附注 |

### 3.5 自动评分机制

1. **确定性程序验证**（Terminal-Bench、SciCode、CORE-Bench、MLE-bench、RE-Bench）：将数值与参考值比较，带容差或使用指标函数。可复现性最好。
2. **参考生成 + 加权 scorer**（ASI-Bench）：先用脚本生成 ground truth，再经有效性检查和 gate，最后对多个产出加权打分，并做 scorer 重放验证。
3. **LLM-as-judge**（PaperBench rubric、BixBench 开放题、DiscoveryBench 假设匹配、ScienceAgentBench 图表、AI Scientist 评审）：适合开放产出，但存在偏差和不稳定，因此需要额外的 judge 评测（如 PaperBench 的 JudgeEval）。
4. **奖励接口标准化**：Harbor 统一约定写 `/logs/verifier/reward.txt`（单标量）或 `reward.json`（多维），使任务可以跨框架复用。Terminal-Bench Science 进一步把它限定为 0/1，诊断信息写到单独文件。

---

## 4. 现有工作的局限

1. **LLM-judge 评分不稳定**：PaperBench、BixBench、DiscoveryBench、AI Scientist 依赖 LLM 判定，结果受 judge 模型版本、提示词和位置偏差影响，而且 judge 与被测模型可能同源，产生自我偏好。AI Scientist 的自动 reviewer“接近人类”只是统计意义上的一致，不能替代同行评审。
2. **隐藏测试泄漏与奖励作弊**：如果测试文件与 agent 在同一容器，agent 可能读取、修改测试或硬编码答案。基于公开论文或竞赛的任务还存在训练数据污染。Harbor 的分离 verifier 是针对这一问题的工程解法，但很多旧基准没有这层隔离。
3. **只看最终正确性，不看研究效果和效率**：二元 pass/fail 无法区分“接近正确”和“完全失败”，也不衡量方法是否合理、实验是否充分、用时和成本是多少。ASI-Bench 的数据显示花费更多（单次约 684–2,728 美元）并不能稳定换来更高分。
4. **二元 reward 难以区分弱模型**：Terminal-Bench Science 的目标解决率是 10–20%，SciCode 最佳成绩只有 4.6%。在这种难度下，7B 级小模型基本全为 0，无法用来比较训练改进。
5. **环境难复现**：依赖 GPU、长时运行、外部数据下载或网络，不同硬件下数值结果有差异。CORE-Bench 本身就说明了“复现已发表代码”有多难。
6. **方差大、样本少**：ASI-Bench 指出许多配置跨运行方差大，部分结果只有单次运行；60–100 个任务的规模下置信区间很宽。
7. **科学计算和计算生物覆盖少**：Auto Research 类基准以 ML 为主；生物类基准以问答或单步分析为主，缺少需要数值求解（ODE/PDE、优化、随机模拟、序列统计）且结果可确定性验证的项目级任务。
8. **评测成本高**：项目级任务需要多小时的 agent 运行和大量 token，限制了多次采样和小团队复现。
9. **harness 与模型混淆**：得分同时受脚手架、工具和时间预算影响，跨论文横向比较需要谨慎。

---

## 5. 对本项目的启示

- **采用 Harbor 兼容格式**：每题包含 `task.toml`、`instruction.md`、`environment/Dockerfile`、`solution/solve.sh`、`tests/test.sh`，可直接在 Harbor 或 Terminal-Bench 生态中运行。
- **确定性数值 verifier**：以计算生物问题为载体（如动力学 ODE 参数拟合、序列比对统计、种群遗传模拟、蛋白或代谢网络稳态求解），用带容差的数值比对评分，不使用 LLM judge。
- **隐藏测试隔离**：参考答案和测试数据只放在 verifier 镜像中，verifier 采用 separate 模式且不联网。尽量使用随机种子或程序化生成的数据，降低泄漏和污染风险。
- **连续分 + 二元分并存**：`reward.json` 同时输出 `pass`（0/1，按 Harbor 约定）和连续分（例如按误差分段的部分分），避免二元 reward 无法区分弱模型。
- **多维指标**：正确性（数值是否达标）、研究效果（误差或指标相对 baseline 和参考解的归一化提升，参考 RE-Bench）、效率（用时、步数、token）、可复现性（verifier 是否能用 agent 留下的代码重新跑出结果）。
- **指导分级**：借鉴 ASI-Bench 的 B1–B3 设置，同一题提供“给方法 / 只给方向 / 只给目标”三种 instruction，测量自主性。
- **小模型基线**：用 7B 级开源模型加简单 agent 循环跑出基线，配合参考解（oracle）和空解（nop）验证 verifier 的正确性与区分度，确保分数分布不会全部堆在 0。

---

## 6. 参考文献

1. Zhou, J., Sun, Z., et al. *ASI-Bench: At the Dawn of Artificial Superintelligence.* arXiv:2608.17271, 2026. https://arxiv.org/abs/2608.17271
2. Terminal-Bench Science（任务仓库与贡献指南）. https://github.com/harbor-framework/terminal-bench-science ；征稿说明 https://www.tbench.ai/news/tbsci-contribution-call ；排行榜 https://vals.ai/benchmarks/terminal-bench-science
3. Harbor 任务格式文档. https://docs.harborframework.com/tasks/overview
4. Merrill, M. A., et al. *Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces.* ICLR 2026 / arXiv:2601.11868. https://github.com/harbor-framework/terminal-bench
5. Chan, J. S., et al. *MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering.* arXiv:2410.07095, 2024. https://arxiv.org/abs/2410.07095
6. Wijk, H., et al. (METR). *RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts.* arXiv:2411.15114, 2024. https://arxiv.org/abs/2411.15114
7. Starace, G., et al. *PaperBench: Evaluating AI's Ability to Replicate AI Research.* arXiv:2504.01848, 2025. https://arxiv.org/abs/2504.01848
8. Huang, Q., et al. *MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation.* arXiv:2310.03302, 2023. https://arxiv.org/abs/2310.03302
9. Nathani, D., et al. *MLGym: A New Framework and Benchmark for Advancing AI Research Agents.* arXiv:2502.14499, 2025. https://arxiv.org/abs/2502.14499
10. Chen, Z., et al. *ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery.* arXiv:2410.05080, 2024. https://arxiv.org/abs/2410.05080
11. Majumder, B. P., et al. *DiscoveryBench: Towards Data-Driven Discovery with Large Language Models.* arXiv:2407.01725, 2024. https://arxiv.org/abs/2407.01725
12. Siegel, Z. S., et al. *CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark.* arXiv:2409.11363, 2024. https://arxiv.org/abs/2409.11363
13. Tian, M., et al. *SciCode: A Research Coding Benchmark Curated by Scientists.* arXiv:2407.13168, 2024. https://arxiv.org/abs/2407.13168
14. Laurent, J. M., et al. *LAB-Bench: Measuring Capabilities of Language Models for Biology Research.* arXiv:2407.10362, 2024. https://arxiv.org/abs/2407.10362
15. Mitchener, L., et al. *BixBench: a Comprehensive Benchmark for LLM-based Agents in Computational Biology.* arXiv:2503.00096, 2025. https://arxiv.org/abs/2503.00096
16. Lu, C., et al. *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery.* arXiv:2408.06292, 2024. https://arxiv.org/abs/2408.06292
17. Yamada, Y., et al. *The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search.* arXiv:2504.08066, 2025. https://arxiv.org/abs/2504.08066
