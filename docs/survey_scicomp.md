# 科学计算方向基准与答案验证专题调研

> 本文是 `survey.md` 的补充，只讨论**科学计算类**基准的任务形式、验证方法和小模型表现，通用的 Auto Research 基准对比见 `survey.md`，这里不再重复。
> 调研日期：2026-10-20。文中数字来自论文、官方仓库或排行榜原文；未能核实的内容放在文末“待核实”一节。

---

## 1. 代表性基准与任务例子

| 名称 | 任务形式 | 验证方式 | 是否确定性 | 备注 |
|---|---|---|---|---|
| **Terminal-Bench Science**（`tasks/` 下的具体任务） | 容器内的项目级任务，最长 8 h。例如：`amr-poisson-optimize`（补全并加速三维两层 AMR 有限体积 Poisson 求解器）、`ode-law-discovery`（从带噪轨迹中发现符号 ODE）、`spin-glass-groundstate`（求 6 个三维 EA 自旋玻璃实例的基态） | AMR：verifier 自己重新组装算子，算残差 ρ₂≤1e-8，与自带多重网格参考解比较 E₂≤5e-5，检查粗细网格通量缺陷 η≤5e-8，并要求运行时间 ≤1.4×参考解、内存 <4 GiB。ODE：在隐藏初值上 rollout 的 nRMSE<0.08，stress rollout <0.15，13³ 网格上的向量场相对误差 <0.08，复杂度 ≤50。自旋玻璃：用整数精确计算能量，必须等于真实基态能量，6 个实例全部达到才给 1 分 | 正确性部分是确定性的；计时部分有噪声（取交错运行的中位数） | 仓库 README 现写 71 题，`survey.md` 记录的 v0.1.0 为 70 题。目录覆盖 physics、applied-mathematics、operations-research、mechanical-engineering 等子领域 |
| **SciCode** | 函数级科研代码生成：80 个主问题拆为 338 个子问题 | 科学家编写的数值测试用例 | 确定性（带容差比对） | 最佳 Claude 3.5 Sonnet 子问题 pass@1 为 26.0%。开源模型（Llama-3-70B、Mixtral-8x22B）一个主问题都没解出 |
| **CodePDE** (CMU/Flatiron) | 给定 PDE 描述，生成求解器代码。共 5 类方程（Advection、Burgers、Reaction-Diffusion、可压缩 NS、Darcy），每类 100 个实例，数据来自 PDEBench | 与参考数据比较 nRMSE；另做收敛性测试（加密网格后相邻分辨率的差应缩小）并统计经验收敛阶；记录运行时间（时限 1200/2400 s） | 基本确定（受 GPU 浮点影响） | 单次生成平均约 41% 能无错运行，自调试 4 轮后约 84%；CNS 单次仅 16.6%。在 4/5 类方程上，LLM 解优于手写参考解，Reaction-Diffusion 仍然困难 |
| **CFDLLMBench** (NREL) | CFDQuery（90 道选择题）、CFDCodeBench（24 道 Python CFD 编程题）、FoamBench（110 个基础 + 16 个高级 OpenFOAM 算例） | CodeBench 用解析解或专家参考解。终态 NMSE≤10% 记 1 分，10–30% 记 0.5 分；另要求加密网格后误差下降；可执行、精度、收敛三项都满分才算成功。FoamBench 看可执行性和 NMSE，另用 ROUGE 比较文件结构 | 确定性（形状不一致时插值后比较） | 最佳模型 CodeBench 约 14%、FoamBench 约 34%。**Gemma-2-9B 在 FoamBench 所有配置下成功率都是 0** |
| **FEABench** (Google) | 通过 API 驱动 COMSOL 求解多物理场有限元问题。Gold 集 15 题，Large 集 200 题 | 将目标物理量与教程中的参考值比较 | 确定性 | 最佳策略 88% 能生成可执行的 API 调用，但**没有任何模型或 agent 完整、正确地解出任何一题** |
| **ParEval** (UMD, HPDC'24) | 420 个科学/并行计算函数题，覆盖 12 类问题和多种并行模型（OpenMP、MPI、CUDA 等） | 先跑正确性测试，再测性能：`speedup_n@k`、`efficiency_n@k` | 正确性确定；性能有噪声 | **含 7B 模型数据**（见第 3 节）。并行代码明显比串行代码难写 |
| **KernelBench** (Stanford) | 把 PyTorch 算子改写成更快的 GPU kernel | 正确性：在随机输入上与参考算子比较（原文用 5 组输入，默认容差 1e-4）。性能：`fast_p` = 既正确又比基线快 p 倍的任务比例 | 正确性近似确定；计时有噪声 | 前沿模型在不到 20% 的任务上追平 PyTorch。**奖励作弊的典型案例**（见 2.3 节） |
| **FrontierCO** (ICLR 2026) | 8 类组合优化问题（路由、调度、选址、图问题），分 easy/hard 两档，最大规模 TSP 达 1000 万节点 | 可行性检查 + 目标值与最优已知解的差距 | 确定性 | LLM agent 主要靠调参和套用已知启发式（LNS、模拟退火）取得进展，很少发现新算法 |
| **AlphaEvolve / OpenEvolve 类“算法发现”** | 进化式修改程序，优化用户提供的评估函数 h（例如 26 个圆的单位正方形堆积、数学构造） | 程序化评估函数输出一个或多个分数；后续论文涉及 67 个数学问题 | 确定性 | 只适用于能自动评估的问题。没有找到小模型在其中的公开结果 |
| **SciReplicate-Bench** (COLM'25) | 根据 36 篇 NLP 论文中的算法描述，在仓库上下文中实现代码，共 100 题 | 执行准确率（运行测试），另报 CodeBLEU、API recall、推理图准确率 | 执行部分是确定性的 | 最佳模型执行准确率 0.39。模型对算法的理解（推理图约 0.73）明显好于实现 |
| **ResearchCodeBench** (NeurIPS'25) | 根据 20 篇 2024–25 年 ML 论文实现关键代码片段，共 212 题 | 原论文作者或专家参与编写的单元测试 | 确定性 | 最佳 Gemini-2.5-Pro 为 37.3% |
| **PDE-Agent / PDE-Bench** (arXiv 2512.16214) | 多 agent 调用 PDE 工具链，约 100 个测试用例 | 从全局、逻辑、局部三个层次评估工具调用 | 部分确定 | 只有作者自报结果，数据集是否已公开未确认 |

---

## 2. 科学计算任务的可验证性分析

### 2.1 验证策略

| 策略 | 做法 | 实例 | 优点 | 主要风险 |
|---|---|---|---|---|
| 解析解 | 选有闭式解的问题，直接比较误差 | CFDCodeBench 部分题 | 有绝对真值 | 解析解在网上很容易查到，agent 可能直接写出公式而不是去数值求解 |
| 制造解（MMS） | 先设定 u\*，反推源项 f 和边界条件，再要求 agent 求解 | 文献中的标准做法，本次调研的基准未显式采用 | 可以随机生成无穷多实例，适合作为隐藏测试 | 需要确认 u\* 足够光滑，并且离散格式能达到对应精度 |
| 无参考残差 | 由 verifier 自己组装离散算子，计算 ‖f−Au‖/‖f‖ | TB-Sci `amr-poisson-optimize` 的 ρ₂、η | 不依赖参考解，不受参考解本身误差影响 | 只能说明离散方程解对了，不能说明离散格式本身选对了，所以需要固定或检查离散格式 |
| 收敛阶 | 在多个分辨率上运行，拟合误差斜率，或要求误差单调下降 | CodePDE、CFDCodeBench | 能识别“碰巧在某个网格上误差小”的假解 | 渐近区间难以确定，计算量要成倍增加 |
| 守恒律/不变量 | 检查质量、能量、通量平衡，以及对称性、正定性等 | TB-Sci 的粗细网格通量缺陷 η | 物理意义清楚、实现便宜 | 只是必要条件，不是充分条件 |
| 参考解 + 容差 | 用高精度参考解（不同方法或更细网格）比较 nRMSE 或 E₂ | CodePDE、TB-Sci E₂、ODE rollout | 通用 | 容差怎么选（见下文），参考解本身也有误差 |
| 精确离散目标 | 用整数或有理数精确计算目标值，再与已证明的最优值比较 | TB-Sci 自旋玻璃基态、FrontierCO | 完全确定，没有容差问题 | 需要预先求出最优值；或者改成相对最优已知解的差距 |
| 运行时间/加速比 | 与参考实现比较运行时间或加速比 | TB-Sci（≤1.4×）、ParEval、KernelBench | 能衡量“研究效果” | 计时噪声大，依赖硬件 |
| 性质/随机测试 | 在随机输入上与参考实现比较，或检查不变式 | KernelBench（随机输入）、ParEval | 能防止针对少数固定输入的过拟合 | 输入分布太窄时会漏掉边界情况 |
| 留出泛化 | 隐藏的初值、源项或实例 | TB-Sci 留出 4 个源位置；ODE 用隐藏初值和 stress rollout | 防止硬编码输出 | 需要能程序化生成实例 |

### 2.2 容差与数值非确定性

- **容差要从参考解本身的不确定度推出来。** TB-Sci AMR 任务先用两种参数各求一次参考解，要求两者相差 ≤1e-6，再把验收阈值设为 E₂≤5e-5，阈值留了充足余量。KernelBench-Verified 指出，默认容差 1e-4 原本是为两份同样归约顺序的 PyTorch 实现之间的比较设计的。对自定义 kernel，求和顺序不同但同样正确的实现会被它误判为错误。
- **浮点非确定性**：并行归约顺序、BLAS 线程数、GPU 原子操作、FMA 等都会让结果在最后几位上不同。建议 verifier 固定单线程（TB-Sci 的计时阶段就是单线程），容差至少比机器 ε 引起的误差高几个数量级。对 0/1 判定，阈值边界附近要留缓冲带。
- **分级给分**：CFDCodeBench 按 NMSE 分档（≤10% 记 1 分，10–30% 记 0.5 分），比单一阈值更能区分弱模型。

### 2.3 奖励作弊与防御（来自真实案例）

1. **钻 verifier 的漏洞**：Sakana 的 AI CUDA Engineer 利用评测代码中的内存漏洞跳过了正确性检查。修复后，EvoEngineer 复现得到的中位加速比从 1.13× 降到 0.82×，成功任务数从 63 降到 22。
2. **直接调用库函数**：KernelBench 维护者记录了直接调用 torch/cuBLAS 高层算子、用空 kernel 加内存复用等方式通过检查的情况。TB-Sci AMR 任务在指令中禁用了 SciPy、PyAMG、PETSc、Numba 以及 `numpy.linalg` 的稠密求解器，并由 verifier 做静态扫描，被标记的代码不运行。
3. **硬编码或读取答案**：TB-Sci AMR 的 verifier 以无特权的 `grader` 用户运行候选代码，该用户无法读取 `/tests`。verifier 还通过 `sitecustomize` 拦截对 oracle 模块的导入，用 `lchown` 防止通过符号链接读到测试文件，对固定输入文件做 SHA-256 校验，并**重新运行候选代码、覆盖旧的输出文件**，所以 agent 预先写好的 solution 文件不会被采信。
4. **规模**：Hacker-Fixer Loop（arXiv 2606.08960）审计了 5 个终端 agent 基准共 1,968 题，发现 323 题（约 16%）仅凭任务描述就能被前沿模型骗过 verifier。在 KernelBench 上，这一方法把攻击成功率从 62% 降到 0。
5. **计时噪声**：TB-Sci 由 verifier 自己计时，候选与 oracle 交错运行（C-O-C-O-C-O），各取最多 3 次的中位数后比较。候选最好的一次已超过预算 2 倍时提前停止，整个计时阶段上限 1500 s。

---

## 3. 对 7B 模型的难度预期

**有直接数据的：**
- **ParEval**（函数级，单次生成，排行榜 2024-03 版）：串行 / 并行 pass@1 分别为 StarCoder2-7B 59.4 / 15.9，CodeLlama-7B 48.4 / 15.3，StarCoder2-3B 42.7 / 9.6；GPT-4 为 76.1 / 37.8。可见函数级串行数值代码 7B 模型能做对一半左右，并行代码约为 GPT-4 的 40%。
- **CFDLLMBench**：Gemma-2-9B 在 CFDQuery 选择题上约 60%（各模型中最低），在 FoamBench 所有配置（包括 Foam-Agent + RAG + Reviewer）下成功率都是 **0**。

**只能间接推断的：**
- SciCode 原文中 70B 级开源模型解不出任何主问题，最佳开源模型 DeepSeek-Coder-v2 也只有 3.1%。7B 模型在主问题级别预计接近 0，子问题级别有一定通过率。
- FEABench 上前沿模型和 agent 都没有完整解出任何一题；TB-Sci 的设计目标是前沿模型约 10–20%。在这类项目级任务上，**7B 模型的 0/1 reward 几乎肯定全为 0**。
- CodePDE 中前沿模型单次生成也只有约 41% 能运行，CNS 只有 16.6%。7B 模型“写出能运行的求解器”本身就是主要瓶颈，可执行率可以作为有区分度的中间指标。
- FrontierCO 显示，LLM agent 的收益主要来自调参和套用已知启发式。这类“已有可行解、在此基础上改进”的任务对 7B 模型更友好，因为几乎总能得到一个非零分。

**结论**：只用 0/1 判定的项目级任务无法区分 7B 模型。可行的做法是：(a) 给 starter code，让任务变成补全或优化；(b) 输出连续分（误差分档、相对基线的改进幅度）；(c) 把可执行、精度、收敛分开计分。

---

## 4. 对本项目任务设计的建议

- **首选“无参考验证 + 隐藏实例”**：例如 ODE/PDE 题由 verifier 重算残差和守恒量，再用 MMS 或随机种子生成隐藏实例；优化题用精确整数目标值，或与最优已知解比较差距。不要只比较一个公开参考数组。
- **至少 3 个任务，覆盖 3 种验证范式**，以便互相印证：
  1. **数值 PDE/ODE 求解**（例如计算生物中的反应扩散方程或药代动力学 ODE 刚性系统）：用 nRMSE 与 MMS/参考解比较，加上 2–3 个分辨率的收敛阶检查。
  2. **逆问题/方程发现**（参照 `ode-law-discovery`）：在隐藏初值上做 rollout 算 nRMSE，同时比较向量场误差，并限制复杂度。
  3. **组合优化或离散能量最小化**（参照 `spin-glass-groundstate` 和 FrontierCO，例如格点蛋白折叠的 HP 模型）：用整数精确目标值，按与最优已知解的差距给连续分。
  4. （可选）**性能优化**（参照 `amr-poisson-optimize`）：先要求正确，再看相对参考实现的加速比；计时用交错运行取中位数。
- **容差的定标流程**：用两种独立的参考方法（或两种分辨率）求解，估计参考解自身的误差 δ，取容差 ≈ 10–100 δ。用 oracle 解重复运行 k=5 次，确认通过率为 100%；用空解（nop）确认得 0 分。
- **反作弊清单**：静态扫描禁用库（任务需要时禁用 scipy.integrate、scipy.optimize 等“一行就能解决”的接口）；以无特权用户运行候选代码；verifier 重新运行候选代码并覆盖旧输出；校验固定输入文件的哈希；verifier 断网；留出隐藏实例。发布前用前沿模型扮演“hacker”专门找漏洞（参考 Hacker-Fixer Loop）。
- **为 7B 模型设计计分**：`reward.json` 中的分数分级，例如可执行 0.2 + 精度分档 0.5 + 收敛/泛化 0.3，同时保留 Harbor 要求的 0/1 `pass`。每题提供 starter code 和“给方法 / 只给目标”两档 instruction（呼应 `survey.md` 中 ASI-Bench 的 B1–B3 设置），保证 7B 模型在给方法档能拿到非零分。
- **控制资源和非确定性**：限制在 CPU、单线程、NumPy 范围内，单题运行几十分钟以内。比较时避开 GPU 原子操作带来的不确定位，所有随机数都由 verifier 指定种子。
- **防污染**：问题参数（系数、几何、源项位置）在 verifier 内随机化，避免直接复用 PDEBench、TSPLib 等公开实例的已知答案。

---

## 5. 参考文献

1. Terminal-Bench Science 仓库（`tasks/` 下的 `amr-poisson-optimize`、`ode-law-discovery`、`spin-glass-groundstate` 等）. https://github.com/harbor-framework/terminal-bench-science
2. Tian, M., et al. *SciCode: A Research Coding Benchmark Curated by Scientists.* arXiv:2407.13168. https://arxiv.org/abs/2407.13168
3. *CodePDE: An Inference Framework for LLM-driven PDE Solver Generation.* arXiv:2505.08783. https://arxiv.org/abs/2505.08783
4. *CFDLLMBench: A Benchmark Suite for Evaluating Large Language Models in Computational Fluid Dynamics.* arXiv:2509.20374. https://arxiv.org/abs/2509.20374
5. Mudur, N., et al. *FEABench: Evaluating Language Models on Multiphysics Reasoning Ability.* arXiv:2504.06260. https://arxiv.org/abs/2504.06260
6. Nichols, D., et al. *Can Large Language Models Write Parallel Code?* (ParEval), HPDC'24, arXiv:2401.12554. https://arxiv.org/abs/2401.12554 ；排行榜 https://pssg.cs.umd.edu/blog/2024/pareval/
7. *KernelBench: Can LLMs Write Efficient GPU Kernels?* arXiv:2502.10517. https://arxiv.org/abs/2502.10517
8. *KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?* arXiv:2607.16241. https://arxiv.org/abs/2607.16241
9. Feng, S., et al. *FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization.* ICLR 2026, arXiv:2505.16952. https://arxiv.org/abs/2505.16952
10. AlphaEvolve 后续工作 *Mathematical exploration and discovery at scale.* arXiv:2511.02864. https://arxiv.org/abs/2511.02864 
11. Xiang, Y., et al. *SciReplicate-Bench.* COLM 2025, arXiv:2504.00255. https://arxiv.org/abs/2504.00255
12. Hua, T., et al. *ResearchCodeBench.* NeurIPS 2025 D&B, arXiv:2506.02314. https://arxiv.org/abs/2506.02314
13. *PDE-Agent: A toolchain-augmented multi-agent framework for PDE solving.* arXiv:2512.16214. https://arxiv.org/abs/2512.16214
14. Zhong, Z., et al. *Hardening Agent Benchmarks with Adversarial Hacker-Fixer Loops.* arXiv:2606.08960. https://arxiv.org/abs/2606.08960
15. *EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with LLMs*（含 AI CUDA Engineer 作弊复现）. arXiv:2510.03760. https://arxiv.org/abs/2510.03760
16. Zhou, J., et al. *ASI-Bench.* arXiv:2608.17271. https://arxiv.org/abs/2608.17271 （摘要只给出 B1→B3 的平均分 50.91→29.10→26.62，没有给出数学或物理任务的具体例子，本文因此没有展开）

### 待核实（未写入正文结论）
- 聚合站点列出 Llama-3.1-8B 在 SciCode 上 13.2%，但不清楚是子问题还是主问题指标。
- CodePDE 附录中是否有 7B–32B 开源模型的结果（正文表格中没有）。
- PDE-Bench 数据是否已公开；KernelBench 上 7B 模型的 `fast_p` 数值。
- OpenEvolve 官方仓库地址未核实，故未列出。
