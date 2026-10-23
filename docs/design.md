# BioCompute-ARB：设计文档

> 领域：**科学计算**，以计算生物学问题为任务载体。
> 调研见 [survey.md](survey.md)（总体）、[survey_scicomp.md](survey_scicomp.md)（科学计算）、
> [survey_compbio.md](survey_compbio.md)（计算生物学的可验证性）。

## 1. 设计目标

| 目标 | 做法 |
|---|---|
| 可运行 | 每条任务自带 starter workspace、公开样例与参考解；本地与 Linux 服务器均可一键运行 |
| 可自动评测 | 每条任务有独立 verifier（`tests/grade.py`），只依赖 numpy/scipy，确定性打分 |
| 可复现 | 隐藏数据由固定种子的生成脚本产生并随仓库冻结；结果记录任务内容哈希；Harbor 兼容 Dockerfile |
| 隔离隐藏测试 | Agent 只能看到 `environment/workspace`；`tests/`、`solution/`、`baselines/` 对 Agent 用户不可读；评测在新目录中进行，提交代码以低权限用户在子进程中执行，参考答案只留在评测父进程内存中 |
| 区分弱模型 | 每条任务同时给出二元 `reward`（是否解决）和连续 `score`（部分分），避免 7B 模型全部 0 分而失去区分度 |

### 为什么选“科学计算 × 计算生物学”

调研结论（[survey_compbio.md](survey_compbio.md)）：可验证性取决于**任务类型**而不是领域。
计算生物学中的算法实现（序列比对）、数值模拟（反应网络 ODE）、合成数据上的参数推断
（已知 ground truth）都可以做到**确定性、客观**的评分；只有开放式生物学解释才需要
LLM/人工评审。本项目三条任务全部落在前两类，且覆盖科学计算的三种验证范式：
**精确最优值**、**高精度参考解 + 容差**、**隐藏真值的参数恢复**。

## 2. 任务格式（兼容 Harbor / Terminal-Bench Science）

```
tasks/<task-id>/
  task.toml            # 元数据、难度、超时、资源（Harbor 字段）+ entrypoint
  instruction.md       # Agent 唯一可见的任务说明（输入、接口、工具、成功条件）
  guidance.md          # 方法指导，仅在 `method` 指导等级时附加给 Agent（见 §6）
  environment/
    Dockerfile         # Harbor 运行用；本地 runner 不需要 Docker
    workspace/         # 拷贝为 Agent 的工作目录：starter 代码 + 公开样例
  solution/            # 参考解（oracle），Agent 不可见
    solve.sh
  tests/               # 隐藏评测，Agent 不可见
    test.sh            # 入口：写 reward.txt（0/1）与 metrics.json（连续分与诊断）
    grade.py           # 独立评测程序
    cases.json         # 冻结的隐藏用例与参考答案
    build_cases.py     # 作者侧生成脚本（固定种子，可重新生成）
    sandbox_worker.py  # 在子进程中运行提交代码，与参考答案隔离
  baselines/           # “看似合理但错误”的基线，用于验证评测能区分常见错误
```

## 3. 三条任务

| | T1 `seq-align-affine` | T2 `reaction-network-stiff` | T3 `seir-changepoint-inference` |
|---|---|---|---|
| 难度 | 简单（算法实现） | 中等（数值方法选择） | 困难（统计推断 + 模型选择） |
| 研究问题 | 实现仿射空位罚分的最优双序列比对（Gotoh），含全局/局部模式 | 从 JSON 描述构建通用质量作用动力学 ODE，并在刚性系统上准确、快速地求解 | 从带噪声的每日病例中估计 R0、干预后 R、干预日期，并预测未来 28 天 |
| 输入 | 两条 DNA 序列 + 打分参数 | 反应网络（物种、初值、反应、速率）+ 时间点 | 70 天病例数、人口、已知潜伏期/传染期、初始感染数、预测窗口 |
| 可用工具 | bash、Python、numpy；禁止现成比对库 | bash、Python、numpy、scipy | bash、Python、numpy、scipy |
| 预期输出 | `align()` 返回 (分数, 比对行 a, 比对行 b) | `simulate()` 返回 (时间点 × 物种) 浓度矩阵 | `fit()` 返回 R0、R_post、change_day、forecast |
| 隐藏测试 | 64 个用例（含边界）+ 2 个 1500×1500 性能用例 | 8 个未见网络：Michaelis–Menten 快结合、Robertson、基因表达二聚化抑制、SIR、3/4 级 MAPK 级联、12/40 物种随机网络 | 12 次合成疫情（3 次无干预），所有参数各不相同 |
| 成功条件 | 全部用例：分数等于最优值、比对合法、重新计分一致；性能用例 ≤ 60 s | 每个网络 `|y−y_ref| ≤ 1e-4·(|y_ref| + 1e-3·max|y_ref|)`，≤ 60 s | ≥ 10/12 次疫情通过：R0、R_post 相对误差 ≤ 10%，change_day 误差 ≤ 3 天（仅当 R 降幅 ≥ 25% 时计分），预测相对 MAE ≤ 0.30 |
| 连续分 | 通过用例比例 | 通过网络比例 | 各疫情满足判据比例的平均 |
| 参考答案来源 | 参考实现 + 小规模穷举（300 组）交叉验证 | Radau(rtol 1e-11) 与 LSODA(rtol 1e-10) 两个独立求解器交叉验证（差异 < 2e-6，比评分容差小 50 倍以上） | 生成参数即真值；预测与无噪声期望曲线比较 |

### 3.1 T1 设计要点
- **不比较比对字符串**，因为最优比对可能不唯一；而是检查三件事：分数等于最优值、去掉 `-`
  后还原输入（局部模式为子串）、按定义重新计分等于报告分数。这样既允许多解，又防止
  “分数对但比对瞎编”。
- 参考实现与指数级穷举在 300 组小输入上逐一对比（规避 Gotoh 原始论文中已知的初始化错误）。
- 静态扫描禁止导入 Biopython、parasail 等现成比对库。

### 3.2 T2 设计要点
- 难点不在“会不会调 `solve_ivp`”，而在**识别刚性并选对方法与容差**：显式 RK45 在刚性网络
  上超时；隐式求解器用默认容差则精度不达标（见 §5 校准结果）。
- 网络包含零级合成、降解、二级（2P→D）反应、催化剂两侧出现等易错细节；隐藏网络与公开
  样例完全不同，防止针对样例过拟合。
- 评分容差混合相对误差与“物种量级 × 1e-3”的绝对下限，避免浓度接近 0 时相对误差失真。

### 3.3 T3 设计要点
- **可识别性**：仅凭发病数据无法同时识别潜伏期、传染期和 R0，因此潜伏期和传染期作为已知
  量给出；报告率设为完整，避免与初始感染数混淆。
- 干预日期只在“病例清晰可见且未到峰值”的窗口内采样；R 降幅 < 25% 时干预日期本身不可识别，
  不对 change_day 计分（参考解在降幅 15% 的样本上误差 7 天，印证了这一点）。
- 包含 3 次无干预疫情，考察 Agent 是否会做**模型选择**而不是硬套变点模型（参考解用 BIC）。
- 预测与无噪声的期望发病曲线比较，避免把观测噪声计入预测误差。
- 阈值按参考解在同分布数据上的最差误差留出余量设定（R 最差 6.4%→阈值 10%；预测最差 0.20→阈值 0.30）。

## 4. 运行环境与隔离

```
harness/runner.py   # 每个 trial：拷贝 workspace → Agent 阶段 → 在 verify/ 新目录中评测 → result.json
harness/agent.py    # 最小工具调用 Agent（OpenAI 兼容接口，vLLM 部署 Qwen3-8B）
harness/sandbox.py  # Agent 命令以低权限用户执行，隐藏目录 chmod 700
```

- **Agent 工具**：`bash`（带超时）、`write_file`、`read_file`、`submit`。Agent 能写代码、
  运行实验、检查公开样例，无网络、不能安装包。
- **隔离**（服务器无 Docker，故采用用户级隔离）：runner 以 root 运行；Agent 命令通过
  `runuser -u agent` 执行，工作目录归 agent 所有；`tests/`、`solution/`、`baselines/` 设为 700。
  评测时提交代码同样以 agent 用户在子进程中运行，只收到调用参数；参考答案只在 root 的评测
  父进程中比较。有 Docker 时可直接用 Harbor 的 `environment/Dockerfile` 运行。
- **单线程**：Agent 与评测均设 `OMP_NUM_THREADS=1`，使计时可比。

## 5. 评价指标

| 维度 | 指标 | 来源 |
|---|---|---|
| 正确性 | `reward`（0/1，是否解决）；多次 trial 的 pass rate、pass@k | `reward.txt` |
| 研究效果 | 连续 `score`；T2 最大缩放误差；T3 各参数相对误差、预测误差、各判据通过率 | `metrics.json` |
| 实验效率 | Agent 墙钟时间、轮数、工具调用数、token 数；提交代码运行时间（T1 性能用例、T2 每网络耗时） | `trajectory.jsonl`、`metrics.json` |
| 可复现性 | 任务内容哈希（`task_digest`）；固定种子数据；评测器对同一提交重复评测结果一致；多 trial 方差 | `result.json`、`summary.json` |

### 评测校准（基线运行检查）

每条任务用三类非 LLM 提交校准评测器：oracle（参考解）应满分；null（未修改的 starter）应 0 分；
“看似合理但错误”的基线应被判失败且只得部分分。

| 任务 | oracle | null | 错误基线 |
|---|---|---|---|
| T1 | reward 1，score 1.00 | 0 / 0.00 | 线性空位罚分：0 / 0.47 |
| T2 | 1 / 1.00（8 个网络共 0.45 s） | 0 / 0.00 | 显式 RK45：0 / 0.25（刚性网络超时）；LSODA 默认容差：0 / 0.00（精度不达标） |
| T3 | 1 / 1.00（12/12） | 0 / 0.00 | 忽略干预的单 R 拟合：0 / 0.36（只通过 3 次无干预疫情） |

## 6. 指导分级（借鉴 ASI-Bench）

同一任务、同一隐藏测试，只改变 Agent 拿到的说明：

| 等级 | Agent 拿到的内容 | 考察能力 |
|---|---|---|
| `goal`（只给目标） | `instruction.md`：问题、接口、数据假设、成功条件；不点名算法 | 自主选择方法 + 实现 |
| `method`（给方法） | `instruction.md` + `guidance.md`：推荐方法与关键细节（T1 三矩阵递推与边界；T2 刚性 → BDF + 收紧容差 + 解析 Jacobian；T3 剖面似然扫描干预日 + BIC 模型选择） | 只考实现 |

两档分数之差衡量“自主选方法”的难度；`method` 档仍失败则说明瓶颈在实现。
`guidance.md` 与隐藏测试一样放在任务目录中，对 Agent 用户不可读，`goal` 档无法偷看。

## 7. 端到端 Agent 评测

- 模型：Qwen3-8B（vLLM，单卡），OpenAI 兼容工具调用（hermes parser）。
- 基线：7 种配置（指导等级 × 思考模式 × 防重复提示 × 公开自检脚本）× 3 任务 × 5 个 trial，
  最多 30 轮、30 分钟；训练后模型每任务 20 个 trial。报告 pass rate、平均 score、效率指标，
  并对失败轨迹做失败模式分类。结果见 [results.md](results.md)。
- 在基线评测基础上，对 Qwen3-8B 做全参数拒绝采样微调和多轮智能体强化学习（ROLL，REINFORCE++ 与
  GRPO），训练方案见 [training.md](training.md)。

## 8. 局限与扩展

- 任务数量少（3 条），难度分布由作者判断，尚未用多模型做难度校准（Terminal-Bench Science
  的目标是前沿模型约 10–20% 解决率）。
- 用户级隔离弱于容器隔离（共享内核与网络命名空间）；正式发布应使用 Harbor + Docker。
- T3 使用确定性 SEIR 生成数据，模型设定与真实世界一致，难度低于真实流行病学（模型误设、
  报告延迟、周末效应）；可逐步加入这些因素，并把指导分级扩展到 ASI-Bench 的四档（含干扰信息）。
- 扩展方向：系统发育似然计算、群体遗传模拟统计、合成单细胞数据上的差异表达
  （precision/recall 评分），以及“给定基线代码做加速”的性能研究型任务。
