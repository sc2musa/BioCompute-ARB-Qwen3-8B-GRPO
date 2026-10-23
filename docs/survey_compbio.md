# 计算生物学任务的可验证性调研（补充 survey.md）

> 目的：回答“计算生物学的可验证性不太好确定”这一顾虑：哪些计算生物任务能被**独立 verifier + 隐藏测试**客观、确定性地评分，哪些不能。
> 调研日期：2026-10-20。通用 Auto Research 基准（MLE-bench、RE-Bench、PaperBench、Harbor 格式等）见 `survey.md`，本文不重复。无法核实的数字不写，存疑处已标注。

---

## 0. 结论先行

- “计算生物”不是一个可验证性等级，**任务类型**才是。算法/数值类任务（比对、ODE 模拟、似然计算）的正确答案由数学唯一确定，可以 exact match 或带容差比对；统计推断类任务在**合成数据 + 已知真值**下也能客观评分，但必须先用参考解标定“理论上能达到的误差”；只有“生物学结论/解读”类任务才离不开 LLM 或人工评审。
- 现有生物 agent 基准的评分方式恰好覆盖这三档：LAB-Bench（选择题精确匹配）、BixBench 开放题（LLM judge）、DREAM 参数估计/网络推断（合成模型 + 隐藏真值）、CASP/CAMEO（盲测 + 结构比对分数）、Terminal-Bench Science 生命科学题（合成隐藏真值 + 多重阈值 + 必须优于基线对照）。
- 对本项目：只要把任务限定在 A/B 两档，并在 B 档中用“隐藏种子重新生成数据 + 对照基线 + 预先标定的容差”，就能做到**确定性评分**（同一份提交，重复评分结果完全一致）。

---

## 1. 代表性基准及其评分方式

| 名称 | 任务形式 | 验证方式 | 确定性 | 备注 |
|---|---|---|---|---|
| **LAB-Bench** (2024) | 2,400 余道生物研究选择题（文献、图表、数据库、序列操作 SeqQA、克隆方案等） | 选择题 exact match；允许选“信息不足”，报告 Accuracy / Precision / Coverage | 高（但只测知识与单步操作） | 不测长程执行 |
| **LABBench2** (Edison, 2026-02) | 1,892 题、11 类；取消选择题，改为开放作答，强调检索与阅读原始材料 | 官方博客未说明评分方式（**未核实**是否用 LLM judge） | 未知 | 新增 PatentQA、TrialQA 等 |
| **BixBench** (2025, v3) | 61 个分析 capsule、205 个问题；agent 在 Jupyter 中分析真实数据 | 开放题：Claude 3.5 Sonnet 对照参考答案给 0/1；选择题：另一 LLM 读 notebook 选项，五次运行多数投票 | **低**（依赖 judge） | 开放题前沿模型约 17% |
| **BixBench3** (Edison, 2026-08) | 20 个“从原始数据复现已发表研究”的任务，共 138 个中间产物（差异表达表、蛋白丰度矩阵、peak 矩阵等） | 每个产物**程序化比对**原研究的参考产物，任务得分 = 复现成功的产物比例 | 中高 | 作者承认：方法若与原文不同，合理的替代方案也会被判低分；具体容差未公开 |
| **BioCoder** (2023/2024) | 2,269 个生信函数级编码题（GitHub 抽取的 Python/Java + 253 道 Rosalind 题） | 运行测试 / fuzz testing，报告 Pass@K | 高 | 函数级，非研究级 |
| **ScienceAgentBench** (2024) | 102 个任务（含生物信息学），输出独立 Python 程序 | 任务专属评测脚本：数值用 `math.isclose`、模型用阈值（如 ROC-AUC ≥ 0.77）；**图表用 GPT-4o 打分**（≥60 算成功） | 数值题高，图表题低 | 同一基准内混用两种评分 |
| **GeneTuring / GeneGPT** (2023) | 16 个模块、1,600 道基因组知识问答（命名、定位、功能、序列比对） | GeneTuring 原文人工评分 48,000 个回答；GeneGPT 对命名、定位用 exact match，基因-疾病关联用 recall，染色体对但位置错给半分 | 中（规则化后高） | 答案是事实性 ID/位置，易被检索直接命中 |
| **BioDiscoveryAgent** (ICLR 2025) | 闭环设计 CRISPR 扰动实验：每轮选 128 个基因、共 5 轮 | 用已完成的全基因组筛选数据查表，统计命中率（hit rate） | 高（查表） | 比贝叶斯优化基线约高 21%（Claude 3.5 Sonnet）；“实验”是对已有数据的回放 |
| **BioML-bench** (2025) | 端到端生物医学 ML（蛋白工程、单细胞、医学影像、药物发现），MLE-bench 式 capsule | 任务专属指标（AUROC、Spearman 等）在隐藏测试集上计算，再与人类排行榜比较 | 高（指标确定，但训练随机性导致跨运行方差） | 后续 BioXArena (2026) 扩展到 76 个任务 |
| **Biomni-Eval1** (2025) | 433 个实例、10 类生物推理任务（GWAS 因果基因、罕见病诊断、CRISPR 递送方式选择等） | 提供 `evaluate()` 返回分数；答案多为基因名、OMIM ID、选项（**具体匹配规则未核实**） | 中高 | Biomni 以 Qwen-32B 做 RL 得到 Biomni-R0 |
| **CASP / CAMEO** | 盲测蛋白结构预测：目标结构尚未公开 | GDT 类分数（依赖叠合，0.5/1/2/4 Å 阈值）、lDDT（无需叠合的局部距离差测试） | 高（给定实验结构，分数是确定函数） | “未来答案”机制天然防污染；CAMEO 每周自动评测 |
| **DREAM6/7 参数估计** | 给定 9 基因调控网络结构，用有限“credits”购买模拟实验数据，估计隐藏参数；第二题还需补出 3 条缺失边 | 合成模型 + 隐藏真值：参数与真值的偏差 + 未测扰动下轨迹预测的偏差 | 高（真值已知） | 与本项目 T2/T3 最接近；论文强调拓扑不完全时难度大增 |
| **DREAM4 in silico 网络推断** | GeneNetWeaver 生成 10/100 基因网络的时间序列与扰动数据 | 对 gold standard 计算 AUROC / AUPR（不计自调控边） | 高 | B 档的标准范式：合成数据、植入真值、排序指标 |
| **Terminal-Bench Science 生命科学** (19 题) | 如 `ambient-rna-correction`、`foraging-cognitive-model`、`protein-active-learning`、`diag-chipseq` | 隐藏真值 + 多重数值阈值 + 必须优于若干平凡对照；二元 0/1 | 高 | 详见 §1.1 |
| **ASI-Bench** (2026) | 60 个项目级任务，含 biology 与 medicine & biostatistics 领域 | 有效性检查 + gate + 加权 scorer | 高 | 生物任务的具体内容与容差在摘要/正文概述中**未查到** |

### 1.1 Terminal-Bench Science 生命科学题的评分设计（读了仓库实际任务）

仓库 `tasks/life-sciences/` 下现有 biology 8 题、medicine 6 题、neuroscience 4 题、ecology 1 题。有三点值得借鉴：

1. **合成数据植入真值**。`ambient-rna-correction` 中，单细胞计数由 verifier 已知的“原生计数 T + 环境 RNA 污染”生成。verifier 持有真实污染率 ρ_c 和原生矩阵，因此可以计算 Pearson 相关（≥0.48）、MAE（≤0.115）、相对误差缩减比例等。
2. **必须打败对照基线**。同一题要求提交结果优于“不校正”“按固定比例 0.65–0.92 缩放”以及一个规定好的“PCA + K-means 置零 marker”对照。这堵住了“输出平凡解也能过阈值”的漏洞。
3. **提交代码、由 verifier 在隐藏数据上运行**。`foraging-cognitive-model` 要求提交 `model.py`。scorer 在 agent 看不到的第 10–14 次 session 上逐 trial 调用，计算 L/L_ceiling：每只动物 ≥0.97、均值 ≥0.993 才算通过。题目还附带反作弊约束：隐状态维数 ≤4 并做实际核查，输出必须依赖 reward，超过 ceiling 太多判为作弊。

---

## 2. 可验证性分级

| 等级 | 典型任务 | 验证方法 | 主要风险 | 对 7B 模型的适配 |
|---|---|---|---|---|
| **A 完全确定、客观** | DP 序列比对最优分（NW/SW/Gotoh）；ODE/动力学模拟与高精度参考对比；给定树和模型的系统发育似然（Felsenstein pruning）；k-mer 计数；给定组装结果的 N50 等统计；给定数据算 π、Tajima's D；在已知可辨识的合成数据上回收参数 | 整数分数 exact match；浮点值用 `|y−y_ref| ≤ atol + rtol·|y_ref|`；对产出做结构有效性检查后**重新计分** | 约定不一致（gap 罚分定义、单位、边界条件）；参考实现自身有 bug；直接调用库函数“作弊”；浮点舍入和求解器差异 | **好**：步骤明确，可设多个测试用例给部分分 |
| **B 客观但有统计噪声** | 合成数据（植入真值）上的差异表达、网络推断（AUPR）、change point 检测；SEIR/动力学参数推断；自写种群遗传模拟器与解析期望对比 | Precision/Recall/AUPR、参数相对误差、预测 CRPS/WIS；**多个隐藏种子数据集取平均**；阈值按“参考估计器的误差分布”标定 | **可辨识性**（结构上或实际上不可辨识的参数无法评分）；单次数据集的运气成分；阈值过严时最优方法也会失败；对 verifier 的数据生成器过拟合 | **中**：需要建模判断，7B 更可能拿部分分，适合连续分 |
| **C 需要 LLM 或人工判断** | “这组差异基因说明了什么通路机制”“给出可检验的生物学假设”“写讨论部分”；真实数据上没有金标准的分析 | LLM judge + rubric，或专家评审 | 不确定、偏差大，judge 与被测模型可能同源；7B 的输出容易被 judge 高估或低估 | **差**：不建议作为主评分 |

**判别规则**：若能写出一个**不读 agent 推理过程**、只看产出文件的 Python 函数，并且参考解稳定通过、空解和平凡解稳定失败，就属于 A 或 B 档；否则属于 C 档。

**几条细化**：

- “种群遗传模拟”要分开看：**给定数据算统计量**属于 A；**让 agent 写模拟器**时，随机数流不同，结果不可能逐位一致，只能对期望或分布做统计检验（如 Watterson θ 的期望、SFS 与 Kingman 解析值比较），属于 B，需要控制误判率（例如用足够大的重复数，使参考实现的误判率 < 10⁻³）。
- 真实数据（如 BixBench3 复现已发表研究）的评分可以程序化，但“参考答案”本身依赖原作者的方法选择，不算纯 A 档。
- 防作弊的通用手段：verifier 容器分离且不联网（见 survey.md）；数据由隐藏种子在评分时生成；要求优于对照基线；检查产出的内部一致性（例如比对结果重新计分要等于声明的分数）。

---

## 3. 三个候选任务评估

### T1 仿射 gap 双序列比对（Gotoh，global + local）

- **确定性**：A 档。给定替换矩阵（如 BLOSUM62，整数）、gap open/extend 和端部 gap 规则，最优分数唯一，可做 **exact match**。最优比对可能不唯一，因此不比较比对字符串本身，而是检查：① 去掉 gap 后还原为原序列（local 则为原序列的子串，且坐标正确）；② verifier 按同一打分规则**重新计分**，结果等于声明分数；③ 声明分数等于参考最优分。三项都满足才算对。
- **可能出错的地方**：
  - **约定歧义**：长度为 k 的 gap 罚 `o + k·e` 还是 `o + (k−1)·e`？是否允许一个序列的 gap 紧挨另一个序列的 gap？global 的端部 gap 罚不罚？local 的空比对是否记 0 分？instruction 必须用公式写死，并给出 2–3 个带答案的公开样例。
  - **参考实现本身有错**：Flouri 等（2015）指出 Gotoh 1982 原文有下标错误和 global 初始化错误，后者还出现在 Gusfield、Waterman 的教材中；Biopython 旧的 `pairwise2` 也被报告会给出次优比对。参考解应该用两套独立实现（例如自写 DP 加上 Biopython `PairwiseAligner` 或 parasail）交叉核对，再加暴力枚举小例子。
  - **奖励作弊**：agent 环境里如果装了比对库，任务就退化成调用库。要么不预装（verifier 离线，无法 pip 安装），要么明确允许并把难度放到规模上。
- **建议设置**：隐藏测试 20–40 对序列，覆盖空序列、单字符、全相同、无相似、长 gap、蛋白和 DNA 两种打分；另设 1–2 个大规模用例（如 2 万 × 2 万），配合内存上限，迫使 agent 实现线性空间算法（Myers–Miller 1988）。分数和坐标用整数 exact match；部分分 = 通过用例比例。
- **7B 适配**：好。算法广为人知，主要考核约定细节和边界条件，能拉开分布。缺点是可能“太教科书”，研究属性弱，需要靠规模、内存和时间限制增加难度。

### T2 通用质量作用定律反应网络 ODE 模拟器（JSON 规格，含刚性系统）

- **确定性**：A 档。初值问题的解唯一（多项式右端局部 Lipschitz）。参考解用 Radau（scipy `solve_ivp(method="Radau")`）在 rtol=1e-10、atol 约 1e-12 × 量级下计算，并用 BDF 或 LSODA 同样高精度交叉核对，二者差异应比判分容差小两个数量级以上。
- **判分建议**：在规定输出时间点上逐物种检查 `|y−y_ref| ≤ atol_i + rtol·|y_ref|`。建议 rtol = 1e-4（对 Radau 参考解宽松 4–6 个数量级）；atol_i 按该物种的特征浓度缩放（如 1e-8 × max|y_ref,i|），避免接近 0 的物种被相对误差放大。再加三类检查：非负性（允许 −1e-10 级别的舍入）、守恒量误差（由化学计量矩阵左零空间自动导出）、时间限制（例如每个系统 ≤ 10–30 s，使显式方法在刚性系统上超时）。
- **可能出错的地方**：
  - **规格歧义**：速率是 `k·∏[S]^ν` 还是包含组合数因子（`k[A]²/2`）？可逆反应拆成两条还是一条？零级源项、体积和单位怎么处理？JSON schema 必须给出速率公式，并附公开样例及其期望输出。
  - **病态系统**：质量作用网络也能出现振荡甚至混沌，长时间积分时误差指数放大，判分会对数值噪声过敏。隐藏测试应避开混沌区和分岔附近，或只取短时间窗口；也可以改为比较周期、稳态值这类稳健量。
  - **刚性测试的写法**：快速酶结合（k_on 比 k_cat 大 6 个数量级）、Robertson 问题这类经典刚性系统，可以区分是否会用隐式方法和 Jacobian。
  - **作弊或捷径**：直接调用 `solve_ivp(Radau)` 是合法做法，任务的难点于是变成规格解析、Jacobian 和规模。可以加入数百个物种的稀疏网络，要求提供解析 Jacobian 或稀疏结构才能在时限内完成，以拉开差距。
- **7B 适配**：好到中。解析 JSON 并调用 scipy 7B 能做；刚性、规模、守恒和时限能形成梯度。部分分 = 通过系统数 / 总数。

### T3 SEIR 参数推断：未知干预 change point + 负二项噪声日发病数

- **确定性**：B 档。verifier 本身确定（同一提交得分固定），但“正确答案”不是一个点，即使最优估计器也有误差。阈值不能拍脑袋，必须先用参考估计器（例如带 change point 的 SEIR 加 NB 似然的最大似然或贝叶斯拟合）在大量同分布合成数据上跑出误差分布，再据此定阈值。
- **可辨识性（关键）**：
  - **只有发病数时 R0 不能单独辨识**。早期增长率 r 与 R0 的关系依赖代际间隔分布。SEIR 中 R0 = (1 + r/σ)(1 + r/γ)（Wallinga & Lipsitch 2007），σ、γ 未知时 R0 无法从数据中唯一确定。**必须给定**潜伏期和传染期（或代际间隔分布），只让 agent 估计 β（或 R0）、干预后的 R、change point τ 和 NB 离散度。
  - **报告率 ρ 与初始感染数、人口规模 N 混淆**：流行早期远未饱和时，ρ 与 I0 基本只以乘积形式出现。要么给定 ρ 和 N，要么不对 ρ 评分。
  - **change point 与报告延迟混淆**：若数据含报告延迟，τ 会整体平移。要么不加延迟，要么给定延迟分布。
  - Roda 等（2020）指出 COVID 早期预测分歧大的主要原因是不可辨识，且复杂模型不一定更可靠；Gostic 等（2020）讨论了 Rt 估计在时间上的偏差和延迟问题。这些文献说明本题的风险是真实存在的。
- **判分建议**：
  - 每个测试用 **5–10 个隐藏种子**生成的数据集，真值参数在合理范围内随机抽取（如 R0 ∈ [1.5, 4]，R_post ∈ [0.5, 1.2]，τ 在第 20–60 天），结果取平均，降低运气成分。
  - 指标：R0 和 R_post 的相对误差（建议以参考估计器 90 分位误差的 1.5 倍为通过线，经验上约 10–20%，需实测标定）；τ 的绝对误差（如 ≤ 3 天）；未来 14–28 天预测，用**无噪声期望发病曲线**而非噪声实现计算对数尺度 MAE 或 WIS/CRPS（预测区间格式参见 Bracher 等 2021），以消除观测噪声带来的方差。
  - **推荐提交“估计器代码”而不是数字**（仿 `foraging-cognitive-model`）：verifier 在评分时用隐藏种子生成新数据并运行 agent 的 `fit(data) -> params, forecast`，固定随机种子和时间上限。这样既防止针对公开数据硬编码答案，也使任务真正考查“方法”。
  - 加对照基线：提交结果必须优于“无 change point 的 SEIR 拟合”和“对最近 7 天取均值外推”。
- **7B 适配**：中。完整写出带 change point 的似然并做优化对 7B 有挑战，但连续分（各指标按误差分段给分）能避免全零。这是三题中最接近“研究”的一题，也最依赖阈值标定。

---

## 4. 结论与建议

1. **可验证性问题有解**：把任务限定在 A 档（T1、T2）和受控的 B 档（T3），全部可由独立 verifier 程序化、确定性地评分，不需要 LLM judge。C 档（生物学解读）不作为主评分，最多作为不计分的附加观察项。
2. **A 档要防“约定歧义”和“参考解有 bug”**：instruction 用公式写死约定并附带答案的公开样例；参考解用至少两套独立实现交叉核对（T1 的 Gotoh 初始化错误就是现实先例）。
3. **B 档要先解决可辨识性**：只对可辨识量评分，不可辨识的量要么作为已知条件给出，要么不评；阈值由参考估计器在同分布数据上的误差分布标定；用多个隐藏种子取平均。
4. **统一的防作弊做法**：verifier 分离且离线；数据由隐藏种子在评分时生成（或提交代码由 verifier 运行）；要求优于平凡对照；检查产出内部一致性（重新计分、守恒量）。
5. **分数设计**：每题同时输出二元 `pass`（全部用例通过）和连续分（通过用例比例或按误差分段打分），保证 7B 基线不会全部为 0。用 oracle 解（应得满分）和 nop 解（应接近 0）验证 verifier。
6. **难度梯度建议**：T1（算法实现）→ T2（数值方法 + 工程）→ T3（统计建模 + 可辨识性判断），恰好覆盖“实现 → 数值 → 推断”三层研究能力，也与 ASI-Bench 的发现一致：从“给方法”到“自己实现方法”，难度变化最大。

---

## 5. 参考文献

1. Laurent, J. M., et al. LAB-Bench. arXiv:2407.10362, 2024. https://arxiv.org/abs/2407.10362
2. Edison Scientific. LABBench2: an improved benchmark, 2026-02. https://edisonscientific.com/news/labbench2-an-improved-benchmark
3. Mitchener, L., et al. BixBench. arXiv:2503.00096 (v3, 2025-10). https://arxiv.org/abs/2503.00096
4. Edison Scientific. BixBench3: benchmarking study-scale computational biology, 2026-08-26. https://advances.edisonscientific.com/research/bixbench3-benchmarking-study-scale-computational-biology/
5. Tang, X., et al. BioCoder. Bioinformatics 2024 / arXiv:2308.16458. https://arxiv.org/abs/2308.16458 ；https://github.com/gersteinlab/biocoder
6. Chen, Z., et al. ScienceAgentBench. arXiv:2410.05080, 2024. https://arxiv.org/abs/2410.05080
7. Hou, W., Ji, Z. GeneTuring. bioRxiv 2023.03.11.532238. https://www.biorxiv.org/content/10.1101/2023.03.11.532238
8. Jin, Q., et al. GeneGPT. arXiv:2304.09667, 2023. https://arxiv.org/abs/2304.09667
9. Roohani, Y., et al. BioDiscoveryAgent. ICLR 2025 / arXiv:2405.17631. https://arxiv.org/abs/2405.17631
10. BioML-bench: Evaluation of AI Agents for End-to-End Biomedical ML. bioRxiv 2025.09.01.673319. https://www.biorxiv.org/content/10.1101/2025.09.01.673319
11. BioXArena. arXiv:2605.15766, 2026. https://arxiv.org/abs/2605.15766
12. Huang, K., et al. Biomni: A General-Purpose Biomedical AI Agent. bioRxiv 2025.05.30.656746. https://www.biorxiv.org/content/10.1101/2025.05.30.656746 ；https://pypi.org/project/biomni/
13. Haas, J., et al. CAMEO complementing CASP12. Proteins 2018. https://pubmed.ncbi.nlm.nih.gov/29178137/ ；CAMEO 评分说明 https://cameo3d.org/cameong_help/3d
14. Mariani, V., et al. lDDT: a local superposition-free score. Bioinformatics 29(21):2722–2728, 2013. https://pmc.ncbi.nlm.nih.gov/articles/PMC3799472
15. Meyer, P., et al. Network topology and parameter estimation (DREAM6/7). BMC Syst Biol 8:13, 2014. https://dspace.mit.edu/handle/1721.1/86005 ；挑战页 https://dreamchallenges.org/dream-6-estimation-of-model-parameters-challenge/
16. Steiert, B., et al. Experimental Design for Parameter Estimation of Gene Regulatory Networks. PLoS ONE 2012. https://pmc.ncbi.nlm.nih.gov/articles/PMC3397990/
17. Marbach, D., et al. Revealing strengths and weaknesses of methods for gene network inference (DREAM). PNAS 2010. https://pmc.ncbi.nlm.nih.gov/articles/PMC2942832
18. Terminal-Bench Science 生命科学任务目录. https://github.com/harbor-framework/terminal-bench-science/tree/main/tasks/life-sciences
19. Zhou, J., et al. ASI-Bench. arXiv:2608.17271, 2026. https://arxiv.org/abs/2608.17271
20. Gotoh, O. An improved algorithm for matching biological sequences. J Mol Biol 162:705–708, 1982. https://doi.org/10.1016/0022-2836(82)90398-9
21. Flouri, T., et al. Are all global alignment algorithms and implementations correct? bioRxiv 031500, 2015. https://www.biorxiv.org/content/10.1101/031500v1
22. Myers, E. W., Miller, W. Optimal alignments in linear space. CABIOS 4(1):11–17, 1988. https://doi.org/10.1093/bioinformatics/4.1.11
23. SciPy `solve_ivp`（Radau/BDF/LSODA）. https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html ；Robertson 刚性问题示例 https://scipython.com/book2/chapter-8-scipy/examples/solving-a-system-of-stiff-odes/
24. Wallinga, J., Lipsitch, M. How generation intervals shape the relationship between growth rates and reproductive numbers. Proc R Soc B 274:599–604, 2007. https://doi.org/10.1098/rspb.2006.3754
25. Roda, W. C., et al. Why is it difficult to accurately predict the COVID-19 epidemic? Infect Dis Model 5:271–281, 2020. https://doi.org/10.1016/j.idm.2020.03.001
26. Gostic, K. M., et al. Practical considerations for measuring the effective reproductive number, Rt. PLOS Comput Biol 16(12):e1008409, 2020. https://doi.org/10.1371/journal.pcbi.1008409
27. Tuncer, N., Le, T. T. Structural and practical identifiability analysis of outbreak models. Math Biosci 299:1–18, 2018. https://doi.org/10.1016/j.mbs.2018.02.004
28. Bracher, J., et al. Evaluating epidemic forecasts in an interval format. PLOS Comput Biol 17(2):e1008618, 2021. https://doi.org/10.1371/journal.pcbi.1008618
