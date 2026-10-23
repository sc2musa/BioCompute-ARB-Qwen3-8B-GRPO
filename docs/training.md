# 训练：Qwen3-8B 的验证器驱动自我改进（RFT + 多轮智能体 RL）

结果与分析见 [results.md](results.md)。本文档说明训练数据、各轮训练的做法、超参数与工程问题。

## 1. 总览

```
Qwen3-8B ──采样(第1轮, 只给目标)──┐
         ──采样(第2轮, 方法指导)──┴─ 筛选 T2 ─→ RFT 第3轮 (不思考) ──┬─ REINFORCE++ 100 步（对照，后期退化）
                                                                    └─ GRPO 60 步 ──→ 交付模型
Qwen3-8B ──采样(第4轮, 方法指导 + 思考)─ 筛选 ─→ RFT 第4轮 (思考模式，对照)
```

| 模型 | 起点 | 方法 | 数据 | 用途 |
|---|---|---|---|---|
| RFT 第 3 轮 | Qwen3-8B | 全参数 SFT（拒绝采样） | 3587 个回合样本，全部来自 T2 | RL 起点 |
| RFT 第 4 轮 | Qwen3-8B | 全参数 SFT（思考模式） | 871 个回合样本，三条任务 | 思考模式对照 |
| REINFORCE++ | RFT 第 3 轮 | 多轮智能体 RL，整批基线 | T1 + T2 训练题 256 道 | 对照 |
| **GRPO** | RFT 第 3 轮 | 多轮智能体 RL，组内基线 | T1 + T2 训练题 256 道 | **交付** |

所有训练信号都来自模型自己的轨迹和评测器打分，没有使用更强的教师模型，也没有人工撰写的解答。

## 2. 训练数据：与测试集隔离的训练题

- 生成器：[training/make_train_tasks.py](../training/make_train_tasks.py)，复用三条任务各自的 `tests/build_cases.py`。
- 规模：每条任务 128 道，共 384 道；每道训练题都是完整的任务目录（公开样例 + 隐藏用例），用同一套评分程序评分。
- 与测试集隔离：
  - 随机种子从 10,000,000 起，测试集种子为 2026102x；
  - 测试集中参数固定的题型（T1 打分参数、T2 的 Robertson / 基因二聚化 / SIR 网络）在训练题中改为随机参数；
  - 测试集的隐藏用例与参考解从不进入训练数据。
- 有效性验证：参考解在 384 道训练题上运行，T1 128/128、T2 128/128、T3 124/128 通过；T3 的 4 道失败题
  干预极弱、参数本身不可识别，予以剔除，**有效训练题 380 道**。

## 3. 拒绝采样微调（RFT）

### 3.1 采样与“方法指导蒸馏”

- 采样脚本：[training/sample_trajectories.sh](../training/sample_trajectories.sh)；8 个 vLLM 副本（每卡一个）。
- 设置与基准 F 配置相同（自检脚本 `check.py`、防重复提示），温度 0.7、top_p 0.8；思考模式采样用 0.6 / 0.95。
- **第 1 轮**：只给目标，每道题 8 条轨迹。基座模型几乎没有全对轨迹，可用数据太少。
- **第 2 轮**：采样时附加方法指导（`guidance.md`），每道题 8 条。T2 的成功轨迹明显增多。
- **训练时去掉指导**（[training/build_sft_data.py](../training/build_sft_data.py)）：方法指导采样的轨迹，
  训练时把第一条用户消息换回只给目标的版本；引用了“guidance”字样的回合不作为训练目标。
  这样模型学的是“在没有提示的情况下也这样做”，相当于把方法指导蒸馏进模型。
- **第 4 轮**：方法指导 + 思考模式，三条任务各采样。

### 3.2 筛选与样本构造

对每道训练题：
- 所有 reward = 1（完整解题）的轨迹保留；
- 若无完整解题，保留同组中得分最高的一条，前提是得分 ≥ 0.5 且高于该题平均分；
- 未提交的轨迹丢弃。

保留轨迹中的**每个 Agent 回合**构成一条样本：输入是模型当时实际看到的完整上下文（Qwen3 对话模板、
工具定义、按评测框架同样的规则折叠旧的工具输出），目标是模型当时的输出；**loss 只计算在模型生成的部分**，
超过 32768 token 的样本丢弃。

| | 第 3 轮 | 第 4 轮 |
|---|---|---|
| 采样来源 | 第 2 轮（方法指导，不思考），只取 T2 | 第 4 轮（方法指导，思考） |
| 覆盖训练题 | 128 道 T2，126 道有保留轨迹 | 380 道，229 道有保留轨迹 |
| 保留轨迹 | 409 条（全部 T2） | 345 条（T2 244、T1 81、T3 20） |
| 训练样本（回合） | 3587 | 871 |

第 3 轮只用 T2：前两轮采样中 T1、T3 的全对轨迹极少，混入大量“部分正确”的轨迹会把模型往错误实现上拉。

### 3.3 训练

| 参数 | 值 |
|---|---|
| 实现 | DeepSpeed ZeRO-3 全参数训练，自写训练循环 [training/sft_deepspeed.py](../training/sft_deepspeed.py)，启动脚本 [training/run_sft_deepspeed.sh](../training/run_sft_deepspeed.sh) |
| 硬件 | 8 × H100 80GB |
| 精度 / 序列长度 | bf16 / 32768 |
| 优化器 | AdamW（β = 0.9/0.95），梯度裁剪 1.0；第 4 轮把优化器状态放到 CPU（[ds_zero3_offload.json](../training/ds_zero3_offload.json)） |
| 学习率 | 1e-5，预热 3%，余弦衰减 |
| 批量 / 轮数 | 全局 32 条样本，2 epochs |
| loss 归一化 | 按 token 平均 |
| 显存 | 只对需要计算 loss 的位置做词表投影；梯度检查点 |
| 结果 | 第 3 轮 224 步、35 分钟，最终 loss 0.022；第 4 轮 54 步、22 分钟，最终 loss 0.118 |

原计划使用 ROLL 的 SFT 流程（Megatron 后端），但该 ROLL 版本需要更新的 megatron-core 与 TransformerEngine，
共享环境不满足；为不改动共享环境，改用 DeepSpeed 直接训练。未使用的 ROLL SFT 配置与补丁保留在
[training/roll_configs/rft_sft_qwen3_8b.yaml](../training/roll_configs/rft_sft_qwen3_8b.yaml) 和
[training/roll_patches/sft_raw_template.diff](../training/roll_patches/sft_raw_template.diff)。

## 4. 多轮智能体强化学习（ROLL）

### 4.1 环境

[training/roll_env/biocompute_env.py](../training/roll_env/biocompute_env.py) 实现 ROLL 的
`AgentNativeStepEnvManager` 接口，与评测完全一致：
- 复用 [harness/agent.py](../harness/agent.py) 的系统提示、工具定义、旧输出折叠和“未调用工具”反馈（F 配置）；
- 按 vLLM hermes 规则解析工具调用，命令以低权限用户在沙箱中执行；
- 模型调用 `submit` 或达到 30 轮 / 30 分钟时，用评测器在训练题的隐藏用例上打分；
- **奖励 = 连续分 + 0.5 × 是否全对**，只在最后一轮给出；
- 自测脚本 [training/roll_env/check_env.py](../training/roll_env/check_env.py)：参考解得 1.5 分，空提交 0 分。

训练题只用 T1 + T2（各 128 道）。T3 的长时间拟合会拖慢采样，且模型在 T3 上几乎没有成功样本。

### 4.2 两种优势估计

两次训练只差优势估计，其他设置完全相同。

| | REINFORCE++（[配置](../training/roll_configs/rl_reinforcepp_biocompute.yaml)） | GRPO（[配置](../training/roll_configs/rl_grpo_biocompute.yaml)） |
|---|---|---|
| 每道题采样数 | 1 | 4（一组，同一个种子 = 同一道题） |
| 回报 | 每一轮都取整局回报（γ = 1） | 同左 |
| 归一化 | 减整批均值、除整批标准差，再全局白化 | 减组内均值、除组内标准差；组内分数全相同时优势为 0 |
| 每步轨迹 | 32 局 | 32 局（8 组 × 4） |
| 步数 | 100 | 60 |

两者都是 ROLL 的 `step_reinforce` 估计器（每轮对话作为一个样本，所有轮次共享整局回报），区别只在
`reward_normalization` 的分组方式。GRPO 设置与 ROLL 自带的 agentic 示例一致。

### 4.3 共同超参数

| 参数 | 值 |
|---|---|
| 起点 | RFT 第 3 轮 |
| 学习率 | 1e-6，常数 |
| KL | 不加（`init_kl_coef: 0`，`use_kl_loss: false`） |
| PPO epoch | 1 |
| 采样 | 温度 1.0，top_p 1.0，每轮最多 4096 token，上下文 32768 |
| 资源划分 | 0–3 号卡训练（DeepSpeed ZeRO-3 + 优化器放 CPU），4–7 号卡跑 vLLM |
| 异步 | `async_generation_ratio: 1`：采样最多领先训练一步 |
| 环境数 | 48 个（REINFORCE++ 48 组 × 1，GRPO 12 组 × 4） |
| 检查点 | 每 10 步 |
| 耗时（墙钟，含启动和保存检查点） | REINFORCE++ 100 步约 8 小时，GRPO 60 步约 7 小时（GRPO 要等一组 4 局都结束才能收齐） |

启动：
```bash
RL_STEPS=60 bash training/run_rl.sh <RFT 第3轮模型> <输出目录> rl_grpo_biocompute
```
平台任务入口为 [training/submit_rl_job.sh](../training/submit_rl_job.sh)（建沙箱用户、保留旧输出目录后调用 `run_rl.sh`）。

### 4.4 对 ROLL 的修改

[training/roll_patches/](../training/roll_patches/) 下三个可重复执行的补丁，由 `run_rl.sh` 自动应用：

| 补丁 | 问题 | 修改 |
|---|---|---|
| `install_biocompute_env.sh` | 环境需要注册到 gem | 复制环境代码并注册 `biocompute` |
| `patch_chunked_logprobs.py` | 32k token 的微批次计算完整词表 float32 logits 时显存不足 | `log_probs_from_logits` 分块 + 检查点计算；GPU 上验证与原实现结果一致 |
| `patch_rollout_get_batch.py` | `get_batch` 每轮要等所有环境组交回一局，一个跑满 30 分钟的慢局会卡住整批（每步采样时间 80 → 394 → 1376 秒） | 某组交回一局后立刻重新领取，去掉按轮屏障 |

### 4.5 调试过程中的失败运行

失败的运行都改名保留在服务器上，没有覆盖：
- 步数被读成字符串、缺少 tensorboard：改为 `oc.decode` 读取，日志改为 stdout；
- 训练与推理同卡时 CUDA IPC 报错：改为训练、推理分卡；
- 工具调用参数被解析成 dict，导致旧调用折叠出错：`parse_tool_call_parameter_to_dict: false`；
- 同步模式和 32 个环境时被慢局卡住：改为异步 + 48 个环境 + 去掉按轮屏障的补丁；
- 长样本（17k–19k token）显存不足或触发微批上限断言：分块 log-prob 补丁 + 微批上限 32768。

## 5. 评测

所有模型都只在冻结测试集上评测，与基线使用同一套脚本：F 配置（不思考）与 G 配置（思考），每任务 20 局。
结果、训练过程曲线和失败分析见 [results.md](results.md) §4–§6。

## 6. 发布

交付模型（GRPO 最终检查点）上传到 Hugging Face（公开），链接与 revision 见 [README](../README.md)。
权重格式为 safetensors，推荐用不思考模式（`enable_thinking=False`）配合本仓库的 Agent 使用。
