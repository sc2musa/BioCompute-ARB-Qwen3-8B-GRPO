# BioCompute-ARB: an Auto Research Benchmark for scientific computing in computational biology

A small, runnable, automatically graded benchmark that tests whether an AI agent can carry out
computational-biology research tasks end to end: write code, run experiments, check its own
results and submit. It follows the task organisation of
[Terminal-Bench Science](https://github.com/harbor-framework/terminal-bench-science) (Harbor
format) and borrows graded method guidance from
[ASI-Bench](https://arxiv.org/abs/2608.17271).

| Task | Difficulty | Research skill | Verification |
|---|---|---|---|
| [`seq-align-affine`](tasks/seq-align-affine/instruction.md) | easy | implement optimal pairwise alignment with affine gaps (global + local) | exact optimal score + alignment validity + re-scoring on 66 hidden cases |
| [`reaction-network-stiff`](tasks/reaction-network-stiff/instruction.md) | medium | general mass-action ODE simulator that handles stiff networks | 8 unseen networks vs. cross-checked high-accuracy reference, 60 s limit each |
| [`seir-changepoint-inference`](tasks/seir-changepoint-inference/instruction.md) | hard | infer R0, post-intervention R and intervention day from noisy cases; forecast | 12 hidden synthetic outbreaks with known ground truth |

Every task is graded by an independent program; hidden tests and reference solutions are never
visible to the agent. Each run produces a binary `reward` (solved or not) and a continuous
`score` (partial credit), so weak models still yield an informative signal.

## Trained model

| | |
|---|---|
| Hugging Face | **[SC2meisah/BioCompute-ARB-Qwen3-8B-GRPO](https://huggingface.co/SC2meisah/BioCompute-ARB-Qwen3-8B-GRPO)** |
| Revision | `a0115dfc0783354f2d5b18d14317c60cc3379327` |
| Base model | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Training | rejection-sampling fine-tuning on self-generated T2 trajectories, then 60 steps of multi-turn agentic GRPO (ROLL) on freshly generated training instances; no teacher model, hidden tests never used |
| Use | non-thinking mode (`enable_thinking=False`) with the agent in `harness/agent.py`, configuration F |

Held-out test results, 20 trials per task, configuration F (goal-only instruction, non-thinking,
repeat guard, public `check.py`):

| Model | T2 solved / mean score | T1 mean score | T3 mean score |
|---|---|---|---|
| Qwen3-8B | 0/20 / 0.656 | 0.154 | 0.031 |
| + RFT (round 3) | 11/20 / 0.550 | 0.047 | 0.000 |
| + RFT + REINFORCE++ (final, step 99) | 7/20 / 0.350 | 0.064 | 0.000 |
| **+ RFT + GRPO (final, step 59) — released** | **19/20 / 0.950** | 0.068 | 0.000 |

No model solves T1 or T3 in this setting. The gain is specific to the task with occasional
successes and to the non-thinking mode it was trained in (10/20 on T2 with thinking enabled).
See [docs/results.md](docs/results.md) for all configurations, training curves and failure analysis.

## Documents

Design and analysis documents are in Chinese.

- [docs/survey.md](docs/survey.md): survey of Auto Research / AI Scientist / Research Agent / RSI benchmarks
- [docs/survey_scicomp.md](docs/survey_scicomp.md): verification in scientific-computing benchmarks
- [docs/survey_compbio.md](docs/survey_compbio.md): verifiability of computational-biology tasks
- [docs/design.md](docs/design.md): benchmark design, metrics, isolation, calibration
- [docs/results.md](docs/results.md): baseline and trained-model results, RL dynamics, failure modes, difficulty, limitations
- [docs/training.md](docs/training.md): rejection-sampling fine-tuning and multi-turn agentic RL (REINFORCE++, GRPO) of Qwen3-8B

## Repository layout

```
tasks/<task>/          Harbor-style task: task.toml, instruction.md, guidance.md,
                       environment/ (Dockerfile + agent workspace), solution/, tests/, baselines/
harness/runner.py      run oracle / null / baselines / LLM agent, then grade in a fresh directory
harness/agent.py       minimal tool-calling agent (bash, write_file, read_file, submit)
harness/sandbox.py     unprivileged-user sandbox for docker-less Linux hosts
scripts/               environment setup, vLLM serving, experiment matrix, analysis, regrading
training/              training-task generator, trajectory sampling, SFT data builder,
                       DeepSpeed SFT, ROLL RL environment (roll_env/), configs and patches
runs/                  committed evaluation results and training logs (see runs/README.md)
```

## Quick start (CPU only, no model needed)

Requires Python 3.10+ with `numpy` and `scipy`.

```bash
python -m harness.runner --agent oracle   --tasks all --out runs/local_oracle    # reward 1 on every task
python -m harness.runner --agent null     --tasks all --out runs/local_null      # reward 0 on every task
python -m harness.runner --agent baseline --tasks all --out runs/local_baselines # plausible-but-wrong solutions
```

Grade any workspace directly (Harbor layout is the default; paths can be overridden):

```bash
TESTS_DIR=tasks/seq-align-affine/tests WORKSPACE_DIR=path/to/workspace \
VERIFIER_LOG_DIR=out bash tasks/seq-align-affine/tests/test.sh
```

## End-to-end agent evaluation (Linux + 1 GPU)

```bash
cp configs/server.env.example configs/server.env   # set paths, optional existing env/model
bash scripts/setup_env.sh                           # env, model, sandbox user (skips what exists)
bash scripts/serve_vllm.sh &                        # Qwen3-8B behind an OpenAI-compatible API
sudo bash scripts/run_matrix.sh runs/matrix 5       # 5 configurations x 3 tasks x 5 trials
python scripts/analyze.py runs/matrix               # tables + failure modes
```

The runner must run as root when `--sandbox-user` is used: agent commands and submitted code
then execute as that unprivileged user, while hidden tests stay root-only. With Docker
available, each task can instead be run by Harbor using `environment/Dockerfile`.

Evaluate the released model in configuration F:

```bash
bash scripts/serve_vllm_multi.sh 8 <downloaded-model-dir> biocompute-grpo
sudo python -m harness.runner --agent llm --tasks all --trials 20 --sandbox-user agent \
  --guidance goal --repeat-guard --public-check --workers 12 --model biocompute-grpo \
  --base-url http://127.0.0.1:8000/v1,...,http://127.0.0.1:8007/v1 --out runs/eval_grpo
```

## Training (8 x H100)

```bash
# fresh training instances (new seeds, never the hidden test cases)
python training/make_train_tasks.py --out <train-dir> --per-task 128
# self-generated trajectories (GUIDANCE=method appends guidance.md while sampling)
GUIDANCE=method bash training/sample_trajectories.sh <train-dir> <oracle-run-dir> <sample-dir> 8
# verifier-filtered SFT data (guidance is removed from the training prompts)
python training/build_sft_data.py --runs <sample-dir> --tokenizer <Qwen3-8B> \
  --tasks-root <train-dir> --out sft.jsonl
# full-parameter rejection-sampling fine-tuning (DeepSpeed ZeRO-3)
bash training/run_sft_deepspeed.sh sft.jsonl <rft-dir>
# multi-turn agentic GRPO with ROLL (ROLL_DIR in configs/server.env)
RL_STEPS=60 bash training/run_rl.sh <rft-dir> <rl-dir> rl_grpo_biocompute
```

Details and hyperparameters: [docs/training.md](docs/training.md).

## Reproducibility

- Hidden data are produced by seeded generators (`tests/build_cases.py`) and frozen in
  `tests/cases.json`; every result records a content hash of the task directory.
- Reference answers are cross-validated (brute force for T1, two independent stiff solvers for
  T2, known generating parameters for T3).
- Verification is deterministic: the same submission always receives the same reward and score.
