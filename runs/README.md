# Run artefacts

Evaluation results and training logs behind [docs/results.md](../docs/results.md) and
[docs/training.md](../docs/training.md). Server paths are replaced by `$ROOT` / `$SHARED` and
internal hosts by `<ip>` / `<host>`.

```
baseline_matrix/<config>/         base Qwen3-8B, configurations A-G, 5 trials per task
                                  (analysis.md: tables, failure modes, verification behaviour)
eval_F/<model>/                   configuration F (non-thinking), 20 trials per task
  qwen3-8b, rft3, reinforcepp_step50, reinforcepp_step99, grpo_step40, grpo_step59 (released)
eval_G/<model>/                   configuration G (thinking), 20 trials per task
  qwen3-8b, rft4, grpo_step40, grpo_step59
training/
  sft_round3/, sft_round4/        RFT data summary and per-step loss
  rl_reinforcepp/, rl_grpo/       metrics.jsonl (one line per RL step), episodes.jsonl (one line
                                  per training episode), config_used.yaml, train_tasks.txt
```

Each evaluation directory holds `summary.json` (per-task pass rate, mean score, efficiency, stop
reasons) and, per trial, `result.json` (reward, score, grader metrics, agent statistics),
`trajectory.jsonl` (full agent transcript with tool calls and outputs), the grader output under
`verify/` and the submitted code under `app-*/`.

Tables and failure modes for a group of runs (each subdirectory is treated as one configuration):
`python scripts/analyze.py runs/eval_F`.
