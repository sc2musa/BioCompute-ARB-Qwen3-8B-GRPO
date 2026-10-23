"""Re-run the verifier on the final workspaces of finished trials (no agent re-run).

Usage (as root when a sandbox user was used):
    python scripts/regrade.py <matrix-or-run-dir> [--tasks t1,t2] [--sandbox-user agent] [--write]

Used (1) after fixing a verifier/test-data bug, and (2) to check that grading is
deterministic: unchanged tasks must reproduce the stored reward and score exactly.
With --write, result.json is updated and the old values are kept under "previous".
"""

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.runner import TASKS_DIR, task_digest, verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tasks", default=None, help="comma-separated task names (default: all)")
    ap.add_argument("--sandbox-user", default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tasks-root", default=str(TASKS_DIR),
                    help="task directory the run used (training instances live outside tasks/)")
    args = ap.parse_args()
    if args.sandbox_user:
        os.umask(0o077)
    wanted = set(args.tasks.split(",")) if args.tasks else None
    paths = [p for p in sorted(Path(args.run_dir).rglob("result.json"))
             if not wanted or json.loads(p.read_text())["task"].split("[")[0] in wanted]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        tags = list(pool.map(lambda p: regrade_one(p, args), paths))
    print(f"regraded {len(tags)} trials: {tags.count('same')} identical, {tags.count('CHANGED')} changed")


def regrade_one(result_path, args):
    """Re-verify one trial; return "same" or "CHANGED"."""
    result = json.loads(result_path.read_text())
    task = result["task"].split("[")[0]
    trial_dir = result_path.parent
    app_dir = trial_dir / result.get("workspace", "app")
    task_dir = Path(args.tasks_root) / task
    reward, metrics = verify(task_dir, trial_dir, app_dir, args.sandbox_user)
    new_score = metrics.get("score", 0.0)
    tag = "same" if (reward, round(new_score, 9)) == (result["reward"], round(result["score"], 9)) else "CHANGED"
    print(f"{tag:7s} {trial_dir.relative_to(args.run_dir)}: reward {result['reward']}->{reward} "
          f"score {result['score']:.3f}->{new_score:.3f}", flush=True)
    new_digest = task_digest(task_dir)
    # Rewrite when the outcome or the task version changed, so stored diagnostics stay current
    if args.write and (tag == "CHANGED" or new_digest != result["task_digest"]):
        result["previous"] = {k: result[k] for k in ("reward", "score", "task_digest")}
        result.update(reward=reward, score=new_score, metrics=metrics, task_digest=new_digest)
        result_path.write_text(json.dumps(result, indent=1))
    return tag


if __name__ == "__main__":
    main()
