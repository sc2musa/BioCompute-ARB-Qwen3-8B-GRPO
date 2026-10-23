"""Run benchmark tasks end to end: agent phase in a fresh workspace, then independent grading.

Examples (from the repo root):
    python -m harness.runner --agent oracle --tasks all --out runs/oracle
    python -m harness.runner --agent null --tasks all --out runs/null
    python -m harness.runner --agent baseline --tasks all --out runs/baselines
    python -m harness.runner --agent llm --tasks seq-align-affine --trials 5 \
        --base-url http://127.0.0.1:8000/v1 --model qwen3-8b --sandbox-user agent --out runs/qwen3-8b

Layout per trial: <out>/<task>/<trial>/{app-<random>/, verify/, trajectory.jsonl, result.json}.
The agent never sees the task directory; grading copies the final workspace into
verify/app and runs tests/test.sh there (the submission is executed as the sandbox user).

Guidance levels (cf. ASI-Bench): `goal` gives only instruction.md; `method` appends the
task's guidance.md with the intended method.

Isolation with --sandbox-user (runner as root): task directories become root-only, every
trial directory is traverse-only (0711) and the workspace has a random name, so concurrent
trials cannot find each other's workspaces; all other runner output is root-only (umask 077).
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

from harness.agent import AgentConfig, run_agent
from harness.sandbox import (MEMORY_LIMIT_KB, SINGLE_THREAD_ENV, Sandbox, cpu_slot, give_to_user,
                             kill_leftovers, lock_hidden_dirs, run_in_group, set_cpu_slots)

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "tasks"


def task_digest(task_dir):
    """Content hash of a task directory, recorded for reproducibility."""
    h = hashlib.sha256()
    for path in sorted(Path(task_dir).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            h.update(str(path.relative_to(task_dir)).replace("\\", "/").encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def entrypoint_file(task_dir):
    for line in (Path(task_dir) / "task.toml").read_text().splitlines():
        if line.strip().startswith("entrypoint"):
            return line.split("=", 1)[1].strip().strip('"').split(":")[0]
    raise ValueError(f"no entrypoint in {task_dir}/task.toml")


def verifier_timeout(task_dir, default=1800.0):
    """[verifier] timeout_sec from task.toml (Harbor field)."""
    section = None
    for line in (Path(task_dir) / "task.toml").read_text().splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line
        elif section == "[verifier]" and line.startswith("timeout_sec"):
            return float(line.split("=", 1)[1])
    return default


def run_baseline_agent(kind, task_dir, app_dir):
    """Non-LLM agents: oracle (reference solution), null (untouched stub), baseline files."""
    if kind == "oracle":
        env = dict(os.environ, WORKSPACE_DIR=str(app_dir), SOLUTION_DIR=str(task_dir / "solution"))
        subprocess.run(["bash", str(task_dir / "solution" / "solve.sh")], env=env, check=True)
    elif kind.startswith("baseline:"):
        shutil.copy(task_dir / "baselines" / kind.split(":", 1)[1], app_dir / entrypoint_file(task_dir))
    elif kind != "null":
        raise ValueError(kind)
    return {"agent": kind}


def verify(task_dir, trial_dir, app_dir, sandbox_user):
    """Grade the final workspace in a fresh directory; return (reward, metrics)."""
    verify_dir = trial_dir / "verify"
    shutil.rmtree(verify_dir, ignore_errors=True)
    shutil.copytree(app_dir, verify_dir / "app", symlinks=True)
    out_dir = verify_dir / "logs"
    # Single-threaded BLAS: the submission's own time limits must not depend on host load
    env = dict(os.environ, TESTS_DIR=str(task_dir / "tests"), WORKSPACE_DIR=str(verify_dir / "app"),
               VERIFIER_LOG_DIR=str(out_dir), PYTHON=sys.executable, **SINGLE_THREAD_ENV)
    if sandbox_user:
        os.chmod(verify_dir, 0o711)
        give_to_user(verify_dir / "app", sandbox_user)
        env["GRADER_SANDBOX_PREFIX"] = (f"runuser -u {sandbox_user} -- "
                                        f"prlimit --as={MEMORY_LIMIT_KB * 1024} --")
    start = time.time()
    timeout = verifier_timeout(task_dir)
    with cpu_slot():
        code, stdout, stderr = run_in_group(["bash", str(task_dir / "tests" / "test.sh")],
                                            timeout=timeout, env=env, text=True, errors="replace")
    kill_leftovers(verify_dir, sandbox_user)
    (verify_dir / "verifier_stdout.txt").write_text(stdout + "\n[stderr]\n" + stderr)
    reward_file, metrics_file = out_dir / "reward.txt", out_dir / "metrics.json"
    reward = int(reward_file.read_text().strip()) if reward_file.exists() else 0
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {
        "error": f"verifier timed out after {timeout:.0f}s" if code is None
        else f"verifier failed (exit {code})"}
    metrics["verifier_seconds"] = round(time.time() - start, 1)
    return reward, metrics


PUBLIC_CHECK_NOTE = """

## Self-check script

`/app/check.py` runs your implementation on the public examples and applies the same
criteria as the hidden tests. Run `python check.py` after every change and read its output;
the hidden tests use different data, so passing it is necessary but not sufficient."""


def build_instruction(task_dir, guidance, app_dir, public_check=False):
    text = (task_dir / "instruction.md").read_text()
    if guidance == "method":
        text += "\n\n" + (task_dir / "guidance.md").read_text()
    elif guidance != "goal":
        raise ValueError(guidance)
    if public_check:
        text += PUBLIC_CHECK_NOTE
    return text.replace("/app", str(app_dir))


def run_trial(task_dir, trial_dir, args, trial_idx):
    shutil.rmtree(trial_dir, ignore_errors=True)
    trial_dir.mkdir(parents=True)
    app_dir = trial_dir / f"app-{secrets.token_hex(8)}"
    shutil.copytree(task_dir / "environment" / "workspace", app_dir)
    if getattr(args, "public_check", False):
        shutil.copy(task_dir / "public_check.py", app_dir / "check.py")
    if args.sandbox_user:
        os.chmod(trial_dir, 0o711)
        give_to_user(app_dir, args.sandbox_user)
    started = time.time()
    if args.agent == "llm":
        # Qwen3 recommended sampling: thinking T=0.6/top_p=0.95, non-thinking T=0.7/top_p=0.8
        # Several comma-separated endpoints (one vLLM per GPU) are used round-robin per trial
        urls = args.base_url.split(",")
        base_url = urls[(zlib.crc32(str(trial_dir).encode()) + trial_idx) % len(urls)]
        config = AgentConfig(base_url, args.model, max_turns=args.max_turns,
                             temperature=0.6 if args.thinking else 0.7,
                             top_p=0.95 if args.thinking else 0.8,
                             max_tokens=12288 if args.thinking else 4096,
                             enable_thinking=args.thinking,
                             seed=None if args.seed is None else args.seed + trial_idx,
                             time_budget_sec=args.time_budget, repeat_guard=args.repeat_guard,
                             presence_penalty=args.presence_penalty)
        sandbox = Sandbox(app_dir, user=args.sandbox_user, python_bin=sys.executable)
        instruction = build_instruction(task_dir, args.guidance, app_dir, args.public_check)
        agent_stats = run_agent(instruction, sandbox, config, trial_dir / "trajectory.jsonl")
        agent_stats.update(agent="llm", model=args.model, thinking=args.thinking,
                           guidance=args.guidance, repeat_guard=args.repeat_guard,
                           presence_penalty=args.presence_penalty,
                           public_check=args.public_check, isolated=sandbox.isolated)
    else:
        agent_stats = run_baseline_agent(args.agent, task_dir, app_dir)
    agent_seconds = round(time.time() - started, 1)
    kill_leftovers(app_dir, args.sandbox_user)
    reward, metrics = verify(task_dir, trial_dir, app_dir, args.sandbox_user)
    result = {"task": task_dir.name, "trial": trial_idx, "task_digest": task_digest(task_dir),
              "workspace": app_dir.name, "reward": reward, "score": metrics.get("score", 0.0),
              "agent_seconds": agent_seconds,
              "agent": agent_stats, "metrics": metrics}
    (trial_dir / "result.json").write_text(json.dumps(result, indent=1))
    return result


def summarize(results):
    by_task = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)
    summary = {}
    for task, rs in by_task.items():
        n = len(rs)
        summary[task] = {
            "trials": n,
            "pass_rate": sum(r["reward"] for r in rs) / n,
            "pass_at_k": int(any(r["reward"] for r in rs)),
            "mean_score": sum(r["score"] for r in rs) / n,
            "scores": [round(r["score"], 3) for r in rs],
            "mean_agent_seconds": sum(r["agent_seconds"] for r in rs) / n,
            "mean_tool_calls": sum(r["agent"].get("tool_calls", 0) for r in rs) / n,
            "mean_tokens": sum(r["agent"].get("prompt_tokens", 0) + r["agent"].get("completion_tokens", 0)
                               for r in rs) / n,
            "submit_rate": sum(bool(r["agent"].get("submitted")) for r in rs) / n,
            "mean_repeated_calls": sum(r["agent"].get("repeated_calls", 0) for r in rs) / n,
            "stop_reasons": {s: sum(r["agent"].get("stop_reason", "").split(":")[0] == s for r in rs)
                             for s in {r["agent"].get("stop_reason", "").split(":")[0] for r in rs}},
            "task_digest": rs[0]["task_digest"],
        }
    return summary


def raise_open_file_limit():
    """Use the hard RLIMIT_NOFILE: hundreds of concurrent trials (pipes, logs) exceed 1024."""
    try:
        import resource  # POSIX only
    except ImportError:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="all", help="comma-separated task names or 'all'")
    ap.add_argument("--agent", default="oracle",
                    help="oracle | null | baseline (all files in baselines/) | baseline:<file> | llm")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sandbox-user", default=None, help="unprivileged user for agent + submission")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                    help="OpenAI-compatible endpoint(s); comma-separated list = spread trials over them")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--thinking", action="store_true",
                    help="enable Qwen3 thinking mode (also switches to its recommended sampling)")
    ap.add_argument("--repeat-guard", action="store_true",
                    help="warn the agent when it repeats the same tool call 3 times in 8 calls")
    ap.add_argument("--public-check", action="store_true",
                    help="give the agent check.py, a self-check on the public examples")
    ap.add_argument("--presence-penalty", type=float, default=0.0,
                    help="sampling penalty against repeated text (training sampling only; the "
                         "benchmark configurations use 0)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--time-budget", type=int, default=1800, help="agent wall-clock budget (s)")
    ap.add_argument("--guidance", choices=["goal", "method"], default="goal",
                    help="goal: instruction only; method: instruction + guidance.md")
    ap.add_argument("--workers", type=int, default=1,
                    help="trials to run concurrently (mostly waiting for the model)")
    ap.add_argument("--cpu-slots", type=int, default=0,
                    help="max concurrent agent commands + gradings across trials (0 = unlimited); "
                         "keep below the core count so --workers can exceed it")
    ap.add_argument("--resume", action="store_true",
                    help="skip trials that already have result.json (rerun unfinished ones)")
    ap.add_argument("--tasks-root", default=str(TASKS_DIR),
                    help="directory of task folders (default: the benchmark's tasks/; training "
                         "instances from training/make_train_tasks.py live elsewhere)")
    args = ap.parse_args()
    set_cpu_slots(args.cpu_slots)
    raise_open_file_limit()

    tasks_root = Path(args.tasks_root).resolve()
    names = sorted(p.name for p in tasks_root.iterdir() if (p / "task.toml").exists())
    if args.tasks != "all":
        names = args.tasks.split(",")
    if args.sandbox_user:
        os.umask(0o077)
        os.chmod(tasks_root, 0o711)  # traversable but not listable by the agent
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in names:
        task_dir = tasks_root / name
        if args.sandbox_user:
            lock_hidden_dirs(task_dir)
            (out / name).mkdir(exist_ok=True)
            os.chmod(out / name, 0o711)
        agents = [args.agent]
        if args.agent == "baseline":
            agents = [f"baseline:{p.name}" for p in sorted((task_dir / "baselines").glob("*.py"))]
        for agent in agents:
            run_args = argparse.Namespace(**{**vars(args), "agent": agent})
            label = agent.replace("baseline:", "").replace(".py", "")
            for t in range(args.trials):
                trial_dir = out / name / (f"{label}_{t}" if args.agent == "baseline" else f"trial{t}")
                task_label = f"{name}[{label}]" if args.agent == "baseline" else name
                jobs.append((task_dir, trial_dir, run_args, t, task_label))
    if args.sandbox_user:
        os.chmod(out, 0o711)

    def run_job(job):
        task_dir, trial_dir, run_args, t, task_label = job
        done = trial_dir / "result.json"
        if args.resume and done.exists():  # finished before an interruption: keep it
            return json.loads(done.read_text())
        r = run_trial(task_dir, trial_dir, run_args, t)
        r["task"] = task_label
        print(f"[{task_label} #{t}] reward={r['reward']} score={r['score']:.3f} "
              f"agent={r['agent_seconds']}s", flush=True)
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_job, jobs))
    summary = summarize(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
