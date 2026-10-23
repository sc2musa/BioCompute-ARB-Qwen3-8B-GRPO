"""Aggregate an experiment matrix into tables and failure-mode counts.

Usage: python scripts/analyze.py <matrix-dir> [--json out.json]
<matrix-dir> holds one sub-directory per configuration (as written by scripts/run_matrix.sh),
each containing <task>/<trial>/{result.json, trajectory.jsonl, app-*/}.
Failure modes are assigned by transparent heuristics (see classify()).
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

TASK_ORDER = ["seq-align-affine", "reaction-network-stiff", "seir-changepoint-inference"]
ENTRYPOINTS = {"seq-align-affine": "align.py", "reaction-network-stiff": "simulate.py",
               "seir-changepoint-inference": "fit.py"}
LOOP_THRESHOLD = 10


def final_code(trial_dir, task):
    apps = sorted(trial_dir.glob("app-*"))
    path = apps[0] / ENTRYPOINTS[task] if apps else None
    return path.read_text(errors="replace") if path and path.exists() else ""


def grader_errors(metrics):
    rows = metrics.get("details") or metrics.get("failures") or []
    return [str(r.get("error") or r.get("reason") or "") for r in rows]


def method_tags(task, code):
    """Which numerical approach the final code uses (T2/T3 only)."""
    tags = []
    if task == "reaction-network-stiff":
        if re.search(r"""method\s*=\s*['"](BDF|Radau|LSODA)""", code) or "odeint" in code:
            tags.append("implicit")
        elif "solve_ivp" in code:
            tags.append("explicit(default RK45)")
        if re.search(r"rtol\s*=\s*1e-([6-9]|1\d)", code):
            tags.append("tight-rtol")
    if task == "seir-changepoint-inference":
        if re.search(r"for\s+\w+\s+in\s+range\(.*\)", code) and "change" in code:
            tags.append("scans-change-day")
        if re.search(r"bic|aic", code, re.I):
            tags.append("model-selection")
    return tags


def classify(task, result, code):
    agent, metrics = result["agent"], result["metrics"]
    errs = " ".join(grader_errors(metrics))
    if result["reward"] == 1:
        return "solved"
    if str(agent.get("stop_reason", "")).startswith("api_error"):
        return "api/context error"
    if "NotImplementedError" in errs or not code.strip():
        return "no implementation"
    if "timeout after" in errs and result["score"] < 1:
        return "too slow (grader timeout)"
    if re.search(r"error: .*(Traceback|Error)", errs) and result["score"] == 0:
        return "crashes on hidden inputs"
    if agent.get("repeated_calls", 0) >= LOOP_THRESHOLD and not agent.get("submitted"):
        return "stuck in repetition loop"
    if agent.get("submitted"):
        return "submitted wrong/partial solution"
    return "ran out of turns/time"


def behaviour(trial_dir):
    """Verification behaviour from the trajectory.

    silent_submit: the agent submitted right after a command that exited 0 with no output
    (e.g. `python fit.py` on a module without a main) -- it mistook "no crash" for "correct".
    """
    events = [json.loads(line) for line in (trial_dir / "trajectory.jsonl").open(encoding="utf-8")]
    tools = [e for e in events if e["type"] == "tool"]
    runs = [e for e in tools if e["name"] == "bash"]
    empty = [e for e in runs if re.sub(r"\[harness note\].*", "", e["output"], flags=re.S).strip() == "[exit code 0]"]
    last_run = None
    for e in tools:
        if e["name"] == "submit":
            break
        if e["name"] == "bash":
            last_run = e
    submitted = any(e["name"] == "submit" for e in tools)
    silent = submitted and last_run is not None and last_run in empty
    truncated = sum(e["type"] == "assistant" and e.get("finish_reason") == "length" for e in events)
    used_check = any("check.py" in (e.get("arguments") or "") for e in runs)
    return {"bash_runs": len(runs), "empty_runs": len(empty), "silent_submit": silent,
            "truncated": truncated, "used_check": used_check}


def load(matrix_dir):
    rows = []
    for config_dir in sorted(p for p in Path(matrix_dir).iterdir() if p.is_dir()):
        for result_path in sorted(config_dir.glob("*/trial*/result.json")):
            result = json.loads(result_path.read_text())
            task = result["task"]
            code = final_code(result_path.parent, task)
            rows.append({"config": config_dir.name, "task": task, "result": result,
                         "mode": classify(task, result, code), "methods": method_tags(task, code),
                         "behaviour": behaviour(result_path.parent)})
    return rows


def behaviour_table(rows):
    lines = ["| config | trials | mean score | silent-success submit | empty runs / trial | "
             "truncated replies | used check.py |", "|---|---|---|---|---|---|---|"]
    for c in sorted({r["config"] for r in rows}):
        rs = [r for r in rows if r["config"] == c]
        b = [r["behaviour"] for r in rs]
        n = len(rs)
        lines.append(f"| {c} | {n} | {sum(r['result']['score'] for r in rs) / n:.3f} "
                     f"| {sum(x['silent_submit'] for x in b)}/{n} "
                     f"| {sum(x['empty_runs'] for x in b) / n:.1f} "
                     f"| {sum(x['truncated'] for x in b)} "
                     f"| {sum(x['used_check'] for x in b)}/{n} |")
    return "\n".join(lines)


def table(rows):
    configs = sorted({r["config"] for r in rows})
    lines = ["| config | task | pass | mean score | submit | tool calls | repeats | tokens (k) | agent s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in configs:
        for t in TASK_ORDER:
            rs = [r for r in rows if r["config"] == c and r["task"] == t]
            if not rs:
                continue
            n = len(rs)
            a = [r["result"]["agent"] for r in rs]
            lines.append(
                f"| {c} | {t} | {sum(r['result']['reward'] for r in rs)}/{n} "
                f"| {sum(r['result']['score'] for r in rs) / n:.2f} "
                f"| {sum(bool(x.get('submitted')) for x in a)}/{n} "
                f"| {sum(x.get('tool_calls', 0) for x in a) / n:.1f} "
                f"| {sum(x.get('repeated_calls', 0) for x in a) / n:.1f} "
                f"| {sum(x.get('prompt_tokens', 0) + x.get('completion_tokens', 0) for x in a) / n / 1000:.0f} "
                f"| {sum(r['result']['agent_seconds'] for r in rs) / n:.0f} |")
    return "\n".join(lines)


def failure_table(rows):
    modes = Counter(r["mode"] for r in rows)
    lines = ["| failure mode | " + " | ".join(TASK_ORDER) + " | total |", "|---" * (len(TASK_ORDER) + 2) + "|"]
    for mode, total in modes.most_common():
        per = [sum(r["mode"] == mode and r["task"] == t for r in rows) for t in TASK_ORDER]
        lines.append(f"| {mode} | " + " | ".join(map(str, per)) + f" | {total} |")
    return "\n".join(lines)


def method_table(rows):
    lines = ["| config | task | approaches in final code (count) |", "|---|---|---|"]
    for c in sorted({r["config"] for r in rows}):
        for t in TASK_ORDER[1:]:
            tags = Counter(tag for r in rows if r["config"] == c and r["task"] == t for tag in r["methods"])
            if any(r["config"] == c and r["task"] == t for r in rows):
                lines.append(f"| {c} | {t} | " + (", ".join(f"{k}: {v}" for k, v in tags.items()) or "-") + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix_dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = load(args.matrix_dir)
    print(f"## Results ({len(rows)} trials)\n\n" + table(rows))
    print("\n## Failure modes\n\n" + failure_table(rows))
    print("\n## Numerical approach in final code\n\n" + method_table(rows))
    print("\n## Verification behaviour\n\n" + behaviour_table(rows))
    if args.json:
        Path(args.json).write_text(json.dumps(
            [{"config": r["config"], "task": r["task"], "reward": r["result"]["reward"],
              "score": r["result"]["score"], "mode": r["mode"], "methods": r["methods"],
              "behaviour": r["behaviour"], "agent": r["result"]["agent"]} for r in rows], indent=1))


if __name__ == "__main__":
    main()
