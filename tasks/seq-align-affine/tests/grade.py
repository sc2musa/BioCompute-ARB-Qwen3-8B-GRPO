"""Verifier for seq-align-affine.

Usage: python grade.py --workspace /app --out /logs/verifier
Writes reward.txt (1 = all hidden cases pass, else 0) and metrics.json (partial scores).
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbox_worker import run_submission  # noqa: E402

CORRECTNESS_TIMEOUT_SEC = 300
PERF_TIMEOUT_SEC = 60
# Ready-made aligners would trivialise the task; the instruction forbids them.
BANNED_PATTERNS = [rf"^\s*(from|import)\s+{lib}\b" for lib in
                   ("Bio", "parasail", "edlib", "pyopa", "skbio", "swalign", "pairwise2")]


def rescore(a, b, p):
    score, prev = 0, None
    for x, y in zip(a, b):
        if x != "-" and y != "-":
            score += p["match"] if x == y else p["mismatch"]
            prev = None
            continue
        kind = "a" if x == "-" else "b"
        score += p["gap_extend"] if prev == kind else p["gap_open"]
        prev = kind
    return score


def check_case(case, result):
    """Return (passed, reason)."""
    if not result.get("ok"):
        return False, "error: " + str(result.get("error", ""))[-300:]
    value = result["value"]
    if not (isinstance(value, list) and len(value) == 3):
        return False, "return value must be (score, aligned_a, aligned_b)"
    score, al_a, al_b = value
    args = case["args"]
    p = {k: args[k] for k in ("match", "mismatch", "gap_open", "gap_extend")}
    if not isinstance(score, (int, float)) or score != case["expected_score"]:
        return False, f"score {score!r} != optimal {case['expected_score']}"
    if not (isinstance(al_a, str) and isinstance(al_b, str)) or len(al_a) != len(al_b):
        return False, "aligned strings must be str of equal length"
    if any(x == "-" and y == "-" for x, y in zip(al_a, al_b)):
        return False, "column with two gaps"
    raw_a, raw_b = al_a.replace("-", ""), al_b.replace("-", "")
    if args["mode"] == "global":
        if raw_a != args["seq_a"] or raw_b != args["seq_b"]:
            return False, "global alignment does not spell the full input sequences"
    elif raw_a not in args["seq_a"] or raw_b not in args["seq_b"]:
        return False, "local alignment is not a pair of substrings"
    if rescore(al_a, al_b, p) != score:
        return False, "reported score does not match the returned alignment"
    return True, "ok"


def static_scan(workspace):
    hits = []
    for path in Path(workspace).rglob("*.py"):
        text = path.read_text(errors="ignore")
        hits += [f"{path.name}: {pat}" for pat in BANNED_PATTERNS
                 if re.search(pat, text, re.MULTILINE)]
    return hits


def grade(workspace):
    cases = json.loads((Path(__file__).resolve().parent / "cases.json").read_text())
    module = Path(workspace) / "align.py"
    banned = static_scan(workspace)
    if not module.exists() or banned:
        reason = "align.py missing" if not module.exists() else f"banned library use: {banned}"
        return {"pass": False, "score": 0.0, "reason": reason}
    details = []
    results = run_submission(module, "align", [c["args"] for c in cases["cases"]],
                             CORRECTNESS_TIMEOUT_SEC, workdir=workspace)
    for case, res in zip(cases["cases"], results):
        ok, why = check_case(case, res)
        details.append({"id": case["id"], "pass": ok, "reason": why, "seconds": res.get("seconds")})
    for case in cases["perf_cases"]:  # one process per perf case, own time limit
        res = run_submission(module, "align", [case["args"]], PERF_TIMEOUT_SEC, workdir=workspace)[0]
        ok, why = check_case(case, res)
        details.append({"id": case["id"], "pass": ok, "reason": why, "seconds": res.get("seconds")})
    n_pass = sum(d["pass"] for d in details)
    return {
        "pass": n_pass == len(details),
        "score": n_pass / len(details),
        "n_pass": n_pass,
        "n_cases": len(details),
        "correctness_pass_rate": sum(d["pass"] for d in details if not d["id"].startswith("perf"))
        / len(cases["cases"]),
        "perf_seconds": [d["seconds"] for d in details if d["id"].startswith("perf")],
        "failures": [d for d in details if not d["pass"]][:10],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="/app")
    ap.add_argument("--out", default="/logs/verifier")
    args = ap.parse_args()
    metrics = grade(args.workspace)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.txt").write_text("1" if metrics["pass"] else "0")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: v for k, v in metrics.items() if k != "failures"}))


if __name__ == "__main__":
    main()
