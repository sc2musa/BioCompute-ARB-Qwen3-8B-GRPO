"""Verifier for reaction-network-stiff.

Usage: python grade.py --workspace /app --out /logs/verifier
Each hidden network runs in its own child process with a wall-clock limit. A trajectory
passes when |y - y_ref| <= RTOL * (|y_ref| + FLOOR * max_t |y_ref_species|) everywhere.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbox_worker import run_submission  # noqa: E402

RTOL = 1e-4
FLOOR = 1e-3
TIMEOUT_PER_NETWORK_SEC = 60


def check(case, result):
    """Return (passed, max_scaled_error or None, reason)."""
    if not result.get("ok"):
        return False, None, "error: " + str(result.get("error", ""))[-300:]
    ref = np.asarray(case["reference"], dtype=float)
    try:
        y = np.asarray(result["value"], dtype=float)
    except (TypeError, ValueError):
        return False, None, "return value is not a numeric array"
    if y.shape != ref.shape:
        return False, None, f"shape {y.shape} != expected {ref.shape}"
    if not np.all(np.isfinite(y)):
        return False, None, "non-finite values"
    scale = np.abs(ref).max(axis=0)
    err = float((np.abs(y - ref) / (np.abs(ref) + FLOOR * scale + 1e-300)).max())
    if err > RTOL:
        return False, err, f"max scaled error {err:.2e} > {RTOL:.0e}"
    return True, err, "ok"


def grade(workspace):
    cases = json.loads((Path(__file__).resolve().parent / "cases.json").read_text())
    module = Path(workspace) / "simulate.py"
    if not module.exists():
        return {"pass": False, "score": 0.0, "reason": "simulate.py missing"}
    details = []
    for case in cases:
        call = {"network": case["network"], "t_eval": case["t_eval"]}
        res = run_submission(module, "simulate", [call], TIMEOUT_PER_NETWORK_SEC, workdir=workspace)[0]
        ok, err, why = check(case, res)
        details.append({"id": case["network"]["name"], "pass": ok, "max_scaled_error": err,
                        "seconds": res.get("seconds"), "reason": why})
    n_pass = sum(d["pass"] for d in details)
    return {
        "pass": n_pass == len(details),
        "score": n_pass / len(details),
        "n_pass": n_pass,
        "n_cases": len(details),
        "total_seconds": sum(d["seconds"] or TIMEOUT_PER_NETWORK_SEC for d in details),
        "details": details,
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
    summary = {k: v for k, v in metrics.items() if k != "details"}
    summary["per_case"] = {d["id"]: d["pass"] for d in metrics.get("details", [])}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
