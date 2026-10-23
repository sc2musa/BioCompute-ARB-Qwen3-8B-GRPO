"""Verifier for seir-changepoint-inference.

Usage: python grade.py --workspace /app --out /logs/verifier
The submitted fit() runs on each hidden outbreak in its own child process; estimates are
compared with the hidden ground truth. Thresholds were calibrated on the reference
solution (see docs/design.md).
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbox_worker import run_submission  # noqa: E402

TIMEOUT_PER_OUTBREAK_SEC = 120
R_REL_TOL = 0.10
CHANGE_DAY_TOL = 3.0
CLEAR_DROP = 0.25          # change_day is only scored when R drops by at least 25%
FORECAST_RMAE_TOL = 0.30
MIN_PASS_FRACTION = 10 / 12


def _num(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def check(case, result):
    """Return per-criterion booleans plus raw errors for one outbreak."""
    if not result.get("ok"):
        return {"error": str(result.get("error", ""))[-300:]}
    out, truth = result["value"], case["truth"]
    if not isinstance(out, dict):
        return {"error": "fit() must return a dict"}
    crit, errs = {}, {}
    for key in ("R0", "R_post"):
        v = out.get(key)
        errs[key] = abs(v / truth[key] - 1.0) if _num(v) else None
        crit[key] = errs[key] is not None and errs[key] <= R_REL_TOL
    if truth["change_day"] is not None and truth["R_post"] <= (1 - CLEAR_DROP) * truth["R0"]:
        v = out.get("change_day")
        errs["change_day"] = abs(v - truth["change_day"]) if _num(v) else None
        crit["change_day"] = errs["change_day"] is not None and errs["change_day"] <= CHANGE_DAY_TOL
    fc, mu = out.get("forecast"), truth["expected_forecast"]
    if isinstance(fc, list) and len(fc) == len(mu) and all(_num(x) for x in fc):
        errs["forecast_rmae"] = sum(abs(a - b) for a, b in zip(fc, mu)) / sum(mu)
        crit["forecast"] = errs["forecast_rmae"] <= FORECAST_RMAE_TOL
    else:
        errs["forecast_rmae"], crit["forecast"] = None, False
    return {"criteria": crit, "errors": errs}


def grade(workspace):
    cases = json.loads((Path(__file__).resolve().parent / "cases.json").read_text())
    module = Path(workspace) / "fit.py"
    if not module.exists():
        return {"pass": False, "score": 0.0, "reason": "fit.py missing"}
    details = []
    for case in cases:
        res = run_submission(module, "fit", [case["inputs"]], TIMEOUT_PER_OUTBREAK_SEC,
                             workdir=workspace)[0]
        d = {"id": case["id"], "seconds": res.get("seconds"), **check(case, res)}
        crit = d.get("criteria", {})
        d["pass"] = bool(crit) and all(crit.values())
        d["score"] = sum(crit.values()) / len(crit) if crit else 0.0
        details.append(d)
    n_pass = sum(d["pass"] for d in details)
    return {
        "pass": n_pass >= MIN_PASS_FRACTION * len(details),
        "score": sum(d["score"] for d in details) / len(details),
        "n_pass": n_pass,
        "n_cases": len(details),
        "criterion_pass_rate": {
            k: sum(d["criteria"][k] for d in details if k in d.get("criteria", {}))
            / max(1, sum(k in d.get("criteria", {}) for d in details))
            for k in ("R0", "R_post", "change_day", "forecast")
        },
        "total_seconds": sum(d["seconds"] or TIMEOUT_PER_OUTBREAK_SEC for d in details),
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
    print(json.dumps({k: v for k, v in metrics.items() if k != "details"}))


if __name__ == "__main__":
    main()
