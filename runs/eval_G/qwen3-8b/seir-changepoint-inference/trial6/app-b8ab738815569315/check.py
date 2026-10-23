"""Public self-check: run fit() on data/example*.json and compare with their ground truth.

Usage: python check.py   (from the workspace). Uses only public data and the same criteria as
the hidden tests (R0 / R_post within 10%, change_day within 3 days for clear interventions,
forecast relative MAE <= 0.30 against the noise-free expected cases).
"""

import glob
import json
import math
import time
import traceback

from fit import fit


def criteria(out, truth):
    res = {}
    for key in ("R0", "R_post"):
        v = out.get(key)
        res[key] = (abs(v / truth[key] - 1) if isinstance(v, (int, float)) and math.isfinite(v) else None, 0.10)
    if truth["change_day"] is not None and truth["R_post"] <= 0.75 * truth["R0"]:
        v = out.get("change_day")
        res["change_day"] = (abs(v - truth["change_day"]) if isinstance(v, (int, float)) else None, 3.0)
    fc, mu = out.get("forecast"), truth["expected_forecast"]
    ok_fc = isinstance(fc, (list, tuple)) and len(fc) == len(mu)
    res["forecast_rmae"] = (sum(abs(a - b) for a, b in zip(fc, mu)) / sum(mu) if ok_fc else None, 0.30)
    return res


def main():
    files = sorted(glob.glob("data/example*.json"))
    n_ok = 0
    for path in files:
        case = json.load(open(path))
        t0 = time.perf_counter()
        try:
            out = fit(**case["inputs"])
            if hasattr(out.get("forecast"), "tolist"):
                out["forecast"] = out["forecast"].tolist()
            res = criteria(out, case["truth"])
            failed = {k: v for k, v in res.items() if v[0] is None or v[0] > v[1]}
            problem = None if not failed else ", ".join(
                f"{k} error {'missing/invalid' if e is None else f'{e:.3g}'} (limit {lim})"
                for k, (e, lim) in failed.items())
            detail = (f" estimates R0={out.get('R0')}, R_post={out.get('R_post')}, "
                      f"change_day={out.get('change_day')}; truth R0={case['truth']['R0']:.3f}, "
                      f"R_post={case['truth']['R_post']:.3f}, change_day={case['truth']['change_day']}")
        except Exception:
            problem, detail = "exception:\n" + traceback.format_exc(limit=3), ""
        n_ok += problem is None
        print(f"{'PASS' if problem is None else 'FAIL'} {path} ({time.perf_counter() - t0:.1f}s)"
              + ("" if problem is None else f": {problem}") + "\n   " + detail)
    print(f"{n_ok}/{len(files)} public examples pass "
          "(hidden tests require 10 of 12 outbreaks; weak interventions are hard even for a "
          "good estimator, so one miss is not necessarily a bug)")


if __name__ == "__main__":
    main()
