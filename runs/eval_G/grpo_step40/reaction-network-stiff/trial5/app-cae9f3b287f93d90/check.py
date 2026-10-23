"""Public self-check: run simulate() on the networks in examples/ and report the error.

Usage: python check.py   (from the workspace). Uses only public data and the same accuracy
formula as the hidden tests: |y - y_ref| <= 1e-4 * (|y_ref| + 1e-3 * max_t |y_ref(species)|).
"""

import glob
import json
import time
import traceback

import numpy as np

from simulate import simulate

RTOL, FLOOR, TIME_LIMIT = 1e-4, 1e-3, 60.0


def main():
    files = sorted(glob.glob("examples/*.json"))
    n_ok = 0
    for path in files:
        case = json.load(open(path))
        t0 = time.perf_counter()
        try:
            y = np.asarray(simulate(case["network"], case["t_eval"]), dtype=float)
            seconds = time.perf_counter() - t0
            ref = np.asarray(case["reference"])
            if y.shape != ref.shape:
                problem = f"shape {y.shape} != expected {ref.shape}"
            else:
                scale = np.abs(ref).max(axis=0)
                err = np.abs(y - ref) / (np.abs(ref) + FLOOR * scale + 1e-300)
                worst = np.unravel_index(np.nanargmax(err), err.shape)
                species = case["network"]["species"][worst[1]]["name"]
                problem = None if err.max() <= RTOL and seconds <= TIME_LIMIT else (
                    f"max scaled error {err.max():.2e} (limit {RTOL:.0e}) at t={case['t_eval'][worst[0]]:g}, "
                    f"species {species}; {seconds:.2f}s")
        except Exception:
            seconds, problem = time.perf_counter() - t0, "exception:\n" + traceback.format_exc(limit=3)
        n_ok += problem is None
        print(f"{'PASS' if problem is None else 'FAIL'} {path} ({seconds:.2f}s)"
              + ("" if problem is None else f": {problem}"))
    print(f"{n_ok}/{len(files)} public examples pass")


if __name__ == "__main__":
    main()
