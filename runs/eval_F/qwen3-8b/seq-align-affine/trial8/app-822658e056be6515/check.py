"""Public self-check: run align() on the public examples in examples.json.

Usage: python check.py   (from the workspace). Uses only public data; the hidden tests apply
the same checks to different sequences and parameters.
"""

import json
import time
import traceback

from align import align


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


def check(case):
    args = case["args"]
    score, al_a, al_b = align(**args)
    p = {k: args[k] for k in ("match", "mismatch", "gap_open", "gap_extend")}
    if score != case["expected_score"]:
        return f"score {score} != optimal {case['expected_score']}"
    if len(al_a) != len(al_b) or any(x == y == "-" for x, y in zip(al_a, al_b)):
        return "aligned rows have different lengths or a double-gap column"
    raw_a, raw_b = al_a.replace("-", ""), al_b.replace("-", "")
    if args["mode"] == "global" and (raw_a, raw_b) != (args["seq_a"], args["seq_b"]):
        return "global alignment does not spell the full inputs"
    if args["mode"] == "local" and (raw_a not in args["seq_a"] or raw_b not in args["seq_b"]):
        return "local alignment rows are not substrings of the inputs"
    if rescore(al_a, al_b, p) != score:
        return f"returned rows re-score to {rescore(al_a, al_b, p)}, not {score}"
    return None


def main():
    cases = json.load(open("examples.json"))
    n_ok = 0
    for case in cases:
        t0 = time.perf_counter()
        try:
            problem = check(case)
        except Exception:
            problem = "exception:\n" + traceback.format_exc(limit=2)
        status = "PASS" if problem is None else "FAIL"
        n_ok += problem is None
        print(f"{status} {case['id']} ({case['args']['mode']}, {time.perf_counter() - t0:.2f}s)"
              + ("" if problem is None else f": {problem}"))
    print(f"{n_ok}/{len(cases)} public examples pass")


if __name__ == "__main__":
    main()
