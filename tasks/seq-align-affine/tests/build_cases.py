"""Author-side script: generate hidden test cases for seq-align-affine.

Not used at grading time. Run from the repo root:
    python tasks/seq-align-affine/tests/build_cases.py
Writes tests/cases.json (hidden) and environment/workspace/examples.json (public).
Reference scores come from solution/align.py, cross-checked by brute force on tiny inputs.
"""

import itertools
import json
import random
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR / "solution"))
from align import align  # noqa: E402

SEED = 20261020
ALPHABET = "ACGT"
PARAM_SETS = [
    dict(match=2, mismatch=-3, gap_open=-5, gap_extend=-2),
    dict(match=1, mismatch=-1, gap_open=-2, gap_extend=-2),   # linear gaps
    dict(match=5, mismatch=-4, gap_open=-10, gap_extend=-1),  # expensive open, cheap extend
    dict(match=3, mismatch=-2, gap_open=-4, gap_extend=-3),
]


def mutate(rng, seq, sub_rate, indel_rate):
    out = []
    i = 0
    while i < len(seq):
        r = rng.random()
        if r < indel_rate / 2:  # deletion of a run
            i += rng.randint(1, 6)
            continue
        if r < indel_rate:  # insertion of a run
            out.extend(rng.choice(ALPHABET) for _ in range(rng.randint(1, 6)))
        ch = seq[i]
        if rng.random() < sub_rate:
            ch = rng.choice(ALPHABET.replace(ch, ""))
        out.append(ch)
        i += 1
    return "".join(out)


def random_seq(rng, n):
    return "".join(rng.choice(ALPHABET) for _ in range(n))


def brute_force_global(a, b, p):
    """Enumerate all alignments of tiny sequences (exponential; len <= 5)."""
    best = None

    def rec(i, j, cols):
        nonlocal best
        if i == len(a) and j == len(b):
            s = rescore(cols, p)
            best = s if best is None else max(best, s)
            return
        if i < len(a) and j < len(b):
            rec(i + 1, j + 1, cols + [(a[i], b[j])])
        if i < len(a):
            rec(i + 1, j, cols + [(a[i], "-")])
        if j < len(b):
            rec(i, j + 1, cols + [("-", b[j])])

    rec(0, 0, [])
    return best


def rescore(cols, p):
    score, prev = 0, None
    for x, y in cols:
        if x != "-" and y != "-":
            score += p["match"] if x == y else p["mismatch"]
            prev = None
            continue
        kind = "a" if x == "-" else "b"
        score += p["gap_extend"] if prev == kind else p["gap_open"]
        prev = kind
    return score


def brute_force_local(a, b, p):
    best = 0
    for i0, i1 in itertools.combinations_with_replacement(range(len(a) + 1), 2):
        for j0, j1 in itertools.combinations_with_replacement(range(len(b) + 1), 2):
            if i1 > i0 and j1 > j0:
                best = max(best, brute_force_global(a[i0:i1], b[j0:j1], p))
    return best


def self_check(rng):
    for _ in range(300):
        a = random_seq(rng, rng.randint(0, 5))
        b = random_seq(rng, rng.randint(0, 5))
        p = rng.choice(PARAM_SETS)
        g = align(a, b, "global", **p)[0]
        assert g == (brute_force_global(a, b, p) or 0), (a, b, p)
        loc = align(a, b, "local", **p)[0]
        assert loc == brute_force_local(a, b, p), (a, b, p)


def make_case(rng, case_id, a, b, mode, params):
    score, _, _ = align(a, b, mode, **params)
    return {"id": case_id, "args": {"seq_a": a, "seq_b": b, "mode": mode, **params},
            "expected_score": score}


def build(rng):
    cases = []
    edge = [("", ""), ("ACGT", ""), ("", "GG"), ("A", "A"), ("A", "C"), ("ACGTACGT", "ACGTACGT"),
            ("AAAAAAAAAA", "TTTTTTTTTT"), ("ACGTTTTTTTTTACGT", "ACGTACGT")]
    for k, (a, b) in enumerate(edge):
        for mode in ("global", "local"):
            cases.append(make_case(rng, f"edge{k}_{mode}", a, b, mode, PARAM_SETS[k % 4]))
    for k in range(48):
        base = random_seq(rng, rng.randint(20, 220))
        other = mutate(rng, base, sub_rate=rng.uniform(0.02, 0.25), indel_rate=rng.uniform(0.0, 0.08))
        if k % 6 == 5:  # local-alignment style: embed a related core in unrelated flanks
            other = random_seq(rng, rng.randint(10, 60)) + other[: len(other) // 2] + random_seq(rng, 30)
        mode = "local" if k % 2 else "global"
        cases.append(make_case(rng, f"rand{k}_{mode}", base, other, mode, PARAM_SETS[k % 4]))
    return cases


def build_perf(rng):
    base = random_seq(rng, 1500)
    other = mutate(rng, base, 0.1, 0.03)
    return [make_case(rng, f"perf_{m}", base, other, m, PARAM_SETS[0]) for m in ("global", "local")]


def main():
    rng = random.Random(SEED)
    self_check(rng)
    hidden = {"cases": build(rng), "perf_cases": build_perf(rng)}
    (TASK_DIR / "tests" / "cases.json").write_text(json.dumps(hidden, indent=1))
    pub_rng = random.Random(SEED + 1)
    public = []
    for k, mode in enumerate(["global", "local", "global", "local"]):
        a = random_seq(pub_rng, 30)
        b = mutate(pub_rng, a, 0.15, 0.05)
        c = make_case(pub_rng, f"example{k}", a, b, mode, PARAM_SETS[k])
        public.append(c)
    out = TASK_DIR / "environment" / "workspace" / "examples.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(public, indent=1))
    print(f"hidden cases: {len(hidden['cases'])} + {len(hidden['perf_cases'])} perf; public: {len(public)}")


if __name__ == "__main__":
    main()
