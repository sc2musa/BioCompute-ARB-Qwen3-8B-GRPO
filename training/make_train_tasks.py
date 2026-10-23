"""Generate training task instances that are disjoint from the frozen benchmark test set.

Usage (from the repo root):
    python training/make_train_tasks.py --out <dir> --per-task 128 [--workers 32]

Every instance is a full task directory (same layout as tasks/<task>/) with freshly generated
public examples and hidden cases, so harness/runner.py can run and grade it unchanged:
    python -m harness.runner --tasks-root <dir> --agent llm ...
Seeds start at TRAIN_SEED_BASE, far from the test-set seeds (2026102x), and the families that
are fixed in the test set (Robertson, gene dimerisation, SIR, scoring parameters) get randomised
parameters, so no training instance reproduces a test case.
"""

import argparse
import concurrent.futures
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
TRAIN_SEED_BASE = 10_000_000
TASK_OFFSETS = {"seq-align-affine": 0, "reaction-network-stiff": 1_000_000,
                "seir-changepoint-inference": 2_000_000}
COPY_FILES = ["task.toml", "instruction.md", "guidance.md", "public_check.py"]
COPY_TEST_FILES = ["grade.py", "sandbox_worker.py", "test.sh"]


def load_builder(task):
    path = TASKS / task / "tests" / "build_cases.py"
    spec = importlib.util.spec_from_file_location(f"build_{task.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- T1: sequence alignment
def random_param_set(rng):
    gap_open = -rng.randint(2, 12)
    return dict(match=rng.randint(1, 5), mismatch=-rng.randint(1, 5),
                gap_open=gap_open, gap_extend=-rng.randint(1, -gap_open))


def gen_align(mod, seed):
    rng = random.Random(seed)
    mod.PARAM_SETS = [random_param_set(rng) for _ in range(4)]
    hidden = {"cases": mod.build(rng), "perf_cases": mod.build_perf(rng)[:1]}
    public = []
    for k, mode in enumerate(["global", "local", "global", "local"]):
        a = mod.random_seq(rng, 30)
        public.append(mod.make_case(rng, f"example{k}", a, mod.mutate(rng, a, 0.15, 0.05), mode,
                                    mod.PARAM_SETS[k]))
    return hidden, {"examples.json": public}


# ---------------------------------------------------------------- T2: reaction networks
def perturb(net, rng, spread=0.5):
    """Scale every rate constant by 10**U(-spread, spread)."""
    out = json.loads(json.dumps(net))
    for r in out["reactions"]:
        r["rate"] = float(r["rate"] * 10 ** rng.uniform(-spread, spread))
    return out


def random_mm(mod, rng, name):
    net = mod.michaelis_menten(10 ** rng.uniform(3, 6.5), 10 ** rng.uniform(0, 3), 10 ** rng.uniform(-1, 1),
                               10 ** rng.uniform(-4, -2), rng.uniform(0.5, 2.0), rng.choice([500.0, 1000.0, 2000.0]))
    return dict(net, name=name)


def random_sir(mod, rng):
    n = 10 ** rng.uniform(3, 5)
    gamma = rng.uniform(0.05, 0.3)
    beta = rng.uniform(1.5, 4.0) * gamma / n
    i0 = rng.uniform(1, 20)
    net = mod.sir()
    net["species"] = mod.species([("S", n - i0), ("I", i0), ("R", 0.0)])
    net["reactions"] = [mod.rxn({"S": 1, "I": 1}, {"I": 2}, beta), mod.rxn({"I": 1}, {"R": 1}, gamma)]
    return net


def gen_network(mod, seed):
    rng = random.Random(seed)
    for _ in range(20):  # resample if the reference cross-check rejects a draw
        try:
            nets = [random_mm(mod, rng, "michaelis_menten_full"), perturb(mod.robertson(), rng, 0.3),
                    perturb(mod.gene_dimer(), rng), random_sir(mod, rng),
                    mod.mapk_cascade(rng, rng.choice([2, 3])), mod.mapk_cascade(rng, rng.choice([3, 4])),
                    mod.random_closed_network(rng, rng.randint(8, 16), rng.randint(8, 20), "random_closed_small"),
                    mod.random_closed_network(rng, rng.randint(25, 40), rng.randint(30, 50), "random_closed_large")]
            cases = []
            for net in nets:
                t_eval, y = mod.reference(net)
                cases.append({"network": net, "t_eval": t_eval.tolist(), "reference": y.tolist()})
            public = {}
            for net in (random_mm(mod, rng, "example_michaelis_menten"),
                        mod.random_closed_network(rng, 6, 6, "example_random_6")):
                t_eval, y = mod.reference(net)
                public[f"examples/{net['name']}.json"] = {"network": net, "t_eval": t_eval.tolist(),
                                                          "reference": y.tolist()}
            return cases, public
        except AssertionError:
            continue
    raise RuntimeError(f"could not generate a valid network instance for seed {seed}")


# ---------------------------------------------------------------- T3: SEIR inference
def gen_seir(mod, seed):
    rng = np.random.default_rng(seed)
    hidden = [mod.sample_case(rng, f"outbreak{k:02d}", intervention=k >= mod.N_NO_INTERVENTION)
              for k in range(mod.N_HIDDEN)]
    rng.shuffle(hidden)
    public = {f"data/example{k}.json": mod.sample_case(rng, f"example{k}", intervention=iv)
              for k, iv in enumerate([True, True, False])}
    return hidden, public


GENERATORS = {"seq-align-affine": gen_align, "reaction-network-stiff": gen_network,
              "seir-changepoint-inference": gen_seir}
PUBLIC_DATA = {"seq-align-affine": ["examples.json"], "reaction-network-stiff": ["examples"],
               "seir-changepoint-inference": ["data"]}


def make_instance(task, index, out_root):
    seed = TRAIN_SEED_BASE + TASK_OFFSETS[task] + index
    hidden, public = GENERATORS[task](load_builder(task), seed)
    src, dst = TASKS / task, Path(out_root) / f"{task}__train{index:04d}"
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "tests").mkdir(parents=True)
    for name in COPY_FILES:
        shutil.copy(src / name, dst / name)
    for name in COPY_TEST_FILES:
        shutil.copy(src / "tests" / name, dst / "tests" / name)
    shutil.copytree(src / "solution", dst / "solution")
    shutil.copytree(src / "environment", dst / "environment")
    workspace = dst / "environment" / "workspace"
    for old in PUBLIC_DATA[task]:  # replace the benchmark's public examples
        p = workspace / old
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    for rel, content in public.items():
        (workspace / rel).parent.mkdir(parents=True, exist_ok=True)
        (workspace / rel).write_text(json.dumps(content, indent=1))
    (dst / "tests" / "cases.json").write_text(json.dumps(hidden))
    (dst / "train_meta.json").write_text(json.dumps({"base_task": task, "seed": seed}))
    return dst.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-task", type=int, default=128)
    ap.add_argument("--tasks", default=",".join(GENERATORS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(t, i) for t in args.tasks.split(",") for i in range(args.per_task)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(make_instance, t, i, args.out) for t, i in jobs]
        for f in concurrent.futures.as_completed(futures):
            print(f.result(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
