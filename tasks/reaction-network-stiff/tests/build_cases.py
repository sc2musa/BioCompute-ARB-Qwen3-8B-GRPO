"""Author-side script: build hidden networks and high-accuracy reference trajectories.

Not used at grading time. Run from the repo root:
    python tasks/reaction-network-stiff/tests/build_cases.py
Reference = Radau (rtol 1e-11) with an independent dense-matrix RHS, cross-checked against
LSODA. Writes tests/cases.json (hidden) and public examples into environment/workspace/.
"""

import json
import random
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

TASK_DIR = Path(__file__).resolve().parents[1]
SEED = 20261021
N_TIMES = 41
CROSS_CHECK_RTOL = 2e-6  # >= 50x below the grading tolerance (1e-4)


def rxn(reactants, products, rate):
    return {"reactants": reactants, "products": products, "rate": rate}


def species(pairs):
    return [{"name": n, "initial": v} for n, v in pairs]


def michaelis_menten(k_on, k_off, k_cat, e0, s0, t_end):
    return {
        "name": "michaelis_menten_full",
        "species": species([("E", e0), ("S", s0), ("C", 0.0), ("P", 0.0)]),
        "reactions": [rxn({"E": 1, "S": 1}, {"C": 1}, k_on), rxn({"C": 1}, {"E": 1, "S": 1}, k_off),
                      rxn({"C": 1}, {"E": 1, "P": 1}, k_cat)],
        "t_end": t_end,
    }


def robertson():
    return {
        "name": "robertson_autocatalysis",
        "species": species([("A", 1.0), ("B", 0.0), ("C", 0.0)]),
        "reactions": [rxn({"A": 1}, {"B": 1}, 0.04), rxn({"B": 2}, {"B": 1, "C": 1}, 3e7),
                      rxn({"B": 1, "C": 1}, {"A": 1, "C": 1}, 1e4)],
        "t_end": 4e4,
    }


def gene_dimer():
    return {
        "name": "gene_expression_dimerisation",
        "species": species([("G", 1.0), ("M", 0.0), ("P", 0.0), ("D", 0.0), ("GD", 0.0)]),
        "reactions": [
            rxn({"G": 1}, {"G": 1, "M": 1}, 2.0),          # transcription
            rxn({"M": 1}, {"M": 1, "P": 1}, 10.0),         # translation
            rxn({"M": 1}, {}, 0.5),                         # mRNA decay
            rxn({"P": 1}, {}, 0.05),                        # protein decay
            rxn({"P": 2}, {"D": 1}, 50.0),                  # dimerisation
            rxn({"D": 1}, {"P": 2}, 1.0),
            rxn({"G": 1, "D": 1}, {"GD": 1}, 1e3),          # repression by dimer
            rxn({"GD": 1}, {"G": 1, "D": 1}, 0.1),
            rxn({}, {"M": 1}, 0.01),                        # leaky basal transcription
        ],
        "t_end": 200.0,
    }


def sir():
    n = 1e4
    return {
        "name": "sir_mass_action",
        "species": species([("S", n - 10), ("I", 10.0), ("R", 0.0)]),
        "reactions": [rxn({"S": 1, "I": 1}, {"I": 2}, 3e-5), rxn({"I": 1}, {"R": 1}, 0.1)],
        "t_end": 160.0,
    }


def mapk_cascade(rng, tiers=3):
    """Phosphorylation cascade: each tier X <-> Xp driven by the active upstream kinase."""
    sp = [("Sig", 0.1)]
    reactions = []
    upstream = "Sig"
    for t in range(tiers):
        x, xp, c1, c2, ph = f"K{t}", f"K{t}p", f"K{t}_c", f"K{t}p_c", f"Ph{t}"
        sp += [(x, 1.0), (xp, 0.0), (c1, 0.0), (c2, 0.0), (ph, 0.3)]
        kon = 10 ** rng.uniform(3, 5)
        reactions += [
            rxn({upstream: 1, x: 1}, {c1: 1}, kon), rxn({c1: 1}, {upstream: 1, x: 1}, kon * 1e-3),
            rxn({c1: 1}, {upstream: 1, xp: 1}, 10 ** rng.uniform(-1, 1)),
            rxn({ph: 1, xp: 1}, {c2: 1}, kon), rxn({c2: 1}, {ph: 1, xp: 1}, kon * 1e-3),
            rxn({c2: 1}, {ph: 1, x: 1}, 10 ** rng.uniform(-1, 1)),
        ]
        upstream = xp
    return {"name": f"mapk_cascade_{tiers}tier", "species": species(sp), "reactions": reactions,
            "t_end": 200.0}


def random_closed_network(rng, n_species, n_reactions, name):
    """Random mass-conserving network (A+B<->C and A<->B types) with stiff rate spread."""
    names = [f"X{i}" for i in range(n_species)]
    reactions = []
    for _ in range(n_reactions):
        a, b, c = rng.sample(names, 3)
        k_f, k_b = 10 ** rng.uniform(-2, 4), 10 ** rng.uniform(-2, 3)
        if rng.random() < 0.5:
            reactions += [rxn({a: 1, b: 1}, {c: 1}, k_f), rxn({c: 1}, {a: 1, b: 1}, k_b)]
        else:
            reactions += [rxn({a: 1}, {b: 1}, k_f), rxn({b: 1}, {a: 1}, k_b)]
    init = [(n, round(rng.uniform(0.0, 2.0), 3)) for n in names]
    return {"name": name, "species": species(init), "reactions": reactions, "t_end": 50.0}


def reference(network):
    """Independent implementation: dense stoichiometry, Radau, finite-difference Jacobian."""
    names = [s["name"] for s in network["species"]]
    idx = {n: i for i, n in enumerate(names)}
    R = len(network["reactions"])
    nu_in, nu_out = np.zeros((R, len(names))), np.zeros((R, len(names)))
    k = np.array([r["rate"] for r in network["reactions"]])
    for j, r in enumerate(network["reactions"]):
        for n, c in r["reactants"].items():
            nu_in[j, idx[n]] += c
        for n, c in r["products"].items():
            nu_out[j, idx[n]] += c
    S = (nu_out - nu_in).T

    def f(_t, y):
        return S @ (k * np.prod(np.power(y[None, :], nu_in), axis=1))

    def jac(_t, y):
        dv = np.zeros((R, len(names)))
        for s in range(len(names)):
            others = nu_in.copy()
            others[:, s] = 0.0
            lowered = np.power(y[s], np.maximum(nu_in[:, s] - 1.0, 0.0))
            dv[:, s] = k * nu_in[:, s] * lowered * np.prod(np.power(y[None, :], others), axis=1)
        return S @ dv

    y0 = np.array([s["initial"] for s in network["species"]], dtype=float)
    t_eval = np.linspace(0.0, network["t_end"], N_TIMES)
    if network["t_end"] > 1e3:  # long horizons: log-spaced sampling shows all time scales
        t_eval = np.concatenate([[0.0], np.logspace(-4, np.log10(network["t_end"]), N_TIMES - 1)])
    # Floating-point round-off can push the last point past t_end (e.g. 2000.0000000000002),
    # which would make solve_ivp(t_span=(0, t_end), t_eval=...) fail; pin it exactly.
    t_eval[-1] = network["t_end"]
    atol = 1e-14 * max(1.0, np.abs(y0).max())
    ref = solve_ivp(f, (0, t_eval[-1]), y0, method="Radau", t_eval=t_eval, rtol=1e-11, atol=atol, jac=jac)
    chk = solve_ivp(f, (0, t_eval[-1]), y0, method="LSODA", t_eval=t_eval, rtol=1e-10, atol=atol, jac=jac)
    assert ref.success and chk.success, network["name"]
    scale = np.abs(ref.y).max(axis=1, keepdims=True)
    err = np.abs(ref.y - chk.y) / (np.abs(ref.y) + 1e-6 * scale + 1e-300)
    assert err.max() < CROSS_CHECK_RTOL, (network["name"], err.max())
    return t_eval, ref.y.T


def main():
    rng = random.Random(SEED)
    hidden = [
        michaelis_menten(1e6, 1e2, 1.0, 1e-3, 1.0, 2000.0),
        robertson(),
        gene_dimer(),
        sir(),
        mapk_cascade(rng, 3),
        mapk_cascade(rng, 4),
        random_closed_network(rng, 12, 14, "random_closed_12"),
        random_closed_network(rng, 40, 50, "random_closed_40"),
    ]
    cases = []
    for net in hidden:
        t_eval, y = reference(net)
        cases.append({"network": net, "t_eval": t_eval.tolist(), "reference": y.tolist()})
        print(f"{net['name']}: {len(net['species'])} species, {len(net['reactions'])} reactions")
    (TASK_DIR / "tests" / "cases.json").write_text(json.dumps(cases))

    public = [dict(michaelis_menten(1e4, 10.0, 2.0, 0.01, 1.0, 500.0), name="example_michaelis_menten"),
              random_closed_network(random.Random(SEED + 1), 6, 6, "example_random_6")]
    ws = TASK_DIR / "environment" / "workspace" / "examples"
    ws.mkdir(parents=True, exist_ok=True)
    for net in public:
        t_eval, y = reference(net)
        (ws / f"{net['name']}.json").write_text(json.dumps(
            {"network": net, "t_eval": t_eval.tolist(), "reference": y.tolist()}, indent=1))


if __name__ == "__main__":
    main()
