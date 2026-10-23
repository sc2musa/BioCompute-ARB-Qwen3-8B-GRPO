"""Plausible-but-wrong baseline: explicit RK45 with tight tolerances.

Correct mass-action RHS, but an explicit method on stiff networks needs tiny steps and
should exceed the per-network time limit.
"""

import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    names = [s["name"] for s in network["species"]]
    idx = {n: i for i, n in enumerate(names)}
    y0 = np.array([s["initial"] for s in network["species"]], dtype=float)

    def rhs(_t, y):
        dy = np.zeros_like(y)
        for r in network["reactions"]:
            v = r["rate"]
            for n, c in r["reactants"].items():
                v *= y[idx[n]] ** c
            for n, c in r["reactants"].items():
                dy[idx[n]] -= c * v
            for n, c in r["products"].items():
                dy[idx[n]] += c * v
        return dy

    sol = solve_ivp(rhs, (0, t_eval[-1]), y0, method="RK45", t_eval=t_eval, rtol=1e-8, atol=1e-12)
    return sol.y.T
