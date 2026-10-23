"""Plausible-but-wrong baseline: implicit LSODA with default tolerances.

Fast and stable, but scipy defaults (rtol=1e-3, atol=1e-6) miss the 1e-4 accuracy
requirement on most networks.
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

    sol = solve_ivp(rhs, (0, t_eval[-1]), y0, method="LSODA", t_eval=t_eval)
    return sol.y.T
