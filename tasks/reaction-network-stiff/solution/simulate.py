"""Reference solution: mass-action reaction network simulator for stiff systems.

Builds stoichiometry matrices once, then integrates with an implicit method (BDF) using the
analytic Jacobian.
"""

import numpy as np
from scipy.integrate import solve_ivp

RTOL = 1e-8
ATOL_FRACTION = 1e-12


def _build(network):
    names = [s["name"] for s in network["species"]]
    index = {name: i for i, name in enumerate(names)}
    n_species, n_reactions = len(names), len(network["reactions"])
    reactant_order = np.zeros((n_reactions, n_species))
    net_change = np.zeros((n_species, n_reactions))
    rates = np.zeros(n_reactions)
    for r, rxn in enumerate(network["reactions"]):
        rates[r] = rxn["rate"]
        for name, coef in rxn.get("reactants", {}).items():
            reactant_order[r, index[name]] += coef
            net_change[index[name], r] -= coef
        for name, coef in rxn.get("products", {}).items():
            net_change[index[name], r] += coef
    y0 = np.array([float(s["initial"]) for s in network["species"]])
    return rates, reactant_order, net_change, y0


def simulate(network, t_eval):
    rates, order, change, y0 = _build(network)
    t_eval = np.asarray(t_eval, dtype=float)
    # Per-reaction list of (species, order) pairs for fast propensity/Jacobian evaluation
    terms = [[(s, int(o)) for s, o in enumerate(row) if o] for row in order]

    def propensities(y):
        v = rates.copy()
        for r, term in enumerate(terms):
            for s, o in term:
                v[r] *= y[s] ** o
        return v

    def rhs(_t, y):
        return change @ propensities(y)

    def jac(_t, y):
        dv = np.zeros((len(rates), len(y)))  # d propensity_r / d y_s
        for r, term in enumerate(terms):
            for s, o in term:
                partial = rates[r] * o * y[s] ** (o - 1)
                for s2, o2 in term:
                    if s2 != s:
                        partial *= y[s2] ** o2
                dv[r, s] = partial
        return change @ dv

    scale = max(np.abs(y0).max(), 1e-300)
    sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), y0, method="BDF", t_eval=t_eval,
                    jac=jac, rtol=RTOL, atol=ATOL_FRACTION * scale)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T
