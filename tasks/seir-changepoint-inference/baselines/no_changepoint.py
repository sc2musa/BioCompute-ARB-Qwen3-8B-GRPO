"""Plausible-but-incomplete baseline: SEIR fit that ignores the intervention.

Fits a single R to the whole series (Poisson likelihood). Expected to fail R_post, change_day
and the forecast on outbreaks with an intervention, and to bias R0.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    cases = np.asarray(cases, dtype=float)
    n = len(cases)
    sigma, gamma = 1.0 / latent_period, 1.0 / infectious_period

    def incidence(r, days):
        def rhs(y, _t):
            s, e, i, _c = y
            inf = r * gamma * s * i / population
            return [-inf, inf - sigma * e, sigma * e - gamma * i, sigma * e]
        y = odeint(rhs, [population - initial_infectious, 0, initial_infectious, 0],
                   np.arange(days + 1.0))
        return np.maximum(np.diff(y[:, 3]), 1e-9)

    def nll(r):
        mu = incidence(r, n)
        return np.sum(mu - cases * np.log(mu))

    r = minimize_scalar(nll, bounds=(0.5, 6.0), method="bounded").x
    return {"R0": r, "R_post": r, "change_day": None,
            "forecast": incidence(r, n + horizon)[n:].tolist()}
