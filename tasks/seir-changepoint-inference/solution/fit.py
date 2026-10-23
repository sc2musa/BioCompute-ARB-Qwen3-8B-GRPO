"""Reference solution: SEIR change-point inference by profile maximum likelihood.

For every candidate change day on an integer grid, fit (R0, R_post) by maximising a
negative-binomial likelihood of the observed daily cases; keep the best day.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.special import gammaln

NB_DISPERSION = 30.0
MIN_SEGMENT_DAYS = 5


def _incidence(params, population, i0, latent, infectious, change_day, n_days):
    r0, r_post = params
    sigma, gamma = 1.0 / latent, 1.0 / infectious

    def rhs(y, t):
        s, e, i, _c = y
        beta = (r0 if t < change_day else r_post) * gamma
        infection = beta * s * i / population
        return [-infection, infection - sigma * e, sigma * e - gamma * i, sigma * e]

    t = np.arange(n_days + 1, dtype=float)
    tcrit = [change_day] if 0 < change_day < n_days else None
    y = odeint(rhs, [population - i0, 0.0, i0, 0.0], t, tcrit=tcrit, rtol=1e-8, atol=1e-6)
    return np.maximum(np.diff(y[:, 3]), 1e-9)


def _nb_nll(mu, cases, k=NB_DISPERSION):
    return -np.sum(gammaln(cases + k) - gammaln(k) + k * np.log(k / (k + mu))
                   + cases * np.log(mu / (k + mu)))


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    cases = np.asarray(cases, dtype=float)
    n = len(cases)
    model = (population, initial_infectious, latent_period, infectious_period)

    def nll(log_params, change_day):
        mu = _incidence(np.exp(log_params), *model, change_day, n)
        return _nb_nll(mu, cases)

    options = {"xatol": 1e-5, "fatol": 1e-6, "maxiter": 400}
    # Null model: a single R for the whole period (change day beyond the data)
    null = minimize(lambda x: nll(np.array([x[0], x[0]]), n + 1), [np.log(2.5)],
                    method="Nelder-Mead", options=options)
    best = None
    for change_day in range(MIN_SEGMENT_DAYS, n - MIN_SEGMENT_DAYS + 1):
        res = minimize(nll, np.log([2.5, 1.5]), args=(change_day,), method="Nelder-Mead",
                       options=options)
        if best is None or res.fun < best[0]:
            best = (res.fun, res.x, change_day)
    # BIC model selection: the change-point model has two extra parameters
    if 2 * null.fun + np.log(n) <= 2 * best[0] + 3 * np.log(n):
        r0 = r_post = float(np.exp(null.x[0]))
        change_day = None
    else:
        r0, r_post = (float(v) for v in np.exp(best[1]))
        change_day = float(best[2])
    forecast = _incidence((r0, r_post), *model, n + 1 if change_day is None else change_day,
                          n + horizon)[n:]
    return {"R0": r0, "R_post": r_post, "change_day": change_day, "forecast": forecast.tolist()}
