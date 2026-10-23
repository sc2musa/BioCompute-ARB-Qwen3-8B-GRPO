import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
import scipy.special

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    C0 = 0.0
    y0 = [S0, E0, I0, R0, C0]

    def seir_model(y, t, R0_val, R_post_val, change_day):
        S, E, I, R, C = y
        if t < change_day:
            beta = R0_val * gamma
        else:
            beta = R_post_val * gamma
        dS = -beta * S * I / population
        dE = beta * S * I / population - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        dC = sigma * E
        return [dS, dE, dI, dR, dC]

    def compute_expected_cases(R0_val, R_post_val, change_day):
        t1 = np.arange(0, change_day + 1)
        t2 = np.arange(change_day, 70 + 1)
        y1 = odeint(seir_model, y0, t1, args=(R0_val, R_post_val, change_day))
        S1, E1, I1, R1, C1 = y1[-1]
        y2 = odeint(seir_model, [S1, E1, I1, R1, C1], t2, args=(R0_val, R_post_val, change_day))
        C = np.zeros(71)
        for i, t in enumerate(t1):
            C[int(t)] = y1[i, 4]
        for i, t in enumerate(t2):
            C[int(t)] = y2[i, 4]
        expected_cases = [C[d] - C[d-1] for d in range(1, 71)]
        return expected_cases

    def nll(params, change_day):
        R0_val, R_post_val = params
        expected = compute_expected_cases(R0_val, R_post_val, change_day)
        ll = 0.0
        eps = 1e-10
        for y, lam in zip(cases, expected):
            lam = max(lam, eps)
            ll += y * np.log(lam) - lam - scipy.special.gammaln(y + 1)
        return -ll

    def optimize_for_change_day(change_day):
        initial_guess = [2.0, 1.0]
        bounds = [(1e-6, None), (1e-6, None)]
        result = minimize(nll, initial_guess, args=(change_day,), bounds=bounds, method='L-BFGS-B')
        if result.success:
            return result.x[0], result.x[1], -result.fun
        else:
            return 2.0, 1.0, float('inf')

    def compute_expected_cases_no_intervention(R):
        t = np.arange(0, 71)
        y = odeint(seir_model, y0, t, args=(R, R, None))
        C = y[:, 4]
        expected_cases = [C[d] - C[d-1] for d in range(1, 71)]
        return expected_cases

    def nll_no_intervention(params):
        R = params[0]
        expected = compute_expected_cases_no_intervention(R)
        ll = 0.0
        eps = 1e-10
        for y, lam in zip(cases, expected):
            lam = max(lam, eps)
            ll += y * np.log(lam) - lam - scipy.special.gammaln(y + 1)
        return -ll

    initial_guess_no_intervention = [2.0]
    result_no_intervention = minimize(nll_no_intervention, initial_guess_no_intervention, method='L-BFGS-B', bounds=[(1e-6, None)])
    nll_no_intervention_val = -result_no_intervention.fun if result_no_intervention.success else float('inf')
    k_no_intervention = 1
    bic_no_intervention = 2 * nll_no_intervention_val + k_no_intervention * np.log(70)

    best_model = None
    best_bic = float('inf')

    for change_day in range(0, 71):
        if change_day < 5 or change_day > 65:
            continue
        R0_opt, R_post_opt, nll_val = optimize_for_change_day(change_day)
        k_intervention = 2
        bic_intervention = 2 * nll_val + k_intervention * np.log(70)
        if bic_intervention < bic_no_intervention:
            best_model = ('intervention', change_day, R0_opt, R_post_opt)
            best_bic = bic_intervention
        else:
            best_model = ('no_intervention', None, 0.0, 0.0)

    if best_model[0] == 'intervention':
        R0 = best_model[2]
        R_post = best_model[3]
        change_day = best_model[1]
    else:
        R0 = best_model[2]
        R_post = best_model[3]
        change_day = None

    if change_day is not None:
        t1 = np.arange(0, change_day + 1)
        t2 = np.arange(change_day, 70 + 1)
        y1 = odeint(seir_model, y0, t1, args=(R0, R_post, change_day))
        S1, E1, I1, R1, C1 = y1[-1]
        y2 = odeint(seir_model, [S1, E1, I1, R1, C1], t2, args=(R0, R_post, change_day))
        S_forecast, E_forecast, I_forecast, R_forecast, C_forecast = y2[-1]
        t_forecast = np.arange(70, 70 + horizon + 1)
        forecast_y = odeint(seir_model, [S_forecast, E_forecast, I_forecast, R_forecast, C_forecast], t_forecast, args=(R0, R_post, change_day))
        forecast_C = forecast_y[:, 4]
        forecast = [forecast_C[d] - forecast_C[d-1] for d in range(1, horizon + 1)]
    else:
        t = np.arange(0, 71)
        y = odeint(seir_model, y0, t, args=(R0, R0, None))
        S_forecast, E_forecast, I_forecast, R_forecast, C_forecast = y[-1]
        t_forecast = np.arange(70, 70 + horizon + 1)
        forecast_y = odeint(seir_model, [S_forecast, E_forecast, I_forecast, R_forecast, C_forecast], t_forecast, args=(R0, R0, None))
        forecast_C = forecast_y[:, 4]
        forecast = [forecast_C[d] - forecast_C[d-1] for d in range(1, horizon + 1)]

    return {
        'R0': float(R0),
        'R_post': float(R_post),
        'change_day': float(change_day) if change_day is not None else None,
        'forecast': forecast
    }
