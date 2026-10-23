import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    N = population
    S0 = N - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0

    def seir_model(t, y, R0_val, R_post_val, change_day):
        S, E, I, R = y
        beta = R0_val * gamma if t < change_day else R_post_val * gamma
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dI_dt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dI_dt, dRdt]

    def compute_expected_cases(R0_val, R_post_val, change_day):
        t_span = (0, 70)
        y0 = [S0, E0, I0, R0]
        sol = solve_ivp(seir_model, t_span, y0, args=(R0_val, R_post_val, change_day), dense_output=True)
        expected_cases = []
        for d in range(1, 71):
            t_start = d - 1
            t_end = d
            E_start = sol.sol(t_start)[2]
            E_end = sol.sol(t_end)[2]
            integral = (E_start + E_end) * (t_end - t_start) / 2.0
            expected_cases.append(integral * sigma)
        return expected_cases

    def log_likelihood(params, change_day):
        R0_val, R_post_val = params
        expected = compute_expected_cases(R0_val, R_post_val, change_day)
        ll = 0.0
        for d in range(70):
            observed = cases[d]
            mu = expected[d]
            if mu == 0:
                if observed == 0:
                    ll += 0.0
                else:
                    ll -= float('inf')
            else:
                ll += observed * np.log(mu) - mu
        return -ll

    best_ll = -np.inf
    best_R0 = 0.0
    best_R_post = 0.0
    best_change_day = None

    # Optimize change_day with fewer iterations
    for change_day in range(1, 21):  # Reduced from 71 to 21
        initial_guess = [1.0, 1.0]
        result = minimize(log_likelihood, initial_guess, args=(change_day), method='Nelder-Mead', bounds=[(0.01, 100.0), (0.01, 100.0)])
        if result.success:
            current_ll = -result.fun
            if current_ll > best_ll:
                best_ll = current_ll
                best_R0 = result.x[0]
                best_R_post = result.x[1]
                best_change_day = change_day

    # Check no-intervention case
    def log_likelihood_no_change(R0_val):
        expected = compute_expected_cases(R0_val, R0_val, 71)
        ll = 0.0
        for d in range(70):
            observed = cases[d]
            mu = expected[d]
            if mu == 0:
                if observed == 0:
                    ll += 0.0
                else:
                    ll -= float('inf')
            else:
                ll += observed * np.log(mu) - mu
        return -ll
    result = minimize(log_likelihood_no_change, 1.0, method='Nelder-Mead', bounds=[(0.01, 100.0)])
    if result.success:
        best_R0 = result.x[0]
        best_R_post = best_R0
        best_change_day = None

    forecast_cases = []
    if best_change_day is None:
        t_span = (0, 70 + horizon)
        y0 = [S0, E0, I0, R0]
        sol = solve_ivp(seir_model, t_span, y0, args=(best_R0, best_R0, 71), dense_output=True)
        for d in range(71, 71 + horizon):
            t_start = d - 1
            t_end = d
            E_start = sol.sol(t_start)[2]
            E_end = sol.sol(t_end)[2]
            integral = (E_start + E_end) * (t_end - t_start) / 2.0
            forecast_cases.append(integral * sigma)
    else:
        t_span = (0, 70 + horizon)
        y0 = [S0, E0, I0, R0]
        sol = solve_ivp(seir_model, t_span, y0, args=(best_R0, best_R_post, best_change_day), dense_output=True)
        for d in range(71, 71 + horizon):
            t_start = d - 1
            t_end = d
            E_start = sol.sol(t_start)[2]
            E_end = sol.sol(t_end)[2]
            integral = (E_start + E_end) * (t_end - t_start) / 2.0
            forecast_cases.append(integral * sigma)

    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": forecast_cases
    }