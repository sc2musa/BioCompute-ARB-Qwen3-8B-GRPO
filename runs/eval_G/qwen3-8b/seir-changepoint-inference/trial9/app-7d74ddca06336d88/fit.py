import numpy as np
from scipy.optimize import minimize

def simulate_seir(population, initial_infectious, beta, latent_period, infectious_period, days, initial_E=0, initial_S=None, initial_I=None, initial_R=None):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    N = population
    if initial_S is None:
        initial_S = N - initial_infectious
    if initial_I is None:
        initial_I = initial_infectious
    if initial_E is None:
        initial_E = 0
    if initial_R is None:
        initial_R = 0
    S = initial_S
    E = initial_E
    I = initial_I
    R = initial_R
    cases = []
    for _ in range(days):
        k1_S = -beta * S * I / N
        k1_E = beta * S * I / N - sigma * E
        k1_I = sigma * E - gamma * I
        k1_R = gamma * I
        k2_S = -beta * S * I / N
        k2_E = beta * S * I / N - sigma * E
        k2_I = sigma * E - gamma * I
        k2_R = gamma * I
        S += (k1_S + k2_S) / 2
        E += (k1_E + k2_E) / 2
        I += (k1_I + k2_I) / 2
        R += (k1_R + k2_R) / 2
        new_cases = sigma * E
        cases.append(new_cases)
    return cases, S, E, I, R

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    best_change_day = None
    best_R0 = None
    best_R_post = None
    best_total_error = float('inf')
    best_state_vars = None

    for change_day in range(71):
        # First part: days 0 to change_day-1
        def objective_beta0(beta0):
            cases_first, _, _, _, _ = simulate_seir(population, initial_infectious, beta0, latent_period, infectious_period, change_day)
            error = sum((cases_first[i] - cases[i])**2 for i in range(change_day))
            return error
        res_beta0 = minimize(objective_beta0, x0=1.0, bounds=[(0.0, 100.0)], method='L-BFGS-B')
        beta0_opt = res_beta0.x[0]
        cases_first, S_end, E_end, I_end, R_end = simulate_seir(population, initial_infectious, beta0_opt, latent_period, infectious_period, change_day)

        # Second part: days change_day to 69
        observed_second = cases[change_day:70]
        def objective_beta_post(beta_post):
            cases_second, _, _, _, _ = simulate_seir(population, initial_infectious, beta_post, latent_period, infectious_period, 70 - change_day, initial_S=S_end, initial_E=E_end, initial_I=I_end, initial_R=R_end)
            error = sum((cases_second[i] - observed_second[i])**2 for i in range(len(observed_second)))
            return error
        res_beta_post = minimize(objective_beta_post, x0=1.0, bounds=[(0.0, 100.0)], method='L-BFGS-B')
        beta_post_opt = res_beta_post.x[0]
        total_error = res_beta0.fun + res_beta_post.fun

        if total_error < best_total_error:
            best_total_error = total_error
            best_change_day = change_day
            best_R0 = beta0_opt * infectious_period
            best_R_post = beta_post_opt * infectious_period
            best_state_vars = (S_end, E_end, I_end, R_end)

    # Forecast
    beta0 = best_R0 / infectious_period
    beta_post = best_R_post / infectious_period
    cases_first, S_end, E_end, I_end, R_end = simulate_seir(population, initial_infectious, beta0, latent_period, infectious_period, best_change_day)
    cases_second, _, _, _, _ = simulate_seir(population, initial_infectious, beta_post, latent_period, infectious_period, 70 - best_change_day, initial_S=S_end, initial_E=E_end, initial_I=I_end, initial_R=R_end)
    forecast_cases = []
    for _ in range(horizon):
        if len(forecast_cases) < best_change_day:
            current_beta = beta0
        else:
            current_beta = beta_post
        cases_next, S_next, E_next, I_next, R_next = simulate_seir(population, initial_infectious, current_beta, latent_period, infectious_period, 1, initial_S=S_end, initial_E=E_end, initial_I=I_end, initial_R=R_end)
        forecast_cases.append(cases_next[0])
        S_end, E_end, I_end, R_end = S_next, E_next, I_next, R_next
    return {
        'R0': best_R0,
        'R_post': best_R_post,
        'change_day': best_change_day,
        'forecast': forecast_cases
    }