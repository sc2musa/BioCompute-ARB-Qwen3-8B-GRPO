import numpy as np
from scipy.optimize import minimize

def simulate_seir(population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day, cases_length=70):
    sigma = 1 / latent_period
    gamma = 1 / infectious_period
    S = population - initial_infectious
    E = 0
    I = initial_infectious
    R = 0
    new_cases = []
    for day in range(1, cases_length + 1):
        beta = R0 * gamma if day < change_day else R_post * gamma
        new_infections = beta * S * I / population
        E += new_infections - sigma * E
        new_cases_day = sigma * E
        new_cases.append(new_cases_day)
        I += new_cases_day - gamma * I
        S -= new_infections
    return np.array(new_cases), S, E, I, R

def forecast(population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day, horizon):
    sigma = 1 / latent_period
    gamma = 1 / infectious_period
    _, S_final, E_final, I_final, R_final = simulate_seir(population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day, 70)
    forecast_cases = []
    for _ in range(horizon):
        beta = R_post * gamma
        new_infections = beta * S_final * I_final / population
        E_final += new_infections - sigma * E_final
        new_cases_day = sigma * E_final
        forecast_cases.append(new_cases_day)
        I_final += new_cases_day - gamma * I_final
        S_final -= new_infections
    return forecast_cases

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    gamma = 1 / infectious_period
    if len(cases) < 2:
        R0_initial = 3.0
    else:
        t = min(20, len(cases) - 1)
        growth_rate = np.log(cases[t]/cases[0]) / t
        R0_initial = 1 + growth_rate / gamma
    R_post_initial = R0_initial
    if len(cases) > 1:
        # Use rolling window to detect potential change points
        window_size = 5
        diffs = np.diff(cases)
        rolling_mean = np.convolve(diffs, np.ones(window_size), mode='valid')
        change_day_initial = np.argmin(rolling_mean) + 1 + window_size//2
    else:
        change_day_initial = 70.0

    def loss(params):
        R0, R_post, change_day = params
        # Ensure R_post <= R0
        R_post = np.clip(R_post, 0.5, R0)
        simulated_cases, _, _, _, _ = simulate_seir(population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day, len(cases))
        loss = np.sum((simulated_cases - cases) ** 2)
        return loss

    bounds = [(0.5, 5.0), (0.5, 5.0), (0.0, 70.0)]
    result = minimize(loss, [R0_initial, R_post_initial, change_day_initial], bounds=bounds, method='L-BFGS-B')

    R0_opt = result.x[0]
    R_post_opt = result.x[1]
    change_day_opt = result.x[2]

    forecast_cases = forecast(population, initial_infectious, latent_period, infectious_period, R0_opt, R_post_opt, change_day_opt, horizon)

    # Correct intervention detection logic
    if R_post_opt < 0.75 * R0_opt:
        change_day = change_day_opt
    else:
        change_day = None

    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day,
        "forecast": forecast_cases
    }