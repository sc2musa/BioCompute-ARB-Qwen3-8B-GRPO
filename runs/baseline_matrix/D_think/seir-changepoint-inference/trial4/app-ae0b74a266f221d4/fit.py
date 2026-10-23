import numpy as np
from scipy.optimize import minimize

def simulate_model(population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day, days=70):
    S = population - initial_infectious
    E = 0
    I = initial_infectious
    R = 0
    E_history = [0] * (days + 1)
    I_history = [0] * (days + 1)
    expected_cases = [0] * (days + 1)
    gamma = 1.0 / infectious_period
    for d in range(1, days + 1):
        if change_day is None:
            beta = R0 * gamma
        else:
            if d < change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
        new_infected = beta * I * S / population
        E_history[d] = new_infected
        S -= new_infected
        E += new_infected
        if d >= latent_period + 1:
            moved_to_I = E_history[d - latent_period]
            E -= moved_to_I
            I += moved_to_I
            expected_cases[d] = moved_to_I
            I_history[d] = moved_to_I
        if d >= infectious_period + 1:
            moved_to_R = I_history[d - infectious_period]
            I -= moved_to_R
            R += moved_to_R
    return expected_cases[1:days+1]

def estimate_R0(cases, population, initial_infectious, latent_period, infectious_period):
    def loss(R0):
        expected = simulate_model(population, initial_infectious, latent_period, infectious_period, R0, R_post=None, change_day=None, days=70)
        mse = np.mean((np.array(expected) - np.array(cases)) ** 2)
        return mse
    result = minimize(loss, 2.0, method='L-BFGS-B', bounds=[(0.1, 100)])
    return result.x[0]

def detect_change_day(cases, expected_cases, population, initial_infectious, latent_period, infectious_period, R0_est):
    ratios = np.array(cases) / np.array(expected_cases)
    min_ratio_day = np.argmin(ratios) + 1  # days are 1-based
    if ratios[min_ratio_day - 1] < 0.75:
        return min_ratio_day
    else:
        return None

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    R0_est = estimate_R0(cases, population, initial_infectious, latent_period, infectious_period)
    expected_cases = simulate_model(population, initial_infectious, latent_period, infectious_period, R0_est, R_post=None, change_day=None, days=70)
    change_day = detect_change_day(cases, expected_cases, population, initial_infectious, latent_period, infectious_period, R0_est)

    if change_day is not None:
        # Re-estimate R0 and R_post
        # Split the data into before and after change_day
        cases_before = cases[0:change_day-1]
        cases_after = cases[change_day-1:]
        # Estimate R0 for before data
        R0_before = estimate_R0(cases_before, population, initial_infectious, latent_period, infectious_period)
        # Estimate R_post for after data
        def loss_R_post(R_post):
            expected_after = simulate_model(population, initial_infectious, latent_period, infectious_period, R0_before, R_post, change_day, days=len(cases_after))
            mse = np.mean((np.array(expected_after) - np.array(cases_after)) ** 2)
            return mse
        result_R_post = minimize(loss_R_post, 2.0, method='L-BFGS-B', bounds=[(0.1, R0_before)])
        R_post_est = result_R_post.x[0]
    else:
        R_post_est = R0_est

    # Forecast
    # Simulate from day 71 onward using the estimated parameters
    days_forecast = horizon
    S = population - initial_infectious
    E = 0
    I = initial_infectious
    gamma = 1.0 / infectious_period
    beta = R0_est * gamma if change_day is None else (R0_est * gamma if change_day > days_forecast else R_post_est * gamma)
    forecast = []
    for d in range(1, days_forecast + 1):
        new_infected = beta * I * S / population
        E += new_infected
        S -= new_infected
        if d >= latent_period + 1:
            moved_to_I = E - (E - new_infected)  # Simplified for forecast
            E -= moved_to_I
            I += moved_to_I
            forecast.append(moved_to_I)
        else:
            forecast.append(0)
    return {
        "R0": R0_est,
        "R_post": R_post_est,
        "change_day": change_day,
        "forecast": forecast
    }