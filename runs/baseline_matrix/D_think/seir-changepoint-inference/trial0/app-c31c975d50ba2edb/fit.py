import numpy as np
from scipy.optimize import minimize

def simulate_model(population, initial_infectious, sigma, gamma, change_day, R0, R_post):
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    predicted = []

    for day in range(1, 71):
        if day <= change_day:
            beta = R0 * gamma
        else:
            beta = R_post * gamma
        dE = beta * S * I / population - sigma * E
        dI = sigma * E - gamma * I
        E += dE
        I += dI
        S = population - E - I - R
        new_cases = sigma * E
        predicted.append(new_cases)
    return predicted

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    best_R0 = None
    best_R_post = None
    best_change_day = None
    best_log_likelihood = -np.inf

    for change_day in range(0, 71):
        if change_day == 0:
            def objective(R0):
                R_post = R0
                predicted = simulate_model(population, initial_infectious, sigma, gamma, change_day, R0, R_post)
                nll = -np.sum(cases[i] * np.log(predicted[i] + 1e-10) - predicted[i] for i in range(len(cases)))
                return nll
            initial_guess = [1.0]
            result = minimize(objective, initial_guess, bounds=[(0.1, 10.0)])
            if result.success:
                current_log_likelihood = -result.fun
                if current_log_likelihood > best_log_likelihood:
                    best_log_likelihood = current_log_likelihood
                    best_R0 = result.x[0]
                    best_R_post = best_R0
                    best_change_day = change_day
        else:
            def objective(params):
                R0, R_post = params
                predicted = simulate_model(population, initial_infectious, sigma, gamma, change_day, R0, R_post)
                nll = -np.sum(cases[i] * np.log(predicted[i] + 1e-10) - predicted[i] for i in range(len(cases)))
                return nll
            initial_guess = [1.0, 1.0]
            bounds = [(0.1, 10.0), (0.0, 10.0)]
            result = minimize(objective, initial_guess, bounds=bounds)
            if result.success:
                current_log_likelihood = -result.fun
                if current_log_likelihood > best_log_likelihood:
                    best_log_likelihood = current_log_likelihood
                    best_R0 = result.x[0]
                    best_R_post = result.x[1]
                    best_change_day = change_day

    # Forecast
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    for day in range(1, 71):
        if day <= best_change_day:
            beta = best_R0 * gamma
        else:
            beta = best_R_post * gamma
        dE = beta * S * I / population - sigma * E
        dI = sigma * E - gamma * I
        E += dE
        I += dI
        S = population - E - I - R
    forecast = []
    for day in range(71, 71 + horizon):
        if day <= best_change_day:
            beta = best_R0 * gamma
        else:
            beta = best_R_post * gamma
        dE = beta * S * I / population - sigma * E
        dI = sigma * E - gamma * I
        E += dE
        I += dI
        S = population - E - I - R
        new_cases = sigma * E
        forecast.append(new_cases)
    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day if best_change_day != 0 else None,
        "forecast": forecast
    }
