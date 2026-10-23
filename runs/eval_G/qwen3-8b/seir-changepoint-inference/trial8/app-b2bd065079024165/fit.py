import numpy as np
from scipy.optimize import minimize

def simulate_seir(population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day, days=70):
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period

    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0

    beta_pre = R0 * gamma
    beta_post = R_post * gamma

    predicted_cases = []

    for day in range(days):
        beta = beta_pre if day < change_day else beta_post
        new_cases = sigma * E
        predicted_cases.append(new_cases)
        dS = -beta * I * S / population
        dE = beta * I * S / population - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        S += dS
        E += dE
        I += dI
        R += dR

    return predicted_cases

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period

    best_error = float('inf')
    best_R0 = best_R_post = best_change_day = None

    for change_day in range(0, 71):
        def objective(params):
            R0, R_post = params
            predicted = simulate_seir(population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day)
            return np.sum((np.array(predicted) - np.array(cases)) ** 2)

        result = minimize(objective, [2.0, 1.0], method='L-BFGS-B', bounds=[(0.1, 10.0), (0.1, 10.0)])
        if result.success and result.fun < best_error:
            best_error = result.fun
            best_R0, best_R_post = result.x
            best_change_day = change_day

    # Handle no intervention case
    if abs(best_R_post - best_R0) < 0.05:
        best_change_day = None

    # Forecast
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    beta_pre = best_R0 * gamma
    beta_post = best_R_post * gamma

    for day in range(70):
        beta = beta_pre if day < best_change_day else beta_post
        dS = -beta * I * S / population
        dE = beta * I * S / population - (1.0 / latent_period) * E
        dI = (1.0 / latent_period) * E - gamma * I
        dR = gamma * I
        S += dS
        E += dE
        I += dI
        R += dR

    forecast = []
    for _ in range(horizon):
        beta = beta_pre if 70 < best_change_day else beta_post
        dS = -beta * I * S / population
        dE = beta * I * S / population - (1.0 / latent_period) * E
        dI = (1.0 / latent_period) * E - gamma * I
        dR = gamma * I
        S += dS
        E += dE
        I += dI
        R += dR
        forecast.append(sigma * E)

    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": forecast
    }