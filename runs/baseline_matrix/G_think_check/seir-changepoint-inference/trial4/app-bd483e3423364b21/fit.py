import numpy as np
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    def error(params):
        R0, R_post, change_day = params
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        simulated = []
        for d in range(1, 71):
            if d < change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
            delta_S = -beta * I * S / population
            delta_E = beta * I * S / population - sigma * E
            delta_I = sigma * E - gamma * I
            delta_R = gamma * I
            new_cases = sigma * E
            simulated.append(new_cases)
            S += delta_S
            E += delta_E
            I += delta_I
            R += delta_R
        error = sum((simulated[i] - cases[i])**2 for i in range(70))
        return error

    initial_guess = [2.0, 1.0, 30.0]
    result = minimize(error, initial_guess, method='L-BFGS-B')
    best_R0 = result.x[0]
    best_R_post = result.x[1]
    best_change_day = result.x[2]

    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    for d in range(1, 71):
        if d < best_change_day:
            beta = best_R0 * gamma
        else:
            beta = best_R_post * gamma
        delta_S = -beta * I * S / population
        delta_E = beta * I * S / population - sigma * E
        delta_I = sigma * E - gamma * I
        delta_R = gamma * I
        new_cases = sigma * E
        S += delta_S
        E += delta_E
        I += delta_I
        R += delta_R

    forecast = []
    for day in range(71, 70 + horizon + 1):
        if day < best_change_day:
            beta = best_R0 * gamma
        else:
            beta = best_R_post * gamma
        delta_S = -beta * I * S / population
        delta_E = beta * I * S / population - sigma * E
        delta_I = sigma * E - gamma * I
        delta_R = gamma * I
        new_cases = sigma * E
        forecast.append(new_cases)
        S += delta_S
        E += delta_E
        I += delta_I
        R += delta_R

    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": forecast
    }