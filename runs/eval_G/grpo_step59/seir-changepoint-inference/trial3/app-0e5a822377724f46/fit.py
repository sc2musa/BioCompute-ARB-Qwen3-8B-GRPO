import numpy as np
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Calculate rates
    sigma = 1 / latent_period
    gamma = 1 / infectious_period

    # Initial guesses
    R0_initial = 1.0
    R_post_initial = 1.0
    change_day_initial = 0

    # Define error function
    def error_function(params):
        R0, R_post, change_day = params
        beta = [R0 * gamma] * 70
        for d in range(change_day, 70):
            beta[d] = R_post * gamma

        # Initialize state
        S = population - initial_infectious
        E = 0
        I = initial_infectious
        R = 0

        # Simulate
        error = 0
        for d in range(70):
            current_beta = beta[d]
            dS = -current_beta * S * I / population
            dE = current_beta * S * I / population - sigma * E
            dI = sigma * E - gamma * I
            dR = gamma * I

            S += dS
            E += dE
            I += dI
            R += dR

            error += (dI - cases[d]) ** 2

        return error

    # Optimize
    result = minimize(error_function, [R0_initial, R_post_initial, change_day_initial], method='BFGS')

    # Extract results
    R0_best = result.x[0]
    R_post_best = result.x[1]
    change_day_best = int(result.x[2])

    # Forecast
    S = population - initial_infectious
    E = 0
    I = initial_infectious
    R = 0

    for d in range(70):
        current_beta = R0_best * gamma if d < change_day_best else R_post_best * gamma
        dS = -current_beta * S * I / population
        dE = current_beta * S * I / population - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I

        S += dS
        E += dE
        I += dI
        R += dR

    forecast = []
    for d in range(70, 70 + horizon):
        current_beta = R_post_best * gamma
        dS = -current_beta * S * I / population
        dE = current_beta * S * I / population - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I

        S += dS
        E += dE
        I += dI
        R += dR

        forecast.append(dI)

    return {
        "R0": R0_best,
        "R_post": R_post_best,
        "change_day": change_day_best,
        "forecast": forecast
    }