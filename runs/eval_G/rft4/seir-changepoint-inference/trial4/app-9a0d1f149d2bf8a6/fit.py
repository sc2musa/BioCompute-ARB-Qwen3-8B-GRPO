import numpy as np
from scipy.optimize import minimize

def simulate_model(population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    expected_cases = []

    for day in range(1, 71):
        if change_day is not None:
            if day < change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
        else:
            beta = R0 * gamma

        new_S = S - beta * S * I
        new_E = E + beta * S * I - sigma * E
        new_I = I + sigma * E - gamma * I
        new_R = R + gamma * I

        S, E, I, R = new_S, new_E, new_I, new_R

        expected_cases.append(sigma * E)

    return expected_cases

def log_likelihood(expected, observed):
    expected = np.array(expected)
    observed = np.array(observed)
    expected = np.maximum(expected, 1e-10)
    return np.sum( -observed * expected + np.log(expected) )

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    best_change_day = None
    best_R0 = None
    best_R_post = None
    best_forecast = None
    best_likelihood = -np.inf

    # Try all possible change days
    for change_day_candidate in [None] + list(range(1, 71)):
        if change_day_candidate is None:
            # Fit R0 only
            def objective(R0):
                expected = simulate_model(population, initial_infectious, latent_period, infectious_period, R0, R0, change_day_candidate)
                return -log_likelihood(expected, cases)
            # Initial guess for R0
            initial_guess = 2.0
            result = minimize(objective, x0=initial_guess, method='Nelder-Mead')
            R0 = result.x[0]
            R_post = R0
        else:
            # Fit R0 and R_post
            def objective(params):
                R0, R_post = params
                expected = simulate_model(population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day_candidate)
                return -log_likelihood(expected, cases)
            # Initial guess for R0 and R_post
            initial_guess = [2.0, 1.0]
            result = minimize(objective, x0=initial_guess, method='Nelder-Mead')
            R0 = result.x[0]
            R_post = result.x[1]

        # Compute the likelihood for this candidate
        expected = simulate_model(population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day_candidate)
        current_likelihood = log_likelihood(expected, cases)
        if current_likelihood > best_likelihood:
            best_likelihood = current_likelihood
            best_change_day = change_day_candidate
            best_R0 = R0
            best_R_post = R_post

    # Now, forecast the next days
    # Simulate up to day 70
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    for day in range(1, 71):
        if best_change_day is not None:
            if day < best_change_day:
                beta = best_R0 * gamma
            else:
                beta = best_R_post * gamma
        else:
            beta = best_R0 * gamma
        new_S = S - beta * S * I
        new_E = E + beta * S * I - sigma * E
        new_I = I + sigma * E - gamma * I
        new_R = R + gamma * I
        S, E, I, R = new_S, new_E, new_I, new_R

    # Now, simulate from day 71 to 70 + horizon
    forecast = []
    for day in range(71, 70 + horizon + 1):
        if best_change_day is not None:
            if day < best_change_day:
                beta = best_R0 * gamma
            else:
                beta = best_R_post * gamma
        else:
            beta = best_R0 * gamma
        new_S = S - beta * S * I
        new_E = E + beta * S * I - sigma * E
        new_I = I + sigma * E - gamma * I
        new_R = R + gamma * I
        S, E, I, R = new_S, new_E, new_I, new_R
        forecast.append(sigma * E)

    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": forecast
    }