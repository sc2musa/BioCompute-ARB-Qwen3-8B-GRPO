import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert parameters to rates
    sigma = 1 / latent_period
    gamma = 1 / infectious_period

    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0

    # Define the SEIR model
    def seir_model(t, y, beta, R_post, change_day):
        S, E, I, R = y
        if t < change_day:
            beta_t = beta
        else:
            beta_t = R_post * gamma
        dSdt = -beta_t * S * I / population
        dEdt = beta_t * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    # Objective function to minimize
    def objective(params, cases, population, sigma, gamma, horizon):
        beta, R_post, change_day = params
        # Initial conditions
        S0 = population - initial_infectious
        E0 = 0
        I0 = initial_infectious
        R0 = 0

        # Solve the ODE
        sol = solve_ivp(seir_model, [0, 70], [S0, E0, I0, R0], args=(beta, R_post, change_day), dense_output=True)

        # Get the expected number of new cases
        expected_cases = []
        for d in range(1, 71):
            t = d
            S, E, I, R = sol.sol(t)
            new_cases = sigma * E
            expected_cases.append(new_cases)

        # Calculate the negative log-likelihood
        # We'll use a simple Poisson likelihood for the observed cases
        log_likelihood = 0
        for i in range(len(cases)):
            if expected_cases[i] > 0:
                log_likelihood += np.log(np.mean([np.random.poisson(expected_cases[i]) for _ in range(1000)]))

        # Add a penalty for large R_post
        penalty = 0
        if R_post < 0.75 * beta / gamma:
            penalty = 100 * (beta / gamma - R_post)

        return -log_likelihood + penalty

    # Initial guess for R0
    R0_initial = 3.0

    # Optimize the parameters
    result = minimize(objective, [R0_initial, R0_initial, 0], args=(cases, population, sigma, gamma, horizon), method='L-BFGS-B', bounds=[(0.1, 10), (0.1, 10), (0, 70)])

    # Extract the estimated parameters
    beta_est = result.x[0]
    R_post_est = result.x[1]
    change_day_est = result.x[2]

    # Forecast the next horizon days
    forecast = []
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0

    sol = solve_ivp(seir_model, [0, 70], [S0, E0, I0, R0], args=(beta_est, R_post_est, change_day_est), dense_output=True)

    for d in range(71, 71 + horizon):
        t = d
        S, E, I, R = sol.sol(t)
        new_cases = sigma * E
        forecast.append(new_cases)

    return {
        "R0": beta_est / gamma,
        "R_post": R_post_est / gamma,
        "change_day": change_day_est if R_post_est < 0.75 * beta_est / gamma else None,
        "forecast": forecast
    }