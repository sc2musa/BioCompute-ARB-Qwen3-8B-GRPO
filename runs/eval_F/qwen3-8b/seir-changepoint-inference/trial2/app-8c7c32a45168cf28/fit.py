import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert latent and infectious periods to rates
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0

    # Time steps
    days = len(cases)
    time = np.arange(days)

    # Simulate the SEIR model
    def seir_model(params):
        R0_val, R_post_val, change_day = params
        beta = np.full(days, R0_val * gamma)
        beta[change_day:] = R_post_val * gamma

        # Initialize the state variables
        S = np.full(days, S0)
        E = np.full(days, E0)
        I = np.full(days, I0)
        R = np.full(days, R0)

        # Simulate the SEIR model
        for t in range(1, days):
            # Calculate new infections
            new_infections = beta[t-1] * I[t-1] * (S[t-1] / population)
            # Calculate new exposed
            new_exposed = new_infections
            # Calculate new infectious
            new_infectious = sigma * E[t-1]
            # Calculate new removed
            new_removed = gamma * I[t-1]

            # Update the state variables
            S[t] = S[t-1] - new_infections
            E[t] = E[t-1] + new_exposed - new_infectious
            I[t] = I[t-1] + new_infectious - new_removed
            R[t] = R[t-1] + new_removed

        # Calculate the expected cases
        expected_cases = np.zeros(days)
        for t in range(days):
            expected_cases[t] = sigma * E[t]

        # Calculate the negative log-likelihood
        log_likelihood = 0.0
        for t in range(days):
            # Poisson distribution
            log_likelihood += np.log(np.exp(-expected_cases[t]) * (cases[t] ** expected_cases[t]) / np.math.factorial(cases[t]))

        return -log_likelihood

    # Initial guess for parameters
    initial_guess = [3.0, 3.0, 30.0]

    # Optimize the parameters
    result = minimize(seir_model, initial_guess, method='L-BFGS-B', bounds=[(0.1, 10.0), (0.1, 10.0), (1, 70)])

    # Extract the optimal parameters
    R0_opt = result.x[0]
    R_post_opt = result.x[1]
    change_day_opt = result.x[2]

    # Forecast the next horizon days
    forecast = np.zeros(horizon)
    for t in range(horizon):
        # Calculate the expected cases
        expected_cases = np.zeros(days + t + 1)
        for d in range(days + t + 1):
            expected_cases[d] = sigma * E[d]

        # Calculate the forecast
        forecast[t] = expected_cases[-1]

    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt if change_day_opt < 70 else None,
        "forecast": forecast.tolist()
    }