import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert days to time steps
    days = len(cases)
    time = np.arange(days)

    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0

    # SEIR model parameters
    sigma = 1 / latent_period
    gamma = 1 / infectious_period

    # Initial guess for R0 and R_post
    R0_guess = 3.0
    R_post_guess = 3.0
    change_day_guess = None

    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params

        # Initialize the SEIR model
        S = np.zeros(days)
        E = np.zeros(days)
        I = np.zeros(days)
        R = np.zeros(days)

        S[0] = S0
        E[0] = E0
        I[0] = I0
        R[0] = R0

        # Simulate the SEIR model
        for d in range(1, days):
            # Transmission rate
            beta = R0 * gamma if d < change_day else R_post * gamma

            # Update the SEIR model
            dS = -beta * S[d-1] * I[d-1] / population
            dE = beta * S[d-1] * I[d-1] / population - sigma * E[d-1]
            dI = sigma * E[d-1] - gamma * I[d-1]
            dR = gamma * I[d-1]

            S[d] = S[d-1] + dS
            E[d] = E[d-1] + dE
            I[d] = I[d-1] + dI
            R[d] = R[d-1] + dR

        # Calculate the expected number of new cases
        expected_cases = I[1:] - I[:-1]

        # Calculate the negative log-likelihood
        nll = -np.sum(poisson.logpmf(cases, expected_cases))

        return nll

    # Optimize the parameters
    result = minimize(seir_model, [R0_guess, R_post_guess, change_day_guess], method='L-BFGS-B')

    # Extract the optimized parameters
    R0_opt = result.x[0]
    R_post_opt = result.x[1]
    change_day_opt = result.x[2]

    # Forecast the next weeks
    forecast = []
    for d in range(days, days + horizon):
        # Transmission rate
        beta = R0_opt * gamma if d < change_day_opt else R_post_opt * gamma

        # Simulate the SEIR model for the forecast period
        S = np.zeros(1)
        E = np.zeros(1)
        I = np.zeros(1)
        R = np.zeros(1)

        S[0] = S[-1] + (beta * S[-1] * I[-1] / population)
        E[0] = E[-1] + (beta * S[-1] * I[-1] / population - sigma * E[-1])
        I[0] = I[-1] + (sigma * E[-1] - gamma * I[-1])
        R[0] = R[-1] + (gamma * I[-1])

        forecast.append(I[0] - I[-1])

    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt if change_day_opt is not None else None,
        "forecast": forecast
    }