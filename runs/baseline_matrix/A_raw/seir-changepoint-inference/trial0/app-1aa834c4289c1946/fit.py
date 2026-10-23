import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert days to time steps
    days = len(cases)

    # Convert latent and infectious periods to rates
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    # Initialize the SEIR model
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0

    # Store the simulated cases
    simulated_cases = np.zeros(days)

    # Simulate the SEIR model
    for d in range(days):
        # Calculate the number of new cases
        new_cases = np.random.poisson(I * sigma * gamma * population)
        simulated_cases[d] = new_cases

        # Update the compartments
        S -= new_cases
        E += new_cases
        I -= new_cases
        R += new_cases

    # Optimize to find the best fit for R0 and R_post
    def objective(params):
        R0, R_post, change_day = params
        # Simulate the SEIR model with the given parameters
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        simulated_cases = np.zeros(days)

        for d in range(days):
            # Calculate the number of new cases
            if d < change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
            new_cases = np.random.poisson(I * sigma * beta * population)
            simulated_cases[d] = new_cases

            # Update the compartments
            S -= new_cases
            E += new_cases
            I -= new_cases
            R += new_cases

        # Calculate the negative log-likelihood
        return -np.sum(poisson.logpmf(cases, simulated_cases))

    # Initial guess for R0 and R_post
    initial_guess = [3.0, 3.0, 50.0]

    # Optimize
    result = minimize(objective, initial_guess, method='L-BFGS-B')
    R0, R_post, change_day = result.x

    # Forecast future cases
    forecast = []
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0

    for d in range(days, days + horizon):
        # Calculate the number of new cases
        if d < change_day:
            beta = R0 * gamma
        else:
            beta = R_post * gamma
        new_cases = np.random.poisson(I * sigma * beta * population)
        forecast.append(new_cases)

        # Update the compartments
        S -= new_cases
        E += new_cases
        I -= new_cases
        R += new_cases

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day if change_day > 0 else None,
        "forecast": forecast
    }