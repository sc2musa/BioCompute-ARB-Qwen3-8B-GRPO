import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert days to time units
    days = np.arange(1, len(cases) + 1)
    time_units = days / infectious_period

    # Initialize parameters
    R0 = 2.0
    R_post = R0
    change_day = 0

    # Define the SEIR model
    def seir_model(t, R, change_day):
        # Calculate the number of infectious people at time t
        # This is a simplified version that doesn't account for all SEIR dynamics
        # In practice, we would need to solve the SEIR differential equations
        # For simplicity, we'll use a basic approximation
        infectious = np.zeros_like(t)
        infectious[0] = initial_infectious
        for i in range(1, len(t)):
            if t[i] < change_day:
                beta = R0 * infectious_period
            else:
                beta = R_post * infectious_period
            infectious[i] = infectious[i-1] + beta * infectious[i-1] * (population - infectious[i-1]) / population
        return infectious

    # Define the negative log-likelihood function
    def negative_log_likelihood(params):
        R0, R_post, change_day = params
        infectious = seir_model(time_units, R0, change_day)
        # Calculate the expected cases
        expected_cases = infectious[1:] - infectious[:-1]
        # Calculate the negative log-likelihood
        return -np.sum(poisson.logpmf(cases, expected_cases))

    # Optimize the parameters
    result = minimize(negative_log_likelihood, [R0, R_post, change_day], method='L-BFGS-B')
    R0, R_post, change_day = result.x

    # Forecast future cases
    future_days = np.arange(len(cases) + 1, len(cases) + 1 + horizon)
    future_time_units = future_days / infectious_period
    infectious = seir_model(future_time_units, R0, change_day)
    forecast = infectious[1:] - infectious[:-1]

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day if change_day is not None else None,
        "forecast": forecast.tolist()
    }