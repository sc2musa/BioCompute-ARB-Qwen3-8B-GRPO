import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}.
    """
    days = len(cases)
    time_steps = np.arange(days)

    # Define the SEIR model
    def seir_model(params, time):
        R0, R_post, change_day = params
        gamma = 1.0 / infectious_period
        sigma = 1.0 / latent_period
        beta = np.zeros_like(time)
        beta[time < change_day] = R0 * gamma
        beta[time >= change_day] = R_post * gamma

        # Initialize the state variables
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0

        # Simulate the SEIR model
        for t in range(1, days):
            # Calculate the number of new infections
            new_infections = beta[t] * I * S / population
            # Update the state variables
            S -= new_infections
            E += new_infections
            E -= sigma * E
            I += sigma * E - gamma * I
            R += gamma * I

            # Record the number of new cases
            cases[t] = I - I_prev
            I_prev = I

        return cases

    # Estimate R0 and R_post
    # Initial guess for R0 and R_post
    initial_guess = [3.0, 3.0, 0.0]

    # Define the objective function
    def objective(params):
        R0, R_post, change_day = params
        # Simulate the SEIR model with the given parameters
        simulated_cases = seir_model(params, time_steps)
        # Calculate the difference between observed and simulated cases
        difference = np.sum((np.array(cases) - simulated_cases) ** 2)
        return difference

    # Optimize the parameters
    result = minimize(objective, initial_guess, method='L-BFGS-B')
    R0, R_post, change_day = result.x

    # Forecast future cases
    forecast = []
    # Simulate the SEIR model for the next horizon days
    for t in range(days, days + horizon):
        # Calculate the number of new infections
        new_infections = beta[t] * I * S / population
        # Update the state variables
        S -= new_infections
        E += new_infections
        E -= sigma * E
        I += sigma * E - gamma * I
        R += gamma * I

        # Record the number of new cases
        forecast.append(I - I_prev)
        I_prev = I

    return {"R0": R0, "R_post": R_post, "change_day": change_day, "forecast": forecast}