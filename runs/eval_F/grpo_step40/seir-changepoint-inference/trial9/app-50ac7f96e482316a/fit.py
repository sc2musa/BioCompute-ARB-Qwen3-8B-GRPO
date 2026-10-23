import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Parse inputs
    cases = np.array(cases)
    population = np.array(population)
    initial_infectious = np.array(initial_infectious)
    latent_period = np.array(latent_period)
    infectious_period = np.array(infectious_period)
    horizon = np.array(horizon)

    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        # Initialize the state
        S = population - initial_infectious
        E = np.zeros_like(S)
        I = initial_infectious
        R = np.zeros_like(S)

        # Time steps
        t = np.arange(1, 71)

        # Calculate the transmission rate
        beta = np.zeros_like(t)
        for d in range(70):
            if d < change_day:
                beta[d] = R0 * infectious_period
            else:
                beta[d] = R_post * infectious_period

        # Simulate the SEIR model
        for d in range(70):
            # Calculate the new cases
            new_cases = np.zeros_like(S)
            for s in range(len(S)):
                if S[s] > 0:
                    new_cases[s] = S[s] * beta[d] / population[s]
            # Update the state
            E = E + new_cases
            I = I + np.sum(new_cases)
            S = S - np.sum(new_cases)
            R = R + np.sum(new_cases)

        # Return the forecast
        forecast = np.zeros(horizon)
        for d in range(horizon):
            # Calculate the new cases
            new_cases = np.zeros_like(S)
            for s in range(len(S)):
                if S[s] > 0:
                    new_cases[s] = S[s] * beta[69] / population[s]
            # Update the state
            E = E + new_cases
            I = I + np.sum(new_cases)
            S = S - np.sum(new_cases)
            R = R + np.sum(new_cases)
            forecast[d] = np.sum(new_cases)

        return forecast

    # Optimize the parameters
    initial_guess = [3.0, 3.0, 50]
    bounds = [(0.1, 10.0), (0.1, 10.0), (1, 70)]
    result = minimize(lambda x: np.sum((seir_model(x) - cases) ** 2), initial_guess, bounds=bounds)

    # Return the results
    return {
        "R0": result.x[0],
        "R_post": result.x[1],
        "change_day": result.x[2],
        "forecast": seir_model(result.x)
    }