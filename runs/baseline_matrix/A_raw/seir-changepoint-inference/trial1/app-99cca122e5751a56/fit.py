import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import json

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert days to time units
    days = np.arange(1, len(cases) + 1)

    # Convert latent and infectious periods to time units
    latent_period_days = latent_period
    infectious_period_days = infectious_period

    # Initial parameters
    R0 = 3.0
    R_post = R0
    change_day = 40.0  # Initial guess for change day

    # Objective function to minimize
    def objective(params):
        R0, R_post, change_day = params
        # Simulate the SEIR model
        S = population - initial_infectious
        E = 0
        I = initial_infectious
        R = 0
        new_cases = []

        for day in range(1, len(cases) + 1):
            # Calculate the transmission rate
            if day < change_day:
                beta = R0 * infectious_period_days
            else:
                beta = R_post * infectious_period_days

            # Update the number of new cases
            new_cases.append(beta * I)

            # Update the compartments
            dE = beta * I
            dI = (1 - np.exp(-beta * infectious_period_days)) * I
            dR = (1 - np.exp(-beta * infectious_period_days)) * I

            E += dE
            I = I - dI + dR
            R += dR

        # Calculate the negative log-likelihood
        log_likelihood = 0
        for observed, expected in zip(cases, new_cases):
            if expected == 0:
                log_likelihood += 0  # Avoid log(0)
            else:
                log_likelihood += np.log(poisson.pmf(observed, expected))

        return -log_likelihood

    # Optimize the parameters
    result = minimize(objective, [R0, R_post, change_day], method='L-BFGS-B', bounds=[(0.1, 10), (0.1, 10), (1, 70)])
    R0, R_post, change_day = result.x

    # Forecast future cases
    forecast = []
    for day in range(len(cases) + 1, len(cases) + horizon + 1):
        # Calculate the transmission rate
        if day < change_day:
            beta = R0 * infectious_period_days
        else:
            beta = R_post * infectious_period_days

        # Simulate the SEIR model
        S = population - initial_infectious
        E = 0
        I = initial_infectious
        R = 0
        new_cases = []

        for d in range(1, day + 1):
            # Calculate the transmission rate
            if d < change_day:
                beta = R0 * infectious_period_days
            else:
                beta = R_post * infectious_period_days

            # Update the number of new cases
            new_cases.append(beta * I)

            # Update the compartments
            dE = beta * I
            dI = (1 - np.exp(-beta * infectious_period_days)) * I
            dR = (1 - np.exp(-beta * infectious_period_days)) * I

            E += dE
            I = I - dI + dR
            R += dR

        forecast.append(new_cases[-1])

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast
    }

# Test the fit function with the first example
with open('$ROOT/runs/matrix_v2/A_raw/seir-changepoint-inference/trial1/app-99cca122e5751a56/data/example0.json') as f:
    data = json.load(f)
    inputs = data['inputs']
    truth = data['truth']

    result = fit(
        inputs['cases'],
        inputs['population'],
        inputs['initial_infectious'],
        inputs['latent_period'],
        inputs['infectious_period'],
        inputs['horizon']
    )

    print("R0:", result['R0'])
    print("R_post:", result['R_post'])
    print("change_day:", result['change_day'])
    print("forecast:", result['forecast'])