import numpy as np
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Parse inputs
    cases = np.array(cases)
    population = int(population)
    initial_infectious = int(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)

    # Initial state: S, E, I, R
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0

    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        beta = R0 * infectious_period
        gamma = 1.0 / infectious_period
        sigma = 1.0 / latent_period

        # Compute the rate of change
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dI_dt = sigma * E - gamma * I
        dRdt = gamma * I

        return [dSdt, dEdt, dI_dt, dRdt]

    # Solve the ODE
    sol = solve_ivp(seir_model, [0, 70], [S0, E0, I0, R0], method='BDF', rtol=1e-8, atol=1e-12 * population)

    # Extract the results
    S, E, I, R = sol.y

    # Compute the forecast
    forecast = np.zeros(horizon)
    for d in range(horizon):
        # Compute the expected new cases
        expected_cases = I[-1] * (R0 * infectious_period) * (1 - np.exp(-R0 * infectious_period * d))
        forecast[d] = expected_cases

    # Return the results
    return {
        "R0": R0,
        "R_post": R0,
        "change_day": None,
        "forecast": forecast.tolist()
    }