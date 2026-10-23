import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    # Initial parameters
    R0 = 3.0
    R_post = 3.0
    change_day = None

    # Define the SEIR model
    def seir_model(params, t):
        S, E, I, R = params
        beta = params[0] if t < change_day else params[1]
        gamma = 1 / infectious_period
        sigma = 1 / latent_period
        dS = -beta * I * S / population
        dE = beta * I * S / population - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        return [dS, dE, dI, dR]

    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0
    initial_params = [S0, E0, I0, R0]

    # Define the objective function
    def objective(params):
        # Extract parameters
        S, E, I, R = params
        # Calculate the predicted cases
        predicted_cases = np.zeros(len(cases))
        for t in range(len(cases)):
            # Solve the ODE for this day
            sol = solve_ivp(seir_model, [t, t+1], initial_params, method='BDF', rtol=1e-8, atol=1e-12 * population)
            predicted_cases[t] = sol.y[2][-1]  # Infectious cases
        # Calculate the error
        error = np.sum((cases - predicted_cases) ** 2)
        return error

    # Optimize the parameters
    result = minimize(objective, initial_params, method='BFGS', tol=1e-8)
    optimized_params = result.x

    # Extract the results
    R0 = optimized_params[0]
    R_post = optimized_params[1]
    change_day = optimized_params[2]
    forecast = np.zeros(horizon)
    for t in range(horizon):
        # Solve the ODE for the forecast
        sol = solve_ivp(seir_model, [len(cases), len(cases)+t+1], initial_params, method='BDF', rtol=1e-8, atol=1e-12 * population)
        forecast[t] = sol.y[2][-1]  # Infectious cases

    return {"R0": R0, "R_post": R_post, "change_day": change_day, "forecast": forecast.tolist()}