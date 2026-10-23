import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse inputs
    N = population
    S0 = N - initial_infectious
    I0 = initial_infectious
    E0 = 0
    
    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        beta = R0 * gamma if t >= change_day else R0 * gamma
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Define the ODE solver
    sol = solve_ivp(seir_model, [0, 70], [S0, E0, I0, 0], method="BDF", rtol=1e-8, atol=1e-12 * N)
    
    # Calculate the forecast
    forecast = []
    for t in range(71, 70 + horizon):
        # Calculate the expected cases for day t
        expected_cases = sol.y[2][t] - sol.y[2][t-1]
        forecast.append(expected_cases)
    
    return {
        "R0": R0,
        "R_post": R0,
        "change_day": None,
        "forecast": forecast
    }