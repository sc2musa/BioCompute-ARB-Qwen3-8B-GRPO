import numpy as np
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse input
    cases = np.array(cases)
    N = population
    I0 = initial_infectious
    L = latent_period
    G = infectious_period
    
    # Initial state
    S0 = N - I0
    E0 = 0
    I0 = I0
    R0 = 0
    
    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Initial values
    y0 = [S0, E0, I0, R0]
    
    # Solve the ODE
    sol = solve_ivp(seir_model, [0, 70], y0, t_eval=np.arange(70))
    
    # Extract the results
    S, E, I, R = sol.y
    
    # Calculate R0
    R0 = np.max(I) / I0
    
    # Calculate the forecast
    forecast = np.zeros(horizon)
    for t in range(horizon):
        forecast[t] = np.max(I) * np.exp(-gamma * t)
    
    return {"R0": R0, "R_post": R0, "change_day": None, "forecast": forecast.tolist()}