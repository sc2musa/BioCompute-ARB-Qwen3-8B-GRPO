import numpy as np
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse the input
    N = population
    I0 = initial_infectious
    L = latent_period
    G = infectious_period
    
    # Calculate the time steps
    T = len(cases)
    
    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        beta = R0 * G
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Initial conditions
    y0 = [N - I0, 0, I0, 0]
    
    # Solve the ODE
    sol = solve_ivp(seir_model, [0, T], y0, t_eval=np.arange(0, T+1))
    
    # Extract the results
    S = sol.y[0]
    E = sol.y[1]
    I = sol.y[2]
    R = sol.y[3]
    
    # Calculate the number of new cases
    new_cases = I[1:] - I[:-1]
    
    # Return the results
    return {"R0": 0.0, "R_post": 0.0, "change_day": None, "forecast": []}