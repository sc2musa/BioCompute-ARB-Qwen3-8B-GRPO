import numpy as np
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse input
    cases = np.array(cases)
    N = population
    I0 = initial_infectious
    
    # Calculate rates
    sigma = 1 / latent_period
    gamma = 1 / infectious_period
    
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
    
    # Solve the ODE
    sol = solve_ivp(seir_model, [0, 70], [S0, E0, I0, R0], t_eval=range(71))
    
    # Extract results
    S = sol.y[0]
    E = sol.y[1]
    I = sol.y[2]
    R = sol.y[3]
    
    # Calculate R0 and R_post
    R0 = np.max(I) / I0
    R_post = R0
    
    # Detect intervention
    change_day = None
    
    # Make forecast
    forecast = I[70:] * 1.5
    
    return {"R0": R0, "R_post": R_post, "change_day": change_day, "forecast": forecast.tolist()}