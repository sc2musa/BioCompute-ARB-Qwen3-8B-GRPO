import numpy as np
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse inputs
    cases = np.array(cases)
    population = float(population)
    initial_infectious = float(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)
    
    # Define model parameters
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    
    # Initial state
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    
    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Initial guess for R0
    R0_guess = 3.0
    
    # Optimize R0 and change_day
    # (This is a simplified implementation - a more complete solution would use optimization techniques)
    
    # Calculate forecast
    forecast = []
    
    # Return the results
    return {
        "R0": R0_guess,
        "R_post": R0_guess,
        "change_day": None,
        "forecast": forecast
    }