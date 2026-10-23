import numpy as np
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse input
    cases = np.array(cases)
    population = int(population)
    initial_infectious = float(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)
    
    # Calculate gamma and sigma
    gamma = 1 / infectious_period
    sigma = 1 / latent_period
    
    # Initialize populations
    S = population - initial_infectious
    E = 0.0
    I = np.array([initial_infectious])
    R = 0.0
    
    # Define the SEIR model
    def seir_model(t, y, R0):
        S, E, I, R = y
        beta = R0 * gamma
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Set initial conditions
    y0 = [S, E, I, R]
    
    # Solve the ODE
    sol = solve_ivp(seir_model, [0, 70], y0, method='BDF', rtol=1e-8, atol=1e-12 * population, args=(R0,))
    
    # Extract results
    S, E, I, R = sol.y
    
    # Calculate R0
    R0 = (I[70] / I[0]) ** (1 / 70) * population
    
    # Calculate forecast
    forecast = []
    for t in range(71, 71 + horizon):
        # Calculate new cases
        new_cases = I[t] - I[t-1]
        forecast.append(new_cases)
    
    # Return results
    return {"R0": R0, "R_post": R0, "change_day": None, "forecast": forecast}