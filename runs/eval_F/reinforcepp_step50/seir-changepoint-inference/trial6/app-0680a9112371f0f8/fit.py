import numpy as np
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse inputs
    cases = np.array(cases)
    population = int(population)
    initial_infectious = float(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)
    
    # Initialize SEIR model
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    
    # Calculate parameters
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    
    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        # Calculate the transmission rate
        beta = R0 * gamma
        dSdt = -beta * I * S / population
        dEdt = beta * I * S / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Initial conditions
    y0 = [S, E, I, R]
    
    # Solve the ODE
    sol = solve_ivp(seir_model, [0, 70], y0, method="BDF",
                    rtol=1e-8, atol=1e-12 * population)
    
    # Extract the results
    results = sol.y
    
    # Calculate R0
    R0 = (results[2, -1] - results[2, 0]) / (results[1, -1] - results[1, 0])
    
    # Calculate the forecast
    forecast = []
    for d in range(71, 71 + horizon):
        # Calculate the expected cases
        expected = results[2, d] - results[2, d-1]
        forecast.append(expected)
    
    # Return the results
    return {"R0": R0, "R_post": R0, "change_day": None, "forecast": forecast}