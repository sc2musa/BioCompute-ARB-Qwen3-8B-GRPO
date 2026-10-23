import numpy as np
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Convert days to time units
    days = len(cases)
    time = np.linspace(0, days, days)
    
    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0
    
    # Define the SEIR model
    def seir_model(t, y, beta, sigma, gamma):
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Define the objective function
    def objective(beta, t, y):
        S, E, I, R = y
        return np.sum((cases - (I - I0)) ** 2)
    
    # Initial guess for beta
    beta_initial = 1.0
    
    # Solve the optimization problem
    result = solve_ivp(seir_model, [0, days], [S0, E0, I0, R0], t_eval=time, method='BDF', 
                      rtol=1e-8, atol=1e-12 * population, 
                      args=(beta_initial, 1/latent_period, 1/infectious_period))
    
    # Extract the results
    S, E, I, R = result.y[-1]
    
    # Calculate the basic reproduction number
    R0 = np.max(I) / initial_infectious
    
    # Calculate the forecast
    forecast = []
    for t in range(days, days + horizon):
        S, E, I, R = result.y[-1]
        dSdt = -beta_initial * S * I / population
        dEdt = beta_initial * S * I / population - (1/latent_period) * E
        dIdt = (1/latent_period) * E - (1/infectious_period) * I
        dRdt = (1/infectious_period) * I
        forecast.append(I + dIdt)
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R0,
        "change_day": None,
        "forecast": forecast
    }