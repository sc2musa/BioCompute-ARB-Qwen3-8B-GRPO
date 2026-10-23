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
    
    # Initialize SEIR model
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    
    # Calculate beta and gamma
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    
    # Define SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        beta = R0 * gamma
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Solve SEIR model
    sol = solve_ivp(seir_model, [0, 70], [S0, E0, I0, R0], t_eval=np.arange(0, 70+1))
    
    # Extract results
    S, E, I, R = sol.y.T
    
    # Calculate R0
    R0 = np.max(I) / initial_infectious
    
    # Calculate R_post
    R_post = R0
    
    # Calculate change_day
    change_day = None
    
    # Calculate forecast
    forecast = np.zeros(horizon)
    
    # Return results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }