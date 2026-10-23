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
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    
    # Define SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        beta = R0 * infectious_period if t < change_day else R_post * infectious_period
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - E / latent_period
        dIdt = E / latent_period - I / infectious_period
        dRdt = I / infectious_period
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Estimate R0 and R_post
    # This is a simplified implementation - in practice we would use optimization
    # to find the best fit parameters
    
    # Initial guess for R0
    R0 = 3.0
    
    # Simulate the model
    sol = solve_ivp(seir_model, [0, 70], [S0, E0, I0, R0], t_eval=np.arange(70))
    
    # Calculate the fit
    fit_cases = sol.y[2] * population
    
    # Calculate the error
    error = np.sum((cases - fit_cases) ** 2) / np.sum(cases ** 2)
    
    # Adjust R0 based on the error
    R0 = 3.0 * (1 - error)
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R0,
        "change_day": None,
        "forecast": []
    }