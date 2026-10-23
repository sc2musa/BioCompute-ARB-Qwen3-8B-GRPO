import numpy as np
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse parameters
    N = population
    S0 = N - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0
    
    # Calculate gamma and sigma
    gamma = 1 / infectious_period
    sigma = 1 / latent_period
    
    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        beta = R0 * gamma
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Define the change point detection
    def change_point_detection(t, y):
        # This is a simplified version for demonstration
        # In a real implementation, we would use more sophisticated methods
        return 0
    
    # Solve the ODE
    sol = solve_ivp(
        seir_model,
        [0, 70],
        [S0, E0, I0, R0],
        t_eval=np.arange(0, 70, 1),
        method='BDF',
        rtol=1e-8,
        atol=1e-12 * N
    )
    
    # Extract the results
    S, E, I, R = sol.y
    
    # Calculate R0 and R_post
    R0 = np.max(I) / (initial_infectious / 70)
    R_post = R0
    
    # Calculate change_day
    change_day = None
    
    # Calculate forecast
    forecast = [0] * horizon
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast
    }