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
    
    # Initial state
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    
    # Calculate R0
    R0 = 1.0 / (latent_period * infectious_period)
    
    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        # Calculate the transmission rate
        if t < change_day:
            beta = R0 * gamma
        else:
            beta = R_post * gamma
        
        # Calculate the derivatives
        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Solve the ODE
    sol = solve_ivp(seir_model, [0, 70], [S, E, I, R], method="BDF",
                    t_eval=np.arange(0, 70 + 1),
                    rtol=1e-8, atol=1e-12 * population)
    
    # Extract the results
    S, E, I, R = sol.y.T
    
    # Calculate the forecast
    forecast = []
    for t in range(70, 70 + horizon):
        # Calculate the transmission rate
        if t < change_day:
            beta = R0 * gamma
        else:
            beta = R_post * gamma
        
        # Calculate the new cases
        new_cases = beta * S[t] * I[t]
        forecast.append(new_cases)
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast
    }