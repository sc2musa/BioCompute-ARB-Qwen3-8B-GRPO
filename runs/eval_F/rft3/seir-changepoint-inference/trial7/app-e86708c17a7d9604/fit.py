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
    
    # Initialize variables
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
        # Calculate beta based on the current day
        if change_day is not None and t >= change_day:
            beta = R_post * gamma
        else:
            beta = R0 * gamma
        
        dSdt = -beta * I * S / population
        dEdt = beta * I * S / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Initial values
    y0 = [S, E, I, R]
    
    # Solve the ODE
    sol = solve_ivp(seir_model, [0, 70], y0, t_eval=np.arange(70), method='BDF', rtol=1e-8)
    
    # Extract results
    S = sol.y[0]
    E = sol.y[1]
    I = sol.y[2]
    R = sol.y[3]
    
    # Calculate R0
    R0 = (I[1] - I[0]) / (S[1] - S[0]) * population
    
    # Calculate change point
    change_day = None
    for d in range(1, 70):
        if I[d] < I[d-1] * 0.75:
            change_day = d
            break
    
    # Calculate forecast
    forecast = []
    for d in range(70, 70 + horizon):
        # Calculate beta for this day
        if change_day is not None and d >= change_day:
            beta = R_post * gamma
        else:
            beta = R0 * gamma
        
        # Calculate new cases
        new_cases = beta * I[-1] * population / (population - I[-1])
        forecast.append(new_cases)
        
    # Return results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast
    }