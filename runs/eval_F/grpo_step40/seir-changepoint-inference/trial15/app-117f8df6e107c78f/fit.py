import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Convert days to time steps
    days = len(cases)
    time = np.arange(days)
    
    # Initial state
    S = population - initial_infectious
    E = 0
    I = initial_infectious
    R = 0
    
    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        dSdt = -beta(t) * S * I / population
        dEdt = beta(t) * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Define the parameters
    sigma = 1 / latent_period
    gamma = 1 / infectious_period
    
    # Define the objective function
    def objective(params):
        R0, R_post, change_day = params
        
        # Define the transmission rate
        def beta(t):
            if t < change_day:
                return R0 * gamma
            else:
                return R_post * gamma
        
        # Initial state
        S = population - initial_infectious
        E = 0
        I = initial_infectious
        R = 0
        
        # Solve the ODE
        sol = solve_ivp(seir_model, [0, days], [S, E, I, R], t_eval=time, method='BDF', rtol=1e-8)
        
        # Calculate the expected cases
        expected = sol.y[2][1:]  # Infectious cases
        
        # Calculate the error
        error = np.sum((cases - expected) ** 2) / np.sum(expected)
        
        return error
    
    # Initial guess
    initial_guess = [3.0, 3.0, 0]
    
    # Optimize the parameters
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=[(0.1, 10), (0.1, 10), (0, days)])
    
    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Forecast the next weeks
    forecast = []
    for d in range(days, days + horizon):
        # Calculate the expected cases
        S = population - initial_infectious
        E = 0
        I = initial_infectious
        R = 0
        
        # Solve the ODE
        sol = solve_ivp(seir_model, [0, d], [S, E, I, R], t_eval=[d], method='BDF', rtol=1e-8)
        
        # Calculate the expected cases
        expected = sol.y[2][0]  # Infectious cases
        
        forecast.append(expected)
    
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast
    }