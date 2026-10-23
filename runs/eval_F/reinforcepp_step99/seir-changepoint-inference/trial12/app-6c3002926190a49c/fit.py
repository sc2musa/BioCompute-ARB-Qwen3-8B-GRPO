import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse input
    cases = np.array(cases)
    N = population
    I0 = initial_infectious
    L = latent_period
    D = infectious_period
    H = horizon
    
    # Initial state
    S = N - I0
    E = 0
    I = I0
    R = 0
    
    # Time parameters
    T = len(cases)
    
    # Define the SEIR model
    def seir_model(t, S, E, I, R, beta):
        # Calculate new cases
        new_cases = I * beta * (S / N)
        
        # Update state
        S_new = S - new_cases
        E_new = E + new_cases
        I_new = I - (E_new - E) / L
        R_new = R + (I_new - I) / D
        
        return S_new, E_new, I_new, R_new
    
    # Define the objective function
    def objective(beta):
        # Initialize state
        S = N - I0
        E = 0
        I = I0
        R = 0
        
        # Track state over time
        states = np.zeros((T, 4))
        states[0] = [S, E, I, R]
        
        # Track beta over time
        betas = np.zeros(T)
        betas[0] = beta
        
        # Simulate the outbreak
        for t in range(1, T):
            S, E, I, R = seir_model(t, S, E, I, R, beta)
            states[t] = [S, E, I, R]
            betas[t] = beta
        
        # Calculate the error
        error = np.sum((cases - states[:, 2]) ** 2)
        return error
    
    # Optimize the parameters
    result = minimize(objective, x0=1.0, method='BFGS')
    
    # Extract the results
    R0 = result.x[0]
    
    # Detect the changepoint
    change_day = None
    
    # Make forecasts
    forecast = np.zeros(H)
    
    return {"R0": R0, "R_post": R0, "change_day": change_day, "forecast": forecast}