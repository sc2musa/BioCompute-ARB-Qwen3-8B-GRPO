import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse input
    cases = np.array(cases)
    N = population
    I0 = initial_infectious
    L = latent_period
    I = infectious_period
    H = horizon
    
    # Initial state
    S = N - I0
    E = 0
    I = I0
    R = 0
    
    # Time parameters
    T = len(cases)
    
    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        
        # Initialize state
        S = N - I0
        E = 0
        I = I0
        R = 0
        
        # Create time array
        t = np.arange(T)
        
        # Calculate beta(t)
        beta = np.zeros(T)
        for d in range(T):
            if d < change_day:
                beta[d] = R0 * I
            else:
                beta[d] = R_post * I
        
        # Calculate the expected cases
        expected = np.zeros(T)
        for d in range(T):
            if d > 0:
                # Calculate the number of new cases
                new_cases = beta[d-1] * S * (1 - np.exp(-beta[d-1] * L))
                expected[d] = new_cases
                
                # Update state
                S -= new_cases
                E += new_cases
                I = I * np.exp(-beta[d-1] * I)
                R = R * np.exp(-beta[d-1] * R)
        
        # Calculate the error
        error = np.sum((cases - expected) ** 2)
        return error
    
    # Initial guess
    initial_guess = [1.0, 1.0, 0]
    
    # Optimize the parameters
    result = minimize(seir_model, initial_guess, method='BFGS', tol=1e-8)
    
    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Make forecasts
    forecast = np.zeros(H)
    for d in range(H):
        # Calculate the number of new cases
        new_cases = beta[T-1] * S * (1 - np.exp(-beta[T-1] * L))
        forecast[d] = new_cases
        
        # Update state
        S -= new_cases
        E += new_cases
        I = I * np.exp(-beta[T-1] * I)
        R = R * np.exp(-beta[T-1] * R)
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }