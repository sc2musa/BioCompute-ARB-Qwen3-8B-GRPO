import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Convert to numpy arrays
    cases = np.array(cases)
    
    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        
        # Initialize the state
        S = population - initial_infectious
        E = 0
        I = initial_infectious
        R = 0
        
        # Calculate the time steps
        T = len(cases)
        
        # Initialize the results
        results = np.zeros(T)
        
        # Simulate the SEIR model
        for t in range(T):
            # Calculate the transmission rate
            if t < change_day:
                beta = R0 * infectious_period
            else:
                beta = R_post * infectious_period
            
            # Calculate the new cases
            new_cases = beta * I
            
            # Update the state
            S -= new_cases
            E += new_cases
            I = E / latent_period
            R = I / infectious_period
            
            # Store the results
            results[t] = new_cases
        
        # Calculate the error
        error = np.sum((results - cases) ** 2)
        return error
    
    # Set the initial guess
    initial_guess = [3.0, 3.0, 0]
    
    # Set the bounds
    bounds = [(0.1, 10.0), (0.1, 10.0), (0, T)]
    
    # Optimize the parameters
    result = minimize(seir_model, initial_guess, bounds=bounds)
    
    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Calculate the forecast
    forecast = np.zeros(horizon)
    
    # Simulate the SEIR model for the forecast
    for t in range(horizon):
        # Calculate the transmission rate
        if t < change_day:
            beta = R0 * infectious_period
        else:
            beta = R_post * infectious_period
        
        # Calculate the new cases
        new_cases = beta * I
        
        # Update the state
        S -= new_cases
        E += new_cases
        I = E / latent_period
        R = I / infectious_period
        
        # Store the results
        forecast[t] = new_cases
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }