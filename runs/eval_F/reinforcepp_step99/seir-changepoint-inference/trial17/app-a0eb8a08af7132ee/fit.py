import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse input
    cases = np.array(cases)
    days = len(cases)
    
    # Initialize variables
    S = population - initial_infectious
    E = 0
    I = initial_infectious
    R = 0
    
    # Calculate time constants
    sigma = 1 / latent_period
    gamma = 1 / infectious_period
    
    # Define the objective function
    def objective(params):
        R0, R_post, change_day = params
        
        # Set the transmission rate
        beta = R0 * gamma
        
        # Calculate the expected cases
        expected = np.zeros(days)
        
        # Calculate the expected cases for each day
        for d in range(days):
            # Calculate the transmission rate
            if d < change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
            
            # Calculate the expected cases
            expected[d] = beta * I * S / population
            
            # Update the compartments
            S -= expected[d]
            E += expected[d]
            
            # Calculate new exposed cases
            new_exposed = E * sigma
            
            # Update the compartments
            E -= new_exposed
            I += new_exposed
            
            # Calculate new removed cases
            new_removed = I * gamma
            
            # Update the compartments
            I -= new_removed
            R += new_removed
        
        # Calculate the error
        error = np.sum((cases - expected) ** 2)
        
        return error
    
    # Set the initial parameters
    initial_params = [1.0, 1.0, 0]
    
    # Set the bounds for the parameters
    bounds = [(0.0, 10.0), (0.0, 10.0), (0, days)]
    
    # Run the optimization
    result = minimize(objective, initial_params, bounds=bounds)
    
    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Calculate the forecast
    forecast = np.zeros(horizon)
    
    # Calculate the forecast for each day
    for d in range(horizon):
        # Calculate the transmission rate
        if d < change_day:
            beta = R0 * gamma
        else:
            beta = R_post * gamma
        
        # Calculate the expected cases
        expected = beta * I * S / population
        
        # Update the compartments
        S -= expected
        E += expected
        
        # Calculate new exposed cases
        new_exposed = E * sigma
        
        # Update the compartments
        E -= new_exposed
        I += new_exposed
        
        # Calculate new removed cases
        new_removed = I * gamma
        
        # Update the compartments
        I -= new_removed
        R += new_removed
        
        # Store the forecast
        forecast[d] = expected
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }