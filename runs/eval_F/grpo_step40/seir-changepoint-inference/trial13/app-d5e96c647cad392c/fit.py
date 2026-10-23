import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Convert to numpy arrays
    cases = np.array(cases)
    
    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        
        # Initial conditions
        S0 = population - initial_infectious
        E0 = 0
        I0 = initial_infectious
        R0 = 0
        
        # Time steps
        T = len(cases)
        
        # Create the matrix
        A = np.zeros((T, T))
        
        # Fill the matrix
        for t in range(T):
            if t < change_day:
                beta = R0 * infectious_period
            else:
                beta = R_post * infectious_period
            
            # Calculate the number of new cases
            new_cases = beta * I0 * (1 - np.exp(-beta * latent_period))
            
            # Update the matrix
            A[t, t] = new_cases
            
            # Update the state
            S0 = S0 - new_cases
            E0 = E0 + new_cases
            I0 = I0 - new_cases
            R0 = R0 + new_cases
        
        # Return the matrix
        return A
    
    # Define the objective function
    def objective(params):
        A = seir_model(params)
        
        # Calculate the error
        error = np.sum((cases - A) ** 2)
        
        # Return the error
        return error
    
    # Set the initial parameters
    initial_params = [3.0, 3.0, 0]
    
    # Optimize the parameters
    result = minimize(objective, initial_params, method="BFGS")
    
    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Calculate the forecast
    forecast = np.zeros(horizon)
    
    # Calculate the forecast for each day
    for t in range(horizon):
        if t < change_day:
            beta = R0 * infectious_period
        else:
            beta = R_post * infectious_period
        
        # Calculate the number of new cases
        new_cases = beta * I0 * (1 - np.exp(-beta * latent_period))
        
        # Update the forecast
        forecast[t] = new_cases
        
        # Update the state
        S0 = S0 - new_cases
        E0 = E0 + new_cases
        I0 = I0 - new_cases
        R0 = R0 + new_cases
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }