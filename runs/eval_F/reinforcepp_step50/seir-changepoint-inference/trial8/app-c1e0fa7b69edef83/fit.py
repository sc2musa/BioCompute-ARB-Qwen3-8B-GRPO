import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse inputs
    cases = np.array(cases)
    population = np.float64(population)
    initial_infectious = np.float64(initial_infectious)
    latent_period = np.float64(latent_period)
    infectious_period = np.float64(infectious_period)
    horizon = np.int64(horizon)
    
    # Initialize parameters
    R0 = 3.0
    R_post = 3.0
    change_day = None
    
    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        
        # Calculate the effective reproduction number
        effective_R = np.zeros_like(cases)
        effective_R[0] = R0
        for d in range(1, len(cases)):
            if change_day is not None and d == change_day:
                effective_R[d] = R_post
            else:
                effective_R[d] = R0
        
        # Calculate the number of infectious people
        infectious = np.zeros_like(cases)
        infectious[0] = initial_infectious
        for d in range(1, len(cases)):
            infectious[d] = infectious[d-1] * (1 - effective_R[d] / population)
        
        # Calculate the number of new cases
        new_cases = np.zeros_like(cases)
        for d in range(1, len(cases)):
            new_cases[d] = infectious[d-1] * effective_R[d] / population
        
        # Calculate the sum of new cases
        total_new_cases = np.sum(new_cases)
        
        # Return the total new cases
        return total_new_cases
    
    # Optimize the parameters
    result = minimize(seir_model, [R0, R_post, change_day], method="BFGS")
    
    # Extract the optimized parameters
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Forecast the next days
    forecast = np.zeros(horizon)
    infectious = np.zeros(horizon)
    for d in range(horizon):
        if change_day is not None and d == change_day:
            effective_R = R_post
        else:
            effective_R = R0
        
        # Calculate the number of new cases
        new_cases = infectious[-1] * effective_R / population
        
        # Update the infectious count
        infectious = np.append(infectious, new_cases)
        
        # Store the forecast
        forecast[d] = new_cases
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }