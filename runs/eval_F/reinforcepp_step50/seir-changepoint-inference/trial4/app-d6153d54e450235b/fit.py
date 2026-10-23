import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse input
    cases = np.array(cases)
    population = int(population)
    initial_infectious = float(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)
    
    # Initialize parameters
    R0 = 3.0
    R_post = 3.0
    change_day = None
    
    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        
        # Initialize the state
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        
        # Define the time steps
        t = np.arange(1, len(cases) + 1)
        
        # Simulate the SEIR model
        for d in range(1, len(cases) + 1):
            # Calculate the transmission rate
            if d <= change_day:
                beta = R0 * infectious_period
            else:
                beta = R_post * infectious_period
            
            # Calculate the new cases
            new_cases = beta * I * (1 - np.exp(-beta * latent_period))
            
            # Update the state
            E += new_cases
            I = I * np.exp(-beta * infectious_period)
            R += I
        
        return I
    
    # Define the objective function
    def objective(params):
        R0, R_post, change_day = params
        
        # Calculate the forecast
        forecast = np.zeros(horizon)
        
        # Simulate the SEIR model
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        
        # Define the time steps
        t = np.arange(1, len(cases) + 1)
        
        # Simulate the SEIR model
        for d in range(1, len(cases) + 1):
            # Calculate the transmission rate
            if d <= change_day:
                beta = R0 * infectious_period
            else:
                beta = R_post * infectious_period
            
            # Calculate the new cases
            new_cases = beta * I * (1 - np.exp(-beta * latent_period))
            
            # Update the state
            E += new_cases
            I = I * np.exp(-beta * infectious_period)
            R += I
        
        # Calculate the forecast
        for d in range(1, horizon + 1):
            # Calculate the transmission rate
            if d <= change_day:
                beta = R0 * infectious_period
            else:
                beta = R_post * infectious_period
            
            # Calculate the new cases
            new_cases = beta * I * (1 - np.exp(-beta * latent_period))
            
            # Update the state
            E += new_cases
            I = I * np.exp(-beta * infectious_period)
            R += I
            
            forecast[d-1] = new_cases
        
        # Calculate the objective function
        return np.sum((forecast - cases) ** 2)
    
    # Set the bounds for the parameters
    bounds = [(0.1, 10.0), (0.1, 10.0), (0, len(cases))] 
    
    # Optimize the parameters
    result = minimize(objective, [R0, R_post, change_day], bounds=bounds)
    
    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Make the forecast
    forecast = np.zeros(horizon)
    
    # Simulate the SEIR model
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    
    # Define the time steps
    t = np.arange(1, len(cases) + 1)
    
    # Simulate the SEIR model
    for d in range(1, len(cases) + 1):
        # Calculate the transmission rate
        if d <= change_day:
            beta = R0 * infectious_period
        else:
            beta = R_post * infectious_period
        
        # Calculate the new cases
        new_cases = beta * I * (1 - np.exp(-beta * latent_period))
        
        # Update the state
        E += new_cases
        I = I * np.exp(-beta * infectious_period)
        R += I
    
    # Calculate the forecast
    for d in range(1, horizon + 1):
        # Calculate the transmission rate
        if d <= change_day:
            beta = R0 * infectious_period
        else:
            beta = R_post * infectious_period
        
        # Calculate the new cases
        new_cases = beta * I * (1 - np.exp(-beta * latent_period))
        
        # Update the state
        E += new_cases
        I = I * np.exp(-beta * infectious_period)
        R += I
        
        forecast[d-1] = new_cases
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }