import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse inputs
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
        # Extract parameters
        R0, R_post, change_day = params
        
        # Initialize the state
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        
        # Calculate the time steps
        time_steps = len(cases)
        
        # Initialize the forecast
        forecast = np.zeros(horizon)
        
        # Simulate the model
        for t in range(time_steps):
            # Calculate the transmission rate
            if t < change_day:
                beta = R0 * infectious_period
            else:
                beta = R_post * infectious_period
            
            # Calculate the new cases
            new_cases = beta * I * (S / population)
            
            # Update the state
            S -= new_cases
            E += new_cases
            I = E * np.exp(-infectious_period * 1)
            R = I * np.exp(-infectious_period * 1)
            
            # Store the forecast
            forecast[t] = new_cases
        
        # Calculate the loss
        loss = np.sum((forecast - cases) ** 2)
        return loss
    
    # Optimize the parameters
    result = minimize(seir_model, [R0, R_post, change_day], method="BFGS")
    
    # Return the results
    return {
        "R0": result.x[0],
        "R_post": result.x[1],
        "change_day": result.x[2],
        "forecast": forecast
    }