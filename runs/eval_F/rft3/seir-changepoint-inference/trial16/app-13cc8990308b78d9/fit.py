import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse input
    cases = np.array(cases)
    population = np.array(population)
    initial_infectious = np.array(initial_infectious)
    latent_period = np.array(latent_period)
    infectious_period = np.array(infectious_period)
    horizon = np.array(horizon)
    
    # Define parameters
    R0 = 3.0
    R_post = 3.0
    change_day = 0.0  # Default value
    
    # Implement SEIR model
    def seir_model(params):
        # Extract parameters
        R0, R_post, change_day = params
        
        # Initialize state
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        
        # Calculate gamma and sigma
        gamma = 1.0 / infectious_period
        sigma = 1.0 / latent_period
        
        # Simulate SEIR dynamics
        for d in range(len(cases)):
            # Calculate transmission rate
            if d < change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
            
            # Calculate new cases
            new_cases = beta * S * I
            
            # Update state
            E += new_cases
            I += new_cases
            S -= new_cases
            
            # Update R
            R += I * gamma
            
        return R
    
    # Optimize parameters
    result = minimize(seir_model, [R0, R_post, change_day], method='L-BFGS-B', bounds=[(0, 10), (0, 10), (0, 100)])
    
    # Extract results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Forecast future cases
    forecast = []
    for d in range(len(cases), len(cases) + horizon):
        # Calculate transmission rate
        if d < change_day:
            beta = R0 * gamma
        else:
            beta = R_post * gamma
        
        # Calculate new cases
        new_cases = beta * S * I
        
        # Update state
        E += new_cases
        I += new_cases
        S -= new_cases
        
        # Update R
        R += I * gamma
        
        forecast.append(new_cases)
    
    return {"R0": R0, "R_post": R_post, "change_day": change_day, "forecast": forecast}