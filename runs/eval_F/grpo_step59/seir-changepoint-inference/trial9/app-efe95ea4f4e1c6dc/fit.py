import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse inputs
    cases = np.array(cases)
    population = int(population)
    initial_infectious = int(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)
    
    # Initialize variables
    S = population - initial_infectious
    E = 0
    I = initial_infectious
    R = 0
    
    # Calculate gamma and sigma
    gamma = 1 / infectious_period
    sigma = 1 / latent_period
    
    # Define the SEIR model
    def seir_model(beta):
        # Calculate the number of new cases
        new_cases = np.zeros_like(cases)
        for d in range(len(cases)):
            # Calculate the number of new cases
            new_cases[d] = beta * I * S / population
            
            # Update the variables
            S -= new_cases[d]
            E += new_cases[d]
            I += new_cases[d]
            R -= new_cases[d]
        
        return new_cases
    
    # Define the objective function
    def objective_function(beta):
        # Calculate the number of new cases
        new_cases = seir_model(beta)
        
        # Calculate the error
        error = np.sum((new_cases - cases) ** 2)
        
        return error
    
    # Optimize the parameters
    result = minimize(objective_function, [1.0], method="L-BFGS-B")
    
    # Extract the results
    R0 = result.x[0]
    
    # Calculate the forecast
    forecast = np.zeros(horizon)
    for d in range(horizon):
        # Calculate the number of new cases
        new_cases = R0 * gamma * S / population
        
        # Update the variables
        S -= new_cases
        E += new_cases
        I += new_cases
        R -= new_cases
        
        forecast[d] = new_cases
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R0,
        "change_day": None,
        "forecast": forecast.tolist()
    }