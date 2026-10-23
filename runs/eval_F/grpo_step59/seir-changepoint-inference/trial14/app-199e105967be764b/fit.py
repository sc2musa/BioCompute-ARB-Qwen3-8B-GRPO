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
    
    # Define parameters
    R0 = 1.0
    R_post = 1.0
    change_day = None
    
    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        
        # Initialize the population
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        
        # Calculate the time steps
        time_steps = len(cases)
        
        # Calculate the beta values
        beta = np.zeros(time_steps)
        for t in range(time_steps):
            if t < change_day:
                beta[t] = R0 * infectious_period
            else:
                beta[t] = R_post * infectious_period
        
        # Calculate the gamma values
        gamma = 1.0 / infectious_period
        
        # Calculate the sigma values
        sigma = 1.0 / latent_period
        
        # Calculate the new cases
        new_cases = np.zeros(time_steps)
        for t in range(time_steps):
            # Calculate the number of new infections
            new_infections = beta[t] * S * I
            
            # Calculate the number of new cases
            new_cases[t] = new_infections * sigma
            
            # Update the population
            S -= new_infections
            E += new_infections
            I += new_infections * sigma
            R += new_infections * sigma
        
        # Return the new cases
        return new_cases
    
    # Define the objective function
    def objective_function(params):
        R0, R_post, change_day = params
        
        # Calculate the new cases
        new_cases = seir_model(params)
        
        # Calculate the error
        error = np.sum((new_cases - cases) ** 2)
        
        # Return the error
        return error
    
    # Set up the optimization
    initial_guess = [1.0, 1.0, 0]
    bounds = [(0.0, 100.0), (0.0, 100.0), (0, len(cases))]
    
    # Run the optimization
    result = minimize(objective_function, initial_guess, bounds=bounds)
    
    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Generate the forecast
    forecast = seir_model([R0, R_post, change_day])
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }