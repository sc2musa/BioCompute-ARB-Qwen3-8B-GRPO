import numpy as np
from scipy.optimize import least_squares


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse inputs
    cases = np.array(cases)
    population = int(population)
    initial_infectious = float(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)
    
    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        
        # Initialize the SEIR compartments
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        
        # Calculate the time steps
        days = len(cases)
        
        # Initialize the forecast
        forecast = np.zeros(horizon)
        
        # Simulate the outbreak
        for d in range(days):
            # Calculate the transmission rate
            if d < change_day:
                beta = R0 * infectious_period
            else:
                beta = R_post * infectious_period
            
            # Calculate the new cases
            new_cases = beta * I * S / population
            
            # Update the compartments
            S -= new_cases
            E += new_cases
            I += E / latent_period
            R += I / infectious_period
            
            # Store the forecast
            if d >= days - horizon:
                forecast[d - days + horizon] = new_cases
        
        return forecast
    
    # Define the objective function
    def objective(params):
        forecast = seir_model(params)
        return forecast - cases
    
    # Set initial parameters
    initial_params = [3.0, 3.0, 0.0]
    
    # Set bounds for parameters
    bounds = [(0.1, 10.0), (0.1, 10.0), (0.0, 70.0)]
    
    # Optimize the parameters
    result = least_squares(objective, initial_params, bounds=bounds)
    
    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Return the results
    return {"R0": R0, "R_post": R_post, "change_day": change_day, "forecast": seir_model([R0, R_post, change_day])}
