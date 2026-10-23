import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse input
    cases = np.array(cases)
    population = int(population)
    initial_infectious = int(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)
    
    # Initialize parameters
    R0 = 3.0
    R_post = 3.0
    change_day = None
    
    # Define the SEIR model
    def seir_model(t, S, E, I, R):
        # Calculate the number of new cases
        if change_day is not None and t >= change_day:
            new_cases = I * R_post * np.exp(-t * latent_period)
        else:
            new_cases = I * R0 * np.exp(-t * latent_period)
        
        # Update the compartments
        dS = -new_cases * S / population
        dE = new_cases * S / population - E / latent_period
        dI = E / latent_period - I / infectious_period
        dR = I / infectious_period
        
        return [dS, dE, dI, dR]
    
    # Define the objective function
    def objective(params):
        # Extract parameters
        R0, R_post, change_day = params
        
        # Convert change_day to float if it's None
        change_day = float(change_day) if change_day is not None else 0.0
        
        # Ensure change_day is within the valid range of days
        if change_day < 0 or change_day > len(cases):
            change_day = 0.0
        
        # Initialize the compartments
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        
        # Calculate the total cases
        total_cases = 0.0
        
        # Simulate the model
        for t in range(len(cases)):
            # Calculate the new cases
            if change_day is not None and t >= change_day:
                new_cases = I * R_post * np.exp(-t * latent_period)
            else:
                new_cases = I * R0 * np.exp(-t * latent_period)
            
            # Update the compartments
            dS = -new_cases * S / population
            dE = new_cases * S / population - E / latent_period
            dI = E / latent_period - I / infectious_period
            dR = I / infectious_period
            
            # Update the compartments
            S += dS
            E += dE
            I += dI
            R += dR
            
            # Add the new cases to the total
            total_cases += new_cases
        
        # Calculate the error
        error = np.sum((cases - total_cases) ** 2)
        
        return error
    
    # Set the initial guess
    initial_guess = [3.0, 3.0, None]
    
    # Optimize the parameters
    result = minimize(objective, initial_guess, method="L-BFGS-B",
                     bounds=[(0.1, 10.0), (0.1, 10.0), (0, len(cases))])
    
    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Forecast the future cases
    forecast = []
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    
    for t in range(len(cases), len(cases) + horizon):
        # Calculate the new cases
        if change_day is not None and t >= change_day:
            new_cases = I * R_post * np.exp(-t * latent_period)
        else:
            new_cases = I * R0 * np.exp(-t * latent_period)
        
        # Update the compartments
        dS = -new_cases * S / population
        dE = new_cases * S / population - E / latent_period
        dI = E / latent_period - I / infectious_period
        dR = I / infectious_period
        
        # Update the compartments
        S += dS
        E += dE
        I += dI
        R += dR
        
        # Add the new cases to the forecast
        forecast.append(new_cases)
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast
    }