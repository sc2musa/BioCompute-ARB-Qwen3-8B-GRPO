import numpy as np
from scipy.optimize import curve_fit


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse parameters
    population = int(population)
    initial_infectious = float(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)
    
    # Initialize variables
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    
    # Calculate time steps
    time_steps = len(cases)
    
    # Define the SEIR model
    def seir_model(beta, t):
        # Calculate the rate of change
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - (1 / latent_period) * E
        dIdt = (1 / latent_period) * E - (1 / infectious_period) * I
        dRdt = (1 / infectious_period) * I
        
        # Update the variables
        S += dSdt
        E += dEdt
        I += dIdt
        R += dRdt
        
        # Return the new cases
        return I
    
    # Define the objective function
    def objective_function(beta):
        # Initialize the variables
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        
        # Calculate the total cases
        total_cases = 0.0
        
        # Iterate through the time steps
        for t in range(time_steps):
            # Calculate the new cases
            new_cases = seir_model(beta, t)
            
            # Update the variables
            S -= new_cases
            E += new_cases
            I += new_cases
            R += new_cases
            
            # Add to the total cases
            total_cases += new_cases
        
        # Return the total cases
        return total_cases
    
    # Set the initial guess for R0
    initial_guess = 3.0
    
    # Optimize the parameters
    result = curve_fit(objective_function, np.arange(time_steps), cases, p0=[initial_guess])
    
    # Extract the results
    R0 = result[0][0]
    
    # Calculate the forecast
    forecast = []
    
    # Iterate through the forecast horizon
    for t in range(horizon):
        # Calculate the new cases
        new_cases = seir_model(R0, t)
        
        # Add to the forecast
        forecast.append(new_cases)
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R0,
        "change_day": None,
        "forecast": forecast
    }