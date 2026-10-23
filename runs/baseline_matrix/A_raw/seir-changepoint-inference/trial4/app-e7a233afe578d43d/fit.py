import numpy as np
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert days to time units
    time_units = 1.0 / (latent_period + infectious_period)
    
    # Convert cases to daily new infections
    daily_new_infections = np.array(cases)
    
    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    
    # Define the SEIR model
    def seir_model(params, t):
        R0_val, R_post_val, change_day = params
        
        # Calculate the effective reproduction number
        if t < change_day:
            R = R0_val
        else:
            R = R_post_val
        
        # Calculate the transmission rate
        beta = R * infectious_period
        
        # Calculate the recovery rate
        gamma = 1.0 / infectious_period
        
        # Calculate the transition rates
        sigma = 1.0 / latent_period
        
        # Calculate the number of new infections
        new_infections = beta * I0 * S0 / population
        
        # Calculate the number of new exposed
        new_exposed = sigma * E0
        
        # Calculate the number of new infectious
        new_infectious = gamma * I0
        
        # Calculate the number of new removed
        new_removed = gamma * I0
        
        # Update the state variables
        S0 -= new_infections
        E0 += new_infections
        I0 += new_exposed
        R0 += new_infectious
        
        return new_infections
    
    # Define the objective function
    def objective_function(params):
        R0_val, R_post_val, change_day = params
        
        # Calculate the predicted cases
        predicted_cases = np.zeros_like(daily_new_infections)
        
        # Calculate the predicted cases for each day
        for t in range(len(daily_new_infections)):
            predicted_cases[t] = seir_model(params, t)
        
        # Calculate the error
        error = np.sum((daily_new_infections - predicted_cases) ** 2)
        
        return error
    
    # Initial guess for the parameters
    initial_guess = [3.0, 3.0, 50.0]
    
    # Optimize the parameters
    result = minimize(objective_function, initial_guess, method='L-BFGS-B')
    
    # Extract the optimized parameters
    R0_opt, R_post_opt, change_day_opt = result.x
    
    # Forecast the next horizon days
    forecast = np.zeros(horizon)
    
    # Calculate the forecast for each day
    for t in range(horizon):
        forecast[t] = seir_model((R0_opt, R_post_opt, change_day_opt), len(daily_new_infections) + t)
    
    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt if change_day_opt > 0 else None,
        "forecast": forecast.tolist()
    }