import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse input
    cases = np.array(cases)
    days = len(cases)
    
    # Initial values
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0
    
    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Define the objective function
    def objective(params):
        beta, sigma, gamma, change_day = params
        
        # Initial values
        S = population - initial_infectious
        E = 0
        I = initial_infectious
        R = 0
        
        # Store results
        results = np.zeros((days, 4))
        results[0] = [S, E, I, R]
        
        # Simulate the outbreak
        for d in range(1, days):
            # Calculate the new cases
            new_cases = np.random.poisson(I * gamma * population)
            
            # Update the state
            S = S - new_cases
            E = E + new_cases
            I = I - new_cases
            R = R + new_cases
            
            # Store results
            results[d] = [S, E, I, R]
        
        # Calculate the error
        error = np.sum((cases - results[:, 2]) ** 2)
        
        return error
    
    # Set initial parameters
    initial_params = [1.0, 1.0, 1.0, 0.0]
    
    # Optimize the parameters
    result = minimize(objective, initial_params, method="L-BFGS-B",
                      bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, days)])
    
    # Extract the results
    beta, sigma, gamma, change_day = result.x
    
    # Calculate R0 and R_post
    R0 = beta * gamma
    R_post = beta * gamma
    
    # Calculate the forecast
    forecast = []
    for d in range(days, days + horizon):
        # Calculate the new cases
        new_cases = np.random.poisson(I * gamma * population)
        
        # Update the state
        S = S - new_cases
        E = E + new_cases
        I = I - new_cases
        R = R + new_cases
        
        # Store results
        forecast.append(new_cases)
    
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast
    }