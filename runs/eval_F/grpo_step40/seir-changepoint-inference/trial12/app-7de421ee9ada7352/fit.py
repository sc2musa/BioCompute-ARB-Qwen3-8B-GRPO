import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Convert to numpy arrays
    cases = np.array(cases)
    population = np.array([population])
    initial_infectious = np.array([initial_infectious])
    latent_period = np.array([latent_period])
    infectious_period = np.array([infectious_period])
    horizon = np.array([horizon])
    
    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        
        # Initialize the state
        S = population - initial_infectious
        E = np.zeros_like(S)
        I = initial_infectious
        R = np.zeros_like(S)
        
        # Calculate the time steps
        T = len(cases)
        
        # Create the beta function
        def beta(t):
            if t < change_day:
                return R0 * infectious_period[0]
            else:
                return R_post * infectious_period[0]
        
        # Simulate the SEIR model
        for t in range(T):
            # Calculate the new cases
            new_cases = beta(t) * I[t] * S[t] / population[0]
            
            # Update the state
            E[t+1] = E[t] + new_cases * latent_period[0]
            I[t+1] = I[t] + E[t+1] - R[t+1]
            R[t+1] = R[t] + I[t+1]
            
            # Update the cases
            cases[t] = new_cases
        
        # Return the final state
        return I[-1]
    
    # Define the objective function
    def objective(params):
        return np.sum((cases - seir_model(params)) ** 2)
    
    # Set initial guess
    initial_guess = [3.0, 3.0, 0]
    
    # Optimize the parameters
    result = minimize(objective, initial_guess, method="L-BFGS-B")
    
    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Forecast the next days
    forecast = np.zeros(horizon[0])
    
    # Calculate the time steps
    T = len(cases)
    
    # Create the beta function
    def beta(t):
        if t < change_day:
            return R0 * infectious_period[0]
        else:
            return R_post * infectious_period[0]
    
    # Simulate the SEIR model
    for t in range(T):
        # Calculate the new cases
        new_cases = beta(t) * I[t] * S[t] / population[0]
        
        # Update the state
        E[t+1] = E[t] + new_cases * latent_period[0]
        I[t+1] = I[t] + E[t+1] - R[t+1]
        R[t+1] = R[t] + I[t+1]
        
        # Update the cases
        cases[t] = new_cases
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast
    }