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
    
    # Define the SEIR model
    def seir_model(beta, cases):
        # Calculate the time steps
        T = len(cases)
        
        # Initialize the state vector
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        
        # Store the results
        results = np.zeros((T, 4))
        results[0] = [S, E, I, R]
        
        # Calculate the time steps
        dt = 1.0
        
        # Calculate the beta values
        beta_values = np.zeros(T)
        beta_values[0] = beta
        
        # Calculate the cases
        for t in range(1, T):
            # Calculate the new cases
            new_cases = I * beta * dt
            
            # Update the state vector
            S -= new_cases
            E += new_cases
            I = E * np.exp(-1.0 / latent_period) * dt
            R = I * np.exp(-1.0 / infectious_period) * dt
            
            # Store the results
            results[t] = [S, E, I, R]
            
            # Update the beta values
            beta_values[t] = beta
        
        return results
    
    # Define the objective function
    def objective_function(beta):
        # Calculate the model results
        results = seir_model(beta, cases)
        
        # Calculate the error
        error = np.sum((results[:, 2] - cases) ** 2)
        
        return error
    
    # Optimize the beta value
    beta_opt = minimize(objective_function, 3.0, method="L-BFGS-B").x[0]
    
    # Calculate the model results
    results = seir_model(beta_opt, cases)
    
    # Calculate the forecast
    forecast = results[-horizon:][:, 2]
    
    # Return the results
    return {
        "R0": beta_opt,
        "R_post": beta_opt,
        "change_day": None,
        "forecast": forecast.tolist()
    }