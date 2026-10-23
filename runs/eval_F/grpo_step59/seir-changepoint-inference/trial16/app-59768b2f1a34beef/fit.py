import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Convert to numpy arrays
    cases = np.array(cases)
    population = np.array([population])
    initial_infectious = np.array([initial_infectious])
    
    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        
        # Initialize the SEIR compartments
        S = population - initial_infectious
        E = np.zeros_like(S)
        I = initial_infectious
        R = np.zeros_like(S)
        
        # Time steps
        T = len(cases)
        
        # Calculate the transmission rates
        beta = np.zeros(T)
        beta[0] = R0 * infectious_period
        
        # If there's a change day, update the transmission rate
        if change_day is not None:
            beta[change_day:] = R_post * infectious_period
        
        # Simulate the SEIR model
        for t in range(1, T):
            # Calculate the new cases
            new_cases = I[t-1] * beta[t-1] * (S[t-1] / population)
            
            # Update the compartments
            E[t] = E[t-1] + new_cases * (1 / latent_period)
            I[t] = I[t-1] + E[t-1] - R[t-1] * (1 / infectious_period)
            S[t] = S[t-1] - new_cases
            R[t] = R[t-1] + I[t-1] * (1 / infectious_period)
        
        # Return the predicted cases
        return I[1:] / population
    
    # Define the objective function
    def objective(params):
        predicted = seir_model(params)
        return np.sum((predicted - cases) ** 2)
    
    # Set initial parameters
    initial_params = [3.0, 3.0, 0]
    
    # Optimize the parameters
    result = minimize(objective, initial_params, method="L-BFGS-B",
                     bounds=[(0.1, 10), (0.1, 10), (0, T-1)])
    
    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Forecast the future cases
    forecast = seir_model([R0, R_post, change_day])
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }