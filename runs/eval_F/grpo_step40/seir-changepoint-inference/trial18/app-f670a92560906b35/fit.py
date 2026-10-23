import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse input
    cases = np.array(cases)
    population = int(population)
    initial_infectious = float(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)
    
    # Initial state
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    
    # Time parameters
    days = len(cases)
    time = np.arange(days)
    
    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        
        # Calculate transmission rates
        beta = np.zeros(days)
        beta[0:change_day] = R0 * infectious_period
        beta[change_day:] = R_post * infectious_period
        
        # Solve the SEIR equations
        S = np.zeros(days)
        E = np.zeros(days)
        I = np.zeros(days)
        R = np.zeros(days)
        
        S[0] = population - initial_infectious
        E[0] = 0.0
        I[0] = initial_infectious
        R[0] = 0.0
        
        for t in range(1, days):
            # Calculate new infections
            new_infections = beta[t-1] * I[t-1] * (S[t-1] / population)
            
            # Update compartments
            S[t] = S[t-1] - new_infections
            E[t] = E[t-1] + new_infections
            I[t] = I[t-1] + E[t-1] * np.exp(-latent_period)
            R[t] = R[t-1] + I[t-1] * np.exp(-infectious_period)
        
        # Calculate forecast
        forecast = np.zeros(horizon)
        for t in range(days, days + horizon):
            # Calculate new infections
            new_infections = beta[t-1] * I[t-1] * (S[t-1] / population)
            
            # Update compartments
            S[t] = S[t-1] - new_infections
            E[t] = E[t-1] + new_infections
            I[t] = I[t-1] + E[t-1] * np.exp(-latent_period)
            R[t] = R[t-1] + I[t-1] * np.exp(-infectious_period)
            
            forecast[t - days] = new_infections
        
        return forecast
    
    # Define the objective function
    def objective(params):
        R0, R_post, change_day = params
        
        # Calculate the forecast
        forecast = seir_model(params)
        
        # Calculate the error
        error = np.sum((forecast - cases) ** 2)
        
        return error
    
    # Initial guess
    initial_guess = [3.0, 3.0, 0]
    
    # Optimize the parameters
    result = minimize(objective, initial_guess, method="L-BFGS-B",
                      bounds=[(0.0, 10.0), (0.0, 10.0), (0, days)])
    
    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Make the forecast
    forecast = seir_model(result.x)
    
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }