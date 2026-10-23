import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Convert days to time units
    days = np.array(cases)
    time = np.arange(len(days))
    
    # Initial values
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0
    
    # Define the SEIR model
    def seir_model(t, S, E, I, R):
        return np.array([
            -beta * I * S / population,
            beta * I * S / population - sigma * E,
            sigma * E - gamma * I,
            gamma * I
        ])
    
    # Define the objective function
    def objective(params):
        beta, gamma, change_day = params
        
        # Initialize the system
        S = S0
        E = E0
        I = I0
        R = R0
        
        # Track the results
        results = np.zeros((len(time), 4))
        results[0] = [S, E, I, R]
        
        # Simulate the system
        for t in range(1, len(time)):
            # Calculate the new values
            dS = seir_model(t, S, E, I, R)[0]
            dE = seir_model(t, S, E, I, R)[1]
            dI = seir_model(t, S, E, I, R)[2]
            dR = seir_model(t, S, E, I, R)[3]
            
            # Update the values
            S += dS
            E += dE
            I += dI
            R += dR
            
            # Store the results
            results[t] = [S, E, I, R]
        
        # Calculate the error
        error = np.sum((results[:, 2] - cases) ** 2)
        
        return error
    
    # Set initial guesses
    initial_guess = [1.0, 1.0, 0.0]
    
    # Optimize the parameters
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=[(0, 10), (0, 10), (0, len(time))])
    
    # Extract the results
    beta, gamma, change_day = result.x
    
    # Calculate R0 and R_post
    R0 = beta * gamma
    R_post = beta * gamma
    
    # Make forecasts
    forecast = np.zeros(horizon)
    
    # Simulate the system for the forecast period
    S = S0
    E = E0
    I = I0
    R = R0
    
    for t in range(len(time), len(time) + horizon):
        # Calculate the new values
        dS = seir_model(t, S, E, I, R)[0]
        dE = seir_model(t, S, E, I, R)[1]
        dI = seir_model(t, S, E, I, R)[2]
        dR = seir_model(t, S, E, I, R)[3]
        
        # Update the values
        S += dS
        E += dE
        I += dI
        R += dR
        
        # Store the forecast
        forecast[t - len(time)] = I
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }