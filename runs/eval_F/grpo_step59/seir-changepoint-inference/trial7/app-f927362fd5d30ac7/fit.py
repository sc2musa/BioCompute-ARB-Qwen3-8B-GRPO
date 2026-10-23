import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Convert days to time units
    days = np.arange(len(cases))
    time = days / 365

    # Initial state
    S = population - initial_infectious
    E = 0
    I = initial_infectious
    R = 0

    # Define the SEIR model
    def seir_model(t, S, E, I, R):
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    # Define the objective function
    def objective(params):
        beta0, beta_post, change_day = params
        
        # Initialize the state
        S = population - initial_infectious
        E = 0
        I = initial_infectious
        R = 0
        
        # Store the results
        results = [I]  # Initial infectious cases
        
        # Run the simulation
        for t in time:
            if t < change_day:
                beta = beta0
            else:
                beta = beta_post
            
            # Calculate the derivatives
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            
            # Update the state
            S += dSdt * dt
            E += dEdt * dt
            I += dIdt * dt
            R += dRdt * dt
            
            # Store the results
            results.append(I)
        
        # Calculate the error
        error = np.sum((np.array(results) - np.array(cases)) ** 2)
        return error

    # Set the parameters
    sigma = 1 / latent_period
    gamma = 1 / infectious_period
    dt = 1 / 365
    
    # Initial guess
    initial_guess = [3.0, 3.0, 0]
    
    # Optimize the parameters
    result = minimize(objective, initial_guess, method="L-BFGS-B")
    
    # Extract the results
    beta0 = result.x[0]
    beta_post = result.x[1]
    change_day = result.x[2]
    
    # Calculate R0 and R_post
    R0 = beta0 * gamma
    R_post = beta_post * gamma
    
    # Forecast the future cases
    forecast = []
    for t in range(len(cases), len(cases) + horizon):
        if t < change_day:
            beta = beta0
        else:
            beta = beta_post
        
        # Calculate the derivatives
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        
        # Update the state
        S += dSdt * dt
        E += dEdt * dt
        I += dIdt * dt
        R += dRdt * dt
        
        # Store the results
        forecast.append(I)
    
    # Return the results
    return {"R0": R0, "R_post": R_post, "change_day": change_day, "forecast": forecast}