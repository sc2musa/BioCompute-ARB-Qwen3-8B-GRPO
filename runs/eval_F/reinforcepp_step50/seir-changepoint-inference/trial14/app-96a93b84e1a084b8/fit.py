import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse inputs
    cases = np.array(cases)
    population = float(population)
    initial_infectious = float(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)
    
    # Time parameters
    days = len(cases)
    dt = 1.0
    
    # Define the SEIR model
    def seir_model(t, S, E, I, R):
        # Calculate the transmission rate
        if t < change_day:
            beta = R0 * gamma
        else:
            beta = R_post * gamma
        
        # Calculate the derivatives
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dI_dt = sigma * E - gamma * I
        dRdt = gamma * I
        
        return [dSdt, dEdt, dI_dt, dRdt]
    
    # Define the objective function
    def objective(params):
        R0, R_post, change_day = params
        gamma = 1.0 / infectious_period
        sigma = 1.0 / latent_period
        
        # Initialize the state
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        
        # Simulate the SEIR model
        for t in range(days):
            # Calculate the transmission rate
            if t < change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
            
            # Calculate the derivatives
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dI_dt = sigma * E - gamma * I
            dRdt = gamma * I
            
            # Update the state
            S += dSdt * dt
            E += dEdt * dt
            I += dI_dt * dt
            R += dRdt * dt
            
        # Calculate the log likelihood
        log_likelihood = 0.0
        for t in range(days):
            # Calculate the expected cases
            expected_cases = I * dt
            
            # Calculate the log likelihood
            log_likelihood += np.log(expected_cases) if expected_cases > 0 else 0
            
        return -log_likelihood
    
    # Set initial guesses
    initial_guess = [3.0, 3.0, 0.0]
    
    # Optimize the parameters
    result = minimize(objective, initial_guess, method="L-BFGS-B",
                     bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, days)])
    
    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Forecast the next days
    forecast = []
    for t in range(days, days + horizon):
        # Calculate the transmission rate
        if t < change_day:
            beta = R0 * gamma
        else:
            beta = R_post * gamma
        
        # Calculate the expected cases
        expected_cases = I * dt
        
        forecast.append(expected_cases)
        
    return {"R0": R0, "R_post": R_post, "change_day": change_day, "forecast": forecast}