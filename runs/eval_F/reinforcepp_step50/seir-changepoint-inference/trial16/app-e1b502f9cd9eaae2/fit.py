import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse input
    cases = np.array(cases)
    population = int(population)
    initial_infectious = int(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)
    
    # Initialize SEIR model
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0
    
    # Define the SEIR model
    def seir_model(t, S, E, I, R):
        # Calculate the rates
        beta = R0 * infectious_period
        gamma = 1 / infectious_period
        sigma = 1 / latent_period
        
        # Calculate the next state
        dS = -beta * S * I / population
        dE = beta * S * I / population - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        
        return [dS, dE, dI, dR]
    
    # Define the objective function
    def objective(params):
        R0, R_post, change_day = params
        
        # Set up the model with the intervention
        beta = R0 * infectious_period
        gamma = 1 / infectious_period
        sigma = 1 / latent_period
        
        # Initialize the model
        S = np.array([population - initial_infectious])
        E = np.array([0])
        I = np.array([initial_infectious])
        R = np.array([0])
        
        # Simulate the model
        for d in range(1, 71):
            # Calculate the next state
            dS = -beta * S[-1] * I[-1] / population
            dE = beta * S[-1] * I[-1] / population - sigma * E[-1]
            dI = sigma * E[-1] - gamma * I[-1]
            dR = gamma * I[-1]
            
            # Update the state
            S = np.append(S, S[-1] + dS)
            E = np.append(E, E[-1] + dE)
            I = np.append(I, I[-1] + dI)
            R = np.append(R, R[-1] + dR)
        
        # Calculate the forecast
        forecast = []
        for d in range(71, 71 + horizon):
            # Calculate the next state
            dS = -beta * S[-1] * I[-1] / population
            dE = beta * S[-1] * I[-1] / population - sigma * E[-1]
            dI = sigma * E[-1] - gamma * I[-1]
            dR = gamma * I[-1]
            
            # Update the state
            S = np.append(S, S[-1] + dS)
            E = np.append(E, E[-1] + dE)
            I = np.append(I, I[-1] + dI)
            R = np.append(R, R[-1] + dR)
            
            # Store the forecast
            forecast.append(I[-1] * population)
        
        # Calculate the error
        error = 0
        for d in range(71, 71 + horizon):
            error += np.abs(forecast[d-71] - cases[d-1])
        
        return error
    
    # Set up the optimization
    initial_guess = [3.0, 3.0, 0]
    bounds = [(0, 10), (0, 10), (0, 70)]
    
    # Run the optimization
    result = minimize(objective, initial_guess, bounds=bounds, method='COBYLA')
    
    # Return the results
    return {
        "R0": result.x[0],
        "R_post": result.x[1],
        "change_day": result.x[2],
        "forecast": result.x[3]
    }