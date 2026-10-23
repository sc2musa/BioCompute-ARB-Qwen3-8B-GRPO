import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert to numpy array
    cases = np.array(cases)
    days = len(cases)

    # Calculate gamma and sigma
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period

    # Initial guess for R0
    R0_guess = 2.0

    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        
        # Initialize the SEIR compartments
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        
        # Track the number of new cases
        predicted_cases = np.zeros(days)
        
        # Simulate the SEIR model
        for d in range(days):
            # Calculate the transmission rate
            if d < change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
            
            # Calculate the number of new infections
            new_infections = beta * I * S / population
            
            # Update the compartments
            S -= new_infections
            E += new_infections
            
            # Calculate the number of new cases (Exposed -> Infectious)
            new_cases = sigma * E
            
            # Update the compartments
            E -= new_cases
            I += new_cases
            
            # Track the predicted cases
            predicted_cases[d] = new_cases
        
        # Calculate the mean squared error
        mse = np.mean((cases - predicted_cases) ** 2)
        return mse

    # Grid search for change point detection
    best_mse = float('inf')
    best_params = (R0_guess, R0_guess, 0)
    
    # Try different change points
    for change_day in range(days):
        # Optimize R0 and R_post for this change point
        result = minimize(seir_model, x0=[R0_guess, R0_guess, change_day], method='L-BFGS-B', bounds=[(1.0, 10.0), (1.0, 10.0), (0, days)])
        R0, R_post, change_day_opt = result.x
        
        # Calculate MSE
        mse = seir_model((R0, R_post, change_day_opt))
        
        # Update best parameters
        if mse < best_mse:
            best_mse = mse
            best_params = (R0, R_post, change_day_opt)

    # Use the best parameters
    R0, R_post, change_day = best_params

    # Forecast the next days
    forecast = []
    
    # Calculate the final state of the SEIR model
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    
    # Simulate the SEIR model up to the end of the observed data
    for d in range(days):
        # Calculate the transmission rate
        if d < change_day:
            beta = R0 * gamma
        else:
            beta = R_post * gamma
        
        # Calculate the number of new infections
        new_infections = beta * I * S / population
        
        # Update the compartments
        S -= new_infections
        E += new_infections
        
        # Calculate the number of new cases (Exposed -> Infectious)
        new_cases = sigma * E
        
        # Update the compartments
        E -= new_cases
        I += new_cases
        
    
    # Simulate the SEIR model for the forecast period
    for _ in range(horizon):
        # Calculate the transmission rate
        beta = R_post * gamma
        
        # Calculate the number of new infections
        new_infections = beta * I * S / population
        
        # Update the compartments
        S -= new_infections
        E += new_infections
        
        # Calculate the number of new cases (Exposed -> Infectious)
        new_cases = sigma * E
        
        # Update the compartments
        E -= new_cases
        I += new_cases
        
        # Track the predicted cases
        forecast.append(new_cases)

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day if change_day > 0 else None,
        "forecast": forecast
    }