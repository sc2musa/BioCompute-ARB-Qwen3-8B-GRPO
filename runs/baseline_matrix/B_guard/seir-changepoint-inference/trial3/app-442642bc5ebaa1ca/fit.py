import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert days to time units
    time_units = 1
    days = np.arange(len(cases))
    
    # Convert latent and infectious periods to time units
    latent_period_units = latent_period * time_units
    infectious_period_units = infectious_period * time_units
    
    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0
    
    # Define the SEIR model
    def seir_model(params):
        R0_est, R_post_est, change_day = params
        
        # Initialize state variables
        S = S0
        E = E0
        I = I0
        R = R0
        
        # Simulate the outbreak
        for day in range(len(cases)):
            # Calculate the number of new cases
            beta = R0_est * gamma if day < change_day else R_post_est * gamma
            new_cases = np.random.poisson(beta * I * S * time_units)
            
            # Update state variables
            S -= new_cases
            E += new_cases
            
            # Update infectious population
            if day >= infectious_period_units:
                R += I
                I = E - I
                E = 0
            
            # Update exposed population
            if day >= latent_period_units:
                I += E
                E = 0
            
        # Calculate the negative log-likelihood
        nll = -np.sum(poisson.logpmf(cases, new_cases))
        return nll
    
    # Define the gamma parameter
    gamma = 1 / infectious_period_units
    
    # Optimize the parameters
    result = minimize(seir_model, x0=[3.0, 3.0, 30], method='L-BFGS-B')
    R0_est, R_post_est, change_day_est = result.x
    
    # Detect change point using a simple method
    # This is a simplified version for demonstration purposes
    # In practice, we would use more sophisticated change point detection
    
    # Calculate the mean of the cases
    mean_cases = np.mean(cases)
    
    # Find the day where the mean changes significantly
    for day in range(len(cases)):
        if np.abs(np.mean(cases[:day+1]) - mean_cases) > 0.1 * mean_cases:
            change_day_est = day + 1
            break
    
    # Forecast future cases
    forecast = []
    for day in range(len(cases), len(cases) + horizon):
        # Calculate the expected number of new cases
        beta = R0_est * gamma if day < change_day_est else R_post_est * gamma
        expected_cases = beta * I * S * time_units
        forecast.append(expected_cases)
    
    return {
        "R0": R0_est,
        "R_post": R_post_est,
        "change_day": change_day_est,
        "forecast": forecast
    }