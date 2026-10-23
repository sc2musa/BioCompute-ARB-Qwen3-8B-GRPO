import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert cases to numpy array
    cases = np.array(cases)
    days = np.arange(len(cases))
    time_units = days / infectious_period

    # Initial parameters
    R0 = 2.0
    R_post = R0
    change_day = 0  # Default to day 0

    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        
        # Calculate the effective reproduction number
        beta = np.zeros(len(cases))
        beta[:change_day] = R0 * infectious_period
        beta[change_day:] = R_post * infectious_period

        # Initialize the SEIR compartments
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0

        # Simulate the SEIR dynamics
        model_cases = np.zeros(len(cases))
        for t in range(len(cases)):
            # Calculate the number of new infections
            new_infections = beta[t] * I / population * S
            
            # Update the compartments
            S -= new_infections
            E += new_infections
            
            # Calculate the number of new cases (Exposed -> Infectious)
            new_cases = E * (1 / latent_period)
            
            # Update the compartments
            I += new_cases
            E -= new_cases
            
            # Record the number of new cases
            model_cases[t] = new_cases

        # Calculate the forecast
        forecast = np.zeros(horizon)
        for t in range(horizon):
            # Calculate the number of new infections
            new_infections = beta[-1] * I / population * S
            
            # Update the compartments
            S -= new_infections
            E += new_infections
            
            # Calculate the number of new cases (Exposed -> Infectious)
            new_cases = E * (1 / latent_period)
            
            # Update the compartments
            I += new_cases
            E -= new_cases
            
            # Record the forecast
            forecast[t] = new_cases

        return model_cases

    # Optimize the parameters
    result = minimize(lambda params: np.sum((cases - seir_model(params)) ** 2), 
                      x0=[R0, R_post, change_day], 
                      bounds=[(0.1, 5.0), (0.1, 5.0), (0, len(cases))], 
                      method='L-BFGS-B')

    # Extract the optimized parameters
    R0_opt = result.x[0]
    R_post_opt = result.x[1]
    change_day_opt = result.x[2]

    # Forecast the next days
    forecast = seir_model([R0_opt, R_post_opt, change_day_opt])

    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt if change_day_opt > 0 else None,
        "forecast": forecast.tolist()
    }