import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Convert to numpy arrays
    cases = np.array(cases)
    
    # Initial state
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0
    
    # Calculate gamma and sigma
    gamma = 1 / infectious_period
    sigma = 1 / latent_period
    
    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Initial guess for R0
    R0_guess = 3.0
    
    # Optimize R0
    def objective(R0):
        beta = R0 * gamma
        sol = solve_ivp(seir_model, [0, 70], [S0, E0, I0, R0], t_eval=range(71), 
                         method='BDF', rtol=1e-8, atol=1e-12 * population)
        predicted_cases = sol.y[2][1:]  # Infectious cases
        return np.sum((predicted_cases - cases) ** 2) / np.sum(cases ** 2)
    
    # Optimize R_post
    def objective_post(R_post):
        beta = R_post * gamma
        sol = solve_ivp(seir_model, [0, 70], [S0, E0, I0, R0], t_eval=range(71), 
                         method='BDF', rtol=1e-8, atol=1e-12 * population)
        predicted_cases = sol.y[2][1:]  # Infectious cases
        return np.sum((predicted_cases - cases) ** 2) / np.sum(cases ** 2)
    
    # Find the change day
    def find_change_day(R0):
        beta = R0 * gamma
        sol = solve_ivp(seir_model, [0, 70], [S0, E0, I0, R0], t_eval=range(71), 
                         method='BDF', rtol=1e-8, atol=1e-12 * population)
        predicted_cases = sol.y[2][1:]  # Infectious cases
        
        # Find the day where the change occurs
        for day in range(70):
            if predicted_cases[day] < predicted_cases[day + 1] * 0.75:
                return day + 1
        return None
    
    # Calculate the forecast
    def forecast(R0):
        beta = R0 * gamma
        sol = solve_ivp(seir_model, [0, 70], [S0, E0, I0, R0], t_eval=range(71), 
                         method='BDF', rtol=1e-8, atol=1e-12 * population)
        predicted_cases = sol.y[2][1:]  # Infectious cases
        
        # Forecast for the next horizon days
        forecast = []
        for day in range(70, 70 + horizon):
            # Calculate the new cases for the next day
            new_cases = predicted_cases[-1] * (1 - np.exp(-beta * (1 / gamma)))
            forecast.append(new_cases)
        return forecast
    
    # Optimize R0
    R0_opt = minimize(objective, R0_guess, method='L-BFGS-B', bounds=[(0.1, 10)]).x[0]
    
    # Optimize R_post
    R_post_opt = minimize(objective_post, R0_opt, method='L-BFGS-B', bounds=[(0.1, 10)]).x[0]
    
    # Find the change day
    change_day_opt = find_change_day(R0_opt)
    
    # Calculate the forecast
    forecast_opt = forecast(R0_opt)
    
    # Return the results
    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt,
        "forecast": forecast_opt
    }