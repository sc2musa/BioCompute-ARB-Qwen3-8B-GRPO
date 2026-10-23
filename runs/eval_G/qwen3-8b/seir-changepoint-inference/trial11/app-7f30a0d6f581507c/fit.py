import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    
    # Function to simulate SEIR model
    def seir_model(y, t, R0, R_post, change_day):
        if change_day is None:
            beta = R0 * gamma
        else:
            beta = R0 * gamma if t < change_day else R_post * gamma
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Function to compute predicted cases
    def predict_cases(R0, R_post, change_day):
        # Initial conditions
        S0 = population - initial_infectious
        E0 = 0.0
        I0 = initial_infectious
        R0_ = 0.0
        y0 = np.array([S0, E0, I0, R0_])
        
        # Time points
        t = np.arange(0, 71)  # days 0 to 70
        
        # Solve ODE
        y = odeint(seir_model, y0, t, args=(R0, R_post, change_day))
        
        # Extract E values
        E = y[:, 1]
        
        # Predicted cases
        predicted = E[1:] * sigma  # E[0] is day 0, E[1:] are days 1-70
        return predicted
    
    # Try all possible change days
    best_error = float('inf')
    best_R0 = best_R_post = None
    best_change_day = None
    
    # Try no intervention
    def objective_no_intervention(R0):
        predicted = predict_cases(R0, R0, None)
        cases_array = np.array(cases).flatten()
        return np.sum((predicted - cases_array) ** 2)
    
    result_no_intervention = minimize(objective_no_intervention, x0=1.0)
    R0_no_intervention = result_no_intervention.x[0]
    error_no_intervention = result_no_intervention.fun
    
    # Try all change days
    for change_day in range(1, 71):
        # Split data
        cases_before = cases[:change_day]
        cases_after = cases[change_day:]
        
        # Optimize R0 for before period
        def objective_R0(R0):
            predicted_before = predict_cases(R0, 1.0, change_day)
            cases_before_array = np.array(cases_before).flatten()
            return np.sum((predicted_before - cases_before_array) ** 2)
        
        result_R0 = minimize(objective_R0, x0=1.0)
        R0_opt = result_R0.x[0]
        
        # Optimize R_post for after period
        def objective_R_post(R_post):
            predicted_after = predict_cases(R0_opt, R_post, change_day)
            cases_after_array = np.array(cases_after).flatten()
            return np.sum((predicted_after - cases_after_array) ** 2)
        
        result_R_post = minimize(objective_R_post, x0=1.0)
        R_post_opt = result_R_post.x[0]
        
        # Total error
        total_error = result_R0.fun + result_R_post.fun
        
        if total_error < best_error:
            best_error = total_error
            best_R0 = R0_opt
            best_R_post = R_post_opt
            best_change_day = change_day
    
    # Compare with no intervention
    if error_no_intervention < best_error:
        best_R0 = R0_no_intervention
        best_R_post = best_R0
        best_change_day = None
    
    # Forecast
    # Simulate up to day 70
    y = odeint(seir_model, [population - initial_infectious, 0.0, initial_infectious, 0.0], np.arange(0, 71), args=(best_R0, best_R_post, best_change_day))
    E = y[:, 1]
    
    # Forecast from day 70 to 70 + horizon
    t_forecast = np.arange(70, 70 + horizon + 1)
    y_forecast = odeint(seir_model, y[-1], t_forecast, args=(best_R0, best_R_post, best_change_day))
    E_forecast = y_forecast[:, 1]
    
    forecast = E_forecast[1:] * sigma
    
    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": forecast.tolist()
    }