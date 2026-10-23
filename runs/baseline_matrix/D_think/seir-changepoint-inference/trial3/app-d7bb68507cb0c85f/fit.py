import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    
    # Function to simulate SEIR model and compute expected cases
    def simulate(R0, R_post, change_day):
        # Define the SEIR ODE system
        def seir_ode(t, y):
            S, E, I, R = y
            beta = R0 * gamma if t < change_day else R_post * gamma
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        
        # Initial conditions
        S0 = population - initial_infectious
        E0 = 0.0
        I0 = initial_infectious
        R0_ = 0.0
        y0 = [S0, E0, I0, R0_]
        
        # Solve ODEs
        t_span = (0, 70)
        t_eval = np.arange(0, 70 + 1)
        sol = solve_ivp(seir_ode, t_span, y0, t_eval=t_eval, method='RK45')
        
        # Extract I(t) values
        I_values = sol.y[2]
        
        # Compute expected cases for each day
        expected_cases = []
        for d in range(1, 71):
            I_prev = I_values[d-1]
            I_curr = I_values[d]
            expected_cases.append(I_curr - I_prev)
        
        return expected_cases, sol.y
    
    # Function to compute error for a given R0 and R_post
    def error(R0, R_post, change_day):
        expected, sol = simulate(R0, R_post, change_day)
        return np.sum((np.array(expected) - np.array(cases)) ** 2)
    
    # Try all possible change_days
    best_error = float('inf')
    best_R0 = best_R_post = best_change_day = None
    
    # Case 1: No intervention (change_day is None)
    def error_no_intervention(R0):
        expected, _ = simulate(R0, R0, None)
        return np.sum((np.array(expected) - np.array(cases)) ** 2)
    
    result = minimize_scalar(error_no_intervention, bounds=(0.1, 10.0), method='bounded')
    if result.success:
        best_error = result.fun
        best_R0 = result.x
        best_R_post = best_R0
        best_change_day = None
    
    # Case 2: Intervention at some day
    for change_day in range(1, 71):
        # Optimize R0 for the first part
        def error_first_part(R0):
            expected_first, _ = simulate(R0, R_post, change_day)
            return np.sum((np.array(expected_first[:change_day]) - np.array(cases[:change_day])) ** 2)
        
        # Optimize R_post for the second part
        def error_second_part(R_post):
            expected_second, _ = simulate(R0, R_post, change_day)
            return np.sum((np.array(expected_second[change_day:]) - np.array(cases[change_day:])) ** 2)
        
        # Optimize R0 for the first part
        result_R0 = minimize_scalar(error_first_part, bounds=(0.1, 10.0), method='bounded')
        if not result_R0.success:
            continue
        
        # Optimize R_post for the second part
        result_R_post = minimize_scalar(error_second_part, bounds=(0.1, result_R0.x), method='bounded')
        if not result_R_post.success:
            continue
        
        # Compute overall error
        expected, sol = simulate(result_R0.x, result_R_post.x, change_day)
        current_error = np.sum((np.array(expected) - np.array(cases)) ** 2)
        
        if current_error < best_error:
            best_error = current_error
            best_R0 = result_R0.x
            best_R_post = result_R_post.x
            best_change_day = change_day
    
    # Forecast the next horizon days
    def forecast(R0, R_post, change_day):
        # Run full simulation to get state at day 70
        expected, sol = simulate(R0, R_post, change_day)
        
        # Get state at day 70
        day_70_state = sol.y[:, 70]
        
        # Simulate from day 70 to day 70 + horizon
        t_span = (70, 70 + horizon)
        t_eval = np.arange(70, 70 + horizon + 1)
        
        # Define ODE function with current parameters
        def forecast_ode(t, y):
            S, E, I, R = y
            beta = R0 * gamma if t < change_day else R_post * gamma
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        
        sol_forecast = solve_ivp(forecast_ode, t_span, day_70_state, t_eval=t_eval, method='RK45')
        
        # Extract I(t) values for forecast days
        I_values_forecast = sol_forecast.y[2]
        
        # Compute expected cases for the next horizon days
        forecast_cases = []
        for d in range(70 + 1, 70 + horizon + 1):
            I_prev = I_values_forecast[d - 70 - 1]
            I_curr = I_values_forecast[d - 70]
            forecast_cases.append(I_curr - I_prev)
        
        return forecast_cases
    
    forecast_cases = forecast(best_R0, best_R_post, best_change_day)
    
    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": forecast_cases
    }
