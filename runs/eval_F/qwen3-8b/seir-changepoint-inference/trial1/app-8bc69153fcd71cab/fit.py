import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}.
    """
    # Define parameters
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    
    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    
    # Time points
    t_span = (0, len(cases))
    t_eval = np.arange(0, len(cases) + 1)
    
    # Objective function
    def objective(params):
        R0, R_post, change_day = params
        
        # Simulate the SEIR model with the intervention
        def seir_model(t, y):
            S, E, I, R = y
            if t < change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        
        # Solve the ODE
        sol = solve_ivp(seir_model, t_span, [S0, E0, I0, R0], t_eval=t_eval, method='RK45')
        
        # Extract the predicted cases
        predicted_cases = sol.y[2][1:]  # Infectious cases
        
        # Calculate the sum of squared errors
        error = np.sum((np.array(cases) - predicted_cases) ** 2)
        
        return error
    
    # Initial guess for parameters
    initial_guess = [3.0, 3.0, 50.0]
    
    # Optimize the parameters
    result = minimize(objective, initial_guess, method='L-BFGS-B')
    
    # Extract the optimized parameters
    R0_opt, R_post_opt, change_day_opt = result.x
    
    # Forecast the next horizon days
    # Extend the time span to include the forecast period
    t_forecast = np.arange(len(cases), len(cases) + horizon + 1)
    
    # Simulate the SEIR model with the optimized parameters
    def seir_model_forecast(t, y):
        S, E, I, R = y
        if t < change_day_opt:
            beta = R0_opt * gamma
        else:
            beta = R_post_opt * gamma
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Solve the ODE for the forecast period
    sol_forecast = solve_ivp(seir_model_forecast, (len(cases), len(cases) + horizon), [S0, E0, I0, R0], t_eval=t_forecast, method='RK45')
    
    # Extract the forecasted cases
    forecast = sol_forecast.y[2][1:]  # Infectious cases
    
    # Return the results
    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt if change_day_opt > 0 else None,
        "forecast": forecast.tolist()
    }