import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def seir_ode(t, y, R0, R_post, change_day, gamma, latent_period, infectious_period, population):
    S, E, I, R = y
    # Calculate beta based on time and change_day
    if t < change_day:
        beta = R0 * gamma
    else:
        beta = R_post * gamma
    # Compute the derivatives
    dSdt = -beta * S * I / population
    dEdt = beta * S * I / population - (1 / latent_period) * E
    dIdt = (1 / latent_period) * E - (1 / infectious_period) * I
    dRdt = (1 / infectious_period) * I
    return [dSdt, dEdt, dIdt, dRdt]

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Calculate gamma
    gamma = 1 / infectious_period
    # Initial guess for parameters
    initial_guess = [1.0, 1.0, 50.0]  # R0=1, R_post=1, change_day=50
    # Bounds for parameters
    bounds = [(0.1, 10.0), (0.1, 10.0), (1, 70)]
    # Optimization
    result = minimize(error_function, initial_guess, args=(cases, population, initial_infectious, latent_period, infectious_period, gamma), method='L-BFGS-B', bounds=bounds)
    # Extract the optimal parameters
    R0_opt = result.x[0]
    R_post_opt = result.x[1]
    change_day_opt = result.x[2]
    # Check if the intervention is present
    if abs(R_post_opt - R0_opt) < 1e-6:
        # No intervention
        change_day_opt = None
    else:
        # Intervention present
        change_day_opt = round(change_day_opt)
    # Forecast the next days
    # Simulate the model for the next horizon days
    y0 = [population - initial_infectious, 0, initial_infectious, 0]
    # Prepare args for solve_ivp
    if change_day_opt is not None:
        args = (R0_opt, R_post_opt, change_day_opt, gamma, latent_period, infectious_period, population)
    else:
        args = (R0_opt, R_post_opt, gamma, latent_period, infectious_period, population)
    sol = solve_ivp(seir_ode, [0, 70], y0, args=args, method='BDF', rtol=1e-8, atol=1e-12 * population)
    t = sol.t
    I_values = sol.y[2]
    I_interp = interp1d(t, I_values, kind='linear', fill_value='extrapolate')
    forecast_cases = np.zeros(horizon)
    for d in range(71, 70 + horizon + 1):
        I_end = I_interp(d)
        I_start = I_interp(d-1)
        forecast_cases[d-71] = I_end - I_start
    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt,
        "forecast": forecast_cases
    }

def error_function(params, cases, population, initial_infectious, latent_period, infectious_period, gamma):
    R0, R_post, change_day = params
    # Initial conditions: S = population - initial_infectious, E = 0, I = initial_infectious, R = 0
    y0 = [population - initial_infectious, 0, initial_infectious, 0]
    # Define the ODE system with the current parameters
    # Solve the ODE for the given time period
    sol = solve_ivp(seir_ode, [0, 70], y0, args=(R0, R_post, change_day, gamma, latent_period, infectious_period, population), method='BDF', rtol=1e-8, atol=1e-12 * population)
    # Extract the time points and I(t) values
    t = sol.t
    I_values = sol.y[2]
    # Create an interpolation function for I(t)
    I_interp = interp1d(t, I_values, kind='linear', fill_value='extrapolate')
    # Compute the predicted cases for each day
    predicted_cases = np.zeros(len(cases))
    for d in range(1, len(cases) + 1):
        # Compute the new cases for day d as the difference between I at the end of the day and the start of the day
        # The end of the day is at t=d, the start is at t=d-1
        # Evaluate I at t=d and t=d-1
        I_end = I_interp(d)
        I_start = I_interp(d-1)
        predicted_cases[d-1] = I_end - I_start
    # Compute the error between predicted and observed cases
    error = np.sum((predicted_cases - cases) ** 2)
    return error