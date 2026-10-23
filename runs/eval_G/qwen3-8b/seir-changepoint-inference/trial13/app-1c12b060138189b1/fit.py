import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def seir_model(t, y, R0, alpha, change_day, gamma, sigma, population):
    S, E, I, R = y
    if t < change_day:
        beta = R0 * gamma
    else:
        beta = R0 * alpha * gamma
    dSdt = -beta * S * I / population
    dEdt = beta * S * I / population - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def objective(params, cases, population, initial_infectious, latent_period, infectious_period):
    R0, alpha, change_day = params
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0_ = 0.0
    y0 = [S0, E0, I0, R0_]
    # Time points
    t_span = (0.0, 70.0)
    t_eval = np.arange(0.0, 70.0 + 1e-6, 1.0)  # To include day 70
    # Solve the ODE
    sol = solve_ivp(lambda t, y: seir_model(t, y, R0, alpha, change_day, gamma, sigma, population),
                    t_span, y0, t_eval=t_eval, method='RK45')
    # Extract E(t) and I(t) at each time point
    E_values = sol.y[1]
    I_values = sol.y[2]
    # Compute expected cases for each day
    expected_cases = []
    for i in range(len(t_eval) - 1):
        # Trapezoidal rule for integral of sigma * E over [t_eval[i], t_eval[i+1]]
        # Time step is 1 day
        E_start = E_values[i]
        E_end = E_values[i+1]
        expected_cases.append( (E_start + E_end) / 2 * sigma * 1 )
    # Compute the error
    # Corrected Poisson negative log likelihood
    error = 0.0
    for observed, expected in zip(cases, expected_cases):
        if expected <= 0:
            # Avoid log(0)
            error += 1e3  # Reduced penalty for negative expected cases
        else:
            error += observed * np.log(expected) - expected
    return error

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Initial guess for R0, alpha, change_day
    initial_guess = [3.0, 0.5, 35.0]  # R0, alpha, change_day (initial guess for change_day closer to true values)
    # Bounds for R0, alpha, change_day
    bounds = [(0.1, 50.0), (0.0, 1.0), (0.0, 70.0)]  # Expanded R0 bounds
    # Constraints: R_post <= R0 (alpha <= 1.0)
    constraints = ({'type': 'ineq', 'fun': lambda x: x[0] - x[1]})  # x[0] - x[1] >= 0
    # Optimize
    result = minimize(objective, initial_guess, args=(cases, population, initial_infectious, latent_period, infectious_period), method='SLSQP', bounds=bounds, constraints=constraints)
    R0_opt, alpha_opt, change_day_opt = result.x
    # Compute R_post
    R_post_opt = R0_opt * alpha_opt
    # Check if change_day is within valid range
    if change_day_opt < 0 or change_day_opt > 70:
        change_day_opt = None
        R_post_opt = R0_opt
    # Now, compute the forecast for days 71 to 70 + horizon
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0_ = 0.0
    y0 = [S0, E0, I0, R0_]
    # Time points for forecast: 70 to 70 + horizon
    t_forecast = np.arange(70.0, 70.0 + horizon + 1e-6, 1.0)
    # Solve the ODE up to the forecast time
    t_span_forecast = (0.0, 70.0 + horizon)
    sol_forecast = solve_ivp(lambda t, y: seir_model(t, y, R0_opt, alpha_opt, change_day_opt, gamma, sigma, population),
                             t_span_forecast, y0, t_eval=t_forecast, method='RK45')
    # Extract E(t) and I(t) at the forecast days
    E_forecast = sol_forecast.y[1]
    I_forecast = sol_forecast.y[2]
    # Compute the expected cases for each forecast day
    forecast_cases = []
    for i in range(len(t_forecast) - 1):
        # Trapezoidal rule for integral of sigma * E over [t_forecast[i], t_forecast[i+1]]
        # Time step is 1 day
        E_start = E_forecast[i]
        E_end = E_forecast[i+1]
        forecast_cases.append( (E_start + E_end) / 2 * sigma * 1 )
    # Now, return the results
    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt,
        "forecast": forecast_cases
    }