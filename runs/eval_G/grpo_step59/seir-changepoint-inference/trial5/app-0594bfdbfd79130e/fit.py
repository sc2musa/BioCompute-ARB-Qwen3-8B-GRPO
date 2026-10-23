import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Parse cases into an array
    cases_array = np.array(cases)

    # Compute sigma and gamma
    sigma = 1 / latent_period
    gamma = 1 / infectious_period

    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0
    y0 = [S0, E0, I0, R0]

    # Define the SEIR model function
    def seir_model(t, y, R0, R_post, change_day):
        S, E, I, R = y
        beta = R0 * gamma if t < change_day else R_post * gamma
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    # Define the error function
    def error_function(params):
        R0, R_post, change_day = params
        # Solve the ODE
        sol = solve_ivp(seir_model, [0, 70], y0, t_eval=np.arange(0, 70, 1), method='BDF', args=(R0, R_post, change_day))
        # Extract the predicted cases
        predicted_cases = sol.y[2][1:]  # The third compartment is I, and the cases are transitions from E to I
        # Compute the error
        error = np.sum((predicted_cases - cases_array) ** 2)
        return error

    # Set up the optimization
    bounds = [
        (0, 10),  # R0
        (0, 10),  # R_post
        (0, 70)   # change_day
    ]
    # Initial guess for the parameters
    initial_guess = [1.0, 1.0, 0.0]

    # Run the optimization
    result = minimize(error_function, initial_guess, bounds=bounds, method='L-BFGS-B')

    # Extract the estimated parameters
    R0_est = result.x[0]
    R_post_est = result.x[1]
    change_day_est = result.x[2]

    # Now, simulate the model with the estimated parameters to get the forecast
    # The forecast is for days 71 to 70 + horizon
    # So, simulate from day 70 to day 70 + horizon
    # The initial conditions for the forecast are the values at day 70
    # Let's get the values at day 70
    sol = solve_ivp(seir_model, [0, 70], y0, t_eval=np.arange(0, 70, 1), method='BDF', args=(R0_est, R_post_est, change_day_est))
    I_at_70 = sol.y[2][-1]
    E_at_70 = sol.y[1][-1]
    S_at_70 = sol.y[0][-1]
    R_at_70 = sol.y[3][-1]

    # Now, simulate from day 70 to day 70 + horizon
    # The initial conditions are S_at_70, E_at_70, I_at_70, R_at_70
    # The time span is from 70 to 70 + horizon
    # The forecast is the number of new cases (E to I) for each day in this span
    # However, the model is deterministic, so we can simulate it
    # The forecast will be the predicted cases for each day in the horizon

    # Define the forecast model
    def forecast_model(t, y):
        S, E, I, R = y
        beta = R0_est * gamma if t < change_day_est else R_post_est * gamma
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    # Solve the ODE for the forecast
    forecast_sol = solve_ivp(forecast_model, [70, 70 + horizon], [S_at_70, E_at_70, I_at_70, R_at_70], t_eval=np.arange(70, 70 + horizon, 1), method='BDF')
    # Extract the predicted cases for the forecast period
    forecast_cases = forecast_sol.y[2][1:]  # The third compartment is I, and the cases are transitions from E to I

    # Return the results
    return {
        "R0": R0_est,
        "R_post": R_post_est,
        "change_day": change_day_est,
        "forecast": forecast_cases.tolist()
    }