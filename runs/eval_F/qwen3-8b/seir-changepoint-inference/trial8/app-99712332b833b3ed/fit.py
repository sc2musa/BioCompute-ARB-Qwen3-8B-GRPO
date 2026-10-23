import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert days to time units
    days = np.arange(1, len(cases) + 1)
    time_units = days / 365

    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0

    # SEIR model
    def seir_model(t, y, beta, gamma, sigma):
        S, E, I, R = y
        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    # Objective function
    def objective(params):
        beta, gamma, sigma, R_post, change_day = params
        # Simulate SEIR model
        sol = solve_ivp(seir_model, [0, time_units[-1]], [S0, E0, I0, R0], 
                        args=(beta, gamma, sigma), dense_output=True)
        # Compute expected cases
        expected_cases = np.zeros(len(cases))
        for i in range(len(cases)):
            t = time_units[i]
            y = sol.sol(t)
            expected_cases[i] = y[2]  # Number of new cases is the change in infectious population
        # Compute error
        error = np.sum((cases - expected_cases) ** 2)
        return error

    # Initial guess for parameters
    initial_guess = [3.0, 1.0 / infectious_period, 1.0 / latent_period, 3.0, 30.0]

    # Optimization
    result = minimize(objective, initial_guess, method='L-BFGS-B')
    beta_opt, gamma_opt, sigma_opt, R_post_opt, change_day_opt = result.x

    # Compute sol for forecast
    sol = solve_ivp(seir_model, [0, time_units[-1]], [S0, E0, I0, R0], 
                    args=(beta_opt, gamma_opt, sigma_opt), dense_output=True)

    # Forecast
    forecast = []
    for day in range(len(cases) + 1, len(cases) + horizon + 1):
        t = time_units[-1] + (day - len(cases)) / 365
        y = sol.sol(t)
        forecast.append(y[2])

    return {
        "R0": beta_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt,
        "forecast": forecast
    }