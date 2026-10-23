import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import warnings

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    y0 = [S0, E0, I0, R0]

    def seir_ode(t, y, R0_param, R_post_param, change_day, gamma_param, sigma_param):
        S, E, I, R = y
        beta = R0_param * gamma_param if t < change_day else R_post_param * gamma_param
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma_param * E
        dIdt = sigma_param * E - gamma_param * I
        dRdt = gamma_param * I
        return [dSdt, dEdt, dIdt, dRdt]

    def negative_log_likelihood(params):
        R0_param, R_post_param, change_day = params
        params_ode = (R0_param, R_post_param, change_day, gamma, sigma)
        sol = solve_ivp(seir_ode, (0, 70), y0, args=params_ode, t_eval=np.arange(0, 71), rtol=1e-6)
        E_values = sol.y[1]
        expected_cases = sigma * E_values[1:71]  # Corrected to use sigma * E for expected cases
        expected_cases = np.maximum(expected_cases, 1e-10)
        nll = -np.sum(cases * np.log(expected_cases) - expected_cases)
        return nll

    # Adjusted initial guess and bounds for better convergence
    initial_guess = [3.0, 1.0, 30.0]  # Lower R_post and change_day initial guess
    bounds = [(0.1, 10.0), (0.1, 5.0), (0.0, 70.0)]
    result = minimize(negative_log_likelihood, initial_guess, method='L-BFGS-B', bounds=bounds)
    R0_opt, R_post_opt, change_day_opt = result.x

    # Correct intervention detection logic
    if R_post_opt < 0.75 * R0_opt:
        # Intervention detected
        pass
    else:
        R_post_opt = R0_opt
        change_day_opt = None

    forecast_days = horizon
    if change_day_opt is None:
        params_ode_forecast = (R0_opt, R0_opt, 0.0, gamma, sigma)
    else:
        params_ode_forecast = (R0_opt, R_post_opt, change_day_opt, gamma, sigma)
    sol_forecast = solve_ivp(seir_ode, (0, 70 + forecast_days), y0, args=params_ode_forecast, t_eval=np.arange(0, 70 + forecast_days + 1), rtol=1e-6)
    E_forecast = sol_forecast.y[1]
    forecast_cases = sigma * E_forecast[71:71 + forecast_days]  # Corrected to use sigma * E for forecast

    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt,
        "forecast": forecast_cases.tolist()
    }