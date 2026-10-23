import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import math

def seir_ode(t, y, beta, beta_post, change_day, sigma, gamma, population):
    S, E, I, R = y
    if t < change_day:
        beta_current = beta
    else:
        beta_current = beta_post
    dSdt = -beta_current * I * S / population
    dEdt = beta_current * I * S / population - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def objective(params, cases, population, initial_infectious, latent_period, infectious_period, change_day):
    beta, beta_post = params
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    sol = solve_ivp(seir_ode, (0, 70), [S0, E0, I0, R0], args=(beta, beta_post, change_day, sigma, gamma, population), method='RK45', t_eval=np.arange(0, 71))
    I_values = sol.y[2]
    lambda_d = np.zeros(70)
    for d in range(1, 71):
        lambda_d[d-1] = I_values[d] - I_values[d-1]
    log_likelihood = 0.0
    for d in range(70):
        observed = cases[d]
        expected = lambda_d[d]
        if expected < 0:
            expected = 0.0
        log_likelihood += observed * np.log(expected) - expected - np.log(math.factorial(observed))
    return -log_likelihood

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    best_log_likelihood = -np.inf
    best_params = None
    best_change_day = None
    for change_day in range(0, 71):
        if change_day == 0:
            def objective_single_beta(beta):
                return objective((beta, beta), cases, population, initial_infectious, latent_period, infectious_period, change_day)
            initial_guess = np.array([1.0])
            result = minimize(objective_single_beta, initial_guess, method='Nelder-Mead')
            if result.success:
                beta_opt, beta_post_opt = result.x[0], result.x[0]
                current_log_likelihood = -result.fun
            else:
                continue
        else:
            initial_guess = np.array([1.0, 0.5])
            result = minimize(objective, initial_guess, args=(cases, population, initial_infectious, latent_period, infectious_period, change_day), method='Nelder-Mead')
            if result.success:
                beta_opt, beta_post_opt = result.x
                current_log_likelihood = -result.fun
            else:
                continue
        if current_log_likelihood > best_log_likelihood:
            best_log_likelihood = current_log_likelihood
            best_params = (beta_opt, beta_post_opt)
            best_change_day = change_day
    beta_opt, beta_post_opt = best_params
    R0 = beta_opt * infectious_period
    R_post = beta_post_opt * infectious_period
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    sol = solve_ivp(seir_ode, (0, 70), [S0, E0, I0, R0], args=(beta_opt, beta_post_opt, best_change_day, sigma, gamma, population), method='RK45', t_eval=np.arange(0, 71))
    I_values = sol.y[2]
    forecast_days = horizon
    forecast_time_span = (70, 70 + forecast_days)
    state_at_70 = sol.y[:, -1]
    sol_forecast = solve_ivp(seir_ode, forecast_time_span, state_at_70, args=(beta_opt, beta_post_opt, best_change_day, sigma, gamma, population), method='RK45', t_eval=np.arange(70, 70 + forecast_days + 1))
    I_forecast = sol_forecast.y[2]
    forecast_cases = [I_forecast[i] - I_forecast[i-1] for i in range(1, forecast_days + 1)]
    change_day_result = best_change_day
    if change_day_result == 0:
        change_day_result = None
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day_result,
        "forecast": forecast_cases
    }