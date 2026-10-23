import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def seir_model(t, y, R0, R_post, change_day, gamma, sigma, population):
    S, E, I, R = y
    if change_day is not None and t >= change_day:
        beta = R_post * gamma
    else:
        beta = R0 * gamma
    dSdt = -beta * S * I / population
    dEdt = beta * S * I / population - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def simulate_model(cases, population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day):
    sigma = 1 / latent_period
    gamma = 1 / infectious_period
    y0 = [population - initial_infectious, 0, initial_infectious, 0]
    t_span = (0, 70)
    t_eval = np.arange(0, 70 + 1)
    sol = solve_ivp(lambda t, y: seir_model(t, y, R0, R_post, change_day, gamma, sigma, population), t_span, y0, t_eval=t_eval)
    predicted_cases = [sol.y[2][i] - sol.y[2][i-1] for i in range(1, 71)]
    return predicted_cases

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1 / latent_period
    gamma = 1 / infectious_period
    best_error = float('inf')
    best_R0 = best_R_post = best_change_day = None

    for change_day in [None] + list(range(1, 71)):
        if change_day is None:
            def objective_R0(R0):
                predicted = simulate_model(cases, population, initial_infectious, latent_period, infectious_period, R0, R0, change_day)
                return np.sum((np.array(predicted) - np.array(cases)) ** 2)
            res = minimize(objective_R0, x0=1.0, bounds=[(0.1, 10.0)])
            R0_opt, R_post_opt = res.x[0], res.x[0]
        else:
            def objective_params(params):
                R0, R_post = params
                predicted = simulate_model(cases, population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day)
                return np.sum((np.array(predicted) - np.array(cases)) ** 2)
            res = minimize(objective_params, x0=[1.0, 1.0], bounds=[(0.1, 10.0), (0.1, 1.0)])
            R0_opt, R_post_opt = res.x

        if res.fun < best_error:
            best_error = res.fun
            best_R0, best_R_post, best_change_day = R0_opt, R_post_opt, change_day

    # Forecast
    t_span_initial = (0, 70)
    t_eval_initial = np.arange(0, 70 + 1)
    sol_initial = solve_ivp(lambda t, y: seir_model(t, y, best_R0, best_R_post, best_change_day, gamma, sigma, population), t_span_initial, [population - initial_infectious, 0, initial_infectious, 0], t_eval=t_eval_initial)
    state_at_70 = sol_initial.y[:, -1]

    t_span_forecast = (70, 70 + horizon)
    t_eval_forecast = np.arange(70, 70 + horizon + 1)
    sol_forecast = solve_ivp(lambda t, y: seir_model(t, y, best_R0, best_R_post, best_change_day, gamma, sigma, population), t_span_forecast, state_at_70, t_eval=t_eval_forecast)
    forecast = [sol_forecast.y[2][i] - sol_forecast.y[2][i-1] for i in range(1, len(t_eval_forecast))]

    return {
        'R0': float(best_R0),
        'R_post': float(best_R_post),
        'change_day': best_change_day,
        'forecast': forecast
    }