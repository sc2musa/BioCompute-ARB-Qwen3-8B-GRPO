import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    N = population

    def seir_model(t, y, R0, R_post, change_day):
        S, E, I, R = y
        beta = R0 * gamma if t < change_day else R_post * gamma
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    def objective(params):
        R0, R_post, change_day = params
        y0 = [population - initial_infectious, 0.0, initial_infectious, 0.0]
        t = np.linspace(0, 70, 71)
        sol = odeint(seir_model, y0, t, args=(R0, R_post, change_day))
        I_values = sol[:, 2]
        expected = [I_values[i+1] - I_values[i] for i in range(70)]
        error = sum((e - c)**2 for e, c in zip(expected, cases))
        return error

    initial_guess = [2.0, 1.0, 0.0]
    bounds = [(0.1, 10.0), (0.1, 2.0), (0.0, 70.0)]
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)

    if not result.success:
        return {
            "R0": 2.0,
            "R_post": 1.0,
            "change_day": 0,
            "forecast": [0.0]*horizon
        }

    R0_opt, R_post_opt, change_day_opt = result.x
    change_day_opt = round(change_day_opt)
    change_day_opt = max(0, min(change_day_opt, 70))
    if abs(R_post_opt - R0_opt) < 0.05:
        change_day_opt = None

    y0 = [population - initial_infectious, 0.0, initial_infectious, 0.0]
    t = np.linspace(0, 70, 71)
    sol = odeint(seir_model, y0, t, args=(R0_opt, R_post_opt, change_day_opt))
    I_values = sol[:, 2]

    t_forecast = np.linspace(70, 70 + horizon, horizon + 1)
    sol_forecast = odeint(seir_model, sol[-1], t_forecast, args=(R0_opt, R_post_opt, change_day_opt))
    I_forecast = sol_forecast[:, 2]
    forecast = [I_forecast[i+1] - I_forecast[i] for i in range(horizon)]

    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt,
        "forecast": forecast
    }