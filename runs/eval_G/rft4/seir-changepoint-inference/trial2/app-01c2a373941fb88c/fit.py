import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    def seir_model(t, y, R0, R_post, change_day, latent_period, infectious_period, population):
        S, E, I, R = y
        gamma = 1.0 / infectious_period
        sigma = 1.0 / latent_period
        if t < change_day:
            beta = R0 * gamma
        else:
            beta = R_post * gamma
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    y0 = [S0, E0, I0, R0]

    def neg_log_likelihood(params):
        R0_val, R_post_val, change_day_val = params
        sol = solve_ivp(
            fun=seir_model,
            t_span=(0, 70),
            y0=y0,
            args=(R0_val, R_post_val, change_day_val, latent_period, infectious_period, population),
            t_eval=np.arange(0, 71, 1),
            method='RK45',
            rtol=1e-2, atol=1e-6
        )
        I_vals = sol.y[2]
        predicted_cases = np.diff(I_vals)
        log_likelihood = np.sum(cases * np.log(predicted_cases + 1e-12) - predicted_cases)
        zero_mask = (predicted_cases == 0) & (cases != 0)
        if np.any(zero_mask):
            log_likelihood = -np.inf
        return -log_likelihood

    initial_R0 = 1.0
    initial_R_post = 1.0
    initial_change_day = 35.0
    initial_params = [initial_R0, initial_R_post, initial_change_day]

    bounds = (
        (0.001, None),
        (0.001, None),
        (0.0, 70.0)
    )

    result = minimize(
        fun=neg_log_likelihood,
        x0=initial_params,
        method='L-BFGS-B',
        bounds=bounds
    )

    R0_opt = result.x[0]
    R_post_opt = result.x[1]
    change_day_opt = result.x[2]

    sol = solve_ivp(
        fun=seir_model,
        t_span=(0, 70 + horizon),
        y0=y0,
        args=(R0_opt, R_post_opt, change_day_opt, latent_period, infectious_period, population),
        t_eval=np.arange(0, 70 + horizon + 1, 1),
        method='RK45',
        rtol=1e-2, atol=1e-6
    )
    I_vals = sol.y[2]
    forecast_days = np.arange(71, 70 + horizon + 1)
    forecast_cases = np.diff(I_vals)[forecast_days - 1]
    forecast = forecast_cases.tolist()

    if change_day_opt < 1 or change_day_opt > 70 or abs(R_post_opt - R0_opt) < 1e-5:
        change_day = None
        R_post = R0_opt
    else:
        change_day = change_day_opt
        R_post = R_post_opt

    return {
        "R0": R0_opt,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast
    }