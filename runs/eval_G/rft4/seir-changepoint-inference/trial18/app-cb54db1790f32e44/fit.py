import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    y0 = [S0, E0, I0, R0]

    def dydt(t, y, R0, R_post, change_day):
        S, E, I, R = y
        if t < change_day:
            beta = R0 * gamma
        else:
            beta = R_post * gamma
        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    def negative_log_likelihood(params):
        R0, R_post, change_day = params
        sol = solve_ivp(dydt, (0, 70), y0, t_eval=np.arange(0, 71), args=(R0, R_post, change_day), rtol=1e-3, atol=1e-8, method='BDF')
        expected_cases = []
        for d in range(1, 71):
            expected = sol.y[2][d] - sol.y[2][d-1]
            expected_cases.append(expected)
        log_likelihood = 0.0
        for d in range(70):
            observed = cases[d]
            expected = expected_cases[d]
            log_likelihood += observed * np.log(expected) - expected
        return -log_likelihood

    initial_guess = [2.0, 1.0, 30.0]
    bounds = [(0.1, 10.0), (0.1, 10.0), (1.0, 70.0)]
    result = minimize(negative_log_likelihood, initial_guess, bounds=bounds, method='L-BFGS-B')
    R0_opt = result.x[0]
    R_post_opt = result.x[1]
    change_day_opt = result.x[2]

    # Check if R_post is close to R0 (no intervention)
    if abs(R_post_opt - R0_opt) < 1e-5:
        change_day_opt = None

    # Forecast
    sol_forecast = solve_ivp(dydt, (0, 70 + horizon), y0, t_eval=np.arange(0, 70 + horizon + 1), args=(R0_opt, R_post_opt, change_day_opt), rtol=1e-3, atol=1e-8, method='BDF')
    forecast = []
    for d in range(71, 70 + horizon + 1):
        expected = sol_forecast.y[2][d] - sol_forecast.y[2][d-1]
        forecast.append(expected)

    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt,
        "forecast": forecast
    }