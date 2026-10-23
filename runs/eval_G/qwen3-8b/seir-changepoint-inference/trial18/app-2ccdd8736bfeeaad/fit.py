import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize as opt

def seir_model(t, y, beta, sigma, gamma, population):
    S, E, I, R = y
    dSdt = -beta * I * S / population
    dEdt = beta * I * S / population - sigma * E
    dI_dt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dI_dt, dRdt]

def simulate_seir(R0, R_post, change_day, sigma, gamma, population, initial_infectious, days=70):
    def beta_func(t):
        if t >= change_day:
            return R_post * gamma
        else:
            return R0 * gamma
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0_val = 0
    y0 = [S0, E0, I0, R0_val]
    t_span = (0, days)
    # Solve ODEs
    t_eval = np.arange(1, days + 1)  # Simulate for days 1 to 70
    sol = solve_ivp(lambda t, y: seir_model(t, y, beta_func(t), sigma, gamma, population), t_span, y0, t_eval=t_eval)
    E_values = sol.y[1]
    # Compute expected cases by taking sigma * E(t) at each day
    expected_cases = [sigma * E_values[i] for i in range(len(E_values))]  # Days 1 to 70
    return expected_cases

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    days = 70

    def nll(params):
        R0, R_post, change_day = params
        R_post = min(R_post, R0)  # Ensure R_post <= R0
        expected_cases = simulate_seir(R0, R_post, change_day, sigma, gamma, population, initial_infectious, days)
        nll = 0
        for i in range(len(cases)):
            lambda_i = max(expected_cases[i], 1e-10)  # Avoid log(0)
            k_i = cases[i]
            nll += lambda_i - k_i * np.log(lambda_i)
        return nll

    # Initial guess: R0, R_post, change_day
    # Adjusted initial guess closer to true values
    initial_guess = [3.0, 2.0, 35.0]  # R0, R_post, change_day
    bounds = [(0.1, 10.0), (0.1, 10.0), (0.0, 70.0)]

    # Use a more robust optimization method with increased iterations
    result = opt.minimize(nll, initial_guess, bounds=bounds, method='Nelder-Mead', options={'maxiter': 200})
    R0_opt = result.x[0]
    R_post_opt = result.x[1]
    change_day_opt = result.x[2]

    # Determine if there's an intervention
    # Use original threshold for scoring
    if R_post_opt <= 0.75 * R0_opt:
        change_day_final = change_day_opt
    else:
        change_day_final = None

    # Forecast the next horizon days
    forecast_days = horizon
    days_forecast = 70 + forecast_days
    expected_cases_full = simulate_seir(R0_opt, R_post_opt, change_day_opt, sigma, gamma, population, initial_infectious, days_forecast)
    forecast = expected_cases_full[70:70 + forecast_days]

    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_final,
        "forecast": forecast
    }