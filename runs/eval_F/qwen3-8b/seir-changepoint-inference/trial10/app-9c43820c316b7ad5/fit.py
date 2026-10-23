import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert periods to rates
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    # Initial conditions: S, E, I, R
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    initial_conditions = [S0, E0, I0, R0]

    # Define the SEIR model
    def seir_model(t, y, beta):
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    # Objective function to minimize
    def objective(params):
        R0_guess, R_post_guess, change_day_guess = params
        beta = R0_guess * gamma
        beta_post = R_post_guess * gamma

        # Simulate the model
        t_span = (0, 70)
        t_eval = np.arange(0, 70 + 1)
        sol = solve_ivp(seir_model, t_span, initial_conditions, args=(beta,), t_eval=t_eval)
        S, E, I, R = sol.y

        # Apply the intervention
        if change_day_guess is not None:
            beta = beta_post if t_eval >= change_day_guess else beta

        # Compute expected cases
        expected_cases = np.zeros_like(cases)
        for i in range(len(cases)):
            expected_cases[i] = I[i] - I[i-1] if i > 0 else I[0]

        # Calculate the negative log-likelihood
        # We assume Poisson distribution for the cases
        return -np.sum(np.log(expected_cases) - cases)

    # Initial guess for R0 and R_post
    initial_guess = [3.0, 3.0, 30.0]  # R0, R_post, change_day

    # Optimization
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=[(0.1, 10.0), (0.1, 10.0), (0, 70)])
    R0, R_post, change_day = result.x

    # Forecasting
    forecast = []
    t_span = (70, 70 + horizon)
    t_eval = np.arange(70, 70 + horizon + 1)
    sol = solve_ivp(seir_model, t_span, initial_conditions, args=(beta,), t_eval=t_eval)
    S, E, I, R = sol.y

    # Apply the intervention
    if change_day is not None:
        beta = beta_post if t_eval >= change_day else beta

    # Compute expected cases for forecast
    for i in range(len(t_eval)):
        forecast.append(I[i] - I[i-1] if i > 0 else I[0])

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day if change_day is not None else None,
        "forecast": forecast
    }