import numpy as np
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Parse inputs
    cases = np.array(cases)
    population = int(population)
    initial_infectious = float(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)

    # Define SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        beta = R0 * gamma if t < change_day else R_post * gamma
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dI_dt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dI_dt, dRdt]

    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    y0 = [S0, E0, I0, R0]

    # Solve ODE
    sol = solve_ivp(seir_model, [0, 70], y0, t_eval=range(70), method='BDF', rtol=1e-8, atol=1e-12 * population)

    # Extract results
    S, E, I, R = sol.y

    # Calculate R0 and R_post
    R0 = np.max(I) / initial_infectious
    R_post = np.max(I[change_day:]) / initial_infectious

    # Forecast future cases
    forecast = np.zeros(horizon)
    for t in range(horizon):
        # Calculate new cases for day t+70
        new_cases = np.max(I[70 + t]) - np.max(I[70 + t - 1])
        forecast[t] = new_cases

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }