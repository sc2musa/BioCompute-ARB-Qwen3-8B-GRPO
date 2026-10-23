import numpy as np
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        # Calculate gamma and sigma
        sigma = 1 / latent_period
        gamma = 1 / infectious_period
        # Determine the current beta
        if change_day is not None:
            beta = R_post * gamma
        else:
            beta = R0 * gamma
        # Calculate the derivatives
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0
    y0 = [S0, E0, I0, R0]

    # Solve the ODE
    sol = solve_ivp(seir_model, [0, 70], y0, method='BDF', rtol=1e-8, atol=1e-12 * population)

    # Extract the solution
    S, E, I, R = sol.y

    # Calculate R0 and R_post
    R0 = np.max(I) / initial_infectious
    R_post = np.max(I[change_day:]) / initial_infectious

    # Forecast the next weeks
    forecast = np.zeros(horizon)
    for d in range(horizon):
        t = 70 + d
        S, E, I, R = sol.y
        # Calculate the current beta
        if change_day is not None:
            beta = R_post * gamma
        else:
            beta = R0 * gamma
        forecast[d] = beta * S * I / population

    return {
        'R0': R0,
        'R_post': R_post,
        'change_day': change_day,
        'forecast': forecast.tolist()
    }