import numpy as np
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        beta = R0 * gamma
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

    # Define the parameters
    sigma = 1 / latent_period
    gamma = 1 / infectious_period

    # Solve the ODE
    sol = solve_ivp(seir_model, [0, 70], y0, t_eval=np.arange(0, 70, 1), method='BDF', rtol=1e-8, atol=1e-12 * population)

    # Extract the solution
    S, E, I, R = sol.y

    # Calculate the forecast
    forecast = []
    for t in range(70, 70 + horizon):
        # Calculate the expected new cases
        if t < len(I):
            expected_new_cases = I[t] - I[t-1]
            forecast.append(expected_new_cases)
        else:
            forecast.append(0)

    # Return the results
    return {
        'R0': R0,
        'R_post': R0,
        'change_day': None,
        'forecast': forecast
    }