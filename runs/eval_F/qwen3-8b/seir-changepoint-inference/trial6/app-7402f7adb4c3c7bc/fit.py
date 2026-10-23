import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # SEIR model parameters
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0

    # Time steps
    days = len(cases)
    time = np.arange(days)

    # Define the SEIR model
    def seir_model(t, y, R0, R_post, change_day):
        S, E, I, R = y
        if change_day is not None:
            beta = R0 * gamma if t < change_day else R_post * gamma
        else:
            beta = R0 * gamma
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    # Optimization
    def objective(params):
        R0_opt, R_post_opt, change_day_opt = params
        # Handle change_day_opt being None
        if change_day_opt is None:
            change_day_opt = days + 1  # Set to a value beyond the observation period
        # Solve the SEIR model
        sol = solve_ivp(seir_model, [0, days-1], [S0, E0, I0, R0], args=(R0_opt, R_post_opt, change_day_opt), dense_output=True)
        new_cases_opt = np.zeros(days)
        for d in range(days):
            if d == 0:
                new_cases_opt[d] = sol.sol(d)[2] - sol.sol(d-1)[2]
            else:
                new_cases_opt[d] = sol.sol(d)[2] - sol.sol(d-1)[2]
        return np.sum((new_cases_opt - cases) ** 2)

    # Initial guess for parameters
    initial_guess = [3.0, 3.0, None]

    # Optimization
    result = minimize(objective, initial_guess, method='L-BFGS-B')
    R0_opt, R_post_opt, change_day_opt = result.x

    # Forecast the next days
    forecast = np.zeros(horizon)
    for d in range(horizon):
        t = days + d
        if change_day_opt is None:
            beta = R0_opt * gamma
        else:
            beta = R0_opt * gamma if t < change_day_opt else R_post_opt * gamma
        # Forecast the number of new cases
        # This is a simplified forecast; a full implementation would require solving the ODE for the forecast period
        forecast[d] = 0.0  # Placeholder for forecast calculation

    # Return the results
    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt,
        "forecast": forecast.tolist()
    }