import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}.
    """
    # Convert latent and infectious periods to rates
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0

    # Time points
    days = np.arange(1, len(cases) + 1)

    # Define the SEIR model
    def seir_model(t, y, R0, R_post, change_day):
        S, E, I, R = y
        # Calculate the transmission rate
        beta = R0 * gamma if t < change_day else R_post * gamma
        # Calculate the derivatives
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    # Estimate the parameters using optimization
    initial_guess = [3.0, 3.0, 30.0]
    result = minimize(lambda params: np.sum((seir_model(days, [S0, E0, I0, R0], params[0], params[1], params[2]) - cases) ** 2), initial_guess)

    # Extract the estimated parameters
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]

    # Forecast the future cases
    # Simulate the SEIR model for the next horizon days
    t_span = (days[-1], days[-1] + horizon)
    t_eval = np.linspace(t_span[0], t_span[1], horizon + 1)
    sol = solve_ivp(lambda t, y: seir_model(t, y, R0, R_post, change_day), t_span, [S0, E0, I0, R0], t_eval=t_eval)
    forecast = sol.y[2][1:]  # Extract the infectious cases

    return {"R0": R0, "R_post": R_post, "change_day": change_day, "forecast": forecast.tolist()}