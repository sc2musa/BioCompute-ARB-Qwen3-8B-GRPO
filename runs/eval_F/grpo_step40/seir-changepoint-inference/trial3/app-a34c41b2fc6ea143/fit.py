import numpy as np
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    # Parse input
    cases = np.array(cases)
    population = int(population)
    initial_infectious = float(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)

    # Initial state
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0

    # Create time array
    time = np.arange(1, len(cases) + 1)

    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    # Solve the ODE
    sol = solve_ivp(seir_model, [0, len(cases)], [S0, E0, I0, R0], t_eval=time)

    # Extract results
    S, E, I, R = sol.y.T

    # Calculate R0 and R_post
    R0 = np.max(I) / initial_infectious
    R_post = R0

    # Find change day
    change_day = None
    for d in range(len(cases)):
        if I[d] < np.max(I) * 0.75:
            change_day = d + 1
            break

    # Make forecast
    forecast = []
    for d in range(len(cases), len(cases) + horizon):
        if change_day is None or d < change_day:
            beta_val = R0 * gamma
        else:
            beta_val = R_post * gamma
        # Calculate new cases
        new_cases = beta_val * S[d-1] * I[d-1] / population
        forecast.append(new_cases)

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast
    }