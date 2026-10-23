import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert days to time units
    days = np.arange(len(cases))
    time_units = days / latent_period

    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0

    # Define the SEIR model
    def seir_model(t, y, beta):
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - E / latent_period
        dIdt = E / latent_period - I / infectious_period
        dRdt = I / infectious_period
        return [dSdt, dEdt, dIdt, dRdt]

    # Objective function to minimize
    def objective(params):
        R0, R_post, change_day = params
        beta = R0 * infectious_period
        beta_post = R_post * infectious_period

        # Simulate the SEIR model with the intervention
        beta_array = np.full(len(cases), beta)
        beta_array[change_day:] = beta_post

        # Solve the ODEs
        sol = solve_ivp(seir_model, [0, len(cases)], [S0, E0, I0, R0], t_eval=time_units.astype(int).tolist(), args=(beta_array,))
        E_sim = sol.y[1]

        # Calculate the expected cases
        expected_cases = np.zeros(len(cases))
        for d in range(len(cases)):
            expected_cases[d] = E_sim[d] - E_sim[d-1] if d > 0 else E_sim[d]

        # Calculate the negative log-likelihood
        # Assuming Poisson distribution for the cases
        log_likelihood = -np.sum(np.log(expected_cases) - cases)
        return log_likelihood

    # Initial guess for parameters
    initial_guess = [3.0, 3.0, 30]

    # Optimize the parameters
    result = minimize(objective, initial_guess, method='L-BFGS-B')
    R0_opt, R_post_opt, change_day_opt = result.x

    # Forecast the next horizon days
    forecast = np.zeros(horizon)
    beta_forecast = R0_opt * infectious_period
    beta_forecast_post = R_post_opt * infectious_period

    # Simulate the SEIR model for the forecast period
    time_forecast = np.arange(len(cases), len(cases) + horizon) / latent_period
    beta_array_forecast = np.full(horizon, beta_forecast)
    beta_array_forecast[change_day_opt:] = beta_forecast_post

    # Solve the ODEs for the forecast period
    sol_forecast = solve_ivp(seir_model, [len(cases), len(cases) + horizon], [S0, E0, I0, R0], t_eval=time_forecast.astype(int).tolist(), args=(beta_array_forecast,))
    E_sim_forecast = sol_forecast.y[1]

    # Calculate the expected cases for the forecast period
    for d in range(horizon):
        forecast[d] = E_sim_forecast[d] - E_sim_forecast[d-1] if d > 0 else E_sim_forecast[d]

    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt if change_day_opt is not None else None,
        "forecast": forecast.tolist()
    }