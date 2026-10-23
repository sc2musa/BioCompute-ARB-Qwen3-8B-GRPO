import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert days to time units
    time_units = 1.0
    days = np.arange(1, len(cases) + 1)
    time = days * time_units

    # Convert latent and infectious periods to time units
    latent_period *= time_units
    infectious_period *= time_units

    # Calculate sigma (1/latent_period)
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    # Define the SEIR model
    def seir_model(t, S, E, I, R, beta, sigma, gamma):
        dSdt = -beta * I * S / population
        dEdt = beta * I * S / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    # Define the objective function
    def objective(params):
        beta, gamma, change_day = params
        # Initial conditions
        S0 = population - initial_infectious
        E0 = 0.0
        I0 = initial_infectious
        R0 = 0.0

        # Time points for the model
        t_model = np.linspace(0, time[-1], len(time))

        # Solve the ODE
        sol = solve_ivp(lambda t, y: seir_model(t, *y, beta, sigma, gamma), [0, time[-1]], [S0, E0, I0, R0], t_eval=t_model)

        # Extract the number of new cases
        new_cases = np.zeros(len(time))
        for i in range(len(time) - 1):
            new_cases[i] = sol.y[2][i + 1] - sol.y[2][i]

        # Calculate the negative log-likelihood
        # Assuming Poisson distribution for the number of new cases
        # with mean equal to the model prediction
        return -np.sum(np.log(new_cases) - new_cases)

    # Initial guess for the parameters
    initial_guess = [3.0, 1.0 / infectious_period, 0.0]

    # Bounds for the parameters
    bounds = [(0.1, 10.0), (0.1, 10.0), (0.0, len(time))]

    # Optimize the parameters
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B', tol=1e-4)

    # Extract the estimated parameters
    beta_opt, gamma_opt, change_day_opt = result.x

    # Calculate R0 and R_post
    R0 = beta_opt / gamma_opt
    R_post = R0

    # If there is an intervention, calculate R_post
    if change_day_opt > 0:
        R_post = beta_opt / gamma_opt

    # Forecast the next weeks
    forecast = []
    for day in range(len(time) + 1, len(time) + 1 + horizon):
        # Solve the ODE for the forecast period
        t_forecast = np.linspace(0, time[-1], len(time) + 1)
        sol_forecast = solve_ivp(lambda t, y: seir_model(t, *y, beta_opt, sigma, gamma_opt), [0, time[-1]], [S0, E0, I0, R0], t_eval=t_forecast)

        # Extract the number of new cases
        new_cases_forecast = np.zeros(len(t_forecast) - 1)
        for i in range(len(t_forecast) - 1):
            new_cases_forecast[i] = sol_forecast.y[2][i + 1] - sol_forecast.y[2][i]

        # Append the forecast for the next day
        forecast.append(new_cases_forecast[-1])

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day_opt,
        "forecast": forecast
    }