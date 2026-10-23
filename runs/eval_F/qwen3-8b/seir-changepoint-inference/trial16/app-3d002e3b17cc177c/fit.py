import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert days to time units
    days = np.arange(1, len(cases) + 1)
    time = days / 365

    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0

    # Parameters
    gamma = 1 / infectious_period
    sigma = 1 / latent_period

    # Objective function
    def objective(params):
        R0_val, R_post_val, change_day_val = params
        
        # Define beta(t)
        def beta(t):
            if t < change_day_val:
                return R0_val * gamma
            else:
                return R_post_val * gamma

        # SEIR model
        def seir(t, y):
            S, E, I, R = y
            dSdt = -beta(t) * S * I / population
            dEdt = beta(t) * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]

        # Solve the ODE
        sol = solve_ivp(seir, [0, 1], [S0, E0, I0, R0], t_eval=time)
        
        # Calculate expected cases
        expected_cases = sol.y[2][1:]  # Infectious cases
        
        # Calculate the objective function
        # We will use the negative log-likelihood assuming Poisson distribution
        # We will use the expected cases as the mean of the Poisson distribution
        # and the actual cases as the observed values
        
        # Calculate the negative log-likelihood
        neg_log_likelihood = -np.sum(np.log(expected_cases) - expected_cases + cases)
        
        # Penalize for change_day_val outside the range
        if change_day_val < 1 or change_day_val > len(cases):
            neg_log_likelihood += 1e6
        
        return neg_log_likelihood

    # Initial guess for parameters
    initial_guess = [3.0, 3.0, len(cases) / 2]

    # Optimization
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=[(0.1, 10.0), (0.1, 10.0), (1, len(cases))])

    # Extract results
    R0_val, R_post_val, change_day_val = result.x

    # Forecast
    forecast_days = np.arange(len(cases) + 1, len(cases) + 1 + horizon)
    forecast_time = forecast_days / 365

    # Define beta(t) for forecasting
    def beta_forecast(t):
        if t < change_day_val:
            return R0_val * gamma
        else:
            return R_post_val * gamma

    # Forecast using the optimized parameters
    sol_forecast = solve_ivp(seir, [0, 1], [S0, E0, I0, R0], t_eval=forecast_time)
    forecast_cases = sol_forecast.y[2][1:]  # Infectious cases

    # Return the results
    return {
        "R0": R0_val,
        "R_post": R_post_val,
        "change_day": change_day_val if change_day_val > 0 else None,
        "forecast": forecast_cases.tolist()
    }