import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert days to time units
    days = np.arange(1, len(cases) + 1)
    time_units = days / infectious_period

    # Initial guess for R0 and R_post
    R0_guess = 3.0
    R_post_guess = 3.0
    change_day_guess = None

    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        beta = np.zeros_like(time_units)
        beta[time_units < change_day] = R0 * infectious_period
        beta[time_units >= change_day] = R_post * infectious_period

        # Solve the SEIR model
        susceptible = np.zeros_like(time_units)
        exposed = np.zeros_like(time_units)
        infectious = np.zeros_like(time_units)
        removed = np.zeros_like(time_units)

        # Initial conditions
        susceptible[0] = population - initial_infectious
        infectious[0] = initial_infectious

        # Time step
        dt = 1 / infectious_period

        # Integrate the SEIR model
        for t in range(1, len(time_units)):
            # Calculate the number of new infections
            new_infections = beta[t-1] * susceptible[t-1] * infectious[t-1] / population

            # Update the compartments
            susceptible[t] = susceptible[t-1] - new_infections
            exposed[t] = exposed[t-1] + new_infections
            infectious[t] = infectious[t-1] + exposed[t-1] - infectious[t-1] * (1 / infectious_period)
            removed[t] = removed[t-1] + infectious[t-1] * (1 / infectious_period)

        # Calculate the expected new cases
        expected_cases = infectious[1:] - infectious[:-1]

        # Calculate the negative log-likelihood
        # Assuming over-dispersed count noise around the model expectation
        # We'll use a Poisson distribution for simplicity
        log_likelihood = -np.sum(np.log(expected_cases) - cases + np.log(np.exp(expected_cases) / expected_cases))

        return log_likelihood

    # Define the objective function
    def objective(params):
        return seir_model(params)

    # Optimize the parameters
    result = minimize(objective, [R0_guess, R_post_guess, change_day_guess], method='L-BFGS-B')

    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]

    # Forecast the next days
    forecast_days = np.arange(len(cases) + 1, len(cases) + 1 + horizon)
    forecast_time_units = forecast_days / infectious_period
    beta_forecast = np.zeros_like(forecast_time_units)
    beta_forecast[forecast_time_units < change_day] = R0 * infectious_period
    beta_forecast[forecast_time_units >= change_day] = R_post * infectious_period

    # Solve the SEIR model for the forecast period
    susceptible_forecast = np.zeros_like(forecast_time_units)
    exposed_forecast = np.zeros_like(forecast_time_units)
    infectious_forecast = np.zeros_like(forecast_time_units)
    removed_forecast = np.zeros_like(forecast_time_units)

    # Initial conditions for the forecast period
    susceptible_forecast[0] = population - initial_infectious
    infectious_forecast[0] = initial_infectious

    # Time step
    dt = 1 / infectious_period

    # Integrate the SEIR model for the forecast period
    for t in range(1, len(forecast_time_units)):
        # Calculate the number of new infections
        new_infections = beta_forecast[t-1] * susceptible_forecast[t-1] * infectious_forecast[t-1] / population

        # Update the compartments
        susceptible_forecast[t] = susceptible_forecast[t-1] - new_infections
        exposed_forecast[t] = exposed_forecast[t-1] + new_infections
        infectious_forecast[t] = infectious_forecast[t-1] + exposed_forecast[t-1] - infectious_forecast[t-1] * (1 / infectious_period)
        removed_forecast[t] = removed_forecast[t-1] + infectious_forecast[t-1] * (1 / infectious_period)

    # Calculate the expected new cases for the forecast period
    forecast = infectious_forecast[1:] - infectious_forecast[:-1]

    # Return the results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day if change_day is not None else None,
        "forecast": forecast.tolist()
    }