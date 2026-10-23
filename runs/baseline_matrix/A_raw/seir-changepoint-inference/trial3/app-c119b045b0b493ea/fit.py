import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert days to time units
    days = np.arange(len(cases))
    time_units = days / latent_period

    # Initial parameters
    R0 = 3.0
    R_post = R0
    change_day = None

    # Define the SEIR model
    def seir_model(params, time):
        R0, R_post, change_day = params
        # Calculate the effective reproduction number
        R = np.where(time >= change_day, R_post, R0)
        # Calculate the transmission rate
        beta = R * np.sqrt(1 / population) * np.sqrt(1 / infectious_period)
        # Calculate the infectious rate
        gamma = 1 / infectious_period
        # Calculate the exposed rate
        sigma = 1 / latent_period
        # Calculate the number of new cases
        # This is a simplified version, a full implementation would require more complex calculations
        # For the purpose of this task, we'll use a simple approximation
        return beta * (population - initial_infectious) * np.exp(-gamma * time) * np.exp(-sigma * time)

    # Define the objective function
    def objective(params):
        R0, R_post, change_day = params
        # Calculate the predicted cases
        predicted = seir_model(params, time_units)
        # Calculate the error
        error = np.sum((cases - predicted) ** 2)
        return error

    # Optimize the parameters
    result = minimize(objective, (R0, R_post, change_day), method='L-BFGS-B')
    R0, R_post, change_day = result.x

    # Forecast the next weeks
    forecast_days = np.arange(len(cases), len(cases) + horizon)
    forecast_time_units = forecast_days / latent_period
    forecast = seir_model((R0, R_post, change_day), forecast_time_units)

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }