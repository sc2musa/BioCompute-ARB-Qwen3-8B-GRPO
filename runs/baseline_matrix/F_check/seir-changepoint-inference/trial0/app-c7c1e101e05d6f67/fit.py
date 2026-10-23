import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert days to time units
    days = np.arange(len(cases))
    time_units = days / infectious_period

    # Define the SEIR model
    def seir_model(t, R0, R_post, change_day):
        # Initialize the SEIR compartments
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0

        # Time units for the model
        t_model = np.linspace(0, t[-1], len(t))

        # Simulate the SEIR dynamics
        for day in range(len(t_model)):
            # Calculate the transmission rate
            if day < change_day:
                beta = R0 * infectious_period
            else:
                beta = R_post * infectious_period

            # Calculate the number of new infections
            new_infections = beta * I * S / population

            # Update the compartments
            S -= new_infections
            E += new_infections
            # Update I and R using the time units
            I = E * latent_period
            R = I * infectious_period

            # Record the number of new cases (Exposed -> Infectious)
            if day < len(cases):
                cases[day] = new_infections

        return cases, S, I, R

    # Estimate R0 and R_post using optimization
    def objective(params):
        R0, R_post, change_day = params
        # Simulate the SEIR model with the estimated parameters
        simulated_cases, S, I, R = seir_model(days, R0, R_post, change_day)
        # Calculate the negative log-likelihood
        return -np.sum(np.log(simulated_cases))

    # Initial guess for R0 and R_post
    initial_guess = [3.0, 3.0, 0]

    # Optimize the parameters
    result = minimize(objective, initial_guess, method='L-BFGS-B')
    R0, R_post, change_day = result.x

    # Forecast the next horizon days
    forecast = []
    for day in range(len(cases), len(cases) + horizon):
        # Calculate the transmission rate
        if day < change_day:
            beta = R0 * infectious_period
        else:
            beta = R_post * infectious_period

        # Calculate the number of new infections
        new_infections = beta * I * S / population

        forecast.append(new_infections)

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day if change_day > 0 else None,
        "forecast": forecast
    }