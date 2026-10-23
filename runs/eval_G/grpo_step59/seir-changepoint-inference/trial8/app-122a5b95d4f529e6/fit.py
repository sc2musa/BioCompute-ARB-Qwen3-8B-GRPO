import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Parse input
    cases = np.array(cases)
    population = np.array(population)
    initial_infectious = np.array(initial_infectious)
    latent_period = np.array(latent_period)
    infectious_period = np.array(infectious_period)
    horizon = np.array(horizon)

    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        # Initialize compartments
        S = population - initial_infectious
        E = 0
        I = initial_infectious
        R = 0

        # Calculate gamma and sigma
        gamma = 1 / infectious_period
        sigma = 1 / latent_period

        # Initialize time steps
        time_steps = np.arange(1, 71)
        expected_cases = np.zeros(70)

        # Simulate the SEIR model
        for t in range(1, 71):
            # Calculate the transmission rate
            if t <= change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma

            # Calculate the number of new cases
            new_cases = beta * I * S / population
            expected_cases[t-1] = new_cases

            # Update compartments
            S -= new_cases
            E += new_cases
            I = sigma * E
            R = gamma * I

        return expected_cases

    # Define the objective function
    def objective_function(params):
        expected_cases = seir_model(params)
        # Calculate the error between expected and observed cases
        error = np.sum((cases - expected_cases) ** 2)
        return error

    # Set initial guess for parameters
    initial_guess = [3.0, 3.0, 30]

    # Optimize the parameters
    result = minimize(objective_function, initial_guess, method='SLSQP', bounds=[(0.1, 10), (0.1, 10), (1, 70)])

    # Extract the optimized parameters
    R0_opt = result.x[0]
    R_post_opt = result.x[1]
    change_day_opt = result.x[2]

    # Forecast future cases
    forecast = seir_model([R0_opt, R_post_opt, change_day_opt])

    # Return the results
    return {
        'R0': R0_opt,
        'R_post': R_post_opt,
        'change_day': change_day_opt,
        'forecast': forecast.tolist()
    }