import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Define the SEIR model
    def seir_model(t, S, E, I, R, beta, sigma, gamma):
        # Calculate the derivatives
        dSdt = -beta * I * S / population
        dEdt = beta * I * S / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    # Define the objective function for optimization
    def objective(params):
        R0, R_post, change_day = params
        # Calculate gamma and sigma
        gamma = 1.0 / infectious_period
        sigma = 1.0 / latent_period
        # Initialize the SEIR model
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        # Simulate the SEIR model
        time = np.arange(1, len(cases) + 1)
        cases_simulated = np.zeros(len(cases))
        I_prev = I
        for t in range(1, len(cases) + 1):
            if t > change_day:
                beta = R_post * gamma
            else:
                beta = R0 * gamma
            dSdt, dEdt, dIdt, dRdt = seir_model(t, S, E, I, R, beta, sigma, gamma)
            S += dSdt
            E += dEdt
            I += dIdt
            R += dRdt
            # Update the cases
            cases_simulated[t - 1] = I - I_prev
            I_prev = I
        # Calculate the objective function
        return np.sum((cases_simulated - cases) ** 2)

    # Initial guess for R0 and R_post
    initial_guess = [2.0, 2.0, 30.0]
    # Optimize the parameters
    result = minimize(objective, initial_guess, method='L-BFGS-B')
    R0_opt, R_post_opt, change_day_opt = result.x

    # Forecast future cases
    forecast = []
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    I_prev = I
    for t in range(len(cases) + 1, len(cases) + horizon + 1):
        if t > change_day_opt:
            beta = R_post_opt * gamma
        else:
            beta = R0_opt * gamma
        dSdt, dEdt, dIdt, dRdt = seir_model(t, S, E, I, R, beta, sigma, gamma)
        S += dSdt
        E += dEdt
        I += dIdt
        R += dRdt
        forecast.append(I - I_prev)
        I_prev = I

    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt if change_day_opt is not None else None,
        "forecast": forecast
    }