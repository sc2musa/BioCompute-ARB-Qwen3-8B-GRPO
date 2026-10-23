import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Define the cost function
    def cost_function(x):
        R0, R_post, change_day = x
        sigma = 1 / latent_period
        gamma = 1 / infectious_period
        S = population - initial_infectious
        E = 0
        I = initial_infectious
        R = 0
        simulated_cases = []
        for day in range(1, 71):
            if day < change_day:
                beta_t = R0 * gamma
            else:
                beta_t = R_post * gamma
            dS_dt = -beta_t * S * I / population
            dE_dt = beta_t * S * I / population - sigma * E
            dI_dt = sigma * E - gamma * I
            dR_dt = gamma * I
            S += dS_dt
            E += dE_dt
            I += dI_dt
            R += dR_dt
            simulated_cases.append(dI_dt)
        error = np.sum((np.array(simulated_cases) - np.array(cases)) ** 2)
        return error

    # Set up constraints and bounds
    constraints = [{'type': 'ineq', 'fun': lambda x: x[1] - x[0]}]
    bounds = [(0, None), (0, None), (0, 70)]
    x0 = [1.0, 1.0, 0.0]

    # Perform the optimization
    result = minimize(cost_function, x0, method='L-BFGS-B', bounds=bounds, constraints=constraints)

    # Extract the optimized parameters
    R0_opt = result.x[0]
    R_post_opt = result.x[1]
    change_day_opt = result.x[2]

    # Forecast the next 'horizon' days
    S = population - initial_infectious
    E = 0
    I = initial_infectious
    R = 0
    forecast = []
    for day in range(71, 71 + horizon):
        if day < change_day_opt:
            beta_t = R0_opt * gamma
        else:
            beta_t = R_post_opt * gamma
        dS_dt = -beta_t * S * I / population
        dE_dt = beta_t * S * I / population - sigma * E
        dI_dt = sigma * E - gamma * I
        dR_dt = gamma * I
        S += dS_dt
        E += dE_dt
        I += dI_dt
        R += dR_dt
        forecast.append(dI_dt)

    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt,
        "forecast": forecast
    }