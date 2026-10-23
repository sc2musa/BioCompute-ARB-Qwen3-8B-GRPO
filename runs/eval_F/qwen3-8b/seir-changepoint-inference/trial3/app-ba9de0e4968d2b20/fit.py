import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert latent and infectious periods to rates
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    # Initial guess for R0
    R0_guess = 2.0
    R_post_guess = R0_guess

    # Define the SEIR model
    def seir_model(params, t):
        R0, R_post, change_day = params
        beta = np.zeros_like(t)
        beta[t < change_day] = R0 * gamma
        beta[t >= change_day] = R_post * gamma

        # Solve the SEIR model
        # Initial conditions: S = population - initial_infectious, E = 0, I = initial_infectious, R = 0
        S0 = population - initial_infectious
        E0 = 0.0
        I0 = initial_infectious
        R0 = 0.0

        # Solve the ODEs
        sol = solve_ivp(lambda t, y: [
            -beta[int(t)] * y[0] * y[2] / population,  # dS/dt
            beta[int(t)] * y[0] * y[2] / population - sigma * y[1],  # dE/dt
            sigma * y[1] - gamma * y[2],  # dI/dt
            gamma * y[2]  # dR/dt
        ], [0, t[-1]], [S0, E0, I0, R0], t_eval=t)

        # Extract the predicted new cases (E -> I transitions)
        predicted_cases = sol.y[1][1:] - sol.y[1][:-1]  # dE/dt
        # Ensure the predicted cases have the same length as the observed cases
        if len(predicted_cases) < len(cases):
            predicted_cases = np.concatenate([predicted_cases, [predicted_cases[-1]]])
        return predicted_cases.tolist()

    # Optimize the parameters
    result = minimize(lambda params: np.sum((np.array(cases) - seir_model(params, np.arange(len(cases))))**2), 
                      x0=[R0_guess, R_post_guess, 0], 
                      bounds=[(0.1, 5.0), (0.1, 5.0), (0, len(cases))])

    # Extract the optimized parameters
    R0_opt, R_post_opt, change_day_opt = result.x

    # Forecast the next horizon days
    forecast_days = np.arange(len(cases), len(cases) + horizon)
    forecast = seir_model([R0_opt, R_post_opt, change_day_opt], forecast_days)

    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt if change_day_opt > 0 else None,
        "forecast": forecast
    }