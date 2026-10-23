import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert days to time units
    days = np.arange(len(cases))
    time_units = days / infectious_period

    # Define SEIR model
    def seir_model(t, y, beta, gamma, sigma):
        S, E, I, R = y
        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    # Objective function
    def objective(params, cases):
        beta, gamma, sigma, change_day = params
        # Initial conditions
        S0 = population - initial_infectious
        E0 = 0.0
        I0 = initial_infectious
        R0 = 0.0
        y0 = [S0, E0, I0, R0]

        # Time points
        t_span = (0, time_units[-1])
        t_eval = time_units

        # Solve ODE
        sol = solve_ivp(seir_model, t_span, y0, args=(beta, gamma, sigma), t_eval=t_eval, method='RK45')
        I = sol.y[2]

        # Calculate expected cases
        expected_cases = np.zeros_like(cases)
        for i in range(len(cases)):
            expected_cases[i] = I[i] - I[i-1] if i > 0 else I[i]

        # Calculate the negative log-likelihood
        # Assuming Poisson distribution for the cases
        return -np.sum(np.log(expected_cases) - cases)

    # Initial guess for parameters
    initial_guess = [3.0, 1.0 / infectious_period, 1.0 / latent_period, 30.0]

    # Optimization
    result = minimize(objective, initial_guess, args=(cases,), method='L-BFGS-B')
    beta, gamma, sigma, change_day = result.x

    # Estimate R0 and R_post
    R0 = beta / gamma
    R_post = R0

    # Detect change point
    if change_day is not None and change_day < len(cases):
        # Adjust the parameters after the change point
        beta_post = beta * (R_post / R0)
        # Re-solve the ODE with the new beta
        # This is a placeholder for a more sophisticated change point detection
        # For the purpose of this example, we will assume the change point is at the detected day
        pass

    # Forecast the next horizon days
    forecast = []
    # This is a placeholder for the forecasting logic
    # In a real implementation, we would simulate the model forward
    # for the next horizon days
    for _ in range(horizon):
        forecast.append(0.0)  # Placeholder for forecast

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast
    }