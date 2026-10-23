import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert parameters to rates
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    # Define the SEIR model
    def seir_model(y, t, beta, sigma, gamma):
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    # Define the negative log-likelihood function
    def nll(params):
        R0, R_post, change_day = params
        # Set up the initial conditions
        S0 = population - initial_infectious
        E0 = 0.0
        I0 = initial_infectious
        R0_ = 0.0
        y0 = [S0, E0, I0, R0_]

        # Define the beta function
        def beta_func(t):
            if t < change_day:
                return R0 * gamma
            else:
                return R_post * gamma

        # Integrate the model
        t = np.arange(0, 70 + 1)
        sol = odeint(seir_model, y0, t, args=(beta_func, sigma, gamma))

        # Compute the expected cases
        expected_cases = sol[:, 2] - sol[:, 2][:-1]
        expected_cases = np.maximum(expected_cases, 0)

        # Compute the negative log-likelihood
        # Assuming Poisson likelihood
        nll_value = -np.sum(np.log(expected_cases) - cases)
        return nll_value

    # Optimize for R0 and R_post with no intervention
    params_no_intervention = [np.log(1.0), np.log(1.0), None]
    result_no_intervention = minimize(nll, params_no_intervention, method='L-BFGS-B', bounds=[(None, None), (None, None), (None, None)])

    # Optimize for R0 and R_post with intervention
    params_intervention = [np.log(1.0), np.log(1.0), 0]
    result_intervention = minimize(nll, params_intervention, method='L-BFGS-B', bounds=[(None, None), (None, None), (0, 70)])

    # Compare the models using BIC
    nll_no_intervention = result_no_intervention.fun
    nll_intervention = result_intervention.fun
    k_no_intervention = 1
    k_intervention = 3
    bic_no_intervention = 2 * nll_no_intervention + k_no_intervention * np.log(70)
    bic_intervention = 2 * nll_intervention + k_intervention * np.log(70)

    # Choose the best model
    if bic_intervention < bic_no_intervention:
        R0 = np.exp(result_intervention.x[0])
        R_post = np.exp(result_intervention.x[1])
        change_day = result_intervention.x[2]
    else:
        R0 = np.exp(result_no_intervention.x[0])
        R_post = np.exp(result_no_intervention.x[1])
        change_day = None

    # Forecast the next horizon days
    t_forecast = np.arange(70, 70 + horizon + 1)
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0_ = 0.0
    y0 = [S0, E0, I0, R0_]

    def beta_func(t):
        if t < change_day:
            return R0 * gamma
        else:
            return R_post * gamma

    sol_forecast = odeint(seir_model, y0, t_forecast, args=(beta_func, sigma, gamma))
    forecast = sol_forecast[:, 2] - sol_forecast[:, 2][:-1]
    forecast = np.maximum(forecast, 0)

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }