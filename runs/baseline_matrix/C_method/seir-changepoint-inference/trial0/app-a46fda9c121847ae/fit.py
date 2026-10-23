import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert to numpy array
    cases = np.array(cases)
    T = len(cases)

    # Parameters
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    dt = 1.0

    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    C0 = 0.0

    # Define SEIR model
    def seir_model(y, t, beta):
        S, E, I, R, C = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        dCdt = sigma * E
        return [dSdt, dEdt, dIdt, dRdt, dCdt]

    # Negative binomial log-likelihood
    def neg_binom_log_likelihood(y, mu, size):
        return np.sum(np.log(scipy.stats.nbinom.pmf(y, size, 1.0 / (1.0 + mu))))

    # Objective function for optimization
    def objective(params):
        R0, R_post, change_day = params
        beta = R0 * gamma
        beta_post = R_post * gamma

        # Integrate the model
        if change_day is None:
            beta_all = beta
        else:
            beta_all = np.zeros(T + 1)
            beta_all[:change_day] = beta
            beta_all[change_day:] = beta_post

        # Solve ODE
        t = np.arange(0, T + 1)
        y0 = [S0, E0, I0, R0, C0]
        sol = odeint(seir_model, y0, t, args=(beta_all,))

        # Compute expected cases
        expected_cases = sol[:, 4][1:] - sol[:, 4][:-1]

        # Compute log-likelihood
        nll = -neg_binom_log_likelihood(cases, expected_cases, 10)
        return nll

    # Optimize for no intervention
    params_no_intervention = [1.0, 1.0, None]
    result_no_intervention = minimize(objective, params_no_intervention, method='Nelder-Mead')

    # Optimize for change point
    params_change_point = [1.0, 1.0, 37]  # Initial guess for change day
    result_change_point = minimize(objective, params_change_point, method='Nelder-Mead')

    # Compare models with BIC
    nll_no_intervention = result_no_intervention.fun
    nll_change_point = result_change_point.fun
    k_no_intervention = 1
    k_change_point = 3
    bic_no_intervention = 2 * nll_no_intervention + k_no_intervention * np.log(T)
    bic_change_point = 2 * nll_change_point + k_change_point * np.log(T)

    if bic_change_point < bic_no_intervention:
        R0 = result_change_point.x[0]
        R_post = result_change_point.x[1]
        change_day = result_change_point.x[2]
    else:
        R0 = result_no_intervention.x[0]
        R_post = result_no_intervention.x[1]
        change_day = None

    # Forecast
    beta = R0 * gamma
    beta_post = R_post * gamma
    if change_day is None:
        beta_all = beta
    else:
        beta_all = np.zeros(T + 1)
        beta_all[:change_day] = beta
        beta_all[change_day:] = beta_post

    t_forecast = np.arange(T, T + horizon + 1)
    y0_forecast = [S0, E0, I0, R0, C0]
    sol_forecast = odeint(seir_model, y0_forecast, t_forecast, args=(beta_all,))
    forecast = sol_forecast[:, 4][1:] - sol_forecast[:, 4][:-1]

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }