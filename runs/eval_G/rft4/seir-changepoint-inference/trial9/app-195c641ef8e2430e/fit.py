import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    y0 = [S0, E0, I0, R0]

    def compute_expected(R0, R_post, change_day):
        def seir_ode(t, y):
            if change_day is None:
                beta = R0 * gamma
            else:
                beta = R0 * gamma if t < change_day else R_post * gamma
            S, E, I, R = y
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        sol = solve_ivp(
            fun=seir_ode,
            t_span=(0, 70),
            y0=y0,
            t_eval=np.arange(0, 71),
            method='RK45'
        )
        expected = np.zeros(70)
        for d in range(1, 71):
            E_prev = sol.y[1][d-1]
            E_curr = sol.y[1][d]
            expected[d-1] = sigma * (E_prev + E_curr) * 0.5 * 1
        return expected

    def neg_log_likelihood(params, change_day=None):
        if change_day is None:
            R0 = params[0]
            R_post = R0
            expected = compute_expected(R0, R_post, change_day)
        else:
            R0, R_post = params
            expected = compute_expected(R0, R_post, change_day)
        log_likelihood = 0.0
        for d in range(70):
            observed = cases[d]
            mu = expected[d]
            if mu == 0:
                log_likelihood += -observed * np.log(1e-10)
            else:
                log_likelihood += observed * np.log(mu) - mu
        return -log_likelihood

    # Try no-intervention case
    result_no_intervention = minimize(
        fun=lambda x: neg_log_likelihood(x, change_day=None),
        x0=[1.0],
        method='L-BFGS-B',
        bounds=[(0.1, 10.0)]
    )
    best_likelihood = result_no_intervention.fun
    best_params = (None, result_no_intervention.x[0], result_no_intervention.x[0], None)

    # Try intervention cases
    for change_day in range(1, 71):
        result_intervention = minimize(
            fun=lambda x: neg_log_likelihood(x, change_day=change_day),
            x0=[1.0, 1.0],
            method='L-BFGS-B',
            bounds=[(0.1, 10.0), (0.1, 10.0)]
        )
        current_likelihood = result_intervention.fun
        if current_likelihood < best_likelihood:
            best_likelihood = current_likelihood
            best_params = (change_day, result_intervention.x[0], result_intervention.x[1], None)

    change_day, R0, R_post, _ = best_params

    def compute_forecast(R0, R_post, change_day):
        def seir_ode(t, y):
            if change_day is None:
                beta = R0 * gamma
            else:
                beta = R0 * gamma if t < change_day else R_post * gamma
            S, E, I, R = y
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        sol = solve_ivp(
            fun=seir_ode,
            t_span=(0, 70 + horizon),
            y0=y0,
            t_eval=np.arange(0, 70 + horizon + 1),
            method='RK45'
        )
        forecast = np.zeros(horizon)
        for d in range(71, 70 + horizon + 1):
            E_prev = sol.y[1][d-1]
            E_curr = sol.y[1][d]
            forecast[d-71] = sigma * (E_prev + E_curr) * 0.5 * 1
        return forecast

    forecast = compute_forecast(R0, R_post, change_day)

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day if change_day is not None else None,
        "forecast": forecast.tolist()
    }