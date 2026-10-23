import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    def simulate_model(R0, R_post, change_day):
        def seir_ode(t, y):
            S, E, I, R = y
            if t < change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        t_span = (0, 70)
        y0 = [population - initial_infectious, 0, initial_infectious, 0]
        sol = solve_ivp(seir_ode, t_span, y0, t_eval=np.arange(0, 71), method='RK45')
        S, E, I, R = sol.y
        expected_cases = [I[d] - I[d-1] for d in range(1, 71)]
        return expected_cases

    best_R0 = 1.0
    best_R_post = 1.0
    best_change_day = 0
    best_likelihood = -np.inf

    for change_day in range(0, 71):
        if change_day == 0:
            def neg_log_likelihood_R0(R0):
                expected = simulate_model(R0, R0, 0)
                log_likelihood = sum(obs * np.log(exp) - exp for obs, exp in zip(cases, expected) if exp > 0)
                return -log_likelihood
            result = minimize(neg_log_likelihood_R0, x0=[1.0], method='Nelder-Mead')
            current_likelihood = -result.fun
            if current_likelihood > best_likelihood:
                best_likelihood = current_likelihood
                best_R0 = result.x[0]
                best_R_post = best_R0
                best_change_day = 0
        else:
            def neg_log_likelihood(params):
                R0, R_post = params
                expected = simulate_model(R0, R_post, change_day)
                log_likelihood = sum(obs * np.log(exp) - exp for obs, exp in zip(cases, expected) if exp > 0)
                return -log_likelihood
            result = minimize(neg_log_likelihood, x0=[1.0, 1.0], method='Nelder-Mead')
            current_likelihood = -result.fun
            if current_likelihood > best_likelihood:
                best_likelihood = current_likelihood
                best_R0 = result.x[0]
                best_R_post = result.x[1]
                best_change_day = change_day

    def forecast_model(R0, R_post, change_day):
        def seir_ode(t, y):
            S, E, I, R = y
            if t < change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        t_span = (0, 70 + horizon)
        y0 = [population - initial_infectious, 0, initial_infectious, 0]
        sol = solve_ivp(seir_ode, t_span, y0, t_eval=np.arange(0, 70 + horizon + 1), method='RK45')
        S, E, I, R = sol.y
        forecast = [I[d] - I[d-1] for d in range(71, 70 + horizon + 1)]
        return forecast

    forecast = forecast_model(best_R0, best_R_post, best_change_day)
    return {
        "R0": float(best_R0),
        "R_post": float(best_R_post),
        "change_day": float(best_change_day) if best_change_day != 0 else None,
        "forecast": forecast
    }