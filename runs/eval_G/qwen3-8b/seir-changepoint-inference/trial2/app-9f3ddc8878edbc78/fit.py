import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize as optimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1 / latent_period
    gamma = 1 / infectious_period
    initial_S = population - initial_infectious
    initial_E = 0
    initial_I = initial_infectious
    initial_R = 0
    initial_conditions = (initial_S, initial_E, initial_I, initial_R)

    best_error = float('inf')
    best_R0 = None
    best_R_post = None
    best_change_day = None
    best_forecast = []

    for change_day_candidate in range(0, 71):
        # Split cases
        cases_before = cases[:change_day_candidate]
        cases_after = cases[change_day_candidate:]

        # Optimize R0 for before segment
        def error_R0(R):
            time_span = max(change_day_candidate, 1)
            def seir_model(t, y):
                S, E, I, R_ = y
                dSdt = -R * gamma * S * I / population
                dEdt = R * gamma * S * I / population - sigma * E
                dIdt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dIdt, dRdt]
            t_eval = np.arange(0, time_span + 1e-6, 1)
            sol = solve_ivp(seir_model, [0, time_span], initial_conditions, t_eval=t_eval)
            predicted_cases = []
            for d in range(1, int(time_span) + 1):
                E_d_minus_1 = sol.y[1][d-1]
                E_d = sol.y[1][d]
                predicted = sigma * (E_d_minus_1 + E_d) / 2 * 1
                predicted_cases.append(predicted)
            error = sum((predicted - observed)**2 for predicted, observed in zip(predicted_cases, cases_before))
            return error

        result = optimize.minimize(error_R0, x0=1.0, bounds=[(0.1, 10.0)], method='L-BFGS-B')
        R0 = result.x[0]
        
        # Simulate before segment with R0
        def seir_model_before(t, y):
            S, E, I, R_ = y
            dSdt = -R0 * gamma * S * I / population
            dEdt = R0 * gamma * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        t_eval_first = np.arange(0, change_day_candidate + 1e-6, 1)
        sol_before = solve_ivp(seir_model_before, [0, change_day_candidate], initial_conditions, t_eval=t_eval_first)
        initial_after = sol_before.y[:, change_day_candidate]
        
        # Optimize R_post for after segment
        def error_R_post(R_post):
            time_span_after = max(70 - change_day_candidate, 1)
            def seir_model_after(t, y):
                S, E, I, R_ = y
                dSdt = -R_post * gamma * S * I / population
                dEdt = R_post * gamma * S * I / population - sigma * E
                dIdt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dIdt, dRdt]
            t_eval_after = np.arange(change_day_candidate, 70 + 1e-6, 1)
            sol_after = solve_ivp(seir_model_after, [change_day_candidate, 70], initial_after, t_eval=t_eval_after)
            predicted_cases_after = []
            for d in range(change_day_candidate + 1, 71):
                E_d_minus_1 = sol_after.y[1][d - change_day_candidate - 1]
                E_d = sol_after.y[1][d - change_day_candidate]
                predicted = sigma * (E_d_minus_1 + E_d) / 2 * 1
                predicted_cases_after.append(predicted)
            observed_cases_after = cases_after
            error = sum((predicted - observed)**2 for predicted, observed in zip(predicted_cases_after, observed_cases_after))
            return error
        
        result_post = optimize.minimize(error_R_post, x0=1.0, bounds=[(0.1, 10.0)], method='L-BFGS-B')
        R_post = result_post.x[0]
        total_error = result.fun + result_post.fun
        
        # Update best parameters
        if total_error < best_error:
            best_error = total_error
            best_R0 = R0
            best_R_post = R_post
            best_change_day = change_day_candidate

    # Generate forecast
    def seir_model_full(t, y):
        S, E, I, R_ = y
        if t < best_change_day:
            R = best_R0
        else:
            R = best_R_post
        dSdt = -R * gamma * S * I / population
        dEdt = R * gamma * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    t_eval_full = np.arange(0, 70 + 1e-6, 1)
    sol_full = solve_ivp(seir_model_full, [0, 70], initial_conditions, t_eval=t_eval_full)
    initial_forecast = sol_full.y[:, 70]

    def seir_model_forecast(t, y):
        S, E, I, R_ = y
        R = best_R_post
        dSdt = -R * gamma * S * I / population
        dEdt = R * gamma * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    t_eval_forecast = np.arange(70, 70 + horizon + 1e-6, 1)
    sol_forecast = solve_ivp(seir_model_forecast, [70, 70 + horizon], initial_forecast, t_eval=t_eval_forecast)
    forecast = []
    for d in range(70 + 1, 70 + horizon + 1):
        E_d_minus_1 = sol_forecast.y[1][d - 70 - 1]
        E_d = sol_forecast.y[1][d - 70]
        predicted = sigma * (E_d_minus_1 + E_d) / 2 * 1
        forecast.append(predicted)

    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": forecast
    }
