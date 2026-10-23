import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    days = len(cases)
    best_error = float('inf')
    best_R0 = 0.0
    best_R_post = 0.0
    best_change_day = 0
    best_state = None

    def seir_model(t, y, beta):
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    for change_day in range(0, days + 1):
        if change_day == 0:
            def error_func(R0):
                beta = R0 * gamma
                t_span = (0, days)
                t_eval = np.arange(0, days + 1)
                sol = solve_ivp(lambda t, y: seir_model(t, y, beta), t_span, [population - initial_infectious, 0, initial_infectious, 0], t_eval=t_eval, dense_output=True)
                I = sol.sol(t_eval)[2]
                predicted_cases = np.diff(I)
                error = np.sum((predicted_cases - cases)**2)
                return error
            result = minimize_scalar(error_func, bounds=(0.1, 10.0), method='bounded')
            current_error = result.fun
            R0 = result.x
        else:
            first_days = change_day - 1
            second_days = days - change_day

            if first_days < 1:
                # First period has no data, optimize R_post for entire data
                def error_func_R_post(R_post):
                    beta = R_post * gamma
                    t_span = (0, days)
                    t_eval = np.arange(0, days + 1)
                    sol = solve_ivp(lambda t, y: seir_model(t, y, beta), t_span, [population - initial_infectious, 0, initial_infectious, 0], t_eval=t_eval, dense_output=True)
                    I = sol.sol(t_eval)[2]
                    predicted_cases = np.diff(I)
                    error = np.sum((predicted_cases - cases)**2)
                    return error
                result_R_post = minimize_scalar(error_func_R_post, bounds=(0.0, 10.0), method='bounded')
                R_post = result_R_post.x
                error_second = result_R_post.fun
                current_error = error_second
            else:
                # Optimize R0 for first period
                def error_func_R0(R0):
                    beta = R0 * gamma
                    t_span = (0, first_days)
                    t_eval = np.arange(0, first_days + 1)
                    sol = solve_ivp(lambda t, y: seir_model(t, y, beta), t_span, [population - initial_infectious, 0, initial_infectious, 0], t_eval=t_eval, dense_output=True)
                    I = sol.sol(t_eval)[2]
                    predicted_cases = np.diff(I)
                    error = np.sum((predicted_cases - cases[:first_days])**2)
                    return error
                result_R0 = minimize_scalar(error_func_R0, bounds=(0.1, 10.0), method='bounded')
                R0 = result_R0.x
                error_first = result_R0.fun

                # Optimize R_post for second period
                def error_func_R_post(R_post):
                    beta = R_post * gamma
                    t_span_first = (0, first_days)
                    t_eval_first = np.arange(0, first_days + 1)
                    sol_first = solve_ivp(lambda t, y: seir_model(t, y, R0 * gamma), t_span_first, [population - initial_infectious, 0, initial_infectious, 0], t_eval=t_eval_first, dense_output=True)
                    state_first = sol_first.sol(first_days)
                    t_span_second = (first_days, days)
                    t_eval_second = np.arange(first_days, days + 1)
                    sol_second = solve_ivp(lambda t, y: seir_model(t, y, beta), t_span_second, state_first, t_eval=t_eval_second, dense_output=True)
                    I = sol_second.sol(t_eval_second)[2]
                    predicted_cases = np.diff(I)
                    error = np.sum((predicted_cases - cases[first_days:])**2)
                    return error
                result_R_post = minimize_scalar(error_func_R_post, bounds=(0.0, R0), method='bounded')
                R_post = result_R_post.x
                error_second = result_R_post.fun
                current_error = error_first + error_second

        if current_error < best_error:
            best_error = current_error
            best_R0 = R0
            best_R_post = R_post if change_day != 0 else R0
            best_change_day = change_day

    if best_change_day == 0:
        t_span = (0, days)
        t_eval = np.arange(0, days + 1)
        sol = solve_ivp(lambda t, y: seir_model(t, y, best_R0 * gamma), t_span, [population - initial_infectious, 0, initial_infectious, 0], t_eval=t_eval, dense_output=True)
        state_at_end = sol.sol(days)
    else:
        first_days = best_change_day - 1
        t_span_first = (0, first_days)
        t_eval_first = np.arange(0, first_days + 1)
        sol_first = solve_ivp(lambda t, y: seir_model(t, y, best_R0 * gamma), t_span_first, [population - initial_infectious, 0, initial_infectious, 0], t_eval=t_eval_first, dense_output=True)
        state_first = sol_first.sol(first_days)
        t_span_second = (first_days, days)
        t_eval_second = np.arange(first_days, days + 1)
        sol_second = solve_ivp(lambda t, y: seir_model(t, y, best_R_post * gamma), t_span_second, state_first, t_eval=t_eval_second, dense_output=True)
        state_at_end = sol_second.sol(days)

    t_span_forecast = (days, days + horizon)
    t_eval_forecast = np.arange(days, days + horizon + 1)
    beta_forecast = best_R_post * gamma if best_change_day != 0 else best_R0 * gamma
    sol_forecast = solve_ivp(lambda t, y: seir_model(t, y, beta_forecast), t_span_forecast, state_at_end, t_eval=t_eval_forecast, dense_output=True)
    I_forecast = sol_forecast.sol(t_eval_forecast)[2]
    forecast = np.diff(I_forecast).tolist()

    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": forecast
    }