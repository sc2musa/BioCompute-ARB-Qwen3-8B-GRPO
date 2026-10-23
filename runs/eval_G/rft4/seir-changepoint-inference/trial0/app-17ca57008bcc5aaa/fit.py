import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def seir_ode(t, y, beta, sigma, gamma, population):
    S, E, I, R = y
    dSdt = -beta * S * I / population
    dEdt = beta * S * I / population - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    initial_S = population - initial_infectious
    initial_E = 0.0
    initial_I = initial_infectious
    initial_R = 0.0
    initial_conditions = [initial_S, initial_E, initial_I, initial_R]

    best_log_likelihood = -np.inf
    best_R0 = 0.0
    best_R_post = 0.0
    best_change_day = None
    best_forecast = []

    candidate_change_days = [None] + list(range(1, 71))

    for change_day in candidate_change_days:
        if change_day is None:
            start_day1 = 0
            end_day1 = 70
            observed_cases1 = cases
            observed_cases2 = []
        else:
            start_day1 = 0
            end_day1 = change_day - 1
            observed_cases1 = cases[:end_day1]
            start_day2 = change_day
            end_day2 = 70
            observed_cases2 = cases[end_day1:]

        # Fit R0 for the first interval
        if len(observed_cases1) > 0:
            def objective_R0(R):
                beta = R * gamma
                t_span = (0, end_day1)
                t_eval = np.arange(0, end_day1 + 1)
                sol = solve_ivp(seir_ode, t_span, initial_conditions, args=(beta, sigma, gamma, population), method='RK45', t_eval=t_eval)
                expected = []
                for d in range(start_day1, end_day1 + 1):
                    start_t = d - 1
                    end_t = d
                    indices = np.where((sol.t >= start_t) & (sol.t <= end_t))[0]
                    if len(indices) == 0:
                        expected_cases = 0
                    else:
                        t_vals = sol.t[indices]
                        e_vals = sol.y[1][indices]
                        integral = sigma * np.trapz(e_vals, t_vals)
                        expected_cases = integral
                    expected.append(expected_cases)
                expected = [max(e, 1e-10) for e in expected]
                nll = sum((obs - exp) + np.log(exp) for obs, exp in zip(observed_cases1, expected))
                return nll
            result_R0 = minimize(objective_R0, x0=1.0, method='L-BFGS-B', bounds=[(0.1, 10.0)])
            R0 = result_R0.x[0]
        else:
            R0 = 0.0

        # Fit R_post for the second interval
        if len(observed_cases2) > 0:
            if change_day is not None:
                beta_R0 = R0 * gamma
                t_span = (0, end_day1)
                t_eval = np.arange(0, end_day1 + 1)
                sol1 = solve_ivp(seir_ode, t_span, initial_conditions, args=(beta_R0, sigma, gamma, population), method='RK45', t_eval=t_eval)
                final_state1 = sol1.y[:, -1]
                initial_conditions2 = final_state1
            else:
                initial_conditions2 = initial_conditions

            def objective_R_post(R):
                beta = R * gamma
                t_span = (start_day2, end_day2)
                t_eval = np.arange(start_day2, end_day2 + 1)
                sol = solve_ivp(seir_ode, t_span, initial_conditions2, args=(beta, sigma, gamma, population), method='RK45', t_eval=t_eval)
                expected = []
                for d in range(start_day2, end_day2 + 1):
                    start_t = d - 1
                    end_t = d
                    indices = np.where((sol.t >= start_t) & (sol.t <= end_t))[0]
                    if len(indices) == 0:
                        expected_cases = 0
                    else:
                        t_vals = sol.t[indices]
                        e_vals = sol.y[1][indices]
                        integral = sigma * np.trapz(e_vals, t_vals)
                        expected_cases = integral
                    expected.append(expected_cases)
                expected = [max(e, 1e-10) for e in expected]
                nll = sum((obs - exp) + np.log(exp) for obs, exp in zip(observed_cases2, expected))
                return nll
            result_R_post = minimize(objective_R_post, x0=1.0, method='L-BFGS-B', bounds=[(0.1, 10.0)])
            R_post = result_R_post.x[0]
        else:
            R_post = 0.0

        # Compute total likelihood
        nll1 = 0
        if len(observed_cases1) > 0:
            beta_R0 = R0 * gamma
            t_span = (0, end_day1)
            t_eval = np.arange(0, end_day1 + 1)
            sol1 = solve_ivp(seir_ode, t_span, initial_conditions, args=(beta_R0, sigma, gamma, population), method='RK45', t_eval=t_eval)
            expected1 = []
            for d in range(start_day1, end_day1 + 1):
                start_t = d - 1
                end_t = d
                indices = np.where((sol1.t >= start_t) & (sol1.t <= end_t))[0]
                if len(indices) == 0:
                    expected_cases = 0
                else:
                    t_vals = sol1.t[indices]
                    e_vals = sol1.y[1][indices]
                    integral = sigma * np.trapz(e_vals, t_vals)
                    expected_cases = integral
                expected1.append(expected_cases)
            expected1 = [max(e, 1e-10) for e in expected1]
            nll1 = sum((obs - exp) + np.log(exp) for obs, exp in zip(observed_cases1, expected1))
        else:
            nll1 = 0

        nll2 = 0
        if len(observed_cases2) > 0:
            beta_R_post = R_post * gamma
            t_span = (start_day2, end_day2)
            t_eval = np.arange(start_day2, end_day2 + 1)
            sol2 = solve_ivp(seir_ode, t_span, initial_conditions2, args=(beta_R_post, sigma, gamma, population), method='RK45', t_eval=t_eval)
            expected2 = []
            for d in range(start_day2, end_day2 + 1):
                start_t = d - 1
                end_t = d
                indices = np.where((sol2.t >= start_t) & (sol2.t <= end_t))[0]
                if len(indices) == 0:
                    expected_cases = 0
                else:
                    t_vals = sol2.t[indices]
                    e_vals = sol2.y[1][indices]
                    integral = sigma * np.trapz(e_vals, t_vals)
                    expected_cases = integral
                expected2.append(expected_cases)
            expected2 = [max(e, 1e-10) for e in expected2]
            nll2 = sum((obs - exp) + np.log(exp) for obs, exp in zip(observed_cases2, expected2))
        else:
            nll2 = 0

        total_nll = nll1 + nll2
        if total_nll < best_log_likelihood:
            best_log_likelihood = total_nll
            best_R0 = R0
            best_R_post = R_post
            best_change_day = change_day

            # Compute forecast
            if change_day is not None:
                beta_R0 = best_R0 * gamma
                t_span = (0, end_day1)
                t_eval = np.arange(0, end_day1 + 1)
                sol1 = solve_ivp(seir_ode, t_span, initial_conditions, args=(beta_R0, sigma, gamma, population), method='RK45', t_eval=t_eval)
                final_state1 = sol1.y[:, -1]
                beta_R_post = best_R_post * gamma
                t_span = (start_day2, end_day2)
                t_eval = np.arange(start_day2, end_day2 + 1)
                sol2 = solve_ivp(seir_ode, t_span, final_state1, args=(beta_R_post, sigma, gamma, population), method='RK45', t_eval=t_eval)
                final_state2 = sol2.y[:, -1]
            else:
                beta_R = best_R0 * gamma
                t_span = (0, 70)
                t_eval = np.arange(0, 71)
                sol = solve_ivp(seir_ode, t_span, initial_conditions, args=(beta_R, sigma, gamma, population), method='RK45', t_eval=t_eval)
                final_state = sol.y[:, -1]

            forecast_days = 70 + horizon
            t_forecast = np.arange(70, forecast_days + 1)
            if change_day is not None:
                beta_forecast = best_R_post * gamma
                sol_forecast = solve_ivp(seir_ode, (70, forecast_days), final_state2, args=(beta_forecast, sigma, gamma, population), method='RK45', t_eval=t_forecast)
            else:
                beta_forecast = best_R0 * gamma
                sol_forecast = solve_ivp(seir_ode, (70, forecast_days), final_state, args=(beta_forecast, sigma, gamma, population), method='RK45', t_eval=t_forecast)

            forecast = []
            for d in range(71, forecast_days + 1):
                start_t = d - 1
                end_t = d
                indices = np.where((sol_forecast.t >= start_t) & (sol_forecast.t <= end_t))[0]
                if len(indices) == 0:
                    expected_cases = 0
                else:
                    t_vals = sol_forecast.t[indices]
                    e_vals = sol_forecast.y[1][indices]
                    integral = sigma * np.trapz(e_vals, t_vals)
                    expected_cases = integral
                forecast.append(expected_cases)
            best_forecast = forecast

    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": best_forecast
    }