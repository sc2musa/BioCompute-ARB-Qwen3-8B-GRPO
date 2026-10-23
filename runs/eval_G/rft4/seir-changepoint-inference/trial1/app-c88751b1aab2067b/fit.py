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

    best_nll = float('inf')
    best_R0 = None
    best_R_post = None
    best_change_day = None

    # No intervention case
    def objective_no_intervention(R0_candidate):
        def ode(t, y):
            S, E, I, R = y
            beta = R0_candidate * gamma
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dI_dt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dI_dt, dRdt]
        t_eval = np.arange(1, 71)
        sol = solve_ivp(ode, [0, 70], y0, t_eval=t_eval, method='RK45')
        E_values = sol.y[1]
        predicted_cases = sigma * E_values * 1.0
        nll = 0.0
        for d in range(len(t_eval)):
            observed = cases[d]
            predicted = predicted_cases[d]
            if predicted == 0:
                predicted = 1e-10
            nll += -observed * np.log(predicted) + predicted
        return nll

    # Use Nelder-Mead with scalar x0
    result_no_intervention = minimize(objective_no_intervention, x0=1.0, method='Nelder-Mead')
    R0_no_intervention = result_no_intervention.x[0]
    nll_no_intervention = objective_no_intervention(R0_no_intervention)

    # Grid search for change_day
    for change_day in range(1, 71):
        first_period_cases = cases[:change_day]
        second_period_cases = cases[change_day:]

        # Optimize R0 for first period
        def objective_R0(R0_candidate):
            def ode_first(t, y):
                S, E, I, R = y
                beta = R0_candidate * gamma
                dSdt = -beta * S * I / population
                dEdt = beta * S * I / population - sigma * E
                dI_dt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dI_dt, dRdt]
            t_eval = np.arange(1, change_day + 1)
            sol = solve_ivp(ode_first, [0, change_day], y0, t_eval=t_eval, method='RK45')
            E_values = sol.y[1]
            predicted_cases = sigma * E_values * 1.0
            nll = 0.0
            for d in range(len(t_eval)):
                observed = first_period_cases[d]
                predicted = predicted_cases[d]
                if predicted == 0:
                    predicted = 1e-10
                nll += -observed * np.log(predicted) + predicted
            return nll

        result_R0 = minimize(objective_R0, x0=1.0, method='Nelder-Mead')
        R0_opt = result_R0.x[0]
        nll_R0 = objective_R0(R0_opt)

        # Integrate first period with R0_opt
        def ode_first(t, y):
            S, E, I, R = y
            beta = R0_opt * gamma
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dI_dt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dI_dt, dRdt]
        t_eval = np.arange(1, change_day + 1)
        sol_first = solve_ivp(ode_first, [0, change_day], y0, t_eval=t_eval, method='RK45')
        y_initial_second = sol_first.y[:, -1]

        # Optimize R_post for second period
        def objective_R_post(R_post_candidate):
            def ode_second(t, y):
                S, E, I, R = y
                beta = R_post_candidate * gamma
                dSdt = -beta * S * I / population
                dEdt = beta * S * I / population - sigma * E
                dI_dt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dI_dt, dRdt]
            t_eval_second = np.arange(change_day + 1, 71)
            sol_second = solve_ivp(ode_second, [change_day, 70], y_initial_second, t_eval=t_eval_second, method='RK45')
            E_values = sol_second.y[1]
            predicted_cases = sigma * E_values * 1.0
            nll = 0.0
            for d in range(len(t_eval_second)):
                observed = second_period_cases[d]
                predicted = predicted_cases[d]
                if predicted == 0:
                    predicted = 1e-10
                nll += -observed * np.log(predicted) + predicted
            return nll

        result_R_post = minimize(objective_R_post, x0=1.0, method='Nelder-Mead')
        R_post_opt = result_R_post.x[0]
        nll_R_post = objective_R_post(R_post_opt)

        # Total NLL
        total_nll = nll_R0 + nll_R_post
        if total_nll < best_nll:
            best_nll = total_nll
            best_R0 = R0_opt
            best_R_post = R_post_opt
            best_change_day = change_day

    # Compare with no intervention case
    if nll_no_intervention < best_nll:
        best_R0 = R0_no_intervention
        best_R_post = best_R0
        best_change_day = None

    # Forecast
    def forecast_model(R0, R_post, change_day):
        # Integrate up to day 70
        if change_day is None:
            def ode(t, y):
                S, E, I, R = y
                beta = R0 * gamma
                dSdt = -beta * S * I / population
                dEdt = beta * S * I / population - sigma * E
                dI_dt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dI_dt, dRdt]
            t_eval = np.arange(1, 71)
            sol = solve_ivp(ode, [0, 70], y0, t_eval=t_eval, method='RK45')
            y_final = sol.y[:, -1]
        else:
            # First period
            def ode_first(t, y):
                S, E, I, R = y
                beta = R0 * gamma
                dSdt = -beta * S * I / population
                dEdt = beta * S * I / population - sigma * E
                dI_dt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dI_dt, dRdt]
            t_eval_first = np.arange(1, change_day + 1)
            sol_first = solve_ivp(ode_first, [0, change_day], y0, t_eval=t_eval_first, method='RK45')
            y_initial_second = sol_first.y[:, -1]
            
            # Second period
            def ode_second(t, y):
                S, E, I, R = y
                beta = R_post * gamma
                dSdt = -beta * S * I / population
                dEdt = beta * S * I / population - sigma * E
                dI_dt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dI_dt, dRdt]
            t_eval_second = np.arange(change_day + 1, 71)
            sol_second = solve_ivp(ode_second, [change_day, 70], y_initial_second, t_eval=t_eval_second, method='RK45')
            y_final = sol_second.y[:, -1]

        # Forecast from day 71 to 70 + horizon
        if change_day is None:
            def ode_forecast(t, y):
                S, E, I, R = y
                beta = R0 * gamma
                dSdt = -beta * S * I / population
                dEdt = beta * S * I / population - sigma * E
                dI_dt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dI_dt, dRdt]
            t_eval_forecast = np.arange(71, 70 + horizon + 1)
            sol_forecast = solve_ivp(ode_forecast, [70, 70 + horizon], y_final, t_eval=t_eval_forecast, method='RK45')
            E_forecast = sol_forecast.y[1]
            predicted_forecast = sigma * E_forecast * 1.0
        else:
            # Forecast after intervention
            def ode_forecast(t, y):
                S, E, I, R = y
                beta = R_post * gamma
                dSdt = -beta * S * I / population
                dEdt = beta * S * I / population - sigma * E
                dI_dt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dI_dt, dRdt]
            t_eval_forecast = np.arange(71, 70 + horizon + 1)
            sol_forecast = solve_ivp(ode_forecast, [70, 70 + horizon], y_final, t_eval=t_eval_forecast, method='RK45')
            E_forecast = sol_forecast.y[1]
            predicted_forecast = sigma * E_forecast * 1.0

        return predicted_forecast.tolist()

    forecast = forecast_model(best_R0, best_R_post, best_change_day)

    return {
        "R0": float(best_R0),
        "R_post": float(best_R_post),
        "change_day": best_change_day if best_change_day is not None else None,
        "forecast": forecast
    }