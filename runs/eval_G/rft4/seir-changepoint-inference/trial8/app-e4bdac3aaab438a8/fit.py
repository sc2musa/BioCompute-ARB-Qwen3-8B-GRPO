import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    gamma = 1 / infectious_period
    sigma = 1 / latent_period

    # No-intervention model
    def error_no_intervention(params):
        R0 = params[0]
        def seir_ode_no_intervention(t, y, R0, gamma, sigma, population):
            S, E, I, R = y
            beta_t = R0 * gamma
            dSdt = -beta_t * S * I / population
            dEdt = beta_t * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        sol = solve_ivp(seir_ode_no_intervention, [0, 70], [population - initial_infectious, 0, initial_infectious, 0], args=(R0, gamma, sigma, population), method='RK45', t_eval=np.arange(0, 71))
        E_values = sol.y[1]
        predicted = []
        for d in range(1, 71):
            if d == 1:
                E_start = E_values[0]
                E_end = E_values[1]
            else:
                E_start = E_values[d-1]
                E_end = E_values[d]
            integral = (E_start + E_end) * sigma / 2
            predicted.append(integral)
        error = np.sum((np.array(predicted) - np.array(cases)) ** 2)
        return error

    result_no_intervention = minimize(error_no_intervention, x0=[1.0], method='L-BFGS-B')
    R0_no_intervention = result_no_intervention.x[0]
    error_no_intervention_val = result_no_intervention.fun

    # Intervention model
    best_error = float('inf')
    best_params = None
    best_change_day = None

    # Sample change_days every 5 days to reduce computation
    for change_day in range(1, 71, 5):
        def error_intervention(params):
            R0, R_post = params
            def seir_ode_intervention(t, y, R0, R_post, change_day, gamma, sigma, population):
                S, E, I, R = y
                if t < change_day:
                    beta_t = R0 * gamma
                else:
                    beta_t = R_post * gamma
                dSdt = -beta_t * S * I / population
                dEdt = beta_t * S * I / population - sigma * E
                dIdt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dIdt, dRdt]
            sol = solve_ivp(seir_ode_intervention, [0, 70], [population - initial_infectious, 0, initial_infectious, 0], args=(R0, R_post, change_day, gamma, sigma, population), method='RK45', t_eval=np.arange(0, 71))
            E_values = sol.y[1]
            predicted = []
            for d in range(1, 71):
                if d == 1:
                    E_start = E_values[0]
                    E_end = E_values[1]
                else:
                    E_start = E_values[d-1]
                    E_end = E_values[d]
                integral = (E_start + E_end) * sigma / 2
                predicted.append(integral)
            error = np.sum((np.array(predicted) - np.array(cases)) ** 2)
            return error
        
        result_intervention = minimize(error_intervention, x0=[R0_no_intervention, R0_no_intervention], method='Nelder-Mead')
        R0_intervention = result_intervention.x[0]
        R_post_intervention = result_intervention.x[1]
        error_intervention_val = result_intervention.fun

        if error_intervention_val < best_error:
            best_error = error_intervention_val
            best_params = (R0_intervention, R_post_intervention, change_day)

    if best_error < error_no_intervention_val:
        R0, R_post, change_day = best_params
    else:
        R0 = R0_no_intervention
        R_post = R0
        change_day = None

    # Forecast
    forecast_end = 70 + horizon
    def seir_ode_forecast(t, y, R0, R_post, change_day, gamma, sigma, population):
        S, E, I, R = y
        if t < change_day:
            beta_t = R0 * gamma
        else:
            beta_t = R_post * gamma
        dSdt = -beta_t * S * I / population
        dEdt = beta_t * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    if change_day is None:
        sol_forecast = solve_ivp(seir_ode_forecast, [0, forecast_end], [population - initial_infectious, 0, initial_infectious, 0], args=(R0, R0, 71, gamma, sigma, population), method='RK45', t_eval=np.arange(0, forecast_end + 1))
        E_values_forecast = sol_forecast.y[1]
        forecast = []
        for d in range(71, forecast_end + 1):
            if d == 71:
                E_start = E_values_forecast[70]
                E_end = E_values_forecast[71]
            else:
                E_start = E_values_forecast[d-1]
                E_end = E_values_forecast[d]
            integral = (E_start + E_end) * sigma / 2
            forecast.append(integral)
    else:
        sol_forecast = solve_ivp(seir_ode_forecast, [0, forecast_end], [population - initial_infectious, 0, initial_infectious, 0], args=(R0, R_post, change_day, gamma, sigma, population), method='RK45', t_eval=np.arange(0, forecast_end + 1))
        E_values_forecast = sol_forecast.y[1]
        forecast = []
        for d in range(71, forecast_end + 1):
            if d == 71:
                E_start = E_values_forecast[70]
                E_end = E_values_forecast[71]
            else:
                E_start = E_values_forecast[d-1]
                E_end = E_values_forecast[d]
            integral = (E_start + E_end) * sigma / 2
            forecast.append(integral)

    return {
        "R0": float(R0),
        "R_post": float(R_post),
        "change_day": float(change_day) if change_day is not None else None,
        "forecast": forecast
    }