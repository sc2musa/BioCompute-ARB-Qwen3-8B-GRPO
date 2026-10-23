import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    t = np.arange(0, 71)
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    y0 = [S0, E0, I0, R0]

    best_model = None
    best_bic = np.inf

    # No intervention case
    def objective_no_intervention(params):
        R0 = params[0]
        def model(t, y):
            S, E, I, R = y
            beta = R0 * gamma
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        y_int = odeint(model, y0, t)
        E_values = y_int[:, 1]
        expected_cases = sigma * E_values
        nll = 0.0
        for i in range(len(cases)):
            if expected_cases[i] < 1e-10:
                expected_cases[i] = 1e-10
            nll += cases[i] * np.log(expected_cases[i]) - expected_cases[i]
        return nll

    result_no_intervention = minimize(objective_no_intervention, x0=[1.0], method='Nelder-Mead')
    R0_no_intervention = result_no_intervention.x[0]
    nll_no_intervention = objective_no_intervention([R0_no_intervention])
    bic_no_intervention = 2 * nll_no_intervention + 1 * np.log(len(cases))

    if bic_no_intervention < best_bic:
        best_model = {
            'R0': R0_no_intervention,
            'R_post': R0_no_intervention,
            'change_day': None,
            'nll': nll_no_intervention,
        }
        best_bic = bic_no_intervention

    # Intervention cases
    for change_day in range(0, 71):
        def objective_change_day(params):
            R0, R_post = params
            def model(t, y):
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
            y_int = odeint(model, y0, t)
            E_values = y_int[:, 1]
            expected_cases = sigma * E_values
            nll = 0.0
            for i in range(len(cases)):
                if expected_cases[i] < 1e-10:
                    expected_cases[i] = 1e-10
                nll += cases[i] * np.log(expected_cases[i]) - expected_cases[i]
            return nll

        result_change_day = minimize(objective_change_day, x0=[2.0, 1.0], method='Nelder-Mead')
        R0_change_day = result_change_day.x[0]
        R_post_change_day = result_change_day.x[1]
        nll_change_day = objective_change_day([R0_change_day, R_post_change_day])
        bic_change_day = 2 * nll_change_day + 3 * np.log(len(cases))

        if bic_change_day < best_bic:
            best_model = {
                'R0': R0_change_day,
                'R_post': R_post_change_day,
                'change_day': change_day,
                'nll': nll_change_day,
            }
            best_bic = bic_change_day

    selected_model = best_model

    # Forecast
    if selected_model['change_day'] is None:
        R0_forecast = selected_model['R0']
        R_post_forecast = R0_forecast
        def model_forecast(t, y):
            S, E, I, R = y
            beta = R0_forecast * gamma
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        y_int_forecast = odeint(model_forecast, y0, t)
        y_initial_forecast = y_int_forecast[-1]
        t_forecast = np.arange(70, 70 + horizon + 1)
        y_int_forecast_future = odeint(model_forecast, y_initial_forecast, t_forecast)
        E_values_forecast = y_int_forecast_future[:, 1]
        expected_cases_forecast = sigma * E_values_forecast
        forecast = expected_cases_forecast.tolist()
    else:
        R0_forecast = selected_model['R0']
        R_post_forecast = selected_model['R_post']
        change_day_forecast = selected_model['change_day']
        def model_forecast(t, y):
            S, E, I, R = y
            if t < change_day_forecast:
                beta = R0_forecast * gamma
            else:
                beta = R_post_forecast * gamma
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        y_int_forecast = odeint(model_forecast, y0, t)
        y_initial_forecast = y_int_forecast[-1]
        t_forecast = np.arange(70, 70 + horizon + 1)
        y_int_forecast_future = odeint(model_forecast, y_initial_forecast, t_forecast)
        E_values_forecast = y_int_forecast_future[:, 1]
        expected_cases_forecast = sigma * E_values_forecast
        forecast = expected_cases_forecast.tolist()

    return {
        "R0": selected_model['R0'],
        "R_post": selected_model['R_post'],
        "change_day": selected_model['change_day'],
        "forecast": forecast,
    }