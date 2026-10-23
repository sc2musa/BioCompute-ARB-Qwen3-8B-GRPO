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

    best_error = float('inf')
    best_R0 = 2.0
    best_R_post = 1.0
    best_change_day = None

    # No-intervention case
    def objective_no_intervention(params):
        R0 = params[0]
        def seir_ode(t, y):
            beta = R0 * gamma
            S, E, I, R = y
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        t_span = (0, 70)
        sol = solve_ivp(seir_ode, t_span, y0, t_eval=np.arange(0, 71))
        I_values = sol.y[2]
        predicted_cases = np.diff(I_values)
        error = np.sum((predicted_cases - cases)**2)
        return error
    result_no_intervention = minimize(objective_no_intervention, [2.0], method='L-BFGS-B', bounds=[(0.1, None)])
    if result_no_intervention.success:
        current_error = result_no_intervention.fun
        if current_error < best_error:
            best_error = current_error
            best_R0 = result_no_intervention.x[0]
            best_R_post = best_R0
            best_change_day = None

    # Change-day cases
    for change_day in range(1, 71, 5):  # Check every 5 days
        def objective(params):
            R0, R_post = params
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
            sol = solve_ivp(seir_ode, t_span, y0, t_eval=np.arange(0, 71))
            I_values = sol.y[2]
            predicted_cases = np.diff(I_values)
            error = np.sum((predicted_cases - cases)**2)
            return error
        initial_guess = [best_R0, best_R0 * 0.5]  # R_post <= R0
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=[(0.1, None), (0.1, best_R0)])
        if result.success:
            current_error = result.fun
            if current_error < best_error:
                best_error = current_error
                best_R0 = result.x[0]
                best_R_post = result.x[1]
                best_change_day = change_day

    # Compute state at day 70 using best parameters
    def forecast_ode(t, y):
        S, E, I, R = y
        beta = best_R0 * gamma if best_change_day is None else (best_R_post * gamma if 70 >= best_change_day else best_R0 * gamma)
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    t_span_forecast = (0, 70)
    sol_forecast = solve_ivp(forecast_ode, t_span_forecast, y0, t_eval=np.arange(0, 71))
    state_at_70 = sol_forecast.y[:, 70]

    # Forecast
    def forecast_ode_final(t, y):
        S, E, I, R = y
        beta = best_R0 * gamma if best_change_day is None else (best_R_post * gamma if 70 >= best_change_day else best_R0 * gamma)
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    t_span_forecast = (70, 70 + horizon)
    sol_forecast = solve_ivp(forecast_ode_final, t_span_forecast, state_at_70, t_eval=np.arange(70, 70 + horizon + 1))
    I_values_forecast = sol_forecast.y[2]
    predicted_forecast = np.diff(I_values_forecast)
    forecast = predicted_forecast.tolist()

    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": forecast
    }