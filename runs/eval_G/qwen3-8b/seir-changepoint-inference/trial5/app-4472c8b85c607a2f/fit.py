import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    # No intervention case
    def error_no_intervention(R0):
        beta = R0 * gamma
        t = np.arange(0, 71)
        initial_state = [population - initial_infectious, 0, initial_infectious, 0]
        def model(t, state):
            S, E, I, R = state
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        sol = odeint(model, initial_state, t)
        I_total = sol[:, 2]
        new_cases_sim = np.zeros(len(I_total))
        for d in range(1, len(I_total)):
            new_cases_sim[d] = I_total[d] - I_total[d-1]
        error = np.sum((new_cases_sim[1:71] - cases) ** 2)
        return error

    res_no_intervention = minimize(error_no_intervention, x0=1.0, method='L-BFGS-B', bounds=[(0.1, 10.0)])
    best_R0_no_intervention = res_no_intervention.x[0]
    best_error_no_intervention = res_no_intervention.fun

    best_error = best_error_no_intervention
    best_R0 = best_R0_no_intervention
    best_R_post = best_R0_no_intervention
    best_change_day = None

    for change_day in range(0, 71):
        def error_with_intervention(params):
            R0, R_post = params
            beta_initial = R0 * gamma
            beta_post = R_post * gamma
            t1 = np.arange(0, change_day + 1)
            initial_state = [population - initial_infectious, 0, initial_infectious, 0]
            def model1(t, state):
                S, E, I, R = state
                dSdt = -beta_initial * S * I / population
                dEdt = beta_initial * S * I / population - sigma * E
                dIdt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dIdt, dRdt]
            sol1 = odeint(model1, initial_state, t1)
            I1 = sol1[:, 2]
            t2 = np.arange(change_day, 71)
            initial_state2 = sol1[-1, :]
            def model2(t, state):
                S, E, I, R = state
                dSdt = -beta_post * S * I / population
                dEdt = beta_post * S * I / population - sigma * E
                dIdt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dIdt, dRdt]
            sol2 = odeint(model2, initial_state2, t2)
            I_total = np.concatenate((I1, sol2[:, 2]))
            new_cases_sim = np.zeros(len(I_total))
            for d in range(1, len(I_total)):
                new_cases_sim[d] = I_total[d] - I_total[d-1]
            error = np.sum((new_cases_sim[1:71] - cases) ** 2)
            return error

        res = minimize(error_with_intervention, x0=[best_R0_no_intervention, best_R0_no_intervention], method='L-BFGS-B', bounds=[(0.1, 10.0), (0.1, 10.0)])
        current_error = res.fun
        current_R0 = res.x[0]
        current_R_post = res.x[1]

        if current_error < best_error:
            best_error = current_error
            best_R0 = current_R0
            best_R_post = current_R_post
            best_change_day = change_day

    if best_change_day is not None:
        if best_R_post <= 0.75 * best_R0:
            pass
        else:
            best_change_day = None
            best_R_post = best_R0

    if best_change_day is not None:
        R0_best = best_R0
        R_post_best = best_R_post
        beta_initial = R0_best * gamma
        beta_post = R_post_best * gamma
        t1 = np.arange(0, best_change_day + 1)
        initial_state = [population - initial_infectious, 0, initial_infectious, 0]
        def model1(t, state):
            S, E, I, R = state
            dSdt = -beta_initial * S * I / population
            dEdt = beta_initial * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        sol1 = odeint(model1, initial_state, t1)
        I1 = sol1[:, 2]
        t2 = np.arange(best_change_day, 71)
        initial_state2 = sol1[-1, :]
        def model2(t, state):
            S, E, I, R = state
            dSdt = -beta_post * S * I / population
            dEdt = beta_post * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        sol2 = odeint(model2, initial_state2, t2)
        I_total = np.concatenate((I1, sol2[:, 2]))
        forecast_days = horizon
        t_forecast = np.arange(70, 70 + forecast_days + 1)
        initial_state_forecast = sol2[-1, :]
        def model_forecast(t, state):
            S, E, I, R = state
            dSdt = -beta_post * S * I / population
            dEdt = beta_post * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        sol_forecast = odeint(model_forecast, initial_state_forecast, t_forecast)
        forecast_new_cases = np.zeros(len(sol_forecast))
        for d in range(1, len(sol_forecast)):
            forecast_new_cases[d] = sol_forecast[d, 2] - sol_forecast[d-1, 2]
        forecast = forecast_new_cases[1:].tolist()
    else:
        R0_best = best_R0
        beta = R0_best * gamma
        t = np.arange(0, 71)
        initial_state = [population - initial_infectious, 0, initial_infectious, 0]
        def model(t, state):
            S, E, I, R = state
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        sol = odeint(model, initial_state, t)
        I_total = sol[:, 2]
        forecast_days = horizon
        t_forecast = np.arange(70, 70 + forecast_days + 1)
        initial_state_forecast = sol[-1, :]
        def model_forecast(t, state):
            S, E, I, R = state
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dIdt, dRdt]
        sol_forecast = odeint(model_forecast, initial_state_forecast, t_forecast)
        forecast_new_cases = np.zeros(len(sol_forecast))
        for d in range(1, len(sol_forecast)):
            forecast_new_cases[d] = sol_forecast[d, 2] - sol_forecast[d-1, 2]
        forecast = forecast_new_cases[1:].tolist()

    return {
        "R0": best_R0,
        "R_post": best_R_post if best_change_day is not None else best_R0,
        "change_day": best_change_day if best_change_day is not None else None,
        "forecast": forecast
    }