import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    days = 70
    best_error = float('inf')
    best_R0 = 0.0
    best_R_post = 0.0
    best_change_day = None

    def simulate_model(R0, R_post, change_day):
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        predicted = []
        previous_I = I
        for day in range(1, 71):
            if day <= change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
            def seir_ode(t, y):
                S, E, I, R = y
                beta_val = beta
                dSdt = -beta_val * S * I / population
                dEdt = beta_val * S * I / population - sigma * E
                dI_dt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dI_dt, dRdt]
            y0 = [S, E, I, R]
            sol = solve_ivp(seir_ode, [day-1, day], y0, method='RK45', t_eval=[day-1, day])
            S, E, I, R = sol.y[:, -1]
            new_cases = I - previous_I
            predicted.append(new_cases)
            previous_I = I
        return predicted

    for change_day in range(0, days + 1):
        def objective(params):
            R0, R_post = params
            predicted = simulate_model(R0, R_post, change_day)
            error = np.sum((np.array(predicted) - np.array(cases)) ** 2)
            return error
        # Initial guess based on early cases
        early_cases = cases[:10]
        R0_initial = 2.0 * np.mean(early_cases) / (initial_infectious * 1.0)
        R_post_initial = R0_initial
        result = minimize(objective, [R0_initial, R_post_initial], method='L-BFGS-B', bounds=[(0.1, 10.0), (0.1, 10.0)])
        R0_opt, R_post_opt = result.x
        error = result.fun
        if error < best_error:
            best_error = error
            best_R0 = R0_opt
            best_R_post = R_post_opt
            best_change_day = change_day

    # Forecast
    def forecast_model(R0, R_post, change_day):
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        previous_I = I
        for day in range(1, 71):
            if day <= change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
            def seir_ode(t, y):
                S, E, I, R = y
                beta_val = beta
                dSdt = -beta_val * S * I / population
                dEdt = beta_val * S * I / population - sigma * E
                dI_dt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dI_dt, dRdt]
            y0 = [S, E, I, R]
            sol = solve_ivp(seir_ode, [day-1, day], y0, method='RK45', t_eval=[day-1, day])
            S, E, I, R = sol.y[:, -1]
            previous_I = I
        forecast = []
        for day in range(71, 70 + horizon + 1):
            if day <= change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
            def seir_ode(t, y):
                S, E, I, R = y
                beta_val = beta
                dSdt = -beta_val * S * I / population
                dEdt = beta_val * S * I / population - sigma * E
                dI_dt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dI_dt, dRdt]
            y0 = [S, E, I, R]
            sol = solve_ivp(seir_ode, [day-1, day], y0, method='RK45', t_eval=[day-1, day])
            S, E, I, R = sol.y[:, -1]
            new_cases = I - previous_I
            forecast.append(new_cases)
            previous_I = I
        return forecast

    forecast = forecast_model(best_R0, best_R_post, best_change_day)
    return {
        'R0': best_R0,
        'R_post': best_R_post,
        'change_day': best_change_day,
        'forecast': forecast
    }