import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period

    def simulate_seir(R0, R_post, change_day):
        def seir_model(t, y):
            S, E, I, R = y
            if change_day is None:
                beta = R0 * gamma
            else:
                if t < change_day:
                    beta = R0 * gamma
                else:
                    beta = R_post * gamma
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dI_dt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dI_dt, dRdt]
        S0 = population - initial_infectious
        E0 = 0.0
        I0 = initial_infectious
        R0_val = 0.0
        sol = solve_ivp(
            fun=seir_model,
            t_span=(0, 70),
            y0=[S0, E0, I0, R0_val],
            t_eval=np.arange(1, 71),
            method='RK45'
        )
        predicted_cases = []
        for d in range(1, 71):
            if d == 1:
                prev_I = I0
            else:
                prev_I = sol.y[2][d-2]
            curr_I = sol.y[2][d-1]
            predicted_cases.append(curr_I - prev_I)
        return np.array(predicted_cases)

    def log_likelihood(params):
        R0, R_post, change_day = params
        predicted_cases = simulate_seir(R0, R_post, change_day)
        epsilon = 1e-10
        predicted_cases = np.maximum(predicted_cases, epsilon)
        nll = 0.0
        for d in range(70):
            observed = cases[d]
            predicted = predicted_cases[d]
            nll += observed * np.log(predicted) - predicted
        return nll

    R0_guess = 3.0
    R_post_guess = 1.5
    change_day_guess = 37.0

    R0_low = 0.1
    R0_high = 100.0
    R_post_low = 0.1
    R_post_high = 100.0

    res = minimize(
        fun=log_likelihood,
        x0=[R0_guess, R_post_guess, change_day_guess],
        bounds=[(R0_low, R0_high), (R_post_low, R_post_high), (0, 70)],
        method='L-BFGS-B'
    )

    R0_opt = res.x[0]
    R_post_opt = res.x[1]
    change_day_opt = res.x[2]

    if abs(R_post_opt - R0_opt) < 1e-3:
        change_day_opt = None

    def forecast_model(R0, R_post, change_day):
        def seir_model(t, y):
            S, E, I, R = y
            if change_day is None:
                beta = R0 * gamma
            else:
                if t < change_day:
                    beta = R0 * gamma
                else:
                    beta = R_post * gamma
            dSdt = -beta * S * I / population
            dEdt = beta * S * I / population - sigma * E
            dI_dt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dI_dt, dRdt]
        S0 = population - initial_infectious
        E0 = 0.0
        I0 = initial_infectious
        R0_val = 0.0
        sol = solve_ivp(
            fun=seir_model,
            t_span=(0, 70 + horizon),
            y0=[S0, E0, I0, R0_val],
            t_eval=np.arange(1, 70 + horizon + 1),
            method='RK45'
        )
        forecast_cases = []
        for d in range(71, 70 + horizon + 1):
            if d == 71:
                prev_I = I0
            else:
                prev_I = sol.y[2][d-2]
            curr_I = sol.y[2][d-1]
            forecast_cases.append(curr_I - prev_I)
        return forecast_cases

    forecast_cases = forecast_model(R0_opt, R_post_opt, change_day_opt)
    forecast = forecast_cases[:horizon]

    return {
        "R0": R0_opt,
        "R_post": R_post_opt,
        "change_day": change_day_opt,
        "forecast": forecast
    }