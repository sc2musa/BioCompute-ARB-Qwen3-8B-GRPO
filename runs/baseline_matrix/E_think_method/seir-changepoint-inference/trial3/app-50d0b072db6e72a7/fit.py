import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0
    y0 = [S0, E0, I0, R0]
    t = np.arange(0, 71)
    
    def seir_model(y, t, R0, R_post, change_day, sigma, gamma, population):
        beta = R0 * gamma if t < change_day else R_post * gamma
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    def nll_no_intervention(params, cases, t, sigma, gamma, population, y0):
        R = params[0]
        sol = odeint(lambda y, t: seir_model(y, t, R, R, 0, sigma, gamma, population), y0, t)
        expected_cases = np.zeros(len(cases))
        for i in range(len(cases)):
            expected = sigma * sol[i+1][1]
            if expected <= 0:
                expected = 1e-10
            expected_cases[i] = expected
        nll = 0
        for i in range(len(cases)):
            nll += cases[i] * np.log(expected_cases[i]) - expected_cases[i]
        return nll
    
    initial_guess_no_intervention = [2.0]
    bounds_no_intervention = [(1e-6, None)]
    result_no_intervention = minimize(nll_no_intervention, initial_guess_no_intervention, args=(cases, t, sigma, gamma, population, y0), bounds=bounds_no_intervention, method='L-BFGS-B')
    R0_no_intervention = result_no_intervention.x[0]
    nll_no_intervention_val = result_no_intervention.fun
    
    n = len(cases)
    bic_no_intervention = 2 * nll_no_intervention_val + 1 * np.log(n)
    
    best_nll = float('inf')
    best_R0 = 0
    best_R_post = 0
    best_change_day = None
    best_model = 'no_intervention'
    
    for change_day_candidate in range(1, 71):
        def nll_intervention(params, cases, t, sigma, gamma, population, y0, change_day):
            R0, R_post = params
            t1 = np.arange(0, change_day_candidate + 1)
            sol1 = odeint(lambda y, t: seir_model(y, t, R0, R_post, change_day_candidate, sigma, gamma, population), y0, t1)
            y_initial = sol1[-1]
            t2 = np.arange(change_day_candidate, 71)
            sol2 = odeint(lambda y, t: seir_model(y, t, R0, R_post, change_day_candidate, sigma, gamma, population), y_initial, t2)
            expected_cases = np.zeros(len(cases))
            for i in range(len(cases)):
                d = i + 1
                if d <= change_day_candidate:
                    E = sol1[d][1]
                else:
                    E = sol2[d - change_day_candidate][1]
                expected = sigma * E
                if expected <= 0:
                    expected = 1e-10
                expected_cases[i] = expected
            nll = 0
            for i in range(len(cases)):
                nll += cases[i] * np.log(expected_cases[i]) - expected_cases[i]
            return nll
        
        initial_guess_intervention = [R0_no_intervention, 0.5 * R0_no_intervention]
        bounds_intervention = [(1e-6, None), (1e-6, None)]
        result_intervention = minimize(nll_intervention, initial_guess_intervention, args=(cases, t, sigma, gamma, population, y0, change_day_candidate), bounds=bounds_intervention, method='L-BFGS-B')
        nll_intervention_val = result_intervention.fun
        
        k_intervention = 2
        bic_intervention = 2 * nll_intervention_val + k_intervention * np.log(n)
        
        if bic_intervention < bic_no_intervention:
            R0_candidate = result_intervention.x[0]
            R_post_candidate = result_intervention.x[1]
            if R_post_candidate <= 0.75 * R0_candidate:
                if nll_intervention_val < best_nll:
                    best_nll = nll_intervention_val
                    best_R0 = R0_candidate
                    best_R_post = R_post_candidate
                    best_change_day = change_day_candidate
                    best_model = 'intervention'
    
    if best_model == 'intervention':
        R0 = best_R0
        R_post = best_R_post
        change_day = best_change_day
    else:
        R0 = R0_no_intervention
        R_post = R0
        change_day = None
    
    if change_day is None:
        t_forecast = np.arange(0, 70 + horizon + 1)
        sol_forecast = odeint(lambda y, t: seir_model(y, t, R0, R_post, 0, sigma, gamma, population), y0, t_forecast)
        forecast = [sigma * sol_forecast[d][1] for d in range(70, 70 + horizon + 1)]
    else:
        t1 = np.arange(0, change_day + 1)
        sol1 = odeint(lambda y, t: seir_model(y, t, R0, R_post, change_day, sigma, gamma, population), y0, t1)
        y_initial = sol1[-1]
        t2 = np.arange(change_day, 70 + horizon + 1)
        sol2 = odeint(lambda y, t: seir_model(y, t, R0, R_post, change_day, sigma, gamma, population), y_initial, t2)
        forecast = []
        for d in range(70, 70 + horizon + 1):
            if d >= change_day:
                index = d - change_day
                E = sol2[index][1]
            else:
                E = sol1[d][1]
            expected = sigma * E
            if expected <= 0:
                expected = 1e-10
            forecast.append(expected)
    
    return {
        "R0": float(R0),
        "R_post": float(R_post),
        "change_day": float(change_day) if change_day is not None else None,
        "forecast": forecast
    }
