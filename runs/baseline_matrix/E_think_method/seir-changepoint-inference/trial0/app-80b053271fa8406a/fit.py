import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    days = 70
    
    def seir_model(y, t, beta):
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    def compute_nll(expected, observed):
        nll = 0.0
        for e, o in zip(expected, observed):
            if e == 0:
                # Avoid log(0) by adding a small epsilon
                e = 1e-10
            nll += -o * np.log(e) + e
        return nll
    
    def simulate_seir(R0, R_post, change_day):
        if change_day is None:
            beta = R0 * gamma
            t = np.linspace(0, days, days + 1)
            y0 = [population - initial_infectious, 0.0, initial_infectious, 0.0]
            sol = integrate.odeint(seir_model, y0, t, args=(beta,))
        else:
            t1 = np.linspace(0, change_day, int(change_day) + 1)
            t2 = np.linspace(change_day, days, days - int(change_day) + 1)
            t1 = t1[:int(change_day) + 1]
            t2 = t2[:days - int(change_day) + 1]
            
            # Simulate before change_day
            beta1 = R0 * gamma
            y0 = [population - initial_infectious, 0.0, initial_infectious, 0.0]
            sol1 = integrate.odeint(seir_model, y0, t1, args=(beta1,))
            
            # Simulate after change_day
            beta2 = R_post * gamma
            sol2 = integrate.odeint(seir_model, sol1[-1], t2, args=(beta2,))
            
            # Combine solutions
            sol = np.vstack((sol1, sol2))
            t = np.concatenate((t1, t2))
        
        # Compute cumulative incidence C(t)
        C = [0.0]
        for i in range(1, len(sol)):
            delta_t = t[i] - t[i-1]
            integral = (sol[i-1][1] + sol[i][1]) * delta_t / 2
            C.append(C[-1] + integral * sigma)
        
        # Expected cases for days 1 to 70
        expected = [C[d] - C[d-1] for d in range(1, days + 1)]
        return expected
    
    # No-intervention model
    def no_intervention_objective(R0):
        expected = simulate_seir(R0, R0, None)
        return compute_nll(expected, cases)
    
    result = minimize(no_intervention_objective, x0=[1.0], method='Nelder-Mead')
    best_r0_no_intervention = result.x[0]
    expected_no_intervention = simulate_seir(best_r0_no_intervention, best_r0_no_intervention, None)
    nll_no_intervention = compute_nll(expected_no_intervention, cases)
    
    # Change-point models
    best_nll = float('inf')
    best_r0 = None
    best_r_post = None
    best_change_day = None
    
    for change_day in range(1, days + 1):
        def change_intervention_objective(params):
            R0, R_post = params
            expected = simulate_seir(R0, R_post, change_day)
            return compute_nll(expected, cases)
        
        result = minimize(change_intervention_objective, x0=[1.0, 0.5], method='Nelder-Mead')
        R0_opt, R_post_opt = result.x
        expected = simulate_seir(R0_opt, R_post_opt, change_day)
        nll = compute_nll(expected, cases)
        
        if nll < best_nll:
            best_nll = nll
            best_r0 = R0_opt
            best_r_post = R_post_opt
            best_change_day = change_day
    
    # BIC comparison
    n = len(cases)
    bic_no_intervention = 2 * nll_no_intervention + 1 * np.log(n)
    bic_change_point = 2 * best_nll + 2 * np.log(n)
    
    if bic_change_point < bic_no_intervention:
        selected_r0 = best_r0
        selected_r_post = best_r_post
        selected_change_day = best_change_day
    else:
        selected_r0 = best_r0_no_intervention
        selected_r_post = selected_r0
        selected_change_day = None
    
    # Forecast
    days_forecast = days + horizon
    t_forecast = np.linspace(0, days_forecast, days_forecast + 1)
    
    if selected_change_day is None:
        beta = selected_r0 * gamma
        y0 = [population - initial_infectious, 0.0, initial_infectious, 0.0]
        sol = integrate.odeint(seir_model, y0, t_forecast, args=(beta,))
    else:
        t1 = np.linspace(0, selected_change_day, int(selected_change_day) + 1)
        t2 = np.linspace(selected_change_day, days_forecast, days_forecast - int(selected_change_day) + 1)
        t1 = t1[:int(selected_change_day) + 1]
        t2 = t2[:days_forecast - int(selected_change_day) + 1]
        
        # Simulate before change_day
        beta1 = selected_r0 * gamma
        y0 = [population - initial_infectious, 0.0, initial_infectious, 0.0]
        sol1 = integrate.odeint(seir_model, y0, t1, args=(beta1,))
        
        # Simulate after change_day
        beta2 = selected_r_post * gamma
        sol2 = integrate.odeint(seir_model, sol1[-1], t2, args=(beta2,))
        
        # Combine solutions
        sol = np.vstack((sol1, sol2))
        t_forecast = np.concatenate((t1, t2))
    
    # Compute cumulative incidence C(t) for forecast
    C = [0.0]
    for i in range(1, len(sol)):
        delta_t = t_forecast[i] - t_forecast[i-1]
        integral = (sol[i-1][1] + sol[i][1]) * delta_t / 2
        C.append(C[-1] + integral * sigma)
    
    # Forecast cases for days 71 to 70 + horizon
    forecast = [C[d] - C[d-1] for d in range(days + 1, days_forecast + 1)]
    
    return {
        "R0": float(selected_r0),
        "R_post": float(selected_r_post),
        "change_day": float(selected_change_day) if selected_change_day is not None else None,
        "forecast": forecast
    }
