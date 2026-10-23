import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import scipy.stats as stats

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    best_model = None
    best_likelihood = -np.inf

    # No intervention case
    def neg_log_likelihood_R0(params):
        R0 = params[0]
        beta = R0 * gamma
        
        def seir_model(t, y):
            S, E, I, R = y
            dSdt = -beta * S * I
            dEdt = beta * S * I - sigma * E
            dI_dt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dI_dt, dRdt]
        
        t_span = (0, 70)
        y0 = [population - initial_infectious, 0, initial_infectious, 0]
        sol = solve_ivp(seir_model, t_span, y0, t_eval=np.arange(0, 71), method='BDF', rtol=1e-5, atol=1e-9)
        
        expected = []
        for d in range(1, 71):
            t_start = d - 1
            t_end = d
            idx_start = np.where(sol.t >= t_start)[0][0]
            idx_end = np.where(sol.t >= t_end)[0][0]
            E_start = sol.y[1, idx_start]
            E_end = sol.y[1, idx_end]
            expected_case = (E_start + E_end) / 2 * sigma * (t_end - t_start)
            expected.append(expected_case)
        
        return -np.sum(stats.poisson.logpmf(cases, expected))

    result = minimize(neg_log_likelihood_R0, x0=[1.0], method='L-BFGS-B', bounds=[(0.1, 10.0)])
    R0_no_intervention = result.x[0]
    likelihood_no_intervention = -result.fun

    if likelihood_no_intervention > best_likelihood:
        best_likelihood = likelihood_no_intervention
        best_model = {
            'R0': R0_no_intervention,
            'R_post': R0_no_intervention,
            'change_day': None,
            'forecast': []
        }

    # Intervention cases
    for t_c in range(1, 71, 10):  # Check every 10th day
        def neg_log_likelihood(params):
            R0, R_post = params
            beta_before = R0 * gamma
            beta_after = R_post * gamma
            
            def seir_model(t, y):
                S, E, I, R = y
                if t < t_c:
                    dSdt = -beta_before * S * I
                    dEdt = beta_before * S * I - sigma * E
                else:
                    dSdt = -beta_after * S * I
                    dEdt = beta_after * S * I - sigma * E
                dI_dt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dI_dt, dRdt]
            
            t_span1 = (0, t_c)
            y0 = [population - initial_infectious, 0, initial_infectious, 0]
            sol1 = solve_ivp(seir_model, t_span1, y0, t_eval=np.arange(0, t_c + 1), method='BDF', rtol=1e-5, atol=1e-9)
            
            t_span2 = (t_c, 70)
            y0_2 = sol1.y[:, -1]
            sol2 = solve_ivp(seir_model, t_span2, y0_2, t_eval=np.arange(t_c, 71), method='BDF', rtol=1e-5, atol=1e-9)
            
            t_all = np.concatenate([sol1.t, sol2.t])
            y_all = np.concatenate([sol1.y, sol2.y], axis=1)
            
            expected = []
            for d in range(1, 71):
                t_start = d - 1
                t_end = d
                idx_start = np.where(t_all >= t_start)[0][0]
                idx_end = np.where(t_all >= t_end)[0][0]
                E_start = y_all[1, idx_start]
                E_end = y_all[1, idx_end]
                expected_case = (E_start + E_end) / 2 * sigma * (t_end - t_start)
                expected.append(expected_case)
            
            return -np.sum(stats.poisson.logpmf(cases, expected))

        result = minimize(neg_log_likelihood, x0=[1.0, 1.0], method='L-BFGS-B', bounds=[(0.1, 10.0), (0.1, 10.0)])
        R0 = result.x[0]
        R_post = result.x[1]
        likelihood = -result.fun
        
        if R_post < 0.75 * R0:
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_model = {
                    'R0': R0,
                    'R_post': R_post,
                    'change_day': t_c,
                    'forecast': []
                }

    # Forecast
    if best_model['change_day'] is None:
        beta = best_model['R0'] * gamma
        
        def seir_model(t, y):
            S, E, I, R = y
            dSdt = -beta * S * I
            dEdt = beta * S * I - sigma * E
            dI_dt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dI_dt, dRdt]
        
        t_span = (0, 70 + horizon)
        y0 = [population - initial_infectious, 0, initial_infectious, 0]
        sol = solve_ivp(seir_model, t_span, y0, t_eval=np.arange(0, 70 + horizon + 1), method='BDF', rtol=1e-5, atol=1e-9)
        
        forecast = []
        for d in range(71, 70 + horizon + 1):
            t_start = d - 1
            t_end = d
            idx_start = np.where(sol.t >= t_start)[0][0]
            idx_end = np.where(sol.t >= t_end)[0][0]
            E_start = sol.y[1, idx_start]
            E_end = sol.y[1, idx_end]
            expected_case = (E_start + E_end) / 2 * sigma * (t_end - t_start)
            forecast.append(expected_case)
    else:
        t_c = best_model['change_day']
        R0 = best_model['R0']
        R_post = best_model['R_post']
        beta_before = R0 * gamma
        beta_after = R_post * gamma
        
        def seir_model(t, y):
            S, E, I, R = y
            if t < t_c:
                dSdt = -beta_before * S * I
                dEdt = beta_before * S * I - sigma * E
            else:
                dSdt = -beta_after * S * I
                dEdt = beta_after * S * I - sigma * E
            dI_dt = sigma * E - gamma * I
            dRdt = gamma * I
            return [dSdt, dEdt, dI_dt, dRdt]
        
        t_span = (0, 70 + horizon)
        y0 = [population - initial_infectious, 0, initial_infectious, 0]
        sol = solve_ivp(seir_model, t_span, y0, t_eval=np.arange(0, 70 + horizon + 1), method='BDF', rtol=1e-5, atol=1e-9)
        
        forecast = []
        for d in range(71, 70 + horizon + 1):
            t_start = d - 1
            t_end = d
            idx_start = np.where(sol.t >= t_start)[0][0]
            idx_end = np.where(sol.t >= t_end)[0][0]
            E_start = sol.y[1, idx_start]
            E_end = sol.y[1, idx_end]
            expected_case = (E_start + E_end) / 2 * sigma * (t_end - t_start)
            forecast.append(expected_case)
    
    best_model['forecast'] = forecast
    return best_model
