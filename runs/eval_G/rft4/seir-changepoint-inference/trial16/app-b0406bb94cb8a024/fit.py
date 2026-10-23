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

    def seir_model(t, y, beta):
        S, E, I, R = y
        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dI_dt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dI_dt, dRdt]

    def compute_expected(beta, start_day, end_day):
        t_span = (0, end_day)
        sol = solve_ivp(lambda t, y: seir_model(t, y, beta), t_span, y0, t_eval=np.arange(0, end_day + 1))
        E = sol.y[1]
        expected = []
        for d in range(start_day, end_day + 1):
            expected_cases = sigma * (E[d-1] + E[d]) / 2
            expected.append(expected_cases)
        return np.array(expected)

    best_likelihood = -np.inf
    best_params = {}

    for change_day in range(1, 71):
        # First segment
        def objective_R0(R0_candidate):
            R0 = R0_candidate
            beta = R0 * gamma
            expected_first = compute_expected(beta, 1, change_day)
            observed = cases[:change_day]
            expected = np.maximum(expected_first, 1e-10)
            return -np.sum(expected - observed + np.log(expected))

        result_R0 = minimize(objective_R0, x0=1.0, method='Nelder-Mead', bounds=[(0.1, 10.0)])
        R0_opt = result_R0.x

        # Second segment
        def objective_Rpost(Rpost_candidate):
            Rpost = Rpost_candidate
            beta = Rpost * gamma
            expected_second = compute_expected(beta, change_day + 1, 70)
            observed = cases[change_day:]
            expected = np.maximum(expected_second, 1e-10)
            return -np.sum(expected - observed + np.log(expected))

        result_Rpost = minimize(objective_Rpost, x0=1.0, method='Nelder-Mead', bounds=[(0.1, 10.0)])
        Rpost_opt = result_Rpost.x

        # Total likelihood
        total_likelihood = -result_R0.fun - result_Rpost.fun

        if total_likelihood > best_likelihood:
            best_likelihood = total_likelihood
            best_params = {
                'R0': R0_opt,
                'R_post': Rpost_opt,
                'change_day': change_day
            }

    # Handle no intervention case
    if best_params['R_post'] >= 0.75 * best_params['R0']:
        best_params['change_day'] = None
        best_params['R_post'] = best_params['R0']

    # Simulate full 70 days with best parameters to get final state
    if best_params['change_day'] is None:
        beta_full = best_params['R0'] * gamma
        sol_full = solve_ivp(lambda t, y: seir_model(t, y, beta_full), (0, 70), y0, t_eval=np.arange(0, 71))
        final_state = sol_full.y[:, -1]
    else:
        beta_first = best_params['R0'] * gamma
        beta_second = best_params['R_post'] * gamma
        # Simulate first segment
        sol_first = solve_ivp(lambda t, y: seir_model(t, y, beta_first), (0, best_params['change_day']), y0, t_eval=np.arange(0, best_params['change_day'] + 1))
        # Simulate second segment
        sol_second = solve_ivp(lambda t, y: seir_model(t, y, beta_second), (best_params['change_day'], 70), sol_first.y[:, -1], t_eval=np.arange(best_params['change_day'], 71))
        final_state = sol_second.y[:, -1]

    # Forecast
    def forecast_model(beta, start_day, end_day, initial_state):
        t_span = (start_day, end_day)
        sol = solve_ivp(lambda t, y: seir_model(t, y, beta), t_span, initial_state, t_eval=np.arange(start_day, end_day + 1))
        E = sol.y[1]
        expected = []
        for d in range(start_day, end_day + 1):
            expected_cases = sigma * (E[d-1] + E[d]) / 2
            expected.append(expected_cases)
        return np.array(expected)

    if best_params['change_day'] is None:
        beta_forecast = best_params['R0'] * gamma
        forecast = forecast_model(beta_forecast, 71, 70 + horizon, final_state)
    else:
        beta_forecast_first = best_params['R0'] * gamma
        beta_forecast_second = best_params['R_post'] * gamma
        forecast = forecast_model(beta_forecast_first, 71, best_params['change_day'] + 1, final_state)
        forecast += forecast_model(beta_forecast_second, best_params['change_day'] + 1, 70 + horizon, final_state)

    return {
        'R0': best_params['R0'],
        'R_post': best_params['R_post'],
        'change_day': best_params['change_day'],
        'forecast': forecast.tolist()
    }