import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    t_span = (0.0, 70.0)
    t_eval = np.linspace(0.0, 70.0, 1000)

    best_log_likelihood = -np.inf
    best_R0 = None
    best_R_post = None
    best_c = None

    for c in range(0, 71):  # c from 0 to 70 inclusive
        # Initial conditions
        S0 = population - initial_infectious
        E0 = 0.0
        I0 = initial_infectious
        R0 = 0.0
        y0 = [S0, E0, I0, R0]

        # Define the objective function for optimization
        def objective(params):
            R0_opt, R_post_opt = params
            # Define the ODE function with the current c
            def ode_func(t, y):
                S, E, I, R = y
                beta = R0_opt * gamma if t < c else R_post_opt * gamma
                dSdt = -beta * S * I / population
                dEdt = beta * S * I / population - sigma * E
                dI_dt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dI_dt, dRdt]

            # Solve the ODE
            sol = solve_ivp(ode_func, t_span, y0, t_eval=t_eval, method='RK45')

            # Compute the expected cases for each day
            expected_cases = []
            for d in range(1, 71):
                t_start = d - 1
                t_end = d
                mask = (t_eval > t_start) & (t_eval <= t_end)
                t_sub = t_eval[mask]
                E_sub = sol.y[1][mask]
                integral = np.trapz(E_sub, t_sub) * sigma
                expected_cases.append(integral)

            # Compute the log-likelihood
            log_likelihood = 0.0
            for observed, expected in zip(cases, expected_cases):
                # Assuming Poisson likelihood
                log_likelihood += observed * np.log(expected) - expected
            return -log_likelihood  # Negative for minimization

        # Initial guess for R0 and R_post
        initial_guess = [3.0, 2.0]  # Adjusted initial guess
        result = minimize(objective, initial_guess, method='Nelder-Mead', bounds=[(0.1, 10.0), (0.1, 10.0)])
        R0_opt, R_post_opt = result.x
        log_likelihood_opt = -result.fun

        # Compare with the best so far
        if log_likelihood_opt > best_log_likelihood:
            best_log_likelihood = log_likelihood_opt
            best_R0 = R0_opt
            best_R_post = R_post_opt
            best_c = c

    # Determine change_day
    if best_R_post <= 0.75 * best_R0:
        change_day = best_c
    else:
        change_day = None
        best_R_post = best_R0

    # Compute forecast
    t_forecast = np.linspace(0.0, 70.0 + horizon, 1000)
    def ode_func(t, y):
        S, E, I, R = y
        beta = best_R0 * gamma if t < best_c else best_R_post * gamma
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dI_dt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dI_dt, dRdt]

    sol_forecast = solve_ivp(ode_func, (0.0, 70.0 + horizon), y0, t_eval=t_forecast, method='RK45')
    forecast = []
    for d in range(71, 70 + horizon + 1):
        t_start = d - 1
        t_end = d
        mask = (t_forecast > t_start) & (t_forecast <= t_end)
        t_sub = t_forecast[mask]
        E_sub = sol_forecast.y[1][mask]
        integral = np.trapz(E_sub, t_sub) * sigma
        forecast.append(integral)

    return {
        'R0': float(best_R0),
        'R_post': float(best_R_post),
        'change_day': change_day,
        'forecast': forecast
    }
