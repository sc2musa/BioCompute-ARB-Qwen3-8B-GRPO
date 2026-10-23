import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    def seir_ode(t, y, R0, R_post, change_day):
        S, E, I, R = y
        beta = R0 * gamma if t < change_day else R_post * gamma
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    def simulate_model(R0, R_post, change_day):
        y0 = [population - initial_infectious, 0.0, initial_infectious, 0.0]
        t_span = (0.0, 70.0)
        sol = solve_ivp(seir_ode, t_span, y0, args=(R0, R_post, change_day), method='RK45', t_eval=np.arange(0, 71, 1))
        I_values = sol.y[2]
        predicted = [I_values[i] - I_values[i-1] for i in range(1, 71)]
        return predicted

    best_error = float('inf')
    best_R0 = best_R_post = best_change_day = None

    # Efficient change_day search
    for change_day in range(0, 72):
        # Estimate R0 and R_post using optimization
        def error(params):
            R0, R_post = params
            predicted = simulate_model(R0, R_post, change_day)
            return np.sum((np.array(predicted) - np.array(cases)) ** 2)

        # Initial guess based on first few days
        initial_guess = [2.0, 1.0]
        if change_day == 0:
            initial_guess = [2.0, 2.0]
        elif change_day == 71:
            initial_guess = [2.0, 2.0]
        else:
            initial_guess = [2.0, 1.0]

        result = minimize(error, x0=initial_guess, method='L-BFGS-B', bounds=[(0.1, 10.0), (0.1, 10.0)])
        R0, R_post = result.x
        predicted = simulate_model(R0, R_post, change_day)
        current_error = np.sum((np.array(predicted) - np.array(cases)) ** 2)

        if current_error < best_error:
            best_error = current_error
            best_R0, best_R_post, best_change_day = R0, R_post, change_day

    # Post-processing: handle no-intervention case
    if best_R_post >= 0.75 * best_R0:
        best_change_day = 71
        best_R_post = best_R0

    # Forecast
    y0 = [population - initial_infectious, 0.0, initial_infectious, 0.0]
    t_span = (70.0, 70.0 + horizon)
    sol = solve_ivp(seir_ode, t_span, y0, args=(best_R0, best_R_post, best_change_day), method='RK45', t_eval=np.arange(70.0, 70.0 + horizon + 1, 1))
    I_values = sol.y[2]
    forecast = [I_values[i] - I_values[i-1] for i in range(1, horizon + 1)]

    return {
        'R0': best_R0,
        'R_post': best_R_post,
        'change_day': best_change_day,
        'forecast': forecast
    }