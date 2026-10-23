import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    def seir_ode(t, y, beta):
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    best_error = float('inf')
    best_R0 = best_R_post = best_change_day = None

    for change_day in range(0, 71):
        observed_first = cases[:change_day]
        observed_second = cases[change_day:]

        # Fit R0
        def error_R0(R0):
            S = population - initial_infectious
            E = 0
            I = initial_infectious
            R = 0
            state = [S, E, I, R]
            sim = []
            for day in range(1, change_day + 1):
                beta = R0 * gamma
                sol = solve_ivp(lambda t, y: seir_ode(t, y, beta), (day-1, day), state, method='RK45', t_eval=[day-1, day])
                new_state = sol.y[:, -1]
                sim.append(new_state[2] - state[2])
                state = new_state
            return np.sum((np.array(sim) - observed_first)**2)

        res_R0 = minimize(error_R0, x0=1.0, bounds=[(0.1, 10.0)])
        R0_opt = res_R0.x[0]

        # Simulate first segment with R0_opt to get final state
        S = population - initial_infectious
        E = 0
        I = initial_infectious
        R = 0
        state = [S, E, I, R]
        sim_first = []
        for day in range(1, change_day + 1):
            beta = R0_opt * gamma
            sol = solve_ivp(lambda t, y: seir_ode(t, y, beta), (day-1, day), state, method='RK45', t_eval=[day-1, day])
            new_state = sol.y[:, -1]
            sim_first.append(new_state[2] - state[2])
            state = new_state

        # Fit R_post
        def error_R_post(R_post):
            sim_second = []
            for day in range(change_day + 1, 71):
                beta = R_post * gamma
                sol = solve_ivp(lambda t, y: seir_ode(t, y, beta), (day-1, day), state, method='RK45', t_eval=[day-1, day])
                new_state = sol.y[:, -1]
                sim_second.append(new_state[2] - state[2])
                state = new_state
            return np.sum((np.array(sim_second) - observed_second)**2)

        res_R_post = minimize(error_R_post, x0=1.0, bounds=[(0.1, 10.0)])
        R_post_opt = res_R_post.x[0]

        # Final simulation
        S = population - initial_infectious
        E = 0
        I = initial_infectious
        R = 0
        state = [S, E, I, R]
        sim_full = []
        for day in range(1, 71):
            beta = R0_opt * gamma if day <= change_day else R_post_opt * gamma
            sol = solve_ivp(lambda t, y: seir_ode(t, y, beta), (day-1, day), state, method='RK45', t_eval=[day-1, day])
            new_state = sol.y[:, -1]
            sim_full.append(new_state[2] - state[2])
            state = new_state

        error = np.sum((np.array(sim_full) - cases)**2)

        if error < best_error:
            best_error = error
            best_R0 = R0_opt
            best_R_post = R_post_opt
            best_change_day = change_day

    # Forecast
    forecast = []
    S = population - initial_infectious
    E = 0
    I = initial_infectious
    R = 0
    state = [S, E, I, R]
    for day in range(71, 71 + horizon):
        beta = best_R0 * gamma if day <= best_change_day else best_R_post * gamma
        sol = solve_ivp(lambda t, y: seir_ode(t, y, beta), (day-1, day), state, method='RK45', t_eval=[day-1, day])
        new_state = sol.y[:, -1]
        forecast.append(new_state[2] - state[2])
        state = new_state

    return {
        'R0': best_R0,
        'R_post': best_R_post,
        'change_day': best_change_day,
        'forecast': forecast
    }