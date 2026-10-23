import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1 / latent_period
    gamma = 1 / infectious_period
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0
    initial_conditions = [S0, E0, I0, R0]

    def forecast_cases(R0, R_post, change_day):
        state = initial_conditions.copy()
        for day in range(1, 71):
            t_span = (day-1, day)
            def ode(t, y, beta):
                S, E, I, R = y
                dSdt = -beta * S * I / population
                dEdt = beta * S * I / population - sigma * E
                dI_dt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dI_dt, dRdt]
            if day <= change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
            sol = solve_ivp(ode, t_span, state, t_eval=[day-1, day], args=(beta,))
            state = sol.y[:, -1]
        forecast = []
        beta = R_post * gamma if change_day < 70 else R0 * gamma
        for day in range(71, 71 + horizon):
            t_span = (day-1, day)
            def ode(t, y, beta):
                S, E, I, R = y
                dSdt = -beta * S * I / population
                dEdt = beta * S * I / population - sigma * E
                dI_dt = sigma * E - gamma * I
                dRdt = gamma * I
                return [dSdt, dEdt, dI_dt, dRdt]
            sol = solve_ivp(ode, t_span, state, t_eval=[day-1, day], args=(beta,))
            new_cases = sol.y[2][-1] - sol.y[2][0]
            forecast.append(new_cases)
            state = sol.y[:, -1]
        return forecast

    best_loss = float('inf')
    best_params = None

    for change_day in range(0, 72):
        if change_day == 0:
            days = list(range(1, 71))
            observed = cases
            def loss(R):
                beta = R * gamma
                cases_sim = []
                state = initial_conditions.copy()
                for day in days:
                    t_span = (day-1, day)
                    def ode(t, y, beta):
                        S, E, I, R = y
                        dSdt = -beta * S * I / population
                        dEdt = beta * S * I / population - sigma * E
                        dI_dt = sigma * E - gamma * I
                        dRdt = gamma * I
                        return [dSdt, dEdt, dI_dt, dRdt]
                    sol = solve_ivp(ode, t_span, state, t_eval=[day-1, day], args=(beta,))
                    new_cases = sol.y[2][-1] - sol.y[2][0]
                    cases_sim.append(new_cases)
                    state = sol.y[:, -1]
                return np.sum((np.array(cases_sim) - np.array(observed))**2)
            result = minimize(loss, x0=1.0)
            R = result.x[0]
            loss_total = result.fun
            params = (R, None, 0)
        elif change_day == 71:
            days = list(range(1, 71))
            observed = cases
            def loss(R):
                beta = R * gamma
                cases_sim = []
                state = initial_conditions.copy()
                for day in days:
                    t_span = (day-1, day)
                    def ode(t, y, beta):
                        S, E, I, R = y
                        dSdt = -beta * S * I / population
                        dEdt = beta * S * I / population - sigma * E
                        dI_dt = sigma * E - gamma * I
                        dRdt = gamma * I
                        return [dSdt, dEdt, dI_dt, dRdt]
                    sol = solve_ivp(ode, t_span, state, t_eval=[day-1, day], args=(beta,))
                    new_cases = sol.y[2][-1] - sol.y[2][0]
                    cases_sim.append(new_cases)
                    state = sol.y[:, -1]
                return np.sum((np.array(cases_sim) - np.array(observed))**2)
            result = minimize(loss, x0=1.0)
            R = result.x[0]
            loss_total = result.fun
            params = (R, None, 71)
        else:
            days_before = list(range(1, change_day+1))
            days_after = list(range(change_day+1, 71))
            observed_before = cases[:change_day]
            observed_after = cases[change_day:]

            def optimize_before(R):
                beta = R * gamma
                cases_sim = []
                state = initial_conditions.copy()
                for day in days_before:
                    t_span = (day-1, day)
                    def ode(t, y, beta):
                        S, E, I, R = y
                        dSdt = -beta * S * I / population
                        dEdt = beta * S * I / population - sigma * E
                        dI_dt = sigma * E - gamma * I
                        dRdt = gamma * I
                        return [dSdt, dEdt, dI_dt, dRdt]
                    sol = solve_ivp(ode, t_span, state, t_eval=[day-1, day], args=(beta,))
                    new_cases = sol.y[2][-1] - sol.y[2][0]
                    cases_sim.append(new_cases)
                    state = sol.y[:, -1]
                return np.sum((np.array(cases_sim) - np.array(observed_before))**2)

            def optimize_after(R):
                beta = R * gamma
                cases_sim = []
                state = initial_conditions.copy()
                for day in days_before:
                    t_span = (day-1, day)
                    def ode(t, y, beta):
                        S, E, I, R = y
                        dSdt = -beta * S * I / population
                        dEdt = beta * S * I / population - sigma * E
                        dI_dt = sigma * E - gamma * I
                        dRdt = gamma * I
                        return [dSdt, dEdt, dI_dt, dRdt]
                    sol = solve_ivp(ode, t_span, state, t_eval=[day-1, day], args=(beta,))
                    new_cases = sol.y[2][-1] - sol.y[2][0]
                    cases_sim.append(new_cases)
                    state = sol.y[:, -1]
                for day in days_after:
                    t_span = (day-1, day)
                    def ode(t, y, beta):
                        S, E, I, R = y
                        dSdt = -beta * S * I / population
                        dEdt = beta * S * I / population - sigma * E
                        dI_dt = sigma * E - gamma * I
                        dRdt = gamma * I
                        return [dSdt, dEdt, dI_dt, dRdt]
                    sol = solve_ivp(ode, t_span, state, t_eval=[day-1, day], args=(beta,))
                    new_cases = sol.y[2][-1] - sol.y[2][0]
                    cases_sim.append(new_cases)
                    state = sol.y[:, -1]
                return np.sum((np.array(cases_sim) - np.array(observed_after))**2)

            result_before = minimize(optimize_before, x0=1.0)
            R0 = result_before.x[0]
            result_after = minimize(optimize_after, x0=1.0)
            R_post = result_after.x[0]
            loss_total = result_before.fun + result_after.fun
            params = (R0, R_post, change_day)

        if loss_total < best_loss:
            best_loss = loss_total
            best_params = params

    R0, R_post, change_day = best_params

    return {
        'R0': R0,
        'R_post': R_post,
        'change_day': change_day,
        'forecast': forecast_cases(R0, R_post, change_day)
    }

if __name__ == '__main__':
    import json
    with open('$ROOT/runs/eval_round4/rft4_G/seir-changepoint-inference/trial19/app-e643064838a70ec8/data/example1.json') as f:
        data = json.load(f)
    cases = data['inputs']['cases']
    population = data['inputs']['population']
    initial_infectious = data['inputs']['initial_infectious']
    latent_period = data['inputs']['latent_period']
    infectious_period = data['inputs']['infectious_period']
    horizon = data['inputs']['horizon']
    result = fit(cases, population, initial_infectious, latent_period, infectious_period, horizon)
    print(json.dumps(result, indent=2))