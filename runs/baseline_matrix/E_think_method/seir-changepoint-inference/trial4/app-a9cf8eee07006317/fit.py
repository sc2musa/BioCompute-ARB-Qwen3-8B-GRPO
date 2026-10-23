import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

def seir_model(y, t, beta, sigma, gamma, population):
    S, E, I, R = y
    dSdt = -beta * S * I / population
    dEdt = beta * S * I / population - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    candidate_change_days = list(range(5, 66))  # days 5 to 65 inclusive

    best_nll = -np.inf
    best_R0 = 0.0
    best_R_post = 0.0
    best_change_day = None
    best_model = 'no_intervention'

    # Fit no-intervention model
    def no_intervention_nll(R):
        beta = R * gamma
        t = np.arange(0, 70 + 1)
        y0 = [population - initial_infectious, 0.0, initial_infectious, 0.0]
        sol = odeint(seir_model, y0, t, args=(beta, sigma, gamma, population))
        expected_cases = [sigma * sol[d][1] for d in range(1, 71)]
        nll = 0.0
        for observed, expected in zip(cases, expected_cases):
            if expected == 0:
                if observed == 0:
                    nll += 0.0
                else:
                    nll += float('inf')
            else:
                nll += observed * np.log(expected) - expected
        return nll

    result_no_intervention = minimize(lambda x: -no_intervention_nll(np.exp(x[0])), x0=[0.0], method='L-BFGS-B')
    best_R = np.exp(result_no_intervention.x[0])
    nll_no_intervention = -result_no_intervention.fun
    bic_no_intervention = 2 * nll_no_intervention + 1 * np.log(70)

    for change_day in candidate_change_days:
        def change_point_nll(log_R0, log_R_post):
            R0 = np.exp(log_R0)
            R_post = np.exp(log_R_post)
            beta1 = R0 * gamma
            beta2 = R_post * gamma
            t1 = np.arange(0, change_day + 1)
            y0 = [population - initial_infectious, 0.0, initial_infectious, 0.0]
            sol1 = odeint(seir_model, y0, t1, args=(beta1, sigma, gamma, population))
            y1 = sol1[-1]
            t2 = np.arange(change_day, 70 + 1)
            sol2 = odeint(seir_model, y1, t2, args=(beta2, sigma, gamma, population))
            sol = np.vstack((sol1, sol2[1:]))
            expected_cases = [sigma * sol[d][1] for d in range(1, 71)]
            nll = 0.0
            for observed, expected in zip(cases, expected_cases):
                if expected == 0:
                    if observed == 0:
                        nll += 0.0
                    else:
                        nll += float('inf')
                else:
                    nll += observed * np.log(expected) - expected
            return nll

        result = minimize(lambda x: -change_point_nll(x[0], x[1]), x0=[0.0, 0.0], method='Nelder-Mead')
        best_R0_candidate = np.exp(result.x[0])
        best_R_post_candidate = np.exp(result.x[1])
        nll_candidate = -result.fun
        bic_candidate = 2 * nll_candidate + 2 * np.log(70)

        if nll_candidate > best_nll:
            best_nll = nll_candidate
            best_R0 = best_R0_candidate
            best_R_post = best_R_post_candidate
            best_change_day = change_day
            best_model = 'change_point'

    if best_model == 'change_point':
        bic_change_point = 2 * best_nll + 2 * np.log(70)
        if bic_change_point < bic_no_intervention:
            pass
        else:
            best_R0 = best_R
            best_R_post = best_R
            best_change_day = None
    else:
        pass

    if best_change_day is None:
        beta = best_R0 * gamma
        t = np.arange(0, 70 + horizon + 1)
        y0 = [population - initial_infectious, 0.0, initial_infectious, 0.0]
        sol = odeint(seir_model, y0, t, args=(beta, sigma, gamma, population))
        expected_cases_forecast = [sigma * sol[d][1] for d in range(71, 70 + horizon + 1)]
    else:
        beta1 = best_R0 * gamma
        beta2 = best_R_post * gamma
        t1 = np.arange(0, best_change_day + 1)
        y0 = [population - initial_infectious, 0.0, initial_infectious, 0.0]
        sol1 = odeint(seir_model, y0, t1, args=(beta1, sigma, gamma, population))
        y1 = sol1[-1]
        t2 = np.arange(best_change_day, 70 + horizon + 1)
        sol2 = odeint(seir_model, y1, t2, args=(beta2, sigma, gamma, population))
        sol = np.vstack((sol1, sol2[1:]))
        expected_cases_forecast = [sigma * sol[d][1] for d in range(71, 70 + horizon + 1)]

    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": expected_cases_forecast
    }
