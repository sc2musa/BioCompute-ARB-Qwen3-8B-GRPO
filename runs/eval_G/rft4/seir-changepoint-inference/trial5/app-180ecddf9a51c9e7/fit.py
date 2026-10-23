import numpy as np
import math
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    cases_list = np.array(cases)
    best_log_likelihood = -float('inf')
    best_R0 = 0.0
    best_alpha = 0.0
    best_t_change = None

    # Check all possible t_change values
    for t_change in range(1, 71):
        def nll(params):
            R0, alpha = params
            R_post = R0 * alpha
            S = population - initial_infectious
            E = 0.0
            I = initial_infectious
            R = 0.0
            expected = []
            for day in range(1, 71):
                if day < t_change:
                    beta = R0 * gamma
                else:
                    beta = R_post * gamma
                dS = -beta * S * I
                dE = beta * S * I - sigma * E
                dI = sigma * E - gamma * I
                dR = gamma * I
                # Clamp to zero to avoid negative values
                S = max(0.0, S + dS)
                E = max(0.0, E + dE)
                I = max(0.0, I + dI)
                R = max(0.0, R + dR)
                expected_cases = E * sigma
                expected.append(expected_cases)
            log_likelihood = 0.0
            for d in range(70):
                mu = expected[d]
                if mu == 0:
                    log_likelihood += -cases_list[d] * math.log(1e-10)
                else:
                    log_likelihood += -mu + cases_list[d] * math.log(mu)
            return -log_likelihood
        # Initial guess for R0 and alpha
        initial_guess = [1.0, 1.0]
        result = minimize(nll, initial_guess, method='Nelder-Mead', bounds=[(0.0, 100.0), (0.0, 1.0)])
        if result.success:
            current_log_likelihood = -result.fun
            if current_log_likelihood > best_log_likelihood:
                best_log_likelihood = current_log_likelihood
                best_R0 = result.x[0]
                best_alpha = result.x[1]
                best_t_change = t_change

    # Check no-intervention model
    def nll_no_intervention(params):
        R0, alpha = params
        R_post = R0 * alpha
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        expected = []
        for day in range(1, 71):
            beta = R0 * gamma
            dS = -beta * S * I
            dE = beta * S * I - sigma * E
            dI = sigma * E - gamma * I
            dR = gamma * I
            # Clamp to zero to avoid negative values
            S = max(0.0, S + dS)
            E = max(0.0, E + dE)
            I = max(0.0, I + dI)
            R = max(0.0, R + dR)
            expected_cases = E * sigma
            expected.append(expected_cases)
        log_likelihood = 0.0
        for d in range(70):
            mu = expected[d]
            if mu == 0:
                log_likelihood += -cases_list[d] * math.log(1e-10)
            else:
                log_likelihood += -mu + cases_list[d] * math.log(mu)
        return -log_likelihood
    result_no_intervention = minimize(nll_no_intervention, [1.0, 1.0], method='Nelder-Mead', bounds=[(0.0, 100.0), (0.0, 1.0)])
    if -result_no_intervention.fun > best_log_likelihood:
        best_log_likelihood = -result_no_intervention.fun
        best_R0 = result_no_intervention.x[0]
        best_alpha = result_no_intervention.x[1]
        best_t_change = None

    # Forecast
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    forecast = []
    for day in range(71, 70 + horizon + 1):
        if best_t_change is not None and day >= best_t_change:
            beta = best_R0 * best_alpha * gamma
        else:
            beta = best_R0 * gamma
        dS = -beta * S * I
        dE = beta * S * I - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        # Clamp to zero to avoid negative values
        S = max(0.0, S + dS)
        E = max(0.0, E + dE)
        I = max(0.0, I + dI)
        R = max(0.0, R + dR)
        expected_cases = E * sigma
        forecast.append(expected_cases)
    return {
        "R0": float(best_R0),
        "R_post": float(best_R0 * best_alpha),
        "change_day": best_t_change if best_t_change is not None else None,
        "forecast": forecast
    }