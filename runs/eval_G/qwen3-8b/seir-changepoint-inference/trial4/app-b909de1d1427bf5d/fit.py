import numpy as np
from scipy.optimize import minimize

def simulate_seir(population, initial_infectious, R0, R_post, change_day, latent_period, infectious_period, days):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    expected_cases = []
    for day in range(days):
        if change_day is None:
            beta = R0 * gamma
        else:
            beta = R0 * gamma if day < change_day else R_post * gamma
        new_infections = beta * I * S / population
        new_infections = np.clip(new_infections, 0, None)
        E += new_infections
        I += sigma * E - gamma * I
        R += gamma * I
        new_cases = sigma * E
        expected_cases.append(new_cases)
        S -= new_infections
        S = np.clip(S, 0, None)
        E = np.clip(E, 0, None)
        I = np.clip(I, 0, None)
    return np.array(expected_cases)

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    best_log_likelihood = -np.inf
    best_R0 = None
    best_R_post = None
    best_change_day = None
    best_forecast = None

    # Try all possible change days
    for change_day in range(0, 71):
        def objective(params):
            R0, R_post = params
            expected = simulate_seir(population, initial_infectious, R0, R_post, change_day, latent_period, infectious_period, len(cases))
            log_likelihood = 0.0
            eps = 1e-10
            for observed, expected_val in zip(cases, expected):
                expected_val = max(expected_val, eps)
                if observed == 0:
                    log_likelihood += np.log(expected_val) * observed - expected_val
                else:
                    log_likelihood += observed * np.log(expected_val) - expected_val
            return -log_likelihood

        # Initial guess closer to true values
        initial_guess = [3.5, 3.5]  # Adjusted initial guess for higher R0
        bounds = [(0.1, 20.0), (0.1, 20.0)]
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        if result.success:
            current_log_likelihood = -result.fun
            if current_log_likelihood > best_log_likelihood:
                best_log_likelihood = current_log_likelihood
                best_R0 = result.x[0]
                best_R_post = result.x[1]
                best_change_day = change_day
                forecast = simulate_seir(population, initial_infectious, best_R0, best_R_post, best_change_day, latent_period, infectious_period, horizon)
                best_forecast = forecast

    # Handle no-intervention case
    def no_intervention_objective(R0):
        expected = simulate_seir(population, initial_infectious, R0, R0, None, latent_period, infectious_period, len(cases))
        log_likelihood = 0.0
        eps = 1e-10
        for observed, expected_val in zip(cases, expected):
            expected_val = max(expected_val, eps)
            if observed == 0:
                log_likelihood += np.log(expected_val) * observed - expected_val
            else:
                log_likelihood += observed * np.log(expected_val) - expected_val
        return -log_likelihood

    # Initial guess closer to true values
    initial_guess_no_intervention = [3.5]
    result_no_intervention = minimize(no_intervention_objective, initial_guess_no_intervention, bounds=[(0.1, 20.0)], method='L-BFGS-B')
    if result_no_intervention.success:
        current_log_likelihood_no_intervention = -result_no_intervention.fun
        if current_log_likelihood_no_intervention > best_log_likelihood:
            best_log_likelihood = current_log_likelihood_no_intervention
            best_R0 = result_no_intervention.x[0]
            best_R_post = best_R0
            best_change_day = None
            forecast = simulate_seir(population, initial_infectious, best_R0, best_R_post, best_change_day, latent_period, infectious_period, horizon)
            best_forecast = forecast

    # Final check for no-intervention case
    if best_R_post == best_R0:
        best_change_day = None

    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": best_forecast.tolist()
    }