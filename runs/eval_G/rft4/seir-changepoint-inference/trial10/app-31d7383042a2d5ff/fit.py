import numpy as np
from scipy.optimize import minimize

def simulate_period(R, initial_S, initial_E, initial_I, initial_R, population, sigma, gamma, days):
    S = initial_S
    E = initial_E
    I = initial_I
    R_val = initial_R
    expected_cases = []
    for _ in range(days):
        beta = R * gamma
        delta_S = -beta * S * I / population
        delta_E = beta * S * I / population - sigma * E
        delta_I = sigma * E - gamma * I
        delta_R = gamma * I
        S += delta_S
        E += delta_E
        I += delta_I
        R_val += delta_R
        expected = sigma * E
        expected_cases.append(expected)
    return S, E, I, R_val, expected_cases

def compute_log_likelihood(expected, observed):
    log_likelihood = 0.0
    for e, o in zip(expected, observed):
        if e < 1e-10:
            e = 1e-10
        log_likelihood += -o * np.log(e) + e
    return log_likelihood

def fit_R(cases_period, initial_S, initial_E, initial_I, initial_R, population, sigma, gamma, days):
    def objective(R):
        S, E, I, R_val, expected = simulate_period(R, initial_S, initial_E, initial_I, initial_R, population, sigma, gamma, days)
        return -compute_log_likelihood(expected, cases_period)
    result = minimize(objective, 1.0, bounds=[(0.0001, 100.0)])
    return result.x[0]

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    best_log_likelihood = -np.inf
    best_R0 = 0.0
    best_R_post = 0.0
    best_change_day = 0.0
    for change_day_candidate in range(0, 71):
        if change_day_candidate == 0:
            # No change, fit R0 for all days
            initial_S = population - initial_infectious
            initial_E = 0.0
            initial_I = initial_infectious
            initial_R = 0.0
            R0 = fit_R(cases, initial_S, initial_E, initial_I, initial_R, population, sigma, gamma, 70)
            R_post = R0
            # Simulate to get expected cases
            S, E, I, R_val, expected = simulate_period(R0, initial_S, initial_E, initial_I, initial_R, population, sigma, gamma, 70)
            log_likelihood = compute_log_likelihood(expected, cases)
        else:
            # Split into two periods
            cases_period1 = cases[:change_day_candidate]
            cases_period2 = cases[change_day_candidate:]
            # Simulate first period to get initial state for second period
            initial_S1 = population - initial_infectious
            initial_E1 = 0.0
            initial_I1 = initial_infectious
            initial_R1 = 0.0
            R0 = fit_R(cases_period1, initial_S1, initial_E1, initial_I1, initial_R1, population, sigma, gamma, change_day_candidate)
            # Get state after first period
            S1, E1, I1, R1, _ = simulate_period(R0, initial_S1, initial_E1, initial_I1, initial_R1, population, sigma, gamma, change_day_candidate)
            # Fit R_post for second period
            R_post = fit_R(cases_period2, S1, E1, I1, R1, population, sigma, gamma, len(cases_period2))
            # Compute log likelihood for both periods
            # First period
            S1, E1, I1, R1, expected1 = simulate_period(R0, initial_S1, initial_E1, initial_I1, initial_R1, population, sigma, gamma, change_day_candidate)
            log_likelihood1 = compute_log_likelihood(expected1, cases_period1)
            # Second period
            S2, E2, I2, R2, expected2 = simulate_period(R_post, S1, E1, I1, R1, population, sigma, gamma, len(cases_period2))
            log_likelihood2 = compute_log_likelihood(expected2, cases_period2)
            log_likelihood = log_likelihood1 + log_likelihood2
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_R0 = R0
            best_R_post = R_post
            best_change_day = float(change_day_candidate)
    # Simulate forecast
    initial_S = population - initial_infectious
    initial_E = 0.0
    initial_I = initial_infectious
    initial_R = 0.0
    for d in range(70):
        beta = best_R0 * gamma if d < best_change_day else best_R_post * gamma
        delta_S = -beta * initial_S * initial_I / population
        delta_E = beta * initial_S * initial_I / population - sigma * initial_E
        delta_I = sigma * initial_E - gamma * initial_I
        delta_R = gamma * initial_I
        initial_S += delta_S
        initial_E += delta_E
        initial_I += delta_I
        initial_R += delta_R
    forecast = []
    current_day = 71
    for _ in range(horizon):
        if current_day < best_change_day:
            beta = best_R0 * gamma
        else:
            beta = best_R_post * gamma
        delta_S = -beta * initial_S * initial_I / population
        delta_E = beta * initial_S * initial_I / population - sigma * initial_E
        delta_I = sigma * initial_E - gamma * initial_I
        delta_R = gamma * initial_I
        initial_S += delta_S
        initial_E += delta_E
        initial_I += delta_I
        initial_R += delta_R
        forecast.append(sigma * initial_E)
        current_day += 1
    return {
        'R0': best_R0,
        'R_post': best_R_post,
        'change_day': best_change_day,
        'forecast': forecast
    }