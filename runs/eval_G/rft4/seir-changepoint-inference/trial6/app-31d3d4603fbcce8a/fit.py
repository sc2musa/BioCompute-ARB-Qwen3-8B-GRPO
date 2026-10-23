import numpy as np
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    def compute_expected_cases(R, cases_list):
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R_state = 0.0
        beta = R * gamma
        expected = []
        for day in range(len(cases_list)):
            expected_cases = sigma * E
            expected.append(expected_cases)
            delta_S = -beta * I * S / population * 1.0
            delta_E = (beta * I * S / population - sigma * E) * 1.0
            delta_I = (sigma * E - gamma * I) * 1.0
            delta_R = gamma * I * 1.0
            S += delta_S
            E += delta_E
            I += delta_I
            R_state += delta_R
        return expected

    def log_likelihood(expected, observed):
        total = 0.0
        for e, o in zip(expected, observed):
            if e == 0 and o == 0:
                continue
            elif e == 0:
                return -float('inf')
            elif o == 0:
                total += -e
            else:
                total += -e + o * np.log(e)
        return total

    best_log_likelihood = -float('inf')
    best_R0 = 1.0
    best_R_post = 1.0
    best_change_day = None

    # Try all possible change days from 1 to 70
    for change_day in range(1, 71):
        pre_cases = cases[:change_day - 1]
        post_cases = cases[change_day - 1:]

        # Estimate R0 for pre period
        if len(pre_cases) == 0:
            R0 = 1.0
        else:
            def objective_R0(R):
                expected = compute_expected_cases(R, pre_cases)
                ll = log_likelihood(expected, pre_cases)
                return -ll
            result_R0 = minimize(objective_R0, 1.0, method='L-BFGS-B')
            if result_R0.success:
                R0 = result_R0.x[0]
            else:
                R0 = 1.0

        # Estimate R_post for post period
        if len(post_cases) == 0:
            R_post = R0
        else:
            def objective_R_post(R):
                expected = compute_expected_cases(R, post_cases)
                ll = log_likelihood(expected, post_cases)
                return -ll
            result_R_post = minimize(objective_R_post, 1.0, method='L-BFGS-B')
            if result_R_post.success:
                R_post = result_R_post.x[0]
            else:
                R_post = R0

        # Compute total log-likelihood
        ll_pre = 0.0
        if len(pre_cases) > 0:
            expected_pre = compute_expected_cases(R0, pre_cases)
            ll_pre = log_likelihood(expected_pre, pre_cases)

        ll_post = 0.0
        if len(post_cases) > 0:
            expected_post = compute_expected_cases(R_post, post_cases)
            ll_post = log_likelihood(expected_post, post_cases)

        total_ll = ll_pre + ll_post

        if total_ll > best_log_likelihood:
            best_log_likelihood = total_ll
            best_R0 = R0
            best_R_post = R_post
            best_change_day = change_day

    # Try the case with no intervention
    pre_cases = cases
    post_cases = []

    def objective_R0(R):
        expected = compute_expected_cases(R, pre_cases)
        ll = log_likelihood(expected, pre_cases)
        return -ll
    result_R0 = minimize(objective_R0, 1.0, method='L-BFGS-B')
    if result_R0.success:
        R0 = result_R0.x[0]
    else:
        R0 = 1.0
    expected = compute_expected_cases(R0, pre_cases)
    ll = log_likelihood(expected, pre_cases)
    if ll > best_log_likelihood:
        best_log_likelihood = ll
        best_R0 = R0
        best_R_post = R0
        best_change_day = None

    # Simulate the entire 70 days to get the state variables at day 70
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R_state = 0.0
    beta_pre = best_R0 * gamma
    beta_post = best_R_post * gamma

    for day in range(70):
        if best_change_day is None:
            beta = beta_pre
        else:
            if day < best_change_day:
                beta = beta_pre
            else:
                beta = beta_post
        # Compute expected cases for this day (not used for simulation)
        expected_cases = sigma * E
        # Update the state variables
        delta_S = -beta * I * S / population * 1.0
        delta_E = (beta * I * S / population - sigma * E) * 1.0
        delta_I = (sigma * E - gamma * I) * 1.0
        delta_R = gamma * I * 1.0
        S += delta_S
        E += delta_E
        I += delta_I
        R_state += delta_R

    # Now, simulate the forecast period
    forecast = []
    if best_change_day is None:
        beta_forecast = beta_pre
    else:
        beta_forecast = beta_post
    for _ in range(horizon):
        expected_cases = sigma * E
        forecast.append(expected_cases)
        delta_S = -beta_forecast * I * S / population * 1.0
        delta_E = (beta_forecast * I * S / population - sigma * E) * 1.0
        delta_I = (sigma * E - gamma * I) * 1.0
        delta_R = gamma * I * 1.0
        S += delta_S
        E += delta_E
        I += delta_I
        R_state += delta_R

    return {
        "R0": float(best_R0),
        "R_post": float(best_R_post),
        "change_day": best_change_day if best_change_day is not None else None,
        "forecast": forecast
    }