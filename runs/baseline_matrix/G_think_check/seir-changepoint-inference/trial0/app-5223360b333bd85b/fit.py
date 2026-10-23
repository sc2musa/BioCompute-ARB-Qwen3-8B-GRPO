import numpy as np

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    best_error = float('inf')
    best_R0 = None
    best_R_post = None
    best_change_day = None

    # Iterate over possible change_days
    for change_day_candidate in range(0, 71):  # days 0 to 70
        # Split the data into before and after
        before_cases = cases[:change_day_candidate]  # days 1 to change_day_candidate
        after_cases = cases[change_day_candidate:]  # days change_day_candidate+1 to 70

        # Fit R0 for before_cases
        R0_candidate = fit_R0(before_cases, population, initial_infectious, latent_period, infectious_period)
        # Fit R_post for after_cases
        R_post_candidate = fit_R_post(after_cases, population, initial_infectious, latent_period, infectious_period)

        # Simulate the entire 70 days with these parameters to compute error
        expected_cases = simulate_full_model(population, initial_infectious, latent_period, infectious_period, R0_candidate, R_post_candidate, change_day_candidate)
        # Compute error between observed cases and expected cases
        error = sum((observed - expected)**2 for observed, expected in zip(cases, expected_cases))
        if error < best_error:
            best_error = error
            best_R0 = R0_candidate
            best_R_post = R_post_candidate
            best_change_day = change_day_candidate

    # Forecast the next horizon days
    forecast = forecast_cases(population, initial_infectious, latent_period, infectious_period, best_R0, best_R_post, best_change_day, horizon)
    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": forecast
    }


def fit_R0(segment_cases, population, initial_infectious, latent_period, infectious_period):
    # Perform a grid search over R0 values
    best_error = float('inf')
    best_R0 = 1.0
    for R0_candidate in np.linspace(0.5, 5.0, 100):
        # Simulate the model for the segment_cases days with R0_candidate
        sigma = 1.0 / latent_period
        gamma = 1.0 / infectious_period
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        expected_segment_cases = []
        for day in range(1, len(segment_cases) + 1):
            beta = R0_candidate * gamma
            new_infections = beta * S * I / population
            E_new = E + new_infections - sigma * E
            I_new = I + sigma * E - gamma * I
            S = S - new_infections
            E = E_new
            I = I_new
            R = R + gamma * I
            new_cases = sigma * E
            expected_segment_cases.append(new_cases)
        # Compute error
        error = sum((observed - expected)**2 for observed, expected in zip(segment_cases, expected_segment_cases))
        if error < best_error:
            best_error = error
            best_R0 = R0_candidate
    return best_R0


def fit_R_post(segment_cases, population, initial_infectious, latent_period, infectious_period):
    # Perform a grid search over R_post values
    best_error = float('inf')
    best_R_post = 1.0
    for R_post_candidate in np.linspace(0.5, 5.0, 100):
        # Simulate the model for the segment_cases days with R_post_candidate
        sigma = 1.0 / latent_period
        gamma = 1.0 / infectious_period
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        expected_segment_cases = []
        for day in range(1, len(segment_cases) + 1):
            beta = R_post_candidate * gamma
            new_infections = beta * S * I / population
            E_new = E + new_infections - sigma * E
            I_new = I + sigma * E - gamma * I
            S = S - new_infections
            E = E_new
            I = I_new
            R = R + gamma * I
            new_cases = sigma * E
            expected_segment_cases.append(new_cases)
        # Compute error
        error = sum((observed - expected)**2 for observed, expected in zip(segment_cases, expected_segment_cases))
        if error < best_error:
            best_error = error
            best_R_post = R_post_candidate
    return best_R_post


def simulate_full_model(population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    expected_cases = []
    for day in range(1, 71):
        if day < change_day:
            beta = R0 * gamma
        else:
            beta = R_post * gamma
        new_infections = beta * S * I / population
        E_new = E + new_infections - sigma * E
        I_new = I + sigma * E - gamma * I
        S = S - new_infections
        E = E_new
        I = I_new
        R = R + gamma * I
        new_cases = sigma * E
        expected_cases.append(new_cases)
    return expected_cases


def forecast_cases(population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    # Simulate up to day 70
    for day in range(1, 71):
        if day < change_day:
            beta = R0 * gamma
        else:
            beta = R_post * gamma
        new_infections = beta * S * I / population
        E_new = E + new_infections - sigma * E
        I_new = I + sigma * E - gamma * I
        S = S - new_infections
        E = E_new
        I = I_new
        R = R + gamma * I
    # Forecast the next horizon days
    forecast = []
    for day in range(71, 71 + horizon):
        if day < change_day:
            beta = R0 * gamma
        else:
            beta = R_post * gamma
        new_infections = beta * S * I / population
        E_new = E + new_infections - sigma * E
        I_new = I + sigma * E - gamma * I
        S = S - new_infections
        E = E_new
        I = I_new
        R = R + gamma * I
        new_cases = sigma * E
        forecast.append(new_cases)
    return forecast
