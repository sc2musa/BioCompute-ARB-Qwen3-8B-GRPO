import numpy as np

def simulate_model_before(R, population, initial_infectious, latent_period, infectious_period, days):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    beta = R * gamma

    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R_ = 0.0

    simulated_cases = []

    for _ in range(days):
        dS = -beta * S * I / population
        dE = beta * S * I / population - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I

        S += dS
        E += dE
        I += dI
        R_ += dR

        new_cases = sigma * E * 1.0
        simulated_cases.append(new_cases)

    return simulated_cases

def fit_R(observed_cases, population, initial_infectious, latent_period, infectious_period):
    best_error = float('inf')
    best_R = None

    for R in np.linspace(0.1, 10.0, 100):
        simulated_cases = simulate_model_before(R, population, initial_infectious, latent_period, infectious_period, len(observed_cases))
        error = sum((simulated_cases[i] - observed_cases[i])**2 for i in range(len(observed_cases)))
        if error < best_error:
            best_error = error
            best_R = R

    return best_R

def simulate_full_model(R0, R_post, change_day, population, initial_infectious, latent_period, infectious_period):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R_ = 0.0

    simulated_cases = []

    # Handle None change_day
    if change_day is None:
        change_day = 71  # Ensure all days use R0

    for day in range(70):
        if day < change_day:
            beta = R0 * gamma
        else:
            beta = R_post * gamma

        dS = -beta * S * I / population
        dE = beta * S * I / population - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I

        S += dS
        E += dE
        I += dI
        R_ += dR

        new_cases = sigma * E * 1.0
        simulated_cases.append(new_cases)

    return simulated_cases, S, E, I, R_

def simulate_forecast(R_post, compartments, population, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    beta = R_post * gamma

    S, E, I, R_ = compartments

    forecast = []

    for _ in range(horizon):
        dS = -beta * S * I / population
        dE = beta * S * I / population - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I

        S += dS
        E += dE
        I += dI
        R_ += dR

        new_cases = sigma * E * 1.0
        forecast.append(new_cases)

    return forecast

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    best_error = float('inf')
    best_R0 = None
    best_R_post = None
    best_change_day = None
    best_forecast = []

    for change_day_candidate in [None] + list(range(1, 71)):
        if change_day_candidate is None:
            before_cases = cases
            after_cases = []
            R0_candidate = fit_R(before_cases, population, initial_infectious, latent_period, infectious_period)
            R_post_candidate = R0_candidate
        else:
            before_days = change_day_candidate - 1
            before_cases = cases[:before_days]
            after_cases = cases[before_days:]
            R0_candidate = fit_R(before_cases, population, initial_infectious, latent_period, infectious_period) if before_cases else None
            R_post_candidate = fit_R(after_cases, population, initial_infectious, latent_period, infectious_period) if after_cases else None
            # Handle missing R values
            if R0_candidate is None:
                R0_candidate = R_post_candidate
            if R_post_candidate is None:
                R_post_candidate = R0_candidate

        # Simulate full model
        if change_day_candidate is None:
            simulated_cases, _, _, _, _ = simulate_full_model(R0_candidate, R_post_candidate, 71, population, initial_infectious, latent_period, infectious_period)
        else:
            simulated_cases, _, _, _, _ = simulate_full_model(R0_candidate, R_post_candidate, change_day_candidate, population, initial_infectious, latent_period, infectious_period)

        error = sum((simulated_cases[i] - cases[i])**2 for i in range(len(cases)))

        if error < best_error:
            best_error = error
            best_R0 = R0_candidate
            best_R_post = R_post_candidate
            best_change_day = change_day_candidate

    simulated_full_cases, S_final, E_final, I_final, R_final = simulate_full_model(best_R0, best_R_post, best_change_day, population, initial_infectious, latent_period, infectious_period)
    forecast = simulate_forecast(best_R_post, (S_final, E_final, I_final, R_final), population, latent_period, infectious_period, horizon)

    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": forecast,
    }