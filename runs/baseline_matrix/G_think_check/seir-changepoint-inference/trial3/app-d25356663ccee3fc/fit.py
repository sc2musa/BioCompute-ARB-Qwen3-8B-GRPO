import numpy as np
from scipy.optimize import minimize

def simulate(beta, initial_S, initial_E, initial_I, initial_R, days, sigma, gamma, population):
    S = initial_S
    E = initial_E
    I = initial_I
    R = initial_R
    cases = []
    compartments = []
    for _ in range(days):
        new_cases = sigma * E
        cases.append(new_cases)
        compartments.append((S, E, I, R))
        dS = -beta * S * I / population
        dE = beta * S * I / population - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        S += dS
        E += dE
        I += dI
        R += dR
    compartments.append((S, E, I, R))
    return cases, compartments

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    initial_S = population - initial_infectious
    initial_E = 0.0
    initial_I = initial_infectious
    initial_R = 0.0

    best_error = float('inf')
    best_R0 = None
    best_R_post = None
    best_change_day = None

    # Try no intervention
    def error_no_intervention(R0):
        beta_val = R0 * gamma
        simulated_cases, _ = simulate(beta_val, initial_S, initial_E, initial_I, initial_R, 70, sigma, gamma, population)
        error = sum((simulated_cases[i] - cases[i])**2 for i in range(70))
        return error

    result = minimize(error_no_intervention, x0=1.0, method='L-BFGS-B', bounds=[(0.1, 10.0)])
    if result.success:
        error = result.fun
        if error < best_error:
            best_error = error
            best_R0 = result.x[0]
            best_R_post = best_R0
            best_change_day = None

    # Try all possible change_days
    for change_day in range(1, 71):
        observed_cases_before = cases[:change_day-1]
        observed_cases_after = cases[change_day-1:]

        # Optimize R0 for before part
        def error_before(R0):
            beta_val = R0 * gamma
            simulated_before_cases, simulated_before_compartments = simulate(beta_val, initial_S, initial_E, initial_I, initial_R, change_day-1, sigma, gamma, population)
            error = sum((simulated_before_cases[i] - observed_cases_before[i])**2 for i in range(len(simulated_before_cases)))
            return error

        result_before = minimize(error_before, x0=1.0, method='L-BFGS-B', bounds=[(0.1, 10.0)])
        if not result_before.success:
            continue
        R0_best = result_before.x[0]

        # Optimize R_post for after part
        def error_after(R_post):
            beta_after = R_post * gamma
            # Simulate before part
            simulated_before_cases, simulated_before_compartments = simulate(R0_best * gamma, initial_S, initial_E, initial_I, initial_R, change_day-1, sigma, gamma, population)
            final_before_S, final_before_E, final_before_I, final_before_R = simulated_before_compartments[-1]
            # Simulate after part
            days_after = 71 - change_day
            simulated_after_cases, _ = simulate(beta_after, final_before_S, final_before_E, final_before_I, final_before_R, days_after, sigma, gamma, population)
            error = sum((simulated_after_cases[i] - observed_cases_after[i])**2 for i in range(len(simulated_after_cases)))
            return error

        result_after = minimize(error_after, x0=1.0, method='L-BFGS-B', bounds=[(0.1, 10.0)])
        if not result_after.success:
            continue
        R_post_best = result_after.x[0]

        total_error = result_before.fun + result_after.fun
        if total_error < best_error:
            best_error = total_error
            best_R0 = R0_best
            best_R_post = R_post_best
            best_change_day = change_day

    # Generate forecast
    forecast_cases = []
    if best_change_day is not None:
        # Simulate up to the change_day
        simulated_before_cases, simulated_before_compartments = simulate(best_R0 * gamma, initial_S, initial_E, initial_I, initial_R, best_change_day-1, sigma, gamma, population)
        final_before_S, final_before_E, final_before_I, final_before_R = simulated_before_compartments[-1]
        # Simulate after part for data days
        days_after_data = 71 - best_change_day
        simulated_after_data_cases, _ = simulate(best_R_post * gamma, final_before_S, final_before_E, final_before_I, final_before_R, days_after_data, sigma, gamma, population)
        # Get compartments at the end of data
        final_after_data_S, final_after_data_E, final_after_data_I, final_after_data_R = simulated_after_data_cases[-1]
        # Simulate forecast days
        S_forecast = final_after_data_S
        E_forecast = final_after_data_E
        I_forecast = final_after_data_I
        R_forecast = final_after_data_R
        for _ in range(horizon):
            new_cases = sigma * E_forecast
            forecast_cases.append(new_cases)
            dS = -best_R_post * gamma * S_forecast * I_forecast / population
            dE = best_R_post * gamma * S_forecast * I_forecast / population - sigma * E_forecast
            dI = sigma * E_forecast - gamma * I_forecast
            dR = gamma * I_forecast
            S_forecast += dS
            E_forecast += dE
            I_forecast += dI
            R_forecast += dR
    else:
        # No intervention, simulate all 70 days
        simulated_cases, compartments = simulate(best_R0 * gamma, initial_S, initial_E, initial_I, initial_R, 70, sigma, gamma, population)
        final_S, final_E, final_I, final_R = compartments[-1]
        # Simulate forecast days
        S_forecast = final_S
        E_forecast = final_E
        I_forecast = final_I
        R_forecast = final_R
        for _ in range(horizon):
            new_cases = sigma * E_forecast
            forecast_cases.append(new_cases)
            dS = -best_R0 * gamma * S_forecast * I_forecast / population
            dE = best_R0 * gamma * S_forecast * I_forecast / population - sigma * E_forecast
            dI = sigma * E_forecast - gamma * I_forecast
            dR = gamma * I_forecast
            S_forecast += dS
            E_forecast += dE
            I_forecast += dI
            R_forecast += dR

    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": forecast_cases,
    }