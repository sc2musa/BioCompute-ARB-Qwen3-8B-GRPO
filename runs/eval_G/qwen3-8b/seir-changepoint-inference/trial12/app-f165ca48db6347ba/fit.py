import numpy as np
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    best_error = float('inf')
    best_change_day = 0
    best_beta_before = 0.0
    best_beta_after = 0.0
    best_state_after_change = (0.0, 0.0, 0.0, 0.0)

    for change_day in range(0, 71):
        before_cases = cases[:change_day]
        after_cases = cases[change_day:]

        # Optimize beta_before
        if len(before_cases) > 0:
            def objective_before(beta):
                S = population - initial_infectious
                E = 0.0
                I = initial_infectious
                R = 0.0
                error = 0.0
                sigma = 1.0 / latent_period
                gamma = 1.0 / infectious_period
                for _ in range(len(before_cases)):
                    new_infections = beta * I * S / population
                    S = max(S - new_infections, 0.0)
                    E_new = E + new_infections - sigma * E
                    I_new = I + sigma * E - gamma * I
                    R_new = R + gamma * I
                    new_cases = sigma * E
                    new_cases = max(new_cases, 0.0)
                    error += (new_cases - before_cases[_])**2
                    E, I, R = E_new, I_new, R_new
                return error
            result_before = minimize(objective_before, x0=3.0, bounds=[(0.0, 100.0)], method='L-BFGS-B')
            beta_before = result_before.x[0]
        else:
            beta_before = 0.0

        # Optimize beta_after
        if len(after_cases) > 0:
            def objective_after(beta):
                S = population - initial_infectious
                E = 0.0
                I = initial_infectious
                R = 0.0
                error = 0.0
                sigma = 1.0 / latent_period
                gamma = 1.0 / infectious_period
                for _ in range(len(after_cases)):
                    new_infections = beta * I * S / population
                    S = max(S - new_infections, 0.0)
                    E_new = E + new_infections - sigma * E
                    I_new = I + sigma * E - gamma * I
                    R_new = R + gamma * I
                    new_cases = sigma * E
                    new_cases = max(new_cases, 0.0)
                    error += (new_cases - after_cases[_])**2
                    E, I, R = E_new, I_new, R_new
                return error
            result_after = minimize(objective_after, x0=3.0, bounds=[(0.0, 100.0)], method='L-BFGS-B')
            beta_after = result_after.x[0]
        else:
            beta_after = 0.0

        # Calculate total error
        total_error = 0.0
        if len(before_cases) > 0:
            total_error += result_before.fun
        if len(after_cases) > 0:
            total_error += result_after.fun

        # Update best parameters
        if total_error < best_error:
            best_error = total_error
            best_change_day = change_day
            best_beta_before = beta_before
            best_beta_after = beta_after

            # Simulate to get final state
            S = population - initial_infectious
            E = 0.0
            I = initial_infectious
            R = 0.0
            for _ in range(change_day):
                new_infections = best_beta_before * I * S / population
                S = max(S - new_infections, 0.0)
                E_new = E + new_infections - (1.0/latent_period)*E
                I_new = I + (1.0/latent_period)*E - (1.0/infectious_period)*I
                R_new = R + (1.0/infectious_period)*I
                E, I, R = E_new, I_new, R_new
            best_state_after_change = (S, E, I, R)

    # Compute R0 and R_post
    gamma = 1.0 / infectious_period
    if best_change_day == 0:
        R0 = best_beta_after / gamma
        R_post = R0
    elif best_change_day == 70:
        R0 = best_beta_before / gamma
        R_post = R0
    else:
        R0 = best_beta_before / gamma
        R_post = best_beta_after / gamma

    # Generate forecast
    forecast = []
    S, E, I, R = best_state_after_change
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    for _ in range(horizon):
        new_infections = best_beta_after * I * S / population
        S = max(S - new_infections, 0.0)
        E_new = E + new_infections - sigma * E
        I_new = I + sigma * E - gamma * I
        R_new = R + gamma * I
        new_cases = sigma * E
        new_cases = max(new_cases, 0.0)
        forecast.append(new_cases)
        E, I, R = E_new, I_new, R_new

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": best_change_day,
        "forecast": forecast
    }