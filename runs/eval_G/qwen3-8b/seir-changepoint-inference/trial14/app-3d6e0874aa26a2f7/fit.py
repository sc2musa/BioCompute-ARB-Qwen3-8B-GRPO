import numpy as np
from scipy.optimize import minimize

def simulate_seir(R, initial_S, initial_E, initial_I, initial_R, days, sigma, gamma, population):
    S = initial_S
    E = initial_E
    I = initial_I
    R_ = initial_R
    simulated_cases = []
    for _ in range(days):
        beta = R * gamma
        new_exposed = beta * I * S / population - sigma * E
        new_infectious = sigma * E - gamma * I
        new_susceptible = -beta * I * S / population
        new_removed = gamma * I
        # Clamp compartments to non-negative
        S = max(S + new_susceptible, 0)
        E = max(E + new_exposed, 0)
        I = max(I + new_infectious, 0)
        R_ = max(R_ + new_removed, 0)
        new_cases = sigma * E
        simulated_cases.append(max(new_cases, 0))
    return simulated_cases, S, E, I, R_

def optimize_R(observed, initial_S, initial_E, initial_I, initial_R, days, sigma, gamma, population):
    # Use first day's cases to estimate initial guess
    if days > 0:
        # Estimate R0 from first day's cases
        # new_cases = sigma * E_initial
        # E_initial = initial_E
        # beta = R * gamma
        # new_exposed = beta * I_initial * S_initial / population - sigma * E_initial
        # But I_initial is initial_infectious, S_initial is population - initial_infectious
        # Assume E_initial is 0 (initial conditions)
        # So new_exposed = beta * I_initial * S_initial / population
        # new_cases = sigma * E_initial (which is 0 initially)
        # This approach may not work for first day
        # Alternative: use average of observed cases divided by (sigma * average E)
        # But E is dynamic, so use initial E
        # Assume E_initial = 0, so first day's cases are due to initial infectious
        # This is a rough estimate
        initial_guess = observed[0] / (sigma * initial_I) * gamma
    else:
        initial_guess = 1.0
    
    def error_func(R):
        simulated, _, _, _, _ = simulate_seir(R, initial_S, initial_E, initial_I, initial_R, days, sigma, gamma, population)
        error = sum((simulated[i] - observed[i])**2 for i in range(len(simulated)))
        return error
    
    # Use L-BFGS-B with bounds to avoid invalid values
    result = minimize(error_func, x0=initial_guess, method='L-BFGS-B', bounds=[(0.1, 10.0)])
    return result.x[0]

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    best_error = float('inf')
    best_R0 = None
    best_R_post = None
    best_change_day = None
    
    # Try no intervention
    initial_S = population - initial_infectious
    initial_E = 0.0
    initial_I = initial_infectious
    initial_R = 0.0
    R0_opt = optimize_R(cases, initial_S, initial_E, initial_I, initial_R, 70, sigma, gamma, population)
    simulated_no_intervention, _, _, _, _ = simulate_seir(R0_opt, initial_S, initial_E, initial_I, initial_R, 70, sigma, gamma, population)
    error_no_intervention = sum((simulated_no_intervention[i] - cases[i])**2 for i in range(70))
    
    # Try all possible change_days
    for change_day in range(0, 71):
        if change_day == 0:
            # Already handled in no_intervention
            continue
        observed_first = cases[0:change_day]
        observed_second = cases[change_day:70]
        R0_opt = optimize_R(observed_first, initial_S, initial_E, initial_I, initial_R, change_day, sigma, gamma, population)
        simulated_first, S_final, E_final, I_final, R_final = simulate_seir(R0_opt, initial_S, initial_E, initial_I, initial_R, change_day, sigma, gamma, population)
        R_post_opt = optimize_R(observed_second, S_final, E_final, I_final, R_final, 70 - change_day, sigma, gamma, population)
        error_first = sum((simulated_first[i] - observed_first[i])**2 for i in range(len(simulated_first)))
        simulated_second, _, _, _, _ = simulate_seir(R_post_opt, S_final, E_final, I_final, R_final, 70 - change_day, sigma, gamma, population)
        error_second = sum((simulated_second[i] - observed_second[i])**2 for i in range(len(simulated_second)))
        total_error = error_first + error_second
        
        if total_error < best_error:
            best_error = total_error
            best_R0 = R0_opt
            best_R_post = R_post_opt
            best_change_day = change_day
    
    # Compare with no intervention scenario
    if error_no_intervention < best_error:
        best_error = error_no_intervention
        best_R0 = R0_opt
        best_R_post = R0_opt
        best_change_day = None
    
    # Compute forecast
    if best_change_day is None:
        R_forecast = best_R0
    else:
        R_forecast = best_R_post
    
    # Get final state after day 70
    if best_change_day is None:
        simulated_full, S_final, E_final, I_final, R_final = simulate_seir(best_R0, initial_S, initial_E, initial_I, initial_R, 70, sigma, gamma, population)
    else:
        simulated_first, S_final, E_final, I_final, R_final = simulate_seir(best_R0, initial_S, initial_E, initial_I, initial_R, best_change_day, sigma, gamma, population)
        simulated_second, _, _, _, _ = simulate_seir(best_R_post, S_final, E_final, I_final, R_final, 70 - best_change_day, sigma, gamma, population)
    
    forecast = []
    S, E, I, R = S_final, E_final, I_final, R_final
    for _ in range(horizon):
        beta = R_forecast * gamma
        new_exposed = beta * I * S / population - sigma * E
        new_infectious = sigma * E - gamma * I
        new_susceptible = -beta * I * S / population
        new_removed = gamma * I
        S = max(S + new_susceptible, 0)
        E = max(E + new_exposed, 0)
        I = max(I + new_infectious, 0)
        R = max(R + new_removed, 0)
        new_cases = sigma * E
        forecast.append(max(new_cases, 0))
    
    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day if best_change_day is not None else None,
        "forecast": forecast
    }