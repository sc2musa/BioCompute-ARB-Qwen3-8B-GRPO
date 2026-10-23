import numpy as np
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    σ = 1.0 / latent_period
    γ = 1.0 / infectious_period

    best_cost = float('inf')
    best_params = None
    best_change_day = None

    # Try all possible change_days, including None
    for change_day_candidate in [None] + list(range(1, 71)):
        if change_day_candidate is None:
            # Optimize β0
            def cost_func(params):
                β0 = params[0]
                S = population - initial_infectious
                E = 0.0
                I = initial_infectious
                R = 0.0
                predicted = []
                for day in range(1, 71):
                    new_cases = σ * E
                    predicted.append(new_cases)
                    # Update compartments
                    S_new = S - β0 * S * I / population
                    E_new = E + β0 * S * I / population - σ * E
                    I_new = I + σ * E - γ * I
                    R_new = R + γ * I
                    # Clip to non-negative values
                    S, E, I, R = max(0.0, S_new), max(0.0, E_new), max(0.0, I_new), max(0.0, R_new)
                cost = sum((predicted[i] - cases[i])**2 for i in range(70))
                return cost
            
            # Use data to estimate initial guess
            initial_guess = [np.mean(cases) * population / initial_infectious * γ * latent_period]  # Heuristic
            result = minimize(cost_func, initial_guess, method='L-BFGS-B', bounds=[(0.0, 100.0)])
            if result.success:
                current_cost = result.fun
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_params = (result.x[0],)
                    best_change_day = None
        else:
            # Optimize β0 and β1
            def cost_func(params):
                β0, β1 = params[0], params[1]
                S = population - initial_infectious
                E = 0.0
                I = initial_infectious
                R = 0.0
                predicted = []
                for day in range(1, 71):
                    if day < change_day_candidate:
                        β = β0
                    else:
                        β = β1
                    new_cases = σ * E
                    predicted.append(new_cases)
                    # Update compartments
                    S_new = S - β * S * I / population
                    E_new = E + β * S * I / population - σ * E
                    I_new = I + σ * E - γ * I
                    R_new = R + γ * I
                    # Clip to non-negative values
                    S, E, I, R = max(0.0, S_new), max(0.0, E_new), max(0.0, I_new), max(0.0, R_new)
                cost = sum((predicted[i] - cases[i])**2 for i in range(70))
                return cost
            
            # Use data to estimate initial guess
            initial_guess = [
                np.mean(cases[:change_day_candidate]) * population / initial_infectious * γ * latent_period,
                np.mean(cases[change_day_candidate-1:]) * population / initial_infectious * γ * latent_period
            ]
            # Add constraint β1 <= β0
            constraints = ({'type': 'ineq', 'fun': lambda x: x[0] - x[1]})
            result = minimize(cost_func, initial_guess, method='L-BFGS-B', bounds=[(0.0, 100.0), (0.0, 100.0)])
            if result.success:
                current_cost = result.fun
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_params = (result.x[0], result.x[1])
                    best_change_day = change_day_candidate

    # Compute R0, R_post, and change_day
    if best_change_day is None:
        β0 = best_params[0]
        R0 = β0 / γ
        R_post = R0
        change_day = None
    else:
        β0, β1 = best_params
        R0 = β0 / γ
        R_post = β1 / γ
        change_day = best_change_day

    # Apply threshold to avoid false positives
    if change_day is not None and R_post > 0.75 * R0:
        change_day = None

    # Forecast the next horizon days
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    for day in range(1, 71):
        if best_change_day is None:
            β = β0
        else:
            if day < best_change_day:
                β = β0
            else:
                β = β1
        # Update compartments
        S_new = S - β * S * I / population
        E_new = E + β * S * I / population - σ * E
        I_new = I + σ * E - γ * I
        R_new = R + γ * I
        # Clip to non-negative values
        S, E, I, R = max(0.0, S_new), max(0.0, E_new), max(0.0, I_new), max(0.0, R_new)

    forecast = []
    for day in range(71, 71 + horizon):
        if best_change_day is None:
            β = β0
        else:
            if day < best_change_day:
                β = β0
            else:
                β = β1
        new_cases = σ * E
        forecast.append(new_cases)
        # Update compartments
        S_new = S - β * S * I / population
        E_new = E + β * S * I / population - σ * E
        I_new = I + σ * E - γ * I
        R_new = R + γ * I
        # Clip to non-negative values
        S, E, I, R = max(0.0, S_new), max(0.0, E_new), max(0.0, I_new), max(0.0, R_new)

    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast
    }