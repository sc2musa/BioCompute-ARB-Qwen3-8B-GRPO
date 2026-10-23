import numpy as np
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    
    def simulate(R0, R_post, change_day):
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        simulated_cases = []
        for day in range(70):
            if day < change_day:
                beta = R0 * gamma
            else:
                beta = R_post * gamma
            # Compute new cases as sigma * E (E at the start of the day)
            new_cases = sigma * E
            simulated_cases.append(new_cases)
            # Update state variables
            dS = -beta * S * I / population
            dE = beta * S * I / population - sigma * E
            dI = sigma * E - gamma * I
            dR = gamma * I
            S += dS
            E += dE
            I += dI
            R += dR
        return simulated_cases

    # Try all possible change_days
    best_error = float('inf')
    best_R0 = None
    best_R_post = None
    best_change_day = None

    for change_day in range(0, 71):
        # Split data into two periods
        period1_days = change_day
        period2_days = 70 - change_day
        
        # Fit R0 for period1
        def objective_R0(R0):
            simulated = simulate(R0, R0, change_day)
            error = np.sum((np.array(simulated[:period1_days]) - np.array(cases[:period1_days]))**2)
            return error
        result_R0 = minimize(objective_R0, x0=1.0)
        R0 = result_R0.x[0]
        
        # Fit R_post for period2
        def objective_R_post(R_post):
            simulated = simulate(R0, R_post, change_day)
            error = np.sum((np.array(simulated[period1_days:]) - np.array(cases[period1_days:]))**2)
            return error
        result_R_post = minimize(objective_R_post, x0=1.0)
        R_post = result_R_post.x[0]
        
        # Compute total error
        simulated = simulate(R0, R_post, change_day)
        error = np.sum((np.array(simulated) - np.array(cases))**2)
        if error < best_error:
            best_error = error
            best_R0 = R0
            best_R_post = R_post
            best_change_day = change_day

    # Forecast
    forecast = []
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    for _ in range(horizon):
        if _ < best_change_day:
            beta = best_R0 * gamma
        else:
            beta = best_R_post * gamma
        # Compute new cases as sigma * E (E at the start of the day)
        new_cases = sigma * E
        forecast.append(new_cases)
        # Update state variables
        dS = -beta * S * I / population
        dE = beta * S * I / population - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        S += dS
        E += dE
        I += dI
        R += dR
    
    return {
        "R0": float(best_R0),
        "R_post": float(best_R_post),
        "change_day": float(best_change_day) if best_change_day is not None else None,
        "forecast": forecast
    }