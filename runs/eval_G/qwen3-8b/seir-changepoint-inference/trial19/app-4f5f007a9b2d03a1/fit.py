import numpy as np
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    # Estimate initial R0 without intervention
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    predicted = []
    for d in range(1, 71):
        beta = 1.0 * gamma
        dS = -beta * S * I / population
        dE = beta * S * I / population - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        new_cases = sigma * E
        predicted.append(new_cases)
        S += dS
        E += dE
        I += dI
        R += dR
    # Use first 70 days to estimate R0
    # This is a placeholder; actual estimation would require more sophisticated method
    initial_R0_guess = 2.0
    initial_R_post_guess = initial_R0_guess

    # Optimize R0 without intervention
    def error_no_intervention(params):
        R0 = params[0]
        S = population - initial_infectious
        E = 0.0
        I = initial_infectious
        R = 0.0
        predicted = []
        for d in range(1, 71):
            beta = R0 * gamma
            dS = -beta * S * I / population
            dE = beta * S * I / population - sigma * E
            dI = sigma * E - gamma * I
            dR = gamma * I
            new_cases = sigma * E
            predicted.append(new_cases)
            S += dS
            E += dE
            I += dI
            R += dR
        # Negative log-likelihood for Poisson
        error_sum = 0.0
        for c, p in zip(cases, predicted):
            error_sum += c * np.log(p + 1e-10) - p
        return error_sum

    result_no_intervention = minimize(error_no_intervention, [initial_R0_guess], bounds=[(0.001, 10.0)])
    initial_R0_guess = result_no_intervention.x[0]

    best_error = float('inf')
    best_R0 = best_R_post = initial_R0_guess
    best_change_day = 71  # Default to no intervention

    for change_day in range(0, 71):
        def error(params):
            R0, R_post = params
            S = population - initial_infectious
            E = 0.0
            I = initial_infectious
            R = 0.0
            predicted = []
            for d in range(1, 71):
                beta = R0 * gamma if d <= change_day else R_post * gamma
                dS = -beta * S * I / population
                dE = beta * S * I / population - sigma * E
                dI = sigma * E - gamma * I
                dR = gamma * I
                new_cases = sigma * E
                predicted.append(new_cases)
                S += dS
                E += dE
                I += dI
                R += dR
            # Negative log-likelihood for Poisson
            error_sum = 0.0
            for c, p in zip(cases, predicted):
                error_sum += c * np.log(p + 1e-10) - p
            return error_sum

        initial_guess = [initial_R0_guess, initial_R0_guess * 0.5]
        bounds = [(0.001, 10.0), (0.001, initial_R0_guess)]
        result = minimize(error, initial_guess, bounds=bounds)
        if result.success:
            current_error = result.fun
            if current_error < best_error:
                best_error = current_error
                best_R0, best_R_post = result.x
                best_change_day = change_day

    # Handle no intervention case
    if best_change_day == 71 or abs(best_R_post - best_R0) < 0.05:
        best_change_day = None

    # Forecast
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    for d in range(1, 71):
        beta = best_R0 * gamma if d <= best_change_day else best_R_post * gamma
        dS = -beta * S * I / population
        dE = beta * S * I / population - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        new_cases = sigma * E
        S += dS
        E += dE
        I += dI
        R += dR

    forecast = []
    for day in range(71, 71 + horizon):
        beta = best_R0 * gamma if day <= best_change_day else best_R_post * gamma
        dS = -beta * S * I / population
        dE = beta * S * I / population - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        new_cases = sigma * E
        forecast.append(new_cases)
        S += dS
        E += dE
        I += dI
        R += dR

    return {
        "R0": best_R0,
        "R_post": best_R_post,
        "change_day": best_change_day,
        "forecast": forecast
    }