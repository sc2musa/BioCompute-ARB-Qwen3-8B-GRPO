import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def simulate_seir(population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day, days=70):
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    if change_day is None:
        beta = lambda t: R0 * gamma
    else:
        beta = lambda t: R0 * gamma if t < change_day else R_post * gamma
    
    def seir_model(t, y):
        S, E, I, R = y
        dSdt = -beta(t) * S * I / population
        dEdt = beta(t) * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    t_span = (0, days)
    t_eval = np.arange(0, days + 1)
    y0 = [population - initial_infectious, 0, initial_infectious, 0]
    sol = solve_ivp(seir_model, t_span, y0, t_eval=t_eval)
    S, E, I, R = sol.y
    expected_cases = np.zeros(days)
    for d in range(days):
        expected_cases[d] = I[d+1] - I[d]
    return S, E, I, R, expected_cases

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    best_loss = float('inf')
    best_params = None
    
    # Try all possible change_days
    for change_day_candidate in range(1, 71):
        def loss(params):
            R0, R_post = params
            expected, _ = simulate_seir(population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day_candidate, days=70)
            loss_val = np.sum((np.array(expected) - np.array(cases)) ** 2)
            return loss_val
        
        initial_guess = [1.0, 1.0]
        # Remove bounds to avoid unpacking error
        result = minimize(loss, initial_guess, method='BFGS')
        if result.success:
            current_loss = result.fun
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = (result.x[0], result.x[1], change_day_candidate)
    
    # Handle no intervention case
    def loss_no_intervention(params):
        R0, R_post = params
        expected, _ = simulate_seir(population, initial_infectious, latent_period, infectious_period, R0, R_post, change_day=None, days=70)
        loss_val = np.sum((np.array(expected) - np.array(cases)) ** 2)
        return loss_val
    
    initial_guess_no_intervention = [1.0, 1.0]
    result_no_intervention = minimize(loss_no_intervention, initial_guess_no_intervention, method='BFGS')
    if result_no_intervention.success:
        current_loss = result_no_intervention.fun
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = (result_no_intervention.x[0], result_no_intervention.x[1], None)
    
    # Extract best parameters
    if best_params is None:
        R0_best = 1.0
        R_post_best = 1.0
        change_day_best = None
    else:
        R0_best, R_post_best, change_day_best = best_params
    
    # Generate forecast
    days_forecast = 70 + horizon
    _, _, _, _, expected_forecast = simulate_seir(population, initial_infectious, latent_period, infectious_period, R0_best, R_post_best, change_day_best, days=days_forecast)
    forecast = expected_forecast[70:70 + horizon].tolist()
    
    return {
        "R0": R0_best,
        "R_post": R_post_best,
        "change_day": change_day_best,
        "forecast": forecast
    }