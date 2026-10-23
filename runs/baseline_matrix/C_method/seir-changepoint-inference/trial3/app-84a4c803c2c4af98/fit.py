import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Define the SEIR model
    def seir_model(y, t, beta, sigma, gamma):
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    # Define the negative log-likelihood function
    def nll(params, cases, population, initial_infectious, latent_period, infectious_period):
        R0, R_post, change_day = params
        # Convert to rates
        sigma = 1 / latent_period
        gamma = 1 / infectious_period
        # Define beta(t)
        def beta(t):
            if t < change_day:
                return R0 * gamma
            else:
                return R_post * gamma
        
        # Integrate the SEIR model
        t = np.arange(0, len(cases) + 1)
        y0 = [population - initial_infectious, 0, initial_infectious, 0]
        sol = odeint(seir_model, y0, t, args=(beta, sigma, gamma))
        
        # Calculate expected cases
        expected_cases = sol[:, 2][1:] - sol[:, 2][:-1]
        
        # Calculate the negative log-likelihood
        # We use a Poisson likelihood for simplicity
        nll = -np.sum(np.log(expected_cases) - cases)
        return nll

    # Optimize the parameters
    # We use a grid search for the change_day
    best_nll = float('inf')
    best_params = None
    best_change_day = None
    
    # Try all possible change days
    for change_day in range(1, len(cases) + 1):
        # Define the negative log-likelihood function with fixed change_day
        def nll_fixed(params):
            R0, R_post = params
            return nll((R0, R_post, change_day), cases, population, initial_infectious, latent_period, infectious_period)
        
        # Optimize using Nelder-Mead
        result = minimize(nll_fixed, x0=[1.0, 1.0], method='Nelder-Mead')
        
        # Update best parameters
        if result.fun < best_nll:
            best_nll = result.fun
            best_params = result.x
            best_change_day = change_day
    
    # Also try the no-intervention model
    def nll_no_intervention(params):
        R0, = params
        return nll((R0, R0, None), cases, population, initial_infectious, latent_period, infectious_period)
    
    result_no_intervention = minimize(nll_no_intervention, x0=[1.0], method='Nelder-Mead')
    
    # Compare the models using BIC
    # For the change point model, k = 3 (R0, R_post, change_day)
    # For the no-intervention model, k = 1 (R0)
    bic_change = 2 * best_nll + 3 * np.log(len(cases))
    bic_no_intervention = 2 * result_no_intervention.fun + 1 * np.log(len(cases))
    
    # Choose the best model
    if bic_change < bic_no_intervention:
        R0, R_post = best_params
        change_day = best_change_day
    else:
        R0 = result_no_intervention.x[0]
        R_post = R0
        change_day = None
    
    # Forecast the next days
    sigma = 1 / latent_period
    gamma = 1 / infectious_period
    def beta(t):
        if t < change_day:
            return R0 * gamma
        else:
            return R_post * gamma
    
    t = np.arange(0, len(cases) + horizon + 1)
    y0 = [population - initial_infectious, 0, initial_infectious, 0]
    sol = odeint(seir_model, y0, t, args=(beta, sigma, gamma))
    
    # Calculate expected cases for the forecast period
    expected_forecast = sol[:, 2][len(cases):] - sol[:, 2][len(cases)-1:]
    
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": expected_forecast.tolist()
    }