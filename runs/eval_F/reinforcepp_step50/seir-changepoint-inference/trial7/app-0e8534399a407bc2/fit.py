import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse inputs
    cases = np.array(cases)
    population = int(population)
    initial_infectious = float(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)
    
    # Initial parameters
    R0 = 2.0
    R_post = 2.0
    change_day = 70  # Default to after the observation period
    
    # Define the SEIR model
    def seir_model(t, S, E, I, R):
        # Calculate the transmission rate
        if t < change_day:
            beta = R0 * infectious_period
        else:
            beta = R_post * infectious_period
        
        # Calculate the transition rates
        sigma = 1.0 / latent_period
        gamma = 1.0 / infectious_period
        
        # Calculate the derivatives
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Define the objective function
    def objective(params):
        R0, R_post, change_day = params
        
        # Initial state
        S0 = population - initial_infectious
        E0 = 0.0
        I0 = initial_infectious
        R0 = 0.0
        
        # Solve the ODE
        sol = solve_ivp(seir_model, [0, 70], [S0, E0, I0, R0], t_eval=np.arange(70), method="BDF",
                        rtol=1e-8, atol=1e-12 * population)
        
        # Calculate the log-likelihood
        log_likelihood = 0.0
        for t in range(70):
            # Calculate the expected cases
            expected_cases = sol.y[2, t] * population
            
            # Calculate the log-likelihood
            log_likelihood += np.log(expected_cases) - expected_cases
            
        return -log_likelihood
    
    # Optimize the parameters
    result = minimize(objective, [R0, R_post, change_day], method="L-BFGS-B",
                     bounds=[(1.0, 10.0), (1.0, 10.0), (0, 70)])
    
    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Forecast the future cases
    forecast = []
    for t in range(70, 70 + horizon):
        # Calculate the expected cases
        expected_cases = sol.y[2, t] * population
        
        forecast.append(expected_cases)
    
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast
    }