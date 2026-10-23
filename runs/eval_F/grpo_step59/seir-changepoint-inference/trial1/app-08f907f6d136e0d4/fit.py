import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse inputs
    cases = np.array(cases)
    population = int(population)
    initial_infectious = float(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)
    
    # Initial state
    S = population - initial_infectious
    E = 0.0
    I = initial_infectious
    R = 0.0
    
    # Define the SEIR model
    def seir_model(t, y, beta):
        S, E, I, R = y
        dSdt = -beta * I * S / population
        dEdt = beta * I * S / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])
    
    # Initial guess for beta
    beta_initial = 1.0
    
    # Optimize beta
    def objective(beta):
        # Solve the ODE
        sol = solve_ivp(seir_model, [0, 70], [S, E, I, R], args=(beta,), method="BDF",
                        rtol=1e-8, atol=1e-12 * population)
        
        # Calculate the log-likelihood
        log_likelihood = 0.0
        for d in range(70):
            # Calculate expected cases
            expected = sol.y[2, d] - sol.y[2, d-1] if d > 0 else sol.y[2, 0]
            # Calculate log-likelihood
            log_likelihood += np.log(expected) if expected > 0 else 0.0
        
        return -log_likelihood
    
    # Set up optimization
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    
    # Optimize beta
    beta_opt = minimize(objective, beta_initial, method="L-BFGS-B",
                        bounds=[(0.0, 1000.0)], tol=1e-8)
    
    # Get the optimal beta
    beta = beta_opt.x[0]
    
    # Calculate R0
    R0 = beta / gamma
    
    # Detect change point
    change_day = None
    
    # Forecast
    forecast = []
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R0,
        "change_day": change_day,
        "forecast": forecast
    }