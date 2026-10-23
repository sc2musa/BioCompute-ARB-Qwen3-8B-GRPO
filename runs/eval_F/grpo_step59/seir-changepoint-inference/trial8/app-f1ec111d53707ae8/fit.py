import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse input
    cases = np.array(cases)
    population = int(population)
    initial_infectious = float(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)
    
    # Initial state
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    
    # Define parameters
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period
    
    # Define the SEIR model
    def seir_model(t, y, beta, gamma, change_day):
        S, E, I, R = y
        dSdt = -beta * I * S / population
        dEdt = beta * I * S / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Define the objective function
    def objective(params):
        beta, gamma, change_day = params
        
        # Initialize the state
        y0 = [S0, E0, I0, R0]
        
        # Create time array
        t = np.arange(1, 71)
        
        # Solve the ODE
        sol = solve_ivp(seir_model, [0, 70], y0, t_eval=t, method="BDF",
                        args=(beta, gamma, change_day), 
                        rtol=1e-8, atol=1e-12 * population)
        
        # Calculate the forecast
        forecast = sol.y[2, 70:] * population
        
        # Calculate the error
        error = np.sum((cases - forecast) ** 2) / np.sum(cases ** 2)
        
        return error
    
    # Set initial guess
    initial_guess = [3.0, 1.0, 0.0]
    
    # Optimize the parameters
    result = minimize(objective, initial_guess, method="L-BFGS-B",
                     bounds=[(0.1, 10.0), (0.1, 10.0), (0.0, 70.0)])
    
    # Extract the results
    beta_opt = result.x[0]
    gamma_opt = result.x[1]
    change_day_opt = result.x[2]
    
    # Calculate R0 and R_post
    R0 = beta_opt / gamma_opt
    R_post = R0 if change_day_opt is None else beta_opt / gamma_opt
    
    # Generate the forecast
    forecast = np.zeros(horizon)
    for day in range(71, 71 + horizon):
        # Calculate the state
        y = solve_ivp(seir_model, [0, day], y0, t_eval=[day], method="BDF",
                      args=(beta_opt, gamma_opt, change_day_opt), 
                      rtol=1e-8, atol=1e-12 * population)
        forecast[day - 71] = y.y[2, 0] * population
    
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day_opt,
        "forecast": forecast.tolist()
    }