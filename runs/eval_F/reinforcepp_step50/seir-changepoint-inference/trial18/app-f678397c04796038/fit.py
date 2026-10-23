import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse parameters
    N = population
    S0 = N - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0
    
    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        dSdt = -beta * I * S / N
        dEdt = beta * I * S / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    # Define the objective function
    def objective(params):
        beta, gamma, change_day = params
        
        # Set initial values
        y0 = [S0, E0, I0, R0]
        
        # Define the time points
        t = np.arange(1, len(cases) + 1)
        
        # Solve the ODE
        sol = solve_ivp(seir_model, [0, t[-1]], y0, t_eval=t, method="BDF",
                        args=(beta, gamma, change_day), 
                        rtol=1e-8, atol=1e-12 * N)
        
        # Calculate the objective
        objective = 0
        for d in range(len(cases)):
            expected = sol.y[2, d] - sol.y[2, d-1]
            actual = cases[d]
            objective += (expected - actual) ** 2
        return objective
    
    # Set initial guesses
    initial_guess = [3.0, 1.0, 0]
    
    # Optimize the parameters
    result = minimize(objective, initial_guess, method="L-BFGS-B",
                     bounds=[(0, 10), (0, 10), (0, 70)])
    
    # Extract the results
    beta = result.x[0]
    gamma = result.x[1]
    change_day = result.x[2]
    
    # Calculate R0 and R_post
    R0 = beta / gamma
    R_post = beta / gamma if change_day is None else beta / gamma
    
    # Forecast the next days
    forecast = []
    for d in range(len(cases), len(cases) + horizon):
        # Solve the ODE for the forecast period
        sol = solve_ivp(seir_model, [0, d], y0, t_eval=[d], method="BDF",
                        args=(beta, gamma, change_day), 
                        rtol=1e-8, atol=1e-12 * N)
        
        # Calculate the forecast
        forecast.append(sol.y[2, 0] - sol.y[2, -1])
    
    # Return the results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast
    }