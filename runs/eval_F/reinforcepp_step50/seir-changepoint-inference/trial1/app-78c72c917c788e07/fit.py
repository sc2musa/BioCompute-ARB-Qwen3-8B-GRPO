import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Convert to numpy arrays
    cases = np.array(cases)
    
    # Define the SEIR model
    def seir_model(params):
        R0, R_post, change_day = params
        
        # Initial conditions
        S0 = population - initial_infectious
        E0 = 0
        I0 = initial_infectious
        R0 = 0
        
        # Calculate the transmission rates
        beta = np.zeros(70)
        beta[0:change_day] = R0 * (1 / infectious_period)
        beta[change_day:] = R_post * (1 / infectious_period)
        
        # Solve the ODE
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            lambda t, y: [S0 - beta[int(t)] * y[1], beta[int(t)] * y[1] - (1 / latent_period) * y[2], (1 / latent_period) * y[2] - (1 / infectious_period) * y[3], (1 / infectious_period) * y[3]],
            [0, 70],
            [S0, E0, I0, R0],
            t_eval=np.arange(70)
        )
        
        # Calculate the predicted cases
        predicted_cases = sol.y[2] * (1 / infectious_period)
        
        return predicted_cases
        
    # Initial guess
    initial_guess = [3.0, 3.0, 30]
    
    # Optimize
    result = minimize(lambda params: np.sum((seir_model(params) - cases) ** 2), initial_guess)
    
    # Extract the results
    R0 = result.x[0]
    R_post = result.x[1]
    change_day = result.x[2]
    
    # Forecast
    forecast = np.zeros(horizon)
    for t in range(horizon):
        # Calculate the transmission rates
        beta = np.zeros(70)
        beta[0:change_day] = R0 * (1 / infectious_period)
        beta[change_day:] = R_post * (1 / infectious_period)
        
        # Solve the ODE
        sol = solve_ivp(
            lambda t, y: [S0 - beta[int(t)] * y[1], beta[int(t)] * y[1] - (1 / latent_period) * y[2], (1 / latent_period) * y[2] - (1 / infectious_period) * y[3], (1 / infectious_period) * y[3]],
            [0, 70 + t],
            [S0, E0, I0, R0],
            t_eval=np.arange(70 + t)
        )
        
        # Calculate the predicted cases
        predicted_cases = sol.y[2] * (1 / infectious_period)
        
        # Store the forecast
        forecast[t] = predicted_cases[-1]
    
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }