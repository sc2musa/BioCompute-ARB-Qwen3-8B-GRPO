import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Parse input
    cases = np.array(cases)
    population = int(population)
    initial_infectious = float(initial_infectious)
    latent_period = float(latent_period)
    infectious_period = float(infectious_period)
    horizon = int(horizon)
    
    # Initial guess
    R0 = 1.0
    R_post = 1.0
    change_day = None
    
    # Define the SEIR model
    def seir_model(t, y):
        S, E, I, R = y
        if change_day is None:
            beta = R0 * infectious_period
        else:
            beta = R_post * infectious_period if t >= change_day else R0 * infectious_period
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - (1 / latent_period) * E
        dI_dt = (1 / latent_period) * E - (1 / infectious_period) * I
        dRdt = (1 / infectious_period) * I
        return [dSdt, dEdt, dI_dt, dRdt]
    
    # Define the objective function
    def objective(params):
        R0, R_post, change_day = params
        # Solve ODE
        sol = solve_ivp(seir_model, [0, 70], [population - initial_infectious, 0, initial_infectious, 0], 
                        method='BDF', t_eval=np.arange(70), 
                        rtol=1e-8, atol=1e-12 * population)
        # Extract results
        S, E, I, R = sol.y
        # Calculate the error
        error = 0.0
        for d in range(70):
            error += (cases[d] - I[d]) ** 2
        return error
    
    # Optimize parameters
    result = minimize(objective, [R0, R_post, 0], method='L-BFGS-B', bounds=[(0.1, 10.0), (0.1, 10.0), (0, 70)])
    R0, R_post, change_day = result.x
    
    # Forecast future cases
    def forecast(t):
        S, E, I, R = sol.y
        beta = R_post * infectious_period if t >= change_day else R0 * infectious_period
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - (1 / latent_period) * E
        dI_dt = (1 / latent_period) * E - (1 / infectious_period) * I
        dRdt = (1 / infectious_period) * I
        return [dSdt, dEdt, dI_dt, dRdt]
    
    # Calculate forecast
    forecast_results = solve_ivp(forecast, [70, 70 + horizon], [S[-1], E[-1], I[-1], R[-1]], 
                                method='BDF', t_eval=np.arange(70, 70 + horizon), 
                                rtol=1e-8, atol=1e-12 * population)
    forecast_cases = forecast_results.y[2]
    
    # Return results
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast_cases.tolist()
    }