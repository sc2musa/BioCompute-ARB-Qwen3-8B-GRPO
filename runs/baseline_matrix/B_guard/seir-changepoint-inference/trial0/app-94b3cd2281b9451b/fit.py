import numpy as np
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}."""
    
    # Convert days to time steps
    days = np.arange(len(cases))
    
    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0
    
    # Convert periods to rates
    sigma = 1 / latent_period
    gamma = 1 / infectious_period
    
    # Function to compute new cases
    def compute_new_cases(beta, change_day, R_post):
        # Initialize state variables
        S = S0
        E = E0
        I = I0
        R = R0
        
        # Track new cases
        new_cases = np.zeros(len(cases))
        
        # Time step
        dt = 1
        
        # Compute new cases for each day
        for d in range(len(cases)):
            # Compute transmission
            if d < change_day:
                beta_t = beta
            else:
                beta_t = beta * (R_post / R0)  # Assuming R_post is a fraction of R0
            
            # SEIR dynamics
            dS = -beta_t * I * S / population * dt
            dE = beta_t * I * S / population * dt - sigma * E * dt
            dI = sigma * E * dt - gamma * I * dt
            dR = gamma * I * dt
            
            # Update state variables
            S += dS
            E += dE
            I += dI
            R += dR
            
            # Record new cases (Exposed -> Infectious)
            new_cases[d] = dE / dt
        
        return new_cases
    
    # Initial guess for R0
    R0_initial = 3.0
    
    # Optimize R0
    result = minimize(lambda beta: np.sum((compute_new_cases(beta, None, 1.0) - cases) ** 2), R0_initial)
    
    # Estimate R0
    R0 = result.x[0]
    
    # Assume no intervention initially
    R_post = R0
    change_day = None
    
    # Detect intervention by comparing growth rates
    # Try different change days and find the one that minimizes the error
    best_error = float('inf')
    best_change_day = None
    best_R_post = R0
    
    # Try change days from 10 to 60
    for change_day_candidate in range(10, 60):
        # Optimize R_post for this change day
        result = minimize(lambda R_post: np.sum((compute_new_cases(R0, change_day_candidate, R_post) - cases) ** 2), R_post)
        
        # Calculate error
        error = result.fun
        
        # Update best parameters
        if error < best_error:
            best_error = error
            best_change_day = change_day_candidate
            best_R_post = result.x[0]
    
    # Update R_post and change_day
    R_post = best_R_post
    change_day = best_change_day if best_change_day is not None else None
    
    # Forecast future cases
    forecast = compute_new_cases(R0, change_day, R_post)[len(cases):len(cases) + horizon]
    
    return {
        "R0": R0,
        "R_post": R_post,
        "change_day": change_day,
        "forecast": forecast.tolist()
    }