import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    """Return {"R0": float, "R_post": float, "change_day": float | None, "forecast": list}.
    """
    # Convert days to time steps
    days = np.arange(len(cases))
    
    # Initial conditions
    S0 = population - initial_infectious
    E0 = 0
    I0 = initial_infectious
    R0 = 0
    
    # Parameters
    gamma = 1.0 / infectious_period
    sigma = 1.0 / latent_period
    
    # Define the SEIR model
    def seir_model(t, S, E, I, R, beta):
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt
    
    # Function to compute the expected cases
    def expected_cases(beta, change_day):
        # Simulate the SEIR model
        t = np.arange(len(cases))
        S = np.zeros(len(cases))
        E = np.zeros(len(cases))
        I = np.zeros(len(cases))
        R = np.zeros(len(cases))
        
        S[0] = S0
        E[0] = E0
        I[0] = I0
        R[0] = R0
        
        for i in range(1, len(cases)):
            # Compute the derivatives
            dSdt, dEdt, dIdt, dRdt = seir_model(t[i-1], S[i-1], E[i-1], I[i-1], R[i-1], beta)
            
            # Update the states
            S[i] = S[i-1] + dSdt
            E[i] = E[i-1] + dEdt
            I[i] = I[i-1] + dIdt
            R[i] = R[i-1] + dRdt
            
        # Compute the expected cases
        expected = np.zeros(len(cases))
        for i in range(len(cases)):
            expected[i] = I[i] - I[i-1] if i > 0 else 0
        
        # Apply the intervention
        if change_day is not None:
            beta_after = beta * (R_post / R0)
            for i in range(int(change_day), len(cases)):
                # Compute the derivatives
                dSdt, dEdt, dIdt, dRdt = seir_model(t[i-1], S[i-1], E[i-1], I[i-1], R[i-1], beta_after)
                
                # Update the states
                S[i] = S[i-1] + dSdt
                E[i] = E[i-1] + dEdt
                I[i] = I[i-1] + dIdt
                R[i] = R[i-1] + dRdt
                
        return expected
    
    # Function to compute the negative log-likelihood
    def negative_log_likelihood(params):
        R0_est, R_post_est, change_day_est = params
        
        # Estimate beta
        beta = R0_est * gamma
        
        # Compute the expected cases
        expected = expected_cases(beta, change_day_est)
        
        # Compute the negative log-likelihood
        # Assuming Poisson distribution for the cases
        return -np.sum(np.log(expected) - cases)
    
    # Initial guess for parameters
    initial_guess = [2.0, 1.0, None]
    
    # Optimize the parameters
    result = minimize(negative_log_likelihood, initial_guess, method='L-BFGS-B')
    
    # Extract the estimated parameters
    R0_est = result.x[0]
    R_post_est = result.x[1]
    change_day_est = result.x[2]
    
    # Forecast the next horizon days
    forecast = np.zeros(horizon)
    
    # Simulate the SEIR model for the forecast period
    t = np.arange(len(cases), len(cases) + horizon)
    S = np.zeros(len(t))
    E = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    
    S[0] = S[-1]
    E[0] = E[-1]
    I[0] = I[-1]
    R[0] = R[-1]
    
    for i in range(1, len(t)):
        # Compute the derivatives
        dSdt, dEdt, dIdt, dRdt = seir_model(t[i-1], S[i-1], E[i-1], I[i-1], R[i-1], beta)
        
        # Update the states
        S[i] = S[i-1] + dSdt
        E[i] = E[i-1] + dEdt
        I[i] = I[i-1] + dIdt
        R[i] = R[i-1] + dRdt
        
    # Compute the expected cases for the forecast period
    for i in range(len(t)):
        forecast[i] = I[i] - I[i-1] if i > 0 else 0
    
    # Apply the intervention if change_day_est is not None
    if change_day_est is not None:
        beta_after = beta * (R_post_est / R0_est)
        for i in range(int(change_day_est), len(t)):
            # Compute the derivatives
            dSdt, dEdt, dIdt, dRdt = seir_model(t[i-1], S[i-1], E[i-1], I[i-1], R[i-1], beta_after)
            
            # Update the states
            S[i] = S[i-1] + dSdt
            E[i] = E[i-1] + dEdt
            I[i] = I[i-1] + dIdt
            R[i] = R[i-1] + dRdt
            
    return {
        "R0": R0_est,
        "R_post": R_post_est,
        "change_day": change_day_est,
        "forecast": forecast.tolist()
    }