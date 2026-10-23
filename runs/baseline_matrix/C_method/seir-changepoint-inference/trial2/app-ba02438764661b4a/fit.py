import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert periods to rates
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    # Define SEIR model with change point
    def seir_model(y, t, beta, change_day):
        S, E, I, R = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        
        # Handle change point
        if t >= change_day:
            beta = beta_post
        
        return [dSdt, dEdt, dIdt, dRdt]

    # Negative binomial log-likelihood
    def neg_binom_log_likelihood(params, cases, population, sigma, gamma, change_day):
        beta, beta_post = params
        
        # Initial conditions
        S0 = population - initial_infectious
        E0 = 0.0
        I0 = initial_infectious
        R0 = 0.0
        y0 = [S0, E0, I0, R0]

        # Time points
        t = np.arange(0, len(cases) + 1)

        # Solve ODE
        sol = odeint(seir_model, y0, t, args=(beta, change_day))
        C = sol[:, 3]  # Cumulative cases
        expected_cases = C[1:] - C[:-1]

        # Calculate log-likelihood
        # Using negative binomial with dispersion around 10-50
        dispersion = 50.0
        log_likelihood = np.sum(np.log(np.random.negative_binomial(expected_cases, dispersion)))
        
        return -log_likelihood

    # Try different change points
    best_nll = float('inf')
    best_params = None
    best_change_day = None
    
    # Try no intervention model
    def no_intervention_nll(params):
        beta = params
        return neg_binom_log_likelihood((beta, beta), cases, population, sigma, gamma, None)

    result_no_intervention = minimize(no_intervention_nll, x0=[3.0], method='L-BFGS-B')
    best_params = result_no_intervention.x
    best_nll = result_no_intervention.fun
    best_change_day = None

    # Try with intervention
    for change_day in range(1, len(cases) + 1):
        # Fit R0 and R_post
        def intervention_nll(params):
            beta, beta_post = params
            return neg_binom_log_likelihood((beta, beta_post), cases, population, sigma, gamma, change_day)

        result = minimize(intervention_nll, x0=[3.0, 3.0], method='L-BFGS-B')
        if result.fun < best_nll:
            best_nll = result.fun
            best_params = result.x
            best_change_day = change_day

    # Compare with no-intervention model using BIC
    k_no_intervention = 1
    k_intervention = 2
    bic_no_intervention = 2 * best_nll + k_no_intervention * np.log(len(cases))
    bic_intervention = 2 * best_nll + k_intervention * np.log(len(cases))

    if bic_intervention > bic_no_intervention:
        # Prefer no-intervention model
        beta = best_params[0]
        beta_post = beta
        change_day = None
    else:
        beta, beta_post = best_params
        change_day = best_change_day

    # Forecast
    def forecast(beta, beta_post, change_day):
        # Initial conditions
        S0 = population - initial_infectious
        E0 = 0.0
        I0 = initial_infectious
        R0 = 0.0
        y0 = [S0, E0, I0, R0]

        # Time points
        t = np.arange(0, len(cases) + horizon + 1)

        # Solve ODE
        sol = odeint(seir_model, y0, t, args=(beta, change_day))
        C = sol[:, 3]  # Cumulative cases
        expected_cases = C[70:] - C[69:-1]
        return expected_cases.tolist()

    forecast = forecast(beta, beta_post, change_day)

    return {
        "R0": beta,
        "R_post": beta_post,
        "change_day": change_day,
        "forecast": forecast
    }