import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Convert parameters to rates
    sigma = 1.0 / latent_period
    gamma = 1.0 / infectious_period

    # Initial conditions: S, E, I, R, C
    S0 = population - initial_infectious
    E0 = 0.0
    I0 = initial_infectious
    R0 = 0.0
    C0 = 0.0
    y0 = [S0, E0, I0, R0, C0]

    # Define the SEIR model with cumulative incidence
    def seir_model(y, t, beta, sigma, gamma):
        S, E, I, R, C = y
        dSdt = -beta * S * I / population
        dEdt = beta * S * I / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        dCdt = sigma * E
        return [dSdt, dEdt, dIdt, dRdt, dCdt]

    # Negative binomial log-likelihood
    def neg_binom_log_likelihood(params, cases, sigma, gamma):
        R0, R_post = params
        beta = R0 * gamma
        beta_post = R_post * gamma

        # Fit the model with the change day
        # We will try different change days and find the one with the best likelihood
        # For simplicity, we will try all possible change days from 1 to 70
        best_likelihood = -np.inf
        best_R0 = R0
        best_R_post = R_post
        best_change_day = None

        for change_day in range(1, 71):
            # Integrate the model with the change day
            t = np.arange(0, 70 + 1)
            beta_array = np.full(70, beta)
            beta_array[change_day - 1:] = beta_post

            # Solve the ODE
            sol = odeint(seir_model, y0, t, args=(beta_array, sigma, gamma))

            # Calculate the expected cases
            expected_cases = sol[:, 4][1:] - sol[:, 4][:-1]

            # Calculate the negative binomial log-likelihood
            dispersion = 10.0
            log_likelihood = np.sum(np.log(1.0 / (np.sqrt(2 * np.pi * dispersion * expected_cases))) - (cases - expected_cases)**2 / (2 * dispersion * expected_cases))

            # Update the best parameters
            if log_likelihood > best_likelihood:
                best_likelihood = log_likelihood
                best_R0 = R0
                best_R_post = R_post
                best_change_day = change_day

        # Fit the no-intervention model
        beta_no_intervention = R0 * gamma
        t_no_intervention = np.arange(0, 70 + 1)
        beta_array_no_intervention = np.full(70, beta_no_intervention)

        sol_no_intervention = odeint(seir_model, y0, t_no_intervention, args=(beta_array_no_intervention, sigma, gamma))
        expected_cases_no_intervention = sol_no_intervention[:, 4][1:] - sol_no_intervention[:, 4][:-1]

        # Calculate the negative binomial log-likelihood for the no-intervention model
        log_likelihood_no_intervention = np.sum(np.log(1.0 / (np.sqrt(2 * np.pi * dispersion * expected_cases_no_intervention))) - (cases - expected_cases_no_intervention)**2 / (2 * dispersion * expected_cases_no_intervention))

        # Compare the models using BIC
        k_change_point = 3  # R0, R_post, change_day
        k_no_intervention = 1  # R0
        bic_change_point = -2 * log_likelihood + k_change_point * np.log(len(cases))
        bic_no_intervention = -2 * log_likelihood_no_intervention + k_no_intervention * np.log(len(cases))

        # Choose the best model
        if bic_change_point < bic_no_intervention:
            R0 = best_R0
            R_post = best_R_post
            change_day = best_change_day
        else:
            R_post = R0
            change_day = None

        # Forecast the next weeks
        t_forecast = np.arange(70, 70 + horizon + 1)
        beta_forecast = np.full(horizon, R_post * gamma)

        sol_forecast = odeint(seir_model, y0, t_forecast, args=(beta_forecast, sigma, gamma))
        forecast = sol_forecast[:, 4][1:] - sol_forecast[:, 4][:-1]

        return {
            R0: R0,
            R_post: R_post,
            change_day: change_day,
            forecast: forecast.tolist()
        }
