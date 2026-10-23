import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


def fit(cases, population, initial_infectious, latent_period, infectious_period, horizon):
    # Define the SEIR model
    def seir_model(t, S, E, I, R, gamma):
        sigma = 1.0 / latent_period
        beta = R0 * gamma
        beta_post = R_post * gamma
        dSdt = -beta * I * S / population
        dEdt = beta * I * S / population - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    # Define the objective function
    def objective(params):
        R0, R_post, change_day = params
        gamma = 1.0 / infectious_period
        # Simulate the SEIR model
        beta = R0 * gamma
        beta_post = R_post * gamma
        # Simulate the model before and after the change day
        # Initial conditions
        S0 = population - initial_infectious
        E0 = 0.0
        I0 = initial_infectious
        R0_val = 0.0
        # Simulate before the change day
        t_span = (0, change_day)
        t_eval = np.linspace(0, change_day, len(cases))
        sol_before = solve_ivp(seir_model, t_span, (S0, E0, I0, R0_val), t_eval=t_eval, args=(gamma,))
        # Simulate after the change day
        t_span_after = (change_day, 70)
        t_eval_after = np.linspace(change_day, 70, len(cases) - int(change_day))
        sol_after = solve_ivp(seir_model, t_span_after, (sol_before.y[0][-1], sol_before.y[1][-1], sol_before.y[2][-1], sol_before.y[3][-1]), t_eval=t_eval_after, args=(gamma,))
        # Extract the expected cases
        expected_cases = np.zeros(len(cases))
        expected_cases[:int(change_day)] = sol_before.y[2][1:]  # Infectious cases
        expected_cases[int(change_day):] = sol_after.y[2][1:]  # Infectious cases
        # Calculate the objective function
        objective_value = np.sum((cases - expected_cases) ** 2)
        return objective_value

    # Initial guess
    initial_guess = [3.0, 3.0, 30.0]

    # Optimize
    result = minimize(objective, initial_guess)

    # Return the results
    return {
        "R0": result.x[0],
        "R_post": result.x[1],
        "change_day": result.x[2],
        "forecast": []
    }
