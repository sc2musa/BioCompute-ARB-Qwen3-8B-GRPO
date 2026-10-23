import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network["species"]
    reactions = network["reactions"]
    n_species = len(species)
    n_reactions = len(reactions)

    # Create A and B matrices
    A = np.zeros((n_reactions, n_species))
    B = np.zeros((n_reactions, n_species))

    for r in range(n_reactions):
        reactants = reactions[r]["reactants"]
        products = reactions[r]["products"]
        # Fill A
        for s in range(n_species):
            name = species[s]["name"]
            if name in reactants:
                A[r, s] = reactants[name]
        # Fill B
        for s in range(n_species):
            name = species[s]["name"]
            if name in products:
                B[r, s] = products[name]

    # Compute S = (B - A).T
    S = (B - A).T

    # Initial concentrations
    y0 = np.array([s["initial"] for s in species])

    # Set tolerances
    max_initial = max(y0) if y0.size > 0 else 0.0
    atol = 1e-12 * max_initial if max_initial > 0 else 1e-12
    rtol = 1e-8

    # Define the ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            rate = reactions[r]["rate"]
            product = 1.0
            for s in range(n_species):
                if A[r, s] > 0:
                    product *= y[s] ** A[r, s]
            v[r] = rate * product
        return S @ v

    # Solve the ODE
    max_t = max(t_eval)
    sol = solve_ivp(dydt, [0, max_t], y0, t_eval=t_eval, method='BDF', rtol=rtol, atol=atol)
    return sol.y.T