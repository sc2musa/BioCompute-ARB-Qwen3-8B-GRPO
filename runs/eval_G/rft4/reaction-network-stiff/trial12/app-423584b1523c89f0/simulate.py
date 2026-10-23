import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    n_species = len(species)
    species_to_index = {s['name']: i for i, s in enumerate(species)}
    n_reactions = len(network['reactions'])

    # Build A and S matrices
    A = np.zeros((n_reactions, n_species))
    S = np.zeros((n_species, n_reactions))

    for r in range(n_reactions):
        reactants = network['reactions'][r]['reactants']
        products = network['reactions'][r]['products']
        # Fill A
        for s in range(n_species):
            s_name = species[s]['name']
            A[r][s] = reactants.get(s_name, 0)
        # Fill S
        for s in range(n_species):
            s_name = species[s]['name']
            stoich_product = products.get(s_name, 0)
            stoich_reactant = reactants.get(s_name, 0)
            S[s][r] = stoich_product - stoich_reactant

    # Define the ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            rate = network['reactions'][r]['rate']
            product = 1.0
            for s in range(n_species):
                if A[r][s] > 0:
                    product *= y[s] ** A[r][s]
            v[r] = rate * product
        return np.dot(S, v)

    # Initial conditions
    y0 = np.array([s['initial'] for s in species])

    # Compute atol
    y_initial = y0
    atol = [1e-12 * max(1, y_initial[i]) for i in range(n_species)]

    # Solve the ODE
    sol = solve_ivp(
        fun=dydt,
        t_span=[0.0, network['t_end']],
        y0=y0,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-8,
        atol=atol
    )

    # Return the solution at t_eval
    return sol.y.T
