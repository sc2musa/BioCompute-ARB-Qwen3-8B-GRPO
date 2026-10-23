import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    reactions = network['reactions']
    n_species = len(species)
    n_reactions = len(reactions)

    # Build A and S matrices
    A = np.zeros((n_reactions, n_species))
    for r in range(n_reactions):
        reactants = reactions[r]['reactants']
        for s in range(n_species):
            name = species[s]['name']
            if name in reactants:
                A[r][s] = reactants[name]

    S = np.zeros((n_species, n_reactions))
    for r in range(n_reactions):
        reactants = reactions[r]['reactants']
        products = reactions[r]['products']
        for s in range(n_species):
            name = species[s]['name']
            product_coeff = products.get(name, 0)
            reactant_coeff = reactants.get(name, 0)
            S[s][r] = product_coeff - reactant_coeff

    # Initial conditions
    y0 = np.array([s['initial'] for s in species])

    # Define ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            rate = reactions[r]['rate']
            product = np.prod(y ** A[r, :])
            v[r] = rate * product
        return S @ v

    # Solve ODE
    t_end = network['t_end']
    sol = solve_ivp(dydt, (0, t_end), y0, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-12)

    # Return concentrations at t_eval
    return sol.y.T
