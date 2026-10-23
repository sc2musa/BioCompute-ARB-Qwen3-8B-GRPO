import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    reactions = network['reactions']
    n_species = len(species)
    n_reactions = len(reactions)

    # Build A (reactant stoichiometry) and S (net stoichiometry) matrices
    A = np.zeros((n_species, n_reactions))
    S = np.zeros((n_species, n_reactions))

    for r, reaction in enumerate(reactions):
        reactants = reaction['reactants']
        products = reaction['products']
        rate = reaction['rate']

        # Fill A matrix
        for s in range(n_species):
            name = species[s]['name']
            A[s][r] = reactants.get(name, 0)

        # Fill S matrix
        for s in range(n_species):
            name = species[s]['name']
            S[s][r] = products.get(name, 0) - reactants.get(name, 0)

    # Initial concentrations
    y0 = np.array([s['initial'] for s in species])

    # Define ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            rate = reactions[r]['rate']
            product = 1.0
            for s in range(n_species):
                name = species[s]['name']
                if A[s][r] > 0:
                    product *= y[s] ** A[s][r]
            v[r] = rate * product
        return S @ v

    # Solve ODE
    t_span = (0.0, network['t_end'])
    sol = solve_ivp(dydt, t_span, y0, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-12)

    # Return solution
    return sol.y.T
