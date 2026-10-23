import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    reactions = network['reactions']
    t_end = network['t_end']
    n_species = len(species)
    n_reactions = len(reactions)

    # Precompute reactant and product indices and stoichiometry for each reaction
    reactant_indices = [[] for _ in range(n_reactions)]
    reactant_stoich = [[] for _ in range(n_reactions)]
    product_indices = [[] for _ in range(n_reactions)]
    product_stoich = [[] for _ in range(n_reactions)]

    # Build A and B matrices
    A = np.zeros((n_reactions, n_species))
    B = np.zeros((n_reactions, n_species))
    for r in range(n_reactions):
        reactants = reactions[r]['reactants']
        products = reactions[r]['products']
        for s in range(n_species):
            species_name = species[s]['name']
            A[r, s] = reactants.get(species_name, 0)
            B[r, s] = products.get(species_name, 0)
        # Collect reactant indices and stoich
        for s in range(n_species):
            if A[r, s] > 0:
                reactant_indices[r].append(s)
                reactant_stoich[r].append(A[r, s])
        # Collect product indices and stoich
        for s in range(n_species):
            if B[r, s] > 0:
                product_indices[r].append(s)
                product_stoich[r].append(B[r, s])

    # Initial concentrations
    y0 = np.array([s['initial'] for s in species])

    # Define the ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            rate = reactions[r]['rate']
            product = 1.0
            # Compute the product of y^stoich for reactants
            for s, stoich in zip(reactant_indices[r], reactant_stoich[r]):
                product *= y[s] ** stoich
            v[r] = rate * product
        # Compute dydt = (B - A) @ v
        dydt = np.dot(B - A, v)
        return dydt

    # Solve the ODE
    sol = solve_ivp(dydt, [0, t_end], y0, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-12)

    # Return the solution
    return sol.y.T
