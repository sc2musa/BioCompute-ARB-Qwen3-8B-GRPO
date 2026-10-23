import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    reactions = network['reactions']
    n_species = len(species)
    n_reactions = len(reactions)

    # Precompute species name to index mapping
    species_dict = {sp['name']: i for i, sp in enumerate(species)}

    # Build A and B matrices
    A = np.zeros((n_species, n_reactions))
    B = np.zeros((n_species, n_reactions))
    reactant_indices = []

    for r_idx, reaction in enumerate(reactions):
        reactants = reaction['reactants']
        products = reaction['products']
        rate = reaction['rate']

        # Fill A matrix (reactants)
        indices = []
        for s, stoich in reactants.items():
            s_idx = species_dict[s]
            A[s_idx, r_idx] = stoich
            indices.append(s_idx)
        reactant_indices.append(indices)

        # Fill B matrix (products)
        for s, stoich in products.items():
            s_idx = species_dict[s]
            B[s_idx, r_idx] = stoich

    # Initial concentrations
    y0 = np.array([sp['initial'] for sp in species])

    # Compute atol
    max_initial = max(sp['initial'] for sp in species)
    atol = 1e-12 * max_initial if max_initial > 1e-12 else 1e-12

    # Define ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            rate = reactions[r_idx]['rate']
            product = 1.0
            for s_idx in reactant_indices[r_idx]:
                product *= y[s_idx] ** A[s_idx, r_idx]
            v[r_idx] = rate * product
        return (B - A) @ v

    # Solve ODE
    sol = solve_ivp(
        fun=dydt,
        t_span=(0.0, network['t_end']),
        y0=y0,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-8,
        atol=atol
    )

    return sol.y.T
