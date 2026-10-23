import numpy as np
from scipy.integrate import solve_ivp


def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    # Parse network
    species = network['species']
    n_species = len(species)
    reactions = network['reactions']
    n_reactions = len(reactions)

    # Initialize concentrations
    y0 = np.array([s['initial'] for s in species])

    # Precompute species indices for faster access
    species_indices = {s['name']: i for i, s in enumerate(species)}

    # Create reactant order matrix A (reactions x species)
    A = np.zeros((n_reactions, n_species))
    for r_idx, reaction in enumerate(reactions):
        for s_name, stoich in reaction['reactants'].items():
            s_idx = species_indices[s_name]
            A[r_idx, s_idx] = stoich

    # Create net stoichiometry matrix S (species x reactions)
    S = np.zeros((n_species, n_reactions))
    for r_idx, reaction in enumerate(reactions):
        for s_name, stoich in reaction['products'].items():
            s_idx = species_indices[s_name]
            S[s_idx, r_idx] += stoich
        for s_name, stoich in reaction['reactants'].items():
            s_idx = species_indices[s_name]
            S[s_idx, r_idx] -= stoich

    # Precompute list of species involved in each reaction
    species_in_reactions = [[] for _ in range(n_reactions)]
    for r_idx, reaction in enumerate(reactions):
        for s_name, stoich in reaction['reactants'].items():
            s_idx = species_indices[s_name]
            species_in_reactions[r_idx].append(s_idx)

    # Define the ODE function
    def dydt(t, y):
        # Calculate reaction rates
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            # Calculate the product of concentrations raised to their stoichiometry
            product = 1.0
            for s_idx in species_in_reactions[r_idx]:
                product *= y[s_idx] ** A[r_idx, s_idx]
            v[r_idx] = reactions[r_idx]['rate'] * product

        # Calculate the derivative of concentrations
        dydt = S @ v
        return dydt

    # Set solver options
    rtol = 1e-8
    atol = 1e-12 * np.max(y0) if np.max(y0) > 0 else 1e-12

    # Solve the ODE
    sol = solve_ivp(dydt, [0, network['t_end']], y0, t_eval=t_eval, method='BDF', rtol=rtol, atol=atol)

    # Return the solution
    return sol.y.T