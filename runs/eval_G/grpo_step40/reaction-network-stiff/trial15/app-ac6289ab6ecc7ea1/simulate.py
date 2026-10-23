import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    # Parse the network
    species = network['species']
    n_species = len(species)
    reactions = network['reactions']
    n_reactions = len(reactions)

    # Initialize matrices
    A = np.zeros((n_species, n_reactions))  # Reactant order matrix
    S = np.zeros((n_species, n_reactions))  # Net stoichiometry matrix

    # Populate matrices
    for r_idx, reaction in enumerate(reactions):
        reactants = reaction['reactants']
        products = reaction['products']
        rate = reaction['rate']

        # Fill reactant order matrix A
        for s_name, stoich in reactants.items():
            s_idx = next(i for i, spec in enumerate(species) if spec['name'] == s_name)
            A[s_idx, r_idx] = stoich

        # Fill net stoichiometry matrix S
        for s_name, stoich in products.items():
            s_idx = next(i for i, spec in enumerate(species) if spec['name'] == s_name)
            S[s_idx, r_idx] += stoich
        for s_name, stoich in reactants.items():
            s_idx = next(i for i, spec in enumerate(species) if spec['name'] == s_name)
            S[s_idx, r_idx] -= stoich

    # Initial concentrations
    y0 = np.array([spec['initial'] for spec in species])

    # Set tolerances
    max_initial = np.max(y0)
    atol = 1e-12 * max_initial if max_initial > 0 else 1e-12
    rtol = 1e-8

    # Define the ODE system
    def dydt(t, y):
        # Compute reaction rates
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            # Calculate the product of concentrations raised to their stoichiometry
            product = 1.0
            for s_idx in range(n_species):
                if A[s_idx, r_idx] > 0:
                    product *= y[s_idx] ** A[s_idx, r_idx]
            v[r_idx] = reactions[r_idx]['rate'] * product

        # Compute the derivative
        return S @ v

    # Solve the ODE
    sol = solve_ivp(dydt, [0, network['t_end']], y0, t_eval=t_eval, method='BDF', rtol=rtol, atol=atol)

    # Return the solution
    return sol.y.T