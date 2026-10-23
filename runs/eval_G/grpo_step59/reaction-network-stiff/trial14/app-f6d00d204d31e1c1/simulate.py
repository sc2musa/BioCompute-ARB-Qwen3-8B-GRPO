import numpy as np
from scipy.integrate import solve_ivp


def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    # Parse network
    species = network['species']
    n_species = len(species)
    reactions = network['reactions']
    n_reactions = len(reactions)

    # Create reactant order matrix A (reactions x species)
    A = np.zeros((n_reactions, n_species))
    for r, reaction in enumerate(reactions):
        for s, stoich in reaction['reactants'].items():
            s_idx = next(i for i, sp in enumerate(species) if sp['name'] == s)
            A[r, s_idx] = stoich

    # Create net stoichiometry matrix S (species x reactions)
    S = np.zeros((n_species, n_reactions))
    for r, reaction in enumerate(reactions):
        for s, stoich in reaction['products'].items():
            s_idx = next(i for i, sp in enumerate(species) if sp['name'] == s)
            S[s_idx, r] += stoich
        for s, stoich in reaction['reactants'].items():
            s_idx = next(i for i, sp in enumerate(species) if sp['name'] == s)
            S[s_idx, r] -= stoich

    # Initial concentrations
    y0 = np.array([sp['initial'] for sp in species])

    # Define ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            # Calculate reaction rate
            rate = reactions[r]['rate']
            reactants = reactions[r]['reactants']
            product_terms = 1.0
            for s, stoich in reactants.items():
                s_idx = next(i for i, sp in enumerate(species) if sp['name'] == s)
                product_terms *= y[s_idx] ** stoich
            v[r] = rate * product_terms
        # Calculate dy/dt
        return S @ v

    # Set tolerances
    rtol = 1e-8
    atol = 1e-12 * np.max(y0) if np.max(y0) > 0 else 1e-12

    # Solve ODE
    sol = solve_ivp(dydt, [0, t_eval[-1]], y0, t_eval=t_eval, method='BDF', rtol=rtol, atol=atol)

    # Return solution
    return sol.y.T