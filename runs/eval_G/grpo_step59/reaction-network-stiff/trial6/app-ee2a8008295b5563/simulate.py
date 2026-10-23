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

    # Create reactant order matrix A (reactions x species)
    A = np.zeros((n_reactions, n_species))
    for r_idx, reaction in enumerate(reactions):
        for s_name, stoich in reaction['reactants'].items():
            s_idx = next(i for i, s in enumerate(species) if s['name'] == s_name)
            A[r_idx, s_idx] = stoich

    # Create net stoichiometry matrix S (species x reactions)
    S = np.zeros((n_species, n_reactions))
    for r_idx, reaction in enumerate(reactions):
        for s_name, stoich in reaction['products'].items():
            s_idx = next(i for i, s in enumerate(species) if s['name'] == s_name)
            S[s_idx, r_idx] += stoich
        for s_name, stoich in reaction['reactants'].items():
            s_idx = next(i for i, s in enumerate(species) if s['name'] == s_name)
            S[s_idx, r_idx] -= stoich

    # Define the ODE system
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            # Calculate reaction rate
            rate = reactions[r_idx]['rate']
            reactants = reactions[r_idx]['reactants']
            product_terms = 1.0
            for s_name, stoich in reactants.items():
                s_idx = next(i for i, s in enumerate(species) if s['name'] == s_name)
                product_terms *= y[s_idx] ** stoich
            v[r_idx] = rate * product_terms
        # Calculate dy/dt
        return S @ v

    # Set tolerances
    rtol = 1e-8
    atol = 1e-12 * np.max(y0) if np.max(y0) > 0 else 1e-12

    # Solve the ODE
    sol = solve_ivp(
        fun=dydt,
        t_span=(0.0, network['t_end']),
        y0=y0,
        t_eval=t_eval,
        method='BDF',
        rtol=rtol,
        atol=atol,
        dense_output=True
    )

    # Return the solution
    return sol.y.T
