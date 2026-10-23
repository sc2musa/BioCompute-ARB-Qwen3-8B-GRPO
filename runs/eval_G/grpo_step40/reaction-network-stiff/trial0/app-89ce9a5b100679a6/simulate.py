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

    # Initial concentrations
    y0 = np.array([s['initial'] for s in species])

    # Define the ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            # Calculate reaction rate
            rate = reactions[r_idx]['rate']
            reactant_product = 1.0
            for s_idx in range(n_species):
                if A[r_idx, s_idx] > 0:
                    reactant_product *= y[s_idx] ** A[r_idx, s_idx]
            v[r_idx] = rate * reactant_product
        # Calculate dy/dt = S @ v
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