import numpy as np
from scipy.integrate import solve_ivp


def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    # Parse the network into matrices
    species = network['species']
    n_species = len(species)
    n_reactions = len(network['reactions'])

    # Initialize matrices
    A = np.zeros((n_reactions, n_species))  # Reactant order matrix
    S = np.zeros((n_species, n_reactions))  # Net stoichiometry matrix
    k = np.zeros(n_reactions)  # Rate constants

    # Populate matrices
    for r, reaction in enumerate(network['reactions']):
        # Populate A matrix (reactant order)
        for s, stoich in reaction['reactants'].items():
            species_idx = next(i for i, sp in enumerate(species) if sp['name'] == s)
            A[r, species_idx] = stoich
        # Populate S matrix (net stoichiometry)
        for s, stoich in reaction['products'].items():
            species_idx = next(i for i, sp in enumerate(species) if sp['name'] == s)
            S[species_idx, r] = stoich
        # Populate rate constants
        k[r] = reaction['rate']

    # Initial concentrations
    y0 = np.array([sp['initial'] for sp in species])

    # Define the ODE system
    def dydt(t, y):
        v = k * y ** A
        return S @ v

    # Define the Jacobian
    def jac(t, y):
        J = np.zeros((n_species, n_species))
        for r in range(n_reactions):
            for s in range(n_species):
                if A[r, s] > 0:
                    J[s, s] += k[r] * A[r, s] * y[s] ** (A[r, s] - 1) * np.prod(y ** A[r, :])
        return J

    # Solve the ODE
    sol = solve_ivp(
        fun=dydt,
        t_span=(0, network['t_end']),
        y0=y0,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-8,
        atol=1e-12 * np.max(y0),
        jac=jac
    )

    # Return the solution
    return sol.y.T