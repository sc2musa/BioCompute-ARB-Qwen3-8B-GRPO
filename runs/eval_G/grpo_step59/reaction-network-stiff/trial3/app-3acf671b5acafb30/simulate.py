import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    # Parse network
    species = network['species']
    reactions = network['reactions']
    n_species = len(species)
    n_reactions = len(reactions)

    # Create species name to index mapping
    species_names = [s['name'] for s in species]
    species_index = {name: idx for idx, name in enumerate(species_names)}

    # Create reactant order matrix A
    A = np.zeros((n_reactions, n_species))
    for r, reaction in enumerate(reactions):
        for s_name, stoich in reaction['reactants'].items():
            s = species_index[s_name]
            A[r, s] = stoich

    # Create stoichiometry matrix S
    S = np.zeros((n_species, n_reactions))
    for r, reaction in enumerate(reactions):
        for s_name, stoich in reaction['products'].items():
            s = species_index[s_name]
            S[s, r] += stoich
        for s_name, stoich in reaction['reactants'].items():
            s = species_index[s_name]
            S[s, r] -= stoich

    # Initial concentrations
    y0 = np.array([s['initial'] for s in species])

    # Define ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            # Compute reaction rate
            rate = reactions[r]['rate']
            product = 1.0
            for s in range(n_species):
                if A[r, s] > 0:
                    product *= y[s] ** A[r, s]
            v[r] = rate * product
        # Compute dy/dt = S @ v
        return S @ v

    # Solve ODE
    sol = solve_ivp(dydt, [0, network['t_end']], y0, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-12 * np.max(y0))

    return sol.y.T