import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    n_species = len(species)
    reactions = network['reactions']
    n_reactions = len(reactions)

    species_names = [s['name'] for s in species]
    species_index = {name: idx for idx, name in enumerate(species_names)}

    # Build reactant-order matrix A and stoichiometry matrix S
    A = np.zeros((n_reactions, n_species))
    S = np.zeros((n_species, n_reactions))
    for r, reaction in enumerate(reactions):
        reactants = reaction['reactants']
        products = reaction['products']
        for s, stoich in reactants.items():
            A[r, species_index[s]] = stoich
        for s, stoich in products.items():
            S[species_index[s], r] += stoich
        for s, stoich in reactants.items():
            S[species_index[s], r] -= stoich

    # Initial concentrations
    y0 = np.array([s['initial'] for s in species])

    # Define ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            rate = reactions[r]['rate']
            product = 1.0
            reactants = reactions[r]['reactants']
            for s, stoich in reactants.items():
                idx = species_index[s]
                product *= y[idx] ** stoich
            v[r] = rate * product
        return S @ v

    # Set solver options
    rtol = 1e-8
    atol = 1e-12 * y0.max() if y0.max() > 1e-12 else 1e-12
    if atol < 1e-12:
        atol = 1e-12

    # Solve ODE
    sol = solve_ivp(dydt, (0, network['t_end']), y0, t_eval=t_eval, method='BDF', rtol=rtol, atol=atol)

    # Return solution
    return sol.y.T
