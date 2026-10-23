import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    reactions = network['reactions']
    n_species = len(species)
    n_reactions = len(reactions)
    species_dict = {s['name']: i for i, s in enumerate(species)}

    # Build stoichiometry matrix S
    S = np.zeros((n_species, n_reactions))
    for j, reaction in enumerate(reactions):
        reactants = reaction['reactants']
        products = reaction['products']
        for species_name, stoich in reactants.items():
            i = species_dict[species_name]
            S[i][j] -= stoich
        for species_name, stoich in products.items():
            i = species_dict[species_name]
            S[i][j] += stoich

    # Initial concentrations
    y0 = np.array([s['initial'] for s in species])

    # Compute absolute tolerance
    max_initial = np.max(y0)
    atol = 1e-12 * max_initial if max_initial > 0 else 1e-12

    # Define ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for j in range(n_reactions):
            reactants = reactions[j]['reactants']
            rate = reactions[j]['rate']
            product = 1.0
            for species_name, stoich in reactants.items():
                i = species_dict[species_name]
                product *= y[i] ** stoich
            v[j] = rate * product
        return S @ v

    # Solve ODE
    sol = solve_ivp(dydt, [0, network['t_end']], y0, t_eval=t_eval, method='BDF', rtol=1e-8, atol=atol)

    return sol.y.T