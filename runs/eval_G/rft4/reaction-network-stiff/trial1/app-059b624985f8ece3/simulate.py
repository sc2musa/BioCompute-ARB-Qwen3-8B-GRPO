import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    reactions = network['reactions']
    n_species = len(species)
    n_reactions = len(reactions)

    species_indices = {s['name']: i for i, s in enumerate(species)}

    A = np.zeros((n_species, n_reactions))
    B = np.zeros((n_species, n_reactions))

    for r in range(n_reactions):
        reactants = reactions[r]['reactants']
        products = reactions[r]['products']
        rate = reactions[r]['rate']
        for s in reactants:
            A[species_indices[s], r] = reactants[s]
        for s in products:
            B[species_indices[s], r] = products[s]

    y0 = np.array([s['initial'] for s in species])

    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            reactants = reactions[r]['reactants']
            rate = reactions[r]['rate']
            product = 1.0
            for s in reactants:
                product *= y[species_indices[s]] ** reactants[s]
            v[r] = rate * product
        return (B - A) @ v

    def jac(t, y):
        dv_dy = np.zeros((n_reactions, n_species))
        for r in range(n_reactions):
            reactants = reactions[r]['reactants']
            rate = reactions[r]['rate']
            product = 1.0
            for s in reactants:
                product *= y[species_indices[s]] ** reactants[s]
            for s in reactants:
                if y[species_indices[s]] == 0:
                    dv_dy[r, species_indices[s]] = 0.0
                else:
                    dv_dy[r, species_indices[s]] = rate * reactants[s] * product / y[species_indices[s]]
        return (B - A) @ dv_dy

    max_initial = max(s['initial'] for s in species)
    rtol = 1e-6
    atol = 1e-10 * max_initial if max_initial > 0 else 1e-12

    sol = solve_ivp(dydt, [0, network['t_end']], y0, t_eval=t_eval, method='BDF', rtol=rtol, atol=atol, jac=jac)

    return sol.y.T