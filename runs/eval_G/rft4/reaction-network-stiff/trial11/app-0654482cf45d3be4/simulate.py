import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    reactions = network['reactions']
    t_end = network['t_end']

    species_index = {s['name']: i for i, s in enumerate(species)}
    n_species = len(species)
    n_reactions = len(reactions)

    B = np.zeros((n_species, n_reactions))
    A = np.zeros((n_reactions, n_species))
    for r in range(n_reactions):
        reactants = reactions[r]['reactants']
        products = reactions[r]['products']
        for species_name, stoich in reactants.items():
            idx = species_index[species_name]
            B[idx, r] -= stoich
            A[r, idx] = stoich
        for species_name, stoich in products.items():
            idx = species_index[species_name]
            B[idx, r] += stoich

    y0 = np.array([s['initial'] for s in species])

    max_initial = max(y0)
    if max_initial == 0:
        atol = 1e-12
    else:
        atol = 1e-12 * max_initial

    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            rate = reactions[r]['rate']
            for species_name, stoich in reactions[r]['reactants'].items():
                idx = species_index[species_name]
                rate *= y[idx] ** stoich
            v[r] = rate
        return B @ v

    def jac(t, y):
        jac_v = np.zeros((n_reactions, n_species))
        for r in range(n_reactions):
            rate = reactions[r]['rate']
            for species_name, stoich in reactions[r]['reactants'].items():
                idx = species_index[species_name]
                rate *= y[idx] ** stoich
            v_r = rate
            for species_name, stoich in reactions[r]['reactants'].items():
                idx = species_index[species_name]
                if y[idx] == 0:
                    jac_v[r, idx] = 0.0
                else:
                    jac_v[r, idx] = v_r * (stoich / y[idx])
        return B @ jac_v

    sol = solve_ivp(
        fun=dydt,
        t_span=(0, t_end),
        y0=y0,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-8,
        atol=atol,
        jac=jac
    )

    return sol.y.T
