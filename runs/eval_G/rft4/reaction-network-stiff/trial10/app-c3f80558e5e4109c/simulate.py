import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    n_species = len(species)
    n_reactions = len(network['reactions'])

    species_names = [s['name'] for s in species]
    species_index = {name: idx for idx, name in enumerate(species_names)}

    A = np.zeros((n_species, n_reactions))
    B = np.zeros((n_species, n_reactions))
    k = np.zeros(n_reactions)

    for r in range(n_reactions):
        reactants = network['reactions'][r]['reactants']
        products = network['reactions'][r]['products']
        rate = network['reactions'][r]['rate']
        k[r] = rate

        for s_name, stoich in reactants.items():
            s_idx = species_index[s_name]
            A[s_idx, r] = stoich
        for s_name, stoich in products.items():
            s_idx = species_index[s_name]
            B[s_idx, r] = stoich

    y0 = np.array([s['initial'] for s in species])

    max_initial = max(s['initial'] for s in species)
    if max_initial == 0:
        atol = 1e-12
    else:
        atol = 1e-12 * max_initial

    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            rate = k[r]
            for s in range(n_species):
                if A[s, r] > 0:
                    rate *= y[s] ** A[s, r]
            v[r] = rate
        dydt = (B - A) @ v
        return dydt

    sol = solve_ivp(
        fun=dydt,
        t_span=(0, network['t_end']),
        y0=y0,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-8,
        atol=atol
    )

    return sol.y.T
