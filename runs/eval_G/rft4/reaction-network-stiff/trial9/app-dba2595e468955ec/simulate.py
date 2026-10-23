import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    n_species = len(species)
    reactions = network['reactions']
    n_reactions = len(reactions)

    A = np.zeros((n_species, n_reactions))
    S = np.zeros((n_species, n_reactions))

    for r_idx, reaction in enumerate(reactions):
        reactants = reaction['reactants']
        products = reaction['products']
        rate = reaction['rate']
        for s_idx, species_dict in enumerate(species):
            species_name = species_dict['name']
            A[s_idx][r_idx] = reactants.get(species_name, 0)
            S[s_idx][r_idx] = products.get(species_name, 0) - reactants.get(species_name, 0)

    y0 = np.array([s['initial'] for s in species])
    max_initial = max(y0)
    atol = 1e-12 * max_initial if max_initial > 0 else 1e-12

    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            rate = reactions[r]['rate']
            product = 1.0
            for s in range(n_species):
                product *= y[s] ** A[s][r]
            v[r] = rate * product
        return S @ v

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
