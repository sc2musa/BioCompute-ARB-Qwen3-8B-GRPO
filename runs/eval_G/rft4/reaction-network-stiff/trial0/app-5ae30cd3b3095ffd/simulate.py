import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    reactions = network['reactions']
    n_species = len(species)
    n_reactions = len(reactions)

    species_names = [s['name'] for s in species]
    species_index = {name: idx for idx, name in enumerate(species_names)}

    A = np.zeros((n_reactions, n_species))
    S = np.zeros((n_species, n_reactions))

    for r in range(n_reactions):
        reactants = reactions[r]['reactants']
        products = reactions[r]['products']
        rate = reactions[r]['rate']

        for s in range(n_species):
            name = species_names[s]
            A[r, s] = reactants.get(name, 0)

        for s in range(n_species):
            name = species_names[s]
            reactant_count = reactants.get(name, 0)
            product_count = products.get(name, 0)
            S[s, r] = product_count - reactant_count

    y0 = np.array([s['initial'] for s in species])
    max_initial = np.max(y0)
    atol = 1e-12 * max_initial if max_initial > 0 else 1e-12
    rtol = 1e-8

    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            rate = reactions[r]['rate']
            product = 1.0
            for s in range(n_species):
                product *= y[s] ** A[r, s]
            v[r] = rate * product
        return S @ v

    sol = solve_ivp(
        fun=dydt,
        t_span=(0.0, network['t_end']),
        y0=y0,
        t_eval=t_eval,
        method='BDF',
        rtol=rtol,
        atol=atol
    )

    return sol.y.T
