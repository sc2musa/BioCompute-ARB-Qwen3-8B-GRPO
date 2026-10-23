import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    reactions = network['reactions']
    n_species = len(species)
    species_names = [s['name'] for s in species]
    species_dict = {name: idx for idx, name in enumerate(species_names)}

    y0 = np.array([s['initial'] for s in species])

    def dydt(t, y):
        dydt = np.zeros(n_species)
        for reaction in reactions:
            reactants = reaction['reactants']
            products = reaction['products']
            rate = reaction['rate']

            # Calculate rate
            if not reactants:
                rate_val = rate
            else:
                product = 1.0
                for s, stoich in reactants.items():
                    idx = species_dict[s]
                    product *= y[idx] ** stoich
                rate_val = rate * product

            # Update derivatives
            for s, stoich in products.items():
                idx = species_dict[s]
                dydt[idx] += stoich * rate_val
            for s, stoich in reactants.items():
                idx = species_dict[s]
                dydt[idx] -= stoich * rate_val
        return dydt

    t_span = (0.0, max(t_eval))
    sol = solve_ivp(dydt, t_span, y0, t_eval=t_eval, method='Radau', rtol=1e-8, atol=1e-10)
    return sol.y.T
