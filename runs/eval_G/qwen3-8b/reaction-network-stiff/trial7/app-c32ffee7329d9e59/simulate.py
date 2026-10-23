import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    n_species = len(species)
    species_names = [s['name'] for s in species]
    species_index = {name: idx for idx, name in enumerate(species_names)}

    y0 = np.array([s['initial'] for s in species])

    def dydt(t, y):
        dydt = np.zeros(n_species)
        for reaction in network['reactions']:
            rate = reaction['rate']
            reactants = reaction['reactants']
            products = reaction['products']

            # Calculate rate based on reactants
            for species_name, stoich in reactants.items():
                if species_name in species_index:
                    idx = species_index[species_name]
                    rate *= y[idx] ** stoich

            # Update dydt for products and reactants
            for species_name, stoich in products.items():
                if species_name in species_index:
                    idx = species_index[species_name]
                    dydt[idx] += stoich * rate
            for species_name, stoich in reactants.items():
                if species_name in species_index:
                    idx = species_index[species_name]
                    dydt[idx] -= stoich * rate
        return dydt

    t_span = (0.0, t_eval[-1])
    sol = solve_ivp(dydt, t_span, y0, t_eval=t_eval, method='Radau', rtol=1e-6, atol=1e-6)
    return sol.y.T
