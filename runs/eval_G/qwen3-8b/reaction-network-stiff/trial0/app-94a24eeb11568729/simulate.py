import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    n_species = len(species)
    initial_concentrations = [s['initial'] for s in species]
    species_names = [s['name'] for s in species]
    species_index = {name: idx for idx, name in enumerate(species_names)}
    reactions = network['reactions']

    def dydt(t, y):
        rates = []
        for reaction in reactions:
            rate = reaction['rate']
            for species_name, stoich in reaction['reactants'].items():
                idx = species_index[species_name]
                rate *= y[idx] ** stoich
            rates.append(rate)

        dydt = np.zeros(n_species)
        for i, reaction in enumerate(reactions):
            for species_name, stoich in reaction['products'].items():
                idx = species_index[species_name]
                dydt[idx] += stoich * rates[i]
            for species_name, stoich in reaction['reactants'].items():
                idx = species_index[species_name]
                dydt[idx] -= stoich * rates[i]
        return dydt

    sol = solve_ivp(dydt, [0.0, network['t_end']], initial_concentrations, t_eval=t_eval, method='Radau', rtol=1e-6, atol=1e-6)
    return sol.y.T
