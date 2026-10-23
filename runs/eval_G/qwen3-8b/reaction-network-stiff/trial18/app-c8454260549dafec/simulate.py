import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    species_names = [s['name'] for s in species]
    species_indices = {name: idx for idx, name in enumerate(species_names)}
    n_species = len(species)
    y0 = np.array([s['initial'] for s in species])

    def ode_func(t, y):
        dydt = np.zeros(n_species)
        for reaction in network['reactions']:
            rate = reaction['rate']
            for species, stoich in reaction['reactants'].items():
                rate *= y[species_indices[species]] ** stoich
            for species in reaction['reactants']:
                stoich = reaction['reactants'][species]
                net = -stoich
                idx = species_indices[species]
                dydt[idx] += net * rate
            for species in reaction['products']:
                stoich = reaction['products'][species]
                net = stoich
                idx = species_indices[species]
                dydt[idx] += net * rate
        return dydt

    t_span = (0.0, network['t_end'])
    sol = solve_ivp(ode_func, t_span, y0, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-9)
    return sol.y.T
