import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    reactions = network['reactions']
    t_end = network['t_end']

    species_index = {species[i]['name']: i for i in range(len(species))}
    y0 = np.array([s['initial'] for s in species])

    def dydt(t, y):
        dy = np.zeros_like(y)
        for reaction in reactions:
            rate = reaction['rate']
            for species_name, stoich in reaction['reactants'].items():
                rate *= y[species_index[species_name]] ** stoich
            net_stoich = {}
            for species_name, stoich in reaction['reactants'].items():
                net_stoich[species_name] = -stoich
            for species_name, stoich in reaction['products'].items():
                net_stoich[species_name] = net_stoich.get(species_name, 0) + stoich
            for species_name, stoich in net_stoich.items():
                dy[species_index[species_name]] += stoich * rate
        return dy

    sol = solve_ivp(dydt, [0, t_end], y0, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-10)
    return sol.y.T