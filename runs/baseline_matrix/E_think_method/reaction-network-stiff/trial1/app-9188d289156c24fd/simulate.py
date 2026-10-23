import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    reactions = network['reactions']
    n_species = len(species)
    n_reactions = len(reactions)

    A = np.zeros((n_reactions, n_species))
    S = np.zeros((n_species, n_reactions))

    for r_idx, reaction in enumerate(reactions):
        reactants = reaction['reactants']
        products = reaction['products']
        rate = reaction['rate']

        for s_idx, species_name in enumerate(species):
            A[r_idx, s_idx] = reactants.get(species_name, 0)

        for s_idx, species_name in enumerate(species):
            reactant_stoich = reactants.get(species_name, 0)
            product_stoich = products.get(species_name, 0)
            S[s_idx, r_idx] = product_stoich - reactant_stoich

    y0 = np.array([s['initial'] for s in species])

    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            product = 1.0
            for s_idx in range(n_species):
                product *= y[s_idx] ** A[r_idx, s_idx]
            v[r_idx] = reactions[r_idx]['rate'] * product
        return np.dot(S, v)

    def jac(t, y):
        J = np.zeros((n_species, n_species))
        for r_idx in range(n_reactions):
            product = 1.0
            for s_idx in range(n_species):
                product *= y[s_idx] ** A[r_idx, s_idx]
            v_r = reactions[r_idx]['rate'] * product
            for j in range(n_species):
                if y[j] == 0:
                    continue
                dv_r_dy_j = (reactions[r_idx]['rate'] * A[r_idx, j] * v_r) / y[j]
                for i in range(n_species):
                    J[i, j] += S[i, r_idx] * dv_r_dy_j
        return J

    sol = solve_ivp(dydt, [0, network['t_end']], y0, t_eval=t_eval, method='BDF',
                    rtol=1e-8, atol=1e-12 * np.max(y0), jac=jac)
    return sol.y.T