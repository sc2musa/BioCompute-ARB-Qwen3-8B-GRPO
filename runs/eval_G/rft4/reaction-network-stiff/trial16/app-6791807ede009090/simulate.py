import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    reactions = network['reactions']
    n_species = len(species)
    n_reactions = len(reactions)

    species_name_to_idx = {s['name']: i for i, s in enumerate(species)}

    A = np.zeros((n_reactions, n_species))
    S = np.zeros((n_species, n_reactions))
    k = np.array([r['rate'] for r in reactions])

    for r_idx, reaction in enumerate(reactions):
        reactants = reaction['reactants']
        products = reaction['products']

        # Fill A and S for this reaction
        for s_name, count in reactants.items():
            s_idx = species_name_to_idx[s_name]
            A[r_idx, s_idx] = count
            S[s_idx, r_idx] -= count
        for s_name, count in products.items():
            s_idx = species_name_to_idx[s_name]
            S[s_idx, r_idx] += count

    y0 = np.array([s['initial'] for s in species])

    def dydt(t, y):
        v = k * np.prod(y ** A, axis=1)
        return S @ v

    def jac(t, y):
        v = k * np.prod(y ** A, axis=1)
        # Avoid division by zero by using a small epsilon
        y_safe = np.maximum(y, 1e-20)
        dv_dy = A * (v[:, np.newaxis] / y_safe)
        return S @ dv_dy

    sol = solve_ivp(
        fun=dydt,
        t_span=(0, network['t_end']),
        y0=y0,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-8,
        atol=1e-12 * max(y0) if max(y0) > 0 else 1e-12,
        jac=jac
    )

    return sol.y.T
