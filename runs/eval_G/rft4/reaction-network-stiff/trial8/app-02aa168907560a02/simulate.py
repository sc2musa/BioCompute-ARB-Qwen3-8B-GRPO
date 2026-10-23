import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    species_names = [s['name'] for s in species]
    n_species = len(species_names)
    species_index = {name: idx for idx, name in enumerate(species_names)}
    n_reactions = len(network['reactions'])

    # Build stoichiometry matrix A and reaction rate constants K
    A = np.zeros((n_species, n_reactions))
    K = np.zeros(n_reactions)
    reaction_reactants = [[] for _ in range(n_reactions)]

    for r_idx, reaction in enumerate(network['reactions']):
        reactants = reaction['reactants']
        products = reaction['products']
        rate = reaction['rate']
        K[r_idx] = rate

        # Store reactant stoichiometry for rate calculation
        for s_name, a_s in reactants.items():
            s_idx = species_index[s_name]
            reaction_reactants[r_idx].append((s_idx, a_s))

        # Build stoichiometry matrix A
        for s_idx in range(n_species):
            s_name = species_names[s_idx]
            A[s_idx][r_idx] = products.get(s_name, 0) - reactants.get(s_name, 0)

    # Initial concentrations
    y0 = np.array([s['initial'] for s in species])
    max_initial = np.max(y0)
    if max_initial == 0:
        atol = 1e-16
    else:
        atol = 1e-12 * max_initial

    # ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            product = 1.0
            for (s_idx, a_s) in reaction_reactants[r]:
                product *= y[s_idx] ** a_s
            v[r] = K[r] * product
        return A @ v

    # Solve ODE with BDF method and tight tolerances
    sol = solve_ivp(
        fun=dydt,
        t_span=[0.0, network['t_end']],
        y0=y0,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-8,
        atol=atol
    )

    return sol.y.T
