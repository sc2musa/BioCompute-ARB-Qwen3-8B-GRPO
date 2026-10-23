import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network['species']
    n_species = len(species)
    species_names = [s['name'] for s in species]
    species_dict = {name: idx for idx, name in enumerate(species_names)}
    initial = np.array([s['initial'] for s in species])

    reactions = network['reactions']
    n_reactions = len(reactions)

    # Create A and S matrices
    A = np.zeros((n_reactions, n_species))
    S = np.zeros((n_species, n_reactions))

    for r_idx, reaction in enumerate(reactions):
        reactants = reaction['reactants']
        products = reaction['products']
        rate = reaction['rate']

        # Fill A matrix (reactant stoichiometry)
        for s_name, stoich in reactants.items():
            s_idx = species_dict[s_name]
            A[r_idx, s_idx] = stoich

        # Fill S matrix (net stoichiometry)
        for s_name, stoich in products.items():
            s_idx = species_dict[s_name]
            S[s_idx, r_idx] += stoich
        for s_name, stoich in reactants.items():
            s_idx = species_dict[s_name]
            S[s_idx, r_idx] -= stoich

    reaction_rates = np.array([r['rate'] for r in reactions])

    # Define the ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            product = np.prod(y ** A[r_idx, :])
            v[r_idx] = reaction_rates[r_idx] * product
        return S @ v

    # Solve the ODE
    sol = solve_ivp(
        fun=dydt,
        t_span=[0.0, network['t_end']],
        y0=initial,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-8,
        atol=1e-12
    )

    return sol.y.T
