import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network["species"]
    n_species = len(species)
    n_reactions = len(network["reactions"])

    species_names = [s["name"] for s in species]
    species_indices = {name: idx for idx, name in enumerate(species_names)}

    # Build A (reactant stoichiometry matrix) and S (net stoichiometry matrix)
    A = np.zeros((n_reactions, n_species))
    S = np.zeros((n_species, n_reactions))

    for r_idx, reaction in enumerate(network["reactions"]):
        reactants = reaction["reactants"]
        products = reaction["products"]
        rate = reaction["rate"]

        # Fill A matrix
        for s_name, stoich in reactants.items():
            s_idx = species_indices[s_name]
            A[r_idx, s_idx] = stoich

        # Fill S matrix
        for s_name, stoich in products.items():
            s_idx = species_indices[s_name]
            S[s_idx, r_idx] += stoich
        for s_name, stoich in reactants.items():
            s_idx = species_indices[s_name]
            S[s_idx, r_idx] -= stoich

    initial_conditions = np.array([s["initial"] for s in species])

    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            rate = network["reactions"][r_idx]["rate"]
            product = np.prod(y ** A[r_idx, :])
            v[r_idx] = rate * product
        return S @ v

    def jacobian(t, y):
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            rate = network["reactions"][r_idx]["rate"]
            product = np.prod(y ** A[r_idx, :])
            v[r_idx] = rate * product
        J_v = np.zeros((n_reactions, n_species))
        for r_idx in range(n_reactions):
            rate = network["reactions"][r_idx]["rate"]
            v_r = v[r_idx]
            for s_idx in range(n_species):
                if A[r_idx, s_idx] > 0:
                    J_v[r_idx, s_idx] = (rate * A[r_idx, s_idx] * v_r) / y[s_idx]
        return S @ J_v

    max_initial = np.max(initial_conditions)
    atol = 1e-12 * max_initial
    sol = solve_ivp(
        fun=dydt,
        t_span=(0.0, network["t_end"]),
        y0=initial_conditions,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-8,
        atol=atol,
        jac=jacobian
    )

    return sol.y.T
