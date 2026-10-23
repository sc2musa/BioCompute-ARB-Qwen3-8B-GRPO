import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    species = network["species"]
    reactions = network["reactions"]
    n_species = len(species)
    n_reactions = len(reactions)

    # Create species index mapping
    species_names = [s["name"] for s in species]
    species_index = {name: idx for idx, name in enumerate(species_names)}

    # Initialize A and B matrices
    A = np.zeros((n_reactions, n_species))
    B = np.zeros((n_reactions, n_species))
    rates = np.array([r["rate"] for r in reactions])

    for r_idx, reaction in enumerate(reactions):
        reactants = reaction["reactants"]
        products = reaction["products"]
        for s_name, stoich in reactants.items():
            s_idx = species_index[s_name]
            A[r_idx, s_idx] = stoich
        for s_name, stoich in products.items():
            s_idx = species_index[s_name]
            B[r_idx, s_idx] = stoich

    # Initial concentrations
    y0 = np.array([s["initial"] for s in species])

    # Compute atol
    max_initial = np.max(y0)
    atol = max_initial * 1e-12 if max_initial > 0 else 1e-12

    # Define the ODE function
    def dydt(y, t):
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            rate = rates[r_idx]
            product = 1.0
            for s_idx in range(n_species):
                if A[r_idx, s_idx] > 0:
                    product *= y[s_idx] ** A[r_idx, s_idx]
            v[r_idx] = rate * product
        return (B - A) @ v

    # Define the Jacobian function
    def jac(y, t):
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            rate = rates[r_idx]
            product = 1.0
            for s_idx in range(n_species):
                if A[r_idx, s_idx] > 0:
                    product *= y[s_idx] ** A[r_idx, s_idx]
            v[r_idx] = rate * product
        # Compute dv_dy
        dv_dy = np.zeros((n_reactions, n_species))
        for r_idx in range(n_reactions):
            for s_idx in range(n_species):
                if A[r_idx, s_idx] > 0:
                    if y[s_idx] == 0.0:
                        dv_dy[r_idx, s_idx] = 0.0
                    else:
                        dv_dy[r_idx, s_idx] = (rates[r_idx] * A[r_idx, s_idx] * v[r_idx]) / y[s_idx]
        # Compute (B - A) @ dv_dy
        return (B - A) @ dv_dy

    # Solve the ODE
    sol = solve_ivp(
        fun=dydt,
        t_span=(0.0, network["t_end"]),
        y0=y0,
        t_eval=t_eval,
        method="BDF",
        jac=jac,
        rtol=1e-8,
        atol=atol
    )

    return sol.y.T
