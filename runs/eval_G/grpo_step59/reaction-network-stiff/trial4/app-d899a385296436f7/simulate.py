import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    species = network["species"]
    n_species = len(species)
    reactions = network["reactions"]
    n_reactions = len(reactions)

    # Parse reactant orders and stoichiometry
    A = np.zeros((n_reactions, n_species))
    S = np.zeros((n_species, n_reactions))

    for r_idx, reaction in enumerate(reactions):
        reactants = reaction["reactants"]
        products = reaction["products"]
        rate = reaction["rate"]

        # Fill reactant orders matrix A
        for s_name, stoich in reactants.items():
            s_idx = next(i for i, s in enumerate(species) if s["name"] == s_name)
            A[r_idx, s_idx] = stoich

        # Fill stoichiometry matrix S
        for s_name, stoich in products.items():
            s_idx = next(i for i, s in enumerate(species) if s["name"] == s_name)
            S[s_idx, r_idx] = stoich

        # Subtract reactant stoichiometry from product stoichiometry
        for s_name, stoich in reactants.items():
            s_idx = next(i for i, s in enumerate(species) if s["name"] == s_name)
            S[s_idx, r_idx] -= stoich

    # Initial concentrations
    y0 = np.array([s["initial"] for s in species])

    # Define the ODE system
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            # Compute reaction rate
            rate = reactions[r_idx]["rate"]
            reactants = reactions[r_idx]["reactants"]
            product_stoich = 0
            for s_name, stoich in reactants.items():
                s_idx = next(i for i, s in enumerate(species) if s["name"] == s_name)
                product_stoich *= y[s_idx] ** stoich
            v[r_idx] = rate * product_stoich

        # Compute dy/dt = S @ v
        return S @ v

    # Solve the ODE system
    sol = solve_ivp(dydt, [0, network["t_end"]], y0, t_eval=t_eval, method="BDF",
                    rtol=1e-8, atol=1e-12 * np.max(y0) if np.max(y0) > 1e-12 else 1e-12)

    # Return the solution
    return sol.y.T