import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Parse network
    species = network["species"]
    n_species = len(species)
    reactions = network["reactions"]
    n_reactions = len(reactions)

    # Create mapping from species name to index
    species_names = [s["name"] for s in species]
    name_to_idx = {name: idx for idx, name in enumerate(species_names)}

    # Create stoichiometry matrix
    A = np.zeros((n_reactions, n_species))  # Reactant order matrix
    S = np.zeros((n_species, n_reactions))  # Net stoichiometry matrix

    for r_idx, reaction in enumerate(reactions):
        reactants = reaction["reactants"]
        products = reaction["products"]
        rate = reaction["rate"]

        # Fill reactant order matrix
        for s_name, stoich in reactants.items():
            s_idx = name_to_idx[s_name]
            A[r_idx, s_idx] = stoich

        # Fill net stoichiometry matrix
        for s_name, stoich in products.items():
            s_idx = name_to_idx[s_name]
            S[s_idx, r_idx] += stoich
        for s_name, stoich in reactants.items():
            s_idx = name_to_idx[s_name]
            S[s_idx, r_idx] -= stoich

    # Initial concentrations
    y0 = np.array([s["initial"] for s in species])

    # Define ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            # Calculate reaction rate
            rate = reactions[r_idx]["rate"]
            reactants = reactions[r_idx]["reactants"]
            product_terms = 1.0
            for s_name, stoich in reactants.items():
                s_idx = name_to_idx[s_name]
                product_terms *= y[s_idx] ** stoich
            v[r_idx] = rate * product_terms

        # Calculate dy/dt = S @ v
        return S @ v

    # Set tolerances
    rtol = 1e-8
    atol = 1e-12 * np.max(y0) if np.max(y0) > 1e-12 else 1e-12

    # Solve ODE
    sol = solve_ivp(dydt, [0, network["t_end"]], y0, t_eval=t_eval, method="BDF",
                    rtol=rtol, atol=atol, dense_output=True)

    # Return solution
    return sol.y.T