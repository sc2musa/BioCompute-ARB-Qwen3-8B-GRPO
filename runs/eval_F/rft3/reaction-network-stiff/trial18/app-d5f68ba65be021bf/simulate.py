import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    species = network["species"]
    reactions = network["reactions"]
    t_end = network["t_end"]

    # Parse network
    n_species = len(species)
    n_reactions = len(reactions)

    # Create stoichiometry matrix S (species x reactions)
    S = np.zeros((n_species, n_reactions))
    # Create reactant order matrix A (reactions x species)
    A = np.zeros((n_reactions, n_species))

    for r_idx, reaction in enumerate(reactions):
        reactants = reaction["reactants"]
        products = reaction["products"]
        rate = reaction["rate"]

        # Fill reactant order matrix A
        for s_name, stoich in reactants.items():
            s_idx = next(i for i, s in enumerate(species) if s["name"] == s_name)
            A[r_idx, s_idx] = stoich

        # Fill stoichiometry matrix S
        for s_name, stoich in products.items():
            s_idx = next(i for i, s in enumerate(species) if s["name"] == s_name)
            S[s_idx, r_idx] = stoich
        for s_name, stoich in reactants.items():
            s_idx = next(i for i, s in enumerate(species) if s["name"] == s_name)
            S[s_idx, r_idx] -= stoich

    # Initial concentrations
    y0 = np.array([s["initial"] for s in species])

    # Define the ODE system
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            # Calculate reaction rate
            rate = reactions[r_idx]["rate"]
            reactants = reactions[r_idx]["reactants"]
            product_terms = 1.0
            for s_name, stoich in reactants.items():
                s_idx = next(i for i, s in enumerate(species) if s["name"] == s_name)
                product_terms *= y[s_idx] ** stoich
            v[r_idx] = rate * product_terms

        # Calculate dy/dt = S @ v
        return S @ v

    # Set solver options
    rtol = 1e-8
    atol = 1e-12 * np.max(y0) if np.max(y0) > 0 else 1e-12

    # Solve the ODE
    sol = solve_ivp(dydt, [0, t_end], y0, t_eval=t_eval, method="BDF", rtol=rtol, atol=atol)

    # Return the solution
    return sol.y.T