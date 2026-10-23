import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return concentrations with shape (len(t_eval), n_species)."""
    
    # Parse network
    species = network["species"]
    reactions = network["reactions"]
    n_species = len(species)
    n_reactions = len(reactions)

    # Precompute species indices
    species_indices = {sp["name"] : i for i, sp in enumerate(species)}

    # Create reactant order matrix A (reactions x species)
    A = np.zeros((n_reactions, n_species))
    for r, reaction in enumerate(reactions):
        for s, stoich in reaction["reactants"].items():
            A[r, species_indices[s]] = stoich

    # Create net stoichiometry matrix S (species x reactions)
    S = np.zeros((n_species, n_reactions))
    for r, reaction in enumerate(reactions):
        for s, stoich in reaction["products"].items():
            S[species_indices[s], r] += stoich
        for s, stoich in reaction["reactants"].items():
            S[species_indices[s], r] -= stoich

    # Initial concentrations
    y0 = np.array([sp["initial"] for sp in species])

    # Define ODE function
    def dydt(t, y):
        # Calculate reaction rates
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            # Calculate product of concentrations raised to reactant orders
            product = 1.0
            for s in range(n_species):
                if A[r, s] > 0:
                    product *= y[s] ** A[r, s]
            v[r] = reactions[r]["rate"] * product

        # Calculate dy/dt = S @ v
        return S @ v

    # Set tolerances
    rtol = 1e-8
    atol = 1e-12 * np.max(y0) if np.max(y0) > 1e-12 else 1e-12

    # Solve ODE
    sol = solve_ivp(
        dydt,
        [0, network["t_end"]],
        y0,
        t_eval=t_eval,
        method="BDF",
        rtol=rtol,
        atol=atol
    )

    # Return solution
    return sol.y.T