import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return concentrations with shape (len(t_eval), n_species)."""
    
    # Parse network
    species = network["species"]
    n_species = len(species)
    reactions = network["reactions"]
    n_reactions = len(reactions)

    # Create mapping from species name to index
    species_idx = {sp["name"] : i for i, sp in enumerate(species)}

    # Create reactant order matrix A (reactions x species)
    A = np.zeros((n_reactions, n_species))
    for r, reaction in enumerate(reactions):
        for s, stoich in reaction["reactants"].items():
            A[r, species_idx[s]] = stoich

    # Create stoichiometry matrix S (species x reactions)
    S = np.zeros((n_species, n_reactions))
    for r, reaction in enumerate(reactions):
        for s, stoich in reaction["products"].items():
            S[species_idx[s], r] = stoich
        for s, stoich in reaction["reactants"].items():
            S[species_idx[s], r] -= stoich

    # Initial concentrations
    y0 = np.array([sp["initial"] for sp in species])

    # Define the ODE system
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            # Calculate reaction rate
            reactant_conc = y[A[r] > 0]
            rate = reactions[r]["rate"] * np.prod(reactant_conc ** A[r, A[r] > 0])
            v[r] = rate
        # Calculate dy/dt = S @ v
        return S @ v

    # Set tolerances
    rtol = 1e-8
    atol = 1e-12 * np.max(y0) if np.max(y0) > 0 else 1e-12

    # Solve the ODE
    sol = solve_ivp(dydt, [0, network["t_end"]], y0, t_eval=t_eval, method="BDF",
                    rtol=rtol, atol=atol, dense_output=True)

    # Return solution
    return sol.y.T