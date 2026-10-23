import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Parse network
    species = network["species"]
    reactions = network["reactions"]
    n_species = len(species)
    n_reactions = len(reactions)
    
    # Create species name to index mapping
    species_names = [sp["name"] for sp in species]
    species_idx = {name: i for i, name in enumerate(species_names)}
    
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
    
    # Define the ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            # Calculate reaction rate
            rate = reactions[r]["rate"]
            reactants = reactions[r]["reactants"]
            if reactants:
                # Product of concentrations raised to their stoichiometry
                product = 1.0
                for s, stoich in reactants.items():
                    product *= y[species_idx[s]] ** stoich
                v[r] = rate * product
            else:
                # Zero-order source
                v[r] = rate
        # Calculate dy/dt = S @ v
        return S @ v
    
    # Set tolerances
    rtol = 1e-8
    atol = 1e-12 * np.max(y0) if np.max(y0) > 0 else 1e-12
    
    # Solve the ODE
    sol = solve_ivp(
        fun=dydt,
        t_span=(0, network["t_end"]),
        y0=y0,
        t_eval=t_eval,
        method="BDF",
        rtol=rtol,
        atol=atol
    )
    
    # Return the solution
    return sol.y.T