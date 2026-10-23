import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Parse network
    species = network["species"]
    reactions = network["reactions"]
    n_species = len(species)
    n_reactions = len(reactions)
    
    # Create reactant order matrix A (reactions x species)
    A = np.zeros((n_reactions, n_species))
    for r, reaction in enumerate(reactions):
        for s, stoich in reaction["reactants"].items():
            s_idx = next(i for i, sp in enumerate(species) if sp["name"] == s)
            A[r, s_idx] = stoich
    
    # Create net stoichiometry matrix S (species x reactions)
    S = np.zeros((n_species, n_reactions))
    for r, reaction in enumerate(reactions):
        for s, stoich in reaction["products"].items():
            s_idx = next(i for i, sp in enumerate(species) if sp["name"] == s)
            S[s_idx, r] += stoich
        for s, stoich in reaction["reactants"].items():
            s_idx = next(i for i, sp in enumerate(species) if sp["name"] == s)
            S[s_idx, r] -= stoich
    
    # Initial concentrations
    y0 = np.array([sp["initial"] for sp in species])
    
    # Define ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            # Calculate reaction rate
            rate = reactions[r]["rate"]
            # Calculate product of concentrations raised to reactant orders
            prod = 1.0
            for s, stoich in reactions[r]["reactants"].items():
                s_idx = next(i for i, sp in enumerate(species) if sp["name"] == s)
                prod *= y[s_idx] ** stoich
            v[r] = rate * prod
        # Calculate dy/dt = S @ v
        return S @ v
    
    # Set tolerances
    rtol = 1e-8
    atol = 1e-12 * np.max(y0) if np.max(y0) > 0 else 1e-12
    
    # Solve ODE
    sol = solve_ivp(dydt, [0, network["t_end"]], y0, t_eval=t_eval, method="BDF",
                    rtol=rtol, atol=atol, dense_output=True)
    
    # Return solution
    return sol.y.T