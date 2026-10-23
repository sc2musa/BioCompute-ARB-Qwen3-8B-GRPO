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
    species_index = {s["name"] : i for i, s in enumerate(species)}
    
    # Create reactant order matrix A (reactions x species)
    A = np.zeros((n_reactions, n_species))
    for r, reaction in enumerate(reactions):
        for s_name, stoich in reaction["reactants"].items():
            s = species_index[s_name]
            A[r, s] = stoich
    
    # Create net stoichiometry matrix S (species x reactions)
    S = np.zeros((n_species, n_reactions))
    for r, reaction in enumerate(reactions):
        for s_name, stoich in reaction["products"].items():
            s = species_index[s_name]
            S[s, r] += stoich
        for s_name, stoich in reaction["reactants"].items():
            s = species_index[s_name]
            S[s, r] -= stoich
    
    # Initial concentrations
    y0 = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            # Calculate reaction rate
            rate = reactions[r]["rate"]
            # Calculate product of concentrations raised to reactant orders
            product = 1.0
            for s in range(n_species):
                if A[r, s] > 0:
                    product *= y[s] ** A[r, s]
            v[r] = rate * product
        # Calculate dy/dt = S @ v
        return S @ v
    
    # Set tolerances
    rtol = 1e-8
    atol = 1e-12 * np.max(y0) if np.max(y0) > 1e-12 else 1e-12
    
    # Solve the ODE
    sol = solve_ivp(dydt, [0, network["t_end"]], y0, t_eval=t_eval, method="BDF",
                    rtol=rtol, atol=atol, dense_output=True)
    
    # Return the solution
    return sol.y.T