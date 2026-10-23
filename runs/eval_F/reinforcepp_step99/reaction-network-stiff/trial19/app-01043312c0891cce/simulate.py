import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Parse network
    species = network["species"]
    reactions = network["reactions"]
    n_species = len(species)
    n_reactions = len(reactions)
    
    # Create reactant-order matrix A and net stoichiometry matrix S
    A = np.zeros((n_reactions, n_species))
    S = np.zeros((n_species, n_reactions))
    
    for r, reaction in enumerate(reactions):
        reactants = reaction["reactants"]
        products = reaction["products"]
        rate = reaction["rate"]
        
        # Fill reactant-order matrix A
        for s, stoich in reactants.items():
            s_idx = next(i for i, sp in enumerate(species) if sp["name"] == s)
            A[r, s_idx] = stoich
        
        # Fill net stoichiometry matrix S
        for s, stoich in products.items():
            s_idx = next(i for i, sp in enumerate(species) if sp["name"] == s)
            S[s_idx, r] = stoich
        
        # Subtract reactant stoichiometry from products
        for s, stoich in reactants.items():
            s_idx = next(i for i, sp in enumerate(species) if sp["name"] == s)
            S[s_idx, r] -= stoich
        
    
    # Initial concentrations
    y0 = np.array([sp["initial"] for sp in species])
    
    # Define the ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            # Compute reaction rate
            rate = reactions[r]["rate"]
            reactants = reactions[r]["reactants"]
            
            # Compute product of concentrations raised to their stoichiometry
            product = 1.0
            for s, stoich in reactants.items():
                s_idx = next(i for i, sp in enumerate(species) if sp["name"] == s)
                product *= y[s_idx] ** stoich
            
            v[r] = rate * product
        
        # Compute dy/dt = S @ v
        return S @ v
    
    # Set tolerances
    rtol = 1e-8
    atol = 1e-12 * np.max(y0) if np.max(y0) > 0 else 1e-12
    
    # Solve the ODE
    sol = solve_ivp(dydt, [0, network["t_end"]], y0, t_eval=t_eval, method="BDF",
                    rtol=rtol, atol=atol)
    
    # Return the solution
    return sol.y.T