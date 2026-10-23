import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract species and reactions
    species = network["species"]
    reactions = network["reactions"]
    n_species = len(species)
    
    # Initialize concentrations
    y0 = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def dydt(t, y):
        rates = np.zeros(n_species)
        
        # Calculate reaction rates
        for reaction in reactions:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate of this reaction
            rate_val = rate
            for species_name, stoich in reactants.items():
                idx = next(i for i, s in enumerate(species) if s["name"] == species_name)
                rate_val *= y[idx] ** stoich
            
            # Update the rate vector
            for species_name, stoich in products.items():
                idx = next(i for i, s in enumerate(species) if s["name"] == species_name)
                rates[idx] += rate_val
            
            # Subtract reactant contributions
            for species_name, stoich in reactants.items():
                idx = next(i for i, s in enumerate(species) if s["name"] == species_name)
                rates[idx] -= rate_val * stoich
        
        return rates
    
    # Solve the ODE
    sol = solve_ivp(dydt, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-8)
    
    # Return the solution
    return sol.y.T