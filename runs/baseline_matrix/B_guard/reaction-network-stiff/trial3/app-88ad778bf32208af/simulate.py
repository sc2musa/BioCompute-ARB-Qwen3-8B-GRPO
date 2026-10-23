import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract species and reactions
    species = network["species"]
    n_species = len(species)
    
    # Initial concentrations
    y0 = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def dydt(t, y):
        rates = np.zeros(len(network["reactions"]))
        
        # Calculate reaction rates
        for i, reaction in enumerate(network["reactions"]):
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate of the reaction
            rate_val = rate
            for species_name, stoich in reactants.items():
                idx = next(j for j, s in enumerate(species) if s["name"] == species_name)
                rate_val *= y[idx] ** stoich
            
            rates[i] = rate_val
        
        # Calculate the derivative of each species
        dydt = np.zeros(n_species)
        for i, reaction in enumerate(network["reactions"]):
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = rates[i]
            
            # Update the derivative for each species
            for species_name, stoich in products.items():
                idx = next(j for j, s in enumerate(species) if s["name"] == species_name)
                dydt[idx] += stoich * rate
            
            for species_name, stoich in reactants.items():
                idx = next(j for j, s in enumerate(species) if s["name"] == species_name)
                dydt[idx] -= stoich * rate
        
        return dydt
    
    # Solve the ODE
    sol = solve_ivp(dydt, [0, network["t_end"]], y0, t_eval=t_eval, method='LSODA')
    
    # Return the solution
    return sol.y.T