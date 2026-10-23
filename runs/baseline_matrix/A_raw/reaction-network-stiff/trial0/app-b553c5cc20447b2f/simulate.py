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
        rates = np.zeros(n_species)
        
        # Calculate the rate for each reaction
        for reaction in network["reactions"]:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate of the reaction
            rate_reaction = rate
            for species_name, stoich in reactants.items():
                rate_reaction *= y[species_index[species_name]] ** stoich
            
            # Update the rates for the species
            for species_name, stoich in products.items():
                index = species_index[species_name]
                rates[index] += stoich * rate_reaction
            
            for species_name, stoich in reactants.items():
                index = species_index[species_name]
                rates[index] -= stoich * rate_reaction
        
        return rates
    
    # Create a mapping from species names to indices
    species_index = {s["name"] : i for i, s in enumerate(species)}
    
    # Solve the ODE system
    sol = solve_ivp(dydt, [0, network["t_end"]], y0, t_eval=t_eval, method='LSODA')
    
    # Return the solution
    return sol.y.T
