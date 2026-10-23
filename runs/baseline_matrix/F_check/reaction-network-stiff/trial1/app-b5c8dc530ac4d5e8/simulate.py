import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract species information
    species = network["species"]
    n_species = len(species)
    
    # Extract reaction information
    reactions = network["reactions"]
    
    # Create a dictionary to map species names to indices
    species_indices = {species[i]["name"] : i for i in range(n_species)}
    
    # Initialize concentrations
    concentrations = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def dydt(t, y):
        rates = np.zeros(n_species)
        
        for reaction in reactions:
            # Calculate the rate of this reaction
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the product of concentrations for reactants
            reactant_product = 1.0
            for species_name, stoich in reactants.items():
                if species_name in species_indices:
                    reactant_product *= y[species_indices[species_name]] ** stoich
            
            # Calculate the rate
            reaction_rate = rate * reactant_product
            
            # Update the rates for species involved in this reaction
            for species_name, stoich in products.items():
                if species_name in species_indices:
                    rates[species_indices[species_name]] += reaction_rate * stoich
            
            for species_name, stoich in reactants.items():
                if species_name in species_indices:
                    rates[species_indices[species_name]] -= reaction_rate * stoich
        
        return rates
    
    # Solve the ODE system with tighter tolerances
    sol = solve_ivp(dydt, [t_eval[0], t_eval[-1]], concentrations, t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-8)
    
    # Return the solution
    return sol.y.T