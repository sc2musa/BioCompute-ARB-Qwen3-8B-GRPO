import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract species and reactions
    species = network["species"]
    n_species = len(species)
    
    # Create a dictionary to map species names to indices
    species_index = {species[i]["name"]: i for i in range(n_species)}
    
    # Initial concentrations
    y0 = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def dydt(t, y):
        # Calculate the rate of each reaction
        rates = []
        for reaction in network["reactions"]:
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the rate of the reaction
            rate_value = rate
            for species_name, stoichiometry in reactants.items():
                rate_value *= y[species_index[species_name]] ** stoichiometry
            
            rates.append(rate_value)
        
        # Calculate the change in each species
        dydt = np.zeros(n_species)
        for i, reaction in enumerate(network["reactions"]):
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = rates[i]
            
            # Calculate the change in each species due to this reaction
            for species_name, stoichiometry in products.items():
                idx = species_index[species_name]
                dydt[idx] += stoichiometry * rate
            for species_name, stoichiometry in reactants.items():
                idx = species_index[species_name]
                dydt[idx] -= stoichiometry * rate
        
        return dydt
    
    # Solve the ODE system using LSODA method with tighter tolerances
    sol = solve_ivp(dydt, [0, network["t_end"]], y0, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-8)
    
    # Return the solution
    return sol.y.T