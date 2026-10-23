import numpy as np
from scipy.integrate import solve_ivp


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract species and reactions
    species = network["species"]
    n_species = len(species)
    
    # Extract rate constants and stoichiometry matrices
    stoichiometry = np.zeros((n_species, n_species))
    rate_constants = np.zeros(n_species)
    
    for i, reaction in enumerate(network["reactions"]):
        reactants = reaction["reactants"]
        products = reaction["products"]
        rate = reaction["rate"]
        
        # Update stoichiometry matrix
        for species_name, stoich in products.items():
            species_idx = next(j for j, s in enumerate(species) if s["name"] == species_name)
            stoichiometry[species_idx, species_idx] += stoich
        
        for species_name, stoich in reactants.items():
            species_idx = next(j for j, s in enumerate(species) if s["name"] == species_name)
            stoichiometry[species_idx, species_idx] -= stoich
        
        # Update rate constants
        rate_constants[i] = rate
    
    # Initial concentrations
    initial_concentrations = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def dydt(t, y):
        rates = np.zeros(len(network["reactions"]))
        
        # Calculate reaction rates
        for i, reaction in enumerate(network["reactions"]):
            reactants = reaction["reactants"]
            rate = reaction["rate"]
            
            # Calculate the product of concentrations
            rate_product = 1.0
            for species_name, stoich in reactants.items():
                species_idx = next(j for j, s in enumerate(species) if s["name"] == species_name)
                rate_product *= y[species_idx] ** stoich
            
            rates[i] = rate * rate_product
        
        # Calculate the derivative of concentrations
        dydt = np.zeros(n_species)
        for i, reaction in enumerate(network["reactions"]):
            reactants = reaction["reactants"]
            products = reaction["products"]
            rate = reaction["rate"]
            
            # Calculate the product of concentrations
            rate_product = 1.0
            for species_name, stoich in reactants.items():
                species_idx = next(j for j, s in enumerate(species) if s["name"] == species_name)
                rate_product *= y[species_idx] ** stoich
            
            # Update the derivative
            for species_name, stoich in products.items():
                species_idx = next(j for j, s in enumerate(species) if s["name"] == species_name)
                dydt[species_idx] += stoich * rate * rate_product
            
            for species_name, stoich in reactants.items():
                species_idx = next(j for j, s in enumerate(species) if s["name"] == species_name)
                dydt[species_idx] -= stoich * rate * rate_product
        
        return dydt
    
    # Solve the ODE system
    sol = solve_ivp(dydt, (0, network["t_end"]), initial_concentrations, t_eval=t_eval, method='BDF')
    
    # Return the solution
    return sol.y.T