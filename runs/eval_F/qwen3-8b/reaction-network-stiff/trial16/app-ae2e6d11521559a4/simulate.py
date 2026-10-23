import numpy as np


def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    """Return concentrations with shape (len(t_eval), n_species)."""
    
    # Extract species information
    species = network["species"]
    n_species = len(species)
    
    # Extract reaction information
    reactions = network["reactions"]
    
    # Initialize concentrations
    concentrations = np.array([s["initial"] for s in species])
    
    # Create a list of species names for easy lookup
    species_names = [s["name"] for s in species]
    
    # Create a dictionary to map species names to indices
    species_index = {name: i for i, name in enumerate(species_names)}
    
    # Create a list of reaction rate constants
    rates = [r["rate"] for r in reactions]
    
    # Create a list of reaction stoichiometry matrices
    stoichiometry = []
    for reaction in reactions:
        reactants = reaction["reactants"]
        products = reaction["products"]
        
        # Create a stoichiometry vector for this reaction
        stoichiometry_vector = np.zeros(n_species)
        
        # Subtract reactants
        for species_name, stoich in reactants.items():
            idx = species_index[species_name]
            stoichiometry_vector[idx] -= stoich
        
        # Add products
        for species_name, stoich in products.items():
            idx = species_index[species_name]
            stoichiometry_vector[idx] += stoich
        
        stoichiometry.append(stoichiometry_vector)
    
    # Initialize the result array
    result = np.zeros((len(t_eval), n_species))
    result[0] = concentrations.copy()
    
    # Solve the ODE
    t = 0.0
    for i in range(1, len(t_eval)):
        dt = t_eval[i] - t
        
        # Compute the reaction rates
        reaction_rates = np.zeros(len(reactions))
        for j, reaction in enumerate(reactions):
            reactants = reaction["reactants"]
            rate = rates[j]
            
            # Compute the product of concentrations
            product = 1.0
            for species_name, stoich in reactants.items():
                idx = species_index[species_name]
                if concentrations[idx] < 1e-20:
                    product = 0.0
                    break
                product *= concentrations[idx] ** stoich
            
            reaction_rates[j] = rate * product
        
        # Compute the change in concentrations
        d_conc = np.zeros(n_species)
        for j, reaction in enumerate(reactions):
            stoichiometry_vector = stoichiometry[j]
            d_conc += stoichiometry_vector * reaction_rates[j]
        
        # Update concentrations
        concentrations += d_conc * dt
        
        # Update time
        t = t_eval[i]
        
        # Store the result
        result[i] = concentrations.copy()
    
    return result