import numpy as np


def simulate(network, t_eval):
    """Return an array of shape (len(t_eval), n_species) with concentrations over time."""
    
    # Extract network parameters
    species = network["species"]
    reactions = network["reactions"]
    t_end = network["t_end"]
    
    # Initialize concentrations
    n_species = len(species)
    concentrations = np.array([s["initial"] for s in species])
    
    # Prepare reaction matrix
    stoichiometry = np.zeros((n_species, n_species))
    rate_constants = np.zeros(n_species)
    
    # Build stoichiometry matrix and rate constants
    for reaction in reactions:
        reactants = reaction["reactants"]
        products = reaction["products"]
        rate = reaction["rate"]
        
        # Update stoichiometry matrix
        for species_name, stoich in products.items():
            species_idx = next(i for i, s in enumerate(species) if s["name"] == species_name)
            stoichiometry[species_idx, species_idx] += stoich
        
        for species_name, stoich in reactants.items():
            species_idx = next(i for i, s in enumerate(species) if s["name"] == species_name)
            stoichiometry[species_idx, species_idx] -= stoich
        
        # Update rate constants
        rate_constants[species_idx] = rate
    
    # Prepare time points
    t_points = np.array(t_eval)
    
    # Initialize solution array
    solution = np.zeros((len(t_points), n_species))
    solution[0] = concentrations
    
    # Solve the system using the Rosenbrock method
    for i in range(1, len(t_points)):
        dt = t_points[i] - t_points[i-1]
        
        # Compute the Jacobian matrix
        J = np.zeros((n_species, n_species))
        for j in range(n_species):
            for k in range(n_species):
                J[j, k] = -rate_constants[k] * concentrations[k] ** (k == j) * stoichiometry[j, k]
        
        # Rosenbrock method parameters
        alpha = 1.0
        beta = 1.0
        gamma = 1.0
        
        # Initialize the solution
        y = concentrations.copy()
        
        # Rosenbrock iteration
        for _ in range(10):
            # Compute the right-hand side
            f = np.zeros(n_species)
            for j in range(n_species):
                # Calculate the reaction rate for species j
                rate_j = 0.0
                for reaction in reactions:
                    # Calculate the rate for this reaction
                    reactants = reaction["reactants"]
                    products = reaction["products"]
                    rate = reaction["rate"]
                    
                    # Calculate the product of concentrations for reactants
                    product = 1.0
                    for species_name, stoich in reactants.items():
                        species_idx = next(i for i, s in enumerate(species) if s["name"] == species_name)
                        product *= concentrations[species_idx] ** stoich
                    
                    # Calculate the rate for this reaction
                    rate_j += rate * product
                
                # Calculate the change in concentration for species j
                f[j] = np.sum(rate_j * stoichiometry[j, k] for k in range(n_species))
            
            # Solve the linear system (J + dt * alpha * beta * gamma * stoichiometry) * k = -f
            # This is a simplified version of the Rosenbrock method
            # In practice, we would need to solve the full nonlinear system
            # For simplicity, we'll use a fixed-point iteration
            
            # Update the solution
            y -= dt * f
            
        
        # Update the solution
        solution[i] = y
    
    return solution