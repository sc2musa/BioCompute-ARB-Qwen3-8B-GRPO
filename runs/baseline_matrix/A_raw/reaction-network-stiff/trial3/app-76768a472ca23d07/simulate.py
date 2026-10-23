import numpy as np
from scipy.integrate import solve_ivp


def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    """Return concentrations with shape (len(t_eval), n_species)."""
    
    # Extract species information
    species = network["species"]
    n_species = len(species)
    
    # Extract reaction information
    reactions = network["reactions"]
    
    # Initial concentrations
    y0 = np.array([s["initial"] for s in species])
    
    # Define the ODE system
    def dydt(t, y):
        dy = np.zeros(n_species)
        
        # Calculate the rate of each reaction
        for reaction in reactions:
            # Calculate the rate of the reaction
            rate = reaction["rate"]
            
            # Calculate the product of concentrations raised to their stoichiometric coefficients
            reactant_concentrations = 1.0
            for species_name, stoichiometry in reaction["reactants"].items():
                idx = next(i for i, s in enumerate(species) if s["name"] == species_name)
                reactant_concentrations *= y[idx] ** stoichiometry
            
            # Calculate the change in each species
            for species_name, stoichiometry in reaction["products"].items():
                idx = next(i for i, s in enumerate(species) if s["name"] == species_name)
                dy[idx] += rate * reactant_concentrations * stoichiometry
            
            # Subtract the change for reactants
            for species_name, stoichiometry in reaction["reactants"].items():
                idx = next(i for i, s in enumerate(species) if s["name"] == species_name)
                dy[idx] -= rate * reactant_concentrations * stoichiometry
        
        return dy
    
    # Solve the ODE system
    sol = solve_ivp(
        fun=dydt,
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-6,
        atol=1e-6
    )
    
    # Return the solution
    return sol.y.T