import numpy as np
from scipy.integrate import solve_ivp


def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    # Parse network
    species = network['species']
    n_species = len(species)
    reactions = network['reactions']
    n_reactions = len(reactions)

    # Initialize matrices
    reactant_orders = np.zeros((n_reactions, n_species))
    stoichiometry = np.zeros((n_species, n_reactions))

    # Populate matrices
    for r_idx, reaction in enumerate(reactions):
        reactants = reaction['reactants']
        products = reaction['products']
        rate = reaction['rate']

        # Fill reactant_orders
        for species_name, stoich in reactants.items():
            s_idx = next(i for i, s in enumerate(species) if s['name'] == species_name)
            reactant_orders[r_idx, s_idx] = stoich

        # Fill stoichiometry
        for species_name, stoich in products.items():
            s_idx = next(i for i, s in enumerate(species) if s['name'] == species_name)
            stoichiometry[s_idx, r_idx] += stoich
        for species_name, stoich in reactants.items():
            s_idx = next(i for i, s in enumerate(species) if s['name'] == species_name)
            stoichiometry[s_idx, r_idx] -= stoich

    # Initial concentrations
    y0 = np.array([s['initial'] for s in species])

    # Define the ODE function
    def dydt(t, y):
        # Compute reaction rates
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            # Calculate the product of concentrations raised to their stoichiometry
            product = 1.0
            for s_idx in range(n_species):
                if reactant_orders[r_idx, s_idx] > 0:
                    product *= y[s_idx] ** reactant_orders[r_idx, s_idx]
            v[r_idx] = reactions[r_idx]['rate'] * product

        # Compute the derivative
        dydt = stoichiometry @ v
        return dydt

    # Set tolerances
    rtol = 1e-8
    atol = 1e-12 * np.max(y0) if np.max(y0) > 0 else 1e-12

    # Solve the ODE
    sol = solve_ivp(dydt, [0, t_eval[-1]], y0, t_eval=t_eval, method='LSODA', rtol=rtol, atol=atol)

    # Return the solution
    return sol.y.T