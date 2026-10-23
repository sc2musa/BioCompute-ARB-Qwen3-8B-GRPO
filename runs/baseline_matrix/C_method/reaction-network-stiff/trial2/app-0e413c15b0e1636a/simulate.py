import numpy as np
from scipy.integrate import solve_ivp


def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    # Extract species and reactions
    species = network['species']
    n_species = len(species)
    reactions = network['reactions']
    n_reactions = len(reactions)

    # Initialize concentrations
    y0 = np.array([s['initial'] for s in species])

    # Create reactant-order matrix A and stoichiometry matrix S
    A = np.zeros((n_reactions, n_species))
    S = np.zeros((n_species, n_reactions))

    for r, reaction in enumerate(reactions):
        reactants = reaction['reactants']
        products = reaction['products']
        rate = reaction['rate']

        # Populate reactant-order matrix A
        for s, stoich in reactants.items():
            species_idx = next(i for i, s2 in enumerate(species) if s2['name'] == s)
            A[r, species_idx] = stoich

        # Populate stoichiometry matrix S
        for s, stoich in products.items():
            species_idx = next(i for i, s2 in enumerate(species) if s2['name'] == s)
            S[species_idx, r] = stoich
        for s, stoich in reactants.items():
            species_idx = next(i for i, s2 in enumerate(species) if s2['name'] == s)
            S[species_idx, r] -= stoich

    # Define the ODE system
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            # Calculate reaction rate
            rate = reactions[r]['rate']
            product = 1.0
            for s, stoich in reactions[r]['reactants'].items():
                species_idx = next(i for i, s2 in enumerate(species) if s2['name'] == s)
                product *= y[species_idx] ** stoich
            v[r] = rate * product
        # Calculate dy/dt
        dydt = S @ v
        return dydt

    # Define the Jacobian
    def jac(t, y):
        J = np.zeros((n_species, n_reactions))
        for r in range(n_reactions):
            rate = reactions[r]['rate']
            product = 1.0
            for s, stoich in reactions[r]['reactants'].items():
                species_idx = next(i for i, s2 in enumerate(species) if s2['name'] == s)
                product *= y[species_idx] ** stoich
            # Calculate derivative of reaction rate with respect to each species
            for s in range(n_species):
                if s in reactions[r]['reactants']:
                    stoich = reactions[r]['reactants'][species[s]['name']]
                    J[s, r] = rate * stoich * y[s] ** (stoich - 1) * product / y[s]
                else:
                    J[s, r] = 0.0
        # Calculate Jacobian matrix
        return S @ J

    # Set integration parameters
    rtol = 1e-8
    atol = 1e-12 * np.max(y0)

    # Solve the ODE system
    sol = solve_ivp(dydt, [0, t_eval[-1]], y0, t_eval=t_eval, method='BDF', rtol=rtol, atol=atol, jac=jac)

    # Return the solution
    return sol.y.T