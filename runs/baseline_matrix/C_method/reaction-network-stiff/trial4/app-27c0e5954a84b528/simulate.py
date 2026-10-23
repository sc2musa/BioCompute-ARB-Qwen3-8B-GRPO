import numpy as np
from scipy.integrate import solve_ivp


def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    # Extract species information
    species = network['species']
    n_species = len(species)
    initial_concentrations = np.array([s['initial'] for s in species])

    # Extract reaction information
    reactions = network['reactions']
    n_reactions = len(reactions)

    # Create reactant-order matrix A and net stoichiometry matrix S
    A = np.zeros((n_reactions, n_species))
    S = np.zeros((n_species, n_reactions))

    for r_idx, reaction in enumerate(reactions):
        reactants = reaction['reactants']
        products = reaction['products']
        rate = reaction['rate']

        # Populate reactant-order matrix A
        for s_idx, (species_name, stoich) in enumerate(reactants.items()):
            A[r_idx, s_idx] = stoich

        # Populate net stoichiometry matrix S
        for s_idx, (species_name, stoich) in products.items():
            S[s_idx, r_idx] += stoich
        for s_idx, (species_name, stoich) in reactants.items():
            S[s_idx, r_idx] -= stoich

    # Define the ODE system
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            # Calculate reaction rate
            rate = reactions[r_idx]['rate']
            # Calculate product of concentrations raised to reactant orders
            product = 1.0
            for s_idx, (species_name, stoich) in enumerate(reactions[r_idx]['reactants'].items()):
                product *= y[s_idx] ** stoich
            v[r_idx] = rate * product

        # Calculate dy/dt
        dydt = np.zeros(n_species)
        for s_idx in range(n_species):
            for r_idx in range(n_reactions):
                dydt[s_idx] += S[s_idx, r_idx] * v[r_idx]
        return dydt

    # Define the Jacobian
    def jac(t, y):
        J = np.zeros((n_species, n_reactions))
        for r_idx in range(n_reactions):
            rate = reactions[r_idx]['rate']
            # Calculate product of concentrations raised to reactant orders
            product = 1.0
            for s_idx, (species_name, stoich) in enumerate(reactions[r_idx]['reactants'].items()):
                product *= y[s_idx] ** stoich
            # Calculate derivative of reaction rate with respect to each species
            for s_idx in range(n_species):
                if species[s_idx]['name'] in reactions[r_idx]['reactants']:
                    stoich = reactions[r_idx]['reactants'][species[s_idx]['name']]
                    J[s_idx, r_idx] = rate * stoich * y[s_idx] ** (stoich - 1) * product / y[s_idx]
        return J

    # Set solver options
    rtol = 1e-8
    atol = 1e-12 * np.max(initial_concentrations)

    # Solve the ODE
    sol = solve_ivp(
        fun=dydt,
        t_span=(0.0, network['t_end']),
        y0=initial_concentrations,
        t_eval=t_eval,
        method='BDF',
        jac=jac,
        rtol=rtol,
        atol=atol
    )

    # Return the solution
    return sol.y.T
