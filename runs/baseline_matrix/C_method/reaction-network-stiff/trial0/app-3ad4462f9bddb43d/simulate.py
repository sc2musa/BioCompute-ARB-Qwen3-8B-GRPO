import numpy as np
from scipy.integrate import solve_ivp


def simulate(network: dict, t_eval: list[float]) -> np.ndarray:
    # Parse the network into arrays
    species = network['species']
    n_species = len(species)
    reactions = network['reactions']
    n_reactions = len(reactions)

    # Initialize concentration array
    concentrations = np.array([s['initial'] for s in species])

    # Create reactant-order matrix A (reactions x species)
    A = np.zeros((n_reactions, n_species))
    # Create net stoichiometry matrix S (species x reactions)
    S = np.zeros((n_species, n_reactions))

    for r_idx, reaction in enumerate(reactions):
        reactants = reaction['reactants']
        products = reaction['products']
        rate = reaction['rate']

        # Fill reactant-order matrix A
        for s_idx, stoich in reactants.items():
            A[r_idx, s_idx] = stoich

        # Fill net stoichiometry matrix S
        for s_idx, stoich in products.items():
            S[s_idx, r_idx] += stoich
        for s_idx, stoich in reactants.items():
            S[s_idx, r_idx] -= stoich

    # Define the ODE system
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r_idx in range(n_reactions):
            # Calculate reaction rate
            rate = reactions[r_idx]['rate']
            # Calculate product of concentrations raised to reactant orders
            product = 1.0
            for s_idx, stoich in reactions[r_idx]['reactants'].items():
                product *= y[s_idx] ** stoich
            v[r_idx] = rate * product

        # Calculate dy/dt = S @ v
        return S @ v

    # Define the Jacobian
    def jacobian(t, y):
        J = np.zeros((n_species, n_reactions))
        for r_idx in range(n_reactions):
            rate = reactions[r_idx]['rate']
            reactants = reactions[r_idx]['reactants']
            for s_idx, stoich in reactants.items():
                # Calculate derivative of reaction rate with respect to concentration
                derivative = rate * stoich * y[s_idx] ** (stoich - 1) * np.prod([y[s_idx2] ** reactants[s_idx2] for s_idx2 in reactants if s_idx2 != s_idx])
                J[s_idx, r_idx] = derivative
        # Jacobian for the ODE system is S @ J
        return S @ J

    # Set tolerances
    rtol = 1e-8
    atol = 1e-12 * np.max(concentrations)

    # Solve the ODE
    sol = solve_ivp(
        fun=dydt,
        t_span=(0.0, network['t_end']),
        y0=concentrations,
        t_eval=t_eval,
        method='BDF',
        rtol=rtol,
        atol=atol,
        jac=jacobian
    )

    # Return the solution
    return sol.y.T
