import numpy as np
from scipy.integrate import solve_ivp

def simulate(network: dict, t_eval: list[float]) -> numpy.ndarray:
    species = network['species']
    reactions = network['reactions']
    n_species = len(species)
    n_reactions = len(reactions)

    # Extract initial concentrations
    y0 = [s['initial'] for s in species]

    # Build A matrix (reactant stoichiometry)
    A = np.zeros((n_reactions, n_species))
    for r in range(n_reactions):
        reactants = reactions[r]['reactants']
        for s_name, stoich in reactants.items():
            # Find the index of the species
            s_idx = next(i for i, s in enumerate(species) if s['name'] == s_name)
            A[r, s_idx] = stoich

    # Build S matrix (stoichiometry)
    S = np.zeros((n_species, n_reactions))
    for r in range(n_reactions):
        reactants = reactions[r]['reactants']
        products = reactions[r]['products']
        for s_idx, s in enumerate(species):
            s_name = s['name']
            reactant_stoich = reactants.get(s_name, 0)
            product_stoich = products.get(s_name, 0)
            S[s_idx, r] = product_stoich - reactant_stoich

    # Define the ODE function
    def dydt(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            rate = reactions[r]['rate']
            product = 1.0
            for s in range(n_species):
                product *= y[s] ** A[r, s]
            v[r] = rate * product
        return S @ v

    # Define the Jacobian function
    def jac(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            rate = reactions[r]['rate']
            product = 1.0
            for s in range(n_species):
                product *= y[s] ** A[r, s]
            v[r] = rate * product
        # Compute (dv/dy) matrix
        dv_dy = np.zeros((n_reactions, n_species))
        for r in range(n_reactions):
            for s in range(n_species):
                if y[s] == 0:
                    dv_dy[r, s] = 0.0
                else:
                    dv_dy[r, s] = v[r] * A[r, s] / y[s]
        # Compute J = S @ dv_dy
        J = S @ dv_dy
        return J

    # Set up the solver
    max_initial = max(y0)
    rtol = 1e-8
    atol = 1e-12 * max_initial

    # Solve the ODE
    sol = solve_ivp(
        fun=dydt,
        t_span=(0.0, network['t_end']),
        y0=y0,
        t_eval=t_eval,
        method='BDF',
        jac=jac,
        rtol=rtol,
        atol=atol
    )

    # Return the solution as a numpy array with shape (len(t_eval), n_species)
    return sol.y.T