import numpy as np
from scipy.integrate import solve_ivp

def simulate(network, t_eval):
    species = network['species']
    n_species = len(species)
    species_names = [s['name'] for s in species]
    n_reactions = len(network['reactions'])
    reaction_rates = [r['rate'] for r in network['reactions']]

    # Build A and S matrices
    A = np.zeros((n_reactions, n_species))
    S = np.zeros((n_species, n_reactions))
    for r in range(n_reactions):
        reactants = network['reactions'][r]['reactants']
        products = network['reactions'][r]['products']
        for s in range(n_species):
            A[r][s] = reactants.get(species_names[s], 0)
            S[s][r] = products.get(species_names[s], 0) - reactants.get(species_names[s], 0)

    initial_concentrations = np.array([s['initial'] for s in species])
    max_initial = np.max(initial_concentrations)
    atol = 1e-12 * max_initial if max_initial > 0 else 1e-12

    def f(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            product = 1.0
            for s in range(n_species):
                product *= y[s] ** A[r][s]
            v[r] = reaction_rates[r] * product
        dydt = np.dot(S, v)
        return dydt

    def jacobian(t, y):
        v = np.zeros(n_reactions)
        for r in range(n_reactions):
            product = 1.0
            for s in range(n_species):
                product *= y[s] ** A[r][s]
            v[r] = reaction_rates[r] * product
        # compute dv_dy
        dv_dy = np.zeros((n_reactions, n_species))
        for r in range(n_reactions):
            v_r = v[r]
            for s in range(n_species):
                if y[s] == 0:
                    dv_dy[r][s] = 0.0
                else:
                    dv_dy[r][s] = reaction_rates[r] * A[r][s] * v_r / y[s]
        # compute J = S @ dv_dy
        J = np.dot(S, dv_dy)
        return J

    sol = solve_ivp(
        fun=f,
        t_span=(0.0, network['t_end']),
        y0=initial_concentrations,
        t_eval=t_eval,
        method='BDF',
        rtol=1e-8,
        atol=atol,
        jac=jacobian
    )

    return sol.y.T