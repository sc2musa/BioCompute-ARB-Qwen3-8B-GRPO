## Method guidance

1. Parse the network once into arrays: a reactant-order matrix `A` (reactions x species) and a
   net stoichiometry matrix `S = products - reactants` (species x reactions). Remember
   zero-order sources (empty reactants), degradation (empty products), stoichiometry 2
   (`v = k * [P]**2`) and species on both sides (net change 0).
2. Right-hand side: `dy/dt = S @ v(y)` with `v_r = k_r * prod_s y_s ** A[r, s]`.
3. The systems are **stiff** (rate constants span up to ~10 orders of magnitude). Explicit
   methods such as RK45 will need millions of steps and time out. Use an implicit method:
   `scipy.integrate.solve_ivp(..., method="BDF")` (or `"Radau"` / `"LSODA"`).
4. The default tolerances (`rtol=1e-3`) are far too loose for the 1e-4 accuracy bound. Use
   about `rtol=1e-8` and an absolute tolerance scaled to the problem, e.g.
   `atol = 1e-12 * max(initial concentrations)` (fall back to `1e-12` if every initial
   concentration is 0). The bound is relative per species, so species that stay tiny (e.g.
   the ~1e-5 intermediate of the Robertson system) must be accurate too. There the absolute
   tolerance decides: `atol=1e-9` already gives errors near 1e-3 on that species, while
   `atol=1e-12` passes even with `rtol=1e-6`.
5. Optional: an analytic Jacobian. Without `jac`, BDF/Radau estimate it by finite differences,
   which is accurate and fast enough for these networks, so get a correct solution first.
   If you add it, mind the shapes: `dv/dy` is `(n_reactions, n_species)`, `S` is
   `(n_species, n_reactions)`, and `jac` must return `S @ dv_dy` with shape
   `(n_species, n_species)`. For every reactant `s` of reaction `r` (`A[r,s] > 0`),
   `dv_r/dy_s = k_r * A[r,s] * y_s**(A[r,s]-1) * prod_{s' != s} y_s'**A[r,s']`, and 0 otherwise
   (loop over the reactants only: `0**-1` would give `inf * 0 = nan`).
6. Pass `t_eval` to the solver and return `sol.y.T`. Check your error on both files in
   `/app/examples/` with the same formula the grader uses.
