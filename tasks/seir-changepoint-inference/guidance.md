## Method guidance

1. **Forward model.** Integrate SEIR with an extra cumulative-incidence state `C`
   (`dC/dt = sigma * E`). Expected cases on day `d` are `C(d) - C(d-1)`. Integrate on the
   integer grid `0..T` and make the solver respect the discontinuity in `beta` at the change
   day (e.g. `scipy.integrate.odeint(..., tcrit=[change_day])` or integrate the two
   segments separately). Convention: `beta` switches at time `t = change_day`
   (`R0` for `t < change_day`), so day `change_day + 1`, the interval
   `(change_day, change_day + 1]`, is the first day fully under `R_post`.
2. **Likelihood.** The counts are over-dispersed; a negative-binomial log-likelihood
   (dispersion around 10–50) or a Poisson likelihood both work for point estimates.
   Optimise over `log R0` and `log R_post` so the parameters stay positive.
3. **Change day.** The likelihood is not smooth in the change day. Profile it: for each
   integer candidate day (leaving a few days at both ends), fit `(R0, R_post)` with
   Nelder–Mead and keep the day with the best likelihood.
4. **Model selection.** Also fit a no-intervention model (one `R` for all days) and
   prefer it unless the change-point model is clearly better, e.g. by BIC
   (`2*NLL + k*ln(n)` with `k = 1` vs `k = 3`). Then return `R_post = R0` and
   `change_day = None`.
5. **Forecast.** Integrate the fitted model to day `70 + horizon` and return the expected
   cases for days 71..70+horizon.
6. Validate on `/app/data/example*.json` (they include the truth) with the grading formulas
   before submitting, and keep each fit well under the 120 s limit. The profile runs about
   60 two-parameter fits, so make one likelihood evaluation cheap (`odeint` on the integer
   grid takes milliseconds; avoid re-creating solvers with dense output). `check.py` fits
   three outbreaks, which can exceed the bash tool's default 120 s timeout: pass a larger
   `timeout` when running it.
