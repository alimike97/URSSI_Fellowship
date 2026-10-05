# simwatchdog - Development Plan

`simwatchdog` is a Python package that detects silent failures in ML- and ROM-driven fluid simulations at runtime, and corrects them. It can be run offline over a folder of saved snapshots or wrapped around a solver loop.

This plan covers the first development stage. Edit freely; anything marked **Open** still needs a decision.

## Ground rules

- **No new methods.** Use established tools (numpy, scipy, scikit-learn, `opinf`, PyTorch). Small standard formulas (e.g. an EnKF analysis step) are written directly in numpy rather than adding a heavy dependency.
- **No classes of our own.** Same style as CompFlowLab: modules of plain functions, state carried in dicts (`monitor_param`, `report`), one topic per file.
- **1D CFD only for now.** Sod shock tube first, then colliding shocks and 1D flames.
- **Clean, modular, installable.** Dependencies declared in `pyproject.toml`, heavy ones as optional extras.

## Roles of the FOM data

FOM results come from CompFlowLab and are treated as ground truth (equivalent to experimental data). At this stage the FOM solution is assumed available at every grid point and every saved step. It has three roles, kept separate in the code:

| Role | Used by | Notes |
| --- | --- | --- |
| Training data | testbed surrogates | Only the window `[0, t_train]` |
| Observations | correction, innovation detector | Through an observation mask `obs_indices` (all cells for now, sparse later) |
| Ground truth | scoring | Defines the true failure onset; never seen by reference-free detectors |

There is no parameter sweep at this stage: one FOM trajectory per case.

## Data contract

- **Snapshot:** array of shape `(n_var, n_cell)`, ghost cells removed, as in CompFlowLab's `<iter>iteration_cons.npy` / `_prim.npy` / `_res.npy`.
- **Metadata dict:** grid `x`, `dt`, iteration, time, variable names, gas properties (e.g. `gamma`).
- **Residual callback (optional):** a function supplied by the solver that returns the governing-equation residual for a given state.

## Public API

```python
monitor_param    = init_monitor(config)                      # fit detectors, load memory
report           = check_snapshot(monitor_param, Q, meta)    # scores, flags, failure mode
Q_fixed, report  = correct_snapshot(monitor_param, Q, report)
```

Stateful items (score history for temporal checks, fitted scikit-learn objects, the snapshot memory) live inside `monitor_param`.

## Package layout

```
src/simwatchdog/
  io/        snapshot_read_func.py, input_read_func.py
  detect/    physics_func.py, ood_func.py, residual_func.py,
             temporal_func.py, innovation_func.py
  correct/   projection_func.py, memory_func.py, assimilation_func.py
  monitor/   monitor_func.py
  testbed/   opinf_model.py, pod_regression_model.py, cnn_model.py,
             scoring_func.py
benchmarks/  sod_shock_tube/, colliding_shocks/, free_flame/
tests/
pyproject.toml
```

Input files are TOML. They select the case, the surrogate and its failure knobs, the detectors and thresholds, and the correction method.

## Dependencies

| Group | Packages |
| --- | --- |
| Core | numpy, scipy, scikit-learn |
| `[testbed]` | opinf, torch, compflowlab, matplotlib |
| `[dev]` | pytest, ruff |

Python 3.11 or later (needed for the standard-library TOML reader and already implied by CompFlowLab's numpy pin).

## Detection

Each detector is a function returning a scalar score and a flag. Detectors are listed by name in the input file.

| Family | Checks | Needs FOM at runtime? |
| --- | --- | --- |
| Physics | NaN/inf, positivity of density/pressure/temperature, mass fractions in [0,1] and summing to 1, mass/momentum/energy drift, entropy, total-variation growth | No |
| Out-of-distribution | POD projection error, Mahalanobis distance in latent space, IsolationForest / LOF | No (fitted on training window) |
| Residual | Governing-equation residual of the surrogate state via the solver callback | No |
| Temporal | CUSUM / change-point on any score above | No |
| Innovation | Mismatch between surrogate state and observations at `obs_indices` | Yes |

The first four families are reference-free and are scored separately from the innovation family, so the results stay meaningful for the case where no observations exist.

## Testbed surrogates

All three train on `[0, t_train]` of one FOM trajectory and roll out to the end, so they fail by time extrapolation.

| Category | Method | Tool | Expected failure |
| --- | --- | --- | --- |
| ROM | Operator inference | `opinf` | Oscillations, instability on advection-dominated flow |
| Classical ML | POD coefficients regressed against time (GPR or kernel ridge) | scikit-learn | Extrapolation beyond the training window |
| Neural network | Autoregressive 1D CNN next-step predictor | PyTorch | Rollout drift |

Failure knobs set in the input file: training window, number of POD modes, rollout length, regularisation, training noise.

## Scoring

- True failure onset: first step where surrogate error against FOM exceeds a threshold.
- Per detector: detection delay, false alarms before onset, misses.
- Per correction: error against FOM before and after.

## Correction

| Tier | Method | Tool |
| --- | --- | --- |
| 0 | Signal only, or request solver fallback / rollback | - |
| 1 | Constraint projection: clip, renormalise mass fractions, restore conservation | numpy / scipy |
| 2 | Memory + assimilation: nearest FOM snapshots in POD space as prior/ensemble, EnKF-style analysis step against observations at `obs_indices` | scikit-learn `NearestNeighbors`, numpy |

Correction can itself be a silent failure (pulling a new solution toward stored data), so every correction is logged in the report and scored against FOM.

## Order of work

1. **Skeleton** - `pyproject.toml`, package layout, TOML input reader, CompFlowLab snapshot loader, first tests.
2. **First vertical slice (Sod)** - OpInf surrogate, physics and OOD detectors, scoring.
3. **Other surrogates** - POD + scikit-learn regression, PyTorch CNN.
4. **Remaining detectors** - residual, temporal, innovation.
5. **Correction** - tiers 0 to 2, with before/after scoring.
6. **Widen** - colliding shocks and flame cases, online wrapping inside the CompFlowLab loop, failure-mode taxonomy file, CI and documentation.

## Status (2026-10-04)

Run a benchmark with `python main.py benchmarks/sod_shock_tube/input_file.toml` in the `simwatchdog` conda environment. Results (arrays, `summary.json`, `summary.png`) go to `<method>_results/` next to the input file.

| Step | State |
| --- | --- |
| 1. Skeleton | Done |
| 2. First vertical slice (Sod, OpInf) | Done |
| 3. Other surrogates | POD regression done. CNN drafted but parked: untuned, slow to train on CPU |
| 4. Remaining detectors | Done: temporal (CUSUM), innovation, and residual with a built-in Rusanov flux plus a hook for a solver's own flux divergence |
| 5. Correction | Projection and memory + ensemble Kalman update done, applied snapshot by snapshot. Restarting the surrogate from the corrected state not started |
| 6. Widen | Not started |

Findings on Sod so far (training on the first 40 % of the run):

- OpInf and POD regression both fail silently within 15 to 25 saved steps after the training window: finite, smooth, wrong.
- The conservation check is the first reference-free detector to fire, at or just before the true onset. Positivity, total variation and Mahalanobis fire 160+ steps late.
- The residual check catches POD regression one step after the onset with no false alarms, but is 156 steps late on OpInf: the 10-mode reconstruction of the shock is oscillatory, so its residual is already high inside the training window. The built-in flux divergence agrees with CompFlowLab's saved residual to 2 to 3 %.
- `pod_projection` cannot see a POD-based surrogate, whose output lies in the training basis by construction.
- Thresholds have to be calibrated on the surrogate's own output over the training window, otherwise the conservation and total-variation checks flag its normal truncation error. The same calibration hides a surrogate that is already wrong inside the training window (seen with the untuned CNN).
- Memory correction with 20 observed cells brings the error from 0.36 to 0.0015 when the library covers the whole run. With a training-window-only library it does not help, and projection alone does not help either.

## Needed from Ali

- ~~Sod FOM results~~ Received, in `data/sod_shock_tube/FOM_results/`.
- Later: FOM results for colliding shocks and a 1D flame, in the same gathered format. Sod only for now.

## Open

- ~~Where FOM data lives~~ Decided: out of git; the path is given in the input file.
- ~~Error threshold for "true failure onset"~~ Decided: set in the `[scoring]` section of the input file (norm, variable, value); defaults are relative L1, worst variable, 5%. The full error curve is saved so the threshold can be tuned per case.
- Later extensions: sparse observations, parameter sweeps, neural operators (FNO), automated input generation.
