# Session memory
Updated: 2026-09-23 17:02

## Current task
Session closed with /wrap. The read-only audit of central-pressure estimation for TC Risk tracks is complete; no pipeline code changed. April's uncommitted work was committed at the wrap.

## Context / why
A slide described p_c as a 0.6/0.4 weighted average of CLIMADA's IBTrACS fit and a "Knaff & Zehr 2007 Dvorak wind-pressure relationship, also implemented within CLIMADA". Bobby wanted to know where the weights came from and whether the description is right. It is not; details below.

## Where we are
Findings, all from code, git, VS Code local history, and the KZ07 paper text:
- Weights 0.6/0.4 (RMW 0.4/0.3/0.3, ROCI 0.5/0.5) are hardcoded in `ensemble_estimates()`; never fitted; no rationale anywhere. First appear in untracked scripts in `~/repos/climada_python/notebooks` (2025-10-28), committed here 2025-11-19 (51b6543). No earlier version exists anywhere reachable (see DEAD_ENDS 2026-09-23).
- Pipeline always uses the blend: `tc_risk_functions.py` builds a Namespace with `use_ensemble=True` (lines ~530, ~562); no CLI flag; the Jan–Feb 2026 `--use_ensemble` flag used argparse `type=bool` so it could not be turned off.
- `estimate_pressure_dvorak()` matches neither Dvorak basin table nor Knaff & Zehr 2007. Against the Dvorak Atlantic column it is 7–34 hPa too deep from 65 kt up; against the NW Pacific column (which is Atkinson & Holliday 1977 rounded) it is 1–9 hPa deeper from 77–155 kt. KZ07 Eq. 7 (from the paper): MSLP = 23.286 − 0.483 Vsrm − (Vsrm/24.254)² − 12.587 S − 0.483 lat + Penv; Vsrm = Vmax − 1.5 c^0.63; S = V500/V500c. With S=1, c=10 kt, KZ07 is within −13..+5 hPa of CLIMADA.
- CLIMADA (clone v6.1.0, full git history) has no Knaff/Zehr/Dvorak code; only `_estimate_pressure`/`_estimate_vmax` linear IBTrACS fits (1980–2020, r² 0.8727) plus `estimate_rmw`/`estimate_roci`. CLIMADA citation: Aznar-Siguan & Bresch 2019, GMD 12, 3085–3097.
- Ensemble RMW/ROCI never reach CLIMADA: the CLIMADA input file has no radius fields; the 99_climada runner zero-fills `radius_max_wind`/`radius_oci`; CLIMADA `tctrack_to_si` re-estimates RMW from the (blended) central pressure. ROCI is unused by the windfield.
- Hazard stage runs default H08, where peak wind derives from pressure deficit, so the blend shifts peak wind vs CLIMADA-only: −21% at 35 kt, ~0 at 64 kt, +5..+8% at 85–160 kt (Atlantic); crossover ~64 kt NA, ~80 kt WP.
- `estimate_roci_knaff()` misuses KZ07 Eq. 6 (Rmax in km, with lat−25) as ROCI in nm.
- TC Risk runs BayesPoisson bins chunked to ≤5 years (`MAX_PERIOD_DURATION = 5`; file `bayespoisson_time_bins_max_bin_5.csv`). BayesPoisson = Poisson MLE with synchronized breaks across basins, 10–30-yr bins, BIC selection; not Bayesian in the prior/posterior sense.
- `detect_time_bins.ipynb`: 1965–69 rows are zero-fill for 7 of 8 runs (MPI-ESM1-2-HR has real counts); those years enter the scaling multiplier, so changing `YEAR_RANGES` moves multipliers up to ±12%. The reload `combined_df = pd.read_csv(output_path)` needs `keep_default_na=False` or basin "NA" becomes NaN. Bin lines in `plot_basin_model_timeseries` are silently skipped when `bins_df` is not in the kernel.
- KZ07 PDF: `~/.claude/projects/-mnt-share-homes-bcreiner-repos-idd-climate-models/6061bfcd-0c0b-456b-a507-8323a903a5b5/tool-results/knaffzehr2007.pdf` (text extractable via zlib in the venv; no pdftotext on the node).

## Next steps
1. Bobby decides the central-pressure estimator (keep blend / CLIMADA-only / KZ07 with S=1 / Dvorak by basin) and corrects the slide text.
2. Re-run `detect_time_bins.ipynb` in order with the `keep_default_na=False` reload; rebuild `bins_df` before the bin plots.
3. Delete `notebooks/visualizations/rotate_shapefiles.py`.
4. TOPSIS on `dh_exhaustive_expanded.csv`; final model; uncertainty (carried from April).

## Commit groups
- A `src/idd_climate_models/storm_data/visualizers.py` — committed c818d54 (9/9 tests pass).
- B `pyproject.toml`, `uv.lock` — committed e5aeb03.
- C `.claude/STATUS.md`, `.claude/DEAD_ENDS.md`, `.claude/memory.md` — committed at wrap close as "Update session tracking files" (hash in `git log`); `no_zarr` then pushed to origin over SSH.
- `stash@{0}` (Feb 2026 WIP on no_zarr) untouched.

## Resume prompt
We audited the TC central-pressure estimator: the 0.6·CLIMADA + 0.4·"Dvorak" blend in tc_estimation.py is unfitted and always on in the pipeline; the "Dvorak / Knaff & Zehr" curve matches no published relationship (KZ07 Eq. 7 verified from the paper; the Dvorak NW Pacific column is Atkinson-Holliday 1977); CLIMADA has no KZ code; ensemble RMW/ROCI never reach CLIMADA, which re-derives RMW from the blended pressure; under H08 the blend raises hurricane peak winds 5–8% and lowers sub-hurricane winds. Nothing in the pipeline was changed; the estimator choice is Bobby's. April's plot_all_tracks work and the pyarrow dependency were committed and pushed at the wrap. See "Where we are" for the numbers.
