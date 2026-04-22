"""
stage_grid_dh_v3.py

Stage grid for v3 DH model sweep — rate-based threshold, GLM-only.

Key differences from v2 (stage_grid_dh_expanded.py):
- Threshold is applied to death_rate = total_deaths / exposed_population
- 3 covariate sets instead of full 2^4 factorial (tractability)
- 6 threshold percentiles (70–95) to find the right split point
- GLM-only: NB, gamma, lognormal, Poisson for bulk/tail (ML and GPD removed)
- Poisson added to bulk and tail
- DH model comparisons constrained to same covariate set across all stages
- Predictions constrained in rate space: bulk in (0, u], tail in [u, inf)

Run with:
    python src/idd_climate_models/tc_models/stage_grid_dh_v3.py
"""

import json
from pathlib import Path

# ----------------------------------------------------------------------------
# COVARIATE SETS — 3 representative points in the covariate space
# ----------------------------------------------------------------------------
COVAR_SETS = [
    'none',                  # intercept only
    'wind_sdi',              # continuous covariates only
    'wind_sdi_basin_island', # full model
]

# Threshold percentiles to sweep
THRESHOLDS = [70, 75, 80, 85, 90, 95]

# Distributions
S1_DISTS   = ['statsmodels_logistic']
S2_DISTS   = ['statsmodels_logistic']
BULK_DISTS = ['statsmodels_nb', 'statsmodels_gamma', 'statsmodels_lognormal',
              'statsmodels_poisson']
TAIL_DISTS = ['statsmodels_gamma', 'statsmodels_lognormal', 'statsmodels_poisson']


def build_stage_grid():
    """Build the v3 DH stage grid."""
    stages = []

    # s1: binary on all rows, no threshold
    for dist in S1_DISTS:
        for covars in COVAR_SETS:
            stages.append({
                'stage_type': 's1',
                'dist': dist,
                'covars': covars,
            })

    # pos_binary (s2): binary on positives, rate-based threshold
    for dist in S2_DISTS:
        for covars in COVAR_SETS:
            for thr in THRESHOLDS:
                stages.append({
                    'stage_type': 'pos_binary',
                    'dist': dist,
                    'covars': covars,
                    'threshold_pct': thr,
                })

    # dh_bulk: count on low-rate positives
    for dist in BULK_DISTS:
        for covars in COVAR_SETS:
            for thr in THRESHOLDS:
                stages.append({
                    'stage_type': 'dh_bulk',
                    'dist': dist,
                    'covars': covars,
                    'threshold_pct': thr,
                })

    # tail: count/EVT on high-rate positives
    for dist in TAIL_DISTS:
        for covars in COVAR_SETS:
            for thr in THRESHOLDS:
                stages.append({
                    'stage_type': 'tail',
                    'dist': dist,
                    'covars': covars,
                    'threshold_pct': thr,
                })

    return stages


def count_dh_models(stages):
    """Count same-covariate DH model combinations per threshold."""
    total = 0
    for thr in THRESHOLDS:
        s1    = [s for s in stages if s['stage_type'] == 's1']
        s2    = [s for s in stages if s['stage_type'] == 'pos_binary'   and s['threshold_pct'] == thr]
        bulk  = [s for s in stages if s['stage_type'] == 'dh_bulk'      and s['threshold_pct'] == thr]
        tail  = [s for s in stages if s['stage_type'] == 'tail'         and s['threshold_pct'] == thr]
        # Same-covariate constraint: multiply dist combos × covar sets
        n = (len(S1_DISTS) * len(S2_DISTS) * len(BULK_DISTS) * len(TAIL_DISTS)) * len(COVAR_SETS)
        total += n
    return total


def main():
    stages = build_stage_grid()
    print(f"Total stages: {len(stages)}")
    print(f"Covariate sets: {COVAR_SETS}")
    print(f"Thresholds: {THRESHOLDS}")

    by_type = {}
    for s in stages:
        t = s['stage_type']
        by_type[t] = by_type.get(t, 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")

    n_models = count_dh_models(stages)
    print(f"\nTotal same-covariate DH model combinations: {n_models:,}")
    print(f"  = {len(COVAR_SETS)} covar sets × "
          f"{len(S1_DISTS)*len(S2_DISTS)*len(BULK_DISTS)*len(TAIL_DISTS)} dist combos "
          f"× {len(THRESHOLDS)} thresholds")

    out_dir = Path('/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/stage_results_dh_v3')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'stage_grid.json'
    with open(out_path, 'w') as f:
        json.dump(stages, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == '__main__':
    main()
