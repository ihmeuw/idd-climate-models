"""
stage_grid_dh_only.py

Minimal grid: only non-ML DH stages with 3 covariate sets.
"""

import json
from pathlib import Path

# Covariate sets
COVAR_SETS = ['none', 'wind_sdi', 'wind_sdi_basin_island']

# Thresholds
THRESHOLDS = [70, 75, 80, 85, 90, 95]

# Distributions per stage type (non-ML only)
S1_DISTS = ['statsmodels_logistic']
S2_DISTS = ['statsmodels_logistic']
BULK_DISTS = ['statsmodels_nb', 'statsmodels_gamma', 'statsmodels_lognormal',
              'statsmodels_poisson', 'statsmodels_tweedie']
TAIL_DISTS = ['scipy_gpd', 'statsmodels_nb', 'statsmodels_gamma',
              'statsmodels_lognormal', 'statsmodels_poisson']


def build_stage_grid():
    """Build the minimal DH stage grid."""
    stages = []

    # s1: binary, all rows, no threshold
    for dist in S1_DISTS:
        for covars in COVAR_SETS:
            stages.append({
                'stage_type': 's1',
                'dist': dist,
                'covars': covars,
            })

    # pos_binary (s2): binary on positives, threshold-dependent
    for dist in S2_DISTS:
        for covars in COVAR_SETS:
            for thr in THRESHOLDS:
                stages.append({
                    'stage_type': 'pos_binary',
                    'dist': dist,
                    'covars': covars,
                    'threshold_pct': thr,
                })

    # dh_bulk: count on bulk positives
    for dist in BULK_DISTS:
        for covars in COVAR_SETS:
            for thr in THRESHOLDS:
                stages.append({
                    'stage_type': 'dh_bulk',
                    'dist': dist,
                    'covars': covars,
                    'threshold_pct': thr,
                })

    # tail: count/EVT on exceedances
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


def main():
    stages = build_stage_grid()
    print(f"Total stages: {len(stages)}")
    
    # Count by type
    by_type = {}
    for s in stages:
        t = s['stage_type']
        by_type[t] = by_type.get(t, 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")

    # Write to file
    out_path = Path('/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/stage_results_dh/stage_grid.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(stages, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == '__main__':
    main()
