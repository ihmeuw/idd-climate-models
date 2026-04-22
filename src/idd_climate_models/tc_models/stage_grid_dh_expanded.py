"""
stage_grid_dh_expanded.py

Expanded covariate grid for DH model selection.

Based on lessons learned from initial sweep:
- Only 70% threshold (best performing)
- No poisson/tweedie (poor performance)
- Expanded covariate combinations (especially SDI variations)
- No ML models

Run with: python -m idd_climate_models.tc_models.stage_grid_dh_expanded
"""

import json
from pathlib import Path

# ----------------------------------------------------------------------------
# COVARIATE SETS - Full 2^4 factorial (no log transforms)
# ----------------------------------------------------------------------------
# Features: wind, sdi, basin, island
# 16 combinations from presence/absence of each

COVAR_SETS = [
    # 0 features (intercept only)
    'none',
    
    # 1 feature
    'wind',
    'sdi',
    'basin',
    'island',
    
    # 2 features
    'wind_sdi',
    'wind_basin',
    'wind_island',
    'sdi_basin',
    'sdi_island',
    'basin_island',
    
    # 3 features
    'wind_sdi_basin',
    'wind_sdi_island',
    'wind_basin_island',
    'sdi_basin_island',
    
    # 4 features (full model)
    'wind_sdi_basin_island',
]

# Only 70% threshold - identified as optimal
THRESHOLDS = [70]

# Distributions - exclude poisson and tweedie (poor performers)
S1_DISTS = ['statsmodels_logistic']
S2_DISTS = ['statsmodels_logistic']
BULK_DISTS = ['statsmodels_nb', 'statsmodels_gamma', 'statsmodels_lognormal']
TAIL_DISTS = ['scipy_gpd', 'statsmodels_nb', 'statsmodels_gamma', 'statsmodels_lognormal']


def build_stage_grid():
    """Build the expanded DH stage grid."""
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


def count_dh_models(stages):
    """
    Count total DH model combinations.
    
    DH model = s1 × s2 × bulk × tail
    For fixed threshold: s1 unique, s2/bulk/tail per-threshold
    """
    s1_stages = [s for s in stages if s['stage_type'] == 's1']
    s2_stages = [s for s in stages if s['stage_type'] == 'pos_binary' and s['threshold_pct'] == 70]
    bulk_stages = [s for s in stages if s['stage_type'] == 'dh_bulk' and s['threshold_pct'] == 70]
    tail_stages = [s for s in stages if s['stage_type'] == 'tail' and s['threshold_pct'] == 70]
    
    # Cartesian product
    n_models = len(s1_stages) * len(s2_stages) * len(bulk_stages) * len(tail_stages)
    return n_models


def main():
    stages = build_stage_grid()
    print(f"Total stages: {len(stages)}")
    print(f"Covariate sets: {len(COVAR_SETS)}")
    print(f"Thresholds: {THRESHOLDS}")
    
    # Count by type
    by_type = {}
    for s in stages:
        t = s['stage_type']
        by_type[t] = by_type.get(t, 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")
    
    # Count DH models
    n_models = count_dh_models(stages)
    print(f"\nTotal DH model combinations: {n_models:,}")
    print(f"  = {len(COVAR_SETS)} s1 × {len(COVAR_SETS)} s2 × {len(BULK_DISTS)*len(COVAR_SETS)} bulk × {len(TAIL_DISTS)*len(COVAR_SETS)} tail")

    # Write to file
    out_dir = Path('/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/stage_results_dh_expanded')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'stage_grid.json'
    with open(out_path, 'w') as f:
        json.dump(stages, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == '__main__':
    main()
