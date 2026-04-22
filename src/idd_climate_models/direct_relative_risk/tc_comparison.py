"""
tc_comparison.py  [DEPRECATED — 2026-03-23]

This script implemented the old spec-based comparison (~1,100 model specs via
a monolithic combinatorial grid). It has been superseded by the stage-based
architecture in tc_models/:

    stage_grid.py           # Generate 2011 unique stages
    orchestrate_stages.py   # Submit to cluster
    analyze_stages.py       # Assemble + rank full models
    analyze_dh.py           # DH deep-dive
    notebooks/dh_model_analysis.ipynb  # Primary analysis

Do not run this script for new comparisons. It is retained for reference only
and will be deleted once a final model is selected.

--- Original docstring below ---

Comprehensive model comparison for tropical cyclone mortality prediction.

Uses the tc_models package for evaluation.  The script auto-discovers all
registered model families and builds a full combinatorial grid:
  family × covariate-set × log-transform choice × model-type (hurdle/single)

To add a new model family, register it in tc_models/models/__init__.py and
rerun this script — no other changes needed.
"""

import multiprocessing
import os
import pickle
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.simplefilter('once', category=RuntimeWarning)
warnings.simplefilter('once', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from idd_climate_models.tc_models import (
    load_tc_data, run_multiple_models, run_model_evaluation,
    DEFAULT_SEEDS, DEFAULT_FOLDS,
)
from idd_climate_models.tc_models.models import (
    REGISTRY, SINGLE_ONLY, HURDLE_ONLY, DH_FAMILIES, POT_FAMILIES,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
TEST_MODE = False
VERBOSE = False

SEEDS = [123] if TEST_MODE else list(DEFAULT_SEEDS)
K_FOLDS = 2 if TEST_MODE else DEFAULT_FOLDS

COVARS = (
    ['wind_sdi']
    if TEST_MODE
    else [
        'wind_sdi', 'wind_sdi_basin', 'wind_sdi_island',
        'wind_sdi_basin_island', 'wind_sdi_year',
        'wind_sdi_basin_year', 'wind_sdi_island_year',
        'wind_sdi_basin_island_year',
    ]
)

LOG_COMBOS = [
    {'max_wind_speed': False, 'sdi': False},
    {'max_wind_speed': True,  'sdi': False},
    {'max_wind_speed': False, 'sdi': True},
    {'max_wind_speed': True,  'sdi': True},
]

N_JOBS = int(os.getenv('SLURM_CPUS_PER_TASK', max(1, multiprocessing.cpu_count() - 1)))
SAVE_MODELS = True
OUTPUT_DIR = Path('tc_comparison_output')
MODELS_DIR = OUTPUT_DIR / 'fitted_models'

print(f"Mode: {'TEST' if TEST_MODE else 'FULL'}")
print(f"Seeds: {SEEDS} | Folds: {K_FOLDS}")
print(f"Workers: {N_JOBS}")
print(f"Covariate sets: {COVARS}")

# ============================================================================
# LOAD DATA
# ============================================================================
tc_df = load_tc_data()
nz_tc_df = tc_df[tc_df['death_y_n'] == 1].copy()

print(f"\nData loaded: {len(tc_df):,} observations")
print(f"  Deaths > 0: {tc_df['death_y_n'].sum():,} ({tc_df['death_y_n'].mean()*100:.1f}%)")
print(f"  Year range: {tc_df['data_year'].min()}-{tc_df['data_year'].max()}")

# ============================================================================
# BUILD MODEL SPECS — auto-discovers from REGISTRY
# ============================================================================

# DH/POT configuration
DH_BULK_DISTS = ['nb', 'lognormal', 'gamma', 'poisson']
DH_TAIL_DISTS = ['lognormal', 'gamma', 'gpd', 'poisson']  # nb excluded (convergence issues)
THRESHOLD_PCTS = [80, 90]  # subset for comparison script


def _log_label(log_covs):
    """Generate label for log transform combo."""
    parts = []
    if log_covs['max_wind_speed']:
        parts.append('log(wind)')
    if log_covs['sdi']:
        parts.append('log(sdi)')
    return ' & '.join(parts) if parts else 'no log'


def _build_specs():
    """Generate the full combinatorial grid of model specs."""
    specs = []
    all_families = sorted(REGISTRY.keys())

    for fam, cv, log_covs in product(all_families, COVARS, LOG_COMBOS):
        # Skip island-variant families when covariates don't include island
        if 'island' in fam and 'island' not in cv:
            continue

        # Determine which types this family supports
        if fam in SINGLE_ONLY:
            types = ['single']
        elif fam in HURDLE_ONLY:
            types = ['hurdle']
        else:
            types = ['hurdle', 'single']

        log_str = _log_label(log_covs)

        for mtype in types:
            # DH: expand bulk × tail × threshold
            if fam in DH_FAMILIES:
                for bulk in DH_BULK_DISTS:
                    for tail in DH_TAIL_DISTS:
                        for pct in THRESHOLD_PCTS:
                            specs.append({
                                'type': mtype,
                                'family': fam,
                                'covars': cv,
                                'log_main_covs': log_covs,
                                'bulk_dist': bulk,
                                'tail_dist': tail,
                                'threshold_pct': pct,
                                'label': f"DH {bulk}/{tail}@{pct} ({cv}, {log_str})",
                            })

            # POT: expand threshold
            elif fam in POT_FAMILIES:
                for pct in THRESHOLD_PCTS:
                    specs.append({
                        'type': mtype,
                        'family': fam,
                        'covars': cv,
                        'log_main_covs': log_covs,
                        'threshold_pct': pct,
                        'tail_covars': 'none',
                        'label': f"POT@{pct} ({cv}, {log_str})",
                    })

            # Standard families
            else:
                specs.append({
                    'type': mtype,
                    'family': fam,
                    'covars': cv,
                    'log_main_covs': log_covs,
                    'label': f"{mtype.title()} {fam.upper()} ({cv}, {log_str})",
                })

    return specs


model_specs = _build_specs()
print(f"\nTotal models to evaluate: {len(model_specs)}")

# ============================================================================
# RUN ALL MODELS
# ============================================================================
print("\n" + "=" * 60)
print("RUNNING EVALUATION")
print("=" * 60 + "\n")

multi = run_multiple_models(
    tc_df, model_specs, seeds=SEEDS, k_folds=K_FOLDS, verbose=VERBOSE,
)

oos = multi['summary']

# Collect raw fold-level results for distribution plots
raw_oos = []
for label, result in multi['results'].items():
    df = result['oos_raw'].copy()
    df['label'] = label
    raw_oos.append(df)
raw_oos = pd.concat(raw_oos, ignore_index=True)

# In-sample metrics (already computed by run_multiple_models)
insample_rows = []
for label, result in multi['results'].items():
    row = dict(
        label=label,
        type=result['spec']['type'],
        family=result['spec']['family'],
        covars=result['spec']['covars'],
    )
    row.update(result['insample'])
    insample_rows.append(row)
insample = pd.DataFrame(insample_rows)

# ============================================================================
# ANALYSIS & PLOTTING
# ============================================================================
print("\n" + "=" * 60)
print("GENERATING REPORT")
print("=" * 60 + "\n")

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

OUTPUT_DIR.mkdir(exist_ok=True)

# --- Data overview ---
print("Dataset Overview:")
overview = {
    'Total Observations': len(tc_df),
    'Deaths > 0': tc_df['death_y_n'].sum(),
    'Zero Death Proportion': f"{(1 - tc_df['death_y_n'].mean())*100:.1f}%",
    'Mean Deaths (All)': f"{tc_df['total_deaths'].mean():.1f}",
    'Mean Deaths (Non-Zero)': f"{nz_tc_df['total_deaths'].mean():.1f}",
    'Mean Death Rate per 100k (Non-Zero)': f"{nz_tc_df['death_rate'].mean()*1e5:.2f}",
    'Events >10,000 deaths': (tc_df['total_deaths'] > 10000).sum(),
    'Basins': tc_df['basin'].nunique(),
    'Year Range': f"{tc_df['data_year'].min()}-{tc_df['data_year'].max()}"
}
for k, v in overview.items():
    print(f"  {k}: {v}")

# --- In-sample results ---
print("\n--- In-Sample: Top 15 by Rate MAE ---")
top15_is_rate = insample.nsmallest(15, 'mae_rate')[
    ['label', 'type', 'covars', 'mae_rate', 'rmse_rate', 'mae_log', 'cor_rate']
]
print(top15_is_rate.to_string(index=False))

print("\n--- In-Sample: Top 15 by Count MAE ---")
top15_is_count = insample.nsmallest(15, 'mae_count')[
    ['label', 'type', 'covars', 'mae_count', 'rmse_count', 'cor_count', 'zero_acc']
]
print(top15_is_count.to_string(index=False))

# In-sample rank plot
fig, ax = plt.subplots(figsize=(9, 5))
insample['rank_rate'] = insample['mae_rate'].rank()
insample['rank_count'] = insample['mae_count'].rank()
for mtype in insample['type'].unique():
    subset = insample[insample['type'] == mtype]
    ax.scatter(subset['rank_rate'], subset['rank_count'],
               label=mtype, alpha=0.7, s=50)
ax.plot([0, max(insample['rank_rate'])], [0, max(insample['rank_count'])],
        'k--', alpha=0.5, label='diagonal')
ax.set_xlabel('Rank by Rate MAE')
ax.set_ylabel('Rank by Count MAE')
ax.set_title('In-Sample: Rate Rank vs Count Rank')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'insample_rank_agreement.png')
plt.close()

# --- Out-of-sample results ---
print("\n--- OOS: Top 15 by Rate MAE ---")
top15_oos_rate = oos.nsmallest(15, 'mae_rate')[
    ['label', 'type', 'covars', 'mae_rate', 'sd_mae_rate',
     'rmse_rate', 'mae_log', 'cor_rate']
]
print(top15_oos_rate.to_string(index=False))

print("\n--- OOS: Top 15 by Count MAE ---")
top15_oos_count = oos.nsmallest(15, 'mae_count')[
    ['label', 'type', 'covars', 'mae_count', 'sd_mae_count',
     'rmse_count', 'cor_count', 'zero_acc']
]
print(top15_oos_count.to_string(index=False))

# Covariate effect bar plot
n_families = oos['family'].nunique()
ncols = min(4, n_families)
nrows = (n_families + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
axes = np.array(axes).flatten()
covar_effect = oos.groupby(['family', 'covars'])['mae_rate'].mean().reset_index()
for i, fam in enumerate(sorted(covar_effect['family'].unique())):
    if i >= len(axes):
        break
    subset = covar_effect[covar_effect['family'] == fam]
    axes[i].bar(subset['covars'], subset['mae_rate'], alpha=0.8)
    axes[i].set_title(f"{fam}")
    axes[i].set_xlabel('Covariates')
    axes[i].set_ylabel('Mean MAE Rate')
    axes[i].tick_params(axis='x', rotation=45)
for j in range(i + 1, len(axes)):
    axes[j].axis('off')
plt.suptitle('Covariate Effect on Rate MAE by Family', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'covariate_effect.png')
plt.close()

# Rate MAE heatmap
heatmap_data = oos.groupby(['family', 'covars'])['mae_rate'].mean().unstack()
fig, ax = plt.subplots(figsize=(max(10, len(COVARS) * 1.2), max(6, len(heatmap_data) * 0.5)))
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax,
            cbar_kws={'label': 'MAE Rate (per 100k)'})
ax.set_title('Rate MAE: Family x Covariate')
ax.set_xlabel('Covariates')
ax.set_ylabel('Family')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'rate_heatmap.png')
plt.close()

# Top 10 boxplot
top10_labels = oos.nsmallest(10, 'mae_rate')['label'].tolist()
top10_data = raw_oos[raw_oos['label'].isin(top10_labels)].copy()
fig, ax = plt.subplots(figsize=(12, 6))
top10_data['label'] = pd.Categorical(
    top10_data['label'], categories=top10_labels[::-1], ordered=True
)
sns.boxplot(data=top10_data, y='label', x='mae_rate', ax=ax, palette='viridis')
ax.set_xlabel('MAE Rate (per 100k)')
ax.set_ylabel(None)
ax.set_title('Top 10 Models: Rate MAE Distribution')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'top10_boxplot.png')
plt.close()

# Seed stability
seed_stability = (
    raw_oos[raw_oos['label'].isin(top10_labels)]
    .groupby(['seed', 'label'])['mae_rate'].mean()
    .reset_index()
)
seed_labels = sorted(seed_stability['seed'].unique())
seed_positions = {s: i for i, s in enumerate(seed_labels)}
seed_stability['seed_pos'] = seed_stability['seed'].map(seed_positions)

fig, ax = plt.subplots(figsize=(10, 6))
for label in top10_labels:
    subset = seed_stability[seed_stability['label'] == label]
    ax.plot(subset['seed_pos'], subset['mae_rate'], marker='o', label=label, linewidth=2)
ax.set_xticks(range(len(seed_labels)))
ax.set_xticklabels([str(s) for s in seed_labels])
ax.set_xlabel('Seed')
ax.set_ylabel('Mean MAE Rate (per 100k)')
ax.set_title('Rate MAE Stability Across Seeds')
ax.legend(fontsize=7, loc='best')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'seed_stability.png')
plt.close()

# OOS rank agreement
fig, ax = plt.subplots(figsize=(9, 5))
oos['rank_rate'] = oos['mae_rate'].rank()
oos['rank_count'] = oos['mae_count'].rank()
for fam in sorted(oos['family'].unique()):
    subset = oos[oos['family'] == fam]
    ax.scatter(subset['rank_rate'], subset['rank_count'],
               label=fam, alpha=0.7, s=50)
ax.plot([0, max(oos['rank_rate'])], [0, max(oos['rank_count'])], 'k--', alpha=0.5)
ax.set_xlabel('Rank by Rate MAE')
ax.set_ylabel('Rank by Count MAE')
ax.set_title('OOS: Rate Rank vs Count Rank')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'oos_rank_agreement.png')
plt.close()

# In-sample vs OOS agreement
combined = insample[['label', 'mae_rate', 'mae_count']].rename(
    columns={'mae_rate': 'is_mae_rate', 'mae_count': 'is_mae_count'}
).merge(
    oos[['label', 'mae_rate', 'mae_count']].rename(
        columns={'mae_rate': 'oos_mae_rate', 'mae_count': 'oos_mae_count'}
    ), on='label'
)
combined['is_rank_rate'] = combined['is_mae_rate'].rank()
combined['oos_rank_rate'] = combined['oos_mae_rate'].rank()
combined['is_rank_count'] = combined['is_mae_count'].rank()
combined['oos_rank_count'] = combined['oos_mae_count'].rank()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(combined['is_rank_rate'], combined['oos_rank_rate'], alpha=0.6, s=30, c='steelblue')
ax1.plot([0, max(combined['is_rank_rate'])], [0, max(combined['oos_rank_rate'])], 'k--', alpha=0.5)
z = np.polyfit(combined['is_rank_rate'], combined['oos_rank_rate'], 1)
p = np.poly1d(z)
ax1.plot(combined['is_rank_rate'], p(combined['is_rank_rate']), 'r-', linewidth=2)
ax1.set_xlabel('In-Sample Rank')
ax1.set_ylabel('OOS Rank')
ax1.set_title('Rate MAE: In-Sample vs OOS Rank')

ax2.scatter(combined['is_rank_count'], combined['oos_rank_count'], alpha=0.6, s=30, c='coral')
ax2.plot([0, max(combined['is_rank_count'])], [0, max(combined['oos_rank_count'])], 'k--', alpha=0.5)
z = np.polyfit(combined['is_rank_count'], combined['oos_rank_count'], 1)
p = np.poly1d(z)
ax2.plot(combined['is_rank_count'], p(combined['is_rank_count']), 'r-', linewidth=2)
ax2.set_xlabel('In-Sample Rank')
ax2.set_ylabel('OOS Rank')
ax2.set_title('Count MAE: In-Sample vs OOS Rank')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'is_vs_oos_agreement.png')
plt.close()

# Overfitting plot
top20_oos = combined.nsmallest(20, 'oos_rank_rate')
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(top20_oos))
width = 0.35
ax.barh(x - width / 2, top20_oos['is_mae_rate'], width, label='In-Sample', alpha=0.8)
ax.barh(x + width / 2, top20_oos['oos_mae_rate'], width, label='Out-of-Sample', alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels(top20_oos['label'], fontsize=8)
ax.set_xlabel('MAE Rate (per 100k)')
ax.set_title('Top 20 OOS Models: Overfitting Gap (Rate MAE)')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'overfitting_gap.png')
plt.close()

# --- Conclusions ---
best_rate = oos.nsmallest(1, 'mae_rate').iloc[0]
best_count = oos.nsmallest(1, 'mae_count').iloc[0]
rank_cor = np.corrcoef(combined['is_rank_rate'], combined['oos_rank_rate'])[0, 1]
best_covars = oos.groupby('covars')['mae_rate'].mean().idxmin()
best_family = oos.groupby('family')['mae_rate'].mean().idxmin()
rate_count_cor = np.corrcoef(oos['mae_rate'], oos['mae_count'])[0, 1]

print("\n" + "=" * 60)
print("CONCLUSIONS")
print("=" * 60)
print(f"\nBest model by Rate MAE: {best_rate['label']}")
print(f"  Family: {best_rate['family']}, Covars: {best_rate['covars']}")
print(f"  MAE Rate: {best_rate['mae_rate']:.3f}")

print(f"\nBest model by Count MAE: {best_count['label']}")
print(f"  Family: {best_count['family']}, Covars: {best_count['covars']}")
print(f"  MAE Count: {best_count['mae_count']:.1f}")

print(f"\nBest covariate set: {best_covars}")
print(f"Best model family: {best_family}")
print(f"  {'ML outperforms parametric' if best_family in ('rf', 'xgb') else 'Parametric outperforms ML'}")

print(f"\nSingle vs hurdle: {best_rate['type']} wins by rate MAE")

print(f"\nRate/count agreement (correlation = {rate_count_cor:.2f}):")
if rate_count_cor > 0.7:
    print("  Metrics largely agree.")
else:
    print("  Metrics disagree — metric choice matters.")

print(f"\nIn-sample reliability (rank correlation = {rank_cor:.2f}):")
if rank_cor > 0.7:
    print("  In-sample ranking is a reasonable OOS proxy.")
else:
    print("  In-sample ranking is a poor OOS proxy — CV is essential.")

print(
    f"\n{'WARNING: TEST MODE results only.' if TEST_MODE else f'{K_FOLDS}-fold CV x {len(SEEDS)} seeds = {K_FOLDS * len(SEEDS)} test sets per model.'}"
)

print(f"\nAll plots saved to: {OUTPUT_DIR.absolute()}")

# ============================================================================
# SAVE RESULTS AND MODELS
# ============================================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

oos.to_csv(OUTPUT_DIR / 'oos_summary.csv', index=False)
print("Saved: oos_summary.csv")

insample.to_csv(OUTPUT_DIR / 'insample_summary.csv', index=False)
print("Saved: insample_summary.csv")

raw_oos.to_csv(OUTPUT_DIR / 'oos_raw_all_folds.csv', index=False)
print("Saved: oos_raw_all_folds.csv")

if SAVE_MODELS:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving {len(multi['results'])} fitted models...")
    for label, result in multi['results'].items():
        safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
        filepath = MODELS_DIR / f"{safe_label}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
    total_mb = sum(f.stat().st_size for f in MODELS_DIR.glob('*.pkl')) / 1024 / 1024
    print(f"Saved: {len(multi['results'])} model files to {MODELS_DIR}/ (~{total_mb:.1f} MB)")
else:
    print("\nModel saving disabled (set SAVE_MODELS=True to enable)")

print("\nDone!")

