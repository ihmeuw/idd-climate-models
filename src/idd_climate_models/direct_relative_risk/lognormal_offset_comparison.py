"""
lognormal_offset_comparison.py
Minimal example comparing offset handling across model families.

Goal: Verify that the lognormal model handles the exposure offset correctly
by comparing it to a Poisson GLM (which uses statsmodels' built-in offset).

Models compared:
1. Poisson GLM (statsmodels) - gold standard for offset handling
2. Lognormal (sklearn WLS)   - manual offset subtraction
3. Lognormal_sm (statsmodels WLS) - manual offset subtraction with proper inference

Key questions:
- Do all three models produce similar predictions?
- Are the coefficient estimates on the same scale?
- Is the Jensen correction (sigma²/2) being applied correctly?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure we can import from the package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from idd_climate_models.tc_models import load_tc_data
from idd_climate_models.tc_models.models import glm, lognormal
from idd_climate_models.tc_models.features import build_X
from idd_climate_models.tc_models.models.base import sm_const, safe_log_exp


def main():
    print("=" * 80)
    print("LOGNORMAL OFFSET COMPARISON")
    print("=" * 80)

    # ── Load data ──────────────────────────────────────────────────────────────
    tc_df = load_tc_data()
    print(f"\nData: {len(tc_df)} storms, {(tc_df['total_deaths'] > 0).sum()} with deaths")

    # ── Filter to positive deaths only (for S2 comparison) ─────────────────────
    pos = tc_df[tc_df['total_deaths'] > 0].copy()
    print(f"Positive deaths: {len(pos)} storms")

    # ── Build features ─────────────────────────────────────────────────────────
    covars = 'wind_sdi'
    log_main_covs = {'max_wind_speed': False, 'sdi': False}

    X = build_X(pos, covars=covars, log_main_covs=log_main_covs)
    y = pos['total_deaths'].values
    exposure = pos['exposed_population'].values

    print(f"\nCovariates: {list(X.columns)}")
    print(f"X shape: {X.shape}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIT ALL THREE MODELS
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("MODEL FITS (Stage 2 only - count model on positive deaths)")
    print("=" * 80)

    # 1. Poisson GLM (statsmodels) ─────────────────────────────────────────────
    print("\n── 1. Poisson GLM (statsmodels) ──")
    poisson_model = glm.fit_count_sm(X, y, exposure)
    poisson_pred = glm.predict_count_sm(poisson_model, X, exposure)

    print(f"Coefficients:")
    for name, coef, se in zip(
        ['const'] + list(X.columns),
        poisson_model.params,
        poisson_model.bse
    ):
        print(f"  {name:20s}: {coef:10.4f} (SE: {se:.4f})")

    print(f"\nTotal predicted: {poisson_pred.sum():.0f}")
    print(f"Total observed:  {y.sum():.0f}")
    print(f"Ratio:           {poisson_pred.sum() / y.sum():.4f}")

    # 2. Lognormal (sklearn WLS) ───────────────────────────────────────────────
    print("\n── 2. Lognormal (sklearn WLS) ──")
    ln_model = lognormal.fit_count(X, y, exposure)
    ln_pred = lognormal.predict_count(ln_model, X, exposure)

    print(f"Coefficients:")
    print(f"  {'const':20s}: {ln_model.intercept_:10.4f}")
    for name, coef in zip(X.columns, ln_model.coef_):
        print(f"  {name:20s}: {coef:10.4f}")
    print(f"  sigma²:             {ln_model.sigma2:.4f}")
    print(f"  Jensen correction:  exp(σ²/2) = {np.exp(ln_model.sigma2 / 2):.4f}")

    print(f"\nTotal predicted: {ln_pred.sum():.0f}")
    print(f"Total observed:  {y.sum():.0f}")
    print(f"Ratio:           {ln_pred.sum() / y.sum():.4f}")

    # 3. Lognormal_sm (statsmodels WLS) ────────────────────────────────────────
    print("\n── 3. Lognormal_sm (statsmodels WLS) ──")
    ln_sm_model = lognormal.fit_count_sm(X, y, exposure)
    ln_sm_pred = lognormal.predict_count_sm(ln_sm_model, X, exposure)

    print(f"Coefficients:")
    for name, coef, se in zip(
        ['const'] + list(X.columns),
        ln_sm_model.params,
        ln_sm_model.bse
    ):
        print(f"  {name:20s}: {coef:10.4f} (SE: {se:.4f})")
    print(f"  sigma²:             {ln_sm_model.sigma2:.4f}")
    print(f"  Jensen correction:  exp(σ²/2) = {np.exp(ln_sm_model.sigma2 / 2):.4f}")
    print(f"  R²:                 {ln_sm_model.rsquared:.4f}")

    print(f"\nTotal predicted: {ln_sm_pred.sum():.0f}")
    print(f"Total observed:  {y.sum():.0f}")
    print(f"Ratio:           {ln_sm_pred.sum() / y.sum():.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARE PREDICTIONS
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("PREDICTION COMPARISON")
    print("=" * 80)

    # Correlation between predictions
    corr_pois_ln = np.corrcoef(poisson_pred, ln_pred)[0, 1]
    corr_pois_lnsm = np.corrcoef(poisson_pred, ln_sm_pred)[0, 1]
    corr_ln_lnsm = np.corrcoef(ln_pred, ln_sm_pred)[0, 1]

    print(f"\nCorrelation of predictions:")
    print(f"  Poisson vs Lognormal:      {corr_pois_ln:.6f}")
    print(f"  Poisson vs Lognormal_sm:   {corr_pois_lnsm:.6f}")
    print(f"  Lognormal vs Lognormal_sm: {corr_ln_lnsm:.6f}")

    # MAE vs observed
    mae_pois = np.mean(np.abs(y - poisson_pred))
    mae_ln = np.mean(np.abs(y - ln_pred))
    mae_lnsm = np.mean(np.abs(y - ln_sm_pred))

    print(f"\nMAE (vs observed):")
    print(f"  Poisson:      {mae_pois:.2f}")
    print(f"  Lognormal:    {mae_ln:.2f}")
    print(f"  Lognormal_sm: {mae_lnsm:.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # OFFSET VERIFICATION
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("OFFSET VERIFICATION")
    print("=" * 80)

    # For Poisson: E[Y] = exposure * exp(Xβ)
    # For Lognormal: E[Y] = exposure * exp(Xβ + σ²/2)
    #
    # If we normalize by exposure, we should get rate predictions

    rate_obs = y / exposure
    rate_pois = poisson_pred / exposure
    rate_ln = ln_pred / exposure
    rate_lnsm = ln_sm_pred / exposure

    print(f"\nRate predictions (per person):")
    print(f"  Observed mean rate:      {rate_obs.mean():.6e}")
    print(f"  Poisson mean rate:       {rate_pois.mean():.6e}")
    print(f"  Lognormal mean rate:     {rate_ln.mean():.6e}")
    print(f"  Lognormal_sm mean rate:  {rate_lnsm.mean():.6e}")

    # Check if exposure scales predictions correctly
    # If offset is correct: pred ∝ exposure (holding X constant)
    print(f"\nExposure scaling check (correlation of pred with exposure):")
    print(f"  Poisson:      {np.corrcoef(poisson_pred, exposure)[0,1]:.4f}")
    print(f"  Lognormal:    {np.corrcoef(ln_pred, exposure)[0,1]:.4f}")
    print(f"  Lognormal_sm: {np.corrcoef(ln_sm_pred, exposure)[0,1]:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # WHAT LOGNORMAL IS ACTUALLY FITTING
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("WHAT LOGNORMAL IS FITTING")
    print("=" * 80)

    log_rate = np.log(y) - safe_log_exp(exposure)
    print(f"\nTarget variable: log(y) - log(exposure) = log(rate)")
    print(f"  Mean log(rate):   {log_rate.mean():.4f}")
    print(f"  Std log(rate):    {log_rate.std():.4f}")
    print(f"  exp(mean):        {np.exp(log_rate.mean()):.6e}")

    # What does the intercept represent?
    print(f"\nIntercept interpretation:")
    print(f"  Lognormal intercept:    {ln_model.intercept_:.4f}")
    print(f"  Lognormal_sm intercept: {ln_sm_model.params[0]:.4f}")
    print(f"  Poisson intercept:      {poisson_model.params[0]:.4f}")
    print(f"\n  If covariates are centered, intercept ≈ mean log(rate) for lognormal")
    print(f"  For Poisson, intercept = log(baseline rate)")

    # ══════════════════════════════════════════════════════════════════════════
    # VISUAL COMPARISON
    # ══════════════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Predicted vs Observed
    ax = axes[0, 0]
    ax.scatter(y, poisson_pred, alpha=0.3, s=20)
    max_val = max(y.max(), poisson_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title('Poisson GLM')
    ax.set_xlim(0, np.percentile(y, 99))
    ax.set_ylim(0, np.percentile(poisson_pred, 99))

    ax = axes[0, 1]
    ax.scatter(y, ln_pred, alpha=0.3, s=20)
    ax.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title('Lognormal (sklearn)')
    ax.set_xlim(0, np.percentile(y, 99))
    ax.set_ylim(0, np.percentile(ln_pred, 99))

    ax = axes[0, 2]
    ax.scatter(y, ln_sm_pred, alpha=0.3, s=20)
    ax.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title('Lognormal_sm (statsmodels)')
    ax.set_xlim(0, np.percentile(y, 99))
    ax.set_ylim(0, np.percentile(ln_sm_pred, 99))

    # Row 2: Model comparisons
    ax = axes[1, 0]
    ax.scatter(poisson_pred, ln_pred, alpha=0.3, s=20)
    max_pred = max(poisson_pred.max(), ln_pred.max())
    ax.plot([0, max_pred], [0, max_pred], 'r--')
    ax.set_xlabel('Poisson pred')
    ax.set_ylabel('Lognormal pred')
    ax.set_title(f'Poisson vs Lognormal (r={corr_pois_ln:.4f})')

    ax = axes[1, 1]
    ax.scatter(poisson_pred, ln_sm_pred, alpha=0.3, s=20)
    ax.plot([0, max_pred], [0, max_pred], 'r--')
    ax.set_xlabel('Poisson pred')
    ax.set_ylabel('Lognormal_sm pred')
    ax.set_title(f'Poisson vs Lognormal_sm (r={corr_pois_lnsm:.4f})')

    ax = axes[1, 2]
    # Residuals by exposure
    resid_pois = (y - poisson_pred) / exposure * 1e5
    resid_ln = (y - ln_pred) / exposure * 1e5
    ax.scatter(np.log10(exposure), resid_pois, alpha=0.3, s=20, label='Poisson')
    ax.scatter(np.log10(exposure), resid_ln, alpha=0.3, s=20, label='Lognormal')
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('log10(exposure)')
    ax.set_ylabel('Residual rate (per 100k)')
    ax.set_title('Residuals vs Exposure')
    ax.legend()

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'lognormal_comparison.png', dpi=150)
    print(f"\nPlot saved to: lognormal_comparison.png")
    plt.show()

    # ══════════════════════════════════════════════════════════════════════════
    # STATSMODELS SUMMARY
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("STATSMODELS WLS SUMMARY (Lognormal_sm)")
    print("=" * 80)
    print(ln_sm_model.summary())


if __name__ == '__main__':
    main()
