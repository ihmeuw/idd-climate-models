"""
stage_plots.py
Vetting plots for Double Hurdle TC mortality model stages.

Usage:
    from idd_climate_models.tc_models.stage_plots import StagePlotter
    plotter = StagePlotter()

    # All vetting plots for one stage
    plotter.vet_stage(topsis_winner, stage='s1')
    plotter.vet_stage(topsis_winner, stage='bulk')

    # All 4 stages at once
    plotter.vet_model(topsis_winner)
"""

import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .data import load_tc_data
from .features import build_X
from .coefficients import get_feature_names_from_covars


DH_MODELS_DIR = Path('/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/stage_results_dh_v2/models')

_LINE_STYLES = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1))]


class StagePlotter:
    """Diagnostic vetting plots for DH model stages."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or DH_MODELS_DIR
        self._data = None
        self._model_cache = {}

    @property
    def data(self):
        if self._data is None:
            self._data = load_tc_data()
        return self._data

    def load_stage_model(self, stage_id: str):
        """Load fitted model from pkl with caching."""
        if stage_id not in self._model_cache:
            pkl_path = self.models_dir / f'{stage_id}_insample.pkl'
            if not pkl_path.exists():
                raise FileNotFoundError(f"Model not found: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                self._model_cache[stage_id] = pickle.load(f)
        return self._model_cache[stage_id]

    def _get_stage_info(self, model_row: pd.Series, stage: str) -> dict:
        return {
            'sid': model_row[f'{stage}_sid'],
            'cov': model_row[f'{stage}_cov'],
            'dist': model_row[f'{stage}_dist'],
        }

    def _get_subset_for_stage(self, stage: str, threshold_pct: int = 70,
                              plot_rate: bool = True) -> pd.DataFrame:
        """Get data subset and y column appropriate for each stage.

        For bulk/tail, y is death_rate (deaths / exposed_population) when
        plot_rate=True, matching the scale of model predictions (which are
        computed without an exposure offset, i.e. predicted deaths for
        exposure=1 == predicted death rate).  Set plot_rate=False for raw
        death counts.
        """
        data = self.data.copy()
        nonzero = data[data['total_deaths'] > 0]['total_deaths']
        threshold = np.percentile(nonzero, threshold_pct)

        if stage == 's1':
            data['y'] = (data['total_deaths'] > 0).astype(int)
        elif stage == 's2':
            data = data[data['total_deaths'] > 0].copy()
            data['y'] = (data['total_deaths'] > threshold).astype(int)
        elif stage == 'bulk':
            data = data[(data['total_deaths'] > 0) & (data['total_deaths'] <= threshold)].copy()
            data['y'] = data['death_rate'] if plot_rate else data['total_deaths']
        elif stage == 'tail':
            data = data[data['total_deaths'] > threshold].copy()
            data['y'] = data['death_rate'] if plot_rate else data['total_deaths']
        else:
            raise ValueError(f"Unknown stage: {stage}")
        return data

    def _make_pred_df(self, data: pd.DataFrame, n: int, overrides: Optional[dict] = None) -> pd.DataFrame:
        """
        Build a prediction DataFrame of length n with reference covariate values.
        Reference = median for continuous, mode for categorical.
        Log-transformed versions are derived from the raw values.
        Overrides are applied after setting reference values.
        """
        ref = {
            'max_wind_speed': data['max_wind_speed'].median(),
            'sdi': data['sdi'].median(),
            'basin': data['basin'].mode().iloc[0],
            'is_island': int(data['is_island'].mode().iloc[0]),
        }
        if overrides is not None:
            ref.update(overrides)

        # Scalars get broadcast; arrays (e.g. x_grid) are used directly.
        cols = {}
        for k, v in ref.items():
            if isinstance(v, (np.ndarray, pd.Series)):
                cols[k] = np.asarray(v)
            else:
                cols[k] = np.full(n, v)
        df = pd.DataFrame(cols)
        # Keep log versions in sync with raw values
        df['log_max_wind_speed'] = np.log(np.maximum(df['max_wind_speed'].values, 1e-9))
        df['log_sdi'] = np.log(np.maximum(df['sdi'].values, 1e-9))
        return df

    def _predict(self, fitted_model, pred_data: pd.DataFrame, covars: str,
                 sample_data: pd.DataFrame) -> np.ndarray:
        """Build feature matrix from pred_data, align to training columns, predict."""
        feature_names = get_feature_names_from_covars(covars, sample_data)
        X = build_X(pred_data, covars, include_log_exp=False)
        X = sm.add_constant(X, has_constant='add')
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_names]
        return np.asarray(fitted_model.predict(X))

    # ------------------------------------------------------------------
    # Observed data scatter (shared across plot types)
    # ------------------------------------------------------------------

    def _scatter_obs(self, ax: plt.Axes, x_vals: np.ndarray, y_vals: np.ndarray,
                     is_logistic: bool, rng_seed: int = 42,
                     scatter_kws: Optional[dict] = None) -> None:
        kws = {'alpha': 0.2, 's': 10, 'c': 'lightgray', 'zorder': 1}
        if scatter_kws:
            kws.update(scatter_kws)
        rng = np.random.default_rng(rng_seed)
        if is_logistic:
            jitter = rng.uniform(-0.02, 0.02, len(y_vals))
            ax.scatter(x_vals, y_vals + jitter, **kws)
        else:
            ax.scatter(x_vals, np.maximum(y_vals.astype(float), 1e-9), **kws)
            ax.set_yscale('log')

    def _style_ax(self, ax: plt.Axes, x_label: str, is_logistic: bool,
                  plot_rate: bool = True) -> None:
        ax.set_xlabel(x_label)
        if is_logistic:
            ax.set_ylabel('P(Y=1)')
        else:
            ax.set_ylabel('Death rate (deaths / exposed)' if plot_rate else 'Deaths')
        if is_logistic:
            ax.set_ylim(-0.05, 1.05)

    # ------------------------------------------------------------------
    # Continuous-covariate sub-plots
    # ------------------------------------------------------------------

    def _plot_cont_overall(self, ax: plt.Axes, data: pd.DataFrame, fitted_model,
                           covars: str, x_raw_col: str, x_label: str,
                           is_logistic: bool, n_grid: int = 100,
                           scatter_kws: Optional[dict] = None,
                           plot_rate: bool = True) -> None:
        """Single prediction line, all other covariates at reference values."""
        x_vals = data[x_raw_col].values.astype(float)
        self._scatter_obs(ax, x_vals, data['y'].values, is_logistic, scatter_kws=scatter_kws)

        x_grid = np.linspace(x_vals.min(), x_vals.max(), n_grid)
        pred = self._make_pred_df(data, n_grid, {
            x_raw_col: x_grid,
            f'log_{x_raw_col}': np.log(np.maximum(x_grid, 1e-9)),
        })
        y_pred = self._predict(fitted_model, pred, covars, data)
        ax.plot(x_grid, y_pred, color='steelblue', linewidth=2, zorder=2)

        self._style_ax(ax, x_label, is_logistic, plot_rate=plot_rate)
        ax.set_title(f'{x_label} — overall')

    def _plot_cont_by_basin(self, ax: plt.Axes, data: pd.DataFrame, fitted_model,
                            covars: str, x_raw_col: str, x_label: str,
                            is_logistic: bool, n_grid: int = 100,
                            scatter_kws: Optional[dict] = None,
                            plot_rate: bool = True) -> None:
        """Separate prediction line per basin level."""
        x_vals = data[x_raw_col].values.astype(float)
        self._scatter_obs(ax, x_vals, data['y'].values, is_logistic, scatter_kws=scatter_kws)

        x_grid = np.linspace(x_vals.min(), x_vals.max(), n_grid)
        basins = sorted(data['basin'].dropna().unique())
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(basins)))

        for i, basin in enumerate(basins):
            pred = self._make_pred_df(data, n_grid, {
                x_raw_col: x_grid,
                f'log_{x_raw_col}': np.log(np.maximum(x_grid, 1e-9)),
                'basin': basin,
            })
            y_pred = self._predict(fitted_model, pred, covars, data)
            ax.plot(x_grid, y_pred,
                    color=colors[i],
                    linestyle=_LINE_STYLES[i % len(_LINE_STYLES)],
                    linewidth=1.5, label=basin, zorder=2)

        self._style_ax(ax, x_label, is_logistic, plot_rate=plot_rate)
        ax.set_title(f'{x_label} — by basin')
        ax.legend(fontsize=7, loc='best')

    def _plot_cont_by_island(self, ax: plt.Axes, data: pd.DataFrame, fitted_model,
                             covars: str, x_raw_col: str, x_label: str,
                             is_logistic: bool, n_grid: int = 100,
                             scatter_kws: Optional[dict] = None,
                             plot_rate: bool = True) -> None:
        """Prediction lines for is_island=0 and is_island=1."""
        x_vals = data[x_raw_col].values.astype(float)
        self._scatter_obs(ax, x_vals, data['y'].values, is_logistic, scatter_kws=scatter_kws)

        x_grid = np.linspace(x_vals.min(), x_vals.max(), n_grid)

        for island_val, label, color in [(0, 'Non-island', 'steelblue'), (1, 'Island', 'coral')]:
            pred = self._make_pred_df(data, n_grid, {
                x_raw_col: x_grid,
                f'log_{x_raw_col}': np.log(np.maximum(x_grid, 1e-9)),
                'is_island': island_val,
            })
            y_pred = self._predict(fitted_model, pred, covars, data)
            ax.plot(x_grid, y_pred, color=color, linewidth=2, label=label, zorder=2)

        self._style_ax(ax, x_label, is_logistic, plot_rate=plot_rate)
        ax.set_title(f'{x_label} — by island')
        ax.legend(fontsize=7, loc='best')

    # ------------------------------------------------------------------
    # Categorical covariate beeswarm
    # ------------------------------------------------------------------

    def _plot_cat_beeswarm(self, ax: plt.Axes, data: pd.DataFrame, fitted_model,
                           covars: str, cat_col: str, is_logistic: bool,
                           scatter_kws: Optional[dict] = None,
                           plot_rate: bool = True) -> None:
        """
        Jittered dots = observed data per category level.
        Thick horizontal line = model's mean prediction for that category
        (predicted on all observations in that category).
        """
        kws = {'alpha': 0.2, 's': 10, 'c': 'lightgray', 'zorder': 1}
        if scatter_kws:
            kws.update(scatter_kws)

        categories = sorted(data[cat_col].dropna().unique())
        cat_to_x = {cat: i for i, cat in enumerate(categories)}
        rng = np.random.default_rng(42)

        x_base = np.array([cat_to_x[c] for c in data[cat_col]])
        x_jittered = x_base + rng.uniform(-0.3, 0.3, len(data))
        y_vals = data['y'].values.astype(float)

        if is_logistic:
            jitter_y = rng.uniform(-0.02, 0.02, len(y_vals))
            ax.scatter(x_jittered, y_vals + jitter_y, **kws)
        else:
            ax.scatter(x_jittered, np.maximum(y_vals, 1e-9), **kws)
            ax.set_yscale('log')

        for cat in categories:
            cat_subset = data[data[cat_col] == cat]
            mean_pred = self._predict(fitted_model, cat_subset, covars, data).mean()
            x_pos = cat_to_x[cat]
            ax.hlines(mean_pred, x_pos - 0.35, x_pos + 0.35,
                      colors='steelblue', linewidth=3, zorder=2)

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([str(c) for c in categories], rotation=45, ha='right')
        ax.set_xlabel(cat_col)
        if is_logistic:
            ax.set_ylabel('P(Y=1)')
        else:
            ax.set_ylabel('Death rate (deaths / exposed)' if plot_rate else 'Deaths')
        ax.set_title(f'{cat_col} — beeswarm')
        if is_logistic:
            ax.set_ylim(-0.05, 1.05)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def vet_stage(self, model_row: pd.Series, stage: str,
                  threshold_pct: int = 70, figsize: tuple = (16, 12),
                  scatter_kws: Optional[dict] = None,
                  plot_rate: bool = True) -> plt.Figure:
        """
        Produce all vetting plots for one stage of one model.

        Layout (3 rows × 3 cols):
          Row 0: wind speed overall | wind speed by basin | wind speed by island
          Row 1: SDI overall        | SDI by basin        | SDI by island
          Row 2: basin beeswarm     | island beeswarm     | (empty)

        For stages that don't use a covariate, prediction lines are flat.
        For binary stages (s1, s2): y-axis is probability.
        For continuous stages (bulk, tail): y-axis is death rate (deaths /
        exposed_population) when plot_rate=True (default), raw death counts
        when plot_rate=False.

        Parameters
        ----------
        scatter_kws : dict, optional
            Passed to every ax.scatter() call for observed data points.
            Keys are any valid matplotlib scatter kwargs: alpha, s, c, color,
            marker, edgecolors, linewidths, etc.
            Example: scatter_kws={'alpha': 0.8, 's': 20, 'c': 'steelblue'}
        plot_rate : bool, default True
            If True (default), bulk/tail y-axis is death rate.
            If False, y-axis is raw death counts.
        """
        info = self._get_stage_info(model_row, stage)
        data = self._get_subset_for_stage(stage, threshold_pct, plot_rate=plot_rate)
        fitted_model = self.load_stage_model(info['sid'])
        covars = info['cov']
        is_logistic = stage in ('s1', 's2')

        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle(
            f"Stage: {stage.upper()}  |  covars: {covars}  |  dist: {info['dist']}  |  n={len(data)}",
            fontsize=12,
        )

        # Row 0: wind speed
        self._plot_cont_overall(axes[0, 0], data, fitted_model, covars,
                                'max_wind_speed', 'Wind speed', is_logistic,
                                scatter_kws=scatter_kws, plot_rate=plot_rate)
        self._plot_cont_by_basin(axes[0, 1], data, fitted_model, covars,
                                 'max_wind_speed', 'Wind speed', is_logistic,
                                 scatter_kws=scatter_kws, plot_rate=plot_rate)
        self._plot_cont_by_island(axes[0, 2], data, fitted_model, covars,
                                  'max_wind_speed', 'Wind speed', is_logistic,
                                  scatter_kws=scatter_kws, plot_rate=plot_rate)

        # Row 1: SDI
        self._plot_cont_overall(axes[1, 0], data, fitted_model, covars,
                                'sdi', 'SDI', is_logistic,
                                scatter_kws=scatter_kws, plot_rate=plot_rate)
        self._plot_cont_by_basin(axes[1, 1], data, fitted_model, covars,
                                 'sdi', 'SDI', is_logistic,
                                 scatter_kws=scatter_kws, plot_rate=plot_rate)
        self._plot_cont_by_island(axes[1, 2], data, fitted_model, covars,
                                  'sdi', 'SDI', is_logistic,
                                  scatter_kws=scatter_kws, plot_rate=plot_rate)

        # Row 2: categorical beeswarms
        self._plot_cat_beeswarm(axes[2, 0], data, fitted_model, covars, 'basin', is_logistic,
                                scatter_kws=scatter_kws, plot_rate=plot_rate)
        self._plot_cat_beeswarm(axes[2, 1], data, fitted_model, covars, 'is_island', is_logistic,
                                scatter_kws=scatter_kws, plot_rate=plot_rate)
        axes[2, 2].axis('off')

        plt.tight_layout()
        return fig

    def vet_model(self, model_row: pd.Series, threshold_pct: int = 70,
                  scatter_kws: Optional[dict] = None,
                  plot_rate: bool = True) -> dict:
        """Produce vetting plots for all 4 DH model stages."""
        return {
            stage: self.vet_stage(model_row, stage, threshold_pct,
                                  scatter_kws=scatter_kws, plot_rate=plot_rate)
            for stage in ('s1', 's2', 'bulk', 'tail')
        }

    # ------------------------------------------------------------------
    # Prediction DataFrames
    # ------------------------------------------------------------------

    def predict_df(self, model_row: pd.Series) -> pd.DataFrame:
        """
        Insample predictions for every row in the dataset.

        All 4 stage models are applied to every row — no subsetting.
        Bulk and tail models predict death RATE (deaths / exposed_population),
        keeping them in the same space as the s2 classification threshold.
        GLMs are called with exposure=1 to return rate. ML models return
        counts (trained on counts) which are then divided by exposure.

        Columns added:
          pred_s1             P(deaths > 0) for every row
          pred_s2             P(high rate | deaths > 0) for every row
          pred_bulk_rate      E[death rate | bulk model] for every row
          pred_tail_rate      E[death rate | tail model] for every row
          pred_rate           assembled rate: s1*((1-s2)*bulk_rate + s2*tail_rate)
          pred_count          pred_rate * exposed_population
          death_rate          total_deaths / exposed_population (observed)
          rate_threshold      rate-based threshold u for bulk/tail split
          is_bulk             1 if positive & rate <= u, else 0
          is_tail             1 if positive & rate > u, else 0
        """
        from .distributions import DISTRIBUTIONS, ML_MODELS

        data    = self.data.reset_index(drop=True)
        n       = len(data)
        thr_pct = int(model_row['threshold_pct'])

        pos_mask  = data['death_y_n'].values == 1
        rates_all = data['total_deaths'].values / np.maximum(data['exposed_population'].values, 1)
        u       = float(np.percentile(rates_all[pos_mask], thr_pct))
        exp_all = data['exposed_population'].values
        ones    = np.ones(n)

        pred_s1        = np.full(n, np.nan)
        pred_s2        = np.full(n, np.nan)
        pred_bulk_rate = np.full(n, np.nan)
        pred_tail_rate = np.full(n, np.nan)

        for stage_key in ('s1', 's2', 'bulk', 'tail'):
            sid      = model_row[f'{stage_key}_sid']
            cov      = model_row[f'{stage_key}_cov']
            dist     = model_row[f'{stage_key}_dist']
            is_ml    = dist in ML_MODELS
            dist_mod = DISTRIBUTIONS[dist]
            try:
                fitted = self.load_stage_model(sid)
            except FileNotFoundError as e:
                import warnings
                warnings.warn(f'predict_df: {stage_key} — {e}')
                continue

            try:
                if stage_key == 's1':
                    X = build_X(data, cov, include_log_exp=False)
                    pred_s1[:] = np.asarray(dist_mod.predict(fitted, X, task='binary'), dtype=float)

                elif stage_key == 's2':
                    X = build_X(data, cov, include_log_exp=False)
                    pred_s2[:] = np.asarray(dist_mod.predict(fitted, X, task='binary'), dtype=float)

                elif stage_key == 'bulk':
                    # GLM: exposure=1 → rate directly; clip to (0, u]
                    X   = build_X(data, cov, include_log_exp=False)
                    raw = np.asarray(dist_mod.predict(fitted, X, exposure=ones, task='count'), dtype=float)
                    pred_bulk_rate[:] = np.clip(raw, 1e-9, u)

                elif stage_key == 'tail':
                    # GLM: exposure=1 → rate directly; clip to [u, inf)
                    X   = build_X(data, cov, include_log_exp=False)
                    raw = np.asarray(dist_mod.predict(fitted, X, exposure=ones), dtype=float)
                    pred_tail_rate[:] = np.maximum(raw, u)

            except Exception as e:
                import warnings
                warnings.warn(f'predict_df: {stage_key} failed — {e}')

        pred_rate = pred_s1 * ((1 - pred_s2) * pred_bulk_rate + pred_s2 * pred_tail_rate)

        out = data.copy()
        out['pred_s1']        = pred_s1
        out['pred_s2']        = pred_s2
        out['pred_bulk_rate'] = pred_bulk_rate
        out['pred_tail_rate'] = pred_tail_rate
        out['pred_rate']      = pred_rate
        out['pred_count']     = pred_rate * exp_all
        out['death_rate']     = data['total_deaths'] / data['exposed_population']
        out['rate_threshold'] = u
        out['is_bulk']        = (pos_mask & (rates_all <= u)).astype(int)
        out['is_tail']        = (pos_mask & (rates_all > u)).astype(int)
        return out

    def predict_df_oos(self, model_row: pd.Series,
                       seed: Optional[int] = None) -> pd.DataFrame:
        """
        OOS predictions for every row in the dataset.

        All 4 stage models are applied to every test row in each fold.
        Predictions are averaged across seeds (or a single seed if specified).

        Parameters
        ----------
        seed : int or None
            If int, use only that seed's 5 CV folds.
            If None (default), average across all DEFAULT_SEEDS.

        Returns the same column structure as predict_df().
        """
        from .distributions import DISTRIBUTIONS, ML_MODELS
        from .features import align_X as _align
        from .constants import DEFAULT_SEEDS, DEFAULT_FOLDS

        data    = self.data.reset_index(drop=True)
        n       = len(data)
        thr_pct = int(model_row['threshold_pct'])
        seeds   = [seed] if seed is not None else list(DEFAULT_SEEDS)
        k_folds = DEFAULT_FOLDS

        sum_s1   = np.zeros(n)
        sum_s2   = np.zeros(n)
        sum_bulk = np.zeros(n)
        sum_tail = np.zeros(n)
        cnt      = np.zeros(n, dtype=int)  # same fold structure for all stages

        for s in seeds:
            for fold in range(k_folds):
                rng      = np.random.default_rng(s)
                fold_ids = rng.integers(0, k_folds, size=n)
                tr_idx   = np.where(fold_ids != fold)[0]
                te_idx   = np.where(fold_ids == fold)[0]
                train    = data.iloc[tr_idx]
                test     = data.iloc[te_idx]
                exp_te   = test['exposed_population'].values

                train_pos = train[train['death_y_n'] == 1]
                if len(train_pos) == 0:
                    continue
                tr_rates = (train_pos['total_deaths'].values
                            / train_pos['exposed_population'].values)
                u = float(np.percentile(tr_rates, thr_pct))

                for stage_key in ('s1', 's2', 'bulk', 'tail'):
                    sid      = model_row[f'{stage_key}_sid']
                    cov      = model_row[f'{stage_key}_cov']
                    dist     = model_row[f'{stage_key}_dist']
                    is_ml    = dist in ML_MODELS
                    dist_mod = DISTRIBUTIONS[dist]

                    pkl = self.models_dir / f'{sid}_seed{s}_fold{fold}.pkl'
                    if not pkl.exists():
                        continue
                    with open(pkl, 'rb') as f:
                        fitted = pickle.load(f)

                    ones_te = np.ones(len(te_idx))

                    try:
                        if stage_key == 's1':
                            X_tr = build_X(train, cov, include_log_exp=False)
                            X_te = _align(build_X(test, cov, include_log_exp=False),
                                          list(X_tr.columns))
                            preds = np.asarray(
                                dist_mod.predict(fitted, X_te, task='binary'), dtype=float)
                            sum_s1[te_idx] += preds
                            cnt[te_idx]    += 1

                        elif stage_key == 's2':
                            X_tr = build_X(train_pos, cov, include_log_exp=False)
                            X_te = _align(build_X(test, cov, include_log_exp=False),
                                          list(X_tr.columns))
                            preds = np.asarray(
                                dist_mod.predict(fitted, X_te, task='binary'), dtype=float)
                            sum_s2[te_idx] += preds

                        elif stage_key == 'bulk':
                            train_bulk = train_pos[tr_rates <= u]
                            X_tr  = build_X(train_bulk, cov, include_log_exp=False)
                            X_te  = _align(build_X(test, cov, include_log_exp=False),
                                           list(X_tr.columns))
                            raw   = np.asarray(
                                dist_mod.predict(fitted, X_te, exposure=ones_te, task='count'),
                                dtype=float)
                            preds = np.clip(raw, 1e-9, u)
                            sum_bulk[te_idx] += preds

                        elif stage_key == 'tail':
                            train_tail = train_pos[tr_rates > u]
                            X_tr  = build_X(train_tail, cov, include_log_exp=False)
                            X_te  = _align(build_X(test, cov, include_log_exp=False),
                                           list(X_tr.columns))
                            raw   = np.asarray(
                                dist_mod.predict(fitted, X_te, exposure=ones_te),
                                dtype=float)
                            preds = np.maximum(raw, u)
                            sum_tail[te_idx] += preds

                    except Exception as e:
                        import warnings
                        warnings.warn(
                            f'predict_df_oos: {stage_key} seed={s} fold={fold} — {e}')

        safe_cnt       = np.maximum(cnt, 1)
        pred_s1        = np.where(cnt > 0, sum_s1   / safe_cnt, np.nan)
        pred_s2        = np.where(cnt > 0, sum_s2   / safe_cnt, np.nan)
        pred_bulk_rate = np.where(cnt > 0, sum_bulk / safe_cnt, np.nan)
        pred_tail_rate = np.where(cnt > 0, sum_tail / safe_cnt, np.nan)

        pos_mask  = data['death_y_n'].values == 1
        rates_all = data['total_deaths'].values / np.maximum(data['exposed_population'].values, 1)
        u_global  = float(np.percentile(rates_all[pos_mask], thr_pct))
        exp_all   = data['exposed_population'].values

        pred_rate = pred_s1 * ((1 - pred_s2) * pred_bulk_rate + pred_s2 * pred_tail_rate)

        out = data.copy()
        out['pred_s1']        = pred_s1
        out['pred_s2']        = pred_s2
        out['pred_bulk_rate'] = pred_bulk_rate
        out['pred_tail_rate'] = pred_tail_rate
        out['pred_rate']      = pred_rate
        out['pred_count']     = pred_rate * exp_all
        out['death_rate']     = data['total_deaths'] / data['exposed_population']
        out['rate_threshold'] = u_global
        out['is_bulk']        = (pos_mask & (rates_all <= u_global)).astype(int)
        out['is_tail']        = (pos_mask & (rates_all > u_global)).astype(int)
        return out
