"""
stage_diagnostics.py
Diagnostic plots for individual DH model stages.

Usage:
    from idd_climate_models.tc_models.stage_diagnostics import StageDiagnostics
    
    diag = StageDiagnostics()
    
    # Plot logistic stage (s1 or s2)
    diag.plot_logistic(stage_id, covariate='wind_speed_var', title='P(Y>0)')
    
    # Plot bulk stage (NB/gamma/lognormal)
    diag.plot_bulk(stage_id, title='Bulk: E[Y|0<Y<threshold]')
    
    # Plot tail stage (gamma/GPD)
    diag.plot_tail(stage_id, title='Tail: E[Y|Y>threshold]')
    
    # Plot all 4 stages for a model config
    diag.plot_dh_model(model_row)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Union
import pickle

from .constants import STAGE_MODELS_DIR
from .data import load_tc_data
from .features import build_X


# Default path for DH v2 models
DH_MODELS_DIR = Path('/mnt/team/rapidresponse/pub/tropical-storms/direct_risk/stage_results_dh_v2/models')


class StageDiagnostics:
    """Diagnostic plots for DH model stages."""
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Parameters
        ----------
        models_dir : Path to pkl files (default: stage_results_dh_v2/models)
        """
        self.models_dir = models_dir or DH_MODELS_DIR
        self._data = None
        self._cached_models = {}
    
    @property
    def data(self):
        """Lazy-load TC data."""
        if self._data is None:
            self._data = load_tc_data()
        return self._data
    
    def load_model(self, stage_id: str):
        """Load fitted model from pkl."""
        if stage_id not in self._cached_models:
            pkl_path = self.models_dir / f'{stage_id}_insample.pkl'
            if not pkl_path.exists():
                raise FileNotFoundError(f"Model not found: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                self._cached_models[stage_id] = pickle.load(f)
        return self._cached_models[stage_id]
    
    def plot_logistic(self, 
                      stage_id: str,
                      covariate: str = 'wind_speed_var',
                      ax: Optional[plt.Axes] = None,
                      title: Optional[str] = None,
                      show_rug: bool = True,
                      n_grid: int = 100) -> plt.Axes:
        """
        Plot logistic regression: observed 0/1 vs covariate with fitted curve.
        
        Parameters
        ----------
        stage_id : stage ID hash
        covariate : x-axis variable ('wind_speed_var', 'sdi_var', etc.)
        ax : matplotlib axes (creates new if None)
        title : plot title
        show_rug : show rug plot of data points
        n_grid : number of points for fitted curve
        """
        model = self.load_model(stage_id)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        # Get the data that was used to fit
        # Model stores exog_names which tells us the covariates
        exog_names = model.model.exog_names
        
        # Get raw covariate from data
        data = self.data.copy()
        
        # Determine y based on stage type (from model)
        # For s1: y = (deaths > 0)
        # For s2: y = (deaths > threshold) given deaths > 0
        # We'll infer from the model's endog
        y_fitted = model.fittedvalues
        y_obs = model.model.endog
        
        # Get covariate values
        if covariate in data.columns:
            x_raw = data[covariate].values[:len(y_obs)]
        elif 'wind_speed_var' in covariate or 'wind' in covariate.lower():
            x_raw = data['max_wind_speed'].values[:len(y_obs)]
        elif 'sdi' in covariate.lower():
            x_raw = data['sdi'].values[:len(y_obs)]
        else:
            # Try to get from exog
            X = model.model.exog
            if X.shape[1] > 1:
                x_raw = X[:, 1]  # First covariate after intercept
            else:
                x_raw = np.arange(len(y_obs))
        
        # Sort for plotting
        sort_idx = np.argsort(x_raw)
        x_sorted = x_raw[sort_idx]
        y_obs_sorted = y_obs[sort_idx]
        y_fitted_sorted = y_fitted[sort_idx]
        
        # Plot observed (jittered)
        jitter = np.random.uniform(-0.02, 0.02, len(y_obs_sorted))
        ax.scatter(x_sorted, y_obs_sorted + jitter, alpha=0.3, s=20, 
                   c=['red' if y == 0 else 'blue' for y in y_obs_sorted],
                   label='Observed')
        
        # Plot fitted probability curve
        ax.plot(x_sorted, y_fitted_sorted, 'k-', linewidth=2, label='Fitted P(Y=1)')
        
        # Rug plot
        if show_rug:
            ax.plot(x_sorted[y_obs_sorted == 0], 
                    np.zeros(sum(y_obs_sorted == 0)) - 0.05, 
                    '|', color='red', alpha=0.3, markersize=10)
            ax.plot(x_sorted[y_obs_sorted == 1], 
                    np.ones(sum(y_obs_sorted == 1)) + 0.05, 
                    '|', color='blue', alpha=0.3, markersize=10)
        
        ax.set_xlabel(covariate)
        ax.set_ylabel('P(Y=1)')
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Logistic: {stage_id[:8]}')
        
        return ax
    
    def plot_bulk(self,
                  stage_id: str,
                  ax: Optional[plt.Axes] = None,
                  title: Optional[str] = None,
                  log_scale: bool = True) -> plt.Axes:
        """
        Plot bulk stage: observed vs predicted.
        
        Parameters
        ----------
        stage_id : stage ID hash
        ax : matplotlib axes
        title : plot title
        log_scale : use log scale for axes
        """
        model = self.load_model(stage_id)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        y_obs = model.model.endog
        y_pred = model.fittedvalues
        
        # Handle zeros for log scale
        if log_scale:
            y_obs_plot = np.maximum(y_obs, 0.1)
            y_pred_plot = np.maximum(y_pred, 0.1)
        else:
            y_obs_plot = y_obs
            y_pred_plot = y_pred
        
        ax.scatter(y_pred_plot, y_obs_plot, alpha=0.4, s=20)
        
        # 1:1 line
        lims = [min(y_pred_plot.min(), y_obs_plot.min()),
                max(y_pred_plot.max(), y_obs_plot.max())]
        ax.plot(lims, lims, 'r--', linewidth=1, label='1:1')
        
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Observed')
        ax.legend()
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Bulk: {stage_id[:8]}')
        
        return ax
    
    def plot_tail(self,
                  stage_id: str,
                  ax: Optional[plt.Axes] = None,
                  title: Optional[str] = None,
                  log_scale: bool = True) -> plt.Axes:
        """
        Plot tail stage: observed vs predicted (similar to bulk).
        """
        return self.plot_bulk(stage_id, ax=ax, title=title or f'Tail: {stage_id[:8]}', 
                              log_scale=log_scale)
    
    def plot_residuals(self,
                       stage_id: str,
                       ax: Optional[plt.Axes] = None,
                       title: Optional[str] = None) -> plt.Axes:
        """
        Plot residuals vs fitted values.
        """
        model = self.load_model(stage_id)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        y_pred = model.fittedvalues
        residuals = model.resid_response
        
        ax.scatter(y_pred, residuals, alpha=0.4, s=20)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Residuals: {stage_id[:8]}')
        
        return ax
    
    def plot_dh_model(self,
                      model_row: pd.Series,
                      figsize: tuple = (14, 10)) -> plt.Figure:
        """
        Plot diagnostics for all 4 stages of a DH model.
        
        Parameters
        ----------
        model_row : Series with s1_sid, s2_sid, bulk_sid, tail_sid columns
        figsize : figure size
        
        Returns
        -------
        matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # S1: P(Y > 0)
        s1_sid = model_row.get('s1_sid')
        s1_cov = model_row.get('s1_cov', '')
        if s1_sid:
            try:
                self.plot_logistic(s1_sid, ax=axes[0, 0], 
                                   title=f'S1: P(Y>0) | cov={s1_cov}')
            except Exception as e:
                axes[0, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[0, 0].set_title(f'S1: {s1_cov}')
        
        # S2: P(Y > threshold | Y > 0)
        s2_sid = model_row.get('s2_sid')
        s2_cov = model_row.get('s2_cov', '')
        if s2_sid:
            try:
                self.plot_logistic(s2_sid, ax=axes[0, 1],
                                   title=f'S2: P(Y>thr|Y>0) | cov={s2_cov}')
            except Exception as e:
                axes[0, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[0, 1].set_title(f'S2: {s2_cov}')
        
        # Bulk: E[Y | 0 < Y < threshold]
        bulk_sid = model_row.get('bulk_sid')
        bulk_cov = model_row.get('bulk_cov', '')
        bulk_dist = model_row.get('bulk_dist', '')
        if bulk_sid:
            try:
                self.plot_bulk(bulk_sid, ax=axes[1, 0],
                               title=f'Bulk ({bulk_dist}): cov={bulk_cov}')
            except Exception as e:
                axes[1, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[1, 0].set_title(f'Bulk: {bulk_cov}')
        
        # Tail: E[Y | Y > threshold]
        tail_sid = model_row.get('tail_sid')
        tail_cov = model_row.get('tail_cov', '')
        tail_dist = model_row.get('tail_dist', '')
        if tail_sid:
            try:
                self.plot_tail(tail_sid, ax=axes[1, 1],
                               title=f'Tail ({tail_dist}): cov={tail_cov}')
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[1, 1].set_title(f'Tail: {tail_cov}')
        
        plt.tight_layout()
        return fig
    
    def plot_stage_comparison(self,
                              stage_ids: List[str],
                              stage_type: str = 'logistic',
                              labels: Optional[List[str]] = None,
                              figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Compare multiple stages of the same type on one plot.
        
        Parameters
        ----------
        stage_ids : list of stage ID hashes
        stage_type : 'logistic', 'bulk', or 'tail'
        labels : labels for legend
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(stage_ids)))
        
        for i, sid in enumerate(stage_ids):
            label = labels[i] if labels else sid[:8]
            model = self.load_model(sid)
            
            y_pred = model.fittedvalues
            y_obs = model.model.endog
            
            if stage_type == 'logistic':
                # Sort by fitted values for cleaner curves
                sort_idx = np.argsort(y_pred)
                ax.plot(np.arange(len(y_pred)), y_pred[sort_idx], 
                        color=colors[i], label=label, linewidth=2)
            else:
                ax.scatter(y_pred, y_obs, alpha=0.5, color=colors[i], 
                           label=label, s=20)
        
        ax.legend()
        ax.set_title(f'{stage_type.title()} Stage Comparison')
        
        return fig
