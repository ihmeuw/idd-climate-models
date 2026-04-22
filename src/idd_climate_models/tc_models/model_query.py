"""
model_query.py
Utilities for querying and comparing DH models from exhaustive results.

Usage in notebook:
    from idd_climate_models.tc_models.model_query import ModelQuery
    
    # Pass both filtered df (for display) and full df (for queries)
    mq = ModelQuery(df_filtered, df_full=df_unfiltered)
    
    # Get one model by exact config (searches df_full)
    model = mq.get(s1_cov='wind_sdi_basin', s2_cov='sdi_basin', 
                   bulk_cov='wind_sdi', tail_cov='none')
    
    # Get multiple models by enumerating options (searches df_full)
    models = mq.enumerate(
        s1_cov=['wind_sdi_basin', 'wind_sdi_basin_island'],
        s2_cov='sdi_basin',  # fixed
        bulk_cov=['wind_sdi', 'wind'],
        tail_cov='none'
    )
    
    # Compare models
    mq.compare(models, metrics=['mae_rate', 'pred_obs_ratio', 'cov_10'])
"""

import pandas as pd
from typing import Union, List, Dict, Optional
from itertools import product


class ModelQuery:
    """Query and compare DH models from exhaustive results."""
    
    # Default columns for filtering (distributions fixed at NB/gamma for DH)
    CONFIG_COLS = ['s1_cov', 's2_cov', 'bulk_cov', 'tail_cov']
    DIST_COLS = ['s1_dist', 's2_dist', 'bulk_dist', 'tail_dist']
    
    # Default metrics for comparison
    DEFAULT_METRICS = [
        'mae_rate', 'rmse_rate', 'cor_rate', 
        'cov_5', 'cov_10', 'cov_20',
        'pred_obs_ratio',
        'mae_rate_is', 'pred_obs_ratio_is'  # IS metrics if available
    ]
    
    def __init__(self, df: pd.DataFrame, 
                 df_full: Optional[pd.DataFrame] = None,
                 threshold_pct: int = 70,
                 s1_dist: str = 'statsmodels_logistic',
                 s2_dist: str = 'statsmodels_logistic',
                 bulk_dist: str = 'statsmodels_nb',
                 tail_dist: str = 'statsmodels_gamma'):
        """
        Initialize with results DataFrame and default distribution config.
        
        Parameters
        ----------
        df : DataFrame (possibly filtered) for display
        df_full : Optional unfiltered DataFrame for lookups (if None, uses df)
        threshold_pct : threshold percentage (default 70)
        s1_dist, s2_dist, bulk_dist, tail_dist : default distributions
        """
        self.df = df
        self.df_full = df_full if df_full is not None else df
        self.defaults = {
            'threshold_pct': threshold_pct,
            's1_dist': s1_dist,
            's2_dist': s2_dist,
            'bulk_dist': bulk_dist,
            'tail_dist': tail_dist,
        }
    
    def get(self, 
            s1_cov: str,
            s2_cov: str,
            bulk_cov: str,
            tail_cov: str,
            **overrides) -> Optional[pd.Series]:
        """
        Get a single model by exact covariate config.
        Searches df_full (unfiltered) so filtered-out models can be found.
        
        Parameters
        ----------
        s1_cov, s2_cov, bulk_cov, tail_cov : covariate specs for each stage
        **overrides : override default distributions or threshold
        
        Returns
        -------
        Series with model metrics, or None if not found
        """
        config = {**self.defaults, **overrides}
        config.update({
            's1_cov': s1_cov,
            's2_cov': s2_cov,
            'bulk_cov': bulk_cov,
            'tail_cov': tail_cov,
        })
        
        mask = pd.Series([True] * len(self.df_full), index=self.df_full.index)
        for col, val in config.items():
            if col in self.df_full.columns:
                mask &= (self.df_full[col] == val)
        
        matches = self.df_full[mask]
        if len(matches) == 0:
            return None
        if len(matches) > 1:
            print(f"Warning: {len(matches)} matches found, returning first")
        return matches.iloc[0]
    
    def enumerate(self,
                  s1_cov: Union[str, List[str]],
                  s2_cov: Union[str, List[str]],
                  bulk_cov: Union[str, List[str]],
                  tail_cov: Union[str, List[str]],
                  **overrides) -> pd.DataFrame:
        """
        Get all models matching combinations of covariate specs.
        Searches df_full (unfiltered) so filtered-out models can be found.
        
        Parameters
        ----------
        s1_cov, s2_cov, bulk_cov, tail_cov : 
            Single value or list of values for each stage
        **overrides : override default distributions or threshold
        
        Returns
        -------
        DataFrame with all matching models
        """
        # Ensure all are lists
        def to_list(x):
            return x if isinstance(x, list) else [x]
        
        s1_list = to_list(s1_cov)
        s2_list = to_list(s2_cov)
        bulk_list = to_list(bulk_cov)
        tail_list = to_list(tail_cov)
        
        # Build base config from defaults + overrides
        base_config = {**self.defaults, **overrides}
        
        # Filter to matching distributions/threshold first
        mask = pd.Series([True] * len(self.df_full), index=self.df_full.index)
        for col, val in base_config.items():
            if col in self.df_full.columns:
                mask &= (self.df_full[col] == val)
        
        subset = self.df_full[mask].copy()
        
        # Now filter to covariate combinations
        cov_mask = (
            subset['s1_cov'].isin(s1_list) &
            subset['s2_cov'].isin(s2_list) &
            subset['bulk_cov'].isin(bulk_list) &
            subset['tail_cov'].isin(tail_list)
        )
        
        return subset[cov_mask].copy()
    
    def neighbors(self, 
                  model: pd.Series,
                  vary_stages: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Find models that differ by ±1 covariate token in one stage.
        Searches df_full (unfiltered) so filtered-out models can be found.
        
        Parameters
        ----------
        model : Series (from get() or df.iloc[i])
        vary_stages : which stages to vary (default: all 4)
        
        Returns
        -------
        DataFrame of neighboring models
        """
        if vary_stages is None:
            vary_stages = self.CONFIG_COLS
        
        def tokens(cov):
            return set() if cov == 'none' else set(cov.split('_'))
        
        def is_neighbor(row_cov, target_cov):
            t1, t2 = tokens(row_cov), tokens(target_cov)
            return len(t1.symmetric_difference(t2)) == 1
        
        target = {col: model[col] for col in self.CONFIG_COLS}
        
        # Get base subset matching distributions from df_full
        base_config = {**self.defaults}
        mask = pd.Series([True] * len(self.df_full), index=self.df_full.index)
        for col, val in base_config.items():
            if col in self.df_full.columns and col in model.index:
                mask &= (self.df_full[col] == model[col])
        subset = self.df_full[mask]
        
        neighbors = []
        for stage in vary_stages:
            # Match exactly on other stages
            other_mask = pd.Series([True] * len(subset), index=subset.index)
            for s in self.CONFIG_COLS:
                if s != stage:
                    other_mask &= (subset[s] == target[s])
            
            stage_subset = subset[other_mask].copy()
            stage_subset['_is_neighbor'] = stage_subset[stage].apply(
                lambda x: is_neighbor(x, target[stage])
            )
            stage_neighbors = stage_subset[stage_subset['_is_neighbor']].copy()
            stage_neighbors['_varied_stage'] = stage
            stage_neighbors = stage_neighbors.drop(columns=['_is_neighbor'])
            neighbors.append(stage_neighbors)
        
        if not neighbors:
            return pd.DataFrame()
        
        result = pd.concat(neighbors, ignore_index=True)
        return result.sort_values('mae_rate')
    
    def compare(self,
                models: pd.DataFrame,
                metrics: Optional[List[str]] = None,
                show_config: bool = True) -> pd.DataFrame:
        """
        Format models for side-by-side comparison.
        
        Parameters
        ----------
        models : DataFrame of models (from enumerate() or neighbors())
        metrics : metrics to show (default: DEFAULT_METRICS)
        show_config : include covariate config columns
        
        Returns
        -------
        DataFrame formatted for comparison
        """
        if metrics is None:
            metrics = [m for m in self.DEFAULT_METRICS if m in models.columns]
        
        cols = []
        if show_config:
            cols.extend([c for c in self.CONFIG_COLS if c in models.columns])
        cols.extend([m for m in metrics if m in models.columns])
        
        return models[cols].copy()
    
    def diff(self, model1: pd.Series, model2: pd.Series, 
             metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Show difference between two models.
        
        Returns DataFrame with model1, model2, and diff columns.
        """
        if metrics is None:
            metrics = [m for m in self.DEFAULT_METRICS 
                      if m in model1.index and m in model2.index]
        
        config_cols = [c for c in self.CONFIG_COLS 
                      if c in model1.index and c in model2.index]
        
        rows = []
        for col in config_cols + metrics:
            v1, v2 = model1[col], model2[col]
            if col in config_cols:
                diff = '←' if v1 != v2 else ''
            else:
                try:
                    diff = float(v2) - float(v1)
                except:
                    diff = ''
            rows.append({'metric': col, 'model1': v1, 'model2': v2, 'diff': diff})
        
        return pd.DataFrame(rows).set_index('metric')
    
    def compare_to_reference(self, 
                             models: pd.DataFrame,
                             reference: pd.Series,
                             metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare models to a reference, showing only stages that differ.
        
        Parameters
        ----------
        models : DataFrame of models to compare
        reference : Series (the "best" model to compare against)
        metrics : metrics to show
        
        Returns
        -------
        DataFrame with differing stage(s) highlighted and metric deltas
        """
        if metrics is None:
            metrics = [m for m in self.DEFAULT_METRICS if m in models.columns]
        
        ref_config = {c: reference[c] for c in self.CONFIG_COLS}
        
        rows = []
        for idx, row in models.iterrows():
            # Find which stages differ
            diff_stages = []
            for stage in self.CONFIG_COLS:
                if row[stage] != ref_config[stage]:
                    diff_stages.append(stage)
            
            if not diff_stages:
                continue  # Skip if identical to reference
            
            row_data = {
                'diff_stages': ', '.join(diff_stages),
                'n_diff': len(diff_stages),
            }
            
            # Show config for differing stages only
            for stage in diff_stages:
                row_data[f'{stage}_ref'] = ref_config[stage]
                row_data[f'{stage}_model'] = row[stage]
            
            # Add metrics and deltas
            for m in metrics:
                if m in row.index and m in reference.index:
                    row_data[m] = row[m]
                    try:
                        row_data[f'{m}_delta'] = float(row[m]) - float(reference[m])
                    except:
                        pass
            
            rows.append(row_data)
        
        result = pd.DataFrame(rows)
        if len(result) > 0:
            result = result.sort_values('mae_rate' if 'mae_rate' in result.columns else 'n_diff')
        return result
