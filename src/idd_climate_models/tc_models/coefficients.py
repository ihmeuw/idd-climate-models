"""
coefficients.py
Load and display fitted model coefficients for DH (Double Hurdle) models.

Usage:
    from idd_climate_models.tc_models.coefficients import display_dh_model_coefficients
    
    # Display coefficients for a model row from comparison results
    display_dh_model_coefficients(model_row, "TOPSIS Winner")
    
    # Or get raw coefficients dict
    coefs = load_model_coefficients(model_row)
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .features import build_X
from .data import load_tc_data
from .constants import STAGE_RESULTS_DIR


# Default models directory for DH comparison (v2 = corrected covariate aliases)
DH_MODELS_DIR = STAGE_RESULTS_DIR.parent / 'stage_results_dh_v2' / 'models'


def stage_to_id(stage: dict) -> str:
    """Generate stage ID hash from config dict."""
    key = json.dumps(stage, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def get_stage_ids(model_row, threshold_col: str = 'threshold_pct') -> Dict[str, str]:
    """
    Generate all stage IDs for a DH model configuration.
    
    Parameters
    ----------
    model_row : pd.Series or dict-like
        Row from comparison results with columns:
        s1_dist, s1_cov, s2_dist, s2_cov, bulk_dist, bulk_cov, 
        tail_dist, tail_cov, threshold_pct
    threshold_col : str
        Name of threshold column (default 'threshold_pct')
    
    Returns
    -------
    dict : stage_name -> stage_id
    """
    thr = int(model_row[threshold_col])
    
    stages = {
        's1': {
            'stage_type': 's1', 
            'dist': model_row['s1_dist'], 
            'covars': model_row.get('s1_cov', 'none')
        },
        's2': {
            'stage_type': 'pos_binary', 
            'dist': model_row['s2_dist'], 
            'covars': model_row.get('s2_cov', 'none'), 
            'threshold_pct': thr
        },
        'bulk': {
            'stage_type': 'dh_bulk', 
            'dist': model_row['bulk_dist'], 
            'covars': model_row['bulk_cov'], 
            'threshold_pct': thr
        },
        'tail': {
            'stage_type': 'tail', 
            'dist': model_row['tail_dist'], 
            'covars': model_row['tail_cov'], 
            'threshold_pct': thr
        },
    }
    return {k: stage_to_id(v) for k, v in stages.items()}


def get_feature_names_from_covars(covars: str, sample_data: pd.DataFrame) -> list:
    """Get feature names by building X matrix with specified covars."""
    X = build_X(sample_data, covars, include_log_exp=False)
    X = sm.add_constant(X, has_constant='add')
    return X.columns.tolist()


# Cached sample data
_sample_data = None

def _get_sample_data():
    """Load sample data once for feature name reconstruction."""
    global _sample_data
    if _sample_data is None:
        _sample_data = load_tc_data().head(1000)
    return _sample_data


def load_model_coefficients(
    model_row,
    models_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load coefficients from insample models for all 4 DH stages.
    
    Parameters
    ----------
    model_row : pd.Series or dict-like
        Row from comparison results DataFrame
    models_dir : Path, optional
        Directory containing pickled models. Default: DH_MODELS_DIR
    
    Returns
    -------
    dict : stage_name -> {
        'params': array of coefficients,
        'feature_names': list of names,
        'pvalues': array (if available),
        'std_errors': array (if available),
        'stage_id': str,
        'covars': str,
        'dist': str,
    }
    """
    if models_dir is None:
        models_dir = DH_MODELS_DIR
    models_dir = Path(models_dir)
    
    # Use actual stage IDs from the row if available, otherwise recompute
    if 's1_sid' in model_row.index or 's1_sid' in model_row:
        stage_ids = {
            's1': model_row['s1_sid'],
            's2': model_row['s2_sid'],
            'bulk': model_row['bulk_sid'],
            'tail': model_row['tail_sid'],
        }
    else:
        stage_ids = get_stage_ids(model_row)
    
    coefficients = {}
    sample_data = _get_sample_data()
    
    # Map stage to covars and distribution
    stage_info = {
        's1': {'covars': model_row.get('s1_cov', 'none'), 'dist': model_row['s1_dist']},
        's2': {'covars': model_row.get('s2_cov', 'none'), 'dist': model_row['s2_dist']},
        'bulk': {'covars': model_row['bulk_cov'], 'dist': model_row['bulk_dist']},
        'tail': {'covars': model_row['tail_cov'], 'dist': model_row['tail_dist']},
    }
    
    for stage_name, stage_id in stage_ids.items():
        pkl_path = models_dir / f'{stage_id}_insample.pkl'
        info = stage_info[stage_name]
        covars = info['covars']
        dist = info['dist']
        
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                fitted = pickle.load(f)
            
            if hasattr(fitted, 'params'):
                # Try to get feature names from the fitted model first
                if hasattr(fitted, '_X_cols'):
                    feature_names = list(fitted._X_cols)
                else:
                    # Fall back to reconstructing from covars
                    try:
                        feature_names = get_feature_names_from_covars(covars, sample_data)
                    except Exception:
                        feature_names = []
                
                # Distribution-specific extra parameters
                n_params = len(fitted.params)
                if len(feature_names) < n_params:
                    if 'nb' in dist.lower():
                        feature_names.append('alpha')
                    elif 'gamma' in dist.lower():
                        feature_names.append('scale')
                    elif 'lognormal' in dist.lower():
                        feature_names.append('sigma')
                    else:
                        while len(feature_names) < n_params:
                            feature_names.append(f'param_{len(feature_names)}')
                
                coefficients[stage_name] = {
                    'params': fitted.params,
                    'feature_names': feature_names,
                    'stage_id': stage_id,
                    'covars': covars,
                    'dist': dist,
                }
                if hasattr(fitted, 'pvalues'):
                    coefficients[stage_name]['pvalues'] = fitted.pvalues
                if hasattr(fitted, 'bse'):
                    coefficients[stage_name]['std_errors'] = fitted.bse
            else:
                coefficients[stage_name] = {
                    'stage_id': stage_id, 
                    'covars': covars,
                    'dist': dist,
                    'note': 'No params attribute'
                }
        else:
            coefficients[stage_name] = {
                'stage_id': stage_id, 
                'covars': covars,
                'dist': dist,
                'note': 'Model file not found'
            }
    
    return coefficients


def display_dh_model_coefficients(
    model_row, 
    model_name: str = "Model",
    models_dir: Optional[Path] = None,
):
    """
    Display coefficients for all 4 DH stages with feature names.
    
    Parameters
    ----------
    model_row : pd.Series or dict-like
        Row from comparison results DataFrame
    model_name : str
        Label to print in header
    models_dir : Path, optional
        Directory containing pickled models
    """
    coefs = load_model_coefficients(model_row, models_dir)
    
    # Model config summary
    print(f"{'='*70}")
    print(f"  {model_name.upper()}")
    print(f"{'='*70}")
    print(f"  threshold: {int(model_row['threshold_pct'])}%")
    print(f"  bulk: {model_row['bulk_dist']} ({model_row['bulk_cov']})")
    print(f"  tail: {model_row['tail_dist']} ({model_row['tail_cov']})")
    print()
    
    for stage in ['s1', 's2', 'bulk', 'tail']:
        info = coefs[stage]
        covars_str = info.get('covars', 'unknown')
        dist_str = info.get('dist', 'unknown')
        print(f"--- {stage.upper()} [{dist_str}, {covars_str}] (stage_id: {info['stage_id']}) ---")
        
        if 'params' not in info:
            print(f"    {info.get('note', 'No data')}")
            print()
            continue
        
        params = info['params']
        pvals = info.get('pvalues')
        ses = info.get('std_errors')
        feature_names = info.get('feature_names', [f'x{i}' for i in range(len(params))])
        
        # Handle length mismatch
        if len(feature_names) != len(params):
            feature_names = [f'x{i}' for i in range(len(params))]
        
        # Build coefficient table
        rows = []
        for i, name in enumerate(feature_names):
            coef = params[i] if isinstance(params, np.ndarray) else params.iloc[i]
            row = {'feature': name, 'coef': coef}
            if pvals is not None and i < len(pvals):
                pv = pvals[i] if isinstance(pvals, np.ndarray) else pvals.iloc[i]
                row['pvalue'] = pv
                # Significance stars
                if pv < 0.001:
                    row['sig'] = '***'
                elif pv < 0.01:
                    row['sig'] = '**'
                elif pv < 0.05:
                    row['sig'] = '*'
                elif pv < 0.1:
                    row['sig'] = '.'
                else:
                    row['sig'] = ''
            if ses is not None and i < len(ses):
                row['std_err'] = ses[i] if isinstance(ses, np.ndarray) else ses.iloc[i]
            rows.append(row)
        
        coef_df = pd.DataFrame(rows)
        
        # Format display
        if 'pvalue' in coef_df.columns:
            coef_df['coef'] = coef_df['coef'].map('{:+.4f}'.format)
            coef_df['pvalue'] = coef_df['pvalue'].map('{:.4f}'.format)
            if 'std_err' in coef_df.columns:
                coef_df['std_err'] = coef_df['std_err'].map('{:.4f}'.format)
                display_cols = ['feature', 'coef', 'std_err', 'pvalue', 'sig']
            else:
                display_cols = ['feature', 'coef', 'pvalue', 'sig']
        else:
            coef_df['coef'] = coef_df['coef'].map('{:+.4f}'.format)
            display_cols = ['feature', 'coef']
        
        print(coef_df[display_cols].to_string(index=False))
        print()


def coefficients_to_dataframe(model_row, models_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Return coefficients as a tidy DataFrame.
    
    Returns DataFrame with columns: stage, feature, coef, std_err, pvalue, sig
    """
    coefs = load_model_coefficients(model_row, models_dir)
    
    rows = []
    for stage in ['s1', 's2', 'bulk', 'tail']:
        info = coefs[stage]
        if 'params' not in info:
            continue
        
        params = info['params']
        pvals = info.get('pvalues')
        ses = info.get('std_errors')
        feature_names = info.get('feature_names', [f'x{i}' for i in range(len(params))])
        
        for i, name in enumerate(feature_names):
            row = {
                'stage': stage,
                'dist': info.get('dist'),
                'covars': info.get('covars'),
                'feature': name,
                'coef': params[i] if isinstance(params, np.ndarray) else params.iloc[i],
            }
            if pvals is not None and i < len(pvals):
                row['pvalue'] = pvals[i] if isinstance(pvals, np.ndarray) else pvals.iloc[i]
            if ses is not None and i < len(ses):
                row['std_err'] = ses[i] if isinstance(ses, np.ndarray) else ses.iloc[i]
            rows.append(row)
    
    return pd.DataFrame(rows)
