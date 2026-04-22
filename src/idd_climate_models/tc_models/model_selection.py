"""
model_selection.py

Multi-criteria model selection pipeline:
1. Borda Count ranking
2. Pareto Dominance
3. Kendall Tau metric agreement
4. Friedman + Nemenyi test (or Pairwise Dominance at scale)
5. TOPSIS ranking
6. Configuration clustering
7. Winner profile analysis

Usage:
    from idd_climate_models.tc_models.model_selection import (
        borda_rank, pareto_frontier, kendall_tau_heatmap,
        friedman_nemenyi, pairwise_dominance_summary,
        topsis_rank, cluster_configurations, winner_profile, 
        run_full_pipeline
    )
"""

from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster


# ══════════════════════════════════════════════════════════════════════════════
# METRIC CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_METRICS = {
    # OOS metrics
    'oos_mae_rate': 'lower',
    'oos_rmse_rate': 'lower',
    'oos_mae_log': 'lower',
    'oos_cor_rate': 'higher',
    'oos_zero_acc': 'higher',
    'oos_cov_5': 'higher',
    'oos_cov_10': 'higher',
    'oos_cov_20': 'higher',
    # IS metrics
    'is_mae_rate': 'lower',
    'is_rmse_rate': 'lower',
    'is_mae_log': 'lower',
    'is_cor_rate': 'higher',
    'is_zero_acc': 'higher',
    'is_cov_5': 'higher',
    'is_cov_10': 'higher',
    'is_cov_20': 'higher',
}

# Calibration metrics (special handling - closer to 1 is better)
CALIBRATION_METRICS = ['oos_pred_obs_ratio', 'is_pred_obs_ratio']


def get_metric_ranks(df: pd.DataFrame, metrics: Dict[str, str],
                     calibration_metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute ranks for all models on each metric.
    
    Args:
        df: DataFrame with models as rows
        metrics: Dict mapping metric name -> 'lower' or 'higher' (what's better)
        calibration_metrics: List of metrics where closer to 1 is better
        
    Returns:
        DataFrame with same index, columns = metric_rank for each metric
    """
    calibration_metrics = calibration_metrics or []
    ranks = pd.DataFrame(index=df.index)
    
    for metric, direction in metrics.items():
        if metric not in df.columns:
            continue
        
        if metric in calibration_metrics:
            # For calibration: rank by |value - 1|
            ranks[f'{metric}_rank'] = np.abs(df[metric] - 1).rank(method='average')
        elif direction == 'lower':
            ranks[f'{metric}_rank'] = df[metric].rank(method='average')
        else:  # higher is better
            ranks[f'{metric}_rank'] = df[metric].rank(method='average', ascending=False)
    
    return ranks


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: BORDA COUNT
# ══════════════════════════════════════════════════════════════════════════════

def borda_rank(df: pd.DataFrame, metrics: Dict[str, str],
               calibration_metrics: Optional[List[str]] = None,
               plot: bool = True, figsize: Tuple[int, int] = (12, 6)
               ) -> Tuple[pd.DataFrame, int]:
    """
    Compute Borda count (sum of ranks) for each model.
    
    Args:
        df: DataFrame with models as rows, metrics as columns
        metrics: Dict mapping metric name -> 'lower' or 'higher'
        calibration_metrics: Metrics where closer to 1 is better
        plot: Whether to plot Borda score vs rank
        figsize: Figure size
        
    Returns:
        (df_with_borda, cutpoint_index)
        - df_with_borda: Original df with 'borda_score' and 'borda_rank' columns added
        - cutpoint_index: Suggested index where quality drops off (elbow detection)
    """
    ranks = get_metric_ranks(df, metrics, calibration_metrics)
    
    result = df.copy()
    result['borda_score'] = ranks.sum(axis=1)
    result['borda_rank'] = result['borda_score'].rank(method='average')
    result = result.sort_values('borda_score')
    
    # Elbow detection: find where second derivative is maximized
    scores = result['borda_score'].values
    n = len(scores)
    
    if n > 10:
        # Compute second differences (discrete second derivative)
        d1 = np.diff(scores)
        d2 = np.diff(d1)
        
        # Find the point of maximum acceleration (biggest jump in slope)
        # Skip first/last few points to avoid edge effects
        margin = max(5, n // 20)
        search_range = slice(margin, len(d2) - margin)
        cutpoint = margin + np.argmax(d2[search_range])
    else:
        cutpoint = n // 2
    
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(1, n + 1), scores, 'b-', linewidth=1.5, label='Borda Score')
        ax.axvline(cutpoint, color='red', linestyle='--', alpha=0.7, 
                   label=f'Cut point (rank {cutpoint})')
        ax.fill_between(range(1, cutpoint + 1), 0, scores[:cutpoint], 
                        alpha=0.2, color='green', label='Top tier')
        ax.set_xlabel('Model Rank')
        ax.set_ylabel('Borda Score (lower = better)')
        ax.set_title(f'Borda Count Ranking ({len(metrics)} metrics, {n} models)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"\n=== Borda Count Summary ===")
        print(f"Total models: {n}")
        print(f"Metrics used: {len(metrics)}")
        print(f"Suggested cutpoint: rank {cutpoint} (top {cutpoint} models)")
        print(f"Top tier Borda range: {scores[0]:.1f} - {scores[cutpoint-1]:.1f}")
        print(f"Bottom tier starts at: {scores[cutpoint]:.1f}")
    
    return result, cutpoint


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: PARETO DOMINANCE
# ══════════════════════════════════════════════════════════════════════════════

def pareto_frontier(df: pd.DataFrame, metrics: Dict[str, str],
                    calibration_metrics: Optional[List[str]] = None,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Find Pareto-dominant (non-dominated) models.
    
    A model is dominated if another model beats or ties it on ALL metrics.
    
    Args:
        df: DataFrame with models
        metrics: Dict mapping metric name -> 'lower' or 'higher'
        calibration_metrics: Metrics where closer to 1 is better
        verbose: Print summary
        
    Returns:
        DataFrame with only the non-dominated models
    """
    calibration_metrics = calibration_metrics or []
    
    # Build normalized values where higher = better for all
    values = np.zeros((len(df), len(metrics)))
    metric_list = list(metrics.keys())
    
    for i, (metric, direction) in enumerate(metrics.items()):
        if metric not in df.columns:
            continue
        col = df[metric].values
        
        if metric in calibration_metrics:
            # Transform: closer to 1 is better -> negate |x - 1|
            values[:, i] = -np.abs(col - 1)
        elif direction == 'lower':
            values[:, i] = -col
        else:
            values[:, i] = col
    
    # Find non-dominated indices
    n = len(df)
    dominated = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            # Check if j dominates i (j >= i on all, j > i on at least one)
            if np.all(values[j] >= values[i]) and np.any(values[j] > values[i]):
                dominated[i] = True
                break
    
    pareto_df = df.iloc[~dominated].copy()
    
    if verbose:
        print(f"\n=== Pareto Frontier ===")
        print(f"Total models: {n}")
        print(f"Non-dominated: {len(pareto_df)} ({100*len(pareto_df)/n:.1f}%)")
        print(f"Dominated: {n - len(pareto_df)}")
    
    return pareto_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: KENDALL TAU AGREEMENT
# ══════════════════════════════════════════════════════════════════════════════

def kendall_tau_heatmap(df: pd.DataFrame, metrics: Dict[str, str],
                        calibration_metrics: Optional[List[str]] = None,
                        figsize: Tuple[int, int] = (12, 10)) -> pd.DataFrame:
    """
    Compute pairwise Kendall tau correlations between metric rankings.
    
    Args:
        df: DataFrame with models
        metrics: Dict mapping metric name -> direction
        calibration_metrics: Metrics where closer to 1 is better
        figsize: Figure size for heatmap
        
    Returns:
        Correlation matrix DataFrame
    """
    calibration_metrics = calibration_metrics or []
    ranks = get_metric_ranks(df, metrics, calibration_metrics)
    
    metric_names = [c.replace('_rank', '') for c in ranks.columns]
    n_metrics = len(metric_names)
    
    tau_matrix = np.zeros((n_metrics, n_metrics))
    
    for i in range(n_metrics):
        for j in range(n_metrics):
            if i == j:
                tau_matrix[i, j] = 1.0
            elif i < j:
                tau, _ = stats.kendalltau(ranks.iloc[:, i], ranks.iloc[:, j])
                tau_matrix[i, j] = tau
                tau_matrix[j, i] = tau
    
    tau_df = pd.DataFrame(tau_matrix, index=metric_names, columns=metric_names)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(tau_matrix, dtype=bool), k=1)
    sns.heatmap(tau_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, vmin=-1, vmax=1, mask=mask)
    ax.set_title('Kendall Tau Correlation Between Metric Rankings')
    plt.tight_layout()
    plt.show()
    
    # Interpretation
    avg_tau = tau_matrix[np.triu_indices_from(tau_matrix, k=1)].mean()
    disagreeing = (tau_matrix < 0).sum() // 2
    
    print(f"\n=== Metric Agreement Analysis ===")
    print(f"Average pairwise Kendall tau: {avg_tau:.3f}")
    print(f"Pairs with negative correlation: {disagreeing}")
    
    if avg_tau > 0.5:
        print("Interpretation: Metrics largely agree. Flat top tier likely reflects genuine equivalence.")
    elif avg_tau > 0.2:
        print("Interpretation: Moderate agreement. Some trade-offs between metrics.")
    else:
        print("Interpretation: Low agreement. Flat top tier may be aggregation artifact (metric cancellation).")
    
    return tau_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: FRIEDMAN + NEMENYI
# ══════════════════════════════════════════════════════════════════════════════

def friedman_nemenyi(df: pd.DataFrame, metrics: Dict[str, str],
                     calibration_metrics: Optional[List[str]] = None,
                     top_n: int = 30, alpha: float = 0.05,
                     plot_cd: bool = True, figsize: Tuple[int, int] = (12, 8)
                     ) -> Tuple[float, float, Optional[pd.DataFrame]]:
    """
    Run Friedman test and Nemenyi post-hoc test on top models.
    
    Note: Friedman test treats models as "treatments" and metrics as "blocks".
    
    Args:
        df: DataFrame with models (should be pre-sorted by some criterion)
        metrics: Dict mapping metric name -> direction
        calibration_metrics: Metrics where closer to 1 is better
        top_n: Number of top models to compare
        alpha: Significance level
        plot_cd: Whether to plot critical difference diagram
        figsize: Figure size
        
    Returns:
        (friedman_stat, p_value, nemenyi_result)
    """
    calibration_metrics = calibration_metrics or []
    
    # Take top N models
    df_top = df.head(top_n).copy()
    ranks = get_metric_ranks(df_top, metrics, calibration_metrics)
    
    n_models = len(df_top)
    n_metrics = ranks.shape[1]
    
    # Friedman test: are there significant differences among models?
    # Ranks per model across metrics (transpose: rows=metrics, cols=models)
    rank_matrix = ranks.values.T
    
    try:
        stat, p_value = stats.friedmanchisquare(*rank_matrix.T.tolist())
    except Exception as e:
        print(f"Friedman test failed: {e}")
        return np.nan, np.nan, None
    
    print(f"\n=== Friedman Test (top {n_models} models) ===")
    print(f"Chi-squared statistic: {stat:.3f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value > alpha:
        print(f"Result: No significant difference at alpha={alpha} (models are statistically equivalent)")
        return stat, p_value, None
    
    print(f"Result: Significant difference detected at alpha={alpha}")
    
    # Nemenyi post-hoc test
    # Critical difference: CD = q_alpha * sqrt(k*(k+1)/(6*n))
    # where k = number of models, n = number of metrics
    # q_alpha values for Nemenyi test (approximations for large k)
    q_alpha_table = {
        0.10: 2.291,
        0.05: 2.569,
        0.01: 3.144,
    }
    q_alpha = q_alpha_table.get(alpha, 2.569)
    
    cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_metrics))
    
    # Average ranks per model
    avg_ranks = ranks.mean(axis=1)
    
    print(f"\nNemenyi Critical Difference: {cd:.3f}")
    print(f"Models with avg rank difference < {cd:.3f} are statistically indistinguishable")
    
    # Build groups of indistinguishable models
    sorted_idx = avg_ranks.sort_values().index
    sorted_ranks = avg_ranks[sorted_idx].values
    
    if plot_cd:
        _plot_critical_difference(sorted_idx, sorted_ranks, cd, figsize)
    
    nemenyi_df = pd.DataFrame({
        'model_idx': sorted_idx,
        'avg_rank': sorted_ranks,
    })
    
    return stat, p_value, nemenyi_df


def _plot_critical_difference(model_ids, avg_ranks, cd, figsize):
    """
    Plot Demšar-style critical difference diagram.
    
    Models are shown on a horizontal axis ordered by average rank.
    Horizontal bars connect groups of models that are statistically 
    indistinguishable (within CD of each other).
    """
    n = len(model_ids)
    
    # Sort by rank (best = lowest rank on left)
    sorted_order = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[sorted_order]
    sorted_ids = [model_ids[i] for i in sorted_order]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Horizontal layout: ranks on x-axis, models labeled
    rank_min, rank_max = sorted_ranks.min(), sorted_ranks.max()
    ax.set_xlim(rank_min - 1, rank_max + 1)
    
    # Plot models as points on the rank axis
    # Left side: best models (low rank), Right side: worst (high rank)
    y_top = 0.9  # Where the best models go
    y_bottom = 0.1  # Where the worst models go
    
    # Split models into two rows to avoid label overlap
    top_half = n // 2
    
    for i, (mid, rank) in enumerate(zip(sorted_ids, sorted_ranks)):
        if i < top_half:
            y = y_top
            va = 'bottom'
        else:
            y = y_bottom
            va = 'top'
        
        ax.plot(rank, 0.5, 'ko', markersize=6, zorder=3)
        
        # Draw vertical line from point to label
        label_y = y
        ax.plot([rank, rank], [0.5, label_y], 'k-', linewidth=0.5, alpha=0.5)
        ax.annotate(f'{mid}', (rank, label_y), ha='center', va=va, fontsize=7, rotation=0)
    
    # Draw the horizontal axis line
    ax.axhline(0.5, color='black', linewidth=1, zorder=1)
    
    # Find cliques of indistinguishable models and draw horizontal bars
    # A clique is a maximal set where all pairs are within CD
    # Simpler approach: draw bars for contiguous runs within CD of each other
    
    cliques = _find_cd_cliques(sorted_ranks, cd)
    
    bar_y_positions = np.linspace(0.55, 0.75, len(cliques))
    
    for bar_idx, (start_idx, end_idx) in enumerate(cliques):
        if end_idx > start_idx:  # Only draw if >1 model in clique
            left_rank = sorted_ranks[start_idx]
            right_rank = sorted_ranks[end_idx]
            y_bar = bar_y_positions[bar_idx % len(bar_y_positions)]
            
            # Horizontal bar over the clique
            ax.plot([left_rank, right_rank], [y_bar, y_bar], 'b-', linewidth=3, solid_capstyle='round')
    
    # Draw CD scale bar in upper left
    cd_bar_x = rank_min + 0.5
    cd_bar_y = 0.85
    ax.plot([cd_bar_x, cd_bar_x + cd], [cd_bar_y, cd_bar_y], 'k-', linewidth=2)
    ax.annotate(f'CD = {cd:.2f}', (cd_bar_x + cd/2, cd_bar_y + 0.03), ha='center', fontsize=9)
    
    ax.set_xlabel('Average Rank (lower = better)')
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(f'Critical Difference Diagram ({n} models)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def _find_cd_cliques(sorted_ranks, cd):
    """
    Find cliques of models that are all within CD of each other.
    
    Uses a greedy approach: for each model, extend the clique as far right
    as possible while all pairs remain within CD.
    
    Returns list of (start_idx, end_idx) tuples.
    """
    n = len(sorted_ranks)
    cliques = []
    used = set()
    
    for i in range(n):
        if i in used:
            continue
        
        # Extend clique from i as far as possible
        j = i
        while j + 1 < n and (sorted_ranks[j + 1] - sorted_ranks[i]) <= cd:
            j += 1
        
        if j > i:  # Found a clique of size > 1
            cliques.append((i, j))
            # Mark all in this clique as used
            for k in range(i, j + 1):
                used.add(k)
    
    return cliques
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 (alternative): PAIRWISE DOMINANCE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def pairwise_dominance_summary(df: pd.DataFrame, metrics: Dict[str, str],
                                calibration_metrics: Optional[List[str]] = None,
                                plot: bool = True, figsize: Tuple[int, int] = (12, 6)
                                ) -> pd.DataFrame:
    """
    Simpler alternative to Friedman/Nemenyi for comparing many models.
    
    For each model pair, counts how often A beats B across metrics.
    Returns a dominance score (number of models this model beats on majority of metrics).
    
    Much faster than Friedman at large scale and gives intuitive results.
    
    Args:
        df: DataFrame with models (should be pre-filtered to top tier)
        metrics: Dict mapping metric name -> 'lower' or 'higher'
        calibration_metrics: Metrics where closer to 1 is better
        plot: Whether to plot dominance distribution
        figsize: Figure size
        
    Returns:
        DataFrame with dominance counts and statistics
    """
    calibration_metrics = calibration_metrics or []
    metric_list = [m for m in metrics if m in df.columns]
    n_metrics = len(metric_list)
    n_models = len(df)
    
    # Build comparison matrix: values[i, j] = value of model i on metric j
    # Transformed so higher = better for all
    values = np.zeros((n_models, n_metrics))
    
    for j, metric in enumerate(metric_list):
        col = df[metric].values
        if metric in calibration_metrics:
            values[:, j] = -np.abs(col - 1)  # Closer to 1 = higher (better)
        elif metrics[metric] == 'lower':
            values[:, j] = -col  # Lower raw = higher transformed = better
        else:
            values[:, j] = col
    
    # For each pair (i, j): count metrics where i beats j
    # i "beats" j on a metric if values[i, k] > values[j, k]
    
    # Efficient computation: for each model, count models it beats on majority
    dominance_counts = np.zeros(n_models, dtype=int)
    wins_per_model = np.zeros(n_models, dtype=int)  # total metric-wins
    
    for i in range(n_models):
        wins_vs_others = 0
        models_beaten = 0
        for j in range(n_models):
            if i == j:
                continue
            # How many metrics does i beat j on?
            wins_i = np.sum(values[i] > values[j])
            wins_vs_others += wins_i
            if wins_i > n_metrics / 2:
                models_beaten += 1
        dominance_counts[i] = models_beaten
        wins_per_model[i] = wins_vs_others
    
    # Build result DataFrame
    result = df.copy()
    result['dominance_count'] = dominance_counts
    result['dominance_pct'] = 100 * dominance_counts / (n_models - 1)
    result['total_metric_wins'] = wins_per_model
    result['dominance_rank'] = result['dominance_count'].rank(ascending=False, method='average')
    result = result.sort_values('dominance_count', ascending=False)
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]))
        
        # Distribution of dominance counts
        ax = axes[0]
        ax.hist(dominance_counts, bins=min(50, n_models // 5), edgecolor='black', alpha=0.7)
        ax.axvline(np.median(dominance_counts), color='red', linestyle='--', 
                   label=f'Median: {np.median(dominance_counts):.0f}')
        ax.set_xlabel('Models Beaten (on majority of metrics)')
        ax.set_ylabel('Count')
        ax.set_title(f'Dominance Distribution ({n_models} models)')
        ax.legend()
        
        # Top models dominance vs Borda rank (if available)
        ax = axes[1]
        if 'borda_rank' in result.columns:
            ax.scatter(result['borda_rank'], result['dominance_count'], alpha=0.5, s=10)
            ax.set_xlabel('Borda Rank')
            ax.set_ylabel('Dominance Count')
            ax.set_title('Dominance vs Borda Rank')
        else:
            ax.scatter(range(n_models), dominance_counts[np.argsort(-dominance_counts)], 
                       alpha=0.5, s=10)
            ax.set_xlabel('Model (sorted by dominance)')
            ax.set_ylabel('Dominance Count')
            ax.set_title('Dominance Curve')
        
        plt.tight_layout()
        plt.show()
    
    print(f"\n=== Pairwise Dominance Summary ===")
    print(f"Models compared: {n_models}")
    print(f"Metrics used: {n_metrics}")
    print(f"Max dominance: {dominance_counts.max()} / {n_models - 1} models beaten")
    print(f"Median dominance: {np.median(dominance_counts):.0f}")
    
    # Top 10 most dominant
    top_cols = ['dominance_rank', 'dominance_count', 'dominance_pct']
    config_cols = ['threshold_pct', 'bulk_dist', 'tail_dist']
    config_cols = [c for c in config_cols if c in result.columns]
    print(f"\nTop 10 most dominant models:")
    print(result.head(10)[top_cols + config_cols].to_string())
    
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: TOPSIS
# ══════════════════════════════════════════════════════════════════════════════

def topsis_rank(df: pd.DataFrame, metrics: Dict[str, str],
                calibration_metrics: Optional[List[str]] = None,
                weights: Optional[Dict[str, float]] = None,
                normalize_method: str = 'minmax',
                verbose: bool = True
                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).
    
    Args:
        df: DataFrame with models
        metrics: Dict mapping metric name -> 'lower' or 'higher'
        calibration_metrics: Metrics where closer to 1 is better
        weights: Optional dict of metric -> weight (default: equal weights)
        normalize_method: 'minmax' (recommended) or 'vector' (original TOPSIS)
        verbose: Print summary
        
    Returns:
        (df_with_topsis, influence_df)
        - df_with_topsis: Original df with 'topsis_score' and 'topsis_rank'
        - influence_df: DataFrame showing metric influence on TOPSIS separation
    """
    calibration_metrics = calibration_metrics or []
    metric_list = [m for m in metrics if m in df.columns]
    n_metrics = len(metric_list)
    n_models = len(df)
    
    # Build decision matrix
    decision = np.zeros((n_models, n_metrics))
    
    for i, metric in enumerate(metric_list):
        col = df[metric].values
        if metric in calibration_metrics:
            # Transform to "higher is better": use e.g. 1 - |x - 1|
            decision[:, i] = 1 - np.abs(col - 1)
        elif metrics[metric] == 'lower':
            decision[:, i] = -col  # Flip so higher = better
        else:
            decision[:, i] = col
    
    # Step 1: Normalize
    if normalize_method == 'minmax':
        # Min-max normalization to [0, 1] — each metric gets equal influence
        col_min = decision.min(axis=0)
        col_max = decision.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1  # Avoid division by zero
        normalized = (decision - col_min) / col_range
    else:
        # Vector normalization (original TOPSIS — preserves variance, can be dominated by high-variance metrics)
        norm = np.sqrt((decision ** 2).sum(axis=0))
        norm[norm == 0] = 1
        normalized = decision / norm
    
    # Step 2: Apply weights
    if weights is None:
        w = np.ones(n_metrics) / n_metrics
    else:
        w = np.array([weights.get(m, 1.0) for m in metric_list])
        w = w / w.sum()
    
    weighted = normalized * w
    
    # Step 3: Ideal and anti-ideal solutions
    ideal = weighted.max(axis=0)
    anti_ideal = weighted.min(axis=0)
    
    # Step 4: Distances
    d_ideal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_anti = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))
    
    # Step 5: TOPSIS score (higher = better)
    topsis_score = d_anti / (d_ideal + d_anti + 1e-10)
    
    result = df.copy()
    result['topsis_score'] = topsis_score
    result['topsis_rank'] = pd.Series(topsis_score).rank(ascending=False, method='average').values
    result = result.sort_values('topsis_rank')
    
    # Compute metric influence: how much does each metric contribute to separation?
    # With minmax normalization, this should be roughly equal (range ≈ 1 for all)
    raw_range = decision.max(axis=0) - decision.min(axis=0)
    influence = pd.DataFrame({
        'metric': metric_list,
        'raw_range': raw_range,
        'normalized_range': weighted.max(axis=0) - weighted.min(axis=0),
        'normalized_std': weighted.std(axis=0),
        'weight': w,
    })
    influence['influence'] = influence['normalized_range'] * influence['weight']
    influence = influence.sort_values('influence', ascending=False)
    
    if verbose:
        print(f"\n=== TOPSIS Ranking (normalize={normalize_method}) ===")
        print(f"Models ranked: {n_models}")
        print(f"Metrics used: {n_metrics}")
        print(f"\nTop 10 by TOPSIS:")
        top_cols = ['topsis_rank', 'topsis_score'] + metric_list[:5]
        print(result.head(10)[top_cols].to_string())
        
        print(f"\nMetric Influence (separation power):")
        print(influence.to_string(index=False))
    
    return result, influence


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: WINNER PROFILE & CONFIGURATION CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def cluster_configurations(df: pd.DataFrame, 
                           config_cols: Optional[List[str]] = None,
                           n_clusters: Optional[int] = None,
                           max_clusters: int = 10,
                           plot: bool = True,
                           figsize: Tuple[int, int] = (14, 6)
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster models by their configuration patterns.
    
    Uses hierarchical clustering on one-hot encoded configurations to identify
    distinct "families" of models that dominate the top tier.
    
    Args:
        df: DataFrame with models (should be pre-filtered to top tier)
        config_cols: Columns that define configuration. Default includes common ones.
        n_clusters: Number of clusters. If None, auto-detect using silhouette score.
        max_clusters: Max clusters to try when auto-detecting
        plot: Whether to plot cluster analysis
        figsize: Figure size
        
    Returns:
        (df_with_clusters, cluster_summary)
        - df_with_clusters: Original df with 'config_cluster' column
        - cluster_summary: DataFrame describing each cluster's characteristics
    """
    if config_cols is None:
        config_cols = ['threshold_pct', 'bulk_dist', 'bulk_cov', 'tail_dist', 'tail_cov',
                       's1_cov', 's2_cov']
    config_cols = [c for c in config_cols if c in df.columns]
    
    if not config_cols:
        print("No configuration columns found!")
        return df, pd.DataFrame()
    
    # One-hot encode configurations
    df_config = df[config_cols].copy()
    dummies = pd.get_dummies(df_config, columns=config_cols, drop_first=False)
    X = dummies.values
    
    # Hierarchical clustering
    Z = linkage(X, method='ward')
    
    # Auto-detect number of clusters if not specified
    if n_clusters is None:
        from sklearn.metrics import silhouette_score
        
        best_k = 2
        best_score = -1
        
        for k in range(2, min(max_clusters + 1, len(df))):
            labels = fcluster(Z, k, criterion='maxclust')
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        
        n_clusters = best_k
        print(f"Auto-detected {n_clusters} clusters (silhouette score: {best_score:.3f})")
    
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    
    result = df.copy()
    result['config_cluster'] = labels
    
    # Build cluster summary
    cluster_summaries = []
    
    for cluster_id in sorted(np.unique(labels)):
        cluster_df = result[result['config_cluster'] == cluster_id]
        size = len(cluster_df)
        
        # Find the modal configuration
        config_str = cluster_df[config_cols].astype(str).agg(' | '.join, axis=1)
        mode_config = Counter(config_str).most_common(1)[0]
        
        # Dominant values for each column
        dominant = {}
        for col in config_cols:
            vc = cluster_df[col].value_counts()
            dominant[col] = f"{vc.index[0]} ({100*vc.iloc[0]/size:.0f}%)"
        
        # Performance stats (if available)
        perf = {}
        for metric in ['oos_mae_rate', 'oos_pred_obs_ratio', 'topsis_score', 'dominance_pct']:
            if metric in cluster_df.columns:
                perf[f'{metric}_mean'] = cluster_df[metric].mean()
        
        summary = {
            'cluster': cluster_id,
            'size': size,
            'pct_of_top_tier': 100 * size / len(df),
            'modal_config': mode_config[0],
            'modal_config_count': mode_config[1],
            **dominant,
            **perf,
        }
        cluster_summaries.append(summary)
    
    cluster_summary = pd.DataFrame(cluster_summaries)
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Cluster sizes
        ax = axes[0]
        cluster_sizes = [len(result[result['config_cluster'] == c]) for c in sorted(np.unique(labels))]
        bars = ax.bar(range(1, n_clusters + 1), cluster_sizes, edgecolor='black')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Models')
        ax.set_title(f'Configuration Clusters ({len(df)} models → {n_clusters} clusters)')
        ax.set_xticks(range(1, n_clusters + 1))
        
        # Performance by cluster (if TOPSIS available)
        ax = axes[1]
        if 'topsis_score' in result.columns:
            cluster_perf = [result[result['config_cluster'] == c]['topsis_score'].mean() 
                           for c in sorted(np.unique(labels))]
            ax.bar(range(1, n_clusters + 1), cluster_perf, edgecolor='black', color='green', alpha=0.7)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Mean TOPSIS Score')
            ax.set_title('Average Performance by Cluster')
            ax.set_xticks(range(1, n_clusters + 1))
        else:
            ax.text(0.5, 0.5, 'Run TOPSIS first for performance comparison', 
                    ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.show()
    
    print(f"\n=== Configuration Clusters ===")
    print(f"Top tier models: {len(df)}")
    print(f"Distinct clusters: {n_clusters}")
    
    # Print cluster details
    for _, row in cluster_summary.iterrows():
        print(f"\nCluster {int(row['cluster'])} ({int(row['size'])} models, {row['pct_of_top_tier']:.1f}% of top tier):")
        for col in config_cols:
            if col in row:
                print(f"  {col}: {row[col]}")
    
    return result, cluster_summary


def winner_profile(df: pd.DataFrame, top_n: int = 10,
                   id_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Analyze what the top models have in common.
    
    Args:
        df: DataFrame sorted by some ranking (e.g., TOPSIS)
        top_n: Number of top models to analyze
        id_cols: Columns that identify model configuration
                 Default: threshold_pct, bulk_dist, bulk_cov, tail_dist, tail_cov
                 
    Returns:
        Summary DataFrame showing common patterns
    """
    if id_cols is None:
        id_cols = ['threshold_pct', 's1_cov', 's2_cov', 
                   'bulk_dist', 'bulk_cov', 'tail_dist', 'tail_cov']
    
    id_cols = [c for c in id_cols if c in df.columns]
    
    top_df = df.head(top_n)
    
    print(f"\n=== Winner Profile (Top {top_n} Models) ===\n")
    
    summary = {}
    for col in id_cols:
        vc = top_df[col].value_counts()
        mode = vc.index[0]
        mode_pct = 100 * vc.iloc[0] / top_n
        unique = vc.shape[0]
        
        summary[col] = {
            'mode': mode,
            'mode_count': vc.iloc[0],
            'mode_pct': mode_pct,
            'unique_values': unique,
        }
        
        print(f"{col}:")
        print(f"  Most common: {mode} ({vc.iloc[0]}/{top_n} = {mode_pct:.0f}%)")
        if unique > 1:
            print(f"  Others: {dict(vc.iloc[1:4])}")
        print()
    
    # Identify strong patterns (>= 70% agreement)
    strong_patterns = {k: v['mode'] for k, v in summary.items() if v['mode_pct'] >= 70}
    
    if strong_patterns:
        print("Strong patterns (>=70% agreement):")
        for k, v in strong_patterns.items():
            print(f"  {k} = {v}")
    else:
        print("No single configuration dominates the top tier (diverse winners).")
    
    return pd.DataFrame(summary).T


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(df: pd.DataFrame, 
                      metrics: Optional[Dict[str, str]] = None,
                      calibration_metrics: Optional[List[str]] = None,
                      borda_cutpoint: Optional[int] = None,
                      skip_friedman: bool = False,
                      top_n_friedman: int = 30,
                      top_n_profile: int = 10,
                      n_clusters: Optional[int] = None,
                      ) -> Dict:
    """
    Run the complete multi-criteria model selection pipeline.
    
    Args:
        df: DataFrame with models as rows
        metrics: Dict mapping metric name -> 'lower' or 'higher'
                 Default: DEFAULT_METRICS
        calibration_metrics: Metrics where closer to 1 is better
                             Default: CALIBRATION_METRICS
        borda_cutpoint: Manual cutpoint for Borda. If None, auto-detect.
        skip_friedman: If True, skip Friedman/Nemenyi and use pairwise dominance instead.
                       Recommended for >500 models.
        top_n_friedman: Number of models for Friedman test (ignored if skip_friedman)
        top_n_profile: Number of models for winner profile
        n_clusters: Number of configuration clusters. If None, auto-detect.
        
    Returns:
        Dict with results from each step
    """
    metrics = metrics or DEFAULT_METRICS
    calibration_metrics = calibration_metrics or CALIBRATION_METRICS
    
    # Filter to available metrics
    metrics = {k: v for k, v in metrics.items() if k in df.columns}
    calibration_metrics = [c for c in calibration_metrics if c in df.columns]
    
    print("="*72)
    print("MULTI-CRITERIA MODEL SELECTION PIPELINE")
    print("="*72)
    print(f"\nInput: {len(df)} models, {len(metrics)} metrics")
    print(f"Metrics: {list(metrics.keys())}")

    results = {}
    
    # Step 1: Borda
    print("\n" + "─"*72)
    print("STEP 1: BORDA COUNT RANKING")
    print("─"*72)
    df_borda, cutpoint = borda_rank(df, metrics, calibration_metrics)
    if borda_cutpoint is not None:
        cutpoint = borda_cutpoint
        print(f"\nUsing manual cutpoint: {cutpoint}")
    results['borda'] = (df_borda, cutpoint)
    
    # Get top tier for remaining analysis
    top_tier = df_borda.head(cutpoint).copy()
    print(f"\n→ Top tier: {cutpoint} models for remaining analysis")
    
    # Step 2: Pareto on top tier only
    print("\n" + "─"*72)
    print("STEP 2: PARETO DOMINANCE (Top Tier Only)")
    print("─"*72)
    pareto_df = pareto_frontier(top_tier, metrics, calibration_metrics)
    results['pareto'] = pareto_df
    
    # Step 3: Kendall Tau (on top tier)
    print("\n" + "─"*72)
    print("STEP 3: KENDALL TAU METRIC AGREEMENT (Top Tier)")
    print("─"*72)
    tau_df = kendall_tau_heatmap(top_tier, metrics, calibration_metrics)
    results['kendall_tau'] = tau_df
    
    # Step 4: Friedman OR Pairwise Dominance
    print("\n" + "─"*72)
    if skip_friedman:
        print("STEP 4: PAIRWISE DOMINANCE (replaces Friedman at scale)")
        print("─"*72)
        dominance_df = pairwise_dominance_summary(top_tier, metrics, calibration_metrics)
        results['pairwise_dominance'] = dominance_df
        results['friedman'] = (None, None, None)  # Placeholder
    else:
        print("STEP 4: FRIEDMAN + NEMENYI TEST")
        print("─"*72)
        stat, pval, nemenyi = friedman_nemenyi(
            df_borda, metrics, calibration_metrics, 
            top_n=min(top_n_friedman, cutpoint)
        )
        results['friedman'] = (stat, pval, nemenyi)
        results['pairwise_dominance'] = None
    
    # Step 5: TOPSIS on top tier only
    print("\n" + "─"*72)
    print("STEP 5: TOPSIS RANKING (Top Tier Only)")
    print("─"*72)
    topsis_df, influence = topsis_rank(top_tier, metrics, calibration_metrics)
    results['topsis'] = (topsis_df, influence)
    
    # Step 6: Configuration clustering
    print("\n" + "─"*72)
    print("STEP 6: CONFIGURATION CLUSTERING")
    print("─"*72)
    clustered_df, cluster_summary = cluster_configurations(
        topsis_df, n_clusters=n_clusters
    )
    results['clusters'] = (clustered_df, cluster_summary)
    
    # Step 7: Winner profile
    print("\n" + "─"*72)
    print("STEP 7: WINNER PROFILE")
    print("─"*72)
    profile = winner_profile(clustered_df, top_n=top_n_profile)
    results['profile'] = profile
    
    print("\n" + "="*72)
    print("PIPELINE COMPLETE")
    print("="*72)
    
    return results
