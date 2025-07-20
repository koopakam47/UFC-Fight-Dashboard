"""
Correlation analysis utilities for UFC fight analytics.

This module provides functions for computing correlations between fight statistics
and win outcomes, with statistical validation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List, Optional
from scipy.stats import pearsonr
from scipy import stats

from .config import DATA_CONFIG, NON_COMBAT_COLUMNS


logger = logging.getLogger(__name__)


def compute_correlations_by_weight_class(
    df: pd.DataFrame, 
    target_col: str = 'won',
    min_sample_size: Optional[int] = None
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Compute correlations between fight statistics and win outcome for each weight class.
    
    Args:
        df (pd.DataFrame): Fighter data
        target_col (str): Target column for correlation (default: 'won')
        min_sample_size (int, optional): Minimum sample size per weight class
        
    Returns:
        Dict: Nested dictionary with weight class -> correlation type -> results
    """
    if min_sample_size is None:
        min_sample_size = DATA_CONFIG['min_sample_size']
    
    results = {}
    weight_classes = df['weight_class'].dropna().unique()
    
    for weight_class in weight_classes:
        logger.info("Computing correlations for weight class: %s", weight_class)
        
        subset = df[df['weight_class'] == weight_class].copy()
        
        if len(subset) < min_sample_size:
            logger.warning(
                "Skipping %s: insufficient sample size (%d < %d)", 
                weight_class, len(subset), min_sample_size
            )
            continue
        
        # Ensure target column exists and is numeric
        if target_col not in subset.columns:
            logger.warning("Target column '%s' not found in %s data", target_col, weight_class)
            continue
        
        subset[target_col] = pd.to_numeric(subset[target_col], errors='coerce')
        
        # Compute different types of correlations
        weight_results = {}
        
        # Overall correlations
        overall_corr = _compute_numeric_correlations(subset, target_col)
        if len(overall_corr) > 0:
            weight_results['top_10_overall'] = overall_corr.sort_values(ascending=False).head(10)
            weight_results['bottom_10_overall'] = overall_corr.sort_values(ascending=True).head(10)
        
        # Non-combat correlations
        non_combat_corr = _compute_non_combat_correlations(subset, target_col)
        if len(non_combat_corr) > 0:
            weight_results['non_combat'] = non_combat_corr.sort_values(ascending=False)
        
        # Stance analysis
        stance_stats = _compute_stance_win_rates(subset, target_col)
        if len(stance_stats) > 0:
            weight_results['stance_win_rates'] = stance_stats
        
        results[weight_class] = weight_results
    
    return results


def _compute_numeric_correlations(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Compute correlations for all numeric columns.
    
    Args:
        df (pd.DataFrame): Data subset
        target_col (str): Target column
        
    Returns:
        pd.Series: Correlation values
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Remove target column from features
    feature_cols = numeric_cols.drop(target_col, errors='ignore')
    
    # Compute correlations
    correlations = {}
    for col in feature_cols:
        try:
            # Remove rows where either column is NaN
            valid_data = df[[col, target_col]].dropna()
            
            if len(valid_data) < 10:  # Need minimum data points
                continue
                
            corr_val, p_value = pearsonr(valid_data[col], valid_data[target_col])
            
            # Only include statistically significant correlations
            if not np.isnan(corr_val) and not np.isnan(p_value):
                correlations[col] = corr_val
                
        except Exception as e:
            logger.debug("Error computing correlation for %s: %s", col, str(e))
            continue
    
    return pd.Series(correlations)


def _compute_non_combat_correlations(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Compute correlations for non-combat related columns.
    
    Args:
        df (pd.DataFrame): Data subset
        target_col (str): Target column
        
    Returns:
        pd.Series: Non-combat correlations
    """
    # Filter for non-combat columns that exist in the data
    available_non_combat = [col for col in NON_COMBAT_COLUMNS if col in df.columns]
    
    if not available_non_combat:
        return pd.Series(dtype=float)
    
    # Compute correlations for non-combat columns
    non_combat_data = df[available_non_combat + [target_col]].copy()
    
    correlations = {}
    for col in available_non_combat:
        try:
            valid_data = non_combat_data[[col, target_col]].dropna()
            
            if len(valid_data) < 10:
                continue
                
            corr_val, p_value = pearsonr(valid_data[col], valid_data[target_col])
            
            if not np.isnan(corr_val) and not np.isnan(p_value):
                correlations[col] = corr_val
                
        except Exception as e:
            logger.debug("Error computing non-combat correlation for %s: %s", col, str(e))
            continue
    
    return pd.Series(correlations)


def _compute_stance_win_rates(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Compute win rates by fighting stance.
    
    Args:
        df (pd.DataFrame): Data subset
        target_col (str): Target column
        
    Returns:
        pd.Series: Win rates by stance
    """
    if 'Stance' not in df.columns:
        return pd.Series(dtype=float)
    
    # Group by stance and compute win rate
    stance_stats = df.groupby('Stance')[target_col].agg(['mean', 'count']).reset_index()
    
    # Filter out stances with very few observations
    stance_stats = stance_stats[stance_stats['count'] >= 10]
    
    # Return as series with stance as index
    return stance_stats.set_index('Stance')['mean'].sort_values(ascending=False)


def validate_correlation_significance(
    df: pd.DataFrame, 
    col1: str, 
    col2: str, 
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Validate statistical significance of correlation between two columns.
    
    Args:
        df (pd.DataFrame): Data
        col1 (str): First column
        col2 (str): Second column
        alpha (float): Significance level
        
    Returns:
        Tuple[float, float, bool]: (correlation, p_value, is_significant)
    """
    # Remove rows with missing values
    valid_data = df[[col1, col2]].dropna()
    
    if len(valid_data) < 10:
        return np.nan, np.nan, False
    
    try:
        correlation, p_value = pearsonr(valid_data[col1], valid_data[col2])
        is_significant = p_value < alpha
        
        return correlation, p_value, is_significant
        
    except Exception as e:
        logger.error("Error validating correlation between %s and %s: %s", col1, col2, str(e))
        return np.nan, np.nan, False


def compute_correlation_confidence_intervals(
    df: pd.DataFrame, 
    feature_col: str, 
    target_col: str, 
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for correlation coefficient.
    
    Args:
        df (pd.DataFrame): Data
        feature_col (str): Feature column
        target_col (str): Target column
        confidence (float): Confidence level
        
    Returns:
        Tuple[float, float, float]: (correlation, lower_bound, upper_bound)
    """
    valid_data = df[[feature_col, target_col]].dropna()
    
    if len(valid_data) < 10:
        return np.nan, np.nan, np.nan
    
    try:
        correlation, _ = pearsonr(valid_data[feature_col], valid_data[target_col])
        n = len(valid_data)
        
        # Fisher's z-transformation for confidence interval
        z = 0.5 * np.log((1 + correlation) / (1 - correlation))
        z_alpha = stats.norm.ppf(1 - (1 - confidence) / 2)
        z_se = 1 / np.sqrt(n - 3)
        
        # Confidence interval for z
        z_lower = z - z_alpha * z_se
        z_upper = z + z_alpha * z_se
        
        # Transform back to correlation scale
        lower_bound = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        upper_bound = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return correlation, lower_bound, upper_bound
        
    except Exception as e:
        logger.error("Error computing confidence interval for %s vs %s: %s", feature_col, target_col, str(e))
        return np.nan, np.nan, np.nan


def summarize_correlation_results(correlation_results: Dict) -> Dict[str, Dict]:
    """
    Create summary statistics for correlation analysis results.
    
    Args:
        correlation_results (Dict): Results from compute_correlations_by_weight_class
        
    Returns:
        Dict: Summary statistics
    """
    summary = {}
    
    for weight_class, results in correlation_results.items():
        weight_summary = {
            'sample_size': 0,
            'top_positive_correlation': np.nan,
            'top_negative_correlation': np.nan,
            'significant_correlations_count': 0,
            'stance_with_highest_win_rate': None
        }
        
        # Top correlations
        if 'top_10_overall' in results and len(results['top_10_overall']) > 0:
            weight_summary['top_positive_correlation'] = results['top_10_overall'].iloc[0]
            
        if 'bottom_10_overall' in results and len(results['bottom_10_overall']) > 0:
            weight_summary['top_negative_correlation'] = results['bottom_10_overall'].iloc[0]
        
        # Count significant correlations (arbitrary threshold of 0.1)
        if 'top_10_overall' in results:
            significant_corr = results['top_10_overall'][abs(results['top_10_overall']) > 0.1]
            weight_summary['significant_correlations_count'] = len(significant_corr)
        
        # Top stance
        if 'stance_win_rates' in results and len(results['stance_win_rates']) > 0:
            weight_summary['stance_with_highest_win_rate'] = results['stance_win_rates'].index[0]
        
        summary[weight_class] = weight_summary
    
    return summary