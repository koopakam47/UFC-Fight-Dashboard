"""
Visualization utilities for UFC fight analytics.

This module provides functions for generating consistent, publication-ready
charts and dashboards for correlation analysis results.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple

from .config import CHART_CONFIG, COLOR_PALETTES, PATHS, get_readable_name


logger = logging.getLogger(__name__)


def setup_matplotlib_style():
    """Configure matplotlib with consistent styling."""
    sns.set_style(CHART_CONFIG['style'])
    plt.rcParams.update({
        'figure.figsize': CHART_CONFIG['figsize'],
        'figure.dpi': CHART_CONFIG['dpi'],
        'font.size': CHART_CONFIG['label_fontsize'],
        'axes.titlesize': CHART_CONFIG['title_fontsize'],
        'axes.labelsize': CHART_CONFIG['label_fontsize'],
        'xtick.labelsize': CHART_CONFIG['tick_fontsize'],
        'ytick.labelsize': CHART_CONFIG['tick_fontsize'],
        'legend.fontsize': CHART_CONFIG['tick_fontsize'],
    })


def generate_correlation_dashboard(
    correlation_results: Dict,
    weight_class: str,
    save_path: Optional[str] = None
) -> str:
    """
    Generate a complete correlation dashboard for a weight class.
    
    Args:
        correlation_results (Dict): Results from correlation analysis
        weight_class (str): Weight class name
        save_path (str, optional): Path to save the dashboard
        
    Returns:
        str: Path where dashboard was saved
    """
    setup_matplotlib_style()
    
    if weight_class not in correlation_results:
        raise ValueError(f"No results found for weight class: {weight_class}")
    
    results = correlation_results[weight_class]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=CHART_CONFIG['figsize'])
    fig.suptitle(f'{weight_class} - UFC Fight Analytics Dashboard', 
                 fontsize=CHART_CONFIG['title_fontsize'] + 2, fontweight='bold')
    
    # 1. Top 10 Correlations
    if 'top_10_overall' in results:
        _plot_correlation_bars(
            axes[0, 0], 
            results['top_10_overall'], 
            'Top 10 Correlations with Victory',
            COLOR_PALETTES['correlations_positive']
        )
    
    # 2. Bottom 10 Correlations  
    if 'bottom_10_overall' in results:
        _plot_correlation_bars(
            axes[0, 1], 
            results['bottom_10_overall'], 
            'Bottom 10 Correlations with Victory',
            COLOR_PALETTES['correlations_negative']
        )
    
    # 3. Non-Combat Correlations
    if 'non_combat' in results:
        _plot_correlation_bars(
            axes[1, 0], 
            results['non_combat'], 
            'Non-Combat Factor Correlations',
            'coolwarm',
            limit=8
        )
    
    # 4. Stance Win Rates
    if 'stance_win_rates' in results:
        _plot_stance_win_rates(
            axes[1, 1], 
            results['stance_win_rates'], 
            'Win Rates by Fighting Stance'
        )
    
    plt.tight_layout()
    
    # Save the dashboard
    if save_path is None:
        os.makedirs(PATHS['visualizations_dir'], exist_ok=True)
        save_path = os.path.join(PATHS['visualizations_dir'], f'{weight_class}_correlation_dashboard.png')
    
    plt.savefig(save_path, dpi=CHART_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    logger.info("Dashboard saved for %s: %s", weight_class, save_path)
    return save_path


def _plot_correlation_bars(
    ax: plt.Axes, 
    correlations: pd.Series, 
    title: str, 
    palette: str,
    limit: Optional[int] = None
) -> None:
    """
    Plot correlation bars on given axes.
    
    Args:
        ax (plt.Axes): Matplotlib axes
        correlations (pd.Series): Correlation values
        title (str): Plot title
        palette (str): Color palette
        limit (int, optional): Limit number of bars shown
    """
    if len(correlations) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Limit the number of correlations if specified
    if limit and len(correlations) > limit:
        correlations = correlations.head(limit)
    
    # Get readable names for display
    readable_names = [get_readable_name(col) for col in correlations.index]
    
    # Create horizontal bar plot
    colors = sns.color_palette(palette, len(correlations))
    bars = ax.barh(range(len(correlations)), correlations.values, color=colors)
    
    # Customize the plot
    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels(readable_names, fontsize=CHART_CONFIG['tick_fontsize'])
    ax.set_xlabel('Correlation with Victory', fontsize=CHART_CONFIG['label_fontsize'])
    ax.set_title(title, fontsize=CHART_CONFIG['title_fontsize'], fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, correlations.values)):
        ax.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}', 
                va='center', ha='left' if value >= 0 else 'right',
                fontsize=CHART_CONFIG['tick_fontsize'])
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Invert y-axis to show highest values at top
    ax.invert_yaxis()
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')


def _plot_stance_win_rates(ax: plt.Axes, win_rates: pd.Series, title: str) -> None:
    """
    Plot win rates by stance.
    
    Args:
        ax (plt.Axes): Matplotlib axes
        win_rates (pd.Series): Win rates by stance
        title (str): Plot title
    """
    if len(win_rates) == 0:
        ax.text(0.5, 0.5, 'No stance data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Create bar plot
    colors = sns.color_palette(COLOR_PALETTES['stance_comparison'], len(win_rates))
    bars = ax.bar(range(len(win_rates)), win_rates.values, color=colors)
    
    # Customize the plot
    ax.set_xticks(range(len(win_rates)))
    ax.set_xticklabels(win_rates.index, rotation=45, ha='right')
    ax.set_ylabel('Win Rate', fontsize=CHART_CONFIG['label_fontsize'])
    ax.set_title(title, fontsize=CHART_CONFIG['title_fontsize'], fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, win_rates.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{value:.3f}',
                ha='center', va='bottom', fontsize=CHART_CONFIG['tick_fontsize'])
    
    # Add horizontal line at 0.5 (50% win rate)
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='50% Win Rate')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')


def generate_summary_chart(correlation_results: Dict, save_path: Optional[str] = None) -> str:
    """
    Generate a summary chart comparing key metrics across weight classes.
    
    Args:
        correlation_results (Dict): Results from correlation analysis
        save_path (str, optional): Path to save the chart
        
    Returns:
        str: Path where chart was saved
    """
    setup_matplotlib_style()
    
    # Extract data for summary
    weight_classes = []
    top_correlations = []
    top_stances = []
    
    for weight_class, results in correlation_results.items():
        weight_classes.append(weight_class)
        
        # Get top positive correlation
        if 'top_10_overall' in results and len(results['top_10_overall']) > 0:
            top_correlations.append(results['top_10_overall'].iloc[0])
        else:
            top_correlations.append(0)
        
        # Get top stance
        if 'stance_win_rates' in results and len(results['stance_win_rates']) > 0:
            top_stances.append(results['stance_win_rates'].index[0])
        else:
            top_stances.append('Unknown')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('UFC Analytics Summary - All Weight Classes', 
                 fontsize=CHART_CONFIG['title_fontsize'] + 2, fontweight='bold')
    
    # Plot 1: Top correlations by weight class
    colors1 = sns.color_palette(COLOR_PALETTES['weight_class'], len(weight_classes))
    bars1 = ax1.bar(range(len(weight_classes)), top_correlations, color=colors1)
    ax1.set_xticks(range(len(weight_classes)))
    ax1.set_xticklabels(weight_classes, rotation=45, ha='right')
    ax1.set_ylabel('Highest Positive Correlation')
    ax1.set_title('Strongest Positive Correlation by Weight Class')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, top_correlations):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{value:.3f}',
                ha='center', va='bottom', fontsize=CHART_CONFIG['tick_fontsize'])
    
    # Plot 2: Stance distribution
    stance_counts = pd.Series(top_stances).value_counts()
    colors2 = sns.color_palette(COLOR_PALETTES['stance_comparison'], len(stance_counts))
    ax2.pie(stance_counts.values, labels=stance_counts.index, autopct='%1.1f%%',
            colors=colors2, startangle=90)
    ax2.set_title('Most Successful Stance by Weight Class')
    
    plt.tight_layout()
    
    # Save the chart
    if save_path is None:
        os.makedirs(PATHS['visualizations_dir'], exist_ok=True)
        save_path = os.path.join(PATHS['visualizations_dir'], 'summary_all_weight_classes.png')
    
    plt.savefig(save_path, dpi=CHART_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    logger.info("Summary chart saved: %s", save_path)
    return save_path


def generate_all_dashboards(correlation_results: Dict) -> List[str]:
    """
    Generate dashboards for all weight classes.
    
    Args:
        correlation_results (Dict): Results from correlation analysis
        
    Returns:
        List[str]: List of paths where dashboards were saved
    """
    saved_paths = []
    
    logger.info("Generating dashboards for %d weight classes", len(correlation_results))
    
    for weight_class in correlation_results.keys():
        try:
            path = generate_correlation_dashboard(correlation_results, weight_class)
            saved_paths.append(path)
        except Exception as e:
            logger.error("Failed to generate dashboard for %s: %s", weight_class, str(e))
    
    # Generate summary chart
    try:
        summary_path = generate_summary_chart(correlation_results)
        saved_paths.append(summary_path)
    except Exception as e:
        logger.error("Failed to generate summary chart: %s", str(e))
    
    logger.info("Generated %d dashboard files", len(saved_paths))
    return saved_paths


def create_custom_correlation_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    save_path: Optional[str] = None
) -> str:
    """
    Create a custom scatter plot showing correlation between two variables.
    
    Args:
        df (pd.DataFrame): Data
        x_col (str): X-axis column
        y_col (str): Y-axis column  
        title (str): Plot title
        save_path (str, optional): Path to save the plot
        
    Returns:
        str: Path where plot was saved
    """
    setup_matplotlib_style()
    
    # Remove missing values
    plot_data = df[[x_col, y_col]].dropna()
    
    if len(plot_data) < 10:
        raise ValueError(f"Insufficient data for correlation plot: {len(plot_data)} observations")
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(plot_data[x_col], plot_data[y_col], alpha=0.6, color='steelblue')
    
    # Add trend line
    z = np.polyfit(plot_data[x_col], plot_data[y_col], 1)
    p = np.poly1d(z)
    ax.plot(plot_data[x_col], p(plot_data[x_col]), "r--", alpha=0.8)
    
    # Calculate correlation
    correlation = plot_data[x_col].corr(plot_data[y_col])
    
    # Labels and title
    ax.set_xlabel(get_readable_name(x_col))
    ax.set_ylabel(get_readable_name(y_col))
    ax.set_title(f'{title}\n(Correlation: {correlation:.3f})')
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    if save_path is None:
        os.makedirs(PATHS['visualizations_dir'], exist_ok=True)
        safe_title = title.replace(' ', '_').replace('/', '_')
        save_path = os.path.join(PATHS['visualizations_dir'], f'correlation_{safe_title}.png')
    
    plt.savefig(save_path, dpi=CHART_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    logger.info("Custom correlation plot saved: %s", save_path)
    return save_path