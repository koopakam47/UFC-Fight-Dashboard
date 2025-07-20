"""
Configuration module for UFC fight analytics.

Contains all styling, path, and display settings used throughout the analysis.
"""

import os

# Chart styling configuration
CHART_CONFIG = {
    'figsize': (16, 12),
    'style': 'whitegrid',
    'palette': 'viridis',
    'title_fontsize': 14,
    'label_fontsize': 12,
    'tick_fontsize': 10,
    'dpi': 300
}

# Color palettes for different chart types
COLOR_PALETTES = {
    'correlations_positive': 'viridis',
    'correlations_negative': 'plasma_r',
    'stance_comparison': 'Set2',
    'weight_class': 'tab10'
}

# Path configuration
PATHS = {
    'data_dir': 'data',
    'raw_data': os.path.join('data', 'data.csv'),
    'cleaned_data': os.path.join('data', 'cleaned_fighter_stats.csv'),
    'visualizations_dir': 'visualizations',
    'notebooks_dir': 'notebooks',
    'output_dir': 'output'
}

# Data processing configuration
DATA_CONFIG = {
    'min_sample_size': 100,  # Minimum number of fights per weight class for analysis
    'correlation_threshold': 0.05,  # Minimum correlation to consider significant
    'top_n_correlations': 10  # Number of top/bottom correlations to display
}

# Human-readable column name mappings
READABLE_NAMES = {
    'won': 'Win Indicator',
    'age': 'Age',
    'Height_cms': 'Height (cm)',
    'Reach_cms': 'Reach (cm)',
    'Weight_lbs': 'Weight (lbs)',
    'total_rounds_fought': 'Total Rounds Fought',
    'total_title_bouts': 'Total Title Bouts',
    'total_time_fought(seconds)': 'Total Fight Time (seconds)',
    'wins': 'Career Wins',
    'losses': 'Career Losses',
    'draw': 'Career Draws',
    'current_win_streak': 'Current Win Streak',
    'current_lose_streak': 'Current Lose Streak',
    'longest_win_streak': 'Longest Win Streak',
    'win_by_Submission': 'Wins by Submission',
    'win_by_KO/TKO': 'Wins by KO/TKO',
    'win_by_Decision_Unanimous': 'Wins by Unanimous Decision',
    'win_by_Decision_Majority': 'Wins by Majority Decision',
    'win_by_Decision_Split': 'Wins by Split Decision',
    'win_by_TKO_Doctor_Stoppage': 'Wins by TKO Doctor Stoppage',
    
    # Statistical abbreviations
    'avg_KD': 'Average Knockdowns',
    'avg_opp_KD': 'Average Opponent Knockdowns',
    'avg_SIG_STR_pct': 'Average Significant Strike %',
    'avg_opp_SIG_STR_pct': 'Average Opponent Significant Strike %',
    'avg_TD_pct': 'Average Takedown %',
    'avg_opp_TD_pct': 'Average Opponent Takedown %',
    'avg_SUB_ATT': 'Average Submission Attempts',
    'avg_opp_SUB_ATT': 'Average Opponent Submission Attempts',
    'avg_REV': 'Average Reversals',
    'avg_opp_REV': 'Average Opponent Reversals',
    'avg_SIG_STR_landed': 'Average Significant Strikes Landed',
    'avg_SIG_STR_att': 'Average Significant Strikes Attempted',
    'avg_TOTAL_STR_landed': 'Average Total Strikes Landed',
    'avg_TOTAL_STR_att': 'Average Total Strikes Attempted',
    'avg_TD_landed': 'Average Takedowns Landed',
    'avg_TD_att': 'Average Takedowns Attempted',
    'avg_HEAD_landed': 'Average Head Strikes Landed',
    'avg_HEAD_att': 'Average Head Strikes Attempted',
    'avg_BODY_landed': 'Average Body Strikes Landed',
    'avg_BODY_att': 'Average Body Strikes Attempted',
    'avg_LEG_landed': 'Average Leg Strikes Landed',
    'avg_LEG_att': 'Average Leg Strikes Attempted',
    'avg_DISTANCE_landed': 'Average Distance Strikes Landed',
    'avg_DISTANCE_att': 'Average Distance Strikes Attempted',
    'avg_CLINCH_landed': 'Average Clinch Strikes Landed',
    'avg_CLINCH_att': 'Average Clinch Strikes Attempted',
    'avg_GROUND_landed': 'Average Ground Strikes Landed',
    'avg_GROUND_att': 'Average Ground Strikes Attempted',
    'avg_CTRL_time(seconds)': 'Average Control Time (seconds)',
    'Stance': 'Fighting Stance'
}

# Non-combat related columns for analysis
NON_COMBAT_COLUMNS = [
    'age', 'Height_cms', 'Reach_cms', 'Weight_lbs',
    'wins', 'losses', 'draw',
    'current_win_streak', 'current_lose_streak', 'longest_win_streak',
    'total_rounds_fought', 'total_time_fought(seconds)', 'total_title_bouts',
    'win_by_Submission', 'win_by_KO/TKO', 'win_by_Decision_Unanimous',
    'win_by_Decision_Majority', 'win_by_Decision_Split', 'win_by_TKO_Doctor_Stoppage'
]

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': os.path.join('output', 'ufc_analytics.log')
}


def get_readable_name(column_name):
    """
    Get human-readable name for a column, with automatic formatting fallback.
    
    Args:
        column_name (str): Original column name
        
    Returns:
        str: Human-readable column name
    """
    if column_name in READABLE_NAMES:
        return READABLE_NAMES[column_name]
    
    # Auto-format unmapped columns
    readable = column_name.replace('_', ' ')
    readable = readable.replace('avg ', '')
    readable = readable.replace('opp ', 'Opponent ')
    readable = readable.replace('TD', 'Takedown')
    readable = readable.replace('KD', 'Knockdown')
    readable = readable.replace('STR', 'Strike')
    readable = readable.replace('SUB ATT', 'Submission Attempt')
    readable = readable.replace('SIG', 'Significant')
    readable = readable.replace('pct', '%')
    readable = readable.replace('att', 'Attempted')
    readable = readable.replace('landed', 'Landed')
    
    return readable.title()


def ensure_directories():
    """Create necessary directories if they don't exist."""
    for path_key, path_value in PATHS.items():
        if path_key.endswith('_dir'):
            os.makedirs(path_value, exist_ok=True)