"""
Legacy data loading script - refactored to use new modular architecture.

This script provides backward compatibility while leveraging the new
UFC analytics package for improved functionality.
"""

import os
import sys

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ufc_analytics.data_loader import load_fighter_data as load_data_new
from ufc_analytics.logging_utils import setup_logging, log_data_summary


def load_fighter_data(force_reload: bool = False):
    """
    Load and clean UFC fighter data.
    
    This function maintains backward compatibility with the original interface
    while using the new modular data loading system.
    
    Args:
        force_reload (bool): Force reload from raw data even if cleaned data exists
        
    Returns:
        pd.DataFrame: Cleaned fighter data in long format
    """
    # Set up logging
    setup_logging(console_output=True)
    
    # Use the new modular data loader
    df = load_data_new(force_reload=force_reload)
    
    # Log summary for user feedback
    log_data_summary(df, "UFC Fighter Data")
    
    return df


if __name__ == "__main__":
    # Load data when script is run directly
    data = load_fighter_data()
    print(f"✅ Loaded dataset with shape: {data.shape}")
    print(f"📊 Available weight classes: {sorted(data['weight_class'].unique())}")
    
    # Check if date column exists before displaying range
    if 'date' in data.columns:
        print(f"📅 Date range: {data['date'].min()} to {data['date'].max()}")
    else:
        print("📅 Date information not available in processed data")
