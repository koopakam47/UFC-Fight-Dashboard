"""
Data loading and preprocessing utilities for UFC fight analytics.

This module handles loading raw UFC fight data, cleaning it, and transforming
it into a format suitable for analysis.
"""

import os
import pandas as pd
import logging
from typing import Optional

from .config import PATHS


logger = logging.getLogger(__name__)


def load_fighter_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Load and clean UFC fighter data.
    
    Args:
        force_reload (bool): Force reload from raw data even if cleaned data exists
        
    Returns:
        pd.DataFrame: Cleaned fighter data in long format
    """
    cleaned_path = PATHS['cleaned_data']
    raw_path = PATHS['raw_data']
    
    # Check if cleaned data exists and we're not forcing reload
    if os.path.exists(cleaned_path) and not force_reload:
        logger.info("Loading cleaned dataset from %s", cleaned_path)
        return pd.read_csv(cleaned_path)
    
    logger.info("Cleaned dataset not found or force_reload=True. Loading and cleaning raw data...")
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")
    
    # Load raw data
    df_raw = pd.read_csv(raw_path)
    logger.info("Loaded raw data with shape: %s", df_raw.shape)
    
    # Clean the data
    df_cleaned = clean_fight_data(df_raw)
    
    # Save cleaned data
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    df_cleaned.to_csv(cleaned_path, index=False)
    logger.info("Cleaned dataset saved to %s", cleaned_path)
    
    return df_cleaned


def clean_fight_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw fight data and convert to long format.
    
    Args:
        df_raw (pd.DataFrame): Raw fight data in wide format
        
    Returns:
        pd.DataFrame: Cleaned data in long format with one row per fighter per fight
    """
    logger.info("Starting data cleaning process...")
    
    # Fix column names (remove special characters)
    df_raw.columns = [col.replace('≤', '') for col in df_raw.columns]
    
    # Split into red and blue fighter data
    red_data = _extract_fighter_data(df_raw, 'R_', 'red')
    blue_data = _extract_fighter_data(df_raw, 'B_', 'blue')
    
    # Combine red and blue data
    df_long = pd.concat([red_data, blue_data], ignore_index=True)
    logger.info("Combined data shape after splitting: %s", df_long.shape)
    
    # Add outcome variable
    df_long = _add_outcome_variable(df_long)
    
    # Clean data types
    df_long = _clean_data_types(df_long)
    
    logger.info("Data cleaning completed. Final shape: %s", df_long.shape)
    return df_long


def _extract_fighter_data(df_raw: pd.DataFrame, prefix: str, corner: str) -> pd.DataFrame:
    """
    Extract fighter data for one corner (red or blue).
    
    Args:
        df_raw (pd.DataFrame): Raw fight data
        prefix (str): Column prefix ('R_' or 'B_')
        corner (str): Corner designation ('red' or 'blue')
        
    Returns:
        pd.DataFrame: Fighter data for specified corner
    """
    # Get fighter-specific columns
    fighter_cols = df_raw.filter(like=prefix).copy()
    
    # Remove prefix from column names
    fighter_cols.columns = fighter_cols.columns.str.replace(prefix, '', regex=False)
    
    # Add corner designation
    fighter_cols['corner'] = corner
    
    # Add shared columns (fight-level data)
    shared_cols = ['Winner', 'title_bout', 'weight_class', 'Referee', 'date', 'location']
    for col in shared_cols:
        if col in df_raw.columns:
            fighter_cols[col] = df_raw[col]
    
    return fighter_cols


def _add_outcome_variable(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary outcome variable indicating if fighter won.
    
    Args:
        df_long (pd.DataFrame): Long format fighter data
        
    Returns:
        pd.DataFrame: Data with 'won' column added
    """
    # Create binary outcome variable
    df_long['won'] = (
        ((df_long['corner'] == 'red') & (df_long['Winner'] == 'Red')) |
        ((df_long['corner'] == 'blue') & (df_long['Winner'] == 'Blue'))
    ).astype(int)
    
    return df_long


def _clean_data_types(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and optimize data types.
    
    Args:
        df_long (pd.DataFrame): Fighter data
        
    Returns:
        pd.DataFrame: Data with optimized types
    """
    # Convert boolean columns
    if 'title_bout' in df_long.columns:
        df_long['title_bout'] = df_long['title_bout'].astype(bool)
    
    # Convert date column
    if 'date' in df_long.columns:
        df_long['date'] = pd.to_datetime(df_long['date'], errors='coerce')
    
    # Ensure numeric columns are properly typed
    numeric_cols = df_long.select_dtypes(include=['object']).columns
    for col in numeric_cols:
        if col not in ['fighter', 'Stance', 'corner', 'Winner', 'weight_class', 'Referee', 'location']:
            df_long[col] = pd.to_numeric(df_long[col], errors='coerce')
    
    return df_long


def filter_data_by_weight_class(df: pd.DataFrame, weight_class: str) -> pd.DataFrame:
    """
    Filter data for a specific weight class.
    
    Args:
        df (pd.DataFrame): Fighter data
        weight_class (str): Weight class name
        
    Returns:
        pd.DataFrame: Filtered data for specified weight class
    """
    if 'weight_class' not in df.columns:
        raise ValueError("DataFrame must contain 'weight_class' column")
    
    filtered_df = df[df['weight_class'] == weight_class].copy()
    logger.info("Filtered data for %s: %d records", weight_class, len(filtered_df))
    
    return filtered_df


def get_available_weight_classes(df: pd.DataFrame) -> list:
    """
    Get list of available weight classes in the data.
    
    Args:
        df (pd.DataFrame): Fighter data
        
    Returns:
        list: Available weight classes
    """
    if 'weight_class' not in df.columns:
        return []
    
    weight_classes = df['weight_class'].dropna().unique().tolist()
    weight_classes.sort()
    
    return weight_classes


def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Validate data quality and return summary statistics.
    
    Args:
        df (pd.DataFrame): Fighter data to validate
        
    Returns:
        dict: Data quality summary
    """
    quality_summary = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicate_records': df.duplicated().sum(),
        'weight_classes': len(df['weight_class'].unique()) if 'weight_class' in df.columns else 0,
        'date_range': None
    }
    
    if 'date' in df.columns:
        date_col = pd.to_datetime(df['date'], errors='coerce')
        quality_summary['date_range'] = {
            'start': date_col.min(),
            'end': date_col.max()
        }
    
    return quality_summary