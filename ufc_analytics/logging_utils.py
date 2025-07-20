"""
Logging utilities for UFC fight analytics.

This module sets up consistent logging across the analytics pipeline.
"""

import logging
import os
from typing import Optional

from .config import LOGGING_CONFIG, PATHS


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> None:
    """
    Set up logging configuration for the analytics pipeline.
    
    Args:
        log_level (str, optional): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file (str, optional): Path to log file
        console_output (bool): Whether to output logs to console
    """
    # Use config defaults if not specified
    if log_level is None:
        log_level = LOGGING_CONFIG['level']
    
    if log_file is None:
        log_file = LOGGING_CONFIG['log_file']
    
    # Create output directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(LOGGING_CONFIG['format'])
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set up module-specific loggers
    for module in ['ufc_analytics', 'ufc_analytics.data_loader', 
                   'ufc_analytics.correlation_analysis', 'ufc_analytics.visualization']:
        logger = logging.getLogger(module)
        logger.setLevel(getattr(logging, log_level.upper()))


def log_data_summary(df, description: str = "Dataset") -> None:
    """
    Log a summary of a DataFrame.
    
    Args:
        df: pandas DataFrame to summarize
        description (str): Description of the dataset
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=== %s Summary ===", description)
    logger.info("Shape: %s", df.shape)
    logger.info("Columns: %d", len(df.columns))
    logger.info("Memory usage: %.2f MB", df.memory_usage(deep=True).sum() / 1024**2)
    
    if 'weight_class' in df.columns:
        weight_classes = df['weight_class'].value_counts()
        logger.info("Weight classes: %s", dict(weight_classes.head()))
    
    # Log missing data
    missing_data = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    missing_pct = (missing_data / total_cells) * 100
    logger.info("Missing data: %d cells (%.2f%%)", missing_data, missing_pct)


def log_correlation_summary(correlation_results: dict) -> None:
    """
    Log a summary of correlation analysis results.
    
    Args:
        correlation_results (dict): Results from correlation analysis
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=== Correlation Analysis Summary ===")
    logger.info("Weight classes analyzed: %d", len(correlation_results))
    
    for weight_class, results in correlation_results.items():
        logger.info("--- %s ---", weight_class)
        
        if 'top_10_overall' in results:
            top_corr = results['top_10_overall'].iloc[0] if len(results['top_10_overall']) > 0 else 0
            logger.info("  Top correlation: %.3f", top_corr)
        
        if 'stance_win_rates' in results and len(results['stance_win_rates']) > 0:
            best_stance = results['stance_win_rates'].index[0]
            best_rate = results['stance_win_rates'].iloc[0]
            logger.info("  Best stance: %s (%.3f win rate)", best_stance, best_rate)


def log_processing_step(step_name: str, start_time: Optional[float] = None) -> None:
    """
    Log the start or completion of a processing step.
    
    Args:
        step_name (str): Name of the processing step
        start_time (float, optional): Start time for calculating duration
    """
    logger = logging.getLogger(__name__)
    
    if start_time is None:
        logger.info("Starting: %s", step_name)
    else:
        import time
        duration = time.time() - start_time
        logger.info("Completed: %s (%.2f seconds)", step_name, duration)


def log_error_with_context(error: Exception, context: str) -> None:
    """
    Log an error with additional context.
    
    Args:
        error (Exception): The exception that occurred
        context (str): Additional context about where the error occurred
    """
    logger = logging.getLogger(__name__)
    logger.error("Error in %s: %s (%s)", context, str(error), type(error).__name__)
    logger.debug("Full traceback:", exc_info=True)