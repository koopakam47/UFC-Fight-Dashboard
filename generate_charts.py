#!/usr/bin/env python3
"""
Chart generation utility for UFC fight analytics.

This script generates correlation dashboards for all weight classes and creates
summary visualizations. It provides comprehensive logging and error handling.
"""

import os
import sys
import time
import argparse
from typing import List, Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ufc_analytics.data_loader import load_fighter_data
from ufc_analytics.correlation_analysis import compute_correlations_by_weight_class
from ufc_analytics.visualization import generate_all_dashboards, setup_matplotlib_style
from ufc_analytics.logging_utils import (
    setup_logging, log_data_summary, log_correlation_summary, 
    log_processing_step, log_error_with_context
)
from ufc_analytics.config import ensure_directories


def generate_all_charts(
    force_reload: bool = False,
    weight_classes: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> List[str]:
    """
    Generate correlation charts for all (or specified) weight classes.
    
    Args:
        force_reload (bool): Force reload data from source
        weight_classes (List[str], optional): Specific weight classes to process
        output_dir (str, optional): Custom output directory
        
    Returns:
        List[str]: Paths to generated chart files
    """
    start_time = time.time()
    log_processing_step("Chart Generation Pipeline")
    
    try:
        # Ensure output directories exist
        ensure_directories()
        
        # Load data
        step_start = time.time()
        log_processing_step("Data Loading")
        
        df = load_fighter_data(force_reload=force_reload)
        log_data_summary(df, "Loaded Fighter Data")
        log_processing_step("Data Loading", step_start)
        
        # Filter for specific weight classes if requested
        if weight_classes:
            available_classes = df['weight_class'].unique()
            invalid_classes = [wc for wc in weight_classes if wc not in available_classes]
            
            if invalid_classes:
                raise ValueError(f"Invalid weight classes: {invalid_classes}. "
                               f"Available: {sorted(available_classes)}")
            
            df = df[df['weight_class'].isin(weight_classes)].copy()
            log_data_summary(df, f"Filtered Data ({len(weight_classes)} weight classes)")
        
        # Compute correlations
        step_start = time.time()
        log_processing_step("Correlation Analysis")
        
        correlation_results = compute_correlations_by_weight_class(df)
        log_correlation_summary(correlation_results)
        log_processing_step("Correlation Analysis", step_start)
        
        # Generate visualizations
        step_start = time.time()
        log_processing_step("Visualization Generation")
        
        setup_matplotlib_style()
        generated_files = generate_all_dashboards(correlation_results)
        
        log_processing_step("Visualization Generation", step_start)
        log_processing_step("Chart Generation Pipeline", start_time)
        
        return generated_files
        
    except Exception as e:
        log_error_with_context(e, "Chart Generation Pipeline")
        raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate UFC fight analytics correlation charts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Generate charts for all weight classes
  %(prog)s --force-reload           # Force reload data and generate charts
  %(prog)s --weight-classes Lightweight Heavyweight  # Specific weight classes
  %(prog)s --output-dir custom_viz  # Custom output directory
  %(prog)s --verbose                # Enable debug logging
        """
    )
    
    parser.add_argument(
        '--force-reload', 
        action='store_true',
        help='Force reload data from raw source'
    )
    
    parser.add_argument(
        '--weight-classes',
        nargs='+',
        help='Specific weight classes to process'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Custom output directory for charts'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--list-weight-classes',
        action='store_true',
        help='List available weight classes and exit'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level=log_level, console_output=True)
    
    try:
        # List weight classes if requested
        if args.list_weight_classes:
            df = load_fighter_data()
            weight_classes = sorted(df['weight_class'].unique())
            print("Available weight classes:")
            for wc in weight_classes:
                count = len(df[df['weight_class'] == wc])
                print(f"  - {wc} ({count:,} records)")
            return
        
        # Generate charts
        generated_files = generate_all_charts(
            force_reload=args.force_reload,
            weight_classes=args.weight_classes,
            output_dir=args.output_dir
        )
        
        print(f"\n✅ Successfully generated {len(generated_files)} chart files:")
        for file_path in generated_files:
            print(f"   📊 {file_path}")
        
        print(f"\n🎯 All charts saved to: {os.path.dirname(generated_files[0]) if generated_files else 'visualizations/'}")
        
    except KeyboardInterrupt:
        print("\n❌ Chart generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error generating charts: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()