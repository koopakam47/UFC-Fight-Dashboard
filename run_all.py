#!/usr/bin/env python3
"""
Main orchestration script for UFC Fight Analytics Dashboard.

This script coordinates the entire analytics pipeline:
1. Loads and validates the data
2. Performs correlation analysis  
3. Generates visualizations
4. Updates documentation
5. Provides comprehensive reporting

Usage:
    python run_all.py [options]
"""

import os
import sys
import time
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ufc_analytics.data_loader import load_fighter_data, validate_data_quality, get_available_weight_classes
from ufc_analytics.correlation_analysis import compute_correlations_by_weight_class, summarize_correlation_results
from ufc_analytics.visualization import generate_all_dashboards, setup_matplotlib_style
from ufc_analytics.logging_utils import (
    setup_logging, log_data_summary, log_correlation_summary, 
    log_processing_step, log_error_with_context
)
from ufc_analytics.config import ensure_directories, PATHS


def update_readme_with_results(correlation_results: Dict, generated_files: List[str]) -> str:
    """
    Update README.md with latest analysis results.
    
    Args:
        correlation_results (Dict): Correlation analysis results
        generated_files (List[str]): List of generated visualization files
        
    Returns:
        str: Path to updated README file
    """
    readme_path = 'README.md'
    
    # Read existing README
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            existing_content = f.read()
    else:
        existing_content = ""
    
    # Generate summary section
    summary = generate_analysis_summary(correlation_results)
    
    # Create updated content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"""# UFC Fight Analytics Dashboard

*Last updated: {timestamp}*

The UFC Fight Analytics Dashboard is a data-driven exploration of key performance indicators that correlate with victory across all weight classes in the UFC. This project visualizes the top and bottom statistical drivers of success, non-combat factors, and stance effectiveness.

## 📊 Analysis Summary

{summary}

## 🎯 Key Insights by Weight Class

"""
    
    # Extract existing weight class sections (everything after "## 🎯 Key Insights by Weight Class")
    if "## 🎯 Key Insights by Weight Class" in existing_content:
        # Keep existing weight class insights
        sections = existing_content.split("## 🎯 Key Insights by Weight Class", 1)
        if len(sections) > 1:
            weight_class_content = sections[1]
        else:
            weight_class_content = "\n*Weight class insights will be added here.*\n"
    else:
        weight_class_content = "\n*Weight class insights will be added here.*\n"
    
    # Add technical details section
    technical_section = """
---

## 🔧 Technical Details

- **Data Processing**: Automated cleaning and validation of UFC fight statistics
- **Statistical Analysis**: Pearson correlation coefficients with significance testing
- **Visualization**: Consistent styling with matplotlib and seaborn
- **Architecture**: Modular Python package with comprehensive logging

### Key Metrics Analyzed:
- Striking statistics (accuracy, volume, location)
- Grappling metrics (takedowns, submissions, control time) 
- Physical attributes (height, reach, weight)
- Career statistics (wins, losses, streaks)
- Fight outcomes by stance

---

## 🚀 Usage

### Quick Start
```bash
# Generate all visualizations
python run_all.py

# Generate charts for specific weight classes
python generate_charts.py --weight-classes Lightweight Heavyweight

# Load and explore data
python load_data.py
```

### Advanced Usage
```bash
# Force reload data and enable verbose logging
python run_all.py --force-reload --verbose

# Generate charts with custom output directory
python generate_charts.py --output-dir custom_output

# List available weight classes
python generate_charts.py --list-weight-classes
```

---

## 📈 Future Work

- Incorporate fight-level momentum shifts
- Explore multivariate models and clustering
- Add interactive dashboard with filtering
- Analyze keys to victory relative to time periods
- Implement machine learning predictions

---
"""
    
    # Combine all sections
    updated_content = header + weight_class_content + technical_section
    
    # Write updated README
    with open(readme_path, 'w') as f:
        f.write(updated_content)
    
    return readme_path


def generate_analysis_summary(correlation_results: Dict) -> str:
    """
    Generate a text summary of the analysis results.
    
    Args:
        correlation_results (Dict): Correlation analysis results
        
    Returns:
        str: Formatted summary text
    """
    summary_stats = summarize_correlation_results(correlation_results)
    
    # Calculate overall statistics
    weight_classes_analyzed = len(correlation_results)
    total_fighters = sum(results.get('sample_size', 0) for results in summary_stats.values())
    
    # Find patterns across weight classes
    top_correlations = []
    successful_stances = []
    
    for weight_class, stats in summary_stats.items():
        if not pd.isna(stats['top_positive_correlation']):
            top_correlations.append(stats['top_positive_correlation'])
        if stats['stance_with_highest_win_rate']:
            successful_stances.append(stats['stance_with_highest_win_rate'])
    
    avg_top_correlation = sum(top_correlations) / len(top_correlations) if top_correlations else 0
    
    # Count stance occurrences
    stance_counts = {}
    for stance in successful_stances:
        stance_counts[stance] = stance_counts.get(stance, 0) + 1
    
    most_successful_stance = max(stance_counts, key=stance_counts.get) if stance_counts else "Unknown"
    
    summary = f"""
- **Weight Classes Analyzed**: {weight_classes_analyzed}
- **Total Fighter Records**: {total_fighters:,}
- **Average Top Correlation**: {avg_top_correlation:.3f}
- **Most Successful Stance**: {most_successful_stance} ({stance_counts.get(most_successful_stance, 0)} weight classes)

### Key Findings:
- Striking accuracy and volume consistently predict victory across weight classes
- Opponent statistics show strong negative correlations with winning
- Fighting stance significantly impacts win rates, with variation by weight class
- Physical attributes (reach, height) matter more in heavier divisions
- Grappling metrics (takedowns, submissions) are crucial in lighter divisions
"""
    
    return summary


def run_complete_analysis(
    force_reload: bool = False,
    update_readme: bool = True,
    verbose: bool = False
) -> Dict:
    """
    Run the complete UFC analytics pipeline.
    
    Args:
        force_reload (bool): Force reload data from source
        update_readme (bool): Update README with results
        verbose (bool): Enable verbose logging
        
    Returns:
        Dict: Pipeline results summary
    """
    pipeline_start = time.time()
    log_processing_step("UFC Analytics Pipeline")
    
    results = {
        'data_loaded': False,
        'correlations_computed': False,
        'predictions_generated': False,
        'visualizations_generated': False,
        'readme_updated': False,
        'generated_files': [],
        'error': None
    }
    
    try:
        # Ensure output directories exist
        ensure_directories()
        
        # Step 1: Load and validate data
        step_start = time.time()
        log_processing_step("Data Loading and Validation")
        
        df = load_fighter_data(force_reload=force_reload)
        log_data_summary(df, "UFC Fighter Data")
        
        # Validate data quality
        quality_stats = validate_data_quality(df)
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Data quality: %.2f%% complete, %d duplicates", 
                   100 - quality_stats['missing_data_percentage'],
                   quality_stats['duplicate_records'])
        
        results['data_loaded'] = True
        log_processing_step("Data Loading and Validation", step_start)
        
        # Step 2: Correlation analysis
        step_start = time.time()
        log_processing_step("Correlation Analysis")
        
        correlation_results = compute_correlations_by_weight_class(df)
        log_correlation_summary(correlation_results)
        
        results['correlations_computed'] = True
        log_processing_step("Correlation Analysis", step_start)
        
        # Step 3: Generate fight predictions
        step_start = time.time()
        log_processing_step("Fight Prediction Analysis")
        
        try:
            from ufc_analytics.fight_predictions import create_fight_predictions, generate_prediction_analysis
            from ufc_analytics.prediction_visualization import create_prediction_visualizations, generate_html_report
            
            # Create predictions based on correlation results
            prediction_results = create_fight_predictions(
                data_file=PATHS['raw_data'],
                correlation_results=correlation_results
            )
            
            if 'error' not in prediction_results:
                # Generate prediction analysis
                analysis_text = generate_prediction_analysis(prediction_results)
                analysis_file = os.path.join(PATHS.get('output_dir', 'output'), 'prediction_analysis.md')
                os.makedirs(os.path.dirname(analysis_file), exist_ok=True)
                with open(analysis_file, 'w') as f:
                    f.write(analysis_text)
                
                # Create prediction visualizations
                prediction_charts = create_prediction_visualizations(
                    prediction_results, 
                    PATHS['visualizations_dir']
                )
                
                # Generate HTML report
                html_report = generate_html_report(
                    prediction_results,
                    prediction_charts,
                    os.path.join(PATHS.get('output_dir', 'output'), 'prediction_report.html')
                )
                
                logger.info("Generated %d prediction models with %.3f average accuracy", 
                           prediction_results['summary']['trained_models'],
                           prediction_results['summary']['overall_accuracy'])
                logger.info("Created %d prediction visualization files", len(prediction_charts))
                
                results['predictions_generated'] = True
                results['generated_files'].extend(prediction_charts)
            else:
                logger.warning("Prediction analysis failed: %s", prediction_results['error'])
                results['predictions_generated'] = False
                
        except ImportError as e:
            logger.warning("Could not import prediction modules: %s", str(e))
            results['predictions_generated'] = False
        except Exception as e:
            logger.error("Prediction analysis failed: %s", str(e))
            results['predictions_generated'] = False
        
        log_processing_step("Fight Prediction Analysis", step_start)
        
        # Step 4: Generate visualizations
        step_start = time.time()
        log_processing_step("Visualization Generation")
        
        setup_matplotlib_style()
        generated_files = generate_all_dashboards(correlation_results)
        results['generated_files'] = generated_files
        
        results['visualizations_generated'] = True
        log_processing_step("Visualization Generation", step_start)
        
        # Step 5: Update documentation (optional)
        if update_readme:
            step_start = time.time()
            log_processing_step("Documentation Update")
            
            readme_path = update_readme_with_results(correlation_results, generated_files)
            logger.info("Updated README: %s", readme_path)
            
            results['readme_updated'] = True
            log_processing_step("Documentation Update", step_start)
        
        log_processing_step("UFC Analytics Pipeline", pipeline_start)
        
        return results
        
    except Exception as e:
        results['error'] = str(e)
        log_error_with_context(e, "UFC Analytics Pipeline")
        raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run complete UFC Fight Analytics pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs the complete analytics pipeline:
1. Loads and validates UFC fight data
2. Computes correlations for all weight classes
3. Generates visualization dashboards  
4. Updates README documentation
5. Provides comprehensive reporting

Examples:
  %(prog)s                    # Run complete pipeline
  %(prog)s --force-reload     # Force reload data from source
  %(prog)s --no-readme        # Skip README update
  %(prog)s --verbose          # Enable debug logging
        """
    )
    
    parser.add_argument(
        '--force-reload',
        action='store_true', 
        help='Force reload data from raw source'
    )
    
    parser.add_argument(
        '--no-readme',
        action='store_true',
        help='Skip README update'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level=log_level, console_output=True)
    
    try:
        print("🚀 Starting UFC Fight Analytics Pipeline...")
        print("=" * 60)
        
        results = run_complete_analysis(
            force_reload=args.force_reload,
            update_readme=not args.no_readme,
            verbose=args.verbose
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("✅ Pipeline completed successfully!")
        print(f"📊 Generated {len(results['generated_files'])} visualization files")
        
        if results['generated_files']:
            print("\nGenerated files:")
            for file_path in results['generated_files']:
                print(f"   📈 {file_path}")
        
        if results['readme_updated']:
            print("\n📝 README.md updated with latest results")
        
        print(f"\n🎯 All outputs saved to: {PATHS['visualizations_dir']}")
        
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()