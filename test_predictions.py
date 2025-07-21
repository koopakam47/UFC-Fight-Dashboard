#!/usr/bin/env python3
"""
Simple test script for UFC fight predictions without heavy dependencies.

This script demonstrates the fight prediction functionality using basic Python
libraries and creates visualizations as SVG files.
"""

import os
import sys
import json
import random

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def ensure_output_directories():
    """Create necessary output directories."""
    directories = ['output', 'visualizations']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)

def load_correlation_results():
    """
    Load existing correlation results if available, otherwise create mock data.
    
    Returns:
        Dict: Correlation results for weight classes
    """
    # Try to load existing correlation data
    correlation_file = 'output/correlation_results.json'
    
    if os.path.exists(correlation_file):
        try:
            with open(correlation_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Could not load correlation results: {e}")
    
    # Create mock correlation data for testing
    weight_classes = [
        'Heavyweight', 'LightHeavyweight', 'Middleweight', 'Welterweight',
        'Lightweight', 'Featherweight', 'Bantamweight', 'Flyweight',
        'WomenBantamweight', 'WomenFlyweight', 'WomenStrawweight'
    ]
    
    mock_correlations = {}
    
    for wc in weight_classes:
        # Create mock correlation data
        features = [
            'R_avg_SIG_STR_pct', 'R_avg_TD_pct', 'R_wins', 'R_current_win_streak',
            'B_avg_SIG_STR_pct', 'B_avg_TD_pct', 'B_wins', 'B_current_win_streak',
            'R_avg_KD', 'B_avg_KD', 'R_Height_cms', 'B_Height_cms'
        ]
        
        # Generate random correlations
        correlations = {}
        for feature in features:
            correlations[feature] = random.uniform(-0.3, 0.3)
        
        # Sort to create top/bottom lists
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        mock_correlations[wc] = {
            'top_10_overall': {item[0]: item[1] for item in sorted_corr[:5]},
            'bottom_10_overall': {item[0]: item[1] for item in sorted_corr[-5:]},
            'sample_size': random.randint(100, 500)
        }
    
    return mock_correlations

def test_predictions():
    """Test the prediction functionality."""
    print("=== UFC Fight Prediction Test ===\n")
    
    # Ensure directories exist
    ensure_output_directories()
    
    # Load or create correlation data
    print("Loading correlation data...")
    correlation_results = load_correlation_results()
    print(f"Loaded correlation data for {len(correlation_results)} weight classes")
    
    # Import prediction modules (only after ensuring they exist)
    try:
        from ufc_analytics.fight_predictions import create_fight_predictions, generate_prediction_analysis
        from ufc_analytics.prediction_visualization import create_prediction_visualizations, generate_html_report
    except ImportError as e:
        print(f"Error importing prediction modules: {e}")
        return
    
    # Check if data file exists
    data_file = 'data/data.csv'
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        print("Please ensure the UFC data file exists.")
        return
    
    print(f"Using data file: {data_file}")
    
    # Create predictions
    print("\nCreating fight predictions...")
    prediction_results = create_fight_predictions(
        data_file=data_file,
        correlation_results=correlation_results
    )
    
    # Check for errors
    if 'error' in prediction_results:
        print(f"Error in predictions: {prediction_results['error']}")
        return
    
    # Display summary
    summary = prediction_results['summary']
    print(f"\n=== Prediction Summary ===")
    print(f"Weight classes analyzed: {summary['total_weight_classes']}")
    print(f"Models trained: {summary['trained_models']}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.3f}")
    
    # Show top performing models
    if prediction_results['models']:
        print(f"\n=== Top Performing Models ===")
        sorted_models = sorted(
            prediction_results['models'].items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        
        for i, (weight_class, model_info) in enumerate(sorted_models[:5]):
            print(f"{i+1}. {weight_class}: {model_info['accuracy']:.3f} accuracy "
                  f"({model_info['train_size']} training samples)")
    
    # Generate analysis text
    print(f"\nGenerating prediction analysis...")
    analysis_text = generate_prediction_analysis(prediction_results)
    
    # Save analysis
    analysis_file = 'output/prediction_analysis.md'
    with open(analysis_file, 'w') as f:
        f.write(analysis_text)
    print(f"Analysis saved to: {analysis_file}")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    chart_files = create_prediction_visualizations(prediction_results, 'visualizations')
    print(f"Created {len(chart_files)} visualization files:")
    for chart_file in chart_files:
        print(f"  - {chart_file}")
    
    # Generate HTML report
    print(f"\nGenerating HTML report...")
    html_report = generate_html_report(
        prediction_results, 
        chart_files, 
        'output/prediction_report.html'
    )
    print(f"HTML report saved to: {html_report}")
    
    # Save prediction results as JSON
    results_file = 'output/prediction_results.json'
    
    # Prepare serializable version
    serializable_results = {}
    for key, value in prediction_results.items():
        if key == 'models':
            serializable_models = {}
            for model_name, model_data in value.items():
                serializable_model = {}
                for k, v in model_data.items():
                    if isinstance(v, (list, tuple)):
                        serializable_model[k] = list(v)
                    else:
                        serializable_model[k] = v
                serializable_models[model_name] = serializable_model
            serializable_results[key] = serializable_models
        else:
            serializable_results[key] = value
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Prediction results saved to: {results_file}")
    
    print(f"\n=== Test Complete ===")
    print(f"Generated files:")
    print(f"  - {analysis_file}")
    print(f"  - {html_report}")
    print(f"  - {results_file}")
    print(f"  - {len(chart_files)} visualization files")

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    
    try:
        test_predictions()
    except Exception as e:
        print(f"Error during prediction test: {e}")
        import traceback
        traceback.print_exc()