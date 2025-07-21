#!/usr/bin/env python3
"""
Simple test for the integrated UFC analytics pipeline including predictions.

This script tests the complete pipeline without requiring heavy ML dependencies.
"""

import os
import sys
import subprocess

def test_individual_components():
    """Test individual components of the system."""
    print("=== Testing Individual Components ===\n")
    
    # Test the standalone prediction system
    print("1. Testing standalone prediction system...")
    try:
        result = subprocess.run([sys.executable, 'test_predictions.py'], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("   ✅ Prediction system test passed")
        else:
            print("   ❌ Prediction system test failed")
            print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Failed to run prediction test: {e}")
    
    print()

def check_generated_files():
    """Check if required files were generated."""
    print("=== Checking Generated Files ===\n")
    
    required_files = [
        'output/prediction_analysis.md',
        'output/prediction_report.html', 
        'output/prediction_results.json',
        'visualizations/prediction_accuracy_by_weight_class.svg',
        'visualizations/training_sample_sizes.svg'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} (missing)")
    
    print()

def display_results_summary():
    """Display a summary of the prediction results."""
    print("=== Prediction Results Summary ===\n")
    
    try:
        import json
        with open('output/prediction_results.json', 'r') as f:
            results = json.load(f)
        
        summary = results.get('summary', {})
        models = results.get('models', {})
        
        print(f"Weight classes analyzed: {summary.get('total_weight_classes', 0)}")
        print(f"Models successfully trained: {summary.get('trained_models', 0)}")
        print(f"Overall average accuracy: {summary.get('overall_accuracy', 0):.3f}")
        
        if models:
            print(f"\nTop 3 performing models:")
            sorted_models = sorted(models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            for i, (weight_class, model_info) in enumerate(sorted_models[:3]):
                print(f"  {i+1}. {weight_class}: {model_info['accuracy']:.3f} accuracy")
        
    except Exception as e:
        print(f"Could not load results: {e}")
    
    print()

def show_file_info():
    """Show information about generated files."""
    print("=== Generated File Information ===\n")
    
    files_info = [
        ('output/prediction_analysis.md', 'Markdown analysis report'),
        ('output/prediction_report.html', 'Interactive HTML report'),
        ('output/prediction_results.json', 'Raw prediction results data'),
        ('visualizations/', 'Visualization directory')
    ]
    
    for file_path, description in files_info:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                svg_files = [f for f in os.listdir(file_path) if f.endswith('.svg')]
                print(f"📁 {file_path}: {description}")
                print(f"   Contains {len(svg_files)} SVG prediction charts")
            else:
                size = os.path.getsize(file_path)
                print(f"📄 {file_path}: {description} ({size:,} bytes)")
        else:
            print(f"❌ {file_path}: Not found")
    
    print()

def test_data_validation():
    """Test data validation functions."""
    print("=== Data Validation Test ===\n")
    
    data_file = 'data/data.csv'
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                line_count = sum(1 for line in f)
            
            file_size = os.path.getsize(data_file)
            print(f"✅ Data file: {data_file}")
            print(f"   Lines: {line_count:,}")
            print(f"   Size: {file_size:,} bytes")
            
            # Check first line for headers
            with open(data_file, 'r') as f:
                first_line = f.readline().strip()
                if 'R_fighter' in first_line and 'B_fighter' in first_line:
                    print("   ✅ Headers look correct")
                else:
                    print("   ⚠️ Header format may be incorrect")
            
        except Exception as e:
            print(f"❌ Error reading data file: {e}")
    else:
        print(f"❌ Data file not found: {data_file}")
    
    print()

def main():
    """Main test function."""
    print("🚀 UFC Analytics Pipeline Integration Test")
    print("=" * 60)
    print()
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run tests
    test_data_validation()
    test_individual_components()
    check_generated_files()
    display_results_summary()
    show_file_info()
    
    print("=" * 60)
    print("✅ Integration test completed!")
    print()
    print("Next steps:")
    print("- Open output/prediction_report.html in a web browser to view the interactive report")
    print("- Review output/prediction_analysis.md for detailed analysis")
    print("- Check visualizations/ directory for SVG charts")
    print()
    print("To run the full pipeline with real correlation data:")
    print("  python run_all.py")

if __name__ == "__main__":
    main()