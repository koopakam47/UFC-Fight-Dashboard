#!/usr/bin/env python3
"""
Easy-to-use UFC fight prediction tool for upcoming fights.

This script allows users to input fighter statistics and get predictions
for upcoming fights using the trained machine learning models.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ufc_analytics.fight_predictions import load_csv_data, parse_numeric_value
except ImportError as e:
    print(f"Error importing prediction modules: {e}")
    print("Please ensure the ufc_analytics package is available.")
    sys.exit(1)


def load_trained_models() -> Optional[Dict]:
    """Load trained models from the prediction results file."""
    results_file = 'output/prediction_results.json'
    
    if not os.path.exists(results_file):
        print("No trained models found. Please run the training pipeline first:")
        print("  python test_predictions.py")
        print("  OR")
        print("  python run_all.py")
        return None
    
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading trained models: {e}")
        return None


def get_weight_classes() -> List[str]:
    """Get list of available weight classes."""
    prediction_results = load_trained_models()
    if not prediction_results or 'models' not in prediction_results:
        return []
    
    return list(prediction_results['models'].keys())


def predict_fight_outcome(red_fighter_stats: Dict[str, float], 
                         blue_fighter_stats: Dict[str, float], 
                         weight_class: str) -> Dict[str, any]:
    """
    Predict the outcome of a fight between two fighters.
    
    Args:
        red_fighter_stats: Statistics for the red corner fighter
        blue_fighter_stats: Statistics for the blue corner fighter  
        weight_class: Weight class for the fight
        
    Returns:
        Dict containing prediction results
    """
    prediction_results = load_trained_models()
    if not prediction_results:
        return {'error': 'No trained models available'}
    
    if weight_class not in prediction_results['models']:
        available_classes = list(prediction_results['models'].keys())
        return {
            'error': f'Weight class "{weight_class}" not available. Available: {available_classes}'
        }
    
    model_info = prediction_results['models'][weight_class]
    weights = model_info['weights']
    feature_names = model_info['feature_names']
    
    # Prepare feature vector
    features = []
    for feature_name in feature_names:
        if feature_name.startswith('R_'):
            # Red fighter feature
            stat_name = feature_name[2:]  # Remove 'R_' prefix
            value = red_fighter_stats.get(stat_name, 0.0)
        elif feature_name.startswith('B_'):
            # Blue fighter feature  
            stat_name = feature_name[2:]  # Remove 'B_' prefix
            value = blue_fighter_stats.get(stat_name, 0.0)
        else:
            value = 0.0
        
        features.append(value)
    
    # Calculate logistic regression prediction (weights[0] is bias)
    x_with_bias = [1.0] + features
    z = sum(w * x for w, x in zip(weights, x_with_bias))
    z = max(-250, min(250, z))  # Prevent overflow
    probability_red_wins = 1 / (1 + math.exp(-z))
    probability_blue_wins = 1 - probability_red_wins
    
    # Determine predicted winner
    if probability_red_wins > 0.5:
        predicted_winner = 'Red'
        confidence = probability_red_wins
    else:
        predicted_winner = 'Blue'
        confidence = probability_blue_wins
    
    return {
        'predicted_winner': predicted_winner,
        'red_win_probability': probability_red_wins,
        'blue_win_probability': probability_blue_wins,
        'confidence': confidence,
        'weight_class': weight_class,
        'model_accuracy': model_info['accuracy'],
        'training_samples': model_info['train_size']
    }


def interactive_prediction():
    """Interactive mode for predicting fights."""
    print("=== UFC Fight Prediction Tool ===\n")
    
    # Load available weight classes
    weight_classes = get_weight_classes()
    if not weight_classes:
        print("No trained models available. Please train models first.")
        return
    
    print("Available weight classes:")
    for i, wc in enumerate(weight_classes, 1):
        print(f"  {i}. {wc}")
    
    # Select weight class
    while True:
        try:
            choice = input(f"\nSelect weight class (1-{len(weight_classes)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(weight_classes):
                weight_class = weight_classes[idx]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    print(f"\nSelected weight class: {weight_class}")
    
    # Define common fighter statistics
    stat_descriptions = {
        'avg_SIG_STR_pct': 'Significant Strike Accuracy (%)',
        'avg_TD_pct': 'Takedown Accuracy (%)', 
        'wins': 'Total Wins',
        'current_win_streak': 'Current Win Streak',
        'avg_KD': 'Average Knockdowns per Fight',
        'avg_opp_KD': 'Average Opponent Knockdowns per Fight',
        'avg_opp_SIG_STR_pct': 'Average Opponent Sig Strike Accuracy (%)',
        'Height_cms': 'Height (cm)'
    }
    
    # Get red fighter stats
    print(f"\n=== Red Corner Fighter ===")
    red_stats = {}
    for stat, description in stat_descriptions.items():
        while True:
            try:
                value = input(f"{description}: ").strip()
                if value:
                    red_stats[stat] = float(value)
                else:
                    red_stats[stat] = 0.0
                break
            except ValueError:
                print("Please enter a valid number.")
    
    # Get blue fighter stats
    print(f"\n=== Blue Corner Fighter ===")
    blue_stats = {}
    for stat, description in stat_descriptions.items():
        while True:
            try:
                value = input(f"{description}: ").strip()
                if value:
                    blue_stats[stat] = float(value)
                else:
                    blue_stats[stat] = 0.0
                break
            except ValueError:
                print("Please enter a valid number.")
    
    # Make prediction
    print(f"\n=== Prediction Results ===")
    result = predict_fight_outcome(red_stats, blue_stats, weight_class)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Weight Class: {result['weight_class']}")
    print(f"Predicted Winner: {result['predicted_winner']} Corner")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"")
    print(f"Detailed Probabilities:")
    print(f"  Red Corner: {result['red_win_probability']:.1%}")
    print(f"  Blue Corner: {result['blue_win_probability']:.1%}")
    print(f"")
    print(f"Model Info:")
    print(f"  Training Accuracy: {result['model_accuracy']:.1%}")
    print(f"  Training Samples: {result['training_samples']}")


def example_prediction():
    """Show an example prediction with sample data."""
    print("=== Example Prediction ===\n")
    
    weight_classes = get_weight_classes()
    if not weight_classes:
        print("No trained models available.")
        return
    
    # Use first available weight class
    weight_class = weight_classes[0]
    
    # Example fighter stats
    red_fighter = {
        'avg_SIG_STR_pct': 45.2,
        'avg_TD_pct': 35.8,
        'wins': 15,
        'current_win_streak': 3,
        'avg_KD': 0.8,
        'avg_opp_KD': 0.3,
        'Height_cms': 180
    }
    
    blue_fighter = {
        'avg_SIG_STR_pct': 42.1,
        'avg_TD_pct': 28.5,
        'wins': 12,
        'current_win_streak': 1,
        'avg_KD': 0.5,
        'avg_opp_KD': 0.7,
        'Height_cms': 175
    }
    
    result = predict_fight_outcome(red_fighter, blue_fighter, weight_class)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Weight Class: {weight_class}")
    print(f"Red Fighter Stats: {red_fighter}")
    print(f"Blue Fighter Stats: {blue_fighter}")
    print(f"")
    print(f"Prediction: {result['predicted_winner']} Corner wins")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Red Corner probability: {result['red_win_probability']:.1%}")
    print(f"Blue Corner probability: {result['blue_win_probability']:.1%}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='UFC Fight Prediction Tool')
    parser.add_argument('--interactive', '-i', action='store_true',
                      help='Run in interactive mode')
    parser.add_argument('--example', '-e', action='store_true',
                      help='Show example prediction')
    parser.add_argument('--list-classes', '-l', action='store_true',
                      help='List available weight classes')
    
    args = parser.parse_args()
    
    if args.list_classes:
        weight_classes = get_weight_classes()
        if weight_classes:
            print("Available weight classes:")
            for wc in weight_classes:
                print(f"  - {wc}")
        else:
            print("No trained models available.")
    elif args.example:
        example_prediction()
    elif args.interactive:
        interactive_prediction()
    else:
        print("UFC Fight Prediction Tool")
        print("")
        print("Usage:")
        print("  python predict_upcoming_fight.py --interactive  # Interactive mode")
        print("  python predict_upcoming_fight.py --example     # Show example")
        print("  python predict_upcoming_fight.py --list-classes # List weight classes")
        print("")
        print("First run the training pipeline to create models:")
        print("  python test_predictions.py")


if __name__ == "__main__":
    import math
    main()