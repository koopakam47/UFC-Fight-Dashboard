"""
Fight outcome prediction module for UFC analytics.

This module provides functions for creating fight predictions based on
correlation analysis and fighter statistics, along with performance evaluation.
"""

import os
import csv
import json
import math
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from .config import DATA_CONFIG, PATHS


def load_csv_data(file_path: str) -> Tuple[List[str], List[List[str]]]:
    """
    Load CSV data without pandas dependency.
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        Tuple[List[str], List[List[str]]]: Headers and data rows
    """
    headers = []
    rows = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        # Clean headers - remove BOM and special characters
        headers = [h.replace('≤', '').replace('﻿', '').strip() for h in headers]
        
        for row in reader:
            rows.append(row)
    
    return headers, rows


def parse_numeric_value(value_str: str) -> float:
    """
    Parse a string value to float, handling empty/missing values.
    
    Args:
        value_str (str): String value to parse
        
    Returns:
        float: Parsed numeric value or 0.0 if invalid
    """
    if not value_str or value_str.strip() == '' or value_str.lower() == 'nan':
        return 0.0
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return 0.0


def create_feature_vector(row: List[str], headers: List[str], feature_columns: List[str]) -> List[float]:
    """
    Create a numeric feature vector from a data row.
    
    Args:
        row (List[str]): Data row
        headers (List[str]): Column headers
        feature_columns (List[str]): Features to extract
        
    Returns:
        List[float]: Numeric feature vector
    """
    header_map = {h: i for i, h in enumerate(headers)}
    features = []
    
    for col in feature_columns:
        if col in header_map:
            idx = header_map[col]
            if idx < len(row):
                features.append(parse_numeric_value(row[idx]))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
    
    return features


def simple_logistic_regression(X: List[List[float]], y: List[int], 
                             learning_rate: float = 0.01, epochs: int = 1000) -> List[float]:
    """
    Simple logistic regression implementation without external dependencies.
    
    Args:
        X (List[List[float]]): Feature matrix
        y (List[int]): Target labels (0 or 1)
        learning_rate (float): Learning rate for gradient descent
        epochs (int): Number of training epochs
        
    Returns:
        List[float]: Learned weights
    """
    n_features = len(X[0]) if X else 0
    weights = [0.0] * (n_features + 1)  # +1 for bias
    
    def sigmoid(z):
        z = max(-250, min(250, z))  # Prevent overflow
        return 1.0 / (1.0 + math.exp(-z))
    
    for epoch in range(epochs):
        for i, (features, label) in enumerate(zip(X, y)):
            # Add bias term
            x_with_bias = [1.0] + features
            
            # Forward pass
            z = sum(w * x for w, x in zip(weights, x_with_bias))
            prediction = sigmoid(z)
            
            # Calculate error
            error = prediction - label
            
            # Update weights
            for j in range(len(weights)):
                weights[j] -= learning_rate * error * x_with_bias[j]
    
    return weights


def predict_with_weights(weights: List[float], features: List[float]) -> float:
    """
    Make prediction using learned weights.
    
    Args:
        weights (List[float]): Model weights
        features (List[float]): Input features
        
    Returns:
        float: Prediction probability
    """
    def sigmoid(z):
        z = max(-250, min(250, z))
        return 1.0 / (1.0 + math.exp(-z))
    
    x_with_bias = [1.0] + features
    z = sum(w * x for w, x in zip(weights, x_with_bias))
    return sigmoid(z)


class CorrelationBasedPredictor:
    """
    Simple predictor based on correlation analysis.
    """
    
    def __init__(self, correlation_results: Dict):
        """
        Initialize with correlation analysis results.
        
        Args:
            correlation_results (Dict): Results from correlation analysis
        """
        self.correlation_results = correlation_results
        self.weight_class_models = {}
        
    def extract_top_features(self, weight_class: str, n_features: int = 10) -> List[str]:
        """
        Extract top predictive features for a weight class.
        
        Args:
            weight_class (str): Weight class name
            n_features (int): Number of top features to extract
            
        Returns:
            List[str]: List of top feature names
        """
        if weight_class not in self.correlation_results:
            return []
        
        results = self.correlation_results[weight_class]
        features = []
        
        # Get top positive correlations
        if 'top_10_overall' in results:
            top_data = results['top_10_overall']
            if hasattr(top_data, 'index'):  # pandas Series
                top_positive = list(top_data.index[:n_features//2])
            else:  # dictionary
                top_positive = list(top_data.keys())[:n_features//2]
            features.extend(top_positive)
        
        # Get top negative correlations (but take absolute value)
        if 'bottom_10_overall' in results:
            bottom_data = results['bottom_10_overall']
            if hasattr(bottom_data, 'index'):  # pandas Series
                top_negative = list(bottom_data.index[:n_features//2])
            else:  # dictionary
                top_negative = list(bottom_data.keys())[:n_features//2]
            features.extend(top_negative)
        
        return features[:n_features]
    
    def prepare_training_data(self, headers: List[str], rows: List[List[str]], 
                            weight_class: str) -> Tuple[List[List[float]], List[int], List[str]]:
        """
        Prepare training data for a specific weight class.
        
        Args:
            headers (List[str]): Data headers
            rows (List[List[str]]): Data rows
            weight_class (str): Weight class to filter
            
        Returns:
            Tuple[List[List[float]], List[int], List[str]]: Features, labels, feature names
        """
        # Find relevant column indices
        header_map = {h: i for i, h in enumerate(headers)}
        weight_class_idx = header_map.get('weight_class', -1)
        winner_idx = header_map.get('Winner', -1)
        
        if weight_class_idx == -1 or winner_idx == -1:
            return [], [], []
        
        # Filter rows for weight class
        filtered_rows = [row for row in rows 
                        if len(row) > weight_class_idx and row[weight_class_idx] == weight_class]
        
        if len(filtered_rows) < 50:  # Minimum sample size
            return [], [], []
        
        # Get top features for this weight class
        feature_names = self.extract_top_features(weight_class)
        if not feature_names:
            # Fallback to basic features
            feature_names = [col for col in headers if col.startswith(('R_', 'B_')) and 
                           'avg' in col and any(stat in col for stat in 
                           ['SIG_STR_pct', 'TD_pct', 'KD', 'SUB_ATT', 'wins', 'losses'])][:10]
        
        # Create feature matrix and labels
        X = []
        y = []
        
        for row in filtered_rows:
            if len(row) > winner_idx and row[winner_idx] in ['Red', 'Blue']:
                features = create_feature_vector(row, headers, feature_names)
                label = 1 if row[winner_idx] == 'Red' else 0
                
                X.append(features)
                y.append(label)
        
        return X, y, feature_names
    
    def train_weight_class_model(self, headers: List[str], rows: List[List[str]], 
                                weight_class: str) -> Optional[Dict]:
        """
        Train a model for a specific weight class.
        
        Args:
            headers (List[str]): Data headers
            rows (List[List[str]]): Data rows
            weight_class (str): Weight class to train on
            
        Returns:
            Optional[Dict]: Trained model information
        """
        X, y, feature_names = self.prepare_training_data(headers, rows, weight_class)
        
        if len(X) < 50:
            return None
        
        # Split data for training (80%) and testing (20%)
        split_idx = int(0.8 * len(X))
        indices = list(range(len(X)))
        random.shuffle(indices)
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]
        
        # Train simple logistic regression
        weights = simple_logistic_regression(X_train, y_train)
        
        # Make predictions on test set
        predictions = []
        for features in X_test:
            pred_prob = predict_with_weights(weights, features)
            predictions.append(pred_prob)
        
        # Calculate accuracy
        correct = 0
        for pred_prob, true_label in zip(predictions, y_test):
            pred_label = 1 if pred_prob >= 0.5 else 0
            if pred_label == true_label:
                correct += 1
        
        accuracy = correct / len(y_test) if y_test else 0.0
        
        model_info = {
            'weight_class': weight_class,
            'weights': weights,
            'feature_names': feature_names,
            'accuracy': accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_predictions': predictions,
            'test_labels': y_test
        }
        
        self.weight_class_models[weight_class] = model_info
        return model_info


def create_fight_predictions(data_file: str = None, 
                           correlation_results: Dict = None) -> Dict[str, Any]:
    """
    Create fight predictions based on correlation analysis and simple modeling.
    
    Args:
        data_file (str, optional): Path to data file
        correlation_results (Dict, optional): Correlation analysis results
        
    Returns:
        Dict[str, Any]: Prediction results and model performance
    """
    if data_file is None:
        data_file = PATHS['raw_data']
    
    if not os.path.exists(data_file):
        return {'error': f'Data file not found: {data_file}'}
    
    # Load data
    try:
        headers, rows = load_csv_data(data_file)
    except Exception as e:
        return {'error': f'Failed to load data: {str(e)}'}
    
    # Initialize predictor
    predictor = CorrelationBasedPredictor(correlation_results or {})
    
    # Get unique weight classes
    header_map = {h: i for i, h in enumerate(headers)}
    weight_class_idx = header_map.get('weight_class', -1)
    
    if weight_class_idx == -1:
        return {'error': 'weight_class column not found'}
    
    weight_classes = set()
    for row in rows:
        if len(row) > weight_class_idx and row[weight_class_idx]:
            weight_classes.add(row[weight_class_idx])
    
    # Train models for each weight class
    results = {
        'models': {},
        'summary': {
            'total_weight_classes': len(weight_classes),
            'trained_models': 0,
            'overall_accuracy': 0.0
        }
    }
    
    total_accuracy = 0.0
    trained_count = 0
    
    for weight_class in sorted(weight_classes):
        print(f"Training model for {weight_class}...")
        model_info = predictor.train_weight_class_model(headers, rows, weight_class)
        
        if model_info:
            results['models'][weight_class] = model_info
            total_accuracy += model_info['accuracy']
            trained_count += 1
            print(f"  Accuracy: {model_info['accuracy']:.3f}")
        else:
            print(f"  Insufficient data for {weight_class}")
    
    results['summary']['trained_models'] = trained_count
    if trained_count > 0:
        results['summary']['overall_accuracy'] = total_accuracy / trained_count
    
    return results


def generate_prediction_analysis(prediction_results: Dict[str, Any]) -> str:
    """
    Generate a text analysis of prediction results.
    
    Args:
        prediction_results (Dict[str, Any]): Results from create_fight_predictions
        
    Returns:
        str: Analysis text
    """
    if 'error' in prediction_results:
        return f"Error in prediction analysis: {prediction_results['error']}"
    
    summary = prediction_results['summary']
    models = prediction_results['models']
    
    analysis = []
    analysis.append("# UFC Fight Prediction Analysis\n")
    analysis.append(f"## Summary")
    analysis.append(f"- Total weight classes analyzed: {summary['total_weight_classes']}")
    analysis.append(f"- Models successfully trained: {summary['trained_models']}")
    analysis.append(f"- Overall average accuracy: {summary['overall_accuracy']:.3f}\n")
    
    if models:
        analysis.append("## Model Performance by Weight Class\n")
        
        # Sort by accuracy
        sorted_models = sorted(models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for weight_class, model_info in sorted_models:
            analysis.append(f"### {weight_class}")
            analysis.append(f"- Accuracy: {model_info['accuracy']:.3f}")
            analysis.append(f"- Training samples: {model_info['train_size']}")
            analysis.append(f"- Test samples: {model_info['test_size']}")
            analysis.append(f"- Key features: {', '.join(model_info['feature_names'][:5])}")
            analysis.append("")
        
        # Best and worst performing models
        best_model = sorted_models[0]
        worst_model = sorted_models[-1]
        
        analysis.append("## Key Insights\n")
        analysis.append(f"**Best performing weight class**: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.3f})")
        analysis.append(f"**Most challenging weight class**: {worst_model[0]} (Accuracy: {worst_model[1]['accuracy']:.3f})")
        
        avg_accuracy = sum(model['accuracy'] for model in models.values()) / len(models)
        analysis.append(f"**Average accuracy across all weight classes**: {avg_accuracy:.3f}")
        
        # Performance categories
        high_perf = [wc for wc, model in models.items() if model['accuracy'] >= 0.6]
        med_perf = [wc for wc, model in models.items() if 0.5 <= model['accuracy'] < 0.6]
        low_perf = [wc for wc, model in models.items() if model['accuracy'] < 0.5]
        
        analysis.append(f"\n**Performance Distribution**:")
        analysis.append(f"- High performance (≥60%): {len(high_perf)} weight classes")
        analysis.append(f"- Medium performance (50-60%): {len(med_perf)} weight classes") 
        analysis.append(f"- Low performance (<50%): {len(low_perf)} weight classes")
    
    return "\n".join(analysis)


def save_prediction_results(prediction_results: Dict[str, Any], 
                          output_file: str = 'prediction_results.json') -> str:
    """
    Save prediction results to a JSON file.
    
    Args:
        prediction_results (Dict[str, Any]): Prediction results
        output_file (str): Output file name
        
    Returns:
        str: Path to saved file
    """
    output_path = os.path.join(PATHS.get('output_dir', '.'), output_file)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy arrays to lists if needed and handle non-serializable objects
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
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return output_path


def predict_upcoming_fight(red_fighter_stats: Dict[str, float], 
                          blue_fighter_stats: Dict[str, float], 
                          weight_class: str,
                          models_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Predict the outcome of an upcoming fight between two fighters.
    
    Args:
        red_fighter_stats: Statistics for the red corner fighter
        blue_fighter_stats: Statistics for the blue corner fighter  
        weight_class: Weight class for the fight
        models_data: Pre-loaded models data (optional, will load from file if not provided)
        
    Returns:
        Dict containing prediction results
    """
    # Load models data if not provided
    if models_data is None:
        results_file = os.path.join(PATHS.get('output_dir', '.'), 'prediction_results.json')
        if not os.path.exists(results_file):
            return {'error': 'No trained models found. Please run training first.'}
        
        try:
            with open(results_file, 'r') as f:
                models_data = json.load(f)
        except Exception as e:
            return {'error': f'Error loading models: {e}'}
    
    if 'models' not in models_data:
        return {'error': 'No models found in data'}
    
    if weight_class not in models_data['models']:
        available_classes = list(models_data['models'].keys())
        return {
            'error': f'Weight class "{weight_class}" not available. Available: {available_classes}'
        }
    
    model_info = models_data['models'][weight_class]
    weights = model_info['weights']
    feature_names = model_info['feature_names']
    
    # Prepare feature vector
    features = []
    used_features = {}
    
    for feature_name in feature_names:
        if feature_name.startswith('R_'):
            # Red fighter feature
            stat_name = feature_name[2:]  # Remove 'R_' prefix
            value = red_fighter_stats.get(stat_name, 0.0)
            used_features[feature_name] = value
        elif feature_name.startswith('B_'):
            # Blue fighter feature  
            stat_name = feature_name[2:]  # Remove 'B_' prefix
            value = blue_fighter_stats.get(stat_name, 0.0)
            used_features[feature_name] = value
        else:
            value = 0.0
            used_features[feature_name] = value
        
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
        'training_samples': model_info['train_size'],
        'features_used': used_features,
        'model_features': feature_names
    }