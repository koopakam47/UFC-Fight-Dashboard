"""
Visualization module for UFC fight predictions.

Creates charts and plots to analyze prediction performance against actual outcomes.
"""

import os
import math
from typing import Dict, List, Tuple, Any

# Simple plotting without matplotlib initially - create SVG directly
def create_svg_chart(width: int, height: int, title: str) -> str:
    """Create basic SVG chart structure."""
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <style>
        .title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; }}
        .axis-label {{ font-family: Arial, sans-serif; font-size: 12px; }}
        .bar {{ stroke: black; stroke-width: 1; }}
        .text {{ font-family: Arial, sans-serif; font-size: 10px; }}
    </style>
    <text x="{width//2}" y="25" text-anchor="middle" class="title">{title}</text>
'''


def create_bar_chart(data: Dict[str, float], title: str, output_path: str, 
                    width: int = 800, height: int = 600) -> str:
    """
    Create a simple SVG bar chart.
    
    Args:
        data (Dict[str, float]): Data to plot (name -> value)
        title (str): Chart title
        output_path (str): Output file path
        width (int): Chart width
        height (int): Chart height
        
    Returns:
        str: Path to created chart
    """
    if not data:
        return ""
    
    margin = 60
    chart_width = width - 2 * margin
    chart_height = height - 2 * margin
    
    # Calculate bar dimensions
    max_value = max(data.values())
    min_value = min(data.values())
    value_range = max_value - min_value if max_value != min_value else 1
    
    bar_width = chart_width / len(data)
    
    svg_content = create_svg_chart(width, height, title)
    
    # Draw axes
    svg_content += f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black" stroke-width="2"/>\n'
    svg_content += f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black" stroke-width="2"/>\n'
    
    # Draw bars and labels
    for i, (name, value) in enumerate(data.items()):
        x = margin + i * bar_width
        bar_height = (value - min_value) / value_range * chart_height if value_range > 0 else chart_height * 0.5
        y = height - margin - bar_height
        
        # Color based on value (green for high, red for low)
        if value >= 0.6:
            color = "#2E8B57"  # Sea green
        elif value >= 0.5:
            color = "#FFA500"  # Orange
        else:
            color = "#DC143C"  # Crimson
        
        svg_content += f'<rect x="{x}" y="{y}" width="{bar_width * 0.8}" height="{bar_height}" fill="{color}" class="bar"/>\n'
        
        # Value label on top of bar
        svg_content += f'<text x="{x + bar_width * 0.4}" y="{y - 5}" text-anchor="middle" class="text">{value:.3f}</text>\n'
        
        # X-axis label (rotated)
        label_x = x + bar_width * 0.4
        label_y = height - margin + 20
        svg_content += f'<text x="{label_x}" y="{label_y}" text-anchor="middle" class="text" transform="rotate(-45 {label_x} {label_y})">{name}</text>\n'
    
    # Y-axis label
    svg_content += f'<text x="20" y="{height//2}" text-anchor="middle" class="axis-label" transform="rotate(-90 20 {height//2})">Accuracy</text>\n'
    
    # Y-axis ticks
    for i in range(6):
        tick_value = min_value + (i / 5) * value_range
        tick_y = height - margin - (i / 5) * chart_height
        svg_content += f'<line x1="{margin - 5}" y1="{tick_y}" x2="{margin}" y2="{tick_y}" stroke="black"/>\n'
        svg_content += f'<text x="{margin - 10}" y="{tick_y + 3}" text-anchor="end" class="text">{tick_value:.2f}</text>\n'
    
    svg_content += '</svg>'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(svg_content)
    
    return output_path


def create_confusion_matrix_chart(true_labels: List[int], predictions: List[float], 
                                threshold: float, title: str, output_path: str) -> str:
    """
    Create a confusion matrix visualization.
    
    Args:
        true_labels (List[int]): True labels (0 or 1)
        predictions (List[float]): Prediction probabilities
        threshold (float): Classification threshold
        title (str): Chart title
        output_path (str): Output file path
        
    Returns:
        str: Path to created chart
    """
    if not true_labels or not predictions:
        return ""
    
    # Calculate confusion matrix
    tp = fp = tn = fn = 0
    
    for true_label, pred_prob in zip(true_labels, predictions):
        pred_label = 1 if pred_prob >= threshold else 0
        
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        elif true_label == 0 and pred_label == 0:
            tn += 1
        else:
            fn += 1
    
    total = tp + fp + tn + fn
    
    width, height = 400, 400
    margin = 60
    cell_size = (min(width, height) - 2 * margin) // 2
    
    svg_content = create_svg_chart(width, height, title)
    
    # Define matrix
    matrix = [[tp, fp], [fn, tn]]
    labels = [['True Pos', 'False Pos'], ['False Neg', 'True Neg']]
    colors = [['#90EE90', '#FFB6C1'], ['#FFB6C1', '#90EE90']]
    
    # Draw matrix cells
    for i in range(2):
        for j in range(2):
            x = margin + j * cell_size
            y = margin + 40 + i * cell_size
            
            count = matrix[i][j]
            percentage = (count / total * 100) if total > 0 else 0
            
            svg_content += f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{colors[i][j]}" stroke="black" stroke-width="2"/>\n'
            
            # Count and percentage
            text_x = x + cell_size // 2
            text_y = y + cell_size // 2 - 10
            svg_content += f'<text x="{text_x}" y="{text_y}" text-anchor="middle" class="text" font-size="14px">{count}</text>\n'
            svg_content += f'<text x="{text_x}" y="{text_y + 15}" text-anchor="middle" class="text" font-size="12px">({percentage:.1f}%)</text>\n'
            svg_content += f'<text x="{text_x}" y="{text_y + 30}" text-anchor="middle" class="text" font-size="10px">{labels[i][j]}</text>\n'
    
    # Axis labels
    svg_content += f'<text x="{margin + cell_size}" y="{margin + 30}" text-anchor="middle" class="axis-label">Predicted</text>\n'
    svg_content += f'<text x="{margin - 20}" y="{margin + 60 + cell_size}" text-anchor="middle" class="axis-label" transform="rotate(-90 {margin - 20} {margin + 60 + cell_size})">Actual</text>\n'
    
    # Class labels
    svg_content += f'<text x="{margin + cell_size // 2}" y="{margin + 55}" text-anchor="middle" class="text">Red Win</text>\n'
    svg_content += f'<text x="{margin + cell_size + cell_size // 2}" y="{margin + 55}" text-anchor="middle" class="text">Blue Win</text>\n'
    svg_content += f'<text x="{margin - 35}" y="{margin + 60 + cell_size // 2}" text-anchor="middle" class="text">Red Win</text>\n'
    svg_content += f'<text x="{margin - 35}" y="{margin + 60 + cell_size + cell_size // 2}" text-anchor="middle" class="text">Blue Win</text>\n'
    
    svg_content += '</svg>'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(svg_content)
    
    return output_path


def create_roc_curve_chart(true_labels: List[int], predictions: List[float], 
                          title: str, output_path: str) -> str:
    """
    Create a simple ROC curve visualization.
    
    Args:
        true_labels (List[int]): True labels
        predictions (List[float]): Prediction probabilities
        title (str): Chart title
        output_path (str): Output file path
        
    Returns:
        str: Path to created chart
    """
    if not true_labels or not predictions:
        return ""
    
    # Calculate ROC points
    thresholds = [i / 100.0 for i in range(101)]
    roc_points = []
    
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        
        for true_label, pred_prob in zip(true_labels, predictions):
            pred_label = 1 if pred_prob >= threshold else 0
            
            if true_label == 1 and pred_label == 1:
                tp += 1
            elif true_label == 0 and pred_label == 1:
                fp += 1
            elif true_label == 0 and pred_label == 0:
                tn += 1
            else:
                fn += 1
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        roc_points.append((fpr, tpr))
    
    width, height = 500, 500
    margin = 60
    chart_size = min(width, height) - 2 * margin
    
    svg_content = create_svg_chart(width, height, title)
    
    # Draw axes
    svg_content += f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black" stroke-width="2"/>\n'
    svg_content += f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black" stroke-width="2"/>\n'
    
    # Draw diagonal line (random classifier)
    svg_content += f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{margin}" stroke="gray" stroke-width="1" stroke-dasharray="5,5"/>\n'
    
    # Draw ROC curve
    path_data = "M"
    for i, (fpr, tpr) in enumerate(roc_points):
        x = margin + fpr * chart_size
        y = height - margin - tpr * chart_size
        
        if i == 0:
            path_data += f" {x} {y}"
        else:
            path_data += f" L {x} {y}"
    
    svg_content += f'<path d="{path_data}" fill="none" stroke="blue" stroke-width="2"/>\n'
    
    # Calculate AUC (simple trapezoidal rule)
    auc = 0.0
    for i in range(1, len(roc_points)):
        x1, y1 = roc_points[i-1]
        x2, y2 = roc_points[i]
        auc += (x2 - x1) * (y1 + y2) / 2
    
    # Axis labels
    svg_content += f'<text x="{width // 2}" y="{height - 20}" text-anchor="middle" class="axis-label">False Positive Rate</text>\n'
    svg_content += f'<text x="20" y="{height // 2}" text-anchor="middle" class="axis-label" transform="rotate(-90 20 {height // 2})">True Positive Rate</text>\n'
    
    # AUC label
    svg_content += f'<text x="{width - margin - 10}" y="{height - margin - 20}" text-anchor="end" class="text">AUC = {auc:.3f}</text>\n'
    
    # Axis ticks
    for i in range(6):
        tick_value = i / 5
        
        # X-axis ticks
        tick_x = margin + tick_value * chart_size
        svg_content += f'<line x1="{tick_x}" y1="{height - margin}" x2="{tick_x}" y2="{height - margin + 5}" stroke="black"/>\n'
        svg_content += f'<text x="{tick_x}" y="{height - margin + 18}" text-anchor="middle" class="text">{tick_value:.1f}</text>\n'
        
        # Y-axis ticks
        tick_y = height - margin - tick_value * chart_size
        svg_content += f'<line x1="{margin - 5}" y1="{tick_y}" x2="{margin}" y2="{tick_y}" stroke="black"/>\n'
        svg_content += f'<text x="{margin - 10}" y="{tick_y + 3}" text-anchor="end" class="text">{tick_value:.1f}</text>\n'
    
    svg_content += '</svg>'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(svg_content)
    
    return output_path


def create_prediction_visualizations(prediction_results: Dict[str, Any], 
                                   output_dir: str = "visualizations") -> List[str]:
    """
    Create all prediction visualization charts.
    
    Args:
        prediction_results (Dict[str, Any]): Results from fight predictions
        output_dir (str): Output directory for charts
        
    Returns:
        List[str]: List of created chart file paths
    """
    if 'error' in prediction_results:
        return []
    
    created_files = []
    models = prediction_results.get('models', {})
    
    if not models:
        return []
    
    # 1. Model Accuracy Comparison Bar Chart
    accuracy_data = {wc: model['accuracy'] for wc, model in models.items()}
    accuracy_chart = create_bar_chart(
        accuracy_data,
        "Model Accuracy by Weight Class",
        os.path.join(output_dir, "prediction_accuracy_by_weight_class.svg")
    )
    if accuracy_chart:
        created_files.append(accuracy_chart)
    
    # 2. Sample Size Bar Chart
    sample_data = {wc: model['train_size'] for wc, model in models.items()}
    sample_chart = create_bar_chart(
        sample_data,
        "Training Sample Size by Weight Class", 
        os.path.join(output_dir, "training_sample_sizes.svg")
    )
    if sample_chart:
        created_files.append(sample_chart)
    
    # 3. Confusion matrices for best and worst performing models
    sorted_models = sorted(models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    if sorted_models:
        # Best model confusion matrix
        best_wc, best_model = sorted_models[0]
        if 'test_predictions' in best_model and 'test_labels' in best_model:
            best_cm = create_confusion_matrix_chart(
                best_model['test_labels'],
                best_model['test_predictions'],
                0.5,
                f"Confusion Matrix - {best_wc} (Best Model)",
                os.path.join(output_dir, f"confusion_matrix_best_{best_wc.replace(' ', '_')}.svg")
            )
            if best_cm:
                created_files.append(best_cm)
        
        # Worst model confusion matrix
        if len(sorted_models) > 1:
            worst_wc, worst_model = sorted_models[-1]
            if 'test_predictions' in worst_model and 'test_labels' in worst_model:
                worst_cm = create_confusion_matrix_chart(
                    worst_model['test_labels'],
                    worst_model['test_predictions'],
                    0.5,
                    f"Confusion Matrix - {worst_wc} (Most Challenging)",
                    os.path.join(output_dir, f"confusion_matrix_worst_{worst_wc.replace(' ', '_')}.svg")
                )
                if worst_cm:
                    created_files.append(worst_cm)
    
    # 4. ROC curves for selected models
    if sorted_models:
        for i, (wc, model) in enumerate(sorted_models[:3]):  # Top 3 models
            if 'test_predictions' in model and 'test_labels' in model:
                roc_chart = create_roc_curve_chart(
                    model['test_labels'],
                    model['test_predictions'],
                    f"ROC Curve - {wc}",
                    os.path.join(output_dir, f"roc_curve_{wc.replace(' ', '_')}.svg")
                )
                if roc_chart:
                    created_files.append(roc_chart)
    
    return created_files


def generate_html_report(prediction_results: Dict[str, Any], 
                        chart_files: List[str],
                        output_path: str = "prediction_report.html") -> str:
    """
    Generate an HTML report with all prediction analysis and visualizations.
    
    Args:
        prediction_results (Dict[str, Any]): Prediction results
        chart_files (List[str]): List of chart file paths
        output_path (str): Output HTML file path
        
    Returns:
        str: Path to created HTML report
    """
    if 'error' in prediction_results:
        html_content = f"""
        <html><head><title>UFC Prediction Analysis - Error</title></head>
        <body><h1>Error in Prediction Analysis</h1>
        <p>{prediction_results['error']}</p></body></html>
        """
    else:
        summary = prediction_results['summary']
        models = prediction_results['models']
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>UFC Fight Prediction Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; }}
                h3 {{ color: #7f8c8d; }}
                .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; }}
                .model-table {{ border-collapse: collapse; width: 100%; }}
                .model-table th, .model-table td {{ border: 1px solid #bdc3c7; padding: 8px; text-align: left; }}
                .model-table th {{ background-color: #3498db; color: white; }}
                .chart {{ margin: 20px 0; text-align: center; }}
                .accuracy-high {{ color: #27ae60; font-weight: bold; }}
                .accuracy-medium {{ color: #f39c12; font-weight: bold; }}
                .accuracy-low {{ color: #e74c3c; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>UFC Fight Prediction Analysis Report</h1>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <ul>
                    <li><strong>Total weight classes analyzed:</strong> {summary['total_weight_classes']}</li>
                    <li><strong>Models successfully trained:</strong> {summary['trained_models']}</li>
                    <li><strong>Overall average accuracy:</strong> {summary['overall_accuracy']:.3f}</li>
                </ul>
            </div>
            
            <h2>Model Performance by Weight Class</h2>
            <table class="model-table">
                <tr>
                    <th>Weight Class</th>
                    <th>Accuracy</th>
                    <th>Training Samples</th>
                    <th>Test Samples</th>
                    <th>Performance Category</th>
                </tr>
        """
        
        # Sort models by accuracy
        sorted_models = sorted(models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for weight_class, model_info in sorted_models:
            accuracy = model_info['accuracy']
            if accuracy >= 0.6:
                category = '<span class="accuracy-high">High (≥60%)</span>'
            elif accuracy >= 0.5:
                category = '<span class="accuracy-medium">Medium (50-60%)</span>'
            else:
                category = '<span class="accuracy-low">Low (<50%)</span>'
            
            html_content += f"""
                <tr>
                    <td>{weight_class}</td>
                    <td>{accuracy:.3f}</td>
                    <td>{model_info['train_size']}</td>
                    <td>{model_info['test_size']}</td>
                    <td>{category}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Visualizations</h2>
        """
        
        # Add charts
        for chart_file in chart_files:
            if chart_file.endswith('.svg'):
                chart_name = os.path.basename(chart_file).replace('.svg', '').replace('_', ' ').title()
                # Convert to relative path
                rel_path = os.path.relpath(chart_file, os.path.dirname(output_path))
                html_content += f"""
                    <div class="chart">
                        <h3>{chart_name}</h3>
                        <object data="{rel_path}" type="image/svg+xml" width="800" height="600">
                            Chart not available
                        </object>
                    </div>
                """
        
        # Performance insights
        high_perf = [wc for wc, model in models.items() if model['accuracy'] >= 0.6]
        med_perf = [wc for wc, model in models.items() if 0.5 <= model['accuracy'] < 0.6]
        low_perf = [wc for wc, model in models.items() if model['accuracy'] < 0.5]
        
        html_content += f"""
            <h2>Key Insights</h2>
            <ul>
                <li><strong>Best performing weight class:</strong> {sorted_models[0][0]} with {sorted_models[0][1]['accuracy']:.3f} accuracy</li>
                <li><strong>Most challenging weight class:</strong> {sorted_models[-1][0]} with {sorted_models[-1][1]['accuracy']:.3f} accuracy</li>
                <li><strong>High performance models:</strong> {len(high_perf)} weight classes</li>
                <li><strong>Medium performance models:</strong> {len(med_perf)} weight classes</li>
                <li><strong>Low performance models:</strong> {len(low_perf)} weight classes</li>
            </ul>
            
            <h3>Performance Distribution</h3>
            <p>The predictive models show varying effectiveness across different weight classes. 
            This variation could be due to factors such as:</p>
            <ul>
                <li>Sample size differences between weight classes</li>
                <li>Unique fighting styles and meta-games in each division</li>
                <li>Data quality and feature relevance variations</li>
                <li>Competitive balance within divisions</li>
            </ul>
            
            <h2>Methodology</h2>
            <p>This analysis used correlation-based feature selection and simple logistic regression 
            to predict fight outcomes. Each weight class was modeled separately to account for 
            division-specific patterns and fighting styles.</p>
            
            <p><em>Report generated automatically by UFC Fight Analytics Dashboard</em></p>
        </body>
        </html>
        """
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path