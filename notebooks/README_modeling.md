# UFC Fight Outcome Prediction - 2_modeling.ipynb

## Overview
This notebook implements machine learning models to predict UFC fight outcomes based on fighter statistics and historical performance data.

## Requirements
Before running this notebook, ensure you have the following dependencies installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Data Requirements
The notebook expects a CSV file at one of these locations:
- `../data/data.csv`
- `./data/data.csv` 
- `data/data.csv`

The data should contain:
- Fighter statistics with R_ (red corner) and B_ (blue corner) prefixes
- A 'Winner' column indicating either 'Red' or 'Blue'
- Fight metadata (date, location, weight_class, etc.)

## Notebook Structure

### 1. Data Loading and Exploration
- Loads UFC fight data with error handling
- Performs initial data exploration
- Cleans column names and handles special characters

### 2. Data Preprocessing
- Creates binary target variable (red_wins)
- Handles missing values using median imputation
- Selects numeric features and handles categorical variables
- Creates dummy variables for stance and weight class

### 3. Feature Engineering
- Identifies relevant features for modeling
- Handles missing values
- Scales features using StandardScaler

### 4. Model Training
Tests multiple machine learning algorithms:
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble tree-based model
- **Gradient Boosting**: Advanced ensemble method
- **Support Vector Machine**: Non-linear classifier

### 5. Model Evaluation
- Accuracy scores on test set
- AUC-ROC scores for binary classification
- Cross-validation scores for robustness
- Confusion matrices and classification reports

### 6. Feature Importance Analysis
- Identifies most predictive features
- Visualizes feature importance for tree-based models

### 7. Hyperparameter Tuning
- Grid search for optimal parameters
- Compares tuned vs default model performance

### 8. Results Visualization
- Model performance comparison charts
- ROC curves
- Calibration plots
- Feature importance plots

## Expected Output
When run successfully, the notebook will:
1. Load and preprocess the UFC fight data
2. Train and evaluate 4 different ML models
3. Identify the best performing model
4. Show feature importance rankings
5. Provide comprehensive model evaluation metrics
6. Save the best model (if models directory exists)

## Troubleshooting

### Missing Dependencies
If you get import errors, install the required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Data File Not Found
Ensure the UFC fight data CSV file is in one of the expected locations:
- Check that `data/data.csv` exists
- Verify the file contains the expected columns (R_fighter, B_fighter, Winner, etc.)

### Column Name Issues
The notebook automatically handles common column name issues:
- Removes BOM characters (≤, ﻿)
- Strips whitespace
- Cleans special characters

### Memory Issues
For large datasets, you may need to:
- Reduce the number of features
- Use smaller hyperparameter grids
- Implement feature selection

## Model Performance Expectations
With proper UFC fight data, you can expect:
- **Accuracy**: 60-70% (better than random 50%)
- **AUC Score**: 0.65-0.75
- **Best Models**: Usually Random Forest or Gradient Boosting

## Files Generated
- Model pickle files (if models/ directory exists)
- Feature importance rankings
- Performance comparison charts

## Next Steps
After running this notebook successfully:
1. Experiment with additional features
2. Try ensemble methods
3. Implement time-series validation for temporal data
4. Deploy the model for predictions