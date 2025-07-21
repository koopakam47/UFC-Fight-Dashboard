# UFC Fight Prediction Tool

An easy-to-use tool for predicting upcoming UFC fight outcomes using machine learning models trained on historical fighter statistics.

## Quick Start

1. **Train the models** (if not already done):
   ```bash
   python test_predictions.py
   ```

2. **Predict upcoming fights**:
   ```bash
   # Interactive mode - guided input for fighter stats
   python predict_upcoming_fight.py --interactive
   
   # Show example prediction
   python predict_upcoming_fight.py --example
   
   # List available weight classes  
   python predict_upcoming_fight.py --list-classes
   ```

## How It Works

The prediction system uses:
- **Correlation-based feature selection** from existing victory correlation analysis
- **Weight class-specific models** trained separately for each division
- **Logistic regression** for binary classification (Red vs Blue corner)
- **Fighter statistics** like strike accuracy, takedown percentage, win streaks, etc.

## Fighter Statistics Required

When using interactive mode, you'll be prompted for these statistics for both fighters:

- **Significant Strike Accuracy (%)** - Percentage of significant strikes that land
- **Takedown Accuracy (%)** - Percentage of takedown attempts that succeed  
- **Total Wins** - Fighter's career win count
- **Current Win Streak** - Number of consecutive wins
- **Average Knockdowns per Fight** - Knockdowns scored per fight
- **Average Opponent Knockdowns per Fight** - Knockdowns suffered per fight
- **Average Opponent Sig Strike Accuracy (%)** - Quality of opposition faced
- **Height (cm)** - Fighter height in centimeters

## Example Usage

```bash
# Interactive prediction
$ python predict_upcoming_fight.py --interactive

=== UFC Fight Prediction Tool ===

Available weight classes:
  1. Bantamweight
  2. Lightweight  
  3. Welterweight
  # ... more options

Select weight class (1-13): 1

Selected weight class: Bantamweight

=== Red Corner Fighter ===
Significant Strike Accuracy (%): 45.2
Takedown Accuracy (%): 35.8
# ... enter remaining stats

=== Blue Corner Fighter ===  
# ... enter blue fighter stats

=== Prediction Results ===
Weight Class: Bantamweight
Predicted Winner: Red Corner
Confidence: 67.3%

Detailed Probabilities:
  Red Corner: 67.3%
  Blue Corner: 32.7%

Model Info:
  Training Accuracy: 64.5%
  Training Samples: 369
```

## Programmatic Usage

You can also use the prediction function directly in Python:

```python
from ufc_analytics.fight_predictions import predict_upcoming_fight

# Define fighter stats
red_fighter = {
    'avg_SIG_STR_pct': 45.2,
    'avg_TD_pct': 35.8,
    'wins': 15,
    'current_win_streak': 3,
    'avg_KD': 0.8,
    'Height_cms': 180
}

blue_fighter = {
    'avg_SIG_STR_pct': 42.1,
    'avg_TD_pct': 28.5,
    'wins': 12,
    'current_win_streak': 1,
    'avg_KD': 0.5,
    'Height_cms': 175
}

# Make prediction
result = predict_upcoming_fight(red_fighter, blue_fighter, 'Bantamweight')

print(f"Winner: {result['predicted_winner']}")
print(f"Confidence: {result['confidence']:.1%}")
```

## Model Performance

The system has trained models for 13+ weight classes with varying performance:

**Top Performing Weight Classes:**
- Women's divisions typically show 65-70% accuracy
- Bantamweight: ~64.5% accuracy
- Lightweight: ~62.8% accuracy

**Most Challenging:**
- Heavyweight and Middleweight: ~28-32% accuracy (high variability)

## Notes

- **Model accuracy varies by weight class** - some divisions are more predictable than others
- **Predictions are probabilities** - not guarantees of outcomes
- **Feature importance** differs between weight classes based on correlation analysis
- **Missing statistics** default to 0.0 (may affect prediction quality)
- **Models are retrained** each time you run the training pipeline with updated data

## Generated Files

The prediction system creates:
- `output/prediction_results.json` - Trained model data
- `output/prediction_report.html` - Detailed analysis report
- `output/prediction_analysis.md` - Markdown summary
- `visualizations/prediction_*.svg` - Performance charts