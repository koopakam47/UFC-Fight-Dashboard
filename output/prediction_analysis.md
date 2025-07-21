# UFC Fight Prediction Analysis

## Summary
- Total weight classes analyzed: 14
- Models successfully trained: 13
- Overall average accuracy: 0.521

## Model Performance by Weight Class

### OpenWeight
- Accuracy: 1.000
- Training samples: 68
- Test samples: 18
- Key features: B_avg_KD, B_avg_opp_KD, B_avg_SIG_STR_pct, B_avg_opp_SIG_STR_pct, B_avg_TD_pct

### WomenBantamweight
- Accuracy: 0.700
- Training samples: 119
- Test samples: 30
- Key features: B_Height_cms, R_wins, B_avg_SIG_STR_pct, R_current_win_streak, B_avg_KD

### WomenFlyweight
- Accuracy: 0.682
- Training samples: 88
- Test samples: 22
- Key features: R_avg_TD_pct, R_wins, B_avg_KD, B_current_win_streak, B_avg_SIG_STR_pct

### WomenStrawweight
- Accuracy: 0.658
- Training samples: 152
- Test samples: 38
- Key features: R_current_win_streak, B_Height_cms, R_Height_cms, B_avg_SIG_STR_pct, B_avg_KD

### Bantamweight
- Accuracy: 0.645
- Training samples: 369
- Test samples: 93
- Key features: B_Height_cms, R_avg_TD_pct, R_avg_KD, B_current_win_streak, B_avg_TD_pct

### Lightweight
- Accuracy: 0.628
- Training samples: 857
- Test samples: 215
- Key features: R_Height_cms, R_current_win_streak, B_current_win_streak, B_avg_SIG_STR_pct, B_Height_cms

### CatchWeight
- Accuracy: 0.455
- Training samples: 40
- Test samples: 11
- Key features: B_avg_KD, B_avg_opp_KD, B_avg_SIG_STR_pct, B_avg_opp_SIG_STR_pct, B_avg_TD_pct

### Flyweight
- Accuracy: 0.370
- Training samples: 180
- Test samples: 46
- Key features: R_wins, R_avg_TD_pct, B_avg_TD_pct, B_wins, R_avg_KD

### Welterweight
- Accuracy: 0.360
- Training samples: 852
- Test samples: 214
- Key features: B_wins, R_avg_SIG_STR_pct, R_avg_KD, B_current_win_streak, B_Height_cms

### Featherweight
- Accuracy: 0.343
- Training samples: 431
- Test samples: 108
- Key features: R_avg_KD, B_avg_SIG_STR_pct, B_avg_TD_pct, R_wins, R_Height_cms

### LightHeavyweight
- Accuracy: 0.339
- Training samples: 447
- Test samples: 112
- Key features: B_wins, B_current_win_streak, R_avg_TD_pct, R_wins, R_avg_SIG_STR_pct

### Middleweight
- Accuracy: 0.317
- Training samples: 642
- Test samples: 161
- Key features: R_avg_KD, R_current_win_streak, B_Height_cms, B_avg_TD_pct, B_wins

### Heavyweight
- Accuracy: 0.278
- Training samples: 458
- Test samples: 115
- Key features: B_wins, B_avg_SIG_STR_pct, B_avg_TD_pct, R_avg_SIG_STR_pct, B_Height_cms

## Key Insights

**Best performing weight class**: OpenWeight (Accuracy: 1.000)
**Most challenging weight class**: Heavyweight (Accuracy: 0.278)
**Average accuracy across all weight classes**: 0.521

**Performance Distribution**:
- High performance (≥60%): 6 weight classes
- Medium performance (50-60%): 0 weight classes
- Low performance (<50%): 7 weight classes