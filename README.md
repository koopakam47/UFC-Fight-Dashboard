# UFC-Fight-Dashboard
UFC Fight Analytics &amp; Victory Predictor Dashboard
## Lightweight Fight Correlation Dashboard

This dashboard analyzes which fighter statistics are most strongly associated with victory in UFC lightweight bouts. The visualization presents the top ten metrics that correlate positively with winning outcomes, as well as the ten metrics most negatively correlated with winning. These insights help identify which attributes and performance indicators are most predictive of success inside the octagon.

The underlying dataset was transformed from a wide format into a long format to evaluate each fighter independently and eliminate red vs. blue corner bias. Correlations were calculated using only numerical columns, and fights with missing or inconclusive outcomes were excluded to ensure clean data. To improve clarity, all variables have been relabeled with descriptive, human-readable names. This visualization serves as part of a broader effort to uncover data-driven patterns that contribute to winning performances in mixed martial arts.

![Lightweight Correlation Dashboard](notebooks/lightweight_correlation_dashboard.png)
