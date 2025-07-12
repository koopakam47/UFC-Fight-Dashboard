# UFC Fight Analytics Dashboard

The UFC Fight Analytics Dashboard is a data-driven exploration of key performance indicators that correlate with victory across all weight classes in the UFC. This project visualizes the top and bottom statistical drivers of success, non-combat factors, and stance effectiveness. Each weight class is broken down with four visual dashboards:

- **Top 10 Correlations with Winning**
- **Bottom 10 Correlations with Winning**
- **Non-Combat Correlations (e.g., Win Streaks, Age, Titles)**
- **Win Rates by Stance**

## Key Insights by Weight Class

### Bantamweight
![Bantamweight Dashboard](visualizations/Bantamweight_correlation_dashboard.png)

Striking accuracy and ground control are critical. High leg kick volume and takedown success contribute to winning. Fighters using a **Switch stance** had the highest win rate.

---

### Middleweight
![Middleweight Dashboard](visualizations/Middleweight_correlation_dashboard.png)

Aggressive offensive output, especially in total strikes and submission attempts, positively correlates with success. Fighters in a **Switch** or **Southpaw** stance perform best.

---

### Heavyweight
![Heavyweight Dashboard](visualizations/Heavyweight_correlation_dashboard.png)

Power striking metrics matter most. Reach is a surprisingly important factor. Opponent control time and age have a notable negative correlation with winning.

---

### Women’s Strawweight
![WomenStrawweight Dashboard](visualizations/WomenStrawweight_correlation_dashboard.png)

Takedown accuracy and grappling (e.g., submission attempts, ground control) are strong predictors of success. Fighters using a **Switch stance** show the highest win rates.

---

### Women’s Bantamweight
![WomenBantamweight Dashboard](visualizations/WomenBantamweight_correlation_dashboard.png)

Physical attributes (height, weight, reach) and knockout ability are key predictors. Orthodox stance correlates with better outcomes.

---

### Lightweight
![Lightweight Dashboard](visualizations/Lightweight_correlation_dashboard.png)

Success is driven by both offensive grappling and striking metrics. Southpaws hold the edge in win rate, while accuracy and volume are both critical.

---

### Welterweight
![Welterweight Dashboard](visualizations/Welterweight_correlation_dashboard.png)

Winners display high volume and control metrics. Stance appears especially important—**Switch** and **Southpaw** fighters dominate win rates.

---

### Flyweight
![Flyweight Dashboard](visualizations/Flyweight_correlation_dashboard.png)

Offensive body strikes, volume, and accurate striking define success. Win rates favor Orthodox fighters, but Southpaws trail closely.

---

### Light Heavyweight
![LightHeavyweight Dashboard](visualizations/LightHeavyweight_correlation_dashboard.png)

Submission and control metrics drive victory, along with physical traits like reach. Open Stance fighters appear to outperform others.

---

### Featherweight
![Featherweight Dashboard](visualizations/Featherweight_correlation_dashboard.png)

Striking precision and total control time show strong positive correlations. **Southpaw** fighters have the highest win rates, followed by **Switch** stance.

---

### Women’s Flyweight
![WomenFlyweight Dashboard](visualizations/WomenFlyweight_correlation_dashboard.png)

Leg kicks and offensive output are positively correlated. Surprisingly, current win streak had a negative correlation, possibly due to matchup effects.

---

### Women’s Featherweight
![WomenFeatherweight Dashboard](visualizations/WomenFeatherweight_correlation_dashboard.png)

Unique among divisions, opponent striking output shows a positive correlation—likely due to small sample size. Winning is tied to offensive activity and fight experience.

---

### Catchweight
![CatchWeight Dashboard](visualizations/CatchWeight_correlation_dashboard.png)

High-volume striking is a major factor, while opponent striking accuracy and defense are key negatives. Southpaw stance leads win rates.

---

### Openweight
![OpenWeight Dashboard](visualizations/OpenWeight_correlation_dashboard.png)

Victory is driven by grappling (submissions and ground strikes) and experience. Southpaw and Switch stances lead. Physical traits are less predictive here.

---

## Technical Details

- Data was cleaned and enriched from raw UFC stats
- Correlations were computed between each feature and the win outcome (`won`)
- Visualizations generated with Matplotlib and Seaborn
- All stats were mapped to readable forms for accessibility

---

## 🗂 Folder Structure

UFC-Fight-Dashboard/
├── data/
│ └── cleaned_fighter_stats.csv
├── visualizations/
│ └── {WeightClass}_correlation_dashboard.png
├── notebooks/
│ └── analysis_notebooks.ipynb
└── README.md


---

## Future Work

- Incorporate fight-level momentum shifts
- Explore multivariate models and clustering
- Add interactive dashboard with filtering

---

## Contributions

Pull requests are welcome. If you'd like to collaborate, feel free to fork and submit PRs or open an issue to discuss changes.

