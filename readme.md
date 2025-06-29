# Titanic Dataset EDA - Assignment-4


## Table of Contents
- [Overview](#overview)
- [Insights from Outputs](#insights-from-outputs)
- [Requirements](#requirements)
- [Usage](#usage)
- [Files](#files)

## Overview
This project conducts an in-depth Exploratory Data Analysis (EDA) on the Titanic dataset using `sns.load_dataset('titanic')` with `matplotlib`, `seaborn`, and `scipy`. I explored distributions, missing values, outliers, and relationships via histograms, box plots, heatmaps, bar plots, a stacked bar plot, and pairplot. I engineered a `family_size` feature and used a t-test (p<0.05) to confirm age differences between survivors and non-survivors. The analysis revealed skewed age (20–30 peak), high fare outliers (>500), and survival patterns favoring females (70%) and Class 1 (60%), creating a compelling portfolio piece.

## Insights from Outputs
- **Missing Values (missing_values_heatmap.png)**: The deck column is missing in over 75% of the entries, making it too sparse for reliable analysis. Around 20% of age values are also missing, primarily from passengers in Class 3, which may indicate a recording bias. These gaps suggest the need for imputation, with a median age of 28 being a practical choice.
- **Distributions (histograms.png)**: Age peaks at 20–30, fare is skewed with outliers >500, and `family_size` clusters at 1–2, with few large families (4+).
- **Outliers (box_plots.png, age_outlier_scatter.png)**: A small number of passengers were aged above 65 (11 global outliers) whereas in classwise outlier were observed above 55), Fares over 100 were paid by 116 passengers, and family sizes >3.5 (91 outliers) highlight rare cases, with Class 1 showing more age outliers.
- **Correlations (correlation_heatmap.png)**: There’s only a weak correlation between fare and survival (0.1–0.2), suggesting higher fares alone didn’t strongly predict survival
- **Survival Patterns (survival_bar_plots.png, stacked_bar_plot.png)**: Females survived at 70%, Class 1 at 60%, with 95% Class 1 female survival vs. 15% Class 3 male, per stacked plot.
- **Pairwise Insights (pairplot.png)**: Lower age and fare (<40, <50) correlate with survival, seen in KDE overlaps.
- **Stats (statistical_summary.txt)**: Median age 28, mean fare 32.20, t-test p=0.03 shows younger survivors (~28) vs. non-survivors (~31).

## Requirements
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`
- Install dependencies:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn scipy
  ```

## Usage
1. Run the script or notebook:
   ```bash
   python titanic.py
   ```
2. Outputs include high-quality PNGs and text files.
3. View visualizations in the repository.


## Files
- `titanic.py`: Script for EDA and visualizations.
- PNGs:
  - `missing_values_heatmap.png`: Missing data patterns.
  - `histograms.png`: Age, fare, family size distributions.
  - `box_plots.png`: Outliers by passenger class.
  - `correlation_heatmap.png`: Numerical feature correlations.
  - `survival_bar_plots.png`: Survival by sex, class, embarkation.
  - `stacked_bar_plot.png`: Survival % by class and sex.
  - `pairplot.png`: Pairwise relationships by survival.
  - `age_outlier_scatter.png`: Age outliers vs. class.
- `statistical_summary.txt`: Stats and t-test results.
- `outlier_summary.txt`: Outlier counts.
