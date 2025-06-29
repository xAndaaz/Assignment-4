# Titanic Dataset EDA - Assignment-4


## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Files](#files)
- [Insights from Outputs](#insights-from-outputs)
- [Notes](#notes)

## Overview
This project conducts an in-depth Exploratory Data Analysis (EDA) on the Titanic dataset using `sns.load_dataset('titanic')` with `matplotlib`, `seaborn`, and `scipy`. I explored distributions, missing values, outliers, and relationships via histograms, box plots, heatmaps, bar plots, a stacked bar plot, and pairplot. I engineered a `family_size` feature and used a t-test (p<0.05) to confirm age differences between survivors and non-survivors. The analysis revealed skewed age (20–30 peak), high fare outliers (>500), and survival patterns favoring females (70%) and Class 1 (60%), creating a compelling portfolio piece.

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
   python titanic_eda.py
   ```
   or open `titanic_eda.ipynb` in Jupyter.
2. Outputs include high-quality PNGs and text files.
3. View visualizations in the repository.

## Files
- `titanic_eda.py`: Script for EDA and visualizations.
- `titanic_eda.ipynb`: Notebook with inline results.
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

## Insights from Outputs
- **Missing Values (missing_values_heatmap.png)**: Over 75% of `deck` data is missing, making it unreliable, while ~20% of `age` is absent, especially in Class 3, suggesting recording bias and a need for imputation (e.g., median 28).
- **Distributions (histograms.png)**: Age peaks at 20–30, fare is skewed with outliers >500, and `family_size` clusters at 1–2, with few large families (4+).
- **Outliers (box_plots.png, age_outlier_scatter.png)**: Ages >65 (11 outliers globally, 55+ per class), fares >100 (116 outliers), and family sizes >3.5 (91 outliers) highlight rare cases, with Class 1 showing more age outliers.
- **Correlations (correlation_heatmap.png)**: Weak fare-survival link (0.1–0.2), strong `family_size`-`sibsp`/`parch` correlation (0.6–0.7).
- **Survival Patterns (survival_bar_plots.png, stacked_bar_plot.png)**: Females survived at 70%, Class 1 at 60%, with 95% Class 1 female survival vs. 15% Class 3 male, per stacked plot.
- **Pairwise Insights (pairplot.png)**: Lower age and fare (<40, <50) correlate with survival, seen in KDE overlaps.
- **Stats (statistical_summary.txt)**: Median age 28, mean fare 32.20, t-test p=0.03 shows younger survivors (~28) vs. non-survivors (~31).

## Notes
- Dataset loaded via `sns.load_dataset('titanic')`, no CSV needed.
- Requires `jupyter` for notebook (`pip install jupyter`).
- PNGs are 300 DPI for clarity.
- Outlier thresholds (e.g., 64.81 for age) are global; class-specific outliers (55+) reflect local distributions.