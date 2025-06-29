Titanic Dataset EDA
 
Table of Contents

Overview
Requirements
Usage
Files
Insights from Outputs
Notes

Overview
This project performs an in-depth EDA on the Titanic dataset to explore passenger data and survival patterns using matplotlib, seaborn, and scipy. I analyzed distributions, missing values, outliers, and relationships with visualizations like histograms, box plots, heatmaps, bar plots, and a pairplot. I added a family_size feature and a t-test for age differences, producing high-quality PNGs and text summaries.
Requirements

Python 3.8+
Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy
Install dependencies:pip install pandas numpy matplotlib seaborn scikit-learn scipy



Usage

Place titanic.csv in the working directory.
Run the script or notebook:python titanic_eda.py

or open titanic_eda.ipynb in Jupyter.
Outputs:
PNGs: missing_values_heatmap.png, histograms.png, box_plots.png, correlation_heatmap.png, survival_bar_plots.png, stacked_bar_plot.png, pairplot.png, age_outlier_scatter.png
Text: statistical_summary.txt, outlier_summary.txt


View PNGs in the repository.

Files

titanic_eda.py: Script for EDA and visualizations.
titanic_eda.ipynb: Notebook with inline results.
PNGs:
missing_values_heatmap.png: Shows missing data patterns.
histograms.png: Age, fare, and family size distributions.
box_plots.png: Age, fare, and family size by passenger class for outlier detection.
correlation_heatmap.png: Correlations between numerical features.
survival_bar_plots.png: Survival rates by sex, class, and embarkation town.
stacked_bar_plot.png: Survival percentages by class and sex.
pairplot.png: Pairwise relationships by survival.
age_outlier_scatter.png: Age vs. class with global outliers highlighted.


statistical_summary.txt: Summary statistics and t-test results.
outlier_summary.txt: Outlier counts for age, fare, family_size.

Insights from Outputs

Missing Values Heatmap (missing_values_heatmap.png): The heatmap revealed significant missing data in deck (over 70% missing), suggesting it’s unreliable for analysis, and moderate missingness in age (around 20%), indicating a need for imputation (e.g., with median age of 28) before statistical tests. Missing age appeared more frequent in Class 3, hinting at possible recording biases for lower-class passengers.
Histograms (histograms.png): The age distribution is right-skewed with a peak around 20–30, while fare shows a long tail with high values, and family size peaks at 1–2, with few large families (4+), suggesting most passengers traveled alone or with small groups.
Box Plots (box_plots.png): Outliers in age start around 60+ for Class 1 and 55+ for Classes 2 and 3, indicating older passengers were rare. Fare outliers (above ~65) are prominent in Class 1, reflecting luxury tickets, while family size outliers (4+) appear across all classes, with Class 3 showing more variability.
Correlation Heatmap (correlation_heatmap.png): A weak positive correlation (0.1–0.2) between fare and survived suggests higher fares might slightly improve survival odds, while family_size and sibsp/parch show moderate correlation (0.6–0.7), confirming they measure related family aspects.
Survival Bar Plots (survival_bar_plots.png): Survival was higher for females (around 70%) than males (20%), better in Class 1 (60%) than Class 3 (25%), and slightly higher from Cherbourg (40%) than Southampton (33%), indicating gender, class, and embarkation influenced survival.
Stacked Bar Plot (stacked_bar_plot.png): Survival rates show Class 1 females at ~95%, Class 1 males at ~35%, Class 3 females at ~50%, and Class 3 males at ~15%, highlighting a strong class-sex survival gradient, with annotations making these trends clear.
Pairplot (pairplot.png): The pairplot revealed that survivors tend to have lower fare and age (under 40), with a diagonal clustering in KDE plots, suggesting younger, lower-fare passengers had better survival chances.
Age Outlier Scatter (age_outlier_scatter.png): The scatter plot identified 11 global age outliers above 64.81 (mostly in Class 1), aligning with the outlier summary, while class-specific box plots showed outliers at 55+ for Classes 2 and 3, reflecting varying age distributions across classes.
Statistical Summary (statistical_summary.txt): The average age is ~30, fare ~32, and family size ~2, with a t-test p-value of ~0.03, indicating a significant age difference between survivors (younger, ~28) and non-survivors (older, ~31).
Outlier Summary (outlier_summary.txt): 11 age outliers above 64.81 (likely elderly), 116 fare outliers above 65.63 (high-end tickets), and 91 family size outliers above 3.50 (large families) suggest these groups were unusual in the dataset.

Notes

Ensure titanic.csv has columns: survived, pclass, sex, age, sibsp, parch, fare, embarked, class, who, deck, embark_town.
The notebook requires jupyter (pip install jupyter).
Visualizations are saved as 300 DPI PNGs for clarity.
Outliers are calculated globally (e.g., age > 64.81), while box plots show class-specific thresholds (e.g., ~55+), reflecting different analytical perspectives.
