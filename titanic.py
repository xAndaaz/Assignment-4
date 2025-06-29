import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler

# Set seaborn style for clean, professional visuals
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Load the Titanic dataset
df = sns.load_dataset('titanic')
print(df.head())
# Feature Engineering: Create family_size
df['family_size'] = df['sibsp'] + df['parch'] + 1

# 1. Missing Values Heatmap
def missing_values_heatmap():
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Values in Titanic Dataset')
    plt.tight_layout()
    plt.savefig('missing_values_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Histograms for Numerical Features
def plot_histograms():
    numerical_cols = ['age', 'fare', 'family_size']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(numerical_cols):
        sns.histplot(data=df, x=col, kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribution of {col.replace("_", " ").title()}')
        axes[i].set_xlabel(col.replace("_", " ").title())
        axes[i].set_ylabel('Count')
    plt.tight_layout()
    plt.savefig('histograms.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Box Plots for Outlier Detection
def plot_box_plots():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(data=df, x='pclass', y='age', ax=axes[0], palette='muted')
    axes[0].set_title('Age by Passenger Class')
    axes[0].set_xlabel('Passenger Class')
    axes[0].set_ylabel('Age')
    
    sns.boxplot(data=df, x='pclass', y='fare', ax=axes[1], palette='muted')
    axes[1].set_title('Fare by Passenger Class')
    axes[1].set_xlabel('Passenger Class')
    axes[1].set_ylabel('Fare')
    
    sns.boxplot(data=df, x='pclass', y='family_size', ax=axes[2], palette='muted')
    axes[2].set_title('Family Size by Passenger Class')
    axes[2].set_xlabel('Passenger Class')
    axes[2].set_ylabel('Family Size')
    plt.tight_layout()
    plt.savefig('box_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Correlation Heatmap
def correlation_heatmap():
    numerical_cols = ['age', 'fare', 'sibsp', 'parch', 'family_size']
    corr = df[numerical_cols].corr()
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                cbar_kws={'label': 'Correlation Coefficient'}, square=True)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Survival Rate Bar Plots
def survival_bar_plots():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.countplot(data=df, x='sex', hue='survived', ax=axes[0], palette='Set2')
    axes[0].set_title('Survival by Sex')
    axes[0].set_xlabel('Sex')
    axes[0].set_ylabel('Count')
    
    sns.countplot(data=df, x='pclass', hue='survived', ax=axes[1], palette='Set2')
    axes[1].set_title('Survival by Passenger Class')
    axes[1].set_xlabel('Passenger Class')
    axes[1].set_ylabel('Count')
    
    sns.countplot(data=df, x='embark_town', hue='survived', ax=axes[2], palette='Set2')
    axes[2].set_title('Survival by Embarkation Town')
    axes[2].set_xlabel('Embarkation Town')
    axes[2].set_ylabel('Count')
    plt.tight_layout()
    plt.savefig('survival_bar_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Stacked Bar Plot with Percentage Annotations
def stacked_bar_plot():
    # Calculate survival percentages by class and sex
    survival_rates = df.groupby(['pclass', 'sex'])['survived'].mean().unstack()
    survival_rates = survival_rates * 100  # Convert to percentages
    
    fig, ax = plt.subplots(figsize=(10, 6))
    survival_rates.plot(kind='bar', stacked=True, ax=ax, color=['#ff9999', '#66b3ff'])
    ax.set_title('Survival Rate (%) by Passenger Class and Sex')
    ax.set_xlabel('Passenger Class')
    ax.set_ylabel('Survival Rate (%)')
    ax.legend(title='Sex')
    
    # Add percentage labels
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if height > 0:
            ax.text(x + width/2, y + height/2, f'{height:.1f}%', 
                    ha='center', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('stacked_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# 7. Pairplot for Numerical Features
def pairplot():
    numerical_cols = ['age', 'fare', 'family_size', 'survived']
    pair_plot = sns.pairplot(df[numerical_cols], hue='survived', palette='Set2', 
                             diag_kind='kde', plot_kws={'alpha': 0.6})
    pair_plot.fig.suptitle('Pairplot of Numerical Features by Survival', y=1.02)
    pair_plot.fig.savefig('pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()

# 8. Statistical Summary and Hypothesis Testing
def statistical_analysis():
    numerical_cols = ['age', 'fare', 'family_size']
    summary = df[numerical_cols].describe()
    
    # T-test for age between survivors and non-survivors
    age_survived = df[df['survived'] == 1]['age'].dropna()
    age_not_survived = df[df['survived'] == 0]['age'].dropna()
    t_stat, p_value = ttest_ind(age_survived, age_not_survived, equal_var=False)
    
    with open('statistical_summary.txt', 'w') as f:
        f.write("Summary Statistics:\n\n")
        f.write(summary.to_string())
        f.write("\n\nT-test for Age (Survivors vs Non-Survivors):\n")
        f.write(f"T-statistic: {t_stat:.4f}\n")
        f.write(f"P-value: {p_value:.4f}\n")
        f.write("Note: P-value < 0.05 suggests significant age difference.")

# 9. Outlier Detection using IQR
def detect_outliers():
    outlier_info = {}
    for col in ['age', 'fare', 'family_size']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_info[col] = {'count': len(outliers), 'lower_bound': lower_bound, 'upper_bound': upper_bound}
    
    with open('outlier_summary.txt', 'w') as f:
        for col, info in outlier_info.items():
            f.write(f"{col.replace('_', ' ').title()} Outliers:\n")
            f.write(f"Count: {info['count']}\n")
            f.write(f"Lower Bound: {info['lower_bound']:.2f}, Upper Bound: {info['upper_bound']:.2f}\n\n")

# Generate all visualizations and analyses
if __name__ == '__main__':
    missing_values_heatmap()
    plot_histograms()
    plot_box_plots()
    correlation_heatmap()
    survival_bar_plots()
    stacked_bar_plot()
    pairplot()
    statistical_analysis()
    detect_outliers()
    print("Visualizations saved as PNGs: missing_values_heatmap.png, histograms.png, box_plots.png, correlation_heatmap.png, survival_bar_plots.png, stacked_bar_plot.png, pairplot.png")
    print("Summaries saved as: statistical_summary.txt, outlier_summary.txt")