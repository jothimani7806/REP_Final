import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'REP.csv'
OUTPUTS_EDA = ROOT / 'outputs' / 'eda'
OUTPUTS_MODELS = ROOT / 'outputs' / 'models'
OUTPUTS_CLUSTERING = ROOT / 'outputs' / 'clustering'

OUTPUTS_EDA.mkdir(parents=True, exist_ok=True)
OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)
OUTPUTS_CLUSTERING.mkdir(parents=True, exist_ok=True)


def run_eda_and_research():
    print("Starting Module 1 & 2 Research Asset Generation...")
    df_raw = pd.read_csv(DATA_PATH)
    raw_shape = list(df_raw.shape)

    # 1. Data Quality Analysis
    missing_before = df_raw.isnull().sum().to_dict()
    duplicates_count = int(df_raw.duplicated().sum())

    # Data Cleaning
    df = df_raw.copy()
    if 'bloodpressure' in df.columns:
        invalid_bp = int((df['bloodpressure'] <= 0).sum())
        df.loc[df['bloodpressure'] <= 0, 'bloodpressure'] = np.nan
    else:
        invalid_bp = 0

    df = df.drop_duplicates().reset_index(drop=True)
    cleaned_shape = list(df.shape)

    # Dataset Report JSON
    target_stats = {
        'mean': float(df['claim'].mean()),
        'median': float(df['claim'].median()),
        'std': float(df['claim'].std()),
        'min': float(df['claim'].min()),
        'max': float(df['claim'].max()),
        'q25': float(df['claim'].quantile(0.25)),
        'q75': float(df['claim'].quantile(0.75)),
        'skew': float(df['claim'].skew())
    }

    report = {
        'dataset_title': 'REP Health Insurance Claims Dataset',
        'raw_shape': raw_shape,
        'cleaned_shape': cleaned_shape,
        'duplicate_records_removed': duplicates_count,
        'invalid_bp_records_treated': invalid_bp,
        'missing_values_before_cleaning': missing_before,
        'target_variable': 'claim',
        'target_statistics': target_stats,
        'categorical_columns': ['sex', 'hereditary_diseases', 'city', 'job_title'],
        'binary_columns': ['smoker', 'diabetes', 'regular_ex'],
        'numerical_columns': ['age', 'weight', 'bmi', 'bloodpressure', 'no_of_dependents', 'claim']
    }

    with open(OUTPUTS_EDA / 'dataset_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    # 2. EDA Plots Generation (Clean Publication Style)
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # A: Target Distribution (Histogram & KDE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(df['claim'], bins=40, color='#dc2626', edgecolor='white', alpha=0.85)
    ax1.set_title('Claim Distribution (Right-Skewed Actuarial Target)', fontweight='bold', color='#0f172a')
    ax1.set_xlabel('Claim Amount (₹)')
    ax1.set_ylabel('Customer Count')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    bp = ax2.boxplot(df['claim'], vert=True, patch_artist=True,
                     boxprops=dict(facecolor='#fee2e2', color='#dc2626'),
                     medianprops=dict(color='#991b1b', linewidth=2))
    ax2.set_title('Claim Boxplot & Outlier Detection', fontweight='bold', color='#0f172a')
    ax2.set_ylabel('Claim Amount (₹)')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUTS_EDA / 'target_claim_distribution.png', dpi=180)
    plt.close()

    # B: Actuarial Relationships (Smoker, BMI, Age, BP vs Claim)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Smoker vs Claim
    smoker_means = df.groupby('smoker')['claim'].mean()
    axes[0, 0].bar(['Non-Smoker (0)', 'Smoker (1)'], smoker_means.values, color=['#10b981', '#dc2626'], width=0.5)
    axes[0, 0].set_title('Mean Claim by Smoking Status', fontweight='bold', color='#0f172a')
    axes[0, 0].set_ylabel('Average Claim (₹)')
    for i, v in enumerate(smoker_means.values):
        axes[0, 0].text(i, v + 800, f'₹{v:,.0f}', ha='center', fontweight='bold')
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.4)

    # BMI vs Claim scatter
    axes[0, 1].scatter(df['bmi'], df['claim'], alpha=0.3, c=df['smoker'], cmap='coolwarm', edgecolors='none', s=25)
    axes[0, 1].set_title('BMI vs Claim (Color-Coded by Smoker)', fontweight='bold', color='#0f172a')
    axes[0, 1].set_xlabel('BMI')
    axes[0, 1].set_ylabel('Claim Amount (₹)')
    axes[0, 1].grid(True, linestyle='--', alpha=0.3)

    # Age vs Claim scatter
    axes[1, 0].scatter(df['age'], df['claim'], alpha=0.3, color='#dc2626', s=25)
    axes[1, 0].set_title('Age vs Claim Trajectory', fontweight='bold', color='#0f172a')
    axes[1, 0].set_xlabel('Age (years)')
    axes[1, 0].set_ylabel('Claim Amount (₹)')
    axes[1, 0].grid(True, linestyle='--', alpha=0.3)

    # Blood Pressure vs Claim
    axes[1, 1].scatter(df['bloodpressure'], df['claim'], alpha=0.3, color='#475569', s=25)
    axes[1, 1].set_title('Blood Pressure vs Claim', fontweight='bold', color='#0f172a')
    axes[1, 1].set_xlabel('Blood Pressure (mmHg)')
    axes[1, 1].set_ylabel('Claim Amount (₹)')
    axes[1, 1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUTS_EDA / 'feature_claim_relationships.png', dpi=180)
    plt.close()

    # C: Correlation Matrix Heatmap
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap='Reds', vmin=-0.2, vmax=1.0)
    fig.colorbar(cax)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha='left')
    ax.set_yticklabels(num_cols)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.iloc[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color='white' if abs(val) > 0.5 else 'black', fontsize=9)
    plt.title('Correlation Matrix of Actuarial Features', pad=25, fontweight='bold', color='#0f172a')
    plt.tight_layout()
    plt.savefig(OUTPUTS_EDA / 'correlation_matrix.png', dpi=180)
    plt.close()

    print("EDA assets saved to:", OUTPUTS_EDA)


if __name__ == '__main__':
    run_eda_and_research()
