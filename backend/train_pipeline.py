import os
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'REP.csv'
ARTIFACTS = ROOT / 'artifacts'
OUTPUTS = ROOT / 'outputs'
ARTIFACTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET = 'claim'
N_CLUSTERS = 3
HIERARCHICAL_SAMPLE = 3000
CLUSTER_METRIC_SAMPLE = 3000


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def regression_metrics(y_true, y_pred):
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mape': mape(y_true, y_pred),
    }


def build_preprocessor(X):
    numeric = X.select_dtypes(include=['number']).columns.tolist()
    categorical = X.select_dtypes(exclude=['number']).columns.tolist()

    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, numeric),
        ('cat', categorical_pipe, categorical),
    ], remainder='drop')
    return preprocessor, numeric, categorical


def clean_dataset(df):
    df = df.copy()
    # In this dataset, zero blood pressure is physiologically invalid and is treated as missing.
    if 'bloodpressure' in df.columns:
        df.loc[df['bloodpressure'] <= 0, 'bloodpressure'] = np.nan
    # Basic duplicate removal.
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def feature_engineering(df):
    df = df.copy()
    # Keep these engineered features independent of the target.
    if 'bmi' in df.columns:
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[-np.inf, 18.5, 25, 30, np.inf],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        ).astype('object')
    if 'age' in df.columns:
        df['age_group'] = pd.cut(
            df['age'],
            bins=[-np.inf, 25, 40, 55, np.inf],
            labels=['Young', 'Adult', 'Middle_Age', 'Older']
        ).astype('object')
    
    # Simple count of available binary health indicators.
    binary_cols = [c for c in ['smoker', 'diabetes'] if c in df.columns]
    if binary_cols:
        df['condition_count'] = df[binary_cols].fillna(0).sum(axis=1)

    # Actuarial Risk Interaction Features (Crucial for high precision claim modeling)
    bmi_val = df['bmi'].fillna(df['bmi'].median() if 'bmi' in df.columns else 25.0)
    age_val = df['age'].fillna(df['age'].median() if 'age' in df.columns else 35.0)
    bp_val = df['bloodpressure'].fillna(df['bloodpressure'].median() if 'bloodpressure' in df.columns else 120.0)
    smoker_val = df['smoker'].fillna(0) if 'smoker' in df.columns else 0
    diabetes_val = df['diabetes'].fillna(0) if 'diabetes' in df.columns else 0

    df['smoker_bmi'] = smoker_val * bmi_val
    df['smoker_age'] = smoker_val * age_val
    df['diabetes_bp'] = diabetes_val * bp_val
    df['age_bmi'] = (age_val / 10.0) * (bmi_val / 10.0)
    return df


def make_models():
    return {
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.04, max_depth=4,
            min_samples_leaf=2, random_state=RANDOM_STATE, loss='squared_error'
        ),
        'XGBoost': XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=5,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            objective='reg:squarederror', random_state=RANDOM_STATE,
            n_jobs=-1, tree_method='hist'
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=31,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=300, learning_rate=0.04, depth=6,
            loss_function='RMSE', random_seed=RANDOM_STATE,
            verbose=False, allow_writing_files=False
        ),
    }



def choose_model(results):
    # Primary criterion: lowest validation RMSE; tie-break with MAE.
    return min(results.items(), key=lambda kv: (kv[1]['rmse'], kv[1]['mae']))[0]


def fit_cluster_models(X_train_transformed, y_train, X_val_transformed, y_val, train_clusters, val_clusters):
    model_results = {}
    selected = {}
    final_models = {}

    for risk in ['Low', 'Medium', 'High']:
        tr_mask = train_clusters == risk
        va_mask = val_clusters == risk
        if tr_mask.sum() < 50 or va_mask.sum() < 20:
            raise RuntimeError(f'Not enough samples for {risk} risk: train={tr_mask.sum()}, val={va_mask.sum()}')

        results = {}
        for name, model in make_models().items():
            model.fit(X_train_transformed[tr_mask], y_train[tr_mask])
            pred = model.predict(X_val_transformed[va_mask])
            results[name] = regression_metrics(y_val[va_mask], pred)

        best_name = choose_model(results)
        selected[risk] = best_name
        model_results[risk] = results

        # Refit selected model on all training rows belonging to this risk.
        best_model = make_models()[best_name]
        best_model.fit(X_train_transformed[tr_mask], y_train[tr_mask])
        final_models[risk] = best_model

    return model_results, selected, final_models


def main():
    print('Loading:', DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print('Original shape:', df.shape)
    df = clean_dataset(df)
    df = feature_engineering(df)

    if TARGET not in df.columns:
        raise ValueError(f'Missing target column: {TARGET}')

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=RANDOM_STATE
    )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    # PCA is used only for clustering to make GMM and hierarchical clustering practical.
    n_components = min(20, X_train_t.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_cluster = pca.fit_transform(X_train_t)
    X_val_cluster = pca.transform(X_val_t)
    X_test_cluster = pca.transform(X_test_t)

    # ----- CLUSTERING EXPERIMENTS -----
    rng = np.random.default_rng(RANDOM_STATE)
    metric_n = min(CLUSTER_METRIC_SAMPLE, len(X_train_cluster))
    metric_idx = rng.choice(len(X_train_cluster), size=metric_n, replace=False)
    X_metric = X_train_cluster[metric_idx]
    y_metric = y_train.to_numpy()[metric_idx]

    clustering_models = {
        'KMeans': KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=RANDOM_STATE),
        'GMM': GaussianMixture(n_components=N_CLUSTERS, covariance_type='full',
                               random_state=RANDOM_STATE, n_init=3, reg_covar=1e-5),
    }

    cluster_results = {}
    fitted_cluster_models = {}
    train_cluster_labels = {}

    for name, model in clustering_models.items():
        model.fit(X_train_cluster)
        labels_metric = model.predict(X_metric)
        sil = silhouette_score(X_metric, labels_metric)
        db = davies_bouldin_score(X_metric, labels_metric)
        ch = calinski_harabasz_score(X_metric, labels_metric)
        train_labels = model.predict(X_train_cluster)
        cluster_results[name] = {
            'silhouette': float(sil),
            'davies_bouldin': float(db),
            'calinski_harabasz': float(ch),
            'cluster_sizes': {str(int(k)): int(v) for k, v in zip(*np.unique(train_labels, return_counts=True))}
        }
        fitted_cluster_models[name] = model
        train_cluster_labels[name] = train_labels

    # Hierarchical clustering is evaluated on a sample because full 15k clustering is O(n^2).
    h_n = min(HIERARCHICAL_SAMPLE, len(X_train_cluster))
    h_idx = rng.choice(len(X_train_cluster), size=h_n, replace=False)
    hierarchical = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
    h_labels = hierarchical.fit_predict(X_train_cluster[h_idx])
    cluster_results['Hierarchical'] = {
        'silhouette': float(silhouette_score(X_train_cluster[h_idx], h_labels)),
        'davies_bouldin': float(davies_bouldin_score(X_train_cluster[h_idx], h_labels)),
        'calinski_harabasz': float(calinski_harabasz_score(X_train_cluster[h_idx], h_labels)),
        'cluster_sizes_sample': {str(int(k)): int(v) for k, v in zip(*np.unique(h_labels, return_counts=True))}
    }

    # Select among deployable clustering methods. Hierarchical remains a research benchmark
    # because standard AgglomerativeClustering has no native predict() for unseen customers.
    deployable = {k: v for k, v in cluster_results.items() if k in ['KMeans', 'GMM']}
    best_cluster_method = max(
        deployable,
        key=lambda k: (
            deployable[k]['silhouette'],
            -deployable[k]['davies_bouldin'],
            deployable[k]['calinski_harabasz']
        )
    )
    cluster_model = fitted_cluster_models[best_cluster_method]
    raw_train_labels = train_cluster_labels[best_cluster_method]
    raw_val_labels = cluster_model.predict(X_val_cluster)
    raw_test_labels = cluster_model.predict(X_test_cluster)

    # ----- POST-HOC RISK MAPPING -----
    # Cluster assignment uses only X. Mapping is learned from TRAINING claim distributions.
    cluster_claim_means = {}
    for c in sorted(np.unique(raw_train_labels)):
        cluster_claim_means[int(c)] = float(y_train.to_numpy()[raw_train_labels == c].mean())
    ordered = sorted(cluster_claim_means, key=cluster_claim_means.get)
    risk_map = {int(ordered[0]): 'Low', int(ordered[1]): 'Medium', int(ordered[2]): 'High'}

    train_risk = np.array([risk_map[int(c)] for c in raw_train_labels])
    val_risk = np.array([risk_map[int(c)] for c in raw_val_labels])
    test_risk = np.array([risk_map[int(c)] for c in raw_test_labels])

    # ----- GLOBAL MODEL EXPERIMENT -----
    global_results = {}
    global_selected_name = None
    global_models = make_models()
    for name, model in global_models.items():
        model.fit(X_train_t, y_train.to_numpy())
        pred_val = model.predict(X_val_t)
        global_results[name] = regression_metrics(y_val, pred_val)

    global_selected_name = choose_model(global_results)
    global_final_model = make_models()[global_selected_name]
    global_final_model.fit(X_train_t, y_train.to_numpy())
    global_test_pred = global_final_model.predict(X_test_t)
    global_test_metrics = regression_metrics(y_test, global_test_pred)

    # ----- RISK-AWARE MODEL EXPERIMENT -----
    risk_results, risk_selected, risk_models = fit_cluster_models(
        X_train_t, y_train.to_numpy(), X_val_t, y_val.to_numpy(), train_risk, val_risk
    )

    risk_test_pred = np.zeros(len(X_test_t), dtype=float)
    risk_test_metrics = {}
    for risk in ['Low', 'Medium', 'High']:
        mask = test_risk == risk
        pred = risk_models[risk].predict(X_test_t[mask])
        risk_test_pred[mask] = pred
        risk_test_metrics[risk] = regression_metrics(y_test.to_numpy()[mask], pred)

    overall_risk_metrics = regression_metrics(y_test, risk_test_pred)

    # Save artifacts.
    joblib.dump(preprocessor, ARTIFACTS / 'preprocessor.joblib')
    joblib.dump(pca, ARTIFACTS / 'cluster_pca.joblib')
    joblib.dump(cluster_model, ARTIFACTS / 'cluster_model.joblib')
    joblib.dump(global_final_model, ARTIFACTS / 'global_model.joblib')
    for risk, model in risk_models.items():
        joblib.dump(model, ARTIFACTS / f'{risk.lower()}_model.joblib')

    metadata = {
        'target': TARGET,
        'best_cluster_method': best_cluster_method,
        'risk_map_raw_cluster_to_risk': {str(k): v for k, v in risk_map.items()},
        'cluster_claim_means_train': {str(k): v for k, v in cluster_claim_means.items()},
        'cluster_results': cluster_results,
        'global_validation_results': global_results,
        'global_selected_model': global_selected_name,
        'global_test_metrics': global_test_metrics,
        'risk_validation_results': risk_results,
        'risk_selected_models': risk_selected,
        'risk_test_metrics': risk_test_metrics,
        'risk_aware_overall_test_metrics': overall_risk_metrics,
        'feature_columns': X.columns.tolist(),
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'data_shape_after_cleaning': list(df.shape),
    }
    with open(ARTIFACTS / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    comparison = pd.DataFrame([
        {'approach': 'Global', 'model': global_selected_name, **global_test_metrics},
        {'approach': 'Risk-aware', 'model': 'Per-risk selected', **overall_risk_metrics},
    ])
    comparison.to_csv(OUTPUTS / 'final_comparison.csv', index=False)

    cluster_table = pd.DataFrame([
        {'method': k, **v} for k, v in cluster_results.items()
    ])
    cluster_table.to_json(OUTPUTS / 'cluster_results.json', orient='records', indent=2)

    # Generate Research Plots for Module 2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    (OUTPUTS / 'models').mkdir(parents=True, exist_ok=True)
    (OUTPUTS / 'clustering').mkdir(parents=True, exist_ok=True)

    # 1. Residual Analysis Plot
    residuals = y_test.to_numpy() - global_test_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(global_test_pred, residuals, alpha=0.35, color='#dc2626', edgecolors='none', s=25)
    ax1.axhline(0, color='#0f172a', linestyle='--', linewidth=1.5)
    ax1.set_title('Residuals vs Predicted Values (Global Model)', fontweight='bold', color='#0f172a')
    ax1.set_xlabel('Predicted Claim (₹)')
    ax1.set_ylabel('Residual (Actual - Predicted)')
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2.hist(residuals, bins=40, color='#dc2626', edgecolor='white', alpha=0.85)
    ax2.axvline(0, color='#0f172a', linestyle='--', linewidth=1.5)
    ax2.set_title('Residual Distribution (Centered at Zero)', fontweight='bold', color='#0f172a')
    ax2.set_xlabel('Prediction Error (₹)')
    ax2.set_ylabel('Density / Frequency')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS / 'models' / 'residual_analysis.png', dpi=180)
    plt.close()

    # 2. Actual vs Predicted Scatter
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, global_test_pred, alpha=0.35, color='#2563eb', s=25, label='Holdout Test Predictions')
    min_v = min(y_test.min(), global_test_pred.min())
    max_v = max(y_test.max(), global_test_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], color='#dc2626', linestyle='--', linewidth=2, label='Perfect Fit Line (y = x)')
    ax.set_title(f'Actual vs Predicted Claims (R² = {global_test_metrics["r2"]*100:.2f}%)', fontweight='bold', color='#0f172a')
    ax.set_xlabel('Actual Claim (₹)')
    ax.set_ylabel('Predicted Claim (₹)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS / 'models' / 'actual_vs_predicted.png', dpi=180)
    plt.close()

    # 3. Cluster Profiling Statistics
    train_df_profiling = X_train.copy()
    train_df_profiling['claim'] = y_train.to_numpy()
    train_df_profiling['cluster'] = raw_train_labels
    train_df_profiling['risk_tier'] = train_risk

    cluster_profiles = {}
    for c_id in sorted(np.unique(raw_train_labels)):
        c_sub = train_df_profiling[train_df_profiling['cluster'] == c_id]
        cluster_profiles[str(int(c_id))] = {
            'risk_assigned': risk_map[int(c_id)],
            'size': int(len(c_sub)),
            'avg_claim': float(c_sub['claim'].mean()),
            'median_claim': float(c_sub['claim'].median()),
            'avg_age': float(c_sub['age'].mean()),
            'avg_bmi': float(c_sub['bmi'].mean()),
            'smoker_pct': float(c_sub['smoker'].mean() * 100),
            'diabetes_pct': float(c_sub['diabetes'].mean() * 100),
            'exercise_pct': float(c_sub['regular_ex'].mean() * 100) if 'regular_ex' in c_sub.columns else 0.0,
            'avg_bp': float(c_sub['bloodpressure'].mean()) if 'bloodpressure' in c_sub.columns else 0.0,
        }

    with open(OUTPUTS / 'clustering' / 'cluster_profiles.json', 'w', encoding='utf-8') as f:
        json.dump(cluster_profiles, f, indent=2)

    print('\n=== CLUSTER RESULTS ===')
    print(pd.DataFrame(cluster_results).T[['silhouette', 'davies_bouldin', 'calinski_harabasz']])
    print('\nBest deployable clustering method:', best_cluster_method)
    print('Risk mapping:', risk_map)
    print('\n=== GLOBAL MODEL ===')
    print(pd.DataFrame(global_results).T.sort_values('rmse'))
    print('Selected:', global_selected_name)
    print('Test:', global_test_metrics)
    print('\n=== RISK-AWARE MODEL ===')
    print('Selected models:', risk_selected)
    print(pd.DataFrame(risk_test_metrics).T)
    print('Overall:', overall_risk_metrics)
    print('\nArtifacts saved to:', ARTIFACTS)


if __name__ == '__main__':
    main()

