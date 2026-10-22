import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'
OUTPUTS = ROOT / 'outputs' / 'xai'
OUTPUTS.mkdir(parents=True, exist_ok=True)


def engineer(df):
    df = df.copy()
    if 'bmi' in df.columns:
        df['bmi_category'] = pd.cut(df['bmi'], [-float('inf'), 18.5, 25, 30, float('inf')], labels=['Underweight','Normal','Overweight','Obese']).astype('object')
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], [-float('inf'), 25, 40, 55, float('inf')], labels=['Young','Adult','Middle_Age','Older']).astype('object')
    
    binary_cols = [c for c in ['smoker', 'diabetes'] if c in df.columns]
    if binary_cols:
        df['condition_count'] = df[binary_cols].fillna(0).sum(axis=1)

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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='CSV containing customer rows without claim')
    parser.add_argument('--risk', choices=['Low','Medium','High'], default='High')
    parser.add_argument('--row', type=int, default=0)
    args = parser.parse_args()

    with open(ARTIFACTS / 'metadata.json', encoding='utf-8') as f:
        meta = json.load(f)
    pre = joblib.load(ARTIFACTS / 'preprocessor.joblib')
    model = joblib.load(ARTIFACTS / f'{args.risk.lower()}_model.joblib')

    raw = pd.read_csv(args.csv).drop(columns=['claim'], errors='ignore')
    raw = engineer(raw)
    x = raw.iloc[[args.row]]
    xt = pre.transform(x)
    feature_names = pre.get_feature_names_out()

    import numpy as np

    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(xt)
    if isinstance(shap_raw, list):
        shap_values = np.array(shap_raw[0])
    else:
        shap_values = np.array(shap_raw)
    if shap_values.ndim > 1:
        shap_values = shap_values[0]

    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        expected_val = float(np.ravel(expected_val)[0])
    else:
        expected_val = float(expected_val)

    shap_df = pd.DataFrame({'feature': feature_names, 'shap_value': shap_values})
    shap_df['abs_shap'] = shap_df['shap_value'].abs()
    shap_df.sort_values('abs_shap', ascending=False).head(20).to_csv(OUTPUTS / f'{args.risk.lower()}_shap_top20.csv', index=False)

    plt.figure(figsize=(10, 6))
    explanation = shap.Explanation(
        values=shap_values,
        base_values=expected_val,
        data=xt[0],
        feature_names=feature_names
    )
    shap.plots.waterfall(explanation, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(OUTPUTS / f'{args.risk.lower()}_shap_waterfall.png', dpi=180, bbox_inches='tight')
    plt.close()

    # Subsample data for ICE / PDP to make calculation fast and clean
    sample_df = raw.sample(n=min(150, len(raw)), random_state=42).copy()
    from sklearn.pipeline import Pipeline
    wrapped = Pipeline([('preprocessor', pre), ('model', model)])
    for feature in ['age', 'bmi', 'bloodpressure']:
        if feature not in sample_df.columns:
            continue
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            PartialDependenceDisplay.from_estimator(
                wrapped, sample_df, [feature], kind='both', ax=ax, subsample=50, random_state=42
            )
            ax.set_title(f'PDP & ICE for {feature.capitalize()} ({args.risk} Risk Model)')
            fig.tight_layout()
            fig.savefig(OUTPUTS / f'{args.risk.lower()}_ice_pdp_{feature}.png', dpi=180, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f'Error plotting PDP/ICE for {feature}: {e}')

    print('XAI outputs saved to', OUTPUTS)


if __name__ == '__main__':
    main()

