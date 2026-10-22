# Risk-Aware Insurance Claim Prediction

This project implements the agreed research workflow:

1. REP.csv preprocessing and feature engineering
2. Global regression experiment: RF, GBM, XGBoost, LightGBM, CatBoost
3. Risk segmentation experiment: K-Means, GMM, Hierarchical clustering
4. Cluster evaluation and post-hoc Low/Medium/High risk mapping
5. Risk-aware regression experiment: RF, GBM, XGBoost, LightGBM, CatBoost for each risk group
6. Global vs risk-aware final comparison
7. SHAP + ICE + PDP explanations
8. Recommendation engine
9. FastAPI backend
10. React frontend

## Important methodology

- `claim` is never used as an input to clustering or regression features.
- Cluster-to-risk mapping is learned from training-set claim distributions only. Test-set claim values are used only for final evaluation.
- Hierarchical clustering is evaluated on a sample because standard agglomerative clustering has quadratic memory/time behavior and no native `predict()` for new customers.
- K-Means or GMM is selected for deployment because it can assign new customers to a learned cluster.

## Run

```powershell
cd risk_aware_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend\train_pipeline.py
uvicorn backend.app:app --reload --port 8000
```

API: http://127.0.0.1:8000

## XAI

After training, create a CSV with customer feature columns and run:

```powershell
python backend\xai.py --csv your_customer_rows.csv --risk High --row 0
```

The outputs are written to `outputs/xai`.
