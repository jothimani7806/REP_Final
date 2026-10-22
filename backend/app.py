import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from recommendations import generate_recommendations
import database
import ai_service

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'
OUTPUTS = ROOT / 'outputs'

if not (ARTIFACTS / 'metadata.json').exists():
    raise RuntimeError('Artifacts not found. Run backend/train_pipeline.py first.')

with open(ARTIFACTS / 'metadata.json', encoding='utf-8') as f:
    META = json.load(f)

PREPROCESSOR = joblib.load(ARTIFACTS / 'preprocessor.joblib')
PCA_MODEL = joblib.load(ARTIFACTS / 'cluster_pca.joblib')
CLUSTER_MODEL = joblib.load(ARTIFACTS / 'cluster_model.joblib')
GLOBAL_MODEL = joblib.load(ARTIFACTS / 'global_model.joblib')
RISK_MODELS = {
    risk: joblib.load(ARTIFACTS / f'{risk.lower()}_model.joblib')
    for risk in ['Low', 'Medium', 'High']
}

# Pre-initialize TreeExplainers for instantaneous local explanation
EXPLAINERS = {
    risk: shap.TreeExplainer(RISK_MODELS[risk])
    for risk in ['Low', 'Medium', 'High']
}

app = FastAPI(title='Risk-Aware Insurance Prediction API', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Serve outputs static directory (charts, diagrams, csvs)
if OUTPUTS.exists():
    app.mount('/outputs', StaticFiles(directory=str(OUTPUTS)), name='outputs')


class Customer(BaseModel):
    age: float | None = None
    sex: str
    weight: float
    bmi: float | None = None
    hereditary_diseases: str
    no_of_dependents: int
    smoker: int
    city: str
    bloodpressure: float | None = None
    diabetes: int
    regular_ex: int
    job_title: str


def prepare_input(customer: Customer):
    data = customer.model_dump()
    bmi = data.get('bmi')
    if bmi is not None:
        if bmi < 18.5:
            data['bmi_category'] = 'Underweight'
        elif bmi < 25:
            data['bmi_category'] = 'Normal'
        elif bmi < 30:
            data['bmi_category'] = 'Overweight'
        else:
            data['bmi_category'] = 'Obese'
    else:
        data['bmi_category'] = np.nan

    age = data.get('age')
    if age is not None:
        if age <= 25:
            data['age_group'] = 'Young'
        elif age <= 40:
            data['age_group'] = 'Adult'
        elif age <= 55:
            data['age_group'] = 'Middle_Age'
        else:
            data['age_group'] = 'Older'
    else:
        data['age_group'] = np.nan

    data['condition_count'] = int(data.get('smoker', 0) or 0) + int(data.get('diabetes', 0) or 0)
    if data.get('bloodpressure') is not None and data['bloodpressure'] <= 0:
        data['bloodpressure'] = np.nan

    # Actuarial interactions
    bmi_val = float(data['bmi']) if data.get('bmi') is not None else 25.0
    age_val = float(data['age']) if data.get('age') is not None else 35.0
    bp_val = float(data['bloodpressure']) if (data.get('bloodpressure') is not None and data['bloodpressure'] > 0) else 120.0
    smoker_val = float(data.get('smoker', 0) or 0)
    diabetes_val = float(data.get('diabetes', 0) or 0)

    data['smoker_bmi'] = smoker_val * bmi_val
    data['smoker_age'] = smoker_val * age_val
    data['diabetes_bp'] = diabetes_val * bp_val
    data['age_bmi'] = (age_val / 10.0) * (bmi_val / 10.0)

    return pd.DataFrame([data])



def get_risk(x):
    transformed = PREPROCESSOR.transform(x)
    cluster_space = PCA_MODEL.transform(transformed)
    raw_cluster = int(CLUSTER_MODEL.predict(cluster_space)[0])
    risk = META['risk_map_raw_cluster_to_risk'][str(raw_cluster)]
    return risk, raw_cluster


@app.get('/api')
def api_root():
    return {'message': 'Risk-Aware Insurance Prediction API is running'}


@app.get('/health')
def health():
    return {'status': 'ok', 'best_cluster_method': META['best_cluster_method']}


@app.get('/benchmarks')
def benchmarks():
    return META


@app.get('/xai_info/{risk}')
def xai_info(risk: str):
    r_key = risk.capitalize()
    if r_key not in ['Low', 'Medium', 'High']:
        raise HTTPException(status_code=404, detail='Invalid risk tier')
    
    top20_csv = OUTPUTS / 'xai' / f'{r_key.lower()}_shap_top20.csv'
    top_features = []
    if top20_csv.exists():
        df_top = pd.read_csv(top20_csv)
        top_features = df_top.to_dict(orient='records')

    return {
        'risk': r_key,
        'model_name': META['risk_selected_models'].get(r_key, 'Unknown'),
        'waterfall_img': f'/outputs/xai/{r_key.lower()}_shap_waterfall.png',
        'ice_pdp_age': f'/outputs/xai/{r_key.lower()}_ice_pdp_age.png',
        'ice_pdp_bmi': f'/outputs/xai/{r_key.lower()}_ice_pdp_bmi.png',
        'ice_pdp_bp': f'/outputs/xai/{r_key.lower()}_ice_pdp_bloodpressure.png',
        'top_features': top_features,
    }


@app.post('/predict')
def predict(customer: Customer):
    try:
        x = prepare_input(customer)
        transformed = PREPROCESSOR.transform(x)
        feature_names = PREPROCESSOR.get_feature_names_out()
        risk, raw_cluster = get_risk(x)
        
        # Risk-aware prediction
        model = RISK_MODELS[risk]
        prediction = float(model.predict(transformed)[0])
        
        # Global model prediction
        global_prediction = float(GLOBAL_MODEL.predict(transformed)[0])
        diff = prediction - global_prediction
        diff_pct = (diff / global_prediction * 100) if global_prediction != 0 else 0
        
        # Recommendations
        recs = generate_recommendations(customer.model_dump(), risk, prediction)

        # On-the-fly SHAP explanation for this individual customer
        explainer = EXPLAINERS[risk]
        shap_raw = explainer.shap_values(transformed)
        if isinstance(shap_raw, list):
            shap_vals = np.array(shap_raw[0])
        else:
            shap_vals = np.array(shap_raw)
        if shap_vals.ndim > 1:
            shap_vals = shap_vals[0]

        shap_items = []
        for feat, val in zip(feature_names, shap_vals):
            # Clean up display feature names
            clean_feat = feat.replace('num__', '').replace('cat__', '').replace('_', ' ')
            shap_items.append({
                'feature': clean_feat,
                'raw_feature': feat,
                'impact': float(val),
                'direction': 'Increases Claim' if val > 0 else 'Decreases Claim',
                'abs_impact': abs(float(val))
            })
        
        shap_items.sort(key=lambda item: item['abs_impact'], reverse=True)
        top_shap_factors = shap_items[:6]

        result_data = {
            'risk': risk,
            'cluster': raw_cluster,
            'risk_model': META['risk_selected_models'][risk],
            'risk_aware_prediction': round(prediction, 2),
            'global_model': META['global_selected_model'],
            'global_prediction': round(global_prediction, 2),
            'diff': round(diff, 2),
            'diff_pct': round(diff_pct, 1),
            'top_shap_factors': top_shap_factors,
            'recommendations': recs,
            'cluster_claim_means': META.get('cluster_claim_means_train', {})
        }

        # Persist verified prediction record for grounding AI assistant
        try:
            database.save_prediction('demo_user', result_data, customer.model_dump())
        except Exception as db_err:
            print(f"Warning: Failed to persist prediction record: {db_err}")

        return result_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    user_id: str = "demo_user"


@app.get('/api/conversations')
@app.get('/api/conversations/')
def list_conversations(user_id: str = "demo_user"):
    return database.get_user_conversations(user_id=user_id)


@app.post('/api/conversations')
@app.post('/api/conversations/')
def create_new_conversation(payload: dict = None, user_id: str = "demo_user"):
    title = (payload or {}).get("title", "New Consultation")
    return database.create_conversation(user_id=user_id, title=title)


@app.get('/api/conversations/{conversation_id}/messages')
@app.get('/api/conversations/{conversation_id}/messages/')
def get_conversation_messages(conversation_id: str, user_id: str = "demo_user"):
    existing_any = database.get_conversation_any_user(conversation_id)
    if not existing_any:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if existing_any["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied: Conversation belongs to another user")
    messages = database.get_messages(conversation_id)
    return {
        "conversation": existing_any,
        "messages": messages
    }


@app.delete('/api/conversations/{conversation_id}')
@app.delete('/api/conversations/{conversation_id}/')
def delete_user_conversation(conversation_id: str, user_id: str = "demo_user"):
    existing_any = database.get_conversation_any_user(conversation_id)
    if not existing_any:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if existing_any["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied: Conversation belongs to another user")
    deleted = database.delete_conversation(conversation_id, user_id=user_id)
    return {"status": "success", "deleted_id": conversation_id}


@app.get('/api/predictions/latest')
@app.get('/api/predictions/latest/')
def get_user_latest_prediction(user_id: str = "demo_user"):
    pred = database.get_latest_prediction(user_id=user_id)
    return pred or {}


@app.get('/api/predictions/history')
def get_user_prediction_history(user_id: str = "demo_user"):
    return database.get_prediction_history(user_id=user_id)


@app.post('/api/chat')
@app.post('/api/chat/')
def api_chat(payload: ChatRequest):
    user_query = payload.message.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(user_query) > 2000:
        raise HTTPException(status_code=400, detail="Message exceeds maximum length of 2000 characters")

    conv_id = payload.conversation_id
    user_id = payload.user_id or "demo_user"

    # Enforce conversation ownership and validation
    if not conv_id:
        title = user_query[:40] + ("..." if len(user_query) > 40 else "")
        new_conv = database.create_conversation(user_id=user_id, title=title)
        conv_id = new_conv["id"]
    else:
        existing_any = database.get_conversation_any_user(conv_id)
        if not existing_any:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if existing_any["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Access denied: Conversation belongs to another user")

    # Retrieve prior conversation history for context before adding new message
    history = database.get_messages(conv_id)

    # Idempotent user message recording (avoids duplicate user message on retries)
    if not (history and history[-1]["role"] == "user" and history[-1]["content"] == user_query):
        database.add_message(conv_id, role="user", content=user_query)

    # Retrieve verified prediction from database for factual grounding
    verified_pred = database.get_latest_prediction(user_id=user_id)

    # Generate response & references via Gemini AI service
    try:
        ai_response, references = ai_service.generate_ai_response(
            query=user_query,
            conversation_history=history,
            verified_prediction=verified_pred
        )
    except ai_service.GeminiConfigurationError as cfg_err:
        raise HTTPException(status_code=503, detail=str(cfg_err))
    except ai_service.GeminiGenerationError as gen_err:
        raise HTTPException(status_code=502, detail=str(gen_err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(err)}")

    # Save assistant message ONLY upon successful generation
    assistant_msg = database.add_message(
        conv_id,
        role="assistant",
        content=ai_response,
        medical_references=references
    )

    return {
        "conversation_id": conv_id,
        "message": ai_response,
        "medical_references": references,
        "created_at": assistant_msg["created_at"],
        "id": assistant_msg["id"]
    }


@app.post('/chat')
def legacy_chat(payload: dict):
    """Backward compatibility endpoint for legacy callers."""
    question = str(payload.get('question') or payload.get('message', '')).strip()
    verified_pred = database.get_latest_prediction('demo_user')
    if not verified_pred and payload.get('risk'):
        verified_pred = {
            'risk_tier': payload.get('risk'),
            'predicted_premium': payload.get('prediction'),
            'global_prediction': payload.get('global_prediction'),
            'risk_model': payload.get('model_name'),
            'top_factors': payload.get('top_factors', [])
        }
    try:
        ai_response, refs = ai_service.generate_ai_response(question, [], verified_pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {'answer': ai_response, 'medical_references': refs}


FRONTEND_DIST = ROOT / 'frontend' / 'dist'
if FRONTEND_DIST.exists():
    app.mount('/', StaticFiles(directory=str(FRONTEND_DIST), html=True), name='frontend')
else:
    @app.get('/')
    def root():
        return {'message': 'Risk-Aware Insurance Prediction API is running'}


if __name__ == '__main__':
    import os
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
