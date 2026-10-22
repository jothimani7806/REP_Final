# Deployment Guide - Risk-Aware Insurance Premium Prediction System

This project consists of two services:
1. **Backend**: Python FastAPI with pre-trained ML models (Global + Risk-Tier XGBoost, CatBoost, LightGBM, SHAP Explainers, and Gemini/OpenAI chatbot).
2. **Frontend**: Modern React + Vite interactive dashboard.

---

## Recommended Deployment Architecture (100% Free Tier Supported)

- **Backend**: [Render.com](https://render.com) (or [Railway.app](https://railway.app))
- **Frontend**: [Vercel](https://vercel.com) (or [Netlify](https://netlify.com) / Render Static Site)

---

## Step 1: Deploy the Backend API (FastAPI) on Render

1. Sign in to [Render](https://render.com) and click **New +** -> **Web Service**.
2. Connect your GitHub repository: `https://github.com/jothimani7806/REP_Final`.
3. Configure the service settings:
   - **Name**: `rep-backend` (or your choice)
   - **Region**: Closest to you (e.g. Frankfurt, Oregon, Singapore)
   - **Branch**: `main`
   - **Root Directory**: Leave blank (uses repo root)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`
4. Expand **Environment Variables** and add:
   - `GEMINI_API_KEY`: *(Your Google Gemini API Key)*
   - `GEMINI_MODEL`: `gemini-2.5-flash` *(or your preferred model)*
   - `OPENAI_API_KEY`: *(Optional, if using OpenAI)*
5. Click **Create Web Service**.
6. Wait 2-3 minutes for the build to finish. Once deployed, copy your Backend URL (e.g., `https://rep-backend.onrender.com`).
   - You can test it in your browser: `https://rep-backend.onrender.com/health` -> should return `{"status":"ok"}`.

---

## Step 2: Deploy the Frontend (React Vite) on Vercel

1. Sign in to [Vercel](https://vercel.com) and click **Add New...** -> **Project**.
2. Import `jothimani7806/REP_Final`.
3. In the project configuration:
   - **Framework Preset**: `Vite`
   - **Root Directory**: Click *Edit* and select `frontend`
   - **Build Command**: `npm run build` (detected automatically)
   - **Output Directory**: `dist` (detected automatically)
4. Expand **Environment Variables**:
   - Name: `VITE_API_BASE`
   - Value: `https://rep-backend.onrender.com` *(paste your backend URL from Step 1, without a trailing slash)*
5. Click **Deploy**.
6. In ~40 seconds, your frontend will be live at `https://your-project.vercel.app`!

---

## Alternative: Single-Platform Deployment on Render

If you prefer to host both the Backend and Frontend on Render:

1. **Deploy Backend**: Follow Step 1 above.
2. **Deploy Frontend**:
   - In Render dashboard, click **New +** -> **Static Site**.
   - Select repository: `https://github.com/jothimani7806/REP_Final`.
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Publish Directory**: `dist`
   - Under **Environment Variables**, add:
     - `VITE_API_BASE`: `https://rep-backend.onrender.com`
   - Under **Redirects/Rewrites**:
     - Source: `/*`
     - Destination: `/index.html`
     - Action: `Rewrite`
   - Click **Create Static Site**.

---

## Local Verification Commands

To test both services locally anytime:

### Backend:
```bash
# In project root
python -m uvicorn backend.app:app --reload --port 8000
```

### Frontend:
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173).
