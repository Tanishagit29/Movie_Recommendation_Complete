# Movie Recommendation & Ranking System — Complete

## What is included
This folder contains the full end-to-end project, not just the Streamlit UI.

**MovieLens 1M → preprocessing → EDA-ready data → leakage-aware feature engineering → popularity baseline → XGBoost → LightGBM → OOF Stacking → classification evaluation → Precision@10 / Recall@10 / NDCG@10 → feature importance → SHAP → Streamlit.**

Random Forest is not used.

## 1. Install
Windows PowerShell:
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train everything
```powershell
python train.py
```
`train.py` downloads the official MovieLens 1M archive from GroupLens when it is not already in `data/`. It then creates the models and all inference artifacts.

Expected generated folders:
```text
models/
  xgboost_model.pkl
  lightgbm_model.pkl
  stacking_meta_learner.pkl
  user_stats.pkl
  movie_stats.pkl
  user_genre_preferences.pkl
  movie_genre_matrix.pkl
  feature_columns.pkl
  genre_list.pkl
  baseline_scores.pkl
  movies.pkl
  train_history.pkl
outputs/
  model_comparison.csv
  ranking_comparison.csv
  xgb_feature_importance.csv
  lgb_feature_importance.csv
  xgb_feature_importance.png
  shap_summary_xgboost.png
  rating_distribution.png
  top_rated_movies.png
```

## 3. Run the app
```powershell
streamlit run app.py
```

## 4. Pages
- Home
- Recommendations
- Model Performance
- Movie Details
- Explainability

## 5. Important evaluation design
The project treats ratings >= 4 as relevant. The final ranking evaluation uses a temporal train/test split and sampled unseen movies as negatives. Stacking uses out-of-fold predictions for its meta learner, avoiding the common same-data stacking leakage.

For the recommendation task, Precision@10, Recall@10 and NDCG@10 are the primary metrics. Accuracy/Precision/Recall/F1/ROC-AUC are secondary classification diagnostics.

## 6. If you already have trained artifacts
You can skip `python train.py` and copy your existing compatible `.pkl` files into `models/` and comparison CSVs into `outputs/`, then run Streamlit.
