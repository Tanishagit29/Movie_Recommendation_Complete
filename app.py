import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import re
import requests

# Set page configuration
st.set_page_config(
    page_title="CineMatch AI — Cinema Edition",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cinema Red & Black CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Deep Cinema Background */
    .stApp {
        background-color: #080608;
        color: #f1f5f9;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f0a0d !important;
        border-right: 1px solid rgba(229, 9, 20, 0.22);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 7px;
        height: 7px;
    }
    ::-webkit-scrollbar-track {
        background: #080608;
    }
    ::-webkit-scrollbar-thumb {
        background: #3f090d;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #e50914;
    }
    
    /* Cinema Red Marquee Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(42, 6, 12, 0.95) 0%, rgba(15, 8, 12, 0.98) 60%, rgba(8, 6, 8, 0.99) 100%);
        border: 1px solid rgba(229, 9, 20, 0.35);
        border-radius: 16px;
        padding: 26px 30px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(229, 9, 20, 0.16), 0 4px 20px rgba(0, 0, 0, 0.7);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #e50914, #ff3b44, transparent);
    }
    
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #ffffff 10%, #ff4b55 50%, #e50914 90%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        text-shadow: 0 0 25px rgba(229, 9, 20, 0.3);
    }
    
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 0.95rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    
    /* Cinema Metric Tiles */
    .metric-card {
        background: rgba(24, 12, 16, 0.75);
        border: 1px solid rgba(229, 9, 20, 0.22);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(229, 9, 20, 0.6);
        box-shadow: 0 6px 24px rgba(229, 9, 20, 0.25);
    }
    
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* Movie Recommendation Card with Cinema Poster */
    .movie-card {
        background: rgba(22, 11, 15, 0.65);
        border: 1px solid rgba(229, 9, 20, 0.18);
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 16px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        gap: 16px;
        align-items: stretch;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6);
    }
    .movie-card:hover {
        border-color: rgba(229, 9, 20, 0.65);
        background: rgba(34, 14, 21, 0.85);
        transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(229, 9, 20, 0.28);
    }
    
    .poster-container {
        flex-shrink: 0;
        width: 102px;
        min-height: 150px;
        border-radius: 10px;
        overflow: hidden;
        background: #0d060a;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .poster-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
        transition: transform 0.35s ease;
    }
    .movie-card:hover .poster-img {
        transform: scale(1.05);
    }
    
    .movie-details {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .rank-badge {
        background: linear-gradient(135deg, #e50914, #8b0000);
        color: #ffffff;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 3px 9px;
        border-radius: 6px;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 2px 8px rgba(229, 9, 20, 0.4);
    }
    
    .movie-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
        margin-bottom: 4px;
        line-height: 1.3;
    }
    
    .genre-tag {
        background: rgba(229, 9, 20, 0.12);
        color: #fca5a5;
        border: 1px solid rgba(229, 9, 20, 0.28);
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 5px;
        margin-top: 3px;
    }
    
    .score-pill {
        background: linear-gradient(135deg, #e50914, #991b1b);
        color: white;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 3px 10px;
        border-radius: 8px;
        white-space: nowrap;
        box-shadow: 0 2px 10px rgba(229, 9, 20, 0.35);
    }
    
    /* Persona Card */
    .persona-card {
        background: rgba(20, 10, 14, 0.85);
        border: 1px solid rgba(229, 9, 20, 0.25);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    
    /* Explanation Box */
    .explain-box {
        background: rgba(229, 9, 20, 0.08);
        border-left: 4px solid #e50914;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 12px 0 16px 0;
        font-size: 0.92rem;
        line-height: 1.5;
        color: #f1f5f9;
    }
    
    /* Red Buttons & Tabs */
    .stButton > button {
        background: linear-gradient(135deg, #e50914, #991b1b) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(229, 9, 20, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff1e27, #b91c1c) !important;
        box-shadow: 0 6px 20px rgba(229, 9, 20, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Active Tab indicator */
    button[data-baseweb="tab"] {
        color: #94a3b8;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ff4b55 !important;
        border-bottom-color: #e50914 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #e50914 !important;
    }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
M = ROOT / 'models'
O = ROOT / 'outputs'

def load(name):
    return joblib.load(M / name)

@st.cache_resource
def load_all_artifacts():
    items = [
        ('xgboost_model.pkl', 'xgb'),
        ('lightgbm_model.pkl', 'lgb'),
        ('stacking_meta_learner.pkl', 'meta'),
        ('user_stats.pkl', 'user'),
        ('movie_stats.pkl', 'movie'),
        ('user_genre_preferences.pkl', 'upref'),
        ('movie_genre_matrix.pkl', 'genres_df'),
        ('feature_columns.pkl', 'features'),
        ('genre_list.pkl', 'genre_list'),
        ('baseline_scores.pkl', 'baseline'),
        ('movies.pkl', 'movies'),
        ('train_history.pkl', 'history')
    ]
    bundle = {}
    for filename, key in items:
        bundle[key] = load(filename)
    return bundle

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

try:
    d = load_all_artifacts()
    explainer = get_explainer(d['xgb'])
except Exception as e:
    st.error("⚠️ Model artifacts not found or incomplete. Please ensure `python train.py` has completed successfully.")
    st.exception(e)
    st.stop()

# Helper: Extract Clean Title & Year
def extract_year(title):
    match = re.search(r'\((\d{4})\)', title)
    return match.group(1) if match else "N/A"

def clean_movie_title(raw_title):
    t = re.sub(r'\s*\(\d{4}\)', '', raw_title).strip()
    for art in [', The', ', A', ', An', ', Le', ', La', ', Les', ', Il', ', Der', ', Die', ', Das']:
        if t.endswith(art):
            t = art.replace(', ', '') + ' ' + t[:-len(art)]
            break
    return t

# Cached Movie Poster Retrieval from OMDb / Amazon CDN
@st.cache_data(show_spinner=False, ttl=86400)
def get_movie_poster(raw_title):
    clean = clean_movie_title(raw_title)
    year = extract_year(raw_title)
    try:
        params = {'t': clean, 'apikey': 'trilogy'}
        if year != "N/A":
            params['y'] = year
        r = requests.get('http://www.omdbapi.com/', params=params, timeout=3.5)
        if r.status_code == 200:
            data = r.json()
            poster = data.get('Poster')
            if poster and poster != 'N/A' and poster.startswith('http'):
                return poster
        # Second attempt without year constraint
        if year != "N/A":
            r = requests.get('http://www.omdbapi.com/', params={'t': clean, 'apikey': 'trilogy'}, timeout=3.5)
            if r.status_code == 200:
                data = r.json()
                poster = data.get('Poster')
                if poster and poster != 'N/A' and poster.startswith('http'):
                    return poster
    except Exception:
        pass
    # Premium cinematic fallback placeholder
    return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=300&auto=format&fit=crop&q=80"

# Helper: Build Feature Matrix
def feature_rows(user_id, movie_ids, d):
    gm = float(d['history'].rating.mean())
    df = pd.DataFrame({'userId': [int(user_id)] * len(movie_ids), 'movieId': list(map(int, movie_ids))})
    df = df.merge(d['user'], on='userId', how='left').merge(d['movie'], on='movieId', how='left').merge(d['genres_df'], on='movieId', how='left').merge(d['upref'], on='userId', how='left')
    
    for c in ['user_rating_count', 'user_avg_rating', 'user_rating_std']:
        df[c] = df[c].fillna(gm if c != 'user_rating_count' else 0)
    for c in ['movie_rating_count', 'movie_avg_rating', 'movie_rating_std', 'weighted_rating', 'movie_popularity']:
        df[c] = df[c].fillna(gm if 'rating' in c or c == 'weighted_rating' else 0)
    for g in d['genre_list']:
        df[g] = df[g].fillna(0)
        df['user_pref_' + g] = df['user_pref_' + g].fillna(gm)
        
    df['rating_difference'] = df.movie_avg_rating - df.user_avg_rating
    parts = [((df['user_pref_' + g] - 1) / 4).clip(0, 1) * df[g] for g in d['genre_list']]
    df['genre_match_score'] = (sum(parts) / df[d['genre_list']].sum(axis=1).replace(0, np.nan)).fillna(0)
    
    return df[d['features']].replace([np.inf, -np.inf], np.nan).fillna(0)

# Helper: Compute Scores
def compute_scores(uid, ids, model_name, d):
    if model_name == 'Baseline':
        gm = float(d['history'].rating.mean())
        return np.array([d['baseline'].get(int(i), gm) for i in ids])
    
    X = feature_rows(uid, ids, d)
    px = d['xgb'].predict_proba(X)[:, 1]
    if model_name == 'XGBoost':
        return px
    pl = d['lgb'].predict_proba(X)[:, 1]
    if model_name == 'LightGBM':
        return pl
    # Stacking Meta Learner
    return d['meta'].predict_proba(np.column_stack([px, pl]))[:, 1]

# Sidebar Navigation & Settings
with st.sidebar:
    st.markdown("## 🍿 **CINEMATCH**")
    st.caption("Cinema Red Edition • Recommendation & XAI")
    st.markdown("---")
    
    nav_option = st.radio(
        "Navigation",
        ["🎯 Movie Recommendations", "🔍 Explainable AI (XAI)", "📊 Model Performance"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("#### 🎬 **Screening Controls**")
    users = sorted(d['history'].userId.unique())
    selected_user = st.selectbox("Audience Member (User)", users, index=0, help="Choose an active user profile from MovieLens 1M.")
    
    model_choice = st.selectbox(
        "AI Scoring Engine",
        ["Stacking (Ensemble)", "XGBoost", "LightGBM", "Baseline (Popularity)"],
        index=0,
        help="Stacking meta-learner blends predictions from gradient boosted trees."
    )
    model_key = "Stacking" if "Stacking" in model_choice else ("Baseline" if "Baseline" in model_choice else model_choice)
    
    top_k = st.slider("Recommendations (Top K)", min_value=3, max_value=25, value=8, step=1)
    
    all_genres = ["All Genres"] + sorted(d['genre_list'])
    selected_genre = st.selectbox("Genre Filter", all_genres, index=0)
    
    st.markdown("---")
    st.caption("MovieLens 1M • XGBoost • LightGBM • Stacking • SHAP")

# Executive Header Banner (Cinema Marquee)
st.markdown("""
<div class="hero-container">
    <div class="hero-title">CINEMATCH AI DASHBOARD</div>
    <div class="hero-subtitle">
        Premiere Recommendation Engine & Transparent Decision Explainability • MovieLens 1M Benchmark
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPI Highlights
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(d['history']):,}</div>
        <div class="metric-label">Audience Ratings</div>
    </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value" style="color: #ff4b55;">0.7134</div>
        <div class="metric-label">NDCG@10 Benchmark</div>
    </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value" style="color: #fbbf24;">68.04%</div>
        <div class="metric-label">Precision@10 Relevance</div>
    </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: #ffffff;">{len(d['features'])}</div>
        <div class="metric-label">ML Features</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 1: MOVIE RECOMMENDATIONS
# -------------------------------------------------------------
if nav_option == "🎯 Movie Recommendations":
    # User Profile Persona Header
    user_row = d['user'][d['user'].userId == selected_user]
    u_count = int(user_row.user_rating_count.values[0]) if len(user_row) else 0
    u_avg = float(user_row.user_avg_rating.values[0]) if len(user_row) else 3.5
    
    # User Top Genres
    upref_row = d['upref'][d['upref'].userId == selected_user]
    top_genres_str = "None"
    if len(upref_row):
        upref_series = upref_row.drop(columns=['userId'], errors='ignore').iloc[0]
        fav_genres = upref_series.sort_values(ascending=False).head(4)
        fav_badges = [f"<span class='genre-tag'>{g.replace('user_pref_', '')} ({v:.1f}★)</span>" for g, v in fav_genres.items() if not np.isnan(v)]
        top_genres_str = " ".join(fav_badges)

    st.markdown(f"""
    <div class="persona-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="font-size: 1.18rem; font-weight: 700; color: #ffffff; font-family: 'Outfit', sans-serif;">🎟️ Viewer Profile: ID #{selected_user}</span>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">
                    Screened titles: <b style="color:#f8fafc;">{u_count}</b> &nbsp;|&nbsp; Baseline Taste: <b style="color:#fbbf24;">{u_avg:.2f} ★</b>
                </div>
            </div>
            <div style="margin-top: 8px;">
                <span style="font-size: 0.85rem; color: #e2e8f0; margin-right: 8px; font-weight: 500;">Top Genres:</span>
                {top_genres_str}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Candidate Retrieval & Scoring
    with st.spinner(f"Curating screening recommendations with {model_choice}..."):
        seen_movies = set(d['history'].loc[d['history'].userId == selected_user, 'movieId'])
        all_movies_df = d['movies'].copy()
        
        # Genre filter if selected
        if selected_genre != "All Genres":
            all_movies_df = all_movies_df[all_movies_df.genres.str.contains(selected_genre, regex=False, na=False)]
        
        candidate_ids = [m for m in all_movies_df.movieId.astype(int).unique() if m not in seen_movies]
        
        if len(candidate_ids) == 0:
            st.warning(f"No unseen candidates found for user {selected_user} under genre '{selected_genre}'.")
        else:
            # Score candidate movies
            scores = compute_scores(selected_user, candidate_ids, model_key, d)
            
            cand_df = pd.DataFrame({'movieId': candidate_ids, 'match_score': scores})
            cand_df = cand_df.merge(d['movies'], on='movieId')
            cand_df = cand_df.merge(d['movie'], on='movieId', how='left')
            cand_df = cand_df.sort_values('match_score', ascending=False).head(top_k).reset_index(drop=True)
            cand_df['rank'] = np.arange(1, len(cand_df) + 1)
            
            # Save recommended items to session state for quick access in XAI tab
            st.session_state['last_recommendations'] = cand_df
            st.session_state['active_user'] = selected_user

            st.markdown(f"#### 📽️ Top {len(cand_df)} Featured Screenings ({model_choice})")
            
            # Render Cards in 2-column grid
            col_left, col_right = st.columns(2)
            
            for idx, row in cand_df.iterrows():
                target_col = col_left if idx % 2 == 0 else col_right
                title_clean = clean_movie_title(row['title'])
                year = extract_year(row['title'])
                genres = row['genres'].split('|')
                genre_tags = " ".join([f"<span class='genre-tag'>{g}</span>" for g in genres])
                score_pct = int(round(row['match_score'] * 100)) if row['match_score'] <= 1.0 else int(min(100, round(row['match_score'] * 20)))
                
                movie_cnt = int(row['movie_rating_count']) if pd.notnull(row['movie_rating_count']) else 0
                movie_avg = float(row['movie_avg_rating']) if pd.notnull(row['movie_avg_rating']) else 0.0
                
                # Retrieve poster URL
                poster_url = get_movie_poster(row['title'])
                
                with target_col:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="poster-container">
                            <img src="{poster_url}" alt="{title_clean}" class="poster-img" loading="lazy" />
                        </div>
                        <div class="movie-details">
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                    <div>
                                        <span class="rank-badge">#{row['rank']}</span>
                                        <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 500;">{year}</span>
                                    </div>
                                    <div class="score-pill">
                                        {score_pct}% Match
                                    </div>
                                </div>
                                <div class="movie-title">{title_clean}</div>
                                <div style="margin-bottom: 8px;">{genre_tags}</div>
                            </div>
                            <div style="display: flex; gap: 14px; font-size: 0.8rem; color: #94a3b8; border-top: 1px solid rgba(229,9,20,0.15); padding-top: 6px;">
                                <span>⭐ <b style="color:#fbbf24;">{movie_avg:.2f}</b>/5.0</span>
                                <span>👥 <b style="color:#f8fafc;">{movie_cnt:,}</b> reviews</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.info("🔍 **Behind the Screen**: Switch to **Explainable AI (XAI)** in the sidebar to see exact SHAP feature attributions explaining why these titles were picked!")

# -------------------------------------------------------------
# TAB 2: EXPLAINABLE AI (XAI) STUDIO
# -------------------------------------------------------------
elif nav_option == "🔍 Explainable AI (XAI)":
    st.markdown("### 🔍 **Explainable AI (XAI) Studio**")
    st.caption("Cinema Decision Transparency via SHAP (SHapley Additive exPlanations) & Feature Importance")
    
    tab_local, tab_global = st.tabs(["🔬 Local Recommendation Breakdown", "🌐 Global Model Explainability"])
    
    with tab_local:
        st.markdown("#### **Why was this movie chosen?**")
        st.write("Select any recommended movie to inspect its mathematical decision boundary and feature impacts.")
        
        # Load candidate recommendations or default options
        if 'last_recommendations' in st.session_state and len(st.session_state['last_recommendations']) > 0:
            rec_df = st.session_state['last_recommendations']
            movie_options = rec_df['title'].tolist()
            movie_map = dict(zip(rec_df['title'], rec_df['movieId']))
        else:
            sample_movies = d['movies'].head(15)
            movie_options = sample_movies['title'].tolist()
            movie_map = dict(zip(sample_movies['title'], sample_movies['movieId']))
        
        selected_movie_title = st.selectbox("Select Movie to Explain", movie_options, index=0)
        selected_movie_id = movie_map[selected_movie_title]
        
        # Compute exact feature row
        feat_df = feature_rows(selected_user, [selected_movie_id], d)
        
        # Compute SHAP values via TreeExplainer
        shap_values = explainer(feat_df)
        shap_row = shap_values.values[0]
        
        # Create attribution DataFrame
        attrib_df = pd.DataFrame({
            'Feature': d['features'],
            'SHAP_Value': shap_row,
            'Feature_Value': feat_df.iloc[0].values
        }).sort_values(by='SHAP_Value', key=abs, ascending=False)
        
        top_positive = attrib_df[attrib_df['SHAP_Value'] > 0].head(5)
        top_negative = attrib_df[attrib_df['SHAP_Value'] < 0].head(5)
        
        # Retrieve Movie Details & Poster
        movie_poster_url = get_movie_poster(selected_movie_title)
        clean_name = clean_movie_title(selected_movie_title)
        year_str = extract_year(selected_movie_title)
        top_pos_features = top_positive['Feature'].tolist()
        
        # Natural Language Explanation Generator
        narrative = f"The model prioritized **{clean_name} ({year_str})** for **Viewer #{selected_user}** primarily because "
        reasons = []
        if 'genre_match_score' in top_pos_features:
            reasons.append(f"the film's genre profile strongly matches the viewer's historical taste (SHAP uplift: +{top_positive.loc[top_positive['Feature']=='genre_match_score', 'SHAP_Value'].values[0]:.2f})")
        if 'movie_popularity' in top_pos_features or 'movie_rating_count' in top_pos_features:
            reasons.append("the title has a strong, dependable track record of positive audience acclaim")
        if 'rating_difference' in top_pos_features or 'movie_avg_rating' in top_pos_features:
            reasons.append("the film's critical baseline significantly outperforms the user's typical threshold")
        
        if not reasons:
            reasons.append(f"key signals like `{top_pos_features[0] if top_pos_features else 'taste affinity'}` provided positive scoring uplift")
            
        narrative += " and ".join(reasons) + "."
        
        # Spotlight Card with Poster beside Decision Narrative
        st.markdown(f"""
        <div style="background: rgba(24, 11, 15, 0.7); border: 1px solid rgba(229,9,20,0.28); border-radius: 14px; padding: 18px; margin: 16px 0; display: flex; gap: 20px; align-items: center; box-shadow: 0 4px 20px rgba(0,0,0,0.6);">
            <div style="width: 95px; height: 142px; flex-shrink: 0; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.7); border: 1px solid rgba(229,9,20,0.3);">
                <img src="{movie_poster_url}" alt="{clean_name}" style="width: 100%; height: 100%; object-fit: cover;" />
            </div>
            <div style="flex-grow: 1;">
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.3rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
                    {clean_name} <span style="font-size: 0.9rem; color: #94a3b8; font-weight: 400;">({year_str})</span>
                </div>
                <div class="explain-box" style="margin: 6px 0 0 0;">
                    <b>🤖 AI Decision Narrative:</b><br>
                    {narrative}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Cinema-themed Waterfall-style Bar Plot
        fig, ax = plt.subplots(figsize=(10, 4.8), facecolor='#080608')
        ax.set_facecolor('#10090d')
        
        plot_df = pd.concat([top_positive, top_negative]).sort_values(by='SHAP_Value', ascending=True)
        # Red & Emerald Cinema palette
        colors = ['#10b981' if v > 0 else '#e50914' for v in plot_df['SHAP_Value']]
        
        y_pos = np.arange(len(plot_df))
        bars = ax.barh(y_pos, plot_df['SHAP_Value'], color=colors, height=0.62, alpha=0.92)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['Feature'], color='#e2e8f0', fontsize=9.5, fontweight='500')
        ax.axvline(0, color='#64748b', linestyle='--', linewidth=1.0)
        
        # Annotate bars
        for bar in bars:
            width = bar.get_width()
            align = 'left' if width >= 0 else 'right'
            offset = 0.01 if width >= 0 else -0.01
            ax.text(width + offset, bar.get_y() + bar.get_height()/2, f'{width:+.3f}', 
                    va='center', ha=align, color='#f8fafc', fontsize=8.5, weight='bold')
            
        ax.set_xlabel("SHAP Impact on Recommendation Score", color='#94a3b8', fontsize=10)
        ax.set_title(f"SHAP Feature Attributions for '{clean_name}' (User #{selected_user})", color='#ffffff', fontsize=11, weight='bold', pad=12)
        ax.tick_params(colors='#94a3b8', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2e1218')
        ax.spines['bottom'].set_color('#2e1218')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Detailed Attribution Table
        with st.expander("📋 View Complete Feature Value & Attribution Table"):
            display_attrib = attrib_df.head(15).copy()
            display_attrib['SHAP_Value'] = display_attrib['SHAP_Value'].round(4)
            display_attrib['Feature_Value'] = display_attrib['Feature_Value'].round(3)
            st.dataframe(display_attrib, use_container_width=True, hide_index=True)

    with tab_global:
        st.markdown("#### **Global Feature Importance & Explainability**")
        st.write("Understand which overarching attributes influence recommendations across all MovieLens 1M interactions.")
        
        shap_img = O / 'shap_summary_xgboost.png'
        if shap_img.exists():
            st.image(str(shap_img), caption="Global SHAP Beeswarm Plot: Feature Impact on XGBoost Classifier", use_container_width=True)
            st.markdown("""
            > **Reading the SHAP Summary Plot:**
            > - **Red dots** indicate high feature values; **Blue dots** indicate low feature values.
            > - Features with points further to the **right** increase the predicted probability of the user liking the movie (relevant = 1).
            > - **`weighted_rating`** and **`genre_match_score`** demonstrate the largest SHAP variance, making them the most influential drivers of high ranking.
            """)
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            xgb_fi_path = O / 'xgb_feature_importance.csv'
            if xgb_fi_path.exists():
                st.markdown("##### 🌲 XGBoost Feature Importance")
                xgb_fi = pd.read_csv(xgb_fi_path).head(10)
                st.dataframe(xgb_fi, use_container_width=True, hide_index=True)
        with col_g2:
            lgb_fi_path = O / 'lgb_feature_importance.csv'
            if lgb_fi_path.exists():
                st.markdown("##### ⚡ LightGBM Feature Importance")
                lgb_fi = pd.read_csv(lgb_fi_path).head(10)
                st.dataframe(lgb_fi, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 3: MODEL BENCHMARKS
# -------------------------------------------------------------
elif nav_option == "📊 Model Performance":
    st.markdown("### 📊 **Enterprise Model Benchmark & Evaluation**")
    st.caption("Leakage-free Temporal Validation with Out-of-Fold Stacking")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### 🎯 Ranking Metrics (Precision@10 / NDCG@10)")
        st.write("Evaluates the top recommended items against ground truth relevant ratings on a held-out temporal test set.")
        rank_path = O / 'ranking_comparison.csv'
        if rank_path.exists():
            rank_df = pd.read_csv(rank_path).round(4)
            st.dataframe(rank_df, use_container_width=True, hide_index=True)
            
    with col_m2:
        st.markdown("#### 📈 Classification Diagnostics (ROC-AUC / F1)")
        st.write("Evaluates the binary relevance threshold ($Rating \\geq 4$) probability calibration.")
        cls_path = O / 'model_comparison.csv'
        if cls_path.exists():
            cls_df = pd.read_csv(cls_path).round(4)
            st.dataframe(cls_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📉 Dataset Exploratory Visualizations")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        rating_dist = O / 'rating_distribution.png'
        if rating_dist.exists():
            st.image(str(rating_dist), caption="MovieLens 1M Rating Distribution", use_container_width=True)
    with col_v2:
        top_rated = O / 'top_rated_movies.png'
        if top_rated.exists():
            st.image(str(top_rated), caption="Top Rated Movies by Community Engagement", use_container_width=True)
