"""
Movie Recommendation & Ranking System - complete training pipeline.
MovieLens 1M | Popularity Baseline | XGBoost | LightGBM | Stacking | SHAP

Run:
    python train.py

The script downloads MovieLens 1M automatically, trains the models, evaluates
classification and ranking metrics, saves all artifacts, and prepares the
Streamlit app.
"""
from pathlib import Path
import zipfile, urllib.request, ssl, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; MODELS=ROOT/'models'; OUT=ROOT/'outputs'
DATA.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
URL='https://files.grouplens.org/datasets/movielens/ml-1m.zip'
ZIP=DATA/'ml-1m.zip'; EXTRACT=DATA/'ml-1m'
RATINGS_PATH=EXTRACT/'ratings.dat'; MOVIES_PATH=EXTRACT/'movies.dat'
SEED=42

def download_dataset():
    if RATINGS_PATH.exists() and MOVIES_PATH.exists(): return
    print('Downloading MovieLens 1M...')
    ctx=ssl._create_unverified_context()
    with urllib.request.urlopen(URL, context=ctx) as resp, open(ZIP, 'wb') as out_file:
        out_file.write(resp.read())
    with zipfile.ZipFile(ZIP) as z: z.extractall(DATA)

def load_data():
    ratings=pd.read_csv(RATINGS_PATH, sep='::', engine='python', names=['userId','movieId','rating','timestamp'], encoding='latin-1')
    movies=pd.read_csv(MOVIES_PATH, sep='::', engine='python', names=['movieId','title','genres'], encoding='latin-1')
    ratings['timestamp_dt']=pd.to_datetime(ratings['timestamp'], unit='s')
    ratings['relevant']=(ratings['rating']>=4).astype(int)
    ratings=ratings.sort_values('timestamp').reset_index(drop=True)
    return ratings,movies

def make_artifacts(train,movies):
    global_mean=float(train.rating.mean())
    user=train.groupby('userId').agg(user_rating_count=('rating','count'),user_avg_rating=('rating','mean'),user_rating_std=('rating','std')).reset_index()
    user['user_rating_std']=user.user_rating_std.fillna(0)
    movie=train.groupby('movieId').agg(movie_rating_count=('rating','count'),movie_avg_rating=('rating','mean'),movie_rating_std=('rating','std')).reset_index()
    movie['movie_rating_std']=movie.movie_rating_std.fillna(0)
    # Bayesian weighted popularity score, used only as baseline/ranking feature.
    C=global_mean; m=movie.movie_rating_count.quantile(.60)
    movie['weighted_rating']=(movie.movie_rating_count/(movie.movie_rating_count+m))*movie.movie_avg_rating+(m/(movie.movie_rating_count+m))*C
    movie['movie_popularity']=np.log1p(movie.movie_rating_count)
    genres=movies[['movieId','genres']].copy()
    gd=genres['genres'].str.get_dummies(sep='|'); gd['movieId']=genres.movieId.values
    genre_list=[c for c in gd.columns if c!='movieId']
    # User genre preference is calculated only from training interactions.
    ug=train[['userId','movieId','rating']].merge(genres,on='movieId',how='left')
    rows=[]
    for g in genre_list:
        x=ug.assign(w=ug.genres.str.contains(g, regex=False).astype(int))
        x=x[x.w==1].groupby('userId').rating.mean().rename('user_pref_'+g)
        rows.append(x)
    user_pref=pd.concat(rows,axis=1).reset_index()
    return user,movie,gd,user_pref,genre_list,global_mean

def build_features(rows,user_stats,movie_stats,gd,user_pref,genre_list,global_mean):
    df=rows[['userId','movieId']].copy()
    df=df.merge(user_stats,on='userId',how='left').merge(movie_stats,on='movieId',how='left').merge(gd,on='movieId',how='left').merge(user_pref,on='userId',how='left')
    for c in ['user_rating_count','user_avg_rating','user_rating_std']:
        df[c]=df[c].fillna(global_mean if c!='user_rating_count' else 0)
    for c in ['movie_rating_count','movie_avg_rating','movie_rating_std','weighted_rating','movie_popularity']:
        df[c]=df[c].fillna(global_mean if 'rating' in c or c=='weighted_rating' else 0)
    for g in genre_list:
        df[g]=df[g].fillna(0); p='user_pref_'+g; df[p]=df[p].fillna(global_mean)
    df['rating_difference']=df.movie_avg_rating-df.user_avg_rating
    parts=[]
    for g in genre_list:
        parts.append(((df['user_pref_'+g]-1)/4).clip(0,1)*df[g])
    df['genre_match_score']=(sum(parts)/df[genre_list].sum(axis=1).replace(0,np.nan)).fillna(0)
    return df

def sample_negatives(users, all_movies, seen, n_per_user=2, seed=42):
    rng=np.random.default_rng(seed); out=[]
    all_movies=np.asarray(all_movies,dtype=int)
    for u, positives in users.items():
        pool=np.setdiff1d(all_movies,np.fromiter(positives,dtype=int),assume_unique=False)
        n=min(len(pool), max(1,n_per_user))
        if n: out.extend((int(u),int(m)) for m in rng.choice(pool,n,replace=False))
    return pd.DataFrame(out,columns=['userId','movieId'])

def build_training_set(train,movies,user_stats,movie_stats,gd,user_pref,genres,global_mean):
    pos=train.loc[train.relevant==1,['userId','movieId']].drop_duplicates()
    seen=train.groupby('userId').movieId.apply(set).to_dict()
    neg=sample_negatives(seen,movies.movieId.unique(),seen,n_per_user=1,seed=SEED)
    neg['relevant']=0; pos['relevant']=1
    examples=pd.concat([pos[['userId','movieId','relevant']],neg[['userId','movieId','relevant']]],ignore_index=True)
    examples['relevant']=examples['relevant'].fillna(1).astype(int)
    Xdf=build_features(examples,user_stats,movie_stats,gd,user_pref,genres,global_mean)
    features=['user_rating_count','user_avg_rating','user_rating_std','movie_rating_count','movie_avg_rating','movie_rating_std','movie_popularity','rating_difference','genre_match_score']+genres+[f'user_pref_{g}' for g in genres]
    X=Xdf[features].replace([np.inf,-np.inf],np.nan).fillna(0); y=examples.relevant.astype(int)
    return examples,X,y,features

def ranking_metrics(scores, truth, k):
    order=np.argsort(-np.asarray(scores))[:k]; rel=np.asarray(truth)[order]
    p=float(rel.sum()/k); total=max(1,int(np.asarray(truth).sum())); r=float(rel.sum()/total)
    discounts=1/np.log2(np.arange(2,k+2)); dcg=float((rel*discounts).sum())
    ideal=np.sort(np.asarray(truth))[::-1][:k]; idcg=float((ideal*discounts).sum())
    return p,r,(dcg/idcg if idcg else 0.0)

def evaluate_ranking(test, movies, scoring_fn, baseline_fn, k=10, negatives=100):
    rng=np.random.default_rng(SEED); allm=movies.movieId.values; by_user=test.groupby('userId'); vals={'Precision@K':[],'Recall@K':[],'NDCG@K':[],'Baseline_Precision@K':[],'Baseline_Recall@K':[],'Baseline_NDCG@K':[]}
    for u,grp in by_user:
        pos=set(grp.loc[grp.relevant==1,'movieId'])
        if not pos: continue
        seen=set(test.loc[test.userId==u,'movieId'])
        pool=np.setdiff1d(allm,np.fromiter(seen,dtype=int))
        neg=rng.choice(pool,size=min(negatives,len(pool)),replace=False)
        cand=np.array(list(pos)+list(neg),dtype=int); truth=np.array([1 if m in pos else 0 for m in cand])
        s=scoring_fn(int(u),cand); b=baseline_fn(cand)
        p,r,n=ranking_metrics(s,truth,k); bp,br,bn=ranking_metrics(b,truth,k)
        vals['Precision@K'].append(p); vals['Recall@K'].append(r); vals['NDCG@K'].append(n); vals['Baseline_Precision@K'].append(bp); vals['Baseline_Recall@K'].append(br); vals['Baseline_NDCG@K'].append(bn)
    return {k:float(np.mean(v)) if v else 0 for k,v in vals.items()}

def main():
    download_dataset(); ratings,movies=load_data(); split=int(len(ratings)*.8); train=ratings.iloc[:split].copy(); test=ratings.iloc[split:].copy()
    # Basic EDA outputs for the report/presentation.
    plt.figure(figsize=(7,4)); sns.countplot(data=ratings,x='rating'); plt.title('MovieLens Rating Distribution'); plt.tight_layout(); plt.savefig(OUT/'rating_distribution.png',dpi=160); plt.close()
    top=ratings.groupby('movieId').size().sort_values(ascending=False).head(10).rename('ratings').reset_index().merge(movies,on='movieId',how='left')
    plt.figure(figsize=(9,5)); sns.barplot(data=top,y='title',x='ratings'); plt.title('Top 10 Most Rated Movies'); plt.tight_layout(); plt.savefig(OUT/'top_rated_movies.png',dpi=160); plt.close()
    user,movie,gd,user_pref,genres,gm=make_artifacts(train,movies)
    examples,X,y,features=build_training_set(train,movies,user,movie,gd,user_pref,genres,gm)
    print('Training examples:',len(X))
    xgb=XGBClassifier(n_estimators=180,max_depth=6,learning_rate=.08,subsample=.9,colsample_bytree=.9,random_state=SEED,eval_metric='logloss',n_jobs=4)
    lgb=LGBMClassifier(n_estimators=180,num_leaves=31,learning_rate=.05,subsample=.9,colsample_bytree=.9,random_state=SEED,verbosity=-1,n_jobs=4)
    # Out-of-fold base predictions for leakage-safe stacking.
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED); oof=np.zeros((len(X),2))
    for tr,va in skf.split(X,y):
        a=XGBClassifier(n_estimators=180,max_depth=6,learning_rate=.08,subsample=.9,colsample_bytree=.9,random_state=SEED,eval_metric='logloss',n_jobs=4)
        b=LGBMClassifier(n_estimators=180,num_leaves=31,learning_rate=.05,subsample=.9,colsample_bytree=.9,random_state=SEED,verbosity=-1,n_jobs=4)
        a.fit(X.iloc[tr],y.iloc[tr]); b.fit(X.iloc[tr],y.iloc[tr]); oof[va,0]=a.predict_proba(X.iloc[va])[:,1]; oof[va,1]=b.predict_proba(X.iloc[va])[:,1]
    meta=LogisticRegression(max_iter=1000).fit(oof,y)
    xgb.fit(X,y); lgb.fit(X,y)
    # Classification evaluation on temporal test positives + sampled negatives.
    seen=train.groupby('userId').movieId.apply(set).to_dict(); pos=test[['userId','movieId']].copy(); pos['relevant']=(test.rating>=4).astype(int)
    neg=sample_negatives(seen,movies.movieId.unique(),seen,n_per_user=2,seed=SEED+1); neg['relevant']=0
    valid=pd.concat([pos[['userId','movieId','relevant']],neg],ignore_index=True)
    VX=build_features(valid,user,movie,gd,user_pref,genres,gm)[features].replace([np.inf,-np.inf],np.nan).fillna(0); vy=valid.relevant.astype(int)
    px=xgb.predict_proba(VX)[:,1]; pl=lgb.predict_proba(VX)[:,1]; ps=meta.predict_proba(np.column_stack([px,pl]))[:,1]
    rows=[]
    for name,p in [('XGBoost',px),('LightGBM',pl),('Stacking',ps)]:
        pred=(p>=.5).astype(int); rows.append({'Model':name,'Accuracy':accuracy_score(vy,pred),'Precision':precision_score(vy,pred,zero_division=0),'Recall':recall_score(vy,pred,zero_division=0),'F1':f1_score(vy,pred,zero_division=0),'ROC-AUC':roc_auc_score(vy,p)})
    model_comparison=pd.DataFrame(rows)
    allm=movies.movieId.values; base_map=movie.set_index('movieId').weighted_rating.to_dict()
    def make_scorer(model):
        def f(u,cand):
            r=pd.DataFrame({'userId':[u]*len(cand),'movieId':cand}); xx=build_features(r,user,movie,gd,user_pref,genres,gm)[features].fillna(0)
            if model=='XGBoost': return xgb.predict_proba(xx)[:,1]
            if model=='LightGBM': return lgb.predict_proba(xx)[:,1]
            a=xgb.predict_proba(xx)[:,1]; b=lgb.predict_proba(xx)[:,1]; return meta.predict_proba(np.column_stack([a,b]))[:,1]
        return f
    def base(c): return np.array([base_map.get(int(m),gm) for m in c])
    rank_rows=[]
    for name in ['XGBoost','LightGBM','Stacking']:
        rr=evaluate_ranking(test,movies,make_scorer(name),base,k=10,negatives=100)
        rank_rows.append({'Model':name,'Precision@10':rr['Precision@K'],'Recall@10':rr['Recall@K'],'NDCG@10':rr['NDCG@K']})
    br=evaluate_ranking(test,movies,lambda u,c: base(c),base,k=10,negatives=100)
    rank_rows.append({'Model':'Baseline','Precision@10':br['Baseline_Precision@K'],'Recall@10':br['Baseline_Recall@K'],'NDCG@10':br['Baseline_NDCG@K']})
    ranking=pd.DataFrame(rank_rows)
    # Save artifacts.
    joblib.dump(xgb,MODELS/'xgboost_model.pkl'); joblib.dump(lgb,MODELS/'lightgbm_model.pkl'); joblib.dump(meta,MODELS/'stacking_meta_learner.pkl')
    joblib.dump(user,MODELS/'user_stats.pkl'); joblib.dump(movie,MODELS/'movie_stats.pkl'); joblib.dump(user_pref,MODELS/'user_genre_preferences.pkl'); joblib.dump(gd,MODELS/'movie_genre_matrix.pkl'); joblib.dump(features,MODELS/'feature_columns.pkl'); joblib.dump(genres,MODELS/'genre_list.pkl'); joblib.dump(movie.set_index('movieId').weighted_rating.to_dict(),MODELS/'baseline_scores.pkl')
    movies.to_pickle(MODELS/'movies.pkl'); train[['userId','movieId','rating','timestamp','timestamp_dt']].to_pickle(MODELS/'train_history.pkl')
    model_comparison.to_csv(OUT/'model_comparison.csv',index=False); ranking.to_csv(OUT/'ranking_comparison.csv',index=False)
    fi=pd.DataFrame({'Feature':features,'Importance':xgb.feature_importances_}).sort_values('Importance',ascending=False); fi.to_csv(OUT/'xgb_feature_importance.csv',index=False)
    fi2=pd.DataFrame({'Feature':features,'Importance':lgb.feature_importances_}).sort_values('Importance',ascending=False); fi2.to_csv(OUT/'lgb_feature_importance.csv',index=False)
    plt.figure(figsize=(9,6)); sns.barplot(data=fi.head(15),x='Importance',y='Feature'); plt.title('Top 15 XGBoost Features'); plt.tight_layout(); plt.savefig(OUT/'xgb_feature_importance.png',dpi=160); plt.close()
    try:
        import shap
        sample=X.sample(min(1000,len(X)),random_state=SEED); explainer=shap.TreeExplainer(xgb); sv=explainer.shap_values(sample)
        plt.figure(); shap.summary_plot(sv,sample,show=False,max_display=15); plt.tight_layout(); plt.savefig(OUT/'shap_summary_xgboost.png',dpi=160,bbox_inches='tight'); plt.close()
    except Exception as e: print('SHAP skipped:',e)
    print('\nMODEL COMPARISON\n',model_comparison.round(4)); print('\nRANKING COMPARISON\n',ranking.round(4)); print('\nSaved all artifacts to models/ and outputs/.')

if __name__=='__main__': main()
