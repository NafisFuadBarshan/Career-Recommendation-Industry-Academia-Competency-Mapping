"""Pipelines and leakage-safe stacking architecture."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from .data import TEXT_FEATURE,NUMERIC_FEATURES
SEED=42
TOKEN_PATTERN=r"(?u)(?<!\w)[a-zA-Z0-9][a-zA-Z0-9+#./-]*(?!\w)"
def make_preprocessor():
    numeric=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
    return ColumnTransformer([("text",TfidfVectorizer(ngram_range=(1,3),max_features=15000,sublinear_tf=True,token_pattern=TOKEN_PATTERN),TEXT_FEATURE),("numeric",numeric,NUMERIC_FEATURES)],remainder="drop")
def make_models(n_classes,n_estimators=500,cv_folds=5):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.base import clone
    rf=Pipeline([("prep",make_preprocessor()),("model",RandomForestClassifier(n_estimators=n_estimators,max_depth=25,class_weight="balanced",random_state=SEED,n_jobs=-1))])
    xgb=Pipeline([("prep",make_preprocessor()),("model",XGBClassifier(objective="multi:softprob",num_class=n_classes,n_estimators=n_estimators,learning_rate=.03,max_depth=8,subsample=.8,colsample_bytree=.8,tree_method="hist",eval_metric="mlogloss",random_state=SEED,n_jobs=-1,verbosity=0))])
    lgb=Pipeline([("prep",make_preprocessor()),("model",LGBMClassifier(objective="multiclass",n_estimators=n_estimators,learning_rate=.03,num_leaves=40,reg_lambda=.1,class_weight="balanced",random_state=SEED,n_jobs=-1,verbosity=-1))])
    stack=StackingClassifier(estimators=[("rf",clone(rf)),("xgb",clone(xgb)),("lgb",clone(lgb))],final_estimator=LogisticRegression(C=.5,max_iter=2000,solver="lbfgs"),cv=StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=SEED),stack_method="predict_proba",passthrough=False,n_jobs=1)
    return {"Random Forest":rf,"XGBoost":xgb,"LightGBM":lgb,"Stacking Ensemble":stack}
