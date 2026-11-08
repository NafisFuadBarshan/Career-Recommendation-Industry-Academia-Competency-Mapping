"""Reproducible shared-holdout training and artifact generation."""
import json,pickle,platform
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,sklearn
from sklearn.base import clone
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.preprocessing import LabelEncoder
from .data import load_student_data,load_job_data,MODEL_FEATURES,NUMERIC_FEATURES
from .modeling import make_models,SEED
from .competency import build_competency_artifact
def train_project(root,cv_folds=5,n_estimators=500,test_size=.2):
    root=Path(root).resolve(); student_path=root/"datasets/student_dataset_final.xlsx"; jobs_path=root/"datasets/job_portal_dataset.xlsx"
    model_root=root/"models"; out=root/"results/shared_evaluation"; out.mkdir(parents=True,exist_ok=True); model_root.mkdir(exist_ok=True)
    if not student_path.exists() or not jobs_path.exists(): raise FileNotFoundError("Expected both Excel files in datasets/")
    df,original=load_student_data(student_path); encoder=LabelEncoder(); y=encoder.fit_transform(df["Target_Job_Role"])
    smallest=int(pd.Series(y).value_counts().min())
    if smallest<cv_folds: raise ValueError(f"Smallest class has {smallest} rows; CV requires at least {cv_folds}")
    X=df[MODEL_FEATURES].copy(); pos=np.arange(len(df)); train_ix,test_ix=train_test_split(pos,test_size=test_size,random_state=SEED,stratify=y)
    X_train,X_test=X.iloc[train_ix],X.iloc[test_ix]; y_train,y_test=y[train_ix],y[test_ix]
    ids=df["Student_ID"].astype(str) if "Student_ID" in df else df.index.astype(str)
    split=pd.DataFrame({"source_id":ids,"true_role":df["Target_Job_Role"],"split":"train"}); split.loc[test_ix,"split"]="test"; split.to_csv(out/"shared_split_indices.csv",index=False)
    export=[c for c in ["Student_ID","Target_Job_Role","Known_Skills",*NUMERIC_FEATURES] if c in df.columns]; df.iloc[test_ix][export].to_csv(out/"shared_test_set.csv",index=False)
    with (model_root/"label_encoder.pkl").open("wb") as f: pickle.dump(encoder,f)
    names={"Random Forest":("random_forest","random_forest_pipeline.pkl"),"XGBoost":("xgboost","xgboost_pipeline.pkl"),"LightGBM":("lightgbm","lightgbm_pipeline.pkl"),"Stacking Ensemble":("stacking","stacking_pipeline.pkl")}
    models=make_models(len(encoder.classes_),n_estimators,cv_folds); cv=StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=SEED); rows=[]
    for name,estimator in models.items():
        print(f"\nTraining {name} ({cv_folds}-fold CV on training partition)"); scores=cross_val_score(estimator,X_train,y_train,cv=cv,scoring="accuracy",n_jobs=1,error_score="raise")
        fitted=clone(estimator).fit(X_train,y_train); pred=fitted.predict(X_test)
        m={"model":name,"train_samples":len(train_ix),"test_samples":len(test_ix),"accuracy":float(accuracy_score(y_test,pred)),"macro_precision":float(precision_score(y_test,pred,average="macro",zero_division=0)),"macro_recall":float(recall_score(y_test,pred,average="macro",zero_division=0)),"macro_f1":float(f1_score(y_test,pred,average="macro",zero_division=0)),"cv_accuracy_mean":float(scores.mean()),"cv_accuracy_std":float(scores.std())}; rows.append(m)
        report=classification_report(y_test,pred,labels=np.arange(len(encoder.classes_)),target_names=encoder.classes_,zero_division=0); (out/f"{name.lower().replace(' ','_')}_report.txt").write_text(report,encoding="utf-8")
        cm=confusion_matrix(y_test,pred,labels=np.arange(len(encoder.classes_))); fig,ax=plt.subplots(figsize=(12,9)); ConfusionMatrixDisplay(cm,display_labels=encoder.classes_).plot(ax=ax,xticks_rotation=70,colorbar=False); ax.set_title(f"{name} — shared holdout (n={len(test_ix)})"); fig.tight_layout(); fig.savefig(out/f"{name.lower().replace(' ','_')}_confusion_matrix.png",dpi=180); plt.close(fig)
        folder,filename=names[name]; dest=model_root/folder; dest.mkdir(parents=True,exist_ok=True)
        with (dest/filename).open("wb") as f: pickle.dump(fitted,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Holdout accuracy={m['accuracy']:.4f}; macro-F1={m['macro_f1']:.4f}; CV={m['cv_accuracy_mean']:.4f}±{m['cv_accuracy_std']:.4f}")
    comparison=pd.DataFrame(rows); comparison.to_csv(out/"four_model_comparison.csv",index=False)
    jobs=load_job_data(jobs_path); competency=build_competency_artifact(jobs,set(encoder.classes_.astype(str)))
    with (model_root/"competency_engine.pkl").open("wb") as f: pickle.dump(competency,f,protocol=pickle.HIGHEST_PROTOCOL)
    config={"project_version":"2.0.0","student_rows_original":original,"student_rows_used":len(df),"student_rows_without_target_excluded":original-len(df),"job_posting_rows":len(jobs),"train_samples":len(train_ix),"test_samples":len(test_ix),"test_size":test_size,"random_state":SEED,"stratified":True,"cv_folds_on_training_partition":cv_folds,"features":MODEL_FEATURES,"numeric_features":NUMERIC_FEATURES,"degree_used_as_feature":False,"same_holdout_records_for_all_models":True,"preprocessing_fitted_inside_each_pipeline_and_cv_fold":True,"stacking_base_estimators_are_full_pipelines":True,"career_classes":encoder.classes_.tolist(),"job_postings_mapped_to_career_classes":competency["posting_counts"],"unmapped_job_titles":competency["unmapped_job_titles"],"competency_method":competency["method"],"python_version":platform.python_version(),"scikit_learn_version":sklearn.__version__}; (out/"experiment_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    print("\nShared-holdout comparison:"); print(comparison.to_string(index=False)); return comparison
