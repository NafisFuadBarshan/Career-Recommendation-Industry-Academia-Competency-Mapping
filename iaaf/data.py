"""Dataset loading and conservative skill text cleaning."""
import re
from pathlib import Path
import pandas as pd
TARGET="Target_Job_Role"
RAW_SKILLS="Known_Skills"
TEXT_FEATURE="Cleaned_Skills"
NUMERIC_FEATURES=["CGPA","Logic_Score","Technical_Skill_Count","Soft_Skill_Count","Has_Technical_Skills"]
MODEL_FEATURES=[TEXT_FEATURE,*NUMERIC_FEATURES]
STUDENT_REQUIRED={TARGET,RAW_SKILLS,*NUMERIC_FEATURES}
def clean_skill_text(value):
    if pd.isna(value): return ""
    text=str(value).lower().replace("_"," ")
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9+#./\-\s]"," ",text)).strip()
def _check(df, required, label):
    missing=sorted(set(required)-set(df.columns))
    if missing: raise ValueError(f"{label} dataset missing columns: {missing}")
def load_student_data(path):
    df=pd.read_excel(path); _check(df,STUDENT_REQUIRED,"Student"); original=len(df)
    df=df.dropna(subset=[TARGET]).copy().reset_index(drop=True)
    df[TARGET]=df[TARGET].astype(str).str.strip(); df=df[df[TARGET].ne("")].reset_index(drop=True)
    df[TEXT_FEATURE]=df[RAW_SKILLS].map(clean_skill_text)
    for col in NUMERIC_FEATURES:
        df[col]=pd.to_numeric(df[col],errors="coerce")
        if df[col].notna().sum()==0: raise ValueError(f"Numeric feature {col!r} is entirely missing")
    if df[TARGET].nunique()<2: raise ValueError("At least two career labels are required")
    return df,original
def load_job_data(path):
    df=pd.read_excel(path); _check(df,{"Job_Title","Skill_Tags"},"Job-posting"); return df
