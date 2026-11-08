"""Streamlit interface for career prediction and skill-gap analysis."""
from pathlib import Path
import pickle
import pandas as pd
import streamlit as st
from iaaf.competency import analyze_competency_gap
from iaaf.data import clean_skill_text
ROOT=Path(__file__).resolve().parent; MODELS=ROOT/"models"
MODEL_FILES={"Stacking Ensemble":MODELS/"stacking/stacking_pipeline.pkl","Random Forest":MODELS/"random_forest/random_forest_pipeline.pkl","XGBoost":MODELS/"xgboost/xgboost_pipeline.pkl","LightGBM":MODELS/"lightgbm/lightgbm_pipeline.pkl"}
@st.cache_resource
def load_artifacts():
    required=list(MODEL_FILES.values())+[MODELS/"competency_engine.pkl",MODELS/"label_encoder.pkl"]; missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing: raise FileNotFoundError("Train first with `python -m scripts.train_all_models`. Missing: "+", ".join(missing))
    models={}
    for name,path in MODEL_FILES.items():
        with path.open("rb") as f: models[name]=pickle.load(f)
    with (MODELS/"competency_engine.pkl").open("rb") as f: competency=pickle.load(f)
    with (MODELS/"label_encoder.pkl").open("rb") as f: encoder=pickle.load(f)
    return models,encoder,competency
st.set_page_config(page_title="Industry–Academia Alignment Framework",page_icon="🎓",layout="wide")
st.title("🎓 Industry–Academia Alignment Framework"); st.caption("Career prediction and industry-derived competency-gap analysis.")
try: models,encoder,competency=load_artifacts()
except (FileNotFoundError,OSError,pickle.UnpicklingError) as exc: st.error(str(exc)); st.stop()
with st.sidebar:
    st.header("About"); st.write("Research decision-support prototype—not employment guarantees or professional career advice."); selected=st.selectbox("Prediction model",list(MODEL_FILES),index=0)
with st.form("profile_form"):
    st.subheader("Your profile"); left,right=st.columns(2)
    with left:
        cgpa=st.number_input("CGPA",min_value=0.,max_value=4.,value=3.45,step=.01); logic=st.number_input("Logic score",min_value=0.,value=2.,step=.01); tech=st.number_input("Technical skill count",min_value=0,value=3,step=1)
    with right:
        soft=st.number_input("Soft skill count",min_value=0,value=2,step=1); has_tech=st.selectbox("Do you have technical skills?",[1,0],format_func=lambda x:"Yes" if x else "No")
    skills=st.text_area("Your skills",placeholder="Python, SQL, data analysis, machine learning"); submitted=st.form_submit_button("Analyze my career alignment",type="primary",use_container_width=True)
if submitted:
    profile=pd.DataFrame([{"Cleaned_Skills":clean_skill_text(skills),"CGPA":cgpa,"Logic_Score":logic,"Technical_Skill_Count":tech,"Soft_Skill_Count":soft,"Has_Technical_Skills":has_tech}]); model=models[selected]; idx=int(model.predict(profile)[0]); role=str(encoder.inverse_transform([idx])[0])
    st.subheader("Recommended career role"); st.success(role)
    if hasattr(model,"predict_proba"):
        probs=model.predict_proba(profile)[0]; top=probs.argsort()[::-1][:3]; table=pd.DataFrame({"Career role":encoder.inverse_transform(top),"Model probability output":[f"{probs[i]:.1%}" for i in top]}); st.dataframe(table,hide_index=True,use_container_width=True); st.caption("Probability outputs are not calibrated confidence estimates unless separately evaluated.")
    st.subheader("Industry competency alignment"); gap=analyze_competency_gap(skills,role,competency)
    if not gap["available"]: st.warning(gap["reason"])
    else:
        c1,c2=st.columns([1,2]); c1.metric("Text-vector similarity",f"{gap['alignment_percent']:.1f}%"); c2.caption(f"Benchmark based on {gap['benchmark_postings']} mapped job postings. Not an employability percentage.")
        a,b=st.columns(2)
        with a: st.markdown("**Skills represented in both profiles**"); st.write(", ".join(gap["matched_terms"]) if gap["matched_terms"] else "No overlapping terms identified.")
        with b:
            st.markdown("**Prioritized benchmark gaps**")
            if gap["prioritized_gaps"]: st.dataframe(pd.DataFrame(gap["prioritized_gaps"]).rename(columns={"skill":"Skill / term","gap_weight":"Gap weight","industry_weight":"Industry weight","student_weight":"Student weight"}),hide_index=True,use_container_width=True)
            else: st.write("No positive TF-IDF gap dimensions were identified.")
        st.info("Gap rankings are text-based research aids. Validate them with educators and employers before using them as a formal curriculum roadmap.")
st.divider(); st.caption("Research prototype • Record dataset versions, software versions, and experiment settings when reporting results.")
