"""Role-specific industry TF-IDF benchmark and skill gap analysis."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .data import clean_skill_text
ROLE_ALIASES={"Merchandiser":"Sales Executive","Motion Graphics Designer":"Graphic Designer"}
TOKEN_PATTERN=r"(?u)(?<!\w)[a-zA-Z0-9][a-zA-Z0-9+#./-]*(?!\w)"
def build_competency_artifact(jobs,valid_roles):
    required={"Job_Title","Skill_Tags"}; missing=required-set(jobs.columns)
    if missing: raise ValueError(f"Job-posting dataset missing columns: {sorted(missing)}")
    df=jobs.dropna(subset=["Job_Title"]).copy(); df["Job_Title"]=df["Job_Title"].astype(str).str.strip()
    df["Career_Role"]=df["Job_Title"].replace(ROLE_ALIASES); df["skill_text"]=df["Skill_Tags"].map(clean_skill_text)
    df=df[df["Career_Role"].isin(valid_roles)&df["skill_text"].str.len().gt(0)].reset_index(drop=True)
    if df.empty: raise ValueError("No job postings map to student career labels; review title aliases")
    vectorizer=TfidfVectorizer(ngram_range=(1,3),sublinear_tf=True,token_pattern=TOKEN_PATTERN)
    matrix=vectorizer.fit_transform(df["skill_text"]); names=vectorizer.get_feature_names_out(); benchmarks={}; counts={}
    for role in sorted(valid_roles):
        ix=np.flatnonzero(df["Career_Role"].to_numpy()==role)
        if ix.size: benchmarks[role]=np.asarray(matrix[ix].mean(axis=0)).ravel(); counts[role]=int(ix.size)
    return {"vectorizer":vectorizer,"feature_names":names,"benchmarks":benchmarks,"posting_counts":counts,"unmapped_job_titles":sorted(set(jobs["Job_Title"].dropna().astype(str))-set(df["Job_Title"])),"method":"Mean role-specific job-posting TF-IDF vector; cosine similarity; positive benchmark-minus-student dimensions ranked as gaps."}
def analyze_competency_gap(raw_skills,role,artifact,top_n=15):
    if role not in artifact["benchmarks"]: return {"available":False,"reason":f"No mapped job-posting benchmark is available for {role}."}
    student_vec=artifact["vectorizer"].transform([clean_skill_text(raw_skills)]); benchmark=np.asarray(artifact["benchmarks"][role]).reshape(1,-1)
    similarity=float(cosine_similarity(student_vec,benchmark)[0,0]); student=student_vec.toarray().ravel(); demand=benchmark.ravel(); gaps=np.maximum(0,demand-student); names=np.asarray(artifact["feature_names"])
    ix=np.argsort(gaps)[::-1]; prioritized=[{"skill":str(names[i]),"gap_weight":float(gaps[i]),"industry_weight":float(demand[i]),"student_weight":float(student[i])} for i in ix if gaps[i]>0][:top_n]
    matched=np.flatnonzero((student>0)&(demand>0)); matched=matched[np.argsort(student[matched])[::-1]][:top_n]
    return {"available":True,"career":role,"cosine_similarity":similarity,"alignment_percent":similarity*100,"matched_terms":[str(names[i]) for i in matched],"prioritized_gaps":prioritized,"benchmark_postings":int(artifact["posting_counts"].get(role,0)),"interpretation":"Cosine similarity is a text-vector alignment measure, not employability probability."}
