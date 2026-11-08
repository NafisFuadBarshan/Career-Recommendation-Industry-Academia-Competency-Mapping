import numpy as np
import pandas as pd
from iaaf.data import clean_skill_text
from iaaf.competency import build_competency_artifact,analyze_competency_gap
def test_cleaning_preserves_technical_tokens():
    text=clean_skill_text("C++, C#, Node.js, CI/CD & SQL")
    for token in ("c++","c#","node.js","ci/cd","sql"): assert token in text
def test_benchmark_and_gap():
    jobs=pd.DataFrame({"Job_Title":["Data Scientist","Data Scientist","Merchandiser"],"Skill_Tags":["python, sql, machine learning","python, statistics, pandas","sales, negotiation"]})
    artifact=build_competency_artifact(jobs,{"Data Scientist","Sales Executive"}); result=analyze_competency_gap("python, sql","Data Scientist",artifact)
    assert result["available"] and 0<=result["cosine_similarity"]<=1 and result["benchmark_postings"]==2
def test_unknown_role():
    result=analyze_competency_gap("python","Unknown",{"benchmarks":{},"vectorizer":None,"feature_names":np.array([]),"posting_counts":{}}); assert not result["available"]
