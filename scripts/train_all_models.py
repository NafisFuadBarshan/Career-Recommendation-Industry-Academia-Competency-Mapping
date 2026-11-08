"""Train all models: python -m scripts.train_all_models"""
import argparse
from pathlib import Path
from iaaf.training import train_project
def main():
    p=argparse.ArgumentParser(description="Train and evaluate all IAAF models")
    p.add_argument("--cv-folds",type=int,default=5); p.add_argument("--n-estimators",type=int,default=500); p.add_argument("--test-size",type=float,default=.2); a=p.parse_args()
    if a.cv_folds<2: p.error("--cv-folds must be >= 2")
    if a.n_estimators<1: p.error("--n-estimators must be positive")
    if not 0<a.test_size<1: p.error("--test-size must be between 0 and 1")
    train_project(Path(__file__).resolve().parents[1],a.cv_folds,a.n_estimators,a.test_size)
if __name__=="__main__": main()
