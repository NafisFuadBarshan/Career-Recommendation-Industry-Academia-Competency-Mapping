# Industry–Academia Alignment Framework (IAAF)

A reproducible research prototype that predicts career roles from student profiles and compares student skills with role-specific skills extracted from job postings. It includes Random Forest, XGBoost, LightGBM, a stacking ensemble, competency-gap analysis, and a Streamlit interface.

> **Research use only:** predictions are decision-support outputs, not employment guarantees. Cosine similarity is not an employability probability.

## 1. How the system works

```text
Student Excel -> validation / skill cleaning -> shared stratified 80:20 split
                                           -> RF / XGBoost / LightGBM / Stacking
                                           -> same holdout -> metrics and reports
Job Excel -> title aliases -> role-specific TF-IDF benchmarks
Student skills + predicted role -> cosine similarity + ranked positive gaps
                                           -> Streamlit app
```

**Prediction features:** `Known_Skills` (TF-IDF 1–3 grams) plus `CGPA`, `Logic_Score`, `Technical_Skill_Count`, `Soft_Skill_Count`, and `Has_Technical_Skills`. `Degree` is not used. Only records with missing/blank `Target_Job_Role` are excluded; no skill/role heuristic deletes records.

**Leakage controls:** preprocessing lives inside each estimator pipeline and CV fold. All four models use one stratified 80:20 holdout (`random_state=42`). Five-fold CV runs only on the training partition. Stacking base estimators are complete preprocessing/model pipelines and its meta-learner uses out-of-fold class probabilities.

**Competency analysis:** job `Skill_Tags` are grouped by title. Aliases are explicit: `Merchandiser` → `Sales Executive`; `Motion Graphics Designer` → `Graphic Designer`. The mean role TF-IDF vector is compared to student skills; positive benchmark-minus-student dimensions are ranked. TF-IDF cannot fully represent synonyms, proficiency, context, or seniority, so review results with domain experts.

## 2. Project structure

```text
app.py
iaaf/                 # data, models, competency, training modules
scripts/train_all_models.py
tests/test_core.py
datasets/              # supplied Excel workbooks
models/                # generated locally; excluded from Git
results/               # generated locally; excluded from Git
requirements.txt
README.md
.gitignore
```

## 3. Requirements

- Python 3.10+ (64-bit recommended)
- Git
- Both Excel workbooks in `datasets/`
- Internet for first-time package installation

Training can be CPU/RAM intensive, especially stacking with CV.

## 4. Install and run (Windows PowerShell)

Open PowerShell in the folder containing this README.

### Step 1 — Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

If Python 3.11 is unavailable, use an installed Python 3.10+ version: `python -m venv .venv`.

### Step 2 — Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in this PowerShell session, then activate again.

### Step 3 — Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Train the four models

```powershell
python -m scripts.train_all_models
```

This is the full research configuration: five-fold CV and 500 estimators per tree-based learner. For an environment smoke test only (not final research metrics):

```powershell
python -m scripts.train_all_models --cv-folds 3 --n-estimators 50
```

The quick configuration changes the experiment. Do not combine its metrics with full-run results. Every class must contain at least as many rows as the CV fold count.

### Step 5 — Launch the app

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`. Select a model, enter the five numeric profile values, add comma-separated skills, and submit. Review the predicted role, model probability outputs, overlapping terms, and benchmark gaps. Probability outputs are not necessarily calibrated.

## 5. Run tests

```powershell
pytest -q
```

These tests cover text normalization, title aliases, benchmark creation, and gap-analysis output. They do not replace a full training/evaluation run.

## 6. Generated artifacts

Training creates `models/label_encoder.pkl`, `models/competency_engine.pkl`, and one pipeline under each of `models/random_forest/`, `models/xgboost/`, `models/lightgbm/`, and `models/stacking/`. It also creates `results/shared_evaluation/` with a model-comparison CSV, experiment config, shared split/test exports, classification reports, and confusion-matrix images.

Pickle models are generated locally and excluded from Git because they are large and sensitive to Python/library versions. Train after cloning. **Only load pickle files from trusted sources**; untrusted pickle files can execute code.

## 7. Upload to GitHub

### Option A — GitHub website

1. Sign in at [GitHub](https://github.com/) and choose **New repository**.
2. Name it, e.g. `industry-academia-alignment`. Choose Public or Private. Confirm you may publish the datasets before choosing Public.
3. Do not initialize a second README when uploading this complete project. Click **Create repository**.
4. Choose **Add file → Upload files**.
5. Drag in the project contents: `app.py`, `iaaf/`, `scripts/`, `tests/`, `datasets/`, `README.md`, `requirements.txt`, and `.gitignore`. Do not upload `.venv/`, generated model `.pkl` files, or generated result files.
6. Enter a commit message such as `Add IAAF source code and documentation`, then click **Commit changes**.

If the browser uploader rejects the upload, use the Git command line below.

### Option B — Git command line (recommended)

Install Git, open PowerShell in the project root, and run:

```powershell
git init
git add .
git status
```

Review the staged files. Ensure `.venv/`, model `.pkl` files, and generated result files are not staged. Then:

```powershell
git commit -m "Add IAAF project"
git branch -M main
```

Create an empty GitHub repository, copy its HTTPS URL, and connect it:

```powershell
git remote add origin https://github.com/YOUR-USERNAME/industry-academia-alignment.git
git push -u origin main
```

Replace `YOUR-USERNAME` and the repository name with your own. Authenticate if GitHub prompts you.

**Push later updates:**

```powershell
git add .
git status
git commit -m "Describe your update"
git push
```

If `origin` already exists, inspect it with `git remote -v`, then update it with `git remote set-url origin https://github.com/YOUR-USERNAME/industry-academia-alignment.git`.

### Before making the repository public

- Confirm you have permission to redistribute both datasets.
- Remove personal or confidential student information unless publication is explicitly authorized.
- Check that credentials, tokens, and private files are not included.
- Add verified dataset sources, licenses, collection dates, and permitted-use details. Do not invent a license.

## 8. Research and reproducibility cautions

A common holdout makes the four model scores comparable within this experiment, but does not prove external validity or representativeness. Do not use the holdout set for tuning. Class probabilities are not calibrated unless calibration is evaluated. The job benchmark is a static supplied snapshot, not a live market feed. Role aliases are assumptions that should be reviewed against the research protocol. Validate any curriculum recommendations with educators/employers. Record dataset versions, package versions, split settings, and model parameters when publishing.

## 9. Troubleshooting

- **`ModuleNotFoundError`:** activate `.venv` and reinstall `pip install -r requirements.txt`.
- **Missing model files in the app:** run `python -m scripts.train_all_models` from the project root.
- **Class too small for CV:** select a fold count no larger than the smallest class size; do not silently delete labels.
- **Dataset/column error:** verify both original workbooks and required columns in `iaaf/data.py`.
- **GitHub upload fails:** use Git CLI and verify `.gitignore`.

## 10. Citation

Cite this project and the original dataset/manuscript according to your institution's rules. Add complete bibliographic references and verified dataset licensing information here.
