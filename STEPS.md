# ML Assignment 2: Step-by-Step Guide

End-to-end steps for the Machine Learning Assignment 2 (M.Tech AIML): train five classifiers on one dataset, save them, build a Streamlit app, deploy it, and assemble the submission PDF.

**Deadline:** 8 November 2026, 23:59. No extensions and no draft submissions.
**Marks (15):** model implementation and GitHub (10), Streamlit app (4), BITS Virtual Lab screenshot (1).

---

## 1. What is in the project folder

All files are in `Semester 1 2026/ml/assignment2/`.

| File | Purpose |
|---|---|
| `ML_Assignment2_Colab.ipynb` | Notebook that loads the data, trains the five models, computes the metrics and saves the models and test data |
| `app.py` | Streamlit app that loads the saved models and evaluates them on an uploaded test CSV |
| `requirements.txt` | Python packages needed to deploy the app |
| `README.md` | Required README (problem statement, dataset, results, observations) |
| `test_data.csv` | 921-row held-out test set (57 features plus the `spam` target) |
| `model/*.joblib` | Five saved models (logistic regression, decision tree, KNN, naive Bayes, SVM) |
| `model_comparison.csv` | Metric table produced by the notebook |
| `make_nb.py` | Script that generated the notebook. It is not needed for submission. |

The assignment requires this repository layout:

```
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model/   (all five saved models)
```

---

## 2. Dataset choice (Step 1)

Requirements: one public classification dataset with at least 12 features and at least 500 instances.

**Chosen dataset: Spambase** (UCI, https://archive.ics.uci.edu/dataset/94/spambase)

| Property | Value | Requirement |
|---|---|---|
| Instances | 4,601 | >= 500 |
| Features | 57, all numeric | >= 12 |
| Target | `spam` (1 = spam, 0 = not spam), about 39.4% spam | classification |
| Missing values | none | |

Spambase is clean and fits all five models. The data file has no header, so the notebook adds the 57 column names itself.

> Different students may pick the same dataset. The assignment checks for identical datasets, models and outputs. To make your work distinct, change the dataset, the hyperparameters, the split seed or the app layout, and write the README observations in your own words.

---

## 3. Train the models on Google Colab (Step 2)

1. Open https://colab.research.google.com and choose **File > Upload notebook**.
2. Upload `ML_Assignment2_Colab.ipynb`.
3. Choose **Runtime > Run all**.

### What the notebook does

| Notebook section | What happens |
|---|---|
| 0. Setup | Imports, sets `RANDOM_STATE = 42`, creates the `model/` folder |
| 1. Dataset | Downloads the data from UCI, names the 57 columns, checks the size requirements, shows missing values, duplicates and class balance |
| 2. Split | Stratified 80/20 train/test split. Models never see the test portion during training. |
| 3. Models | Builds five scikit-learn pipelines and fits them on the training data |
| 4. Metrics | Computes Accuracy, AUC, Precision, Recall, F1 and MCC on the test set. Also shows training-set metrics to spot overfitting. |
| 5. Plots | Confusion matrix for each model, combined ROC curves and classification reports |
| 6. Save | Writes `model/*.joblib`, `test_data.csv` and `model_comparison.csv`, then reloads the models and checks they reproduce the same metrics |
| 7. Download | Downloads a models zip, `test_data.csv` and `model_comparison.csv` |

### Models and settings

| Model | Preprocessing | Settings |
|---|---|---|
| Logistic Regression | StandardScaler | `max_iter=2000` |
| Decision Tree | none | `max_depth=8` |
| KNN | StandardScaler | `n_neighbors=5` |
| Naive Bayes | none | `GaussianNB` |
| Support Vector Machine | StandardScaler | RBF kernel, `probability=True` |

Why the settings are what they are:
- **Scaling is inside each pipeline.** Logistic regression, KNN and SVM are sensitive to feature scale. Putting the scaler in the saved pipeline means the app does not need a separate scaler file, and training and prediction cannot get out of step.
- **`probability=True` for SVM.** It is needed to get probabilities for AUC. It makes training slower.
- **Tree depth limited to 8.** It reduces overfitting.

### Metrics

| Metric | Meaning |
|---|---|
| Accuracy | Share of all predictions that are correct |
| AUC | Ability to rank spam above non-spam, using predicted probabilities |
| Precision | Of emails predicted spam, the share that are spam |
| Recall | Of real spam, the share that was caught |
| F1 | Harmonic mean of precision and recall |
| MCC | Correlation between predicted and true labels, using all four confusion-matrix cells. It stays informative when classes are imbalanced. |

### Results from my local run

Your Colab run may differ slightly because of library versions. **Use the numbers your own run prints.**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9294 | 0.9702 | 0.9209 | 0.8981 | 0.9093 | 0.8518 |
| Decision Tree | 0.9034 | 0.9154 | 0.9029 | 0.8457 | 0.8734 | 0.7965 |
| KNN | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| Naive Bayes | 0.8339 | 0.9449 | 0.7178 | 0.9532 | 0.8189 | 0.6941 |
| Support Vector Machine | 0.9273 | 0.9667 | 0.9277 | 0.8843 | 0.9055 | 0.8471 |

Logistic Regression leads on Accuracy, AUC, F1 and MCC. Naive Bayes has the highest recall but the lowest precision.

### Version caution

The `.joblib` files only load reliably with the scikit-learn version that saved them. **Use the model files created on Colab, and let Streamlit Cloud install the current scikit-learn.** If you later see an "unpickling / version mismatch" error, either pin the same version in `requirements.txt` (for example `scikit-learn==1.x.x`), or retrain and re-save the models.

---

## 4. Prepare the GitHub repository (Step 3)

1. Create a new **public** repository on GitHub, for example `ml-assignment2-spambase`.
2. Copy these files into it:
   - `app.py`, `requirements.txt`, `README.md`, `test_data.csv`
   - `model/` containing the five `.joblib` files from the Colab download
   - `ML_Assignment2_Colab.ipynb` (the assignment asks for the source code)
3. Leave out `make_nb.py`, the zip files and any local caches.
4. Commit in several steps with clear messages. **Commit history is reviewed** for plagiarism, so avoid a single large commit.

Example commands:

```bash
cd "Semester 1 2026/ml/assignment2"
git init
git add requirements.txt README.md
git commit -m "Add requirements and README"
git add test_data.csv model/
git commit -m "Add trained models and test data"
git add app.py ML_Assignment2_Colab.ipynb
git commit -m "Add Streamlit app and training notebook"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Check on GitHub that the `model/` folder shows five files and the repo is public.

---

## 5. The requirements.txt file (Step 4)

```
streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
joblib
```

Missing packages are the most common cause of a failed deployment. `app.py` imports `joblib`, `matplotlib`, `seaborn`, `pandas` and `sklearn`. All of them are listed, including `joblib`, which is not in the assignment's example list.

---

## 6. The Streamlit app (`app.py`)

Required features and where they are in the app:

| Required feature | Marks | How the app does it |
|---|---|---|
| CSV upload | 1 | Sidebar file uploader. A checkbox can use the bundled `test_data.csv` instead. |
| Model selection dropdown | 1 | Sidebar `selectbox` with the five models |
| Evaluation metrics | 1 | Six metric tiles: Accuracy, AUC, Precision, Recall, F1, MCC |
| Confusion matrix or classification report | 1 | Both are shown, plus an ROC curve |

Also included: a data preview, and a table comparing all five models on the uploaded data.

How it works:
- The selected model is loaded with `joblib.load`. **Nothing is retrained** on the uploaded data, as the assignment requires.
- The model's `feature_names_in_` gives the expected columns. If the CSV lacks any of them or the `spam` column, the app shows an error listing the missing names.
- If the uploaded file has only one class, AUC and the ROC curve are skipped instead of crashing.

### Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then test it by choosing each of the five models and uploading `test_data.csv`.

---

## 7. Deploy on Streamlit Community Cloud (Step 6)

1. Go to https://streamlit.io/cloud and sign in with your GitHub account.
2. Click **New app**.
3. Select your repository.
4. Choose the branch (`main`).
5. Set the main file to `app.py`.
6. Click **Deploy** and wait for the build.
7. Open the app link. It must show the interactive frontend without errors.
8. Test all four features on the live app, then copy the URL.

### Common problems

| Problem | Likely cause and fix |
|---|---|
| `ModuleNotFoundError` | A package is missing from `requirements.txt`. Add it and push. |
| Model loading fails or warns about version | The model was saved with a different scikit-learn version. Pin the version or re-save the models. |
| `FileNotFoundError` for `model/...` | The `model/` folder was not pushed, or has the wrong file names. Names must match `MODEL_FILES` in `app.py`. |
| App sleeps after inactivity | Normal on the free tier. Open the link once before submitting to wake it. |
| Older Streamlit rejects `width="stretch"` | Update Streamlit. A fresh Cloud deploy installs the latest version. |

---

## 8. Complete the README (Step 5)

The README must follow this structure, and its content must also be in the submitted PDF.

| Section | Marks | Status |
|---|---|---|
| a. Problem statement | | Written |
| b. Dataset description | 1 | Written |
| c. GitHub repository link | 1 | **Add your URL** |
| d. Models used: comparison table with all six metrics for the five models | 5 | **Replace with your Colab results** |
| d. Observations on each model, plus overall winner | 3 | **Rewrite in your own words** |

Before submitting:
1. Replace the two URL placeholders in `README.md` (GitHub repo and Streamlit app).
2. Paste the table printed by the notebook's "Markdown table for the README" cell.
3. Rewrite the observations using your own results. Good points to cover are:
   - which model is best and by how much
   - the precision versus recall trade-off (for spam, a false positive blocks real mail)
   - signs of overfitting from the train/test gap
   - why Naive Bayes is weaker (feature independence assumption)
   - why scaling matters for KNN and SVM
4. Name the overall winner and justify it with the metrics.

---

## 9. BITS Virtual Lab screenshot (1 mark)

The assignment must be performed on **BITS Virtual Lab**, and one screenshot of the assignment running there is required as proof.

1. Log in to BITS Virtual Lab.
2. Run the work there, either the notebook or the app.
3. Take **one** screenshot that clearly shows the lab environment and the execution output.
4. If the lab does not work, email csislabsupport@wilp.bits-pilani.ac.in and raj.rohan@wilp.bits-pilani.ac.in with the subject **"NSP1 ML Assignment 2: BITS Lab issue"**.

---

## 10. Assemble the submission PDF

Submit **one PDF**, with these in this order:

1. GitHub repository link
2. Live Streamlit app link
3. BITS Virtual Lab screenshot
4. The full README content (problem statement, dataset, comparison table, observations)

Open both links in a private/incognito window to confirm they work without logging in.

### Final checklist

- [ ] GitHub repo link works and the repo is public
- [ ] Repo has `app.py`, `requirements.txt`, `README.md`, `test_data.csv`, `model/` with five models, and the notebook
- [ ] Commit history shows several meaningful commits
- [ ] Streamlit app link opens an interactive frontend with no errors
- [ ] CSV upload, model dropdown, metrics and confusion matrix or report all work on the live app
- [ ] README has the real comparison table and your own observations
- [ ] BITS Virtual Lab screenshot is included
- [ ] PDF order is: GitHub link, Streamlit link, screenshot, README content
- [ ] Submitted before 8 November 2026, 23:59

---

## 11. Academic integrity

The assignment checks commit history, repository structure and variable names, Streamlit templates without customisation, and identical dataset, model and output combinations across students. Plagiarism gives zero marks. AI tools are allowed for learning support only, not for direct copy-paste submissions. The code in this folder is a working base. Review it, understand each part and customise it before submitting.
