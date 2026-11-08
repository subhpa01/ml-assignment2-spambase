# ML Assignment 2: Complete Step-by-Step Guide (Run on Google Colab)

Follow these steps in order, from running the notebook on Colab to submitting the final PDF.

**Deadline:** 8 November 2026, 23:59. No extensions and no draft submissions.
**Marks (15):** model implementation and GitHub (10), Streamlit app (4), BITS Virtual Lab screenshot (1).

**You need:** a Google account, a GitHub account, and a BITS Virtual Lab login. The dataset downloads automatically from UCI, so there is nothing to upload to Colab except the notebook.

## Overview

| Phase | Steps | Where |
|---|---|---|
| A. Train and save the models | 1 to 6 | Google Colab |
| B. Prepare the project folder | 7 to 9 | Your computer |
| C. Publish and deploy | 10 to 12 | GitHub and Streamlit Cloud |
| D. Write up and submit | 13 to 16 | README, BITS Virtual Lab, PDF |

### Files in `Semester 1 2026/ml/assignment2/`

| File | Purpose |
|---|---|
| `ML_Assignment2_Colab.ipynb` | Notebook: loads data, trains five models, computes metrics, saves models and test data |
| `app.py` | Streamlit app that loads the saved models and evaluates them on an uploaded test CSV |
| `requirements.txt` | Packages for deployment |
| `README.md` | Required README (you must complete it, see Step 13) |
| `test_data.csv` | 921-row held-out test set (57 features plus `spam`) |
| `model/*.joblib` | Five saved models |
| `model_comparison.csv` | Metric table from the notebook |
| `make_nb.py` | Generated the notebook. Not needed for submission. |
| `STEPS.md` | Shorter general guide |

---

# Phase A: Train and save the models on Colab

## Step 1: Open Colab and upload the notebook

1. Go to https://colab.research.google.com and sign in.
2. In the dialog, choose the **Upload** tab (or use **File > Upload notebook**).
3. Select `ML_Assignment2_Colab.ipynb` from `Semester 1 2026/ml/assignment2/`.

## Step 2: Check the runtime

1. Choose **Runtime > Change runtime type**.
2. Keep **Python 3** with **CPU**. No GPU is needed.
3. Click **Save**.

## Step 3: Run all cells

1. Choose **Runtime > Run all** (Ctrl/Cmd + F9).
2. If Colab warns "This notebook was not authored by Google", click **Run anyway**.
3. Wait about 3 to 5 minutes. The SVM is the slowest step.

To go slowly, click each cell and press **Shift + Enter**, always in order. Later cells use variables from earlier ones.

## Step 4: Check the output of each section

| Section | What you should see |
|---|---|
| 0. Setup | `scikit-learn version: ...`. **Write this version down** (Step 7). |
| 1. Dataset | Shape `(4601, 58)`, a preview table, `Missing values: 0`, and a bar chart with 2,788 not-spam and 1,813 spam |
| 2. Split | `train: (3680, 57)  test: (921, 57)`, spam ratio about 0.394 in both |
| 3. Models | Five lines `trained: <model name>` |
| 4. Metrics | Test-set table of Accuracy, AUC, Precision, Recall, F1, MCC, then training-set metrics and the train/test accuracy gap |
| 4. README table | A markdown table printed as text, and `Best model by MCC: ...` |
| 5. Plots | Five confusion matrices, one ROC chart, a classification report per model |
| 6. Save | `Saved: [...five .joblib names...] + test_data.csv (921, 58)` |
| 7. Download | The browser downloads three files |

Expected test-set values from my local run. Yours may differ slightly, so **use your own numbers**:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9294 | 0.9702 | 0.9209 | 0.8981 | 0.9093 | 0.8518 |
| Decision Tree | 0.9034 | 0.9154 | 0.9029 | 0.8457 | 0.8734 | 0.7965 |
| KNN | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| Naive Bayes | 0.8339 | 0.9449 | 0.7178 | 0.9532 | 0.8189 | 0.6941 |
| Support Vector Machine | 0.9273 | 0.9667 | 0.9277 | 0.8843 | 0.9055 | 0.8471 |

Numbers within about 0.01 of these mean the run is fine.

### What the notebook trains

| Model | Preprocessing | Settings |
|---|---|---|
| Logistic Regression | StandardScaler | `max_iter=2000` |
| Decision Tree | none | `max_depth=8` |
| KNN | StandardScaler | `n_neighbors=5` |
| Naive Bayes | none | `GaussianNB` |
| Support Vector Machine | StandardScaler | RBF kernel, `probability=True` |

Each model is a scikit-learn `Pipeline`, so scaling is saved inside the model and the app needs no separate scaler. The split is stratified 80/20 with `random_state=42`. Models are fit on the training part only.

## Step 5: Download the files

Section 7 downloads three files:

| File | Use |
|---|---|
| `model_files.zip` | Unzip to get the `model/` folder with five `.joblib` files |
| `test_data.csv` | The 921-row test set for the app |
| `model_comparison.csv` | The metric table, for reference |

If Chrome blocks the 2nd and 3rd downloads, click **Allow**, then re-run the last cell.

**Alternative:** open the folder icon in the left sidebar, find `model/`, `test_data.csv` and `model_comparison.csv`, right-click each and choose **Download**.

Colab deletes these files when the session ends. Download them before closing the tab.

## Step 6: Keep evidence of the Colab run

Take a screenshot of the metrics table and plots for your own record. This does **not** replace the BITS Virtual Lab screenshot (Step 14).

---

# Phase B: Prepare the project folder

## Step 7: Pin the scikit-learn version

Saved `.joblib` files load safely only with the scikit-learn version that saved them.

1. Take the version printed in Section 0 (for example `1.6.1`).
2. Edit `requirements.txt` so the scikit-learn line reads `scikit-learn==1.7.2` (use your version).

Final `requirements.txt` (pin only scikit-learn):

```
streamlit
scikit-learn==<your version>
numpy
pandas
matplotlib
seaborn
joblib
```

Do not skip this. A version mismatch is the most likely reason the deployed app fails to load the models. `joblib` is listed because `app.py` imports it, even though it is not in the assignment's example list.

## Step 8: Put the Colab files into the project folder

1. Unzip `model_files.zip`. You should get five files:
   - `logistic_regression.joblib`
   - `decision_tree.joblib`
   - `knn.joblib`
   - `naive_bayes.joblib`
   - `svm.joblib`
2. Replace the existing `model/` folder and `test_data.csv` in `Semester 1 2026/ml/assignment2/` with the Colab versions.
3. Keep the file names exactly. `app.py` looks for these names.

The assignment requires this layout in the repository:

```
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model/   (five saved models)
```

## Step 9: Test the app locally

```bash
cd "Semester 1 2026/ml/assignment2"
pip install -r requirements.txt
streamlit run app.py
```

In the browser, check the four required features:

| Required feature | Marks | How to test |
|---|---|---|
| CSV upload | 1 | Upload `test_data.csv` in the sidebar |
| Model selection dropdown | 1 | Pick each of the five models |
| Evaluation metrics | 1 | Six metric tiles appear and match your Colab table |
| Confusion matrix or classification report | 1 | Both appear, plus an ROC curve |

Also try a CSV with a missing column. The app should show an error, not crash.

The app loads the saved models and does not retrain them on the uploaded data, as the assignment requires.

**Optional: preview the app inside Colab** (the final deployment must still be on Streamlit Cloud). Upload `app.py`, `test_data.csv` and the `model/` folder to Colab, then run:

```python
!pip install -q streamlit
!npm install -q localtunnel
```

```python
import urllib
print("Tunnel password (your public IP):",
      urllib.request.urlopen("https://ipv4.icanhazip.com").read().decode().strip())
!streamlit run app.py &>/content/logs.txt &
!npx localtunnel --port 8501
```

Open the `https://....loca.lt` link and enter the printed IP as the password.

---

# Phase C: Publish and deploy

## Step 10: Push to GitHub

1. Create a new **public** repository on GitHub, for example `ml-assignment2-spambase`.
2. The repository must contain:
   - `app.py`, `requirements.txt`, `README.md`, `test_data.csv`
   - `model/` with the five `.joblib` files
   - `ML_Assignment2_Colab.ipynb` (the source code)
3. Leave out `make_nb.py`, zip files and cache folders.
4. Commit in several steps. **Commit history is reviewed** for plagiarism, so avoid one large commit.

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

Check on GitHub that `model/` shows five files and the repository is public.

## Step 11: Deploy on Streamlit Community Cloud

1. Go to https://streamlit.io/cloud and sign in with GitHub.
2. Click **New app**.
3. Select your repository.
4. Choose the branch (`main`).
5. Set the main file to `app.py`.
6. Click **Deploy** and wait for the build.

## Step 12: Test the live app

1. Open the app URL. It must show the interactive frontend with no errors.
2. Upload `test_data.csv` and check all four features from Step 9.
3. Open the URL in a private or incognito window to confirm it works without logging in.
4. Copy the URL.

Free-tier apps go to sleep after inactivity. Open the link shortly before the deadline to wake it.

| Problem | Likely cause and fix |
|---|---|
| `ModuleNotFoundError` | A package is missing from `requirements.txt`. Add it and push. |
| Model loading fails or warns about version | Pin the scikit-learn version (Step 7), or re-save the models on Colab. |
| `FileNotFoundError` for `model/...` | `model/` was not pushed, or file names differ from `MODEL_FILES` in `app.py`. |
| Streamlit rejects `width="stretch"` | The app needs a recent Streamlit. Streamlit Cloud installs the latest, so redeploy. For an older local install, run `pip install -U streamlit`. |

---

# Phase D: Write up and submit

## Step 13: Complete the README

The README must follow this structure, and its content must also be in the submitted PDF.

| Section | Marks | Action |
|---|---|---|
| a. Problem statement | | Already written |
| b. Dataset description | 1 | Already written |
| c. GitHub repository link | 1 | **Add your repo URL and Streamlit URL** |
| d. Models used: comparison table with six metrics for five models | 5 | **Paste your Colab table** (from the "Markdown table for the README" cell) |
| d. Observations for each model, plus overall winner | 3 | **Rewrite in your own words** |

Points to cover in the observations:
- which model is best and by how much
- the precision versus recall trade-off (for spam, a false positive blocks real mail)
- signs of overfitting from the train/test gap shown in Section 4
- why Naive Bayes is weaker (feature independence assumption)
- why scaling matters for KNN, SVM and logistic regression

Name the overall winner and justify it with the metrics. Then commit and push the updated `README.md`.

## Step 14: BITS Virtual Lab screenshot (1 mark)

The assignment must be performed on **BITS Virtual Lab**, and **one** screenshot is required as proof.

1. Log in to BITS Virtual Lab.
2. Run the work there, either the notebook or `streamlit run app.py`.
3. Take one screenshot that clearly shows the lab environment and the execution output.
4. If the lab does not work, email csislabsupport@wilp.bits-pilani.ac.in and raj.rohan@wilp.bits-pilani.ac.in with the subject **"NSP1 ML Assignment 2: BITS Lab issue"**.

## Step 15: Assemble the submission PDF

Submit **one PDF**, with these in this order:

1. GitHub repository link
2. Live Streamlit app link
3. BITS Virtual Lab screenshot
4. The full README content (problem statement, dataset, comparison table, observations)

## Step 16: Final checklist

- [ ] Colab notebook ran top to bottom without errors
- [ ] scikit-learn version pinned in `requirements.txt` (Step 7)
- [ ] Colab-generated `model/` (five files) and `test_data.csv` are in the project folder
- [ ] App works locally with all four required features
- [ ] GitHub repo is public and has `app.py`, `requirements.txt`, `README.md`, `test_data.csv`, `model/`, and the notebook
- [ ] Commit history shows several meaningful commits
- [ ] Streamlit app link opens an interactive frontend with no errors
- [ ] README has the real comparison table and your own observations
- [ ] README has the GitHub and Streamlit links filled in
- [ ] BITS Virtual Lab screenshot is included
- [ ] PDF order is: GitHub link, Streamlit link, screenshot, README content
- [ ] Submitted before 8 November 2026, 23:59

---

## Colab troubleshooting

| Problem | Fix |
|---|---|
| `URLError` or a download error in Section 1 | The UCI server was briefly unreachable. Wait a minute and re-run the cell. |
| `NameError: name 'df' is not defined` | A cell was skipped. Use **Runtime > Run all** or run cells in order. |
| Section 3 is slow | The SVM takes the longest, about a minute. Wait for it. |
| `ModuleNotFoundError` | Run `!pip install scikit-learn pandas matplotlib seaborn joblib` in a new cell, then re-run. |
| Downloads do not start | Allow multiple downloads in the browser and re-run Section 7, or download from the file sidebar. |
| Session disconnected and files are gone | Run all cells again. The random seed is fixed, so results are the same. |
| Metrics differ slightly from the table above | Normal across scikit-learn versions. Use your own numbers. |

## Academic integrity

The assignment checks commit history, repository structure and variable names, uncustomised Streamlit templates, and identical dataset, model and output combinations across students. Plagiarism gives zero marks. AI tools are allowed for learning support only, not for direct copy-paste submissions. The code here is a working base. Review it, understand each part and customise it before submitting. Changing the dataset, hyperparameters, split seed or app layout will make your work distinct, and your README observations must be in your own words.
