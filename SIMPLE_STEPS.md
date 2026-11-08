# ML Assignment 2: Simple Steps

Deadline: **8 November 2026, 23:59**. Submit one PDF. Full details are in [`STEPS.md`](STEPS.md).

Work in `Semester 1 2026/ml/assignment2/`. Everything is already built; you run it, publish it and write it up.

---

## 1. Run the notebook on Colab (about 5 minutes)

1. Open https://colab.research.google.com and choose **File > Upload notebook**.
2. Upload `ML_Assignment2_Colab.ipynb`.
3. Choose **Runtime > Run all**.
4. Note the line `scikit-learn version: ...` near the top of the output.
5. Download the three files the last cell gives you: `model_files.zip`, `test_data.csv`, `model_comparison.csv`.

## 2. Update the project folder

1. Unzip `model_files.zip`. Replace the `model/` folder with it.
2. Replace `test_data.csv` with the downloaded one.
3. In `requirements.txt`, set `scikit-learn==<version from step 1>`. It is currently `1.9.1`.

## 3. Check the app on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

Check these four things work:
- upload `test_data.csv`
- the model dropdown has 5 models
- six metrics are shown
- confusion matrix and classification report are shown

## 4. Put it on GitHub

```bash
gh auth login
gh repo create ml-assignment2-spambase --public --source=. --push
```

Git is already set up with 4 commits. Check on github.com that `model/` has 5 files.

## 5. Deploy on Streamlit

1. Go to https://streamlit.io/cloud and sign in with GitHub.
2. Click **New app**, pick your repo, branch `main`, file `app.py`.
3. Click **Deploy**.
4. Open the link, test the four features, and copy the URL.

## 6. Finish the README

Edit `README.md`:
1. Add your GitHub URL and Streamlit URL.
2. Paste your own metrics table from Colab.
3. Rewrite the model observations in your own words and name the overall winner.

Then push the change:

```bash
git add README.md && git commit -m "Update README with final results" && git push
```

## 7. Take the BITS Virtual Lab screenshot

1. Log in to BITS Virtual Lab and run the notebook or the app there.
2. Take **one** screenshot showing the lab and the output.
3. If the lab fails, email csislabsupport@wilp.bits-pilani.ac.in and raj.rohan@wilp.bits-pilani.ac.in with the subject `NSP1 ML Assignment 2: BITS Lab issue`.

## 8. Make the PDF and submit

One PDF, in this order:
1. GitHub repo link
2. Streamlit app link
3. BITS Virtual Lab screenshot
4. The full README content

---

## Final checklist

- [ ] Notebook ran on Colab, new `model/` and `test_data.csv` copied in
- [ ] scikit-learn version pinned in `requirements.txt`
- [ ] App works locally
- [ ] GitHub repo is public with all files and several commits
- [ ] Streamlit link opens and all four features work
- [ ] README has both links, your metrics and your own observations
- [ ] BITS Lab screenshot taken
- [ ] PDF in the right order, submitted before the deadline

## If something goes wrong

| Problem | Fix |
|---|---|
| App cannot load the models on Streamlit | The scikit-learn version does not match. Pin the Colab version in `requirements.txt` and push. |
| `ModuleNotFoundError` on Streamlit | Add the missing package to `requirements.txt` and push. |
| `FileNotFoundError` for a model | The `model/` folder was not pushed, or a file name changed. |
| Colab cell fails with `NameError` | A cell was skipped. Use **Run all**. |
| Streamlit app is asleep | Open the link to wake it before submitting. |
