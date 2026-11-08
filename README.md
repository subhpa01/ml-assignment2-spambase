# Spam Email Classification - ML Assignment 2

## a. Problem statement

Classify an email as **spam (1)** or **not spam (0)** from numeric text features. Five classifiers are trained on the same dataset and compared on six metrics (Accuracy, AUC, Precision, Recall, F1, MCC). The models are deployed in a Streamlit app that evaluates them on an uploaded test CSV.

## b. Dataset description

- **Source:** Spambase, UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/94/spambase)
- **Instances:** 4,601 emails (about 39.4% spam)
- **Features:** 57 numeric
  - 48 word-frequency features (`word_freq_*`, % of words matching a word)
  - 6 character-frequency features (`char_freq_*`)
  - 3 capital-letter run-length features
- **Target:** `spam` (1 = spam, 0 = not spam)
- **Split:** stratified 80/20 train/test, `random_state=42`. The 921-row test portion is saved as `test_data.csv`.
- **Preprocessing:** `StandardScaler` inside the pipelines of Logistic Regression, KNN and SVM. The Decision Tree and Naive Bayes use raw features.

## c. GitHub repository link

https://github.com/subhpa01/ml-assignment2-spambase

Live Streamlit app: https://ml-assignment2-spambase-qoxm3g3pq28b6oefvchqwz.streamlit.app/

```
assignment2/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- ML_Assignment2_Colab.ipynb   (training notebook)
|-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- svm.joblib
```

## d. Models used

Test-set results (921 emails), computed from the saved models trained with the included notebook (scikit-learn 1.7.2).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9294 | 0.9702 | 0.9209 | 0.8981 | 0.9093 | 0.8518 |
| Decision Tree | 0.9034 | 0.9154 | 0.9029 | 0.8457 | 0.8734 | 0.7965 |
| KNN | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| Naive Bayes | 0.8339 | 0.9449 | 0.7178 | 0.9532 | 0.8189 | 0.6941 |
| Support Vector Machine | 0.9273 | 0.9667 | 0.9277 | 0.8843 | 0.9055 | 0.8471 |

### Observations

Write these in your own words after checking your results. The draft below is based on the numbers above.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best accuracy, AUC, F1 and MCC. Spam is close to linearly separable in the scaled word-frequency space, and the model is fast and interpretable. |
| Decision Tree | Lowest AUC (0.915) and weaker recall than the linear models. Depth is capped at 8, so it is simple, but a single tree gives coarse probability estimates. |
| KNN | Moderate results. Distance-based, so it relies on scaling, and it is slower at prediction time on 57 features. |
| Naive Bayes | Highest recall (0.953) but lowest precision (0.718), so it flags many legitimate emails as spam. The independence assumption does not hold for correlated word frequencies, and Gaussian density fits skewed features poorly. |
| Support Vector Machine | Very close to Logistic Regression, with the highest precision (0.928). The RBF kernel captures non-linear structure but needs more training time. |

**Overall winner:** Logistic Regression has the best Accuracy, AUC, F1 and MCC, with SVM a close second. For a spam filter, where blocking real mail is costly, SVM's slightly higher precision is a reasonable alternative.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit app features

1. CSV upload of test data (needs the 57 feature columns plus `spam`)
2. Model selection dropdown
3. Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
4. Confusion matrix, ROC curve and classification report
5. Comparison table of all five models on the uploaded data

The models are loaded from `model/` and are not retrained on the uploaded data.
