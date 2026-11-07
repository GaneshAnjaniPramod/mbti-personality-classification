# ML Assignment 2 - MBTI Personality Type Classification

**Live app:** TBD
**Author:** BARU GANESH ANJANI PRAMOD (2026AJ05610)

## a. Problem statement

Predict a person's Myers-Briggs personality type (one of 16 classes such as INTJ, ENFP) from their
answers to a 60-item personality questionnaire. This is a **multi-class classification** problem.
Five classifiers are trained on the same data, compared on six evaluation metrics, and deployed in an
interactive Streamlit app that evaluates the saved models on uploaded test data.

## b. Dataset description

| Item | Value |
|---|---|
| Source | Kaggle – "Personality classification Data: 16 Personalities" by anshulmehtakaggl |
| Link | https://www.kaggle.com/datasets/anshulmehtakaggl/60k-responses-of-16-personalities-test-mbt |
| Instances | 60,000 (47,999 train / 12,000 test, stratified 80/20 split) |
| Features | 60 (one per questionnaire statement) |
| Feature type | Integer, Likert-style score from −3 (strongly disagree) to +3 (strongly agree) |
| Target | `Personality` – 16 MBTI types |
| Dropped column | `Response Id` (identifier, not a feature) |
| Missing values | None (0 Missing values across all 60 features and the target) |
| Duplicate values | 0 (0 Duplicate values across all 60 features and the target) |
| Class balance | Balanced: each of the 16 types has 3,734–3,769 rows (6.22%–6.28% of the data) |

**Preprocessing:** dropped the ID column, stratified 80/20 train–test split (`random_state=<yours>`),
`StandardScaler` for Logistic Regression, KNN and SVM (inside a scikit-learn `Pipeline`, so the scaler
is saved with the model). Decision Tree and Naive Bayes use raw scores. The 20% hold-out set is saved as
`test_data.csv` and is the file uploaded to the app.

## c. GitHub repository link
https://github.com/GaneshAnjaniPramod/mbti-personality-classification

```
project-folder/
├── app.py                  # Streamlit app
├── train_models.ipynb      # training + evaluation (run on BITS Virtual Lab)
├── requirements.txt
├── README.md
├── test_data.csv           # held-out test set (features + target)
├── .streamlit/config.toml  # app theme
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    ├── model_info.json      # feature list, classes, versions (used by the app to validate uploads)
    └── metrics_summary.csv
```

## d. Models used

1. **Logistic Regression** – multinomial, `max_iter=2000`, standardized features
2. **Decision Tree** – `max_depth=20`, `min_samples_leaf=5`
3. **K-Nearest Neighbors** – `k=15`, distance-weighted, standardized features
4. **Naive Bayes** – Gaussian (features contain negative values, so Multinomial NB is not applicable)
5. **Support Vector Machine** – linear SVM (`LinearSVC`, `C=0.5`) with `CalibratedClassifierCV` to produce
   class probabilities for AUC. A linear kernel was chosen because RBF-SVC with probability estimates
   is very slow on ~48k rows × 16 classes.

Metrics are computed on the held-out test set. Precision, Recall and F1 are **macro-averaged**; AUC is
**one-vs-rest macro AUC**; MCC is the multi-class Matthews correlation coefficient.

### Comparison table

<!-- Replace with the table printed by Cell 11 of the notebook -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <your observation> |
| Decision Tree | <your observation> |
| KNN | <your observation> |
| Naive Bayes | <your observation> |
| Support Vector Machine | <your observation> |
| **Overall winner for this dataset** | <model + reason> |

## Streamlit app features

- CSV upload of test data (plus a bundled sample and a download button for it)
- Model selection dropdown (all five classifiers)
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix (counts or row %) and full classification report
- Per-axis accuracy (E/I, S/N, T/F, J/P) and side-by-side comparison of all five models
- Models are loaded from `model/`; nothing is retrained in the app

## How to run locally

```bash
pip install -r requirements.txt
jupyter notebook train_models.ipynb   # optional: retrain and regenerate model/ and test_data.csv
streamlit run app.py
```