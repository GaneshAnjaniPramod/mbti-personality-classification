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
| Instances | 59999 (47999 train / 12000 test, stratified 80/20 split) |
| Features | 62 (one per questionnaire statement) |
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
├── mbti_personality_classifier_trained_models.ipynb      # training + evaluation
├── requirements.txt
├── README.md
├── 16P.csv                 # Input Dataset
├── test_data.csv           # held-out test set (features + target)
├── .streamlit/config.toml  # app theme
└── model/
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── logistic_regression.pkl
    ├── metrics_summary.csv
    ├── model_info.json      # feature list, classes, versions (used by the app to validate uploads)
    ├── naive_bayes.pkl
    └── svm.pkl
```

## d. Models used

1. **Logistic Regression** – multinomial, `max_iter=2000`, standardized features
2. **Decision Tree** – `max_depth=20`, `min_samples_leaf=5`
3. **K-Nearest Neighbors** – `k=15`, distance-weighted, standardized features
4. **Naive Bayes** – Gaussian (features contain negative values, so Multinomial NB is not applicable)
5. **Support Vector Machine** – linear SVM (`LinearSVC`, `C=0.5`) with `CalibratedClassifierCV` to produce
   class probabilities for AUC. A linear kernel was chosen because RBF-SVC with probability estimates
   is very slow on ~48k rows × 16 classes.

Metrics are computed on the held-out test set. 
Precision, Recall and F1 are **macro-averaged**; 
AUC is **one-vs-rest macro AUC**; 
MCC is the multi-class Matthews correlation coefficient.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9183 | 0.9928 | 0.9183 | 0.9183 | 0.9182 | 0.9129 |
| Decision Tree | 0.6488 | 0.8649 | 0.6501 | 0.6487 | 0.6486 | 0.6254 |
| KNN | 0.9870 | 0.9947 | 0.9870 | 0.9870 | 0.9870 | 0.9861 |
| Naive Bayes | 0.9118 | 0.9926 | 0.9122 | 0.9118 | 0.9117 | 0.9060 |
| Support Vector Machine | 0.9106 | 0.9915 | 0.9104 | 0.9105 | 0.9104 | 0.9046 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong performance (≈ 91.8% accuracy, 980 errors), and the best of the three linear and probabilistic models. Errors are concentrated in ESFJ, ISFJ and ISFP, which all share the S and F letters. This suggests a straight decision boundary cannot fully separate types that differ on a single axis. |
| Decision Tree | Weakest model by a wide margin (≈ 64.9% accuracy, 4,215 errors). Errors are spread across many types (ISTP, ISFP, INTJ) rather than concentrated, which points to a general limitation rather than one confusing pair of types. A tree splits on one question at a time, while each personality axis depends on the combined effect of many questions, so the tree needs very many splits to approximate it. |
| KNN | Best model (≈ 98.7% accuracy, 156 errors), with every class scoring 0.978–0.995 on precision, recall and F1. Its few errors are spread thinly (13–16 per type), with no type standing out. This suggests people of the same type give very similar answer patterns, forming tight clusters that a distance-based model can use directly. The trade-off is cost: KNN stores the full training set and is the slowest model to predict. |
| Naive Bayes | Close to the linear models (≈ 91.2% accuracy, 1,058 errors) despite assuming every question is independent, which is not true for questions on the same axis. It confuses the same S/F types as Logistic Regression and SVM. Given its simplicity and speed, this is a good result. |
| Support Vector Machine | Nearly identical to Logistic Regression (≈ 91.1% accuracy, 1,073 errors), with the same most-confused types. Since both are linear classifiers, this indicates that the shape of the decision boundary, not the choice of algorithm, is what limits these models. An RBF kernel might close the gap with KNN, but it was too slow to train on this data size. |
| **Overall winner for this dataset** | **KNN**: highest on every metric and roughly 6× fewer errors than the next-best model. Because the 16 classes are balanced (6.22%–6.28% each), accuracy is a fair comparison here and macro and weighted averages are identical. |

## Streamlit app features

- CSV upload of test data (plus a bundled sample and a download button for it)
- Model selection dropdown (all five classifiers)
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Per-axis accuracy (E/I, S/N, T/F, J/P) and side-by-side comparison of all five models


## How to run locally

```bash
pip install -r requirements.txt
jupyter notebook mbti_personality_classifier_trained_models.ipynb
streamlit run app.py
```