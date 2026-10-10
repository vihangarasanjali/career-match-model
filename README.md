# Job Matching Binary Classification

## Overview

This project predicts whether a student/job seeker is a suitable match for a given job using Machine Learning, and serves the model through a FastAPI service.

The target variable is:

* `1` → Job Match
* `0` → Job Not Match

---

## Project Structure

```text
career-match-model/
├── main.py                  # FastAPI prediction service
├── requirements.txt
├── artifacts/               # Dataset (Job_Data.csv)
├── notebooks/
│   ├── model_building.ipynb
│   └── prediction_pipeline.ipynb
└── static/
    └── model/
        ├── model.pickle
        └── feature_columns.pickle
```

---

## Dataset

Dataset used: `Job_Data.csv`

* Original dataset: **10000 records**
* Target: `Job_Match`
* Removed columns: `Student_ID`, `Job_ID`, `Gender`, `Location`, `Age`

### Categorical Features

Converted using One-Hot Encoding:

* `Job_Title`
* `Degree_Program`

---

## Data Preprocessing

1. Loaded and explored the dataset.
2. Checked for duplicate values and missing values.
3. Removed unnecessary ID and unused columns.
4. Applied One-Hot Encoding to categorical features.
5. Separated features (`X`) and target (`y`).
6. Split into training and testing sets (80/20, stratified to preserve the class distribution).

## Feature Scaling

`StandardScaler` was used for models that require or benefit from scaling (Logistic Regression and SVM). For these models the scaler is part of a scikit-learn `Pipeline`, so it is fitted on the training data only and applied automatically at prediction time.

Unscaled data was used for Multinomial Naive Bayes (requires non-negative values), Decision Tree and Random Forest.

---

## Models Tested

* Logistic Regression
* Multinomial Naive Bayes
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

Hyperparameter tuning was performed with `GridSearchCV` (5-fold, scored on F1).

## Evaluation Metrics

Accuracy, Precision, Recall and F1-Score. Because the dataset is imbalanced, F1-score and Recall were given particular attention.

## Model Results

| Model                   |  Accuracy | Precision |    Recall |  F1-Score |
| ----------------------- | --------: | --------: | --------: | --------: |
| Logistic Regression     |     0.903 |     0.795 |     0.912 |     0.849 |
| Multinomial Naive Bayes |     0.854 |     0.749 |     0.772 |     0.76  |
| Decision Tree           |     0.905 |     0.832 |     0.857 |     0.844 |
| Random Forest           |     0.937 | **0.916** |     0.87  |     0.892 |
| SVM                     | **0.959** |     0.915 | **0.952** | **0.933** |

### Selected Model

**Support Vector Machine** (highest test F1-score of 0.933 and highest recall of 0.952).

### Probability Calibration

The SVM was wrapped in `CalibratedClassifierCV` (sigmoid, 5-fold) so that it can output probabilities. The calibrated model scores slightly differently on the test set:

| Accuracy | Precision | Recall | F1-Score |
| -------: | --------: | -----: | -------: |
|    0.957 |     0.916 |  0.943 |    0.929 |

On the test set, about 62% of predicted probabilities fall below 10% and about 21% above 90%, with the remainder spread between. The model is therefore very confident on clear-cut cases and gives intermediate values for borderline ones.

---

## Model Saving

The calibrated SVM was saved with `pickle`:

* `static/model/model.pickle`: the trained, calibrated model (includes the scaler).
* `static/model/feature_columns.pickle`: the feature names and order used during training.

Pickles should be loaded with the same scikit-learn version used for training (`scikit-learn==1.9.0`).

---

## Prediction Pipeline

```text
New Input
    ↓
Remove unused columns
    ↓
One-Hot Encoding
    ↓
Match training feature columns
    ↓
Load saved calibrated SVM
    ↓
Predict class + estimated match probability
```

The training dataset is not needed at prediction time. Scaling is handled inside the saved estimator, so no separate `StandardScaler` step is required.

---

## FastAPI Service

`main.py` exposes the model over HTTP. The model and feature columns are loaded once at startup.

### Setup

```bash
python -m venv env
env\Scripts\activate        # Windows
# source env/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### Run

```bash
python -m uvicorn main:app --reload --port 8000
```

Interactive docs: `http://localhost:8000/docs`

### Endpoints

| Method | Path       | Description                                              |
| ------ | ---------- | -------------------------------------------------------- |
| GET    | `/health`  | Service liveness check                                   |
| GET    | `/options` | Valid `Job_Title` and `Degree_Program` values            |
| POST   | `/predict` | Predict job match for one candidate/job pair             |

### Request (`POST /predict`)

`Academic_Performance` is a **percentage** (0-100). Any CGPA conversion must be done by the caller before sending. `Job_Title` and `Degree_Program` must be values returned by `GET /options`; unknown values return `422`.

```json
{
  "Academic_Performance": 75,
  "Certifications_Count": 2,
  "Internship_Experience": 6,
  "Skill_Programming": 7,
  "Skill_Database_SQL": 7,
  "Skill_Version_Control_Git": 7,
  "Skill_Problem_Solving": 8,
  "Skill_Communication": 7,
  "Required_Programming": 8,
  "Required_Database_SQL": 8,
  "Required_Version_Control_Git": 8,
  "Required_Problem_Solving": 8,
  "Required_Communication": 8,
  "Min_Experience_Months": 6,
  "Job_Title": "<value from /options>",
  "Degree_Program": "<value from /options>"
}
```

### Response

```json
{
  "result": "Job Match",
  "prediction": 1,
  "match_percentage": 82.02
}
```

* `prediction`: `1` for Job Match, `0` for Job Not Match.
* `match_percentage`: the model's estimated probability of a positive match from `predict_proba()`. It is **not** the percentage of job requirements satisfied.

---

## Integration

```text
React (career-match-web)  →  Node.js API (:5000)  →  FastAPI (:8000)  →  Model
```

The Node.js backend maps the frontend fields to the model's column names, converts CGPA to a percentage, and calls this service at `ML_API_URL` (default `http://localhost:8000`).




