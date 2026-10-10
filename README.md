# Job Matching Binary Classification

## Overview

This project aims to predict whether a student/job seeker is a suitable match for a given job using Machine Learning.

The target variable is:

* `1` → Job Match
* `0` → Job Not Match

---

## Dataset

Dataset used:

`Job_Data.csv`

* Original dataset: **10000 records**
* Target: `Job_Match`
* Removed identifier columns:

  * `Student_ID`
  * `Job_ID`
  * `Gender`
  * `Location`
  * `Age`


### Categorical Features

The following categorical features were converted using One-Hot Encoding:

* `Job_Title`
* `Degree_Program`

---

## Data Preprocessing

The following preprocessing steps were completed:

1. Loaded and explored the dataset.
2. Checked for duplicate values and missing values.
3. Removed unnecessary ID columns.
4. Applied One-Hot Encoding to categorical features.
5. Separated features (`X`) and target (`y`).
6. Split the dataset into training and testing sets using an 80/20 split.
7. Used stratified splitting to preserve the target class distribution.


## Feature Scaling

`StandardScaler` was used for models that require or benefit from feature scaling.

Scaled data was used for:

* Logistic Regression
* SVM

Unscaled data was used for:

* Multinomial Naive Bayes
* Decision Tree
* Random Forest

Multinomial Naive Bayes was kept unscaled because it requires non-negative feature values.

---

## Models Tested

The following classification algorithms were evaluated:

* Logistic Regression
* Multinomial Naive Bayes
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

For some models, hyperparameter tuning was performed using `GridSearchCV`.

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

Since the dataset is imbalanced, **F1-score and Recall** were given particular attention rather than relying only on accuracy.

---

## Model Results

Current test results:

| Model                   |  Accuracy | Precision |    Recall |  F1-Score |
| ----------------------- | --------: | --------: | --------: | --------: |
| Logistic Regression     |     0.903 |     0.795 |     0.912 |     0.849 |
| Multinomial Naive Bayes |     0.854 |     0.749 |     0.772 |     0.76  |
| Decision Tree           |     0.905 |     0.832 |     0.857 |     0.844 |
| Random Forest           |     0.937 | **0.916** |     0.87  |     0.892 |
| SVM                     | **0.959** |     0.915 | **0.952** | **0.933** |

### Selected Model

**Support Vector Machine**

It was selected based mainly on its:

* Highest test F1-score: **0.933**
* Highest recall: **0.952** among the tested models

The model provides a good balance between identifying matching and non-matching cases.

---

## Model Saving

The trained Support Vector Machine model was saved using `pickle`.

Current model files:

```text
static/
└── model/
    ├── model.pickle
    └── feature_columns.pickle
```

`model.pickle` contains the trained model.

`feature_columns.pickle` contains the feature names and their order used during training. This is required to ensure that new input has the same feature structure as the training data.

---

## Prediction Pipeline

A prediction pipeline was created for making predictions on new input.

The prediction process is:

```text
New Input
    ↓
Remove unnecessary ID columns
    ↓
One-Hot Encoding
    ↓
Match training feature columns
    ↓
Load saved SVM model
    ↓
Predict Job Match Class
    ↓
Calculate Estimated Match Probability
    ↓
Return Prediction and Match Percentage
```

The original training dataset does not need to be loaded again during prediction. The saved model and feature columns are sufficient, provided that the input preprocessing matches the training process.

The calibrated SVM handles the scaling and classification steps internally because the scaler is included in the saved estimator. No separate StandardScaler operation is required during prediction.

---

## Prediction Output
``` 

The calibrated SVM provides two outputs:

Predicted class: 1 for Job Match or 0 for Job Not Match.

Estimated match probability: The model's estimated probability that the input belongs to the Job Match class.

Example result:

Prediction: Job Match
Estimated Match Probability: 99.4%

The match percentage is obtained using predict_proba() and represents the model's estimated probability of a positive match, not the percentage of job requirements satisfied.

The probability should be validated on unseen test data before being interpreted as a reliable estimate.
```

## Next Steps

* Build the application interface/API for new user input.
* Connect the saved model to the application.
* Create a user-friendly prediction result.
* Test the model with different new inputs.
* Improve and validate the complete prediction system.

