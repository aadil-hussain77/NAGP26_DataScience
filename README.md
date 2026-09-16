# Customer Churn Prediction

## Git Repository

This project is hosted in the following Git repository:

- GitHub: [NAGP26_DataScience](https://github.com/aadil-hussain77/NAGP26_DataScience.git)
- Remote URL: `https://github.com/aadil-hussain77/NAGP26_DataScience.git`
- Default branch: `main`

## Project Overview

This project develops an end-to-end machine learning solution to predict whether a telecom customer is likely to churn.

The objective is to help the customer-retention team identify high-risk customers early and take proactive retention actions.

The project uses the IBM Telco Customer Churn dataset and a Decision Tree Classifier.

## Business Problem

Customer churn directly affects telecom revenue and customer lifetime value. This project predicts the target variable:

```text
Churn
```

Possible values are:

```text
Yes
No
```

A prediction of `Yes` indicates that the customer is likely to leave the company.

## Project Structure

```text
customer_churn_project/
│
├── data/
│   ├── TelcoCustomerChurn.csv
│   └── TelcoCustomerChurn - Data Dictionary.csv
│
├── notebook/
│   └── churn_analysis.ipynb
│
├── model/
│   └── churn_pipeline.joblib
│
├── app.py
├── requirements.txt
├── README.md
└── sample_request.json
```

## Features

The model uses customer demographic, account, service, and billing information, including:

- Customer tenure
- Contract type
- Internet service
- Online security
- Technical support
- Monthly charges
- Total charges
- Payment method
- Paperless billing

Additional engineered features are created before prediction:

- `TenureGroup`
- `ServiceCount`
- `SupportProtectionCount`

## Model

The final model is a regularized Decision Tree Classifier trained using a stratified
70:30 train-test split and `random_state=42`.

Final Decision Tree configuration:

```text
max_depth = 5
min_samples_split = 20
min_samples_leaf = 10
class_weight = balanced
random_state = 42
```

The preprocessing and model are stored together in a scikit-learn pipeline. This prevents
preprocessing inconsistencies between training and API prediction.

On the unseen test set, the final model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 0.7113 |
| Precision for `Churn = Yes` | 0.4741 |
| Recall for `Churn = Yes` | 0.7986 |
| F1 Score for `Churn = Yes` | 0.5950 |

The model correctly identified 448 of 561 actual churners in the test set. The model is
therefore intended as a customer-prioritization tool for retention campaigns rather than
an automatic decision system.

## Data Preparation and Preprocessing

The notebook includes data-quality checks for data types, hidden blank values, duplicates,
feature types, and target distribution.

Key preparation decisions:

- `customerID` is excluded because it is a unique identifier rather than a predictive
  customer attribute.
- Blank values in `TotalCharges` are converted to numeric values and handled safely.
- The dataset is split using a stratified 70:30 train-test split with `random_state=42`.
- Numerical features use median imputation.
- Categorical features use most-frequent-value imputation and one-hot encoding.
- Preprocessing is fitted only on the training set through a scikit-learn pipeline,
  preventing data leakage.
- The pipeline uses `handle_unknown="ignore"` so unseen categories can be processed
  safely at prediction time.

## Exploratory Data Analysis Insights

The notebook contains more than five visualizations covering churn distribution, customer
characteristics, service usage, contract type, billing behaviour, and numerical
relationships.

Key findings include:

- Overall churn rate: 26.54%.
- Month-to-month customers have the highest churn rate: 42.71%.
- Customers with 0-12 months of tenure have the highest churn rate: 47.44%.
- Fiber optic customers have a churn rate of 41.89%.
- Electronic-check customers have a churn rate of 45.29%.
- Customers without technical support have a churn rate of 41.64%, compared with
  15.17% among customers with technical support.
- Churned customers have higher average monthly charges: 74.44, compared with
  61.27 for retained customers.

These segments are suitable priorities for targeted retention campaigns, onboarding
support, service-quality checks, and contract-upgrade offers.

## Model Evaluation and Business Interpretation

The final model is evaluated on the untouched test set using:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

For churn prediction, recall for the `Churn = Yes` class is especially important. A false
negative means a customer who is actually likely to churn is not identified, causing the
business to lose the opportunity to take a proactive retention action. A false positive
may result in an unnecessary retention offer, but this is often less costly than losing a
customer.

The notebook also includes feature-importance analysis and Decision Tree interpretation
to explain the factors influencing churn predictions.

## Model Persistence and Reliability Check

The final preprocessing-and-model pipeline is saved as:

```text
model/churn_pipeline.joblib
```

The saved artifact was reloaded and tested against the original pipeline. All predictions
from the reloaded pipeline exactly matched the original model predictions:

```text
All reloaded-model predictions match: True
```

This confirms that the persisted model can be reused reliably for future predictions.

## Bonus Enhancements

The assignment allows bonus credit for additional activities such as hyperparameter tuning,
class-imbalance handling, comparing additional models, or other meaningful improvements.

This project implements the following bonus-quality enhancements:

- **Hyperparameter tuning:** `GridSearchCV` was used to tune the Decision Tree with
  stratified 5-fold cross-validation on the training data only. The search evaluated
  24 configurations across `max_depth`, `min_samples_leaf`, and `class_weight`.

- **Business-aligned tuning metric:** A custom scorer optimized recall for
  `Churn = Yes`, because missing an actual churner prevents a proactive retention action.

- **Tuning result:** The best GridSearchCV configuration used:

  ```text
  max_depth = 5
  min_samples_leaf = 10
  class_weight = balanced
  ```

  It achieved a mean 5-fold cross-validation churn recall of **0.7844**. On the
  untouched test set, it achieved accuracy of 0.7113, precision of 0.4741, recall of
  0.7986, and F1 score of 0.5950. These results matched the manually regularized
  final Decision Tree and independently validated the final-model design.

- **Class-imbalance handling:** The churn class represents 26.54% of the dataset.
  The final Decision Tree uses `class_weight="balanced"` and evaluation focuses on
  churn precision, recall, F1 score, and the confusion matrix rather than accuracy alone.

- **Feature engineering beyond the minimum:** Three features were created:
  `TenureGroup`, `ServiceCount`, and `SupportProtectionCount`, exceeding the minimum
  requirement of two engineered features.

- **Reliable deployment workflow:** The solution uses a leakage-safe preprocessing and
  modelling pipeline, saved-model reload verification, FastAPI deployment, Pydantic
  schema validation, automatic Swagger documentation, and HTTP 422 errors for invalid
  input.

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Run the notebook

Start JupyterLab from the project folder:

```powershell
jupyter lab
```

Then open:

```text
notebook/churn_analysis.ipynb
```

Run all notebook cells in order to reproduce data preparation, EDA, feature engineering, model training, evaluation, interpretation, and model saving.

## Run the API

From the main project folder, start the FastAPI server:

```powershell
python -m uvicorn app:app --reload
```

Open the interactive API documentation in a browser:

```text
http://127.0.0.1:8000/docs
```

### Health Check

Open the following URL after the server starts:

```text
http://127.0.0.1:8000/
```

Expected response:

```json
{
  "message": "Customer Churn Prediction API is running.",
  "documentation": "/docs"
}
```

## API Endpoint

### `POST /predict`

The endpoint accepts customer data as JSON and returns a churn prediction and churn probability.

Example response:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8501
}
```

## Example Request

Use the `sample_request.json` file as the request body in the FastAPI Swagger UI.

Example request:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 179.0
}
```

## Input Validation

The API validates required fields, valid categorical values, and numeric ranges before prediction.

For example, the API returns HTTP `422 Unprocessable Content` when the request contains an unsupported value such as:

```json
{
  "gender": "Unknown",
  "SeniorCitizen": 3
}
```

This prevents invalid customer data from reaching the model.

## API Test Result

The API was successfully tested with `sample_request.json`.

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8501
}
```