# Customer Segmentation and Credit Risk Modelling

This project implements two key financial analytics tasks: **Customer Segmentation** and **Credit Risk Modelling**. The goal is to demonstrate "interview-ready" code and methodology, moving from raw data to actionable insights.

## Project Structure

```
.
├── credit_risk_modelling.py  # Script for Credit Risk Model
├── customer_segmentation.py  # Script for Customer Segmentation
├── download_data.py          # Script to download data from Kaggle
├── setup_kaggle.py           # Script to setup Kaggle API
├── requirements.txt          # Python dependencies
├── images/                   # Generated plots and visualizations
│   ├── customer_segments_pca.png
│   ├── elbow_method.png
│   ├── roc_curve.png
│   └── confusion_matrix.png
└── README.md                 # Project documentation
```

## 1. Customer Segmentation

**Goal**: Group credit card holders into distinct segments to tailor marketing strategies.

### Steps & Justifications

1.  **Data Loading**:
    *   **Source**: `CC GENERAL.csv` (Kaggle).
    *   **Reason**: Real-world dataset containing usage behaviors like Balance, Purchases, Cash Advance, etc.

2.  **Data Preprocessing**:
    *   **Missing Values**: Imputed with **median**.
    *   **Reason**: Median is robust to outliers, which are common in financial data (e.g., a few very high credit limits).
    *   **Scaling**: Used `StandardScaler`.
    *   **Reason**: K-Means is distance-based (Euclidean). Without scaling, features with large magnitudes (like `BALANCE`) would dominate features with small magnitudes (like `PURCHASES_FREQUENCY`), leading to biased clusters.

3.  **Dimensionality Reduction (PCA)**:
    *   **Method**: Principal Component Analysis (PCA).
    *   **Reason**:
        *   **Noise Reduction**: The dataset has many correlated features. PCA combines them into uncorrelated components.
        *   **Visualization**: Reducing to 2 components allows us to visualize the clusters in a 2D plane.
    *   **Result**: 2 components explain ~47% of the variance.

4.  **Clustering (K-Means)**:
    *   **Method**: K-Means Clustering.
    *   **Reason**: Efficient and easy to interpret for segmentation tasks.
    *   **Optimal K**: Determined using the **Elbow Method**.
    *   **Reason**: The Elbow plot shows where adding more clusters gives diminishing returns in reducing variance (WCSS). We selected **k=4**.

## 2. Credit Risk Modelling

**Goal**: Predict the probability of a borrower defaulting on a loan and determine an optimal approval threshold.

### Steps & Justifications

1.  **Data Loading**:
    *   **Source**: `credit_risk_dataset.csv` (Kaggle).
    *   **Reason**: Contains loan details (amount, interest rate) and borrower details (income, employment length).

2.  **Data Preprocessing**:
    *   **Imputation**: Filled missing values with median.
    *   **Encoding**: One-Hot Encoding for categorical variables (e.g., `loan_grade`).
    *   **Reason**: Machine learning models require numerical input.

3.  **Feature Engineering (WOE & IV)**:
    *   **Concept**: Weight of Evidence (WOE) and Information Value (IV).
    *   **Reason**:
        *   **WOE**: Handles outliers and non-linear relationships by binning features. It transforms categorical variables into a continuous "risk" measure.
        *   **IV**: Measures the predictive power of a feature. It helps in selecting the most important variables (IV > 0.02).
    *   *Note*: In this script, we demonstrated WOE/IV calculation on `loan_grade` but used One-Hot Encoding for the full model for simplicity.

4.  **Modeling (Logistic Regression)**:
    *   **Model**: Logistic Regression.
    *   **Reason**:
        *   **Interpretability**: Coefficients directly relate to the odds of default.
        *   **Industry Standard**: Widely used in credit scoring due to regulatory requirements for explainability.
        *   **Probabilistic Output**: Provides a probability of default (PD), which is essential for risk grading.

5.  **Evaluation (ROC & Threshold)**:
    *   **Metric**: ROC Curve and AUC Score.
    *   **Reason**: Accuracy is misleading for imbalanced datasets (few defaulters). AUC measures the model's ability to rank bad borrowers higher than good ones.
    *   **Optimal Threshold**: Calculated using **Youden's J statistic**.
    *   **Reason**: The default threshold of 0.5 is often not optimal. We found a threshold of **0.35** maximizes the balance between catching defaulters (Recall) and avoiding false alarms.

## How to Run

1.  **Setup Environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Download Data**:
    Ensure you have your Kaggle API key setup (or run `setup_kaggle.py`).
    ```bash
    python3 download_data.py
    ```

3.  **Run Analysis**:
    ```bash
    python3 customer_segmentation.py
    python3 credit_risk_modelling.py
    ```
    *   Check the console output for analysis results.
    *   Check the `images/` folder for generated plots.
