# Classification Model Comparison for Early Stage Diabetes Risk Prediction

**M.Tech-AIML - Machine Learning Assignment 2**

**Author:** MOHAMMAD HAMZA (2025AA05070)

**Date:** February 2026

---

## Problem Statement

The objective of this assignment is to design, implement, and compare **six classification models** on a real-world medical dataset related to early-stage diabetes risk. The models are assessed using multiple performance metrics to gain insight into their predictive behavior, interpretability, and robustness across different evaluation criteria. In addition, an interactive Streamlit web application is built to showcase the trained models and is deployed on Streamlit Community Cloud to enable convenient access and experimentation for end users.

The classification task focuses on predicting whether a patient is at **risk of early-stage diabetes or not**, using a set of clinical signs and symptoms collected through questionnaires at a diabetes hospital. These features include demographic information such as age and gender, along with multiple binary indicators like polyuria, polydipsia, sudden weight loss, weakness, visual blurring, and other diabetes-related conditions.

---

## Dataset Description

**Dataset:** Early Stage Diabetes Risk Prediction Dataset
**Source:** UCI Machine Learning Repository
**Problem Type:** Binary Classification

# Dataset Characteristics:

**Number of Instances:** 520 (patients)

**Number of Features:** 16 attributes (mix of numerical and categorical/symptom-based)

**Target Variable:**

1 = Positive for early-stage diabetes risk

0 = Negative (no early-stage diabetes risk)

- **Class Distribution:**
  - Positive: 320 instances (61.5%)
  - Negative: 200 instances (38.5%)

### Features:

The dataset includes 16 features gathered from patient questionnaires and clinical assessments at a diabetes hospital, capturing symptoms, demographics, and risk factors for early-stage diabetes prediction.

**Feature Groups:**

Demographics - Age (numeric range) and Gender (Male/Female)

Urinary Symptoms - Polyuria (excessive urination)

Thirst Indicators - Polydipsia (excessive thirst)

Weight Changes - Sudden Weight Loss

General Fatigue - Weakness

Hunger Patterns - Polyphagia (excessive hunger)

Infection Signs - Genital Thrush

Vision Issues - Visual Blurring

Skin Conditions - Itching

Mood Effects - Irritability

Healing Problems - Delayed Healing

Nerve-Related - Partial Paresis (muscle weakness)

Mobility Issues - Muscle Stiffness

Hair Loss - Alopecia

Obesity Factor - Obesity (BMI-related)

Class Label - Positive/Negative diabetes risk


**Missing Values:** None

**Data Split:** 80% Training (416 samples) / 20% Testing (104 samples)

---

## Models Used

### Model Comparison Table

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression      | 0.9038	  | 0.9629 | 0.9296	  | 0.9296 | 0.9296 | 0.7781 |
| Decision Tree            | 0.9135	  | 0.9706 | 0.9697	  | 0.9014 | 0.9343 | 0.8127 |
| kNN                      | 0.9038	  | 0.9742 | 0.9552	  | 0.9014 | 0.9275 | 0.7880 |
| Naive Bayes              | 0.9135	  | 0.9620 | 0.9306	  | 0.9437 | 0.9371 | 0.7988 |
| Random Forest (Ensemble) | 0.9327	  | 0.9825 | 0.9444	  | 0.9577 | 0.9510 | 0.8436 |
| XGBoost (Ensemble)       | 0.9808	  | 0.9970 | 1.0000	  | 0.9718 | 0.9857 | 0.9572 |

---

## Model Performance Observations

| ML Model Name                      | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression** | Delivers solid performance with 90.38% accuracy and strong AUC (0.9629), indicating good class separation. Precision and recall are perfectly aligned (0.9296), resulting in a balanced F1 score. The MCC of 0.7781 reflects dependable predictive strength despite potential class imbalance. Its linear structure performs well given the dataset’s feature distribution. |
| **Decision Tree** | Achieves 91.35% accuracy with improved AUC (0.9706) compared to Logistic Regression. Very high precision (0.9697) suggests confident positive predictions, though slightly lower recall (0.9014) indicates some missed positive cases. MCC (0.8127) shows better overall correlation than simpler linear models. While interpretable, it may still capture noise from the training data. |
| **kNN** | Matches Logistic Regression in accuracy (90.38%) but records a higher AUC (0.9742), showing strong ranking capability. Precision (0.9552) is high, while recall (0.9014) is moderate, leading to a well-balanced F1 score. MCC (0.7880) confirms competitive performance. The model benefits from distance-based learning where similar samples share similar outcomes. |
| **Naive Bayes** | Reaches 91.35% accuracy with balanced precision (0.9306) and recall (0.9437), producing a stable F1 score. Despite its independence assumption, it performs competitively with an AUC of 0.9620. MCC (0.7988) indicates consistent predictive reliability. Its computational efficiency makes it suitable for fast inference scenarios. |
| **Random Forest** | Provides strong results with 93.27% accuracy and high AUC (0.9825), demonstrating excellent discrimination ability. High recall (0.9577) reduces false negatives, while precision (0.9444) remains robust. MCC (0.8436) reflects strong overall classification quality. The ensemble approach improves stability and mitigates overfitting compared to a single tree. |
| **XGBoost** | Delivers the best overall performance with 98.08% accuracy and an outstanding AUC of 0.9970. Perfect precision (1.0000) combined with high recall (0.9718) yields the strongest F1 score. MCC (0.9572) confirms superior predictive power across classes. Gradient boosting with regularization enables exceptional generalization on this dataset. |             |

---

## Key Insights

### Best Performing Models:

1. **XGBoost (Ensemble)** - Achieved the highest accuracy (98.08%) with outstanding overall evaluation scores across all performance metrics.
2. **Random Forest (Ensemble)** - Delivered strong results with 93.27% accuracy and consistently high precision–recall balance.

### Model Selection Recommendations:

- **For Production Deployment:** XGBoost (Ensemble) - Highest accuracy (98.08%) with superior overall performance.
- **For Maximum Recall (Safety Focus):** Random Forest (Ensemble) - Highest recall (0.9577), minimizing false negatives.
- **For Best Ranking Capability:** XGBoost (Ensemble) - Outstanding AUC (0.9970), indicating exceptional class separation.
- **For Computational Efficiency:** Naive Bayes - Lightweight model with fast training and inference.
- **For Interpretability:** Logistic Regression - Linear model with transparent and easily explainable coefficients.

### Clinical Context:

In medical diagnosis, **recall** (sensitivity) is often more important than precision to minimize false negatives (missing diabetics cases). XGBoost shows the highest recall (97.18%), closely followed by Random Forest (95.77%), making them both excellent choices for medical condition detection applications where catching all positive cases is critical.

---

## Project Structure

```
ml-classification-model-comparison/
│
├── streamlit_app.py                    # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── train_models.ipynb                  # All Model training Code  
└── model/
    ├── logistic_regression.pkl         # Model outcome files
    ├── decision_tree.pkl
    ├── k_nearest_neighbors.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── scaler.pkl                      # Feature scaler
    ├── feature_names.pkl               # All the features
    ├── confusion_matrices.json         # Confusion Matrics for all the models
    ├── metrics.json
    ├── models_comparison.csv           # More detals of the results
 └── data/
    ├── test_data.csv                   # Test data for comparision table
    ├── test_samples.csv                # Test sample for upload option
```

---

## Installation and Setup

### Prerequisites:

- Python 3.8 or higher
- pip package manager

### Steps to Run Locally:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Hamza-BiTS/ml-classification-model-comparison
   cd ml-classification-model
   ```
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
3. **Train the models:**

   ```bash
   python train_models.ipynb
   ```
4. **Run the Streamlit app:**

   ```bash
   streamlit run streamlit_app.py
   ```
5. **Access the app:**
   Open your browser and navigate to `http://localhost:8501`

---

## Streamlit App Features

The interactive web application includes:

✅ **Dataset Upload Option** - Upload your own CSV test data
✅ **Model Selection Dropdown** - Choose from 6 different models
✅ **Evaluation Metrics Display** - View Accuracy, AUC, Precision, Recall, F1, and MCC
✅ **Confusion Matrix Visualization** - Interactive heatmap of predictions
✅ **Classification Report** - Detailed per-class performance metrics
✅ **Model Comparison Dashboard** - Compare all models side-by-side with visual charts

---

## Deployment on Streamlit Community Cloud

### Steps to Deploy:

1. Push your directory to GitHub repository
2. Go to https://streamlit.io/cloud
3. Sign in with your GitHub account
4. Click **"New App"**
5. Select your repository
6. Choose branch: `main`
7. Select main file: `ml-classification-model/streamlit_app.py`
8. Update App URL: `diabetes-risk-prediction-2026`
9. Click **"Deploy"**

The app will be live within a few minutes at: `https://diabetes-risk-prediction-2026.streamlit.app/`


## Evaluation Metrics Explained

- **Accuracy:** Percentage of correct predictions
- **AUC (Area Under ROC Curve):** Model's ability to distinguish between classes
- **Precision:** Proportion of positive predictions that are actually correct
- **Recall (Sensitivity):** Proportion of actual positives correctly identified
- **F1 Score:** Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient):** Balanced measure for imbalanced datasets (-1 to +1)

---

## Author

**MOHAMMAD HAMZA - 2025AA05070**
