# Classification Model Comparison for Early Stage Diabetes Risk Prediction

**M.Tech-AIML - Machine Learning Assignment 2**

**Author:** MOHAMMAD HAMZA

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
| Logistic Regression      | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree            | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| kNN                      | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes              | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561   | 0.9939 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| XGBoost (Ensemble)       | 0.9561   | 0.9917 | 0.9467    | 0.9861 | 0.9660 | 0.9058 |

---

## Model Performance Observations

| ML Model Name                      | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**      | Achieves the highest accuracy (98.25%) among all models with excellent AUC score (0.9954), demonstrating outstanding discrimination capability. Perfect balance between precision and recall (0.9861) makes it highly reliable for clinical applications. The high MCC (0.9623) confirms strong performance even accounting for class imbalance. Linear decision boundary works exceptionally well for this dataset, likely due to well-separated feature space. |
| **Decision Tree**            | Shows the lowest performance with 91.23% accuracy, indicating tendency to overfit despite max_depth constraint. Lower recall (0.9028) compared to other models means it misses more malignant cases, which is problematic for medical diagnosis. However, high precision (0.9559) shows that when it predicts malignant, it's usually correct. The model's interpretability is valuable but comes at the cost of prediction accuracy.                            |
| **kNN**                      | Demonstrates strong performance (95.61% accuracy) by leveraging local similarity between samples. Excellent recall (0.9722) means it successfully identifies most malignant cases. High AUC (0.9788) indicates good ranking ability. The model benefits from scaled features and appropriate choice of k=5. Performs well because similar tumor characteristics typically indicate similar diagnoses.                                                            |
| **Naive Bayes**              | Achieves respectable 92.98% accuracy despite strong independence assumptions between features. Remarkably high AUC (0.9868) shows excellent probabilistic calibration for ranking predictions. Balanced precision and recall (both 0.9444) indicate consistent performance across both classes. Fast training and prediction make it suitable for rapid screening applications.                                                                                  |
| **Random Forest (Ensemble)** | Delivers strong 95.61% accuracy through ensemble of decision trees, significantly outperforming single decision tree. Highest AUC (0.9939) among all models demonstrates exceptional discriminative ability. Excellent recall (0.9722) minimizes false negatives, critical for Diabetes risk detection. The ensemble approach effectively reduces overfitting while maintaining interpretability through feature importance.                                            |
| **XGBoost (Ensemble)**       | Achieves 95.61% accuracy with best recall (0.9861) among top performers, meaning it catches almost all malignant cases. Very high AUC (0.9917) confirms strong ranking capability. Gradient boosting with regularization prevents overfitting effectively. Slightly lower precision (0.9467) compared to Logistic Regression, but superior recall makes it safer for medical screening where missing Diabetes risk is more costly than false alarms.                    |

---

## Key Insights

### Best Performing Models:

1. **Logistic Regression** - Highest accuracy (98.25%) and excellent all-around metrics
2. **Random Forest** & **XGBoost** & **kNN** (tied at 95.61% accuracy)

### Model Selection Recommendations:

- **For Production Deployment:** Logistic Regression - Best accuracy, interpretability, and balanced metrics
- **For Maximum Recall (Safety):** XGBoost - Catches 98.61% of malignant cases
- **For Probabilistic Ranking:** Random Forest - Highest AUC (0.9939)
- **For Speed:** Naive Bayes - Fastest training and prediction time
- **For Interpretability:** Logistic Regression - Linear coefficients are easy to explain to medical professionals

### Clinical Context:

In medical diagnosis, **recall** (sensitivity) is often more important than precision to minimize false negatives (missing biabetics cases). XGBoost shows the highest recall (98.61%), closely followed by Logistic Regression (98.61%), making them both excellent choices for medical screening applications where catching all positive cases is critical.

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

---

## Technologies Used

- **Python 3.x**
- **scikit-learn** - Machine learning models and metrics
- **XGBoost** - Gradient boosting framework
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization

---

## Evaluation Metrics Explained

- **Accuracy:** Percentage of correct predictions
- **AUC (Area Under ROC Curve):** Model's ability to distinguish between classes
- **Precision:** Proportion of positive predictions that are actually correct
- **Recall (Sensitivity):** Proportion of actual positives correctly identified
- **F1 Score:** Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient):** Balanced measure for imbalanced datasets (-1 to +1)

---

## Author

**MOHAMMAD HAMZA**

---

## License

This project is created for educational purposes as part of Machine Learning Assignment.

---

## Acknowledgments

- Dataset: UCI Machine Learning Repository
- Framework: Streamlit Community
- Libraries: scikit-learn, XGBoost, and the Python data science ecosystem
