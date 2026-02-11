# ML Assignment 2 - Classification Model Comparison

**M.Tech (AIML/DSE) - Machine Learning Assignment 2**
**Author:** Karthik S Kashyap, M.Tech AIML
**Date:** January 2026

---

## Problem Statement

The objective of this assignment is to implement and compare **six different classification models** on a real-world dataset. The models are evaluated using multiple performance metrics to understand their strengths and weaknesses. Additionally, an interactive **Streamlit web application** is developed to demonstrate the models and deployed on **Streamlit Community Cloud** for easy accessibility.

The classification task involves predicting whether a breast tumor is **malignant** or **benign** based on various diagnostic features extracted from digitized images of fine needle aspirates (FNA) of breast masses.

---

## Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset

**Source:** UCI Machine Learning Repository / sklearn.datasets

**Problem Type:** Binary Classification

### Dataset Characteristics:

- **Number of Instances:** 569
- **Number of Features:** 30 (all numerical)
- **Target Variable:**
  - 0 = Malignant (Cancer)
  - 1 = Benign (Non-cancerous)
- **Class Distribution:**
  - Malignant: 212 instances (37.3%)
  - Benign: 357 instances (62.7%)

### Features:

The dataset contains 30 features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

**Feature Groups:**

1. **Radius** - Mean of distances from center to points on the perimeter
2. **Texture** - Standard deviation of gray-scale values
3. **Perimeter** - Tumor perimeter
4. **Area** - Tumor area
5. **Smoothness** - Local variation in radius lengths
6. **Compactness** - (Perimeter² / Area) - 1.0
7. **Concavity** - Severity of concave portions of the contour
8. **Concave Points** - Number of concave portions of the contour
9. **Symmetry** - Symmetry of the tumor
10. **Fractal Dimension** - "Coastline approximation" - 1

For each feature, three values are computed:

- **Mean**
- **Standard Error**
- **Worst** (mean of the three largest values)

This results in 30 features total (10 characteristics × 3 statistical measures).

**Missing Values:** None

**Data Split:** 80% Training (455 samples) / 20% Testing (114 samples)

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
| **Random Forest (Ensemble)** | Delivers strong 95.61% accuracy through ensemble of decision trees, significantly outperforming single decision tree. Highest AUC (0.9939) among all models demonstrates exceptional discriminative ability. Excellent recall (0.9722) minimizes false negatives, critical for cancer detection. The ensemble approach effectively reduces overfitting while maintaining interpretability through feature importance.                                            |
| **XGBoost (Ensemble)**       | Achieves 95.61% accuracy with best recall (0.9861) among top performers, meaning it catches almost all malignant cases. Very high AUC (0.9917) confirms strong ranking capability. Gradient boosting with regularization prevents overfitting effectively. Slightly lower precision (0.9467) compared to Logistic Regression, but superior recall makes it safer for medical screening where missing cancer is more costly than false alarms.                    |

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

In medical diagnosis, **recall** (sensitivity) is often more important than precision to minimize false negatives (missing cancer cases). XGBoost shows the highest recall (98.61%), closely followed by Logistic Regression (98.61%), making them both excellent choices for medical screening applications where catching all positive cases is critical.

---

## Project Structure

```
BreastCancer/
│
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
│
└── model/
    ├── train_models.py                # Model training script
    ├── logistic_regression_model.py   # Saved Logistic Regression model (Python format)
    ├── decision_tree_model.py         # Saved Decision Tree model (Python format)
    ├── knn_model.py                   # Saved KNN model (Python format)
    ├── naive_bayes_model.py           # Saved Naive Bayes model (Python format)
    ├── random_forest_model.py         # Saved Random Forest model (Python format)
    ├── xgboost_model.py               # Saved XGBoost model loader (Python format)
    ├── xgboost_booster.json           # XGBoost booster configuration (JSON format)
    ├── scaler_model.py                # Feature scaler (Python format)
    ├── test_data.csv                  # Test dataset
    └── model_results.csv              # Comparison results
```

---

## Installation and Setup

### Prerequisites:

- Python 3.8 or higher
- pip package manager

### Steps to Run Locally:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kak4bmh/ML
   cd ML/BreastCancer
   ```
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
3. **Train the models:**

   ```bash
   python model/train_models.py
   ```
4. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
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
7. Select main file: `BreastCancer/app.py`
8. Update App URL: `ml-assignment-2025aa05388-breastcancer`
9. Click **"Deploy"**

The app will be live within a few minutes at: `https://ml-assignment-2025aa05388-breastcancer.streamlit.app`

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

**Karthik S Kashyap
Student M.Tech (AIML)**
Work Integrated Learning Programmes Division
BITS Pilani

---

## License

This project is created for educational purposes as part of Machine Learning Assignment.

---

## Acknowledgments

- Dataset: UCI Machine Learning Repository
- Framework: Streamlit Community
- Libraries: scikit-learn, XGBoost, and the Python data science ecosystem
