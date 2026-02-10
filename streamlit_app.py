import os
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Classifier Dashboard",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Early Stage Diabetes Risk – ML Dashboard")
st.caption("Interactive evaluation of multiple ML models on diabetes risk data")
st.markdown("---")

# -----------------------------------------------------------------------------
# LOAD ARTIFACTS (SCALER, METRICS, REPORTS)
# -----------------------------------------------------------------------------
MODEL_DIR = "model"

try:
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "r") as f:
        training_metrics = json.load(f)

    with open(os.path.join(MODEL_DIR, "confusion_matrices.json"), "r") as f:
        training_cm = json.load(f)

    with open(os.path.join(MODEL_DIR, "classification_reports.json"), "r") as f:
        training_cr = json.load(f)

except Exception as e:
    st.error(f"Could not load required model files. Details: {e}")
    st.stop()

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "k_nearest_neighbors.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.header("App Menu")
section = st.sidebar.radio(
    "Go to",
    (
        "Dataset & Setup",
        "Test Split Evaluation",
        "Upload & Evaluate CSV",
    ),
    index=1,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Metrics in Use**")
st.sidebar.write("Accuracy, AUC, Precision, Recall, F1, MCC")
st.sidebar.markdown("**Models in Use**")
st.sidebar.write("Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost")
st.sidebar.markdown("---")
st.sidebar.info("Hint: Start with *Test Split Evaluation* to inspect baseline performance.")

# -----------------------------------------------------------------------------
# HELPER: LOAD TEST DATA
# -----------------------------------------------------------------------------
def load_test_data(path: str = "data/test_data.csv"):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


# -----------------------------------------------------------------------------
# SECTION 1: DATASET & SETUP
# -----------------------------------------------------------------------------
if section == "Dataset & Setup":
    st.header("Dataset Overview & Experiment Design")
    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Dataset Summary")
        st.markdown(
            """
            - **Name:** Early Stage Diabetes Risk Prediction  
            - **Source:** UCI Machine Learning Repository  
            - **Instances:** 520  
            - **Attributes:** 16 (15 predictors + 1 target)  
            """
        )

    with col_right:
        st.subheader("Problem Setup")
        st.markdown(
            """
            - **Setting:** Supervised binary classification  
            - **Target:** Diabetes risk (0 = No, 1 = Yes)  
            - **Split Strategy:** 80% training / 20% testing (104 test samples)  
            """
        )

    st.markdown("---")
    st.subheader("Models Implemented")
    left_models, right_models = st.columns(2)
    model_labels = [
        "1. Logistic Regression",
        "2. Decision Tree",
        "3. K-Nearest Neighbors (KNN)",
        "4. Naive Bayes (Gaussian)",
        "5. Random Forest (Ensemble)",
        "6. XGBoost (Ensemble)",
    ]
    with left_models:
        for label in model_labels[:3]:
            st.write(label)
    with right_models:
        for label in model_labels[3:]:
            st.write(label)

    st.markdown("---")
    st.subheader("Evaluation Criteria")
    metric_labels = [
        "Accuracy",
        "Area Under ROC Curve (AUC)",
        "Precision",
        "Recall",
        "F1 Score",
        "Matthews Correlation Coefficient (MCC)",
    ]
    left_metrics, right_metrics = st.columns(2)
    with left_metrics:
        for m in metric_labels[:3]:
            st.write(f"- {m}")
    with right_metrics:
        for m in metric_labels[3:]:
            st.write(f"- {m}")

    st.markdown("---")
    st.subheader("Input Features")
    st.markdown("The dataset uses symptom-based attributes plus age and obesity indicators.")

    with st.expander("Show feature details"):
        feature_details = {
            "Feature": [
                "Age",
                "Sex",
                "Polyuria",
                "Polydipsia",
                "sudden weight loss",
                "weakness",
                "Polyphagia",
                "Genital thrush",
                "visual blurring",
                "Itching",
                "Irritability",
                "delayed healing",
                "partial paresis",
                "muscle stifness",
                "Alopecia",
                "Obesity",
                "Class",
            ],
            "Description": [
                "Age between 20 and 65",
                "Sex: 1 = Male, 2 = Female",
                "Polyuria: 1 = Yes, 2 = No",
                "Polydipsia: 1 = Yes, 2 = No",
                "Sudden weight loss: 1 = Yes, 2 = No",
                "Weakness: 1 = Yes, 2 = No",
                "Polyphagia: 1 = Yes, 2 = No",
                "Genital thrush: 1 = Yes, 2 = No",
                "Visual blurring: 1 = Yes, 2 = No",
                "Itching: 1 = Yes, 2 = No",
                "Irritability: 1 = Yes, 2 = No",
                "Delayed healing: 1 = Yes, 2 = No",
                "Partial paresis: 1 = Yes, 2 = No",
                "Muscle stiffness: 1 = Yes, 2 = No",
                "Alopecia: 1 = Yes, 2 = No",
                "Obesity: 1 = Yes, 2 = No",
                "Class: 1 = Positive, 2 = Negative",
            ],
        }
        feature_df = pd.DataFrame(feature_details)
        st.dataframe(feature_df, use_container_width=True)

    st.success("Move to *Test Split Evaluation* to inspect baseline model performance on the 20% hold-out set.")

# -----------------------------------------------------------------------------
# SECTION 2: TEST SPLIT EVALUATION (DEFAULT)
# -----------------------------------------------------------------------------
elif section == "Test Split Evaluation":
    st.header("Evaluation on Hold-out Test Set")
    st.info("All metrics in this section are computed using the fixed 20% test split (104 samples).")
    st.markdown("---")

    # Overall comparison table
    st.subheader("Global Model Comparison")
    st.markdown("Test-set performance of all models using pre-computed metrics.")

    comparison_df = pd.DataFrame(training_metrics).T.reset_index()
    comparison_df.columns = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

    for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.4f}")

    st.dataframe(comparison_df, use_container_width=True)
    st.markdown("---")

    # Qualitative observations
    st.subheader("Qualitative Remarks")
    obs = {
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Naive Bayes",
            "Random Forest",
            "XGBoost",
        ],
        "Comment": [
            "Linear baseline with decent trade-off between sensitivity and specificity, also useful for interpreting feature effects.",
            "Can capture non-linear patterns but prone to overfitting if not regularized or pruned properly.",
            "Performance depends strongly on scaling and neighborhood size; can degrade with noisy or high-dimensional data.",
            "Fast and lightweight despite the feature-independence assumption; suitable when quick decisions are required.",
            "Ensemble of trees that gives robust performance and good feature importance ranking.",
            "Typically yields the strongest scores by modelling complex interactions, at the cost of more tuning and computation.",
        ],
    }
    obs_df = pd.DataFrame(obs)
    st.dataframe(obs_df, use_container_width=True)
    st.markdown("---")

    # Detailed view for one model
    st.subheader("Drill-down: Detailed View for Selected Model")
    st.markdown("Pick a model to view metrics, confusion matrix, classification report, and predictions.")

    selected_model = st.selectbox(
        "Model for detailed analysis",
        list(MODEL_FILES.keys()),
        index=0,
        key="test_data_model_select",
    )
    st.markdown("---")

    test_df = load_test_data()
    if test_df is None:
        st.error("Could not locate `data/test_data.csv`. Please ensure it exists.")
        st.stop()

    X_test = test_df[feature_names]
    y_test = test_df["target"]

    model_path = os.path.join(MODEL_DIR, MODEL_FILES[selected_model])
    if not os.path.exists(model_path):
        st.error(f"Model artifact not found at: {model_path}")
        st.stop()

    model = joblib.load(model_path)
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Metrics (from training_metrics for consistency with given JSON)
    st.subheader(f"Metric Summary — {selected_model}")
    metrics = training_metrics.get(selected_model, {})
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Accuracy", f"{metrics.get('Accuracy', 0):.4f}")
        st.metric("AUC", f"{metrics.get('AUC', 0):.4f}")
    with c2:
        st.metric("Precision", f"{metrics.get('Precision', 0):.4f}")
        st.metric("Recall", f"{metrics.get('Recall', 0):.4f}")
    with c3:
        st.metric("F1", f"{metrics.get('F1', 0):.4f}")
        st.metric("MCC", f"{metrics.get('MCC', 0):.4f}")

    st.markdown("---")

    # Confusion matrix (pre-computed)
    st.subheader(f"Confusion Matrix — {selected_model}")
    cm_arr = np.array(training_cm[selected_model])

    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Diabetes Risk", "Diabetes Risk"],
        yticklabels=["No Diabetes Risk", "Diabetes Risk"],
        ax=ax_cm,
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title(f"Confusion Matrix - {selected_model}")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    st.markdown("---")

    # Classification report (pre-computed)
    st.subheader(f"Classification Report — {selected_model}")
    cr_dict = training_cr[selected_model]
    cr_df = pd.DataFrame(cr_dict).transpose()

    for col in cr_df.columns:
        if cr_df[col].dtype in ("float64", "int64"):
            cr_df[col] = cr_df[col].apply(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.floating)) else x
            )

    st.dataframe(cr_df, use_container_width=True)
    st.markdown("---")

    # Predictions table
    st.subheader(f"Predictions on Test Split — {selected_model}")
    out_df = X_test.copy()
    out_df["Actual"] = y_test.values
    out_df["Predicted"] = y_pred
    out_df["Confidence (%)"] = (y_prob * 100).round(2)
    out_df["Correct"] = np.where(y_pred == y_test, "✅ Yes", "❌ No")

    total = len(out_df)
    correct = int((y_pred == y_test).sum())
    incorrect = total - correct
    st.markdown(
        f"Total samples: **{total}**, Correct: **{correct}**, Incorrect: **{incorrect}**"
    )

    with st.expander("Preview first 10 predictions"):
        st.dataframe(out_df.head(10), use_container_width=True)

    st.download_button(
        label="Download full prediction table (CSV)",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name=f"test_predictions_{selected_model.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# SECTION 3: UPLOAD & EVALUATE CSV
# -----------------------------------------------------------------------------
else:
    st.header("Evaluate a Custom CSV File")
    st.markdown(
        "Upload a CSV with the same structure as the original dataset "
        "(all feature columns plus a binary target)."
    )
    st.markdown("---")

    test_df = load_test_data()
    if test_df is not None:
        st.download_button(
            label="Download reference test data (104 rows)",
            data=test_df.to_csv(index=False).encode("utf-8"),
            file_name="test_data_reference.csv",
            mime="text/csv",
            help="Use this file as a template for your own dataset.",
        )
    else:
        st.warning("Reference `test_data.csv` not found under `data/`.")

    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload CSV for evaluation",
        type=["csv"],
    )

    if uploaded is None:
        st.info("Upload a CSV file to start the validation and evaluation workflow.")
    else:
        try:
            df = pd.read_csv(uploaded)

            # ---------------------- BASIC VALIDATIONS ---------------------- #
            if df.empty:
                st.error("The uploaded file contains no rows.")
                st.stop()

            st.success("File loaded successfully.")

            required_cols = set(feature_names + ["target"])
            cols_present = set(df.columns)

            missing_cols = required_cols - cols_present
            extra_cols = cols_present - required_cols

            if missing_cols:
                st.error(
                    "Your CSV is missing the following required columns:\n"
                    + ", ".join(sorted(missing_cols))
                )
                st.info("Use the reference test data above as a template.")
                st.stop()

            if extra_cols:
                st.warning(
                    "Extra columns found (they will be ignored): "
                    + ", ".join(sorted(extra_cols))
                )

            X = df[feature_names]
            y = df["target"]

            # Numeric type check
            try:
                X = X.astype(float)
            except ValueError:
                st.error("Some feature columns contain non-numeric values.")
                bad_cols = []
                for col in feature_names:
                    try:
                        df[col].astype(float)
                    except Exception:
                        bad_cols.append(col)
                if bad_cols:
                    st.error(
                        "Columns with invalid values: " + ", ".join(sorted(bad_cols))
                    )
                st.stop()

            # Missing values in features
            if X.isnull().any().any():
                st.error("Missing values detected in feature columns.")
                missing_features = X.columns[X.isnull().any()].tolist()
                for c in missing_features:
                    st.error(f"- {c}: {X[c].isnull().sum()} missing entries")
                st.info("Handle missing values (e.g., imputation or row removal) before upload.")
                st.stop()

            # Missing values in target
            if y.isnull().any():
                st.error(f"Target column has {int(y.isnull().sum())} missing values.")
                st.stop()

            # Target validation
            uniques = sorted(y.unique())
            if not set(uniques).issubset({0, 1}):
                st.error(f"Target column must be binary (0/1). Found: {uniques}")
                st.stop()

            st.info(f"Dataset ready for evaluation: {len(df)} rows, {len(feature_names)} features.")

            with st.expander("Preview uploaded data (first 10 rows)"):
                st.dataframe(df.head(10), use_container_width=True)

            st.markdown("---")

            # ---------------------- MODEL SELECTION ---------------------- #
            st.subheader("Select Model and Run Evaluation")

            model_choice = st.selectbox(
                "Choose a model",
                list(MODEL_FILES.keys()),
                index=0,
                key="uploaded_model_select",
            )

            model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_choice])
            if not os.path.exists(model_path):
                st.error(f"Model file not found at: {model_path}")
                st.stop()

            model = joblib.load(model_path)

            # ---------------------- EVALUATION ---------------------- #
            X_scaled = scaler.transform(X)
            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]

            acc = accuracy_score(y, preds)
            auc = roc_auc_score(y, probs)
            prec = precision_score(y, preds, zero_division=0)
            rec = recall_score(y, preds, zero_division=0)
            f1 = f1_score(y, preds, zero_division=0)
            mcc = matthews_corrcoef(y, preds)

            st.subheader(f"Evaluation Metrics — {model_choice}")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Accuracy", f"{acc:.4f}")
                st.metric("AUC", f"{auc:.4f}")
            with m2:
                st.metric("Precision", f"{prec:.4f}")
                st.metric("Recall", f"{rec:.4f}")
            with m3:
                st.metric("F1 Score", f"{f1:.4f}")
                st.metric("MCC", f"{mcc:.4f}")

            st.markdown("---")

            # Confusion matrix
            st.subheader(f"Confusion Matrix — {model_choice}")
            cm = confusion_matrix(y, preds)
            fig_u, ax_u = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["No Diabetes", "Diabetes"],
                yticklabels=["No Diabetes", "Diabetes"],
                ax=ax_u,
            )
            ax_u.set_xlabel("Predicted")
            ax_u.set_ylabel("Actual")
            ax_u.set_title(f"Confusion Matrix - {model_choice}")
            st.pyplot(fig_u)
            plt.close(fig_u)

            st.markdown("---")

            # Classification report
            st.subheader(f"Classification Report — {model_choice}")
            cr_live = classification_report(
                y, preds, output_dict=True, zero_division=0
            )
            cr_live_df = pd.DataFrame(cr_live).transpose()
            for col in cr_live_df.columns:
                if cr_live_df[col].dtype in ("float64", "int64"):
                    cr_live_df[col] = cr_live_df[col].apply(
                        lambda x: f"{x:.4f}"
                        if isinstance(x, (int, float, np.floating))
                        else x
                    )
            st.dataframe(cr_live_df, use_container_width=True)

            st.markdown("---")

            # Prediction table
            st.subheader(f"Prediction Table — {model_choice}")
            pred_df = X.copy()
            pred_df["Actual"] = y.values
            pred_df["Predicted"] = preds
            pred_df["Confidence (%)"] = (probs * 100).round(2)
            pred_df["Correct"] = np.where(preds == y, "✅ Yes", "❌ No")

            with st.expander("Preview first 10 predictions"):
                st.dataframe(pred_df.head(10), use_container_width=True)

            c_download_1, c_download_2 = st.columns(2)
            with c_download_1:
                st.download_button(
                    label="Download predictions (CSV)",
                    data=pred_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"uploaded_predictions_{model_choice.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                )

            with c_download_2:
                correct = int((preds == y).sum())
                total = len(preds)
                incorrect = total - correct
                acc_pct = (correct / total) * 100

                summary_text = (
                    f"Model: {model_choice}\n"
                    f"Total predictions: {total}\n"
                    f"Correct: {correct} ({acc_pct:.2f}%)\n"
                    f"Incorrect: {incorrect} ({100 - acc_pct:.2f}%)\n\n"
                    f"Accuracy: {acc:.4f}\n"
                    f"AUC: {auc:.4f}\n"
                    f"Precision: {prec:.4f}\n"
                    f"Recall: {rec:.4f}\n"
                    f"F1: {f1:.4f}\n"
                    f"MCC: {mcc:.4f}\n"
                )

                st.download_button(
                    label="Download textual summary (TXT)",
                    data=summary_text.encode("utf-8"),
                    file_name=f"uploaded_summary_{model_choice.replace(' ', '_').lower()}.txt",
                    mime="text/plain",
                )

        except Exception as exc:
            st.error("An unexpected error occurred while handling your file.")
            st.error(f"Details: {exc}")
            with st.expander("Show full traceback (debug only)"):
                import traceback

                st.code(traceback.format_exc())

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "ML Assignment – Diabetes Risk Prediction | Built with Streamlit"
)
