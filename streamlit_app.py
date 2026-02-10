import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# =====================================================================================
# STREAMLIT PAGE CONFIG
# =====================================================================================
st.set_page_config(
    page_title="Early Stage Diabetes Risk Prediction",
    page_icon="🩸",
    layout="wide"
)

st.title("🩸 Diabetes Risk Prediction – ML Assignment 2")

st.markdown("---")

# =====================================================================================
# LOAD MODELS & PRECOMPUTED METRICS
# =====================================================================================

MODEL_DIR = "model"

try:
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")

    with open(f"{MODEL_DIR}/metrics.json", "r") as f:
        training_metrics = json.load(f)

    with open(f"{MODEL_DIR}/confusion_matrices.json", "r") as f:
        training_cm = json.load(f)

    with open(f"{MODEL_DIR}/classification_reports.json", "r") as f:
        training_cr = json.load(f)

except Exception as e:
    st.error(f"❌ Unable to load model files: {e}")
    st.stop()

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "k_nearest_neighbors.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

# =====================================================================================
# SIDEBAR NAVIGATION
# =====================================================================================

st.sidebar.title("📍 Navigation")

page = st.sidebar.radio(
    "Choose a section:",
    [
        "📊 Dataset Info & Overview",
        "📈 Evaluation on Test Data (DEFAULT)",
        "📤 Upload & Predict on New Data"
    ],
    index=1
)

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Evaluation Metrics Used")
st.sidebar.write("Accuracy, AUC, Precision, Recall, F1, MCC")

st.sidebar.subheader("📌 Models Implemented")
st.sidebar.write("""
- Logistic Regression  
- Decision Tree  
- KNN  
- Naive Bayes  
- Random Forest  
- XGBoost  
""")

st.sidebar.markdown("---")

st.sidebar.info("💡 Tip: Start with 'Evaluation on Test Data' to view training results.")
# =====================================================================================
# PAGE 1: DATASET INFO & OVERVIEW (SHORT VERSION)
# =====================================================================================

if page == "📊 Dataset Info & Overview":
    
    st.header("📊 Dataset Info & Overview")
    
    st.markdown("---")
    
    # Dataset Basic Information
    st.subheader("📁 Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Dataset Name:** Early Stage Diabetes Risk Prediction
        **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset)  
        **Total Instances:** 520
        **Number of Features:** 16  
        **Target Variable:**  
        - 0 = No Diabetes Risk
        - 1 = Diabetes Risk
        """)
    
    with col2:
        st.markdown("""
        **Train-Test Split:** 80-20  
        - Training: 416 instances  
        - Testing: 104 instances  
        
        **Problem Type:** Binary Classification  
        """)
    
    st.markdown("---")
    
    # Models Implemented
    st.subheader("🤖 Models Implemented")
    
    models_list = [
        "1. Logistic Regression",
        "2. Decision Tree",
        "3. K-Nearest Neighbors (KNN)",
        "4. Naive Bayes (Gaussian)",
        "5. Random Forest (Ensemble)",
        "6. XGBoost (Ensemble)"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        for model in models_list[:3]:
            st.write(model)
    
    with col2:
        for model in models_list[3:]:
            st.write(model)
    
    st.markdown("---")
    
    # Evaluation Metrics
    st.subheader("📊 Evaluation Metrics Used")
    
    metrics_list = [
        "1. Accuracy",
        "2. AUC Score",
        "3. Precision",
        "4. Recall",
        "5. F1 Score",
        "6. Matthews Correlation Coefficient (MCC)"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        for metric in metrics_list[:3]:
            st.write(metric)
    
    with col2:
        for metric in metrics_list[3:]:
            st.write(metric)
    
    st.markdown("---")
    
    # Feature Names
    st.subheader("🔢 Features in Dataset")
    
    st.markdown("""
    **13 Features:** age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    """)
    
    with st.expander("📄 View Feature Details"):
        feature_details = {
            "Feature": [
                "Age","Sex","Polyuria","Polydipsia","sudden weight loss","weakness","Polyphagia","Genital thrush",
                "visual blurring","Itching","Irritability","delayed healing","partial paresis","muscle stifness",
                "Alopecia","Obesity","Class"
            ],
            "Description": [
                "Age 1.20-65",
                "Sex 1. Male, 2.Female",	
                "Polyuria 1.Yes, 2.No.",	
                "Polydipsia 1.Yes, 2.No.",
                "sudden weight loss 1.Yes, 2.No.",
                "weakness 1.Yes, 2.No.",
                "Polyphagia 1.Yes, 2.No.",
                "Genital thrush 1.Yes, 2.No.",	
                "visual blurring 1.Yes, 2.No.",
                "Itching 1.Yes, 2.No.",
                "Irritability 1.Yes, 2.No.",
                "delayed healing 1.Yes, 2.No.",
                "partial paresis 1.Yes, 2.No.",	
                "muscle stifness 1.Yes, 2.No.",
                "Alopecia 1.Yes, 2.No.",
                "Obesity 1.Yes, 2.No.",
                "Class 1.Positive, 2.Negative."
            ]
        }
        
        feature_df = pd.DataFrame(feature_details)
        st.dataframe(feature_df, use_container_width=True)
    
    st.markdown("---")
    
    st.success("✅ Navigate to 'Evaluation on Test Data' to see model performance results!")
    # =====================================================================================
# PAGE 2: EVALUATION ON TEST DATA (DEFAULT)
# =====================================================================================

elif page == "📈 Evaluation on Test Data (DEFAULT)":
    
    st.header("📈 Evaluation on Test Data (DEFAULT)")
    
    st.info("ℹ️ Using training test split (20% - 104 rows). All metrics are pre-computed from training.")
    
    st.markdown("---")
    
    # =====================================================================================
    # MODEL COMPARISON TABLE
    # =====================================================================================
    
    st.subheader("📋 Model Comparison Table")
    
    st.markdown("""
    The table below compares all 6 models based on evaluation metrics computed during training on the **20% test split**.
    """)
    
    comparison_df = pd.DataFrame(training_metrics).T
    comparison_df = comparison_df.reset_index()
    comparison_df.columns = ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    
    # Format metrics to 4 decimal places
    for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(comparison_df, use_container_width=True)
    
    st.markdown("---")
    
    # =====================================================================================
    # OBSERVATIONS TABLE
    # =====================================================================================
    
    st.subheader("📝 Model Performance Observations")
    
    observations = {
        "ML Model Name": [
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ],
        "Observation about model performance": [
            "Provides good baseline performance with balanced precision-recall tradeoff; coefficients are interpretable for feature importance analysis.",
            "Tends to overfit on training data; shows lower generalization compared to ensemble methods; sensitive to small data variations.",
            "Performance is moderate; highly sensitive to feature scaling and the choice of k parameter; computationally expensive for large datasets.",
            "Computationally efficient and fast; performs surprisingly well despite the feature independence assumption; good for real-time predictions.",
            "Strong ensemble method with high AUC; robust against overfitting through bagging and feature randomness; provides good feature importance.",
            "Achieves best overall performance across all metrics; effectively captures complex non-linear patterns in data; requires careful hyperparameter tuning."
        ]
    }
    
    observations_df = pd.DataFrame(observations)
    st.dataframe(observations_df, use_container_width=True)
    
    st.markdown("---")
    
    # =====================================================================================
    # MODEL SELECTION FOR DETAILED EVALUATION
    # =====================================================================================
    
    st.subheader("🤖 Detailed Model Evaluation")
    
    st.markdown("Select a model to view detailed evaluation metrics, confusion matrix, and classification report.")
    
    model_choice = st.selectbox(
        "Choose a model for detailed evaluation:",
        list(model_files.keys()),
        index=0,
        key="test_data_model_selection"
    )
    
    st.markdown("---")
    
    # Load test data
    if os.path.exists("data/test_data.csv"):
        test_df = pd.read_csv("data/test_data.csv")
        X_test = test_df[feature_names]
        y_test = test_df["target"]
        
        # Load selected model
        model_path = f"{MODEL_DIR}/{model_files[model_choice]}"
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            st.stop()
        
        model = joblib.load(model_path)
        
        # Make predictions
        X_scaled = scaler.transform(X_test)
        pred = model.predict(X_scaled)
        pred_prob = model.predict_proba(X_scaled)[:, 1]
        
        # =====================================================================================
        # DISPLAY EVALUATION METRICS
        # =====================================================================================
        
        st.subheader(f"📊 Evaluation Metrics — {model_choice}")
        
        # Get metrics from training (STATIC - from metrics.json)
        metrics = training_metrics.get(model_choice, {})
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics.get('Accuracy', 0):.4f}")
        col1.metric("AUC", f"{metrics.get('AUC', 0):.4f}")
        col2.metric("Precision", f"{metrics.get('Precision', 0):.4f}")
        col2.metric("Recall", f"{metrics.get('Recall', 0):.4f}")
        col3.metric("F1", f"{metrics.get('F1', 0):.4f}")
        col3.metric("MCC", f"{metrics.get('MCC', 0):.4f}")
        
        st.markdown("---")
        
        # =====================================================================================
        # CONFUSION MATRIX
        # =====================================================================================
        
        st.subheader(f"🔢 Confusion Matrix — {model_choice}")
        
        cm = np.array(training_cm[model_choice])
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes Risk", "Diabetes Risk"],
            yticklabels=["No Diabetes Risk", "Diabetes Risk"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {model_choice}")
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # =====================================================================================
        # CLASSIFICATION REPORT
        # =====================================================================================
        
        st.subheader(f"📄 Classification Report — {model_choice}")
        
        cr = training_cr[model_choice]
        cr_df = pd.DataFrame(cr).transpose()
        
        # Format numeric columns
        for col in cr_df.columns:
            if cr_df[col].dtype in ['float64', 'int64']:
                cr_df[col] = cr_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        
        st.dataframe(cr_df, use_container_width=True)
        
        st.markdown("---")
        
        # =====================================================================================
        # PREDICTIONS TABLE
        # =====================================================================================
        
        st.subheader(f"🎯 Predictions — {model_choice}")
        
        output = X_test.copy()
        output["Actual"] = y_test.values
        output["Predicted"] = pred
        output["Confidence (%)"] = (pred_prob * 100).round(2)
        output["Correct"] = (output["Predicted"] == output["Actual"]).map({True: "✅ Yes", False: "❌ No"})
        
        st.markdown(f"**Total Predictions:** {len(output)} | **Correct:** {(pred == y_test).sum()} | **Incorrect:** {(pred != y_test).sum()}")
        
        with st.expander("📄 Preview Predictions (First 10 rows)"):
            st.dataframe(output.head(10), use_container_width=True)
        
        # Download predictions
        st.download_button(
            "📥 Download Full Predictions CSV",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{model_choice.replace(' ', '_').lower()}_test_data.csv",
            mime="text/csv"
        )
        
    else:
        st.error("❌ test_data.csv not found. Please ensure the file exists in the data/ folder.")
# =====================================================================================
# PAGE 3: UPLOAD & PREDICT ON NEW DATA (DYNAMIC)
# =====================================================================================

else:  # page == "📤 Upload & Predict on New Data"
    
    st.header("📤 Upload & Predict on New Data")
    
    st.markdown("""
    Upload your own CSV file to test the models on new data. The file must contain the same 13 features plus the target column.
    """)
    
    # =====================================================================================
    # DOWNLOAD TEST DATA BUTTON
    # =====================================================================================
    
    if os.path.exists("data/test_data.csv"):
        test_data_df = pd.read_csv("data/test_data.csv")
        st.download_button(
            "📥 Download Test Data (104 rows) - See Required Format",
            data=test_data_df.to_csv(index=False).encode("utf-8"),
            file_name="test_data.csv",
            mime="text/csv",
            help="Download this file to see the required format for your upload"
        )
    else:
        st.warning("⚠️ test_data.csv not found in data/ folder")
    
    st.markdown("---")
    
    # =====================================================================================
    # FILE UPLOAD (1 MARK - DATASET UPLOAD OPTION)
    # =====================================================================================
    
    uploaded_file = st.file_uploader(
        "📁 Upload your CSV file:",
        type=["csv"],
        help="Upload a CSV with 13 features + target column"
    )
    
    if uploaded_file is not None:
        
        try:
            # =====================================================================================
            # LOAD UPLOADED DATA
            # =====================================================================================
            
            df = pd.read_csv(uploaded_file)
            
            # =====================================================================================
            # VALIDATION 1: CHECK IF FILE IS EMPTY (CRITICAL - MUST BE FIRST)
            # =====================================================================================
            
            if df.empty or len(df) == 0:
                st.error("❌ **The uploaded file is empty (0 rows).**")
                st.error("Please upload a CSV file with at least 1 row of data.")
                st.info("💡 **Tip:** Download the test data file above to see the required format.")
                st.stop()
            
            # SUCCESS MESSAGE (NO ROW COUNT AS REQUESTED)
            st.success("✅ File uploaded successfully!")
            
            # =====================================================================================
            # VALIDATION 2: CHECK REQUIRED COLUMNS
            # =====================================================================================
            
            required_cols = set(feature_names + ["target"])
            df_cols = set(df.columns)
            
            missing = required_cols - df_cols
            extra = df_cols - required_cols
            
            if missing:
                st.error(f"❌ **Missing required columns:** `{', '.join(sorted(missing))}`")
                st.error("**Your CSV must contain all of the following columns:**")
                st.code(", ".join(feature_names + ["target"]), language="text")
                st.info("💡 **Tip:** Download the test data file above to see the correct format.")
                st.stop()
            
            if extra:
                st.warning(f"⚠️ **Extra columns detected** (will be ignored): `{', '.join(sorted(extra))}`")
            
            # =====================================================================================
            # VALIDATION 3: EXTRACT FEATURES AND TARGET
            # =====================================================================================
            
            X_test = df[feature_names]
            y_test = df["target"]
            
            # =====================================================================================
            # VALIDATION 4: CHECK DATA TYPES (NON-NUMERIC VALUES)
            # =====================================================================================
            
            try:
                X_test = X_test.astype(float)
            except ValueError:
                st.error("❌ **Non-numeric values detected in feature columns.**")
                st.error("All feature columns must contain **only numeric values** (integers or floats).")
                st.info("💡 **Tip:** Check your CSV for text values, special characters, or empty cells in numeric columns.")
                
                problematic_cols = []
                for col in feature_names:
                    try:
                        df[col].astype(float)
                    except:
                        problematic_cols.append(col)
                
                if problematic_cols:
                    st.error(f"**Problematic columns:** `{', '.join(problematic_cols)}`")
                
                st.stop()
            
            # =====================================================================================
            # VALIDATION 5: CHECK FOR MISSING VALUES IN FEATURES
            # =====================================================================================
            
            if X_test.isnull().any().any():
                missing_features = X_test.columns[X_test.isnull().any()].tolist()
                missing_counts = X_test[missing_features].isnull().sum().to_dict()
                
                st.error("❌ **Missing values detected in feature columns:**")
                
                for feat, count in missing_counts.items():
                    st.error(f"   - `{feat}`: {count} missing value(s)")
                
                st.error("**Please handle missing values before uploading.**")
                st.info("💡 **Tip:** Use imputation (mean/median/mode) or remove rows with missing values.")
                st.stop()
            
            # =====================================================================================
            # VALIDATION 6: CHECK FOR MISSING VALUES IN TARGET
            # =====================================================================================
            
            if y_test.isnull().any():
                missing_count = y_test.isnull().sum()
                st.error(f"❌ **Missing values detected in target column:** {missing_count} missing value(s)")
                st.error("The **'target'** column must have no missing values.")
                st.info("💡 **Tip:** Remove rows where target is missing or assign appropriate values.")
                st.stop()
            
            # =====================================================================================
            # VALIDATION 7: VALIDATE TARGET VALUES (MUST BE 0 OR 1)
            # =====================================================================================
            
            unique_targets = sorted(y_test.unique())
            
            if not set(unique_targets).issubset({0, 1}):
                st.error(f"❌ **Invalid target values detected:** `{unique_targets}`")
                st.error("The **'target'** column must contain **only 0 and 1**.")
                st.error("   - **0** = No Diabetes Risk")
                st.error("   - **1** = Diabetes Risk")
                st.info("💡 **Tip:** Check your target column for unexpected values or typos.")
                st.stop()
            
            # =====================================================================================
            # MINIMAL DATASET INFO (AS REQUESTED - LIMITED INFO ONLY)
            # =====================================================================================
            
            st.info(f"ℹ️ Dataset loaded: {len(df)} instances, {len(feature_names)} features")
            
            with st.expander("📄 Preview Uploaded Data (First 10 rows)"):
                st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("---")
            
            # =====================================================================================
            # MODEL SELECTION (1 MARK - ONLY SHOWN IF ALL VALIDATIONS PASS)
            # =====================================================================================
            
            st.subheader("🤖 Select Model for Evaluation")
            
            model_choice = st.selectbox(
                "Choose a model:",
                list(model_files.keys()),
                index=0,
                key="upload_data_model_selection"
            )
            
            # Load selected model
            model_path = f"{MODEL_DIR}/{model_files[model_choice]}"
            
            if not os.path.exists(model_path):
                st.error(f"❌ Model file not found: `{model_path}`")
                st.error("Please ensure all model files are present in the `model/` directory.")
                st.stop()
            
            model = joblib.load(model_path)
            
            st.markdown("---")
            
            # =====================================================================================
            # PREDICTIONS & EVALUATION (DYNAMIC - COMPUTED LIVE)
            # =====================================================================================
            
            # Scale features
            X_scaled = scaler.transform(X_test)
            
            # Make predictions
            pred = model.predict(X_scaled)
            pred_prob = model.predict_proba(X_scaled)[:, 1]
            
            # =====================================================================================
            # COMPUTE METRICS (LIVE) (1 MARK - DISPLAY EVALUATION METRICS)
            # =====================================================================================
            
            st.subheader(f"📊 Evaluation Metrics — {model_choice}")
            
            acc = accuracy_score(y_test, pred)
            auc = roc_auc_score(y_test, pred_prob)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, pred)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{acc:.4f}")
                st.metric("AUC Score", f"{auc:.4f}")
            
            with col2:
                st.metric("Precision", f"{prec:.4f}")
                st.metric("Recall", f"{rec:.4f}")
            
            with col3:
                st.metric("F1 Score", f"{f1:.4f}")
                st.metric("MCC", f"{mcc:.4f}")
            
            st.markdown("---")
            
            # =====================================================================================
            # CONFUSION MATRIX (LIVE) (1 MARK - REQUIRED BY PDF)
            # =====================================================================================
            
            st.subheader(f"🔢 Confusion Matrix — {model_choice}")
            
            cm = confusion_matrix(y_test, pred)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Diabetes", "Diabetes"],
                yticklabels=["No Diabetes ", "Diabetes"],
                ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix - {model_choice}")
            st.pyplot(fig)
            plt.close()
            
            st.markdown("---")
            
            # =====================================================================================
            # CLASSIFICATION REPORT (LIVE) (1 MARK - REQUIRED BY PDF)
            # =====================================================================================
            
            st.subheader(f"📄 Classification Report — {model_choice}")
            
            cr = classification_report(y_test, pred, output_dict=True, zero_division=0)
            cr_df = pd.DataFrame(cr).transpose()
            
            # Format numeric columns
            for col in cr_df.columns:
                if cr_df[col].dtype in ['float64', 'int64']:
                    cr_df[col] = cr_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
            
            st.dataframe(cr_df, use_container_width=True)
            
            st.markdown("---")
            
            # =====================================================================================
            # PREDICTIONS TABLE (NO SUMMARY STATISTICS AS REQUESTED)
            # =====================================================================================
            
            st.subheader(f"🎯 Predictions — {model_choice}")
            
            output = X_test.copy()
            output["Actual"] = y_test.values
            output["Predicted"] = pred
            output["Confidence (%)"] = (pred_prob * 100).round(2)
            output["Correct"] = (output["Predicted"] == output["Actual"]).map({True: "✅ Yes", False: "❌ No"})
            
            # Show preview only
            with st.expander("📄 Preview Predictions (First 10 rows)"):
                st.dataframe(output.head(10), use_container_width=True)
            
            # =====================================================================================
            # DOWNLOAD OPTIONS (AS REQUESTED)
            # =====================================================================================
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download full predictions
                st.download_button(
                    label="📥 Download Full Predictions CSV",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name=f"predictions_{model_choice.replace(' ', '_').lower()}_uploaded_data.csv",
                    mime="text/csv",
                    help="Download all predictions with actual values, predicted values, and confidence scores"
                )
            
            with col2:
                # Download prediction summary (OPTIONAL - AS REQUESTED)
                correct_count = (pred == y_test).sum()
                incorrect_count = (pred != y_test).sum()
                accuracy_pct = (correct_count / len(pred)) * 100
                
                summary_text = f"""Prediction Summary for {model_choice}

Total Predictions: {len(output)}
Correct: {correct_count} ({accuracy_pct:.2f}%)
Incorrect: {incorrect_count} ({100 - accuracy_pct:.2f}%)

Evaluation Metrics:
- Accuracy: {acc:.4f}
- AUC Score: {auc:.4f}
- Precision: {prec:.4f}
- Recall: {rec:.4f}
- F1 Score: {f1:.4f}
- MCC: {mcc:.4f}
"""
                
                st.download_button(
                    label="📥 Download Prediction Summary (TXT)",
                    data=summary_text.encode("utf-8"),
                    file_name=f"summary_{model_choice.replace(' ', '_').lower()}_uploaded_data.txt",
                    mime="text/plain",
                    help="Download a text summary of predictions and metrics"
                )
            
        except Exception as e:
            # =====================================================================================
            # EXCEPTION HANDLING (CATCH-ALL FOR UNEXPECTED ERRORS)
            # =====================================================================================
            
            st.error("❌ **An unexpected error occurred while processing your file.**")
            st.error(f"**Error details:** `{str(e)}`")
            st.info("💡 **Troubleshooting tips:**")
            st.markdown("""
            - Ensure your CSV file is properly formatted
            - Check that all columns contain appropriate data types
            - Verify that there are no special characters or encoding issues
            - Try downloading the test data file above and comparing formats
            """)
            
            with st.expander("🔍 View Full Error Traceback (for debugging)"):
                import traceback
                st.code(traceback.format_exc())
    
    else:
        # =====================================================================================
        # NO FILE UPLOADED MESSAGE
        # =====================================================================================
        
        st.info("👆 **Please upload a CSV file to begin evaluation.**")
        
        st.markdown("""
        ### 📋 Required CSV Format
        
        Your CSV file must contain the following columns:
        
        **Features (15 columns):**
        - Age,Sex,Polyuria,Polydipsia,sudden weight loss,weakness,Polyphagia,Genital thrush,visual blurring,Itching,Irritability,delayed healing,partial paresis,muscle stifness,Alopecia,Obesity
        
        **Target (1 column):**
        - status (values: 0 or 1)
        
        **Total:** 16 columns
        
        ### 💡 Tips:
        - Download the test data file above to see the exact format required
        - Ensure all feature columns contain numeric values
        - Ensure target column contains only 0 and 1
        - Remove any rows with missing values
        """)

# =====================================================================================
# FOOTER (REQUIRED - SHOWS ON ALL PAGES)
# =====================================================================================

st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p style='font-size: 16px; margin-bottom: 10px;'>
            💙 <b>ML Assignment 2</b> – BITS Pilani M.Tech AIML/DSE
        </p>
        <p style='font-size: 14px; color: #888;'>
            Developed with Streamlit | Deployed on Streamlit Community Cloud
        </p>
        <p style='font-size: 12px; color: #aaa; margin-top: 10px;'>
            © Early Stage Diabetes Risk Prediction Application
        </p>
    </div>
    """,
    unsafe_allow_html=True
)