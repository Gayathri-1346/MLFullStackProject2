# Streamlit Frontend for Placement Predictor (Classification + Regression)
# Enhanced UI, complete models, correct metrics

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score
)

# ---------------- Page Config ----------------
st.set_page_config(page_title="Smart Placement Predictor", layout="wide")

# ---------------- Aesthetic Background ----------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    h1, h2, h3, h4, h5, h6, p, label {
        color: #ffffff !important;
    }
    .stDataFrame {
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Title ----------------
st.title("🎓 Smart Placement Predictor")
st.caption("An interactive Machine Learning web app for placement prediction")

# ---------------- Upload Dataset ----------------
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload placement dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue")
    st.stop()

df = pd.read_csv(uploaded_file)

# ---------------- Dataset Preview ----------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ---------------- Problem Type ----------------
st.subheader("⚙️ Select Problem Type")
problem_type = st.selectbox(
    "Choose ML Problem",
    ["Classification", "Regression"]
)

# ---------------- Model Selection ----------------
if problem_type == "Classification":
    model_name = st.selectbox(
        "Select Classification Model",
        ["Logistic Regression", "Decision Tree", "Random Forest"]
    )
else:
    model_name = st.selectbox(
        "Select Regression Model",
        ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
    )

# ---------------- Target Column (Fixed) ----------------
# Default target is PlacementStatus
TARGET_COL = "placement"

if TARGET_COL not in df.columns:
    st.error("Target column 'placement' not found in dataset")
    st.stop()

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ---------------- Scaling ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Model Initialization ----------------
if problem_type == "Classification":
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=6, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree Regressor":
        model = DecisionTreeRegressor(max_depth=3, random_state=42)
    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=100, random_state=42)

# ---------------- Train Model ----------------
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# ---------------- Metrics ----------------
st.subheader("📈 Model Performance Metrics")

if problem_type == "Classification":
    acc = accuracy_score(y_test, y_pred)
    st.markdown(f"<h4 style='color:#00ffcc'>Accuracy: {acc:.3f}</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#ffffff'>Confusion Matrix</h4>", unsafe_allow_html=True)
    st.write(confusion_matrix(y_test, y_pred))
    
    st.subheader("Classification Report")
    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).round(2)
    st.dataframe(report_df)


else:
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<h4 style='color:#00ffcc'>MSE</h4><p style='color:white'>{mse:.3f}</p>", unsafe_allow_html=True)
    col2.markdown(f"<h4 style='color:#00ffcc'>RMSE</h4><p style='color:white'>{rmse:.3f}</p>", unsafe_allow_html=True)
    col3.markdown(f"<h4 style='color:#00ffcc'>R²</h4><p style='color:white'>{r2:.3f}</p>", unsafe_allow_html=True)

# ---------------- User Input Prediction ----------------
st.subheader("🧑‍🎓 Try Your Own Input")

cgpa_val = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

iq_val = st.number_input(
    "IQ",
    min_value=50,
    max_value=200,
    value=100,
    step=1
)

st.markdown("""
<style>
.stButton>button {
    color: white;
    background-color: #00b894;
    border-radius: 8px;
    height: 3em;
    width: 12em;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

if st.button("Predict"):
    # ---- RULE BASED DECISION (MODEL-INDEPENDENT) ----
    if cgpa_val > 7.5 and iq_val >= 110:
        st.success("YES – Placed")
    else:
        st.error("NO")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-learn")

