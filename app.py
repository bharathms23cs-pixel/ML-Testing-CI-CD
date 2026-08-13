import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------------
# PAGE TITLE
# -----------------------------------

st.set_page_config(
    page_title="ML Testing Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ML Model Testing Dashboard")
st.write("House Price Prediction - ML Testing and Performance")

# -----------------------------------
# LOAD DATASET
# -----------------------------------

try:
    data = pd.read_csv("house(2).csv")
except Exception as e:
    st.error("Dataset could not be loaded.")
    st.error(str(e))
    st.stop()

# -----------------------------------
# LOAD MODEL
# -----------------------------------

try:
    model = joblib.load("model(1).pkl")
except Exception as e:
    st.error("Model could not be loaded.")
    st.error(str(e))
    st.stop()

# -----------------------------------
# PREPARE DATA
# -----------------------------------

target = "MedHouseVal"

if target not in data.columns:
    st.error("Target column 'MedHouseVal' was not found in the dataset.")
    st.write("Available columns:", list(data.columns))
    st.stop()

X = data.drop(target, axis=1)
y = data[target]

# Use same test split every time
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# PREDICTION
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# CALCULATE TEST RESULTS
# -----------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

# -----------------------------------
# DISPLAY METRICS
# -----------------------------------

st.subheader("📊 Model Test Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "R² Score",
        f"{r2:.4f}"
    )

with col2:
    st.metric(
        "MAE",
        f"{mae:.4f}"
    )

with col3:
    st.metric(
        "RMSE",
        f"{rmse:.4f}"
    )

# -----------------------------------
# TEST STATUS
# -----------------------------------

st.subheader("✅ Test Status")

if r2 >= 0.70:
    st.success("MODEL TEST PASSED")
else:
    st.error("MODEL TEST FAILED")

# -----------------------------------
# GRAPH 1 - ACTUAL VS PREDICTED
# -----------------------------------

st.subheader("📈 Actual vs Predicted Values")

fig1, ax1 = plt.subplots()

ax1.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

ax1.set_xlabel("Actual House Price")
ax1.set_ylabel("Predicted House Price")
ax1.set_title("Actual vs Predicted")

st.pyplot(fig1)

# -----------------------------------
# GRAPH 2 - MODEL METRICS
# -----------------------------------

st.subheader("📊 Model Performance Graph")

metrics = [
    "R² Score",
    "MAE",
    "RMSE"
]

values = [
    r2,
    mae,
    rmse
]

fig2, ax2 = plt.subplots()

ax2.bar(metrics, values)

ax2.set_ylabel("Score")
ax2.set_title("ML Model Performance")

st.pyplot(fig2)

# -----------------------------------
# GRAPH 3 - PREDICTION COMPARISON
# -----------------------------------

st.subheader("📉 Actual and Predicted Comparison")

comparison = pd.DataFrame({
    "Actual": y_test.values[:30],
    "Predicted": y_pred[:30]
})

fig3, ax3 = plt.subplots()

ax3.plot(
    comparison["Actual"],
    label="Actual"
)

ax3.plot(
    comparison["Predicted"],
    label="Predicted"
)

ax3.set_xlabel("Test Sample")
ax3.set_ylabel("House Price")
ax3.set_title("Actual vs Predicted - First 30 Test Samples")

ax3.legend()

st.pyplot(fig3)

# -----------------------------------
# DATASET INFORMATION
# -----------------------------------

st.subheader("📋 Dataset Information")

col4, col5 = st.columns(2)

with col4:
    st.write("Total records:", len(data))

with col5:
    st.write("Number of features:", len(X.columns))

st.dataframe(data.head(10))
