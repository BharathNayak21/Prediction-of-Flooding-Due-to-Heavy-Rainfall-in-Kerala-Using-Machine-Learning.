import streamlit as st
import joblib
import pandas as pd


# =========================================================
# LOAD MODEL AND HISTORICAL DATA
# =========================================================

# Load trained Random Forest model
model = joblib.load("random_forest_model.pkl")

# Load full Kerala rainfall dataset
historical_data = pd.read_csv("data/kerala_rainfall.csv")

# Create June rainfall change
historical_data["June_change"] = abs(
    historical_data["MAY"] - historical_data["JUN"]
)

# Create rainfall-based flood-risk target
KERALA_THRESHOLD = 2049.3

historical_data["Flood_Risk"] = (
    historical_data["Jun-Sep"] > KERALA_THRESHOLD
).astype(int)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Kerala Flood Risk Assessment",
    page_icon="🌧️",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🌧️ Kerala Flood Risk Assessment")

st.write(
    "Machine learning based rainfall analysis for "
    "assessing flood-risk conditions in Kerala."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📌 Project Information")

st.sidebar.write("**Region:** Kerala, India")
st.sidebar.write("**Dataset Period:** 1901–2015")
st.sidebar.write("**Kerala Records:** 115")
st.sidebar.write("**ML Features:** 3")
st.sidebar.write("**Models Tested:** 4")
st.sidebar.write("**Best Model:** Random Forest")
st.sidebar.write("**Risk Threshold:** 2049.3 mm")

st.sidebar.divider()

st.sidebar.info(
    "This system provides a rainfall-based flood-risk signal. "
    "It does not directly predict observed physical flood events."
)


# =========================================================
# INPUT SECTION
# =========================================================

st.header("🌧️ Enter Rainfall Information")

st.write(
    "Enter the rainfall values below to assess the "
    "rainfall-based flood-risk condition."
)

col1, col2, col3 = st.columns(3)

with col1:
    mar_may = st.number_input(
        "March–May Rainfall (mm)",
        min_value=0.0,
        value=300.0,
        step=10.0
    )

with col2:
    june = st.number_input(
        "June Rainfall (mm)",
        min_value=0.0,
        value=600.0,
        step=10.0
    )

with col3:
    june_change = st.number_input(
        "June Rainfall Change (mm)",
        min_value=0.0,
        value=300.0,
        step=10.0
    )


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔍 Predict Flood Risk",
    use_container_width=True
):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Mar-May": [mar_may],
        "June": [june],
        "June_change": [june_change]
    })

    # Model prediction
    prediction = model.predict(input_data)[0]

    # Model probability
    probabilities = model.predict_proba(input_data)[0]

    # Probability of higher-risk class
    high_risk_probability = probabilities[1] * 100


    # =====================================================
    # RISK LEVEL INTERPRETATION
    # =====================================================

    if high_risk_probability < 25:

        risk_level = "LOW RISK"

        risk_message = (
            "The rainfall pattern indicates a relatively "
            "lower flood-risk condition."
        )

        result_function = st.success

    elif high_risk_probability < 50:

        risk_level = "MODERATE RISK"

        risk_message = (
            "The rainfall pattern indicates a moderate "
            "flood-risk condition."
        )

        result_function = st.warning

    elif high_risk_probability < 75:

        risk_level = "HIGH RISK"

        risk_message = (
            "The rainfall pattern indicates an elevated "
            "flood-risk condition."
        )

        result_function = st.warning

    else:

        risk_level = "VERY HIGH RISK"

        risk_message = (
            "The rainfall pattern indicates a strongly "
            "elevated flood-risk condition."
        )

        result_function = st.error


    # =====================================================
    # RESULT SECTION
    # =====================================================

    st.divider()

    st.header("🎯 Prediction Result")

    result_function(
        f"{risk_level} — "
        f"{high_risk_probability:.1f}% estimated probability"
    )

    # Progress bar
    st.progress(
        min(int(high_risk_probability), 100)
    )

    st.write(risk_message)


    # Binary model result
    if prediction == 1:

        st.info(
            "The Random Forest model classified this "
            "rainfall pattern as the higher-risk class."
        )

    else:

        st.success(
            "The Random Forest model classified this "
            "rainfall pattern as the lower-risk class."
        )


    # =====================================================
    # INPUT SUMMARY
    # =====================================================

    st.subheader("📊 Rainfall Assessment")

    assessment_col1, assessment_col2, assessment_col3 = st.columns(3)

    with assessment_col1:

        st.metric(
            "March–May Rainfall",
            f"{mar_may:.1f} mm"
        )

    with assessment_col2:

        st.metric(
            "June Rainfall",
            f"{june:.1f} mm"
        )

    with assessment_col3:

        st.metric(
            "June Rainfall Change",
            f"{june_change:.1f} mm"
        )


    # =====================================================
    # SIMPLE INTERPRETATION
    # =====================================================

    st.subheader("📝 Prediction Interpretation")

    st.write(
        "The model uses the three rainfall features — "
        "March–May rainfall, June rainfall, and June rainfall "
        "change — to classify the entered rainfall pattern."
    )

    st.write(
        f"The model estimated a {high_risk_probability:.1f}% "
        "probability for the higher-risk class."
    )

    st.caption(
        "The probability is the Random Forest model's estimated "
        "probability for the higher-risk class. It should not be "
        "interpreted as the actual probability that a physical "
        "flood will occur."
    )


# =========================================================
# HISTORICAL KERALA RAINFALL
# =========================================================

st.divider()

st.header("📈 Historical Kerala Rainfall Analysis")

st.write(
    "Historical rainfall statistics from the Kerala dataset "
    "used in this project."
)


# =========================================================
# HISTORICAL STATISTICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    average_rainfall = historical_data["Jun-Sep"].mean()

    st.metric(
        "Average Jun-Sep Rainfall",
        f"{average_rainfall:.1f} mm"
    )


with col2:

    maximum_rainfall = historical_data["Jun-Sep"].max()

    st.metric(
        "Maximum Jun-Sep Rainfall",
        f"{maximum_rainfall:.1f} mm"
    )


with col3:

    minimum_rainfall = historical_data["Jun-Sep"].min()

    st.metric(
        "Minimum Jun-Sep Rainfall",
        f"{minimum_rainfall:.1f} mm"
    )


with col4:

    high_risk_years_count = (
        historical_data["Flood_Risk"] == 1
    ).sum()

    st.metric(
        "Higher-Risk Years",
        int(high_risk_years_count)
    )


# =========================================================
# RAINFALL CHART
# =========================================================

st.subheader("📊 Jun-Sep Rainfall by Year")

chart_data = historical_data[
    ["YEAR", "Jun-Sep"]
].copy()

chart_data = chart_data.set_index("YEAR")

st.line_chart(chart_data)


# =========================================================
# THRESHOLD INFORMATION
# =========================================================

st.subheader("🚨 Kerala Rainfall Threshold")

st.write(
    f"The rainfall threshold used to create the historical "
    f"Flood_Risk target is **{KERALA_THRESHOLD} mm** "
    "for Jun-Sep rainfall."
)

st.write(
    "Years with Jun-Sep rainfall above this threshold "
    "are classified as the higher-risk class."
)


# =========================================================
# HISTORICAL HIGH-RISK YEARS
# =========================================================

st.subheader("📅 Historical Higher-Risk Years")

risk_years = historical_data[
    historical_data["Flood_Risk"] == 1
][
    ["YEAR", "Jun-Sep"]
].copy()

risk_years = risk_years.rename(
    columns={
        "YEAR": "Year",
        "Jun-Sep": "Jun-Sep Rainfall (mm)"
    }
)

st.dataframe(
    risk_years,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.header("🤖 Machine Learning Model Performance")

st.write(
    "Four machine learning classification algorithms were "
    "trained and evaluated using the test dataset."
)


# Model results
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "SVM"
    ],

    "Accuracy (%)": [
        73.91,
        60.87,
        78.26,
        73.91
    ],

    "Precision (%)": [
        80.00,
        50.00,
        83.33,
        100.00
    ],

    "Recall (%)": [
        44.44,
        22.22,
        55.56,
        33.33
    ],

    "F1-Score (%)": [
        57.14,
        30.77,
        66.67,
        50.00
    ]
})


# Display results table

st.dataframe(
    results,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# ACCURACY CHART
# =========================================================

st.subheader("📊 Model Accuracy Comparison")

accuracy_chart = results[
    ["Model", "Accuracy (%)"]
].copy()

accuracy_chart = accuracy_chart.set_index("Model")

st.bar_chart(accuracy_chart)


# =========================================================
# BEST MODEL
# =========================================================

st.subheader("🏆 Best Performing Model")

st.success(
    "Random Forest achieved the best overall performance "
    "among the four tested models."
)

best_col1, best_col2, best_col3, best_col4 = st.columns(4)

with best_col1:

    st.metric(
        "Accuracy",
        "78.26%"
    )

with best_col2:

    st.metric(
        "Precision",
        "83.33%"
    )

with best_col3:

    st.metric(
        "Recall",
        "55.56%"
    )

with best_col4:

    st.metric(
        "F1-Score",
        "66.67%"
    )


# =========================================================
# PROJECT WORKFLOW
# =========================================================

st.divider()

st.header("🔄 Project Workflow")

st.write(
    """
    Historical Rainfall Dataset
    ↓
    Kerala Data Extraction
    ↓
    Data Preprocessing
    ↓
    Feature Engineering
    ↓
    Flood-Risk Target Creation
    ↓
    Training and Testing Split
    ↓
    Machine Learning Models
    ↓
    Model Evaluation
    ↓
    Random Forest Selection
    ↓
    Flood-Risk Prediction
    """
)


# =========================================================
# IMPORTANT NOTE
# =========================================================

st.divider()

st.warning(
    "Important: This application provides a rainfall-based "
    "flood-risk signal derived from historical rainfall data. "
    "It is not a real-time flood warning system and does not "
    "directly predict physical flood events."
)


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "Kerala Flood Risk Assessment | "
    "Machine Learning Project"
)