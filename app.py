import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Traffic AI",
    page_icon="🚦",
    layout="wide"
)


# ==========================================
# LOAD DATA AND MODELS
# ==========================================

df = pd.read_csv("data/traffic_data_1000_rows.csv")

classifier = joblib.load(
    "models/traffic_classifier.pkl"
)

regressor = joblib.load(
    "models/traffic_regressor.pkl"
)


# ==========================================
# TITLE
# ==========================================

st.title("🚦 Smart Traffic Prediction & Analysis System")

st.write(
    "An AI-powered system for predicting traffic levels "
    "and estimating vehicle count based on traffic conditions."
)


# ==========================================
# DATASET OVERVIEW
# ==========================================

st.header("📊 Traffic Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        len(df)
    )

with col2:
    st.metric(
        "Features",
        df.shape[1]
    )

with col3:
    st.metric(
        "Traffic Levels",
        df["Traffic_Level"].nunique()
    )

with col4:
    st.metric(
        "Average Vehicles",
        f"{df['Vehicle_Count'].mean():.0f}"
    )


# ==========================================
# VISUALIZATIONS
# ==========================================

st.header("📈 Traffic Analysis")

col1, col2 = st.columns(2)


# Traffic Level Chart
with col1:

    st.subheader("Traffic Level Distribution")

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="Traffic_Level",
        ax=ax
    )

    ax.set_xlabel("Traffic Level")
    ax.set_ylabel("Number of Records")

    st.pyplot(fig)


# Vehicle Count by Hour
with col2:

    st.subheader("Vehicle Count by Hour")

    hourly_data = (
        df.groupby("Hour")["Vehicle_Count"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.lineplot(
        data=hourly_data,
        x="Hour",
        y="Vehicle_Count",
        marker="o",
        ax=ax
    )

    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Vehicle Count")

    st.pyplot(fig)


# ==========================================
# PREDICTION SECTION
# ==========================================

st.header("🚦 Traffic Prediction")

st.write(
    "Enter the traffic conditions to generate predictions."
)


col1, col2 = st.columns(2)


# ==========================================
# INPUTS
# ==========================================

with col1:

    date = st.date_input(
        "Date"
    )

    hour = st.slider(
        "Hour",
        0,
        23,
        12
    )

    day = st.selectbox(
        "Day",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    weather = st.selectbox(
        "Weather",
        [
            "Clear",
            "Cloudy",
            "Rainy",
            "Foggy"
        ]
    )


with col2:

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=5.0,
        max_value=45.0,
        value=27.0
    )

    holiday = st.selectbox(
        "Holiday",
        [
            "No",
            "Yes"
        ]
    )

    road_type = st.selectbox(
        "Road Type",
        [
            "Main Road",
            "Highway",
            "Residential",
            "Intersection"
        ]
    )


# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button(
    "🚦 Predict Traffic",
    use_container_width=True
):

    selected_date = pd.to_datetime(date)

    input_data = pd.DataFrame({

        "Hour": [hour],

        "Day": [day],

        "Weather": [weather],

        "Temperature": [temperature],

        "Holiday": [holiday],

        "Road_Type": [road_type],

        "Year": [selected_date.year],

        "Month": [selected_date.month],

        "Day_of_Week": [selected_date.dayofweek]
    })


    # ======================================
    # PREDICTIONS
    # ======================================

    traffic_prediction = classifier.predict(
        input_data
    )[0]

    vehicle_prediction = regressor.predict(
        input_data
    )[0]


    # ======================================
    # DISPLAY RESULTS
    # ======================================

    st.header("🎯 Prediction Results")

    result1, result2 = st.columns(2)

    with result1:

        st.metric(
            "Predicted Traffic Level",
            traffic_prediction
        )

    with result2:

        st.metric(
            "Estimated Vehicle Count",
            f"{vehicle_prediction:.0f}"
        )


    # ======================================
    # TRAFFIC MESSAGE
    # ======================================

    if traffic_prediction == "High":

        st.warning(
            "⚠️ High traffic conditions are predicted."
        )

    elif traffic_prediction == "Medium":

        st.info(
            "ℹ️ Medium traffic conditions are predicted."
        )

    else:

        st.success(
            "✅ Low traffic conditions are predicted."
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Smart Traffic Prediction & Analysis System | "
    "Machine Learning Project"
)