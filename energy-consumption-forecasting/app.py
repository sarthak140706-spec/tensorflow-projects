import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Smart Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "xgboost_energy_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: `{MODEL_PATH}`")
    st.info("Make sure `xgboost_energy_model.pkl` is committed inside the `models/` folder in your GitHub repo.")
    st.stop()

model = joblib.load(MODEL_PATH)

# =========================
# TITLE
# =========================

st.title("⚡ Smart Energy Consumption Predictor")

st.markdown("""
Predict energy consumption using Machine Learning and energy generation metrics.
""")

st.divider()

# =========================
# TIME INPUT
# =========================

st.subheader("🕒 Time Information")

time_input = st.text_input(
    "Enter Date & Time",
    "2024-07-15 14:00:00"
)

# Convert datetime
try:
    dt = pd.to_datetime(time_input)
except Exception:
    st.error("❌ Invalid date format. Please use: YYYY-MM-DD HH:MM:SS")
    st.stop()

# Extract engineered features
hour = dt.hour
day = dt.day
month = dt.month
year = dt.year
weekday = dt.weekday()

st.info(
    f"""
    Extracted Features:
    
    Hour: {hour}
    
    Day: {day}
    
    Month: {month}
    
    Year: {year}
    
    Weekday: {weekday}
    """
)

st.divider()

# =========================
# INPUT FIELDS
# =========================

st.subheader("📊 Energy Parameters")

col1, col2, col3 = st.columns(3)

features = []

# =========================
# ENGINEERED TIME FEATURES
# =========================

features.append(hour)
features.append(day)
features.append(month)
features.append(year)
features.append(weekday)

# =========================
# COLUMN 1
# =========================

with col1:

    generation_biomass = st.number_input(
        "generation biomass",
        value=300.0
    )
    features.append(generation_biomass)

    generation_fossil_brown_coal_lignite = st.number_input(
        "generation fossil brown coal/lignite",
        value=450.0
    )
    features.append(generation_fossil_brown_coal_lignite)

    generation_fossil_coal_derived_gas = st.number_input(
        "generation fossil coal-derived gas",
        value=120.0
    )
    features.append(generation_fossil_coal_derived_gas)

    generation_fossil_gas = st.number_input(
        "generation fossil gas",
        value=5000.0
    )
    features.append(generation_fossil_gas)

    generation_fossil_hard_coal = st.number_input(
        "generation fossil hard coal",
        value=3200.0
    )
    features.append(generation_fossil_hard_coal)

    generation_fossil_oil = st.number_input(
        "generation fossil oil",
        value=250.0
    )
    features.append(generation_fossil_oil)

    generation_fossil_oil_shale = st.number_input(
        "generation fossil oil shale",
        value=0.0
    )
    features.append(generation_fossil_oil_shale)

    generation_fossil_peat = st.number_input(
        "generation fossil peat",
        value=0.0
    )
    features.append(generation_fossil_peat)

    generation_geothermal = st.number_input(
        "generation geothermal",
        value=90.0
    )
    features.append(generation_geothermal)

# =========================
# COLUMN 2
# =========================

with col2:

    generation_hydro_pumped_storage_aggregated = st.number_input(
        "generation hydro pumped storage aggregated",
        value=200.0
    )
    features.append(generation_hydro_pumped_storage_aggregated)

    generation_hydro_pumped_storage_consumption = st.number_input(
        "generation hydro pumped storage consumption",
        value=150.0
    )
    features.append(generation_hydro_pumped_storage_consumption)

    generation_hydro_run_of_river_and_poundage = st.number_input(
        "generation hydro run-of-river and poundage",
        value=400.0
    )
    features.append(generation_hydro_run_of_river_and_poundage)

    generation_hydro_water_reservoir = st.number_input(
        "generation hydro water reservoir",
        value=800.0
    )
    features.append(generation_hydro_water_reservoir)

    generation_marine = st.number_input(
        "generation marine",
        value=0.0
    )
    features.append(generation_marine)

    generation_nuclear = st.number_input(
        "generation nuclear",
        value=7000.0
    )
    features.append(generation_nuclear)

    generation_other = st.number_input(
        "generation other",
        value=50.0
    )
    features.append(generation_other)

    generation_other_renewable = st.number_input(
        "generation other renewable",
        value=120.0
    )
    features.append(generation_other_renewable)

    generation_solar = st.number_input(
        "generation solar",
        value=2500.0
    )
    features.append(generation_solar)

# =========================
# COLUMN 3
# =========================

with col3:

    generation_waste = st.number_input(
        "generation waste",
        value=180.0
    )
    features.append(generation_waste)

    generation_wind_offshore = st.number_input(
        "generation wind offshore",
        value=600.0
    )
    features.append(generation_wind_offshore)

    generation_wind_onshore = st.number_input(
        "generation wind onshore",
        value=4500.0
    )
    features.append(generation_wind_onshore)

    forecast_solar_day_ahead = st.number_input(
        "forecast solar day ahead",
        value=2400.0
    )
    features.append(forecast_solar_day_ahead)

    forecast_wind_offshore_eday_ahead = st.number_input(
        "forecast wind offshore eday ahead",
        value=550.0
    )
    features.append(forecast_wind_offshore_eday_ahead)

    forecast_wind_onshore_day_ahead = st.number_input(
        "forecast wind onshore day ahead",
        value=4300.0
    )
    features.append(forecast_wind_onshore_day_ahead)

    total_load_forecast = st.number_input(
        "total load forecast",
        value=25000.0
    )
    features.append(total_load_forecast)

    total_load_actual = st.number_input(
        "total load actual",
        value=24800.0
    )
    features.append(total_load_actual)

    price_day_ahead = st.number_input(
        "price day ahead",
        value=65.0
    )
    features.append(price_day_ahead)

    price_actual = st.number_input(
        "price actual",
        value=68.0
    )
    features.append(price_actual)

# =========================
# PREDICTION SECTION
# =========================

st.divider()

if st.button("⚡ Predict Energy Consumption"):

    try:

        # Convert features to numpy array
        input_data = np.array([features])

        # Debugging info
        st.write("Feature Count:", len(features))

        # Prediction
        prediction = model.predict(input_data)

        predicted_value = prediction[0]

        # Success Message
        st.success("Prediction Completed Successfully!")

        # Prediction Metric
        st.metric(
            label="⚡ Predicted Energy Consumption",
            value=f"{predicted_value:.2f}"
        )

        # Consumption Category
        if predicted_value < 100:
            st.info("🟢 Low Energy Consumption")

        elif predicted_value < 300:
            st.warning("🟡 Moderate Energy Consumption")

        else:
            st.error("🔴 High Energy Consumption")

        # Input Summary
        st.subheader("📋 Input Summary")

        feature_names = [
            "hour", "day", "month", "year", "weekday",
            "generation biomass", "generation fossil brown coal/lignite",
            "generation fossil coal-derived gas", "generation fossil gas",
            "generation fossil hard coal", "generation fossil oil",
            "generation fossil oil shale", "generation fossil peat",
            "generation geothermal", "generation hydro pumped storage aggregated",
            "generation hydro pumped storage consumption",
            "generation hydro run-of-river and poundage",
            "generation hydro water reservoir", "generation marine",
            "generation nuclear", "generation other",
            "generation other renewable", "generation solar",
            "generation waste", "generation wind offshore",
            "generation wind onshore", "forecast solar day ahead",
            "forecast wind offshore eday ahead", "forecast wind onshore day ahead",
            "total load forecast", "total load actual",
            "price day ahead", "price actual"
        ]

        summary_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": features
        })

        st.dataframe(summary_df, use_container_width=True)

    except Exception as e:

        st.error("Prediction Error")
        st.code(str(e))

# =========================
# FOOTER
# =========================

st.divider()

st.caption("Built using Streamlit + XGBoost + Machine Learning")