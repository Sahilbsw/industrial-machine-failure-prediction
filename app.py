
import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature names
model = joblib.load("machine_failure_model.pkl")
features = joblib.load("model_features.pkl")

st.set_page_config(
    page_title="Industrial Machine Failure Prediction",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Industrial Machine Failure Prediction")
st.write(
    "Predict machine failure risk using sensor readings "
    "and a trained Random Forest model."
)

st.info(
    "Enter readings in the units shown. This demo uses "
    "the AI4I dataset and is not a substitute for real "
    "industrial safety monitoring."
)

st.header("Enter Machine Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=290.0,
        max_value=310.0,
        value=298.1
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=300.0,
        max_value=320.0,
        value=308.6
    )

    rpm = st.number_input(
        "Rotational Speed (rpm)",
        min_value=1000,
        max_value=3000,
        value=1500
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=100.0,
        value=40.0
    )

with col2:
    tool_wear = st.number_input(
        "Tool Wear (minutes)",
        min_value=0,
        max_value=300,
        value=10
    )

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

if st.button("Predict Failure", type="primary"):

    # Create inputs in the exact training feature order
    input_values = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type_H": int(machine_type == "H"),
        "Type_L": int(machine_type == "L"),
        "Type_M": int(machine_type == "M")
    }

    input_df = pd.DataFrame(
        [[input_values[name] for name in features]],
        columns=features
    )

    # Estimate probability using trained model
    probability = model.predict_proba(input_df)[0][1]

    # Experimental threshold selected during model testing
    threshold = 0.3
    prediction = int(probability >= threshold)

    st.header("Prediction Result")

    st.metric(
        "Estimated Failure Probability",
        f"{probability:.1%}"
    )

    st.progress(float(probability))

    if prediction == 1:
        st.error("⚠️ Failure Risk Detected")
        st.write(
            "Recommendation: Arrange a machine inspection "
            "and review maintenance requirements."
        )
    else:
        st.success("✅ Failure Not Flagged")
        st.write(
            "Recommendation: Continue routine monitoring "
            "and scheduled maintenance."
        )

    st.caption(
        "This probability is a model score, not necessarily "
        "a calibrated real-world probability. Threshold: 0.3."
    )
