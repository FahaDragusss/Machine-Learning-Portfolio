import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.joblib")
w, b, mean, std = model["w"], model["b"], model["mean"], model["std"]

# Page config
st.set_page_config(page_title="Linear Regression App", layout="centered")

# Title
st.title("🏠 Linear Regression Predictor")
st.markdown("Enter 10 feature values to get a model prediction.")

# Initialize session state to store default input features
if "input_features" not in st.session_state:
    st.session_state.input_features = [0.0] * 10

# Feature input section
st.subheader("🔢 Input Features")
features = []
for i in range(10):
    val = st.number_input(
        f"Feature {i+1}",
        value=st.session_state.input_features[i],
        key=f"feature_{i}",
        step=0.1
    )
    features.append(val)

# Prediction logic (example uses mean, std, w, b — assume they're defined)
if st.button("🔍 Predict"):
    x_std = (np.array(features) - mean) / std
    prediction = np.dot(x_std, w) + b
    st.success(f"🎯 Predicted Value: **{float(prediction[0]):.2f}**")


with st.expander("💡 Example Test Inputs"):
    st.write("Each entry shows the true value (ground truth) followed by the 10 input features used for prediction.")

    examples = [
        {
            "true_value": 23.7,
            "features": [1.0, 0.0, 0.0, 0.0, 0.0, 13.0, 12.5, 7.7915, 4.2485, 4.6052]
        },
        {
            "true_value": 35.0,
            "features": [0.0, 1.0, 0.0, 0.0, 0.0, 13.0, 15.1, 7.8240, 4.8040, 4.4773]
        },
        {
            "true_value": 32.4,
            "features": [0.0, 1.0, 0.0, 0.0, 0.0, 13.0, 17.0, 7.7363, 4.6728, 4.2767]
        }
    ]

    for i, ex in enumerate(examples):
        st.write(f"**True Value:** {ex['true_value']}")
        st.json({f"Features": ex['features']})

        if st.button(f"Use Features from Example {i+1}"):
            st.session_state.input_features = ex["features"]
            st.rerun()


# Model Performance Metrics
with st.expander("📊 Model Performance Metrics"):
    st.subheader("🧪 Test Metrics (on unseen data)")
    st.markdown("""
    - **Mean Squared Error (MSE):** 6.4085  
    - **Mean Absolute Error (MAE):** 2.0322  
    - **R² Score:** 0.8841
    """)

    st.subheader("🎓 Training Metrics")
    st.markdown("""
    - **Mean Squared Error (MSE):** 8.9528  
    - **Mean Absolute Error (MAE):** 2.1759  
    - **R² Score:** 0.8550
    """)

# Animation
st.image("regression_animation.gif", caption="Model Training", use_container_width=True)

# Metadata
with st.expander("📂 Model Details"):
    st.write("**Weights:**", w.flatten().tolist())
    st.write("**Bias:**", b)
    st.write("**Mean:**", mean.tolist())
    st.write("**Std Dev:**", std.tolist())

# Dataset used
with st.expander("📚 Dataset Information"):
    st.markdown("**Model:** Multiple Linear Regression")
    st.write("Trained using gradient descent from scratch (no ML libraries used for training).")
    st.write("The model was trained on a dataset with 10 features, including both numerical and categorical data. All features were normalized before training.")

    st.markdown("🔗 **Source Dataset:** [Auto MPG Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/auto-mpg-dataset)")
    st.markdown("💻 **Model & Code Repository:** [GitHub: Machine Learning Portfolio](https://github.com/FahaDragusss/Machine-Learning-Portfolio)")

