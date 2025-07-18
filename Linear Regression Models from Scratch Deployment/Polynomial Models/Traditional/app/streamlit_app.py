import streamlit as st
import numpy as np
import joblib

# --- Query parameter to control navigation ---
query_params = st.query_params
if "enter" in query_params:
    st.session_state["show_landing"] = False
else:
    st.session_state["show_landing"] = True

# --- Landing Page ---
if st.session_state["show_landing"]:
    st.markdown(
        """
        <style>
        .landing-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 95vh;
            text-align: center;
            color: white;
        }

        .landing-title {
            font-size: 3.5em;
            font-weight: bold;
            color: #39FF14;
            margin-bottom: 10px;
        }

        .landing-tagline {
            font-size: 1.2em;
            color: #CCCCCC;
            margin-bottom: 5px;
        }

        .landing-author {
            font-style: italic;
            color: #999999;
            margin-bottom: 30px;
        }

        .button-wrapper {
            margin-top: 20px;
        }

        .enter-button {
            background-color: #212121;
            color: white;
            border: 2px solid #39FF14;
            padding: 10px 25px;
            border-radius: 8px;
            font-size: 1.1em;
            cursor: pointer;
            text-decoration: none;
        }

        .enter-button:hover {
            background-color: #39FF14;
            color: black;
        }
        </style>

        <div class="landing-container">
            <div class="landing-title">Multiple Linear Regression</div>
            <div class="landing-tagline">Built from scratch.</div>
            <div class="landing-author">by FahaDragusss</div>
            <div class="button-wrapper">
                <a href="?enter=true" class="enter-button">Enter App</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()


# Load model
model = joblib.load("model.joblib")
w, b, mean, std = model["w"], model["b"], model["mean"], model["std"]

# Page config
st.set_page_config(page_title="Linear Regression App", layout="centered")

# Title
st.title("🚗💨Linear Regression Predictor")
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
    - **Mean Squared Error (MSE):** 5.6752  
    - **Mean Absolute Error (MAE):** 1.8559  
    - **R² Score:** 0.8999
    """)

    st.subheader("🎓 Training Metrics")
    st.markdown("""
    - **Mean Squared Error (MSE):** 9.6826  
    - **Mean Absolute Error (MAE):** 2.2965  
    - **R² Score:** 0.8451
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

