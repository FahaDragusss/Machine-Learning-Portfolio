import streamlit as st
import numpy as np
import joblib
from itertools import combinations_with_replacement


def polynomial_features(X, degree):
    """
    Generate polynomial features for input matrix X up to a given degree.
    Parameters:
        X: numpy array of shape (m, n)
        degree: int, highest polynomial degree to generate
    Returns:
        X_poly: numpy array of shape (m, num_features)
    """
    m, n = X.shape
    features = [np.ones((m, 1))]  # Bias term: degree 0
    
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n), deg):
            # Multiply columns based on index combo
            col = np.ones(m)
            for idx in combo:
                col *= X[:, idx]
            features.append(col.reshape(-1, 1))
    
    return np.hstack(features)


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
            <div class="landing-title">Polynomial Regression</div>
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
st.set_page_config(page_title="Polynomial Regression App", layout="centered")

# Title
st.title("🚗💨Polynomial Regression Predictor")
st.markdown("Enter Features to predict CO2 emisions of a car.")

# Initialize session state to store default input features
if "input_features" not in st.session_state:
    st.session_state.input_features = [0.0] * 12 # 12 features initialized to 0.0

# Feature input section
st.subheader("🔢 Input Features")
features = []
for i in range(12): # Assuming 12 features based on the dataset
    val = st.number_input(
        f"Feature {i+1}",
        value=float(st.session_state.input_features[i]),  # Cast to float here
        key=f"feature_{i}",
        step=0.1
    )
    features.append(val)

# Prediction logic (example uses mean, std, w, b — assume they're defined)
if st.button("🔍 Predict"):
    features = np.array(features).reshape(1, -1)  # Shape becomes (1, n)
    poly_features = polynomial_features(features, degree=5)  # Generate polynomial features
    x_std = (np.array(poly_features) - mean) / std
    prediction = np.dot(x_std, w) + b
    st.success(f"🎯 Predicted Value: **{float(prediction[0]):.2f}**")


with st.expander("💡 Example Test Inputs"):
    st.write("Each entry shows the true value (ground truth) followed by the input features used for prediction.")

    examples = [
        {
            "true_value": 259,
            "features": [1.5260563034950492,2.9856819377004897,2.6461747973841225,2.8449093838194077,0,0,0,1,0,0,0,0
]
        },
        {
            "true_value": 342,
            "features": [1.791759469228055,3.258096538021482,2.884800712846709,3.109060958860994,0,0,0,0,1,0,0,0
]
        },
        {
            "true_value": 245,
            "features": [1.4586150226995167,2.8678989020441064,2.6100697927420065,2.760009940032921,0,0,0,1,0,0,0,0
]
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
    - **Mean Squared Error (MSE):** 87.8934 
    - **Mean Absolute Error (MAE):** 5.1873  
    - **R² Score:** 0.9677
    """)

    st.subheader("🎓 Training Metrics")
    st.markdown("""
    - **Mean Squared Error (MSE):** 21.6996  
    - **Mean Absolute Error (MAE):** 4.0101 
    - **R² Score:** 0.9894
    """)

# Animation
st.image("cost_convergence.gif", caption="Model Training", use_container_width=True)

st.image("Predicted_vs_actual.png", caption="Predicted vs Actual", use_container_width=True)

# Metadata
with st.expander("📂 Model Details"):
    st.write("Unfortunately, the model weights and bias were a large numpy array and could not be displayed directly in the app. However, they are saved in `.npy` files for further analysis. You can them in my GitHub repository. Github.com/FahaDragusss/Machine-Learning-Cloud-Deployment-Portfolio/Linear Regression Models from Scratch Deployment/Polynomial Models/Traditional/Results")
#    st.write("**Weights:**", w.flatten().tolist())
#    st.write("**Bias:**", b)
#    st.write("**Mean:**", mean.tolist())
#    st.write("**Std Dev:**", std.tolist())

# Dataset used
with st.expander("📚 Dataset Information"):
    st.markdown("**Model:** Simple Polynomial Regression")
    st.write("Trained using gradient descent with adam optimizer from scratch (no ML libraries used for training besides good'ol numpy).")
    st.write("The model was trained on a dataset with multiple features, including both numerical and categorical data. All features were transformed to polynomial features using a funtion created also by scratch then they were normalized before training.")

    st.markdown("🔗 **Source Dataset:** [Vehicle CO2 Emissions Dataset](https://www.kaggle.com/datasets/brsahan/vehicle-co2-emissions-dataset)")
    st.markdown("💻 **Model & Code Repository:** [GitHub: Machine Learning Portfolio](https://github.com/FahaDragusss/Machine-Learning-Portfolio)")

