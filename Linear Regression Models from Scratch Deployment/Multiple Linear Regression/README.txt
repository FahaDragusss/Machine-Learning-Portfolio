# 📘 Multiple Linear Regression (NumPy Only · Vectorized)

This folder contains a **vectorized implementation of multiple linear regression from scratch** using only **NumPy**. The goal was to predict **Miles Per Gallon (MPG)** based on features from a car fuel consumption dataset.

This implementation **outperformed the Scikit-learn model** on the same dataset, both in terms of accuracy and generalizability. A live demo is also deployed on Hugging Face.

---

## 🧮 Mathematical Formulation

### 1. Hypothesis Function

The model predicts MPG using a linear combination of features:

ŷ = Xθ + b


Where:  
- `X` is the feature matrix  
- `θ` is the weight vector  
- `b` is the bias term  
- `ŷ` is the predicted output  

---

### 2. Cost Function (Mean Squared Error)

J(θ, b) = (1/m) * Σ (ŷᵢ - yᵢ)²


Where:  
- `m` is the number of examples  
- `yᵢ` is the true label  
- `ŷᵢ` is the predicted label  

---

### 3. Gradient Descent Update Rules

Gradients are computed and parameters are updated **simultaneously**:

θ := θ - α * ∇θ J(θ)
b := b - α * ∇b J(b)


The model uses fully **vectorized operations** for speed and efficiency.

---

## 📁 Folder Structure

├── Analysis and Visualization/ # Post-processing and model visualizations
├── app/ # Hugging Face deployment files
├── DevSet/ # Dev set for tuning & validation
├── EDA-&-Preprocessing/ # Dataset and preprocessing notebook
├── Implementation/ # Core NumPy implementation
├── Final Model Parameters.txt # Trained model parameters
├── Predicted vs Actual.png
├── regression_animation.gif
└── Residuals.png

---

## ⚙️ Implementation Overview

- `Implementation/`: Core NumPy-based linear regression with simultaneous gradient updates.
- `Analysis and Visualization/`: Includes residuals, prediction comparison, and regression animation.
- `app/`: All files required for Hugging Face deployment.
- `DevSet/`: Used as a validation dragon [Dry run]
- `EDA-&-Preprocessing/`: Includes the raw dataset and preprocessing notebook (Z-score scaling, feature logging).
- `Final Model Parameters.txt`: Final learned weights, bias, mean, and standard deviation.

---

## 📈 Performance Metrics

| Metric              | Train Set         | Test Set          | Scikit-learn Model   |
|---------------------|-------------------|--------------------|-----------------------|
| Mean Squared Error  | 8.9528            | **6.4085**         | 8.0725                |
| Mean Absolute Error | 2.1759            | **2.0322**         | —                     |
| R² Score            | 0.8550            | **0.8841**         | 0.8418                |

**Highlights**:
- My scratch implementation outperformed Scikit-learn on **unseen test data**.
- Training and test performance is consistent → no signs of overfitting or underfitting.
- Both models were trained on the **same cleaned, scaled, and log-transformed data**.
- **Z-score scaling** was applied using training-set mean and std, then used on the test set.

---

## 📐 Final Model Parameters

**Mean**:

[9.58466454e-03, 4.95207668e-01, 6.38977636e-03, 2.07667732e-01,
2.81150160e-01, 1.70894569e+01, 1.54829744e+01, 7.96781229e+00,
5.14426029e+00, 4.60003153e+00]

**Standard Deviation**:

[0.09743099, 0.49997703, 0.07968028, 0.40563758, 0.44956062,
3.68898508, 2.6940307, 0.28290347, 0.53879275, 0.35044875]

**Weights**:

[[-0.78103318], 
[ 0.10291997],
[ 0.13756557],
[-0.37696422],
[ 0.77636853],
[-2.60103245],
[-0.80347032],
[-2.70218615],
[-1.70772922],
[-2.63906737]]

**Bias**:

23.167092651754405

---

## 🚀 Live Demo

Try the model live on Hugging Face Spaces:

👉 [**Hugging Face App**]https://huggingface.co/spaces/FahaDragusss/MLR-scratch-streamlit

---

## 🧠 Notes and Next Steps

- This model will later be revisited for **polynomial regression** to enhance accuracy and generalizability.
- More models like Ridge, Lasso, ElasticNet, and Logistic Regression will follow in other folders (see parent directory).

---



