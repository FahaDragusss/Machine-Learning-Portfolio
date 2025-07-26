import numpy as np
import joblib

# Load model components from .npy files
w = np.load("weights_poly_trad.npy")
b = np.load("bias_poly_trad.npy").item()  # assuming bias is saved as a scalar array
mean = np.load("mean_poly_trad.npy")
std = np.load("std_poly_trad.npy")

# Define the model dictionary
model = {
    "w": w,
    "b": b,
    "mean": mean,
    "std": std
}

# Save the model to a file using joblib
joblib.dump(model, "model.joblib")

print("✅ model.joblib has been saved successfully.")
