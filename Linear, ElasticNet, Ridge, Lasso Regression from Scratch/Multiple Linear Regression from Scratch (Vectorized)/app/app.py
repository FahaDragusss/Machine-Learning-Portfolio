# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

w = model["w"]
b = model["b"]
mean = model["mean"]
std = model["std"]

# Define input structure
class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    x = np.array(data.features).reshape(1, -1)
    x_scaled = (x - mean) / std
    prediction = np.dot(x_scaled, w) + b
    return {"prediction": float(prediction[0][0])}
