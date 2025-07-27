from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import numpy as np
import time

# Initialize app
app = FastAPI()

# CORS setup to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model components on startup
model = joblib.load("backend/model.pkl")
scaler = joblib.load("backend/scaler.pkl")
pca = joblib.load("backend/pca.pkl")

# To store the latest prediction
latest_prediction_data = {"prediction": None, "probability": []}


@app.get("/")
def root():
    return {"message": "Model API is running."}


def run_predictions(df):
    global latest_prediction_data

    for i, row in df.iterrows():
        try:
            sample_df = pd.DataFrame([row])

            # === Feature Engineering ===
            sample_df["Coolant efficiency"] = sample_df["Coolant temp"] / (sample_df["Engine rpm"] + 1)
            sample_df["Lub oil efficiency"] = sample_df["lub oil temp"] / (sample_df["Engine rpm"] + 1)
            sample_df["Pressure coeff"] = sample_df['Fuel pressure'] * sample_df['Coolant pressure'] * sample_df['Lub oil pressure']
            sample_df["Temp coeff"] = sample_df['Coolant temp'] * sample_df['Coolant temp']
            sample_df["Coolant coeff"] = sample_df['Coolant temp'] * sample_df['Coolant pressure']
            sample_df["Lub oil coeff"] = sample_df['lub oil temp'] * sample_df['Lub oil pressure']
            sample_df["Temp effect"] = sample_df['Temp coeff'] / (sample_df['Engine rpm'] + 1)
            sample_df["Pressure effect"] = sample_df['Pressure coeff'] / (sample_df['Engine rpm'] + 1)
            sample_df["Thermal pressure index"] = (
                (sample_df['Coolant temp'] + sample_df['lub oil temp']) *
                (sample_df['Coolant pressure'] + sample_df['Lub oil pressure'])
            )

            # === Transform and Predict ===
            X_scaled = scaler.transform(sample_df)
            X_pca = pca.transform(X_scaled)

            prediction = model.predict(X_pca)[0]
            probability = model.predict_proba(X_pca)[0].tolist()

            latest_prediction_data = {
                "prediction": int(prediction),
                "probability": probability
            }

            time.sleep(1)  # simulate 1 second delay

        except Exception as e:
            latest_prediction_data = {
                "error": str(e)
            }
            break


@app.post("/predict")
async def start_prediction(background_tasks: BackgroundTasks):
    try:
        # Load a large CSV and take 10 random rows
        df = pd.read_csv("data/input.csv").sample(n=10, random_state=None).reset_index(drop=True)

        # Run predictions in background
        background_tasks.add_task(run_predictions, df)

        return {"message": "Prediction started. Poll /latest for updates every second."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/latest")
def get_latest_prediction():
    return latest_prediction_data
