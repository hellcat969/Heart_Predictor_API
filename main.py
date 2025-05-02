from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum
import pandas as pd
import pickle
import logging
import numpy as np
import os

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS
origins = ["https://aethermedix.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class HeartAttackPrediction(BaseModel):
    Age: int
    Sex: int
    BP: int
    Cholesterol: int
    FBS_over_120: int
    Max_HR: int
    Exercise_angina: int
    ST_depression: float

# Load model once per cold start
model_path = os.path.join(os.path.dirname(__file__), "..", "heart_attack_model2.pkl")
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None  # Prevent crash; will raise on prediction

@app.get("/")
def read_root():
    return {"message": "Heart Predictor API is live."}

@app.post("/predict")
def predict_risk(features: HeartAttackPrediction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    try:
        data = pd.DataFrame([features.dict()])
        prediction = model.predict(data)
        result = prediction[0]
        if isinstance(result, np.generic):
            result = result.item()
        return {"risk_factor": result}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail="Prediction failed.")

# Needed for Vercel
handler = Mangum(app)
