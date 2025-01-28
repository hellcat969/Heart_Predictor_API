import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class HeartAttackPrediction(BaseModel): 
    Age: int
    Sex: int
    Cholesterol: int
    Diabetes: int
    Smoking: int
    Obesity: int
    Alcoholic: int
    EHPW: int
    SHPD: int

# Load the model
try:
    with open('heart_attack_model.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    logger.error("Model file not found")
    raise HTTPException(status_code=500, detail="Model file not found")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise HTTPException(status_code=500, detail="Error loading model")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Predictor API"}

@app.post("/predict")
def predict_risk(features: HeartAttackPrediction):
    try:
        # Log the received features
        logger.info(f"Received features: {features}")
        
        # Convert input to DataFrame
        data = pd.DataFrame([features.dict()])
        
        # Predict using the loaded model
        prediction = model.predict(data)
        result = prediction[0]
        
        # Ensure the result is converted to a native Python type
        if isinstance(result, np.generic):
            result = result.item()
        
        logger.info(f"Predicted risk factor: {result}")
        return {"risk_factor": result}
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
