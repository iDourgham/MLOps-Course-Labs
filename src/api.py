from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
from typing import List



# Load the model and transformer
try:
    model = joblib.load("model/model.pkl")
    transformer = joblib.load("column_transformer.pkl")
    logging.info("Model and transformer loaded successfully.")
except Exception as e:
    logging.exception("Failed to load model or transformer.")
    raise e

# FastAPI app
app = FastAPI()

class ChurnInput(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def home():
    logging.info("Root endpoint called.")
    return {"message": "Welcome to the Churn Prediction API"}

@app.get("/health")
def health():
    logging.info("Health check endpoint called.")
    return {"status": "API is healthy"}

@app.post("/predict")
def predict(data: List[ChurnInput]):
    logging.info("Prediction endpoint called.")

    try:
        df = pd.DataFrame([d.model_dump() for d in data])
        logging.info(f"Received input data: {df.to_dict(orient='records')}")

        X = transformer.transform(df)
        prediction = model.predict(X)

        logging.info(f"Prediction result: {prediction.tolist()}")
        return {"prediction": prediction.tolist()}

    except Exception as e:
        logging.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail=str(e))
