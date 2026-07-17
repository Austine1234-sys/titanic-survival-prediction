from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Titanic Survival Prediction API")
model = joblib.load("titanic_model.joblib")
features = joblib.load("titanic_features.joblib")

class PassengerData(BaseModel):
    Pclass: int
    Sex_male: int
    Age: float
    SibSp: int
    Fare: float
    FamilySize: int
    Title_Mr: int
    Title_Miss: int

@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API is running"}

@app.post("/predict")
def predict(data: PassengerData):
    input_dict = data.dict()
    input_arr = np.array([[input_dict[f] for f in features]])
    prediction = model.predict(input_arr)[0]
    probability = model.predict_proba(input_arr)[0][1]
    return {"survived": int(prediction), "survival_probability": round(float(probability), 3)}