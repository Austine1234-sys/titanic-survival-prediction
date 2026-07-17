# Live Demo
Try the app here:https://austine-akhenda-titanic-predictor.streamlit.app/


# titanic-survival-prediction
#Titanic survival prediction model with FastAPI and Streamlit deployment
# Titanic Survival Prediction

## Description
A machine learning project predicting Titanic passenger survival using a tuned Random Forest classifier, deployed as both a REST API and an interactive web app.

## Problem Statement
Predict whether a passenger survived the Titanic disaster based on class, sex, age, family size, and fare — using historical passenger data.

## Model Used
Random Forest Classifier, tuned via RandomizedSearchCV (5-fold cross-validation). Cross-validated accuracy: 83.17%. Features engineered: Title (from name), FamilySize, IsAlone, AgeGroup.

## Technologies Used
Python, pandas, scikit-learn, FastAPI, Streamlit, joblib

## API Endpoints
POST /predict
Request body:
```json
{
  "Pclass": 1,
  "Sex_male": 0,
  "Age": 28,
  "SibSp": 0,
  "Fare": 80,
  "FamilySize": 1,
  "Title_Mr": 0,
  "Title_Miss": 1
}
