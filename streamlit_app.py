import streamlit as st
import joblib
import numpy as np

model = joblib.load("titanic_model.joblib")
features = joblib.load("titanic_features.joblib")

st.title("Titanic Survival Predictor")

pclass = st.selectbox("Passenger Class", [1,2,3])
sex_male = 1 if st.selectbox("Sex", ["male","female"]) == "male" else 0
age = st.slider("Age", 0, 100, 30)
sibsp = st.slider("Siblings/Spouses aboard", 0, 8, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
family_size = sibsp + 1
title_mr = 1 if sex_male == 1 else 0
title_miss = 0

if st.button("Predict"):
    input_dict = {"Pclass": pclass, "Sex_male": sex_male, "Age": age, "SibSp": sibsp,
                  "Fare": fare, "FamilySize": family_size, "Title_Mr": title_mr, "Title_Miss": title_miss}
    input_arr = np.array([[input_dict[f] for f in features]])
    pred = model.predict(input_arr)[0]
    prob = model.predict_proba(input_arr)[0][1]
    st.write("*Survived" if pred==1 else "Did not survive*")
    st.write(f"Survival probability: {prob:.1%}")