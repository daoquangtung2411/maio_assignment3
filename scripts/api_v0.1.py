from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
import pickle
import os
import numpy as np

app = FastAPI()

@app.get('/', include_in_schema=False)

def read_root():
    return RedirectResponse(url='/docs', status_code=status.HTTP_302_FOUND)

@app.post('/predict')
def predict(age:int, sex: int, bmi:float, bp:int,	s1:float, s2: float, s3:float, s4:float, s5:float,s6:float):
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/model_diabetes_v0.1.pkl'))
    scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/scaler_diabetes_v0.1.pkl'))
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail='Model file not found')

    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    X = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    return {'prediction': prediction[0]}