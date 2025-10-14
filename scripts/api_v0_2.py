# pylint: disable=duplicate-code

"""

API version 0.2 that get model health check and predict

"""

import os
import pickle
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import numpy as np

app = FastAPI()

@app.get('/', include_in_schema=False)

def read_root():

    """
    
    Return FastAPI GUI /docs

    """

    return RedirectResponse(url='/docs', status_code=status.HTTP_302_FOUND)

@app.get('/health')
def health_check():

    """
    
    Health check for API
    
    """

    return {"status": "ok", "version": "v0.1"}

class DiabetesFeatures(BaseModel):

    """
    
    Class for JSON parse
    
    """

    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

@app.post('/predict')
def predict(features: DiabetesFeatures):

    """
    
    API for prediction
    :param: features: JSON contains 10 features (Age, Sex, BMI, BP, s1, s2, s3, s4, s5, s6)
    
    """

    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     '../models/model_diabetes_v0.2.pkl'))

    scaler_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     '../models/scaler_diabetes_v0.2.pkl'))

    selector_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     '../models/selector_diabetes_v0.2.pkl'))

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail='Model file not found')

    if not os.path.exists(scaler_path):
        raise HTTPException(status_code=404, detail='Scaler file not found')

    if not os.path.exists(scaler_path):
        raise HTTPException(status_code=404, detail='Selector file not found')

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    with open(selector_path, 'rb') as file:
        selector = pickle.load(file)

    feature = np.array([[
                    features.age,
                    features.sex,
                    features.bmi,
                    features.bp,
                    features.s1,
                    features.s2,
                    features.s3,
                    features.s4,
                    features.s5,
                    features.s6
                    ]])
    feature_scaled = scaler.transform(feature)
    feature_selected = selector.transform(feature_scaled)
    prediction = model.predict(feature_selected)
    return {'prediction': prediction[0]}
