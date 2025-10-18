#  0.50 3500 1 100 1
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from joblib import dump, load
import uvicorn

class InputData(BaseModel):
    x1 : float
    x2 : float
    x3 : float
    x4 : float
    x5 : float

scaler = joblib.load(r"C:\Users\ddevi\Documents\Python Projects\LLM_langchain_Py3.10\ML_EDA_practice\Loadapproval\Modelpicklefile\scaler.pkl")
model = joblib.load(r"C:\Users\ddevi\Documents\Python Projects\LLM_langchain_Py3.10\ML_EDA_practice\Loadapproval\Modelpicklefile\svcmodel.pkl")

app =FastAPI()

@app.post("/predict/")
def predict(input_data : InputData):
    x_values = np.array([[
        input_data.x1,
        input_data.x2,
        input_data.x3,
        input_data.x4,
        input_data.x5
    ]])

    scaled_x_values = scaler.transform(x_values)

    prediction = model.predict(scaled_x_values)

    prediction = int(prediction[0])

    return {"predictions": prediction[0]}

if __name__=="__main__":
    uvicorn.run(app, host ="127.0.0.1", port=8000)