from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json
import os
from pydantic import BaseModel
from typing_extensions import Annotated,List,Literal,TypedDict
import pandas as pd
import pickle
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel





models = "/model/car_price.pkl"

with open(models, "rb") as f:
    model = pickle.load(f)


class InputRes(BaseModel):
    name: str
    company: str
    fuel_type: str
    kms_driven: int
    year: int


app = FastAPI()


@app.post("/predict")
def predict(data: InputRes):

    input_data = pd.DataFrame([
        {
            "name": data.name,
            "company": data.company,
            "fuel_type": data.fuel_type,
            "kms_driven": data.kms_driven,
            "year": data.year
        }
    ])

    response = model.predict(input_data)

    return {"prediction": float(response[0])}