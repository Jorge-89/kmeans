from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from io import StringIO

app = FastAPI()

class Kmeans(BaseModel):
    edad: float 
    salario: float 
    score: float 
    

@app.post('/predict')
async def clasificadores(variables_entrada: Kmeans):
    data = variables_entrada.dict()
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    data_in = [[data['edad'], data['salario'], data['score']]]
    prediction = loaded_model.predict(data_in)
    
   
            
    return int(prediction)

