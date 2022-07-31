from fastapi import FastAPI, File, Form, UploadFile
import uvicorn
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
    
@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}
    

@app.post('/predict')
async def clasificadores(variables_entrada: Kmeans):
    data = variables_entrada.dict()
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    data_in = [[data['edad'], data['salario'], data['score']]]
    prediction = loaded_model.predict(data_in)
    
   
            
    return int(prediction)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)
