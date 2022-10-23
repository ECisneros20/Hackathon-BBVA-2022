#!usr/bin/env python3

from fastapi import FastAPI
from .app.models import PredictionResponse, PredictionRequest
from .app.views import get_prediction

app = FastAPI(docs_url='/')

@app.post('/v1/prediction')
def make_model_prediction(request: PredictionRequest):
    return PredictionResponse(price=get_prediction(request))

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
