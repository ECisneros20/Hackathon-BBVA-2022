#!/usr/bin/env python3

from .models import PredictionRequest
from .utils import get_model, transform_to_dataframe
from src.prepare import prepareData
model = get_model()

def get_prediction(request: PredictionRequest) -> float:
    data_to_predict = transform_to_dataframe(request)
    print("SOY UN DATAFRAME")
    print(data_to_predict)
    data_to_predict = prepareData(data_to_predict)
    # print("preparing data...")
    # print(data_to_predict.columns)

    prediction = model.predict(data_to_predict)[0]
    return max(0, prediction)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
