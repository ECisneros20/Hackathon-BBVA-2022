#!/usr/bin/env python3

from joblib import load
from pydantic import BaseModel
from pandas import DataFrame
import os
from io import BytesIO

def get_model():
    model_path = os.environ.get('MODEL_PATH', 'models/random_forest.joblib')
    with open(model_path, 'rb') as model_file:
        model = load(BytesIO(model_file.read()))
    return model

def transform_to_dataframe(class_model: BaseModel):
    transition_dictionary = {key: [value] for key, value in class_model.dict().items()}
    print(transition_dictionary)
    data_frame = DataFrame(transition_dictionary)
    return data_frame
