import streamlit as st
import pickle

import os
import time

import numpy as np

import daal4py as d4p

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))

def not_implement_error():
    st.error('not implement yet!')

def backend_load_model(model_name, model_backend):
    oneAPI_suffix = '_oneAPI'
    model_name_pattern = parent_path + '/' + '{}' + '.pkl'
    if model_backend == 'oneAPI':
        model_name = model_name_pattern.format(model_name+oneAPI_suffix)
    elif model_backend == 'Sklearn':
        model_name = model_name_pattern.format(model_name)
    else:
        return None
    try:
        with open(model_name, 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        return None

def backend_predict(model, model_backend, data):
    begin = time.time()
    if model_backend == 'oneAPI':
        prediction = (d4p.gbt_classification_prediction(nClasses=2)
            .compute(data, model)
            .prediction)
    else:
        prediction = model.predict(data)
    end = time.time()
    quality_map = {0: 'Not safe to drink', 1: 'Safe to drink'}
    result = np.vectorize(quality_map.get)(prediction.flatten())
    duration = (end - begin) * 1000

    return duration, result

def backend_train(model, data):
    return not_implement_error()
