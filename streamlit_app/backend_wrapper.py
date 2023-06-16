import streamlit as st
import pickle

import os
import time

import numpy as np

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))

def not_implement_error():
    st.error('not implement yet!')

def backend_load_model(model_name):
    # TODO: add oneAPI model
    try:
        with open(parent_path+'/'+model_name+'.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        return None

def backend_predict(model, data):
    begin = time.time()
    prediction = model.predict(data)
    end = time.time()
    quality_map = {0: 'Not safe to drink', 1: 'Safe to drink'}
    result = np.vectorize(quality_map.get)(prediction)

    return (end-begin)*1000, result

def backend_train(model, data):
    return not_implement_error()
