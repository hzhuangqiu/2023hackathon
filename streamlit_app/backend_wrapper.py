import streamlit as st
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
import daal4py as d4p

import sys
import os

import numpy as np

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

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
    prediction = model.predict(data)
    quality_map = {0: 'Not safe to drink', 1: 'Safe to drink'}
    result = np.vectorize(quality_map.get)(prediction)

    return result

def backend_train(model, data):
    return not_implement_error()
