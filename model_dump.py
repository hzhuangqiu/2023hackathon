import pandas as pd

from sklearnex import patch_sklearn
from typing import Any
from collections.abc import Callable

patch_sklearn()

import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
import daal4py as d4p

import warnings

import pickle

warnings.filterwarnings("ignore")

import os
import json

model_name_map = {"lightgbm": "LGBM", "xgboost": "XGBoost", "catboost": "CatBoost"}

def preprocess_data(raw_data_path: str, dest_data_path: str) -> pd.DataFrame:
    if os.path.exists(dest_data_path):
        print(f"Use existing preprocessed data at {dest_data_path}.")
        print(f"Read preprocessed data from {dest_data_path}...")
        return pd.read_csv(dest_data_path)
    else:
        print(f"Read raw data from {raw_data_path}")
        data = pd.read_csv("dataset.csv")
        print(f"Start preprocessing (data shape: {data.shape})...")

        data["Source"].fillna("NA", inplace=True)
        data["Color"].fillna("NA", inplace=True)
        data["Month"].fillna("NA", inplace=True)

        encoder = LabelEncoder()
        data["Source"] = encoder.fit_transform(data["Source"].astype(str))
        data["Color"] = encoder.fit_transform(data["Color"].astype(str))
        data["Month"] = encoder.fit_transform(data["Month"].astype(str))

        data.fillna(data.mean(), inplace=True)

        print(f"Write to {dest_data_path}...")
        data.to_csv(dest_data_path, index=False)
        return data

def get_model(model_type: str) -> Any:
    try:
        with open(model_type + "_best_params.json", "r") as f:
            params = json.load(f)
        if model_type == "lightgbm":
            return LGBMClassifier(
                objective="binary",
                verbosity=-1,
                eval_metric="binary_logloss",
                early_stopping_rounds=100,
                **params
            )
        elif model_type == "xgboost":
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                early_stopping_rounds=30,
                learning_rate=1.0,
                **params
            )
        elif model_type == "catboost":
            return CatBoostClassifier(
                objective="Logloss",
                eval_metric="Logloss",
                early_stopping_rounds=30,
                learning_rate=1.0,
                **params
            )
    except FileNotFoundError:
        raise KeyError(f"Unknow Model Type: {model_type}")

def convert_to_d4p(model: Any, model_type: str) -> Any:
    if model_type == "lightgbm":
        return d4p.get_gbt_model_from_lightgbm(model.booster_)
    elif model_type == "xgboost":
        return d4p.get_gbt_model_from_xgboost(model.get_booster())
    elif model_type == "catboost":
        return d4p.get_gbt_model_from_catboost(model)
    else:
        raise KeyError(f"Unknow Model Type: {model_type}")


def save_onedal_model(model, file="native_binary.txt"):
    if os.path.exists(file):
        print(f"The oneDAL model file {file} already exists, nothing to do.")
    else:
        print(f"Write the oneDAL model to {file} (required by inference.cpp)...")
        daal_buff = model.__getstate__()
        with open(file, "wb") as f:
            f.write(daal_buff)

def save_model(model, file: str):
    if os.path.exists(file):
        print(f"The model file {file} already exists, nothing to do.")
    else:
        print(f"Write the model to {file}...")
        with open(file, "wb") as f:
            pickle.dump(model, f)


def save_test_data(X_test, y_test, file="test_dataset.csv"):
    if os.path.exists(file):
        print(f"The test data {file} already exists, nothing to do.")
    else:
        print(f"Write test data to {file} (required by inference.cpp)...")
        X_test["Target"] = y_test
        X_test.to_csv(file, header=False, index=False)


if __name__ == "__main__":
    data = preprocess_data("dataset.csv", "preprocessed_dataset.csv")

    print("Split the dataset into training and testing sets...")
    data = data.drop(["Index", "Month", "Day", "Time of Day", "Source"], axis=1)
    X = data.drop("Target", axis=1)
    target = data["Target"]
    X_train, X_test, target_train, target_test = train_test_split(
        X, target, test_size=0.33, random_state=42, stratify=target
    )

    for model_type in ["lightgbm", "xgboost", "catboost"]:
        print(f"Start training ({model_type})...")
        begin = time.time()
        model = get_model(model_type)
        model.fit(
            X_train,
            target_train,
            eval_set=[(X_test, target_test)],
        )
        end = time.time()
        print(f"Elapsed time: {end - begin:.2f}s")

        print(f"Start inferring ({model_type} && Normal)...")
        begin = time.time()
        model_prediction = model.predict(X_test)
        end = time.time()
        print(f"Inference time ({model_type} && Normal): {end - begin:.3f}s")
        print(
            classification_report(
                target_test,
                model_prediction,
                target_names=["Safe to drink", "Not safe to drink"],
            )
        )

        save_model(model, model_name_map[model_type] + ".pkl")

        print("Convert the trained model to oneAPI...")
        daal_model = convert_to_d4p(model, model_type)

        print(f"Start inferring ({model_type} && oneAPI)...")
        begin = time.time()
        daal_model_prediction = (
            d4p.gbt_classification_prediction(nClasses=2)
            .compute(X_test, daal_model)
            .prediction
        )
        end = time.time()
        print(f"Inference time ({model_type} && oneAPI): {end - begin:.3f}s")
        print(
            classification_report(
                target_test,
                daal_model_prediction,
                target_names=["Safe to drink", "Not safe to drink"],
            )
        )
        save_model(daal_model, model_name_map[model_type] + "_oneAPI.pkl")
        save_onedal_model(daal_model, file=f"native_binary_{model_type}.txt")

    save_test_data(X_test, target_test)
