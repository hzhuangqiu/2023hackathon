import pandas as pd

from sklearnex import patch_sklearn

patch_sklearn()

import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
import daal4py as d4p

import warnings

warnings.filterwarnings("ignore")

import os

if __name__ == "__main__":
    if os.path.exists("preprocessed_dataset.csv"):
        print("Use existing preprocessed data")
        print("Read preprocessed data...")
        data = pd.read_csv("preprocessed_dataset.csv")
    else:
        print("Read raw data...")
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

        print("Write to preprocessed_dataset.csv...")
        data.to_csv("preprocessed_dataset.csv", index=False)

    print("Split the dataset into training and testing sets...")
    data = data.drop(["Index", "Month", "Day", "Time of Day", "Source"], axis=1)
    X = data.drop("Target", axis=1)
    target = data["Target"]
    X_train, X_test, target_train, target_test = train_test_split(
        X, target, test_size=0.33, random_state=42, stratify=target
    )

    print("Start training...")
    lgbm_model = LGBMClassifier(
        objective="binary",
        n_estimators=10000,
        learning_rate=0.16806501046153502,
        num_leaves=2280,
        max_depth=12,
        min_data_in_leaf=3900,
        lambda_l1=25,
        lambda_l2=60,
        min_gain_to_split=0.3443081671621193,
        bagging_fraction=0.9,
        bagging_freq=1,
        feature_fraction=0.9,
        verbosity=-1,
    )
    begin = time.time()
    lgbm_model.fit(
        X_train,
        target_train,
        eval_set=[(X_test, target_test)],
        eval_metric="binary_logloss",
        early_stopping_rounds=100,
    )
    end = time.time()
    print(f"Elapsed time: {end - begin:.2f}s")

    print("Start inferring (Normal)...")
    begin = time.time()
    lgbm_model_prediction = lgbm_model.predict(X_test)
    end = time.time()
    print(f"Inference time (Normal): {end - begin:.2f}s")
    print(
        classification_report(
            target_test,
            lgbm_model_prediction,
            target_names=["Safe to drink", "Not safe to drink"],
        )
    )

    print("Convert the trained model to oneAPI...")
    daal_lgbm_model = d4p.get_gbt_model_from_lightgbm(lgbm_model.booster_)

    print("Start inferring (oneAPI)...")
    begin = time.time()
    daal_lgbm_model_prediction = (
        d4p.gbt_classification_prediction(nClasses=2)
        .compute(X_test, daal_lgbm_model)
        .prediction
    )
    end = time.time()
    print(f"Inference time (oneAPI): {end - begin:.2f}s")
    print(
        classification_report(
            target_test,
            daal_lgbm_model_prediction,
            target_names=["Safe to drink", "Not safe to drink"],
        )
    )
