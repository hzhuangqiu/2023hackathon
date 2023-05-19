import modin.pandas as pd
from modin.config import Engine, StorageFormat

Engine.put("dask")
StorageFormat.put("pandas")

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

if __name__ == "__main__":
    data = pd.read_csv("dataset.csv")
    print(f"Data shape: {data.shape}")

    data["Source"].fillna("NA", inplace=True)
    data["Color"].fillna("NA", inplace=True)
    data["Month"].fillna("NA", inplace=True)

    encoder = LabelEncoder()
    data["Source"] = encoder.fit_transform(data["Source"].astype(str))
    data["Color"] = encoder.fit_transform(data["Color"].astype(str))
    data["Month"] = encoder.fit_transform(data["Month"].astype(str))

    for feature in data.columns:
        data[feature].fillna(data[feature].mean(), inplace=True)
    print("Preprocessing finished")

    data = data.drop(["Index", "Month", "Day", "Time of Day", "Source"], axis=1)
    X = data.drop("Target", axis=1)
    target = data["Target"]
    X_train, X_test, target_train, target_test = train_test_split(
        X, target, test_size=0.33, random_state=42, stratify=target
    )

    print("Start training...")
    lgbm_model = LGBMClassifier(verbosity=-1)
    begin = time.time()
    lgbm_model.fit(
        X_train,
        target_train,
        eval_set=[(X_test, target_test)],
        early_stopping_rounds=30,
        eval_metric="f1",
    )
    end = time.time()
    print(f"Elapsed time: {end - begin:.2f}s")

    print("Start inferring (Normal)...")
    begin = time.time()
    lgbm_model_prediction = lgbm_model.predict(X_test)
    end = time.time()
    print(f"Inference time (Normal): {(end - begin) / 6.0:.2f}s")
    print(
        classification_report(
            target_test,
            lgbm_model_prediction,
            target_names=["Safe to drink", "Not safe to drink"],
        )
    )

    daal_lgbm_model = d4p.get_gbt_model_from_lightgbm(lgbm_model.booster_)

    print("Start inferring (oneAPI)...")
    begin = time.time()
    daal_lgbm_model_prediction = (
        d4p.gbt_classification_prediction(nClasses=2)
        .compute(X_test, daal_lgbm_model)
        .prediction
    )
    end = time.time()
    print(f"Inference time (oneAPI): {(end - begin) / 6.0:.2f}s")
    print(
        classification_report(
            target_test,
            daal_lgbm_model_prediction,
            target_names=["Safe to drink", "Not safe to drink"],
        )
    )
