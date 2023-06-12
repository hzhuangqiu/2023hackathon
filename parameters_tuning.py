import pandas as pd

from sklearnex import patch_sklearn

patch_sklearn()

import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

import optuna
from optuna.integration import LightGBMPruningCallback

from lightgbm import LGBMClassifier

import warnings

warnings.filterwarnings("ignore")

import os

import numpy as np


def preprocesse_data(raw_data_path: str, dest_data_path: str) -> pd.DataFrame:
    if os.path.exists(dest_data_path):
        print(f"Use existing preprocessed data at {dest_data_path}")
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


def objective(trial, X, y):
    param_grid = {
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.2, 0.95, step=0.1
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=114514)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LGBMClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            callbacks=[LightGBMPruningCallback(trial, "binary_logloss")],
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

    return np.mean(cv_scores)


if __name__ == "__main__":
    data = preprocesse_data("dataset.csv", "preprocessed_dataset.csv")

    data = data.drop(["Index", "Month", "Day", "Time of Day", "Source"], axis=1)
    X = data.drop("Target", axis=1)
    y = data["Target"]

    study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
    func = lambda trial: objective(trial, X, y)
    study.optimize(func, n_trials=100)

    print(f"Best value (binary_logloss): {study.best_value:.5f}")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"\t{key}: {value}")
