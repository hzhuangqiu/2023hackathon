import pandas as pd

from sklearnex import patch_sklearn

patch_sklearn()

import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

import optuna
from optuna.integration import (
    LightGBMPruningCallback,
    XGBoostPruningCallback,
    CatBoostPruningCallback,
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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


def objective_lightgbm(trial, X, y):
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 2000, 8000, step=1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1000, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1000, log=True),
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

        model = LGBMClassifier(objective="binary", verbosity=-1, **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=50,
            callbacks=[LightGBMPruningCallback(trial, "binary_logloss")],
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

    return np.mean(cv_scores)


def objective_xgboost(trial, X, y):
    param_grid = {
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_int("gamma", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.01, 1.0, log=True),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.01, 1.0, log=True
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1000, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1000, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=100),
    }

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=114514)

    cv_scores = np.empty(2)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            objective="binary:logistic",
            verbosity=0,
            learning_rate=1.0,
            use_label_encoder=False,
            **param_grid,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="logloss",
            early_stopping_rounds=30,
            callbacks=[XGBoostPruningCallback(trial, "validation_0-logloss")],
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

    return np.mean(cv_scores)


def objective_catboost(trial, X, y):
    param_grid = {
        "depth": trial.suggest_int("depth", 1, 12),
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=100),
        "max_bin": trial.suggest_int("max_bin", 200, 400),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 300),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0, log=True),
    }

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=114514)

    cv_scores = np.empty(2)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = CatBoostClassifier(
            objective="Logloss", learning_rate=1.0, eval_metric="Logloss", **param_grid
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=30,
            callbacks=[CatBoostPruningCallback(trial, "Logloss")],
        )

        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

    return np.mean(cv_scores)


if __name__ == "__main__":
    data = preprocesse_data("dataset.csv", "preprocessed_dataset.csv")

    data = data.drop(["Index", "Month", "Day", "Time of Day", "Source"], axis=1)
    X = data.drop("Target", axis=1)
    y = data["Target"]

    print("Start hyperparameter tuning (lightgbm)...")
    study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
    func = lambda trial: objective_lightgbm(trial, X, y)
    study.optimize(func, n_trials=100)
    print(f"Best value (lightgbm && logloss): {study.best_value:.5f}")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"\t{key}={value}")

    print("Start hyperparameter tuning (xgboost)...")
    study = optuna.create_study(direction="minimize", study_name="XGB Classifier")
    func = lambda trial: objective_xgboost(trial, X, y)
    study.optimize(func, n_trials=40)
    print(f"Best value (xgboost && logloss): {study.best_value:.5f}")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"\t{key}={value}")

    print("Start hyperparameter tuning (catboost)...")
    study = optuna.create_study(direction="minimize", study_name="CB Classifier")
    func = lambda trial: objective_catboost(trial, X, y)
    study.optimize(func, n_trials=40)
    print(f"Best value (catboost && logloss): {study.best_value:.5f}")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"\t{key}={value}")
