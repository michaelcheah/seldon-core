import dataclasses
from typing import List, Dict, Tuple, Callable, Union
import numpy as np
import pandas as pd
from alibi.explainers import KernelShap
import shap
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from importlib.metadata import version
from numpy import typing as npt
from pathlib import Path
from functools import partial
import json
import joblib
from sklearn.datasets import load_diabetes


@dataclasses.dataclass
class Data:
    X_train: npt.NDArray
    y_train: npt.NDArray
    X_test: npt.NDArray
    y_test: npt.NDArray
    feature_name: List[str]


def get_model_folder(*args) -> Path:
    model_folder = partial(Path, Path(__file__).parent)(*args)
    model_folder.mkdir(parents=True, exist_ok=True)
    return model_folder


PredictorType = Union[LinearRegression]
ExplainerType = Union[KernelShap]
PredictorFunc = Callable[[Data], Tuple[PredictorType, Dict]]
ExplainerFunc = Callable[[PredictorType, Data], Tuple[ExplainerType, Dict]]

PREDICTOR_PATH = get_model_folder("income", "predictors")
EXPLAINER_PATH = get_model_folder("income", "explainers")


def fetch_diabetes_data() -> Data:
    np.random.seed(0)
    # prepare data
    X, y = shap.datasets.diabetes()
    X_train: npt.NDArray; X_test: npt.NDArray; y_train: npt.NDArray; y_test: npt.NDArray
    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), np.array(y), test_size=0.2, random_state=0)

    X_train_summary = shap.kmeans(X_train, 10)
    print(X_train_summary)

    return Data(X_train, y_train, X_test, y_test, X.columns.to_list())


def make_diabetes_linear_regression(data: Data) -> Tuple[PredictorType, Dict]:
    lin_regr = LinearRegression()
    lin_regr.fit(data.X_train, data.y_train)

    model_settings = {
        "name": "diabetes_linear_regression",
        "implementation": "mlserver_sklearn.SKLearnModel",
        "parameters": {
            "uri": "./model.joblib",
            "version": "v0.1.0"
        }
    }

    return lin_regr, model_settings


def make_diabetes_kernel_shap(predictor: PredictorType, data: Data) -> Tuple[KernelShap, Dict]:
    explainer = KernelShap(
        predictor.predict,
        "identity",
        data.feature_name,
        seed=1
    )
    explainer.fit(data.X_train[:100])
    model_settings = {
        "name": "diabetes_kernel_shap",
        "implementation": "mlserver_alibi_explain.AlibiExplainRuntime",
        "parameters": {
            "uri": "./data",
            "version": "v0.1.0",
            "extra": {
                "infer_uri": "http://localhost:8080/v2/models/diabetes_linear_regression/infer",
                "explainer_type": "kernel_shap",
                "explainer_batch": True,
                "infer_output": "predict"
            }
        }
    }
    return explainer, model_settings


def save_dict(path: Path, model_settings: Dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(model_settings, f, ensure_ascii=False, indent=4)


def create(data: Data, predictor_func: PredictorFunc, explainer_funcs: List[ExplainerFunc],
           classifier_folder=PREDICTOR_PATH, explainer_folder=EXPLAINER_PATH):
    model_paths = []

    clf, clf_model_settings = predictor_func(data)
    score = clf.score(data.X_test, data.y_test)
    print("Predictor score: ", score)

    clf_name = clf_model_settings["name"]
    clf_save_path = get_model_folder(classifier_folder / clf_name)

    print(f"Saving predictor and model settings in {clf_save_path}")
    joblib.dump(clf, clf_save_path / "model.joblib")
    save_dict(clf_save_path / "model-settings.json", clf_model_settings)
    model_paths.append(clf_save_path)

    for explainer_func in explainer_funcs:
        explainer, explainer_model_settings = explainer_func(clf, data)

        explainer_name = explainer_model_settings["name"]
        explainer_save_path = get_model_folder(explainer_folder / explainer_name)

        print(f"Saving explainer and model settings in {explainer_save_path}")
        explainer.save(explainer_save_path / "data")
        save_dict(explainer_save_path / "model-settings.json", explainer_model_settings)
        model_paths.append(explainer_save_path)

    for model_path in model_paths:
        save_dict(model_path / "settings.json", {"debug": True})

    print("Example request: ", {
        "inputs": [
            {
                "name": "income",
                "data": [int(i) for i in data.X_test[0]],
                "datatype": "INT64",
                "shape": [1, len(data.X_test[0])]
            }
        ]
    })


def train():
    diabetes_data = fetch_diabetes_data()
    print(diabetes_data.X_test[0:1])

    # classifier = joblib.load(CLASSIFIER_PATH / "model.joblib")

    # Full Linear Regression with Kernel Shap
    create(diabetes_data, make_diabetes_linear_regression, [make_diabetes_kernel_shap])


if __name__ == "__main__":
    train()
