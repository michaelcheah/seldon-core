import dataclasses
from typing import List, Dict, Tuple, Callable, Union
import numpy as np
from alibi.explainers import KernelShap, TreeShap
from alibi.datasets import fetch_adult
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


@dataclasses.dataclass
class Data:
    X_train: npt.NDArray
    Y_train: npt.NDArray
    X_test: npt.NDArray
    Y_test: npt.NDArray
    feature_name: List[str]
    category_map: Dict[str, List[str]]


def get_model_folder(*args) -> Path:
    model_folder = partial(Path, Path(__file__).parent)(*args)
    model_folder.mkdir(parents=True, exist_ok=True)
    return model_folder


PredictorType = Union[RandomForestClassifier, Pipeline]
ExplainerType = Union[KernelShap, TreeShap]
PredictorFunc = Callable[[Data], Tuple[PredictorType, Dict]]
ExplainerFunc = Callable[[PredictorType, Data], Tuple[ExplainerType, Dict]]

PREDICTOR_PATH = get_model_folder("income", "predictors")
EXPLAINER_PATH = get_model_folder("income", "explainers")


def fetch_income_data() -> Data:
    np.random.seed(0)
    # prepare data
    adult = fetch_adult()
    data = adult.data
    target = adult.target
    feature_names = adult.feature_names
    category_map = adult.category_map

    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=0)

    return Data(X_train, Y_train, X_test, Y_test, feature_names, category_map)


def fetch_income_categorical_data() -> Data:
    np.random.seed(0)
    # fetch adult dataset
    adult = fetch_adult()

    # select categorical columns
    categorical_ids = list(adult.category_map.keys())

    # redefine the feature names list and the categorical mapping
    feature_names = [adult.feature_names[i] for i in categorical_ids]
    category_map = {i: x for i, x in enumerate(adult.category_map.values())}

    # drop the numerical columns
    X, Y = adult.data[:, categorical_ids], adult.target
    # split the dataset into train-test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # define one-hot encoding preprocessor
    preprocessor = OneHotEncoder(
        categories=[range(len(x)) for x in category_map.values()],
        handle_unknown="ignore"
    )

    # fit the OHE preprocessor on the training dataset
    preprocessor.fit(X_train)

    # transform train and test to one-hot-encoding
    X_train_ohe = preprocessor.transform(X_train).toarray()
    X_test_ohe = preprocessor.transform(X_test).toarray()

    return Data(X_train_ohe, Y_train, X_test_ohe, Y_test, feature_names, category_map)


def make_income_random_forest(data: Data) -> Tuple[Pipeline, Dict]:
    # adapted from:
    # https://docs.seldon.io/projects/alibi/en/latest/examples/anchor_tabular_adult.html

    ordinal_features = [
        x for x in range(len(data.feature_name)) if x not in list(data.category_map.keys())
    ]
    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_features = list(data.category_map.keys())
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", ordinal_transformer, ordinal_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = RandomForestClassifier(n_estimators=50)

    model_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", clf),
        ]
    )

    model_pipeline.fit(data.X_train, data.Y_train)

    model_settings = {
        "name": "classifier",
        "implementation": "mlserver_sklearn.SKLearnModel",
        "parameters": {
            "uri": "./model.joblib",
            "version": "v0.1.0"
        }
    }

    return model_pipeline, model_settings


def make_income_random_forest_ohe(data: Data) -> Tuple[RandomForestClassifier, Dict]:
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(data.X_train, data.Y_train)

    model_settings = {
        "name": "classifier_ohe",
        "implementation": "mlserver_sklearn.SKLearnModel",
        "parameters": {
            "uri": "./model.joblib",
            "version": "v0.1.0"
        }
    }

    return clf, model_settings


def make_income_kernel_shap(pipeline: Pipeline, data: Data) -> Tuple[KernelShap, Dict]:
    explainer = KernelShap(
        pipeline.predict_proba,
        "logit",
        data.feature_name,
        categorical_names=data.category_map,
        seed=1
    )

    explainer.fit(data.X_train[:100, :])

    model_settings = {
        "name": "kernel_shap_explainer",
        "implementation": "mlserver_alibi_explain.AlibiExplainRuntime",
        "parameters": {
            "uri": "./data",
            "version": "v0.1.0",
            "extra": {
                "infer_uri": "http://localhost:8080/v2/models/classifier/infer",
                "explainer_type": "kernel_shap",
                "explainer_batch": True,
                "infer_output": "predict_proba"
            }
        }
    }

    return explainer, model_settings


def make_income_grouped_kernel_shap(clf: RandomForestClassifier, data: Data) -> Tuple[KernelShap, Dict]:
    # define prediction function for ohe representation
    def predict_fn(x_ohe):
        out = clf.predict_proba(x_ohe)
        return np.clip(out[:, 1], a_min=0.01, a_max=0.99)

    explainer = KernelShap(
        predict_fn,
        "logit",
        data.feature_name,
        categorical_names=data.category_map,
        seed=1
    )

    # compute starting index list and encoding dimension list
    cat_vars_enc_dim = [len(x) for x in data.category_map.values()]
    cat_vars_start_idx = list(np.cumsum([0] + cat_vars_enc_dim[:-1]))

    groups = [list(range(s, s + e)) for s, e in zip(cat_vars_start_idx, cat_vars_enc_dim)]

    explainer.fit(data.X_train[:100], group_names=data.feature_name, groups=groups)

    model_settings = {
        "name": "grouped_kernel_shap_explainer",
        "implementation": "mlserver_alibi_explain.AlibiExplainRuntime",
        "parameters": {
            "uri": "./data",
            "version": "v0.1.0",
            "extra": {
                "infer_uri": "http://localhost:8080/v2/models/classifier_ohe/infer",
                "explainer_type": "kernel_shap",
                "explainer_batch": True,
                "infer_output": "predict_proba"
            }
        }
    }

    return explainer, model_settings


def make_income_tree_shap(clf: RandomForestClassifier, data: Data) -> Tuple[KernelShap, Dict]:
    explainer = TreeShap(
        clf,
        "identity",
        data.feature_name,
        categorical_names=data.category_map,
        seed=1
    )

    # compute starting index list and encoding dimension list
    cat_vars_enc_dim = [len(x) for x in data.category_map.values()]
    cat_vars_start_idx = list(np.cumsum([0] + cat_vars_enc_dim[:-1]))

    groups = [list(range(s, s + e)) for s, e in zip(cat_vars_start_idx, cat_vars_enc_dim)]

    explainer.fit(data.X_train[:100], group_names=data.feature_name, groups=groups)

    model_settings = {
        "name": "tree_shap_explainer",
        "implementation": "mlserver_alibi_explain.AlibiExplainRuntime",
        "parameters": {
            "uri": "./data",
            "version": "v0.1.0",
            "extra": {
                "infer_uri": "http://localhost:8080/v2/models/classifier_ohe/infer",
                "explainer_type": "tree_shap",
                "explainer_batch": True,
                "infer_output": "predict_proba"
            }
        }
    }

    return explainer, model_settings


def save_dict(path: Path, model_settings: Dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(model_settings, f, ensure_ascii=False, indent=4)


def create(data: Data, classifier_func: PredictorFunc, explainer_funcs: List[ExplainerFunc],
           classifier_folder=PREDICTOR_PATH, explainer_folder=EXPLAINER_PATH):
    model_paths = []

    clf, clf_model_settings = classifier_func(data)
    score = clf.score(data.X_test, data.Y_test)
    print("Classifier score: ", score)

    clf_name = clf_model_settings["name"]
    clf_save_path = get_model_folder(classifier_folder / clf_name)

    print(f"Saving classifier and model settings in {clf_save_path}")
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
    full_data = fetch_income_data()
    categorical_data = fetch_income_categorical_data()
    print(full_data.X_test[0:1])

    # classifier = joblib.load(CLASSIFIER_PATH / "model.joblib")

    # Full Classifier with Kernel Shap
    # create(full_data, make_income_classifier, [make_income_kernel_shap])

    # Full Classifier with Kernel Shap
    create(categorical_data, make_income_random_forest_ohe, [make_income_grouped_kernel_shap, make_income_tree_shap])


if __name__ == "__main__":
    assert version("alibi") == "0.9.1", f"unexpected version {version('alibi')}"
    assert version("mlserver-alibi-explain") == "1.3.0.dev3", f"unexpected version {version('mlserver-alibi-explain')}"

    # a = fetch_categorical_data()
    train()
