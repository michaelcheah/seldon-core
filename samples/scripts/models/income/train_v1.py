import dataclasses
from typing import List, Dict, Tuple, Callable, Union, Type, Optional
import numpy as np
from alibi.explainers import KernelShap, TreeShap
from alibi.datasets import fetch_adult
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from importlib.metadata import version
import shap
from numpy import typing as npt
from pathlib import Path
from functools import partial
import json
import joblib
from abc import abstractmethod


@dataclasses.dataclass
class Data:
    X_train: npt.NDArray
    y_train: npt.NDArray
    X_test: npt.NDArray
    y_test: npt.NDArray
    feature_name: Optional[List[str]]
    category_map: Optional[Dict[str, List[str]]] = None


def get_model_folder(*args) -> Path:
    model_folder = partial(Path, Path(__file__).parent)(*args)
    model_folder.mkdir(parents=True, exist_ok=True)
    return model_folder


def save_dict(path: Path, model_settings: Dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(model_settings, f, ensure_ascii=False, indent=4)


SupportedPredictorType = Union[RandomForestClassifier, Pipeline, LinearRegression, MultiOutputRegressor]
SupportedExplainerType = Union[KernelShap, TreeShap]
PredictorFunc = Callable[[Data], Tuple[SupportedPredictorType, Dict]]
ExplainerFunc = Callable[[SupportedPredictorType, Data], Tuple[SupportedExplainerType, Dict]]

PREDICTOR_PATH = get_model_folder("v1_models", "predictors")
EXPLAINER_PATH = get_model_folder("v1_models", "explainers")


class PredictorWrapper:
    predictor: SupportedPredictorType

    def __init__(self, data: Data):
        self.predictor = self.train(data)

    def get_save_path(self, base_path: Path = PREDICTOR_PATH) -> Path:
        return get_model_folder(base_path / self.get_name())

    def save(self, base_path: Path = PREDICTOR_PATH):
        save_path = self.get_save_path(base_path)
        print(f"Saving predictor and model settings in {save_path}")
        joblib.dump(self.predictor, save_path / "model.joblib")

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def train(self, data: Data) -> SupportedPredictorType:
        pass

    @abstractmethod
    def score(self, data: Data) -> float:
        pass


class ExplainerWrapper:
    explainer: SupportedExplainerType
    predictor_name: str

    def __init__(self, data: Data, predictor: PredictorWrapper):
        self.explainer = self.train(data, predictor.predictor)
        self.predictor_name = predictor.get_name()

    def get_save_path(self, base_path: Path = EXPLAINER_PATH) -> Path:
        return get_model_folder(base_path / self.get_name())

    def save(self, base_path: Path = EXPLAINER_PATH):
        save_path = self.get_save_path(base_path)
        print(f"Saving explainer and model settings in {save_path}")
        self.explainer.save(save_path)

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def train(self, data: Data, predictor: SupportedPredictorType) -> SupportedExplainerType:
        pass


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


class IncomeRandomForestClassifier(PredictorWrapper):

    def get_name(self) -> str:
        return "income_random_forest"

    def score(self, data: Data) -> float:
        return self.predictor.score(data.X_test, data.y_test)

    def train(self, data: Data) -> SupportedPredictorType:
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

        model_pipeline.fit(data.X_train, data.y_train)
        return model_pipeline


class IncomeKernelShap(ExplainerWrapper):

    def get_name(self) -> str:
        return "income_kernel_shap"

    def get_extra_model_settings(self) -> Dict:
        return {
            "explainer_type": "kernel_shap",
            "explainer_batch": True,
            "infer_output": "predict_proba"
        }

    def train(self, data: Data, predictor: SupportedPredictorType) -> SupportedExplainerType:
        explainer = KernelShap(
            predictor.predict_proba,
            "logit",
            data.feature_name,
            categorical_names=data.category_map,
            seed=1
        )
        explainer.fit(data.X_train[:100, :])
        return explainer


def fetch_diabetes_data() -> Data:
    np.random.seed(0)
    # prepare data
    X, y = shap.datasets.diabetes()
    X_train: npt.NDArray
    X_test: npt.NDArray
    y_train: npt.NDArray
    y_test: npt.NDArray
    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), np.array(y), test_size=0.2, random_state=0)

    # X_train_summary = shap.kmeans(X_train, 10)
    # print(X_train_summary)

    return Data(X_train, y_train, X_test, y_test, X.columns.to_list())


class DiabetesLinearRegression(PredictorWrapper):

    def get_name(self) -> str:
        return "diabetes_linear_regression"

    def train(self, data: Data) -> SupportedPredictorType:
        lin_regr = LinearRegression()
        lin_regr.fit(data.X_train, data.y_train)
        return lin_regr

    def score(self, data: Data) -> float:
        return self.predictor.score(data.X_test, data.y_test)


class DiabetesKernelShap(ExplainerWrapper):

    def get_name(self) -> str:
        return "diabetes_kernel_shap"

    def get_extra_model_settings(self) -> Dict:
        return {
            "explainer_type": "kernel_shap",
            "explainer_batch": True,
            "infer_output": "predict"
        }

    def train(self, data: Data, predictor: SupportedPredictorType) -> SupportedExplainerType:
        explainer = KernelShap(
            predictor.predict,
            "identity",
            data.feature_name,
            seed=1,
            task="regression"
        )
        explainer.fit(data.X_train[:100])
        return explainer


def fetch_multi_output_regression_dataset(num_features: int = 20, num_targets: int =5):
    np.random.seed(0)
    # Create sample data with sklearn make_regression function
    X, y = make_regression(n_samples=1000, n_features=num_features, n_informative=7, n_targets=5, random_state=0)

    # Convert the data into Pandas Dataframes for easier maniplution and keeping stored column names
    # Create feature column names
    feature_cols = [f"feat{i}" for i in range(num_features)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return Data(X_train, y_train, X_test, y_test, feature_cols)


class MultiOutputRegression(PredictorWrapper):

    def get_name(self) -> str:
        return "multi_output_regression"

    def train(self, data: Data) -> SupportedPredictorType:
        regr = MultiOutputRegressor(Ridge(random_state=0))
        regr.fit(data.X_train, data.y_train)
        return regr

    def score(self, data: Data) -> float:
        return self.predictor.score(data.X_test, data.y_test)


class MultiOutputRegressionKernelShap(ExplainerWrapper):

    def get_name(self) -> str:
        return "multi_output_regression_kernel_shap"

    def get_extra_model_settings(self) -> Dict:
        return {
            "explainer_type": "kernel_shap",
            "explainer_batch": True,
            "infer_output": "predict"
        }

    def train(self, data: Data, predictor: SupportedPredictorType) -> SupportedExplainerType:
        explainer = KernelShap(
            predictor.predict,
            "identity",
            data.feature_name,
            seed=1,
            task="regression"
        )
        explainer.fit(data.X_train[:100])
        return explainer


def generate_artifacts(data: Data, pred_wrapper: Type[PredictorWrapper], expl_wrappers: List[Type[ExplainerWrapper]]):
    pred = pred_wrapper(data)
    print(f"Predictor score: {pred.score(data)}")
    pred.save()

    for expl_wrapper in expl_wrappers:
        expl = expl_wrapper(data, pred)
        expl.save()

    if data.X_test[0][0] < 1:
        example_data = data.X_test[0]
        example_type = "FP64"
    else:
        example_data = [int(i) for i in data.X_test[0]]
        example_type = "INT64"

    print("Example request: ", {
        "inputs": [
            {
                "name": "income",
                "data": list(example_data),
                "datatype": example_type,
                "shape": [1, len(data.X_test[0])]
            }
        ]
    }
          )
    print(data.y_test[0])


def generate_all():
    income_data = fetch_income_data()
    generate_artifacts(income_data, IncomeRandomForestClassifier, [IncomeKernelShap])

    # diabetes_data = fetch_diabetes_data()
    # generate_artifacts(diabetes_data, DiabetesLinearRegression, [DiabetesKernelShap])
    #
    # regression_data = fetch_multi_output_regression_dataset()
    # generate_artifacts(regression_data, MultiOutputRegression, [MultiOutputRegressionKernelShap])


if __name__ == "__main__":
    assert version("alibi") == "0.7.0", f"unexpected version {version('alibi')}"
    # assert version("mlserver-alibi-explain") == "1.3.0.dev3", f"unexpected version {version('mlserver-alibi-explain')}"
    generate_all()
