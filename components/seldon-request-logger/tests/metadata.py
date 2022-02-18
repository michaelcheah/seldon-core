import copy

from tests.utils import RequestType, retry_with_backoff
import pytest
from seldon_deploy_sdk import V1Model
import json
import requests
import os
from dotenv import dotenv_values
import urllib3
from typing import Dict, List, Any
from enum import Enum

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TEST_NAMESPACE = "seldon"


class TestModel(Enum):
    IRIS = "iris"
    IRIS2 = "iris2"
    IRIS_KFSERVING = "iris-kfserving"
    INCOME = "income"
    INCOME_KFSERVING = "income-kfserving"
    CIFAR10 = "cifar10"
    MOVIE_SENTIMENT = "movie-sentiment"


@pytest.fixture(scope="session")
def env_configs():
    config = dotenv_values("tests/.env")
    return config


@pytest.fixture(scope="session")
def deploy_url_and_auth_header(env_configs):
    for expected_field in ["DEPLOY_API_HOST", "OIDC_PROVIDER", "CLIENT_ID", "CLIENT_SECRET",
                           "OIDC_USERNAME", "OIDC_PASSWORD", "OIDC_AUTH_METHOD", "OIDC_SCOPES"]:
        assert expected_field in env_configs

    # TODO: Make this more reliable
    deploy_url = env_configs["DEPLOY_API_HOST"][:-len("/seldon-deploy/api/v1alpha1")]

    # Get access token
    auth_url = f"{deploy_url}/auth/realms/deploy-realm/protocol/openid-connect/token"
    auth_body = {
        "grant_type": env_configs["OIDC_AUTH_METHOD"],
        "client_id": env_configs["CLIENT_ID"],
        "client_secret": env_configs["CLIENT_SECRET"],
        "scope": env_configs["OIDC_SCOPES"]
    }
    r = requests.post(auth_url, data=auth_body, verify=False)
    assert r.status_code == 200, r.text

    access_token = r.json().get("access_token", None)
    auth_header = {'Authorization': f'Bearer {access_token}'}

    check = requests.get(os.path.join(deploy_url, "seldon-deploy", "api", "v1alpha1", "healthcheck"),
                         headers=auth_header, verify=False)
    assert check.status_code == 200, check.text

    return deploy_url, auth_header


@pytest.fixture(scope="session")
def model_metadata():
    return {
        #  To test e2e have to use wizard to deploy some of below AFTER running this script
        #  Use names, uris and artifact types below when filling in wizard.
        #    Same model different versions
        TestModel.IRIS: {
            "uri": "gs://seldon-models/v1.11.2/sklearn/iris",
            "name": "iris",
            "version": "v2.0.0",
            "artifact_type": "SKLEARN",
            "task_type": "classification",
            "tags": {"author": "Bob"},
            "prediction_schema": {"requests": [{"name": "Sepal Length", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Sepal Width", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Petal Length", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Petal Width", "type": "REAL", "data_type": "FLOAT"}],
                                  "responses": [{"name": "Iris Species", "type": "PROBA", "data_type": "FLOAT",
                                                 "schema": [{"name": "Setosa"}, {"name": "Versicolor"},
                                                            {"name": "Virginica"}]}]}
        },
        TestModel.IRIS2: {  # schema made up to test edge cases
            "uri": "gs://seldon-models/sklearn/iris2",
            "name": "dummy",
            "version": "v1.0.0",
            "artifact_type": "SKLEARN",
            "task_type": "classification",
            "tags": {"author": "Noname"},
            "prediction_schema": {"requests": [{"name": "dummy_one_hot", "type": "ONE_HOT", "data_type": "INT",
                                                "schema": [{"name": "dummy_one_hot_1"}, {"name": "dummy_one_hot_2"}]},
                                               {"name": "dummy_categorical", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 2,
                                                "category_map": {"0": "dummy_cat_0", "1": "dummy_cat_1"}},
                                               {"name": "dummy_float", "type": "REAL", "data_type": "FLOAT"}],
                                  "responses": [{"name": "dummy_proba", "type": "PROBA", "data_type": "FLOAT",
                                                 "schema": [{"name": "dummy_proba_0"}, {"name": "dummy_proba_1"}]},
                                                {"name": "dummy_float", "type": "REAL", "data_type": "FLOAT"}]}
        },
        TestModel.IRIS_KFSERVING: {  # kfserving iris
            "uri": "gs://kfserving-samples/models/sklearn/iris",
            "name": "iris-kf",
            "version": "v2.0.0",
            "artifact_type": "SKLEARN",
            "task_type": "classification",
            "tags": {"author": "Jeff"},
            "prediction_schema": {"requests": [{"name": "Sepal Length", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Sepal Width", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Petal Length", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Petal Width", "type": "REAL", "data_type": "FLOAT"}],
                                  "responses": [{"name": "Iris Species", "type": "PROBA", "data_type": "FLOAT",
                                                 "schema": [{"name": "Setosa"}, {"name": "Versicolor"},
                                                            {"name": "Virginica"}]}]}
        },
        TestModel.INCOME: {
            # schema from https://github.com/SeldonIO/ml-prediction-schema/blob/master/examples/income-classifier.json
            "uri": "gs://seldon-models/sklearn/income/model-0.23.2",
            "name": "income",
            "version": "v2.0.0",
            "artifact_type": "SKLEARN",
            "task_type": "classification",
            "tags": {"author": "Fred"},
            "prediction_schema": {"requests": [{"name": "Age", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Workclass", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 9,
                                                "category_map": {"0": "?", "1": "Federal-gov", "2": "Local-gov",
                                                                 "3": "Never-worked", "4": "Private",
                                                                 "5": "Self-emp-inc", "6": "Self-emp-not-inc",
                                                                 "7": "State-gov", "8": "Without-pay"}},
                                               {"name": "Education", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 7,
                                                "category_map": {"0": "Associates", "1": "Bachelors", "2": "Doctorate",
                                                                 "3": "Dropout", "4": "High School grad",
                                                                 "5": "Masters", "6": "Prof-School"}},
                                               {"name": "Marital Status", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 4,
                                                "category_map": {"0": "Married", "1": "Never-Married", "2": "Separated",
                                                                 "3": "Widowed"}},
                                               {"name": "Occupation", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 9,
                                                "category_map": {"0": "?", "1": "Admin", "2": "Blue-Collar",
                                                                 "3": "Military", "4": "Other", "5": "Professional",
                                                                 "6": "Sales", "7": "Service", "8": "White-Collar"}},
                                               {"name": "Relationship", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 6,
                                                "category_map": {"0": "Husband", "1": "Not-in-family",
                                                                 "2": "Other-relative", "3": "Own-child",
                                                                 "4": "Unmarried", "5": "Wife"}},
                                               {"name": "Race", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 5,
                                                "category_map": {"0": "Amer-Indian-Eskimo", "1": "Asian-Pac-Islander",
                                                                 "2": "Black", "3": "Other", "4": "White"}},
                                               {"name": "Sex", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 2, "category_map": {"0": "Female", "1": "Male"}},
                                               {"name": "Capital Gain", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Capital Loss", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Hours per week", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Country", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 11,
                                                "category_map": {"0": "?", "1": "British-Commonwealth", "2": "China",
                                                                 "3": "Euro_1", "4": "Euro_2", "5": "Latin-America",
                                                                 "6": "Other", "7": "SE-Asia", "8": "South-America",
                                                                 "9": "United-States", "10": "Yugoslavia"}}],
                                  "responses": [{"name": "Income", "type": "PROBA", "data_type": "FLOAT",
                                                 "schema": [{"name": "<=$50K"}, {"name": ">$50K"}]}]}
        },
        TestModel.INCOME_KFSERVING: {  # kfserving income
            "uri": "gs://seldon-models/sklearn/income/model",
            "name": "income-kf",
            "version": "v2.0.0",
            "artifact_type": "SKLEARN",
            "task_type": "classification",
            "tags": {"author": "Jim"},
            "prediction_schema": {"requests": [{"name": "Age", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Workclass", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 9,
                                                "category_map": {"0": "?", "1": "Federal-gov", "2": "Local-gov",
                                                                 "3": "Never-worked", "4": "Private",
                                                                 "5": "Self-emp-inc", "6": "Self-emp-not-inc",
                                                                 "7": "State-gov", "8": "Without-pay"}},
                                               {"name": "Education", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 7,
                                                "category_map": {"0": "Associates", "1": "Bachelors", "2": "Doctorate",
                                                                 "3": "Dropout", "4": "High School grad",
                                                                 "5": "Masters", "6": "Prof-School"}},
                                               {"name": "Marital Status", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 4,
                                                "category_map": {"0": "Married", "1": "Never-Married", "2": "Separated",
                                                                 "3": "Widowed"}},
                                               {"name": "Occupation", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 9,
                                                "category_map": {"0": "?", "1": "Admin", "2": "Blue-Collar",
                                                                 "3": "Military", "4": "Other", "5": "Professional",
                                                                 "6": "Sales", "7": "Service", "8": "White-Collar"}},
                                               {"name": "Relationship", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 6,
                                                "category_map": {"0": "Husband", "1": "Not-in-family",
                                                                 "2": "Other-relative", "3": "Own-child",
                                                                 "4": "Unmarried", "5": "Wife"}},
                                               {"name": "Race", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 5,
                                                "category_map": {"0": "Amer-Indian-Eskimo", "1": "Asian-Pac-Islander",
                                                                 "2": "Black", "3": "Other", "4": "White"}},
                                               {"name": "Sex", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 2, "category_map": {"0": "Female", "1": "Male"}},
                                               {"name": "Capital Gain", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Capital Loss", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Hours per week", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Country", "type": "CATEGORICAL", "data_type": "INT",
                                                "n_categories": 11,
                                                "category_map": {"0": "?", "1": "British-Commonwealth", "2": "China",
                                                                 "3": "Euro_1", "4": "Euro_2", "5": "Latin-America",
                                                                 "6": "Other", "7": "SE-Asia", "8": "South-America",
                                                                 "9": "United-States", "10": "Yugoslavia"}}],
                                  "responses": [{"name": "Income", "type": "PROBA", "data_type": "FLOAT",
                                                 "schema": [{"name": "<=$50K"}, {"name": ">$50K"}]}]}
        },
        TestModel.CIFAR10: {  # cifar10
            "uri": "gs://seldon-models/tfserving/cifar10/resnet32",
            "name": "cifar10",
            "version": "v1.0.0",
            "artifact_type": "TENSORFLOW",
            "task_type": "classification",
            "tags": {"author": "Noname"},
            "prediction_schema": {
                "requests": [{"name": "Input Image", "type": "TENSOR", "data_type": "FLOAT", "shape": [32, 32, 3]}],
                "responses": [{"name": "Image Class", "type": "PROBA", "data_type": "FLOAT",
                               "schema": [{"name": "Airplane"}, {"name": "Automobile"}, {"name": "Bird"},
                                          {"name": "Cat"}, {"name": "Deer"}, {"name": "Dog"}, {"name": "Frog"},
                                          {"name": "Horse"}, {"name": "Ship"}, {"name": "Truck"}]}]}
        },
        TestModel.MOVIE_SENTIMENT: {
            "uri": "gs://seldon-models/sklearn/moviesentiment",
            "name": "movie-sentiment",
            "version": "v1.0.0",
            "artifact_type": "SKLEARN",
            "task_type": "classification",
            "tags": {"author": "Noname"},
            "prediction_schema": {
                "requests": [{"name": "Review Comment", "type": "TEXT"}],
                "responses": [{"name": "Sentiment", "type": "ONE_HOT", "data_type": "INT",
                               "schema": [{"name": "Negative"}, {"name": "Positive"}]}],
            }
        },
    }


def convert_to_json(body: V1Model) -> Dict:
    return {
        "URI": body.uri,
        "name": body.name,
        "version": body.version,
        "artifactType": body.artifact_type,
        "taskType": body.task_type,
        "tags": body.tags,
        "metrics": body.metrics,
        "predictionSchema": body.prediction_schema,
        "project": body.project,
    }


@pytest.fixture(scope="session")
def created_models(deploy_url_and_auth_header, model_metadata, env_configs):
    deploy_url, auth_header = deploy_url_and_auth_header
    model_metadata_url = os.path.join(deploy_url, "seldon-deploy", "api", "v1alpha1", "model", "metadata")
    created_models: List[Dict] = []
    try:
        for model_name in model_metadata:
            model = model_metadata[model_name]
            body = V1Model(**model)
            r = requests.post(model_metadata_url, json=convert_to_json(body), headers=auth_header, verify=False)
            if r.status_code == 409:
                print(f"Model already exists: {body.uri} in project {body.project}")
                print("Checking if prediction schema exists...")
                check_prediction_schema_exists(body.uri, deploy_url, auth_header)
                created_models.append(convert_to_json(body))
                continue
            assert r.status_code == 200, r.text
            created_models.append(convert_to_json(body))
    except AssertionError as ae:
        print(ae)
    except Exception as e:
        print(e)

    yield created_models
    for created_model in created_models:
        uri, project = created_model["URI"], created_model["project"]
        r = requests.delete(model_metadata_url, params=created_model, headers=auth_header, verify=False)
        if r.status_code != 200:
            print(f"Could not delete created model: {uri} in project {project}: {r.text}")


def get_deployments_url(test_model: TestModel, deploy_url) -> str:
    v1alpha1_url = os.path.join(deploy_url, "seldon-deploy", "api", "v1alpha1")
    if test_model.value.endswith("kfserving"):
        deployments_url = os.path.join(v1alpha1_url, "namespaces", TEST_NAMESPACE, "inferenceservices")
    else:
        deployments_url = os.path.join(v1alpha1_url, "namespaces", TEST_NAMESPACE, "seldondeployments")
    return deployments_url


def check_prediction_schema_exists(model_uri: str, deploy_url: str, auth_header: Dict):
    assert model_uri != "", "need the model uri"

    model_identifier = {"URI": model_uri}
    r = requests.get(os.path.join(deploy_url, "seldon-deploy", "api", "v1alpha1", "model", "metadata"),
                     params=model_identifier, headers=auth_header, verify=False)
    assert r.status_code == 200, r.text
    models = r.json().get("models", [])
    assert len(models) == 1, "only one model needed"
    schema = models[0].get("predictionSchema", None)
    assert schema is not None, f"cannot find prediction schema in model metadata: {models[0]}"


def create_deployment(test_model: TestModel, model_metadata: Dict, deploy_url_and_auth_header):
    metadata = model_metadata.get(test_model)
    model_uri = metadata["uri"]
    kind = "SeldonDeployment"
    if "kfserving" in test_model.value:
        kind = "InferenceService"

    artifact_type = metadata['artifact_type']

    deploy_url, auth_header = deploy_url_and_auth_header
    deployments_url = get_deployments_url(test_model, deploy_url)
    create_params = {"action": "create", "messages": "creating test deployment"}

    if kind == "SeldonDeployment":
        deployment = {
            "apiVersion": "machinelearning.seldon.io/v1alpha2",
            "kind": "SeldonDeployment",
            "metadata": {
                "name": f"{test_model.value}-test-deployment",
                "namespace": TEST_NAMESPACE,
            },
            "spec": {
                "name": f"{test_model.value}-test-deployment",
                "predictors": [
                    {
                        "graph": {
                            "implementation": f"{artifact_type}_SERVER",
                            "modelUri": model_uri,
                            "name": f"{test_model.value}-test-container",
                        },
                        "name": "default",
                    }
                ]
            },
        }
    else:
        deployment = {
            "apiVersion": "serving.kubeflow.org/v1alpha2",
            "kind": "InferenceService",
            "metadata": {
                "name": f"{test_model.value}-test-deployment",
                "namespace": TEST_NAMESPACE,
            },
            "spec": {
                "default": {
                    "predictor": {
                        artifact_type.lower(): {
                            "storageUri": model_uri,
                        }
                    },
                },
            },
        }
    print(deployment)

    result = requests.post(deployments_url, json=deployment, params=create_params, headers=auth_header, verify=False)
    assert result.status_code == 200, result.text

    def model_running_in_deployment():
        search_param = {
            "DeploymentName": f"{test_model.value}-test-deployment",
            "DeploymentNamespace": TEST_NAMESPACE,
            "DeploymentStatus": "Running",
        }
        r = requests.get(os.path.join(deploy_url, "seldon-deploy", "api", "v1alpha1", "model", "metadata", "runtime"),
                         params=search_param, headers=auth_header, verify=False)
        assert r.status_code == 200, r.text
        runtime_metadata = r.json().get("runtimeMetadata", [])
        assert len(runtime_metadata) > 0, f"cannot find runtime metadata with {search_param}"
        status = runtime_metadata[0].get("deploymentStatus", None)
        assert status is not None, "cannot find status"
        assert status == "Running", "status not running yet"
        model = runtime_metadata[0].get("model", None)
        assert model is not None, "model must be present"
        schema = model.get("predictionSchema", None)
        assert schema is not None, "cannot find prediction schema in runtime metadata model"

        retrieved_model_uri = model.get("URI", "")

        check_prediction_schema_exists(retrieved_model_uri, deploy_url, auth_header)
        print(f"Model and schema has been found in model metadata search: {search_param}")

    retry_with_backoff(model_running_in_deployment)
    return deployment


def delete_deployment(test_model: TestModel, deploy_url_and_auth_header):
    deploy_url, auth_header = deploy_url_and_auth_header
    deployments_url = get_deployments_url(test_model, deploy_url)
    deployment_url = os.path.join(deployments_url, f"{test_model.value}-test-deployment")
    result = requests.delete(deployment_url, params={"action": "delete", "messages": "delete test deployment"},
                             headers=auth_header, verify=False)
    assert result.status_code == 200, result.text


def deploy(test_model, model_metadata, deploy_url_and_auth_header):
    try:
        deployment = create_deployment(test_model, model_metadata, deploy_url_and_auth_header)
    except Exception as e:
        print(e)
        delete_deployment(test_model, deploy_url_and_auth_header)
        raise (e)
    return deployment


@pytest.fixture(scope="class")
def seldon_iris_deployment(created_models, model_metadata, deploy_url_and_auth_header):
    test_model = TestModel.IRIS
    deployment = deploy(test_model, model_metadata, deploy_url_and_auth_header)
    yield deployment
    delete_deployment(test_model, deploy_url_and_auth_header)


def seldon_iris_deployment_info():
    test_model = TestModel.IRIS
    deployment_name = f"{test_model.value}-test-deployment"
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Type": "io.seldon.serving.inference.request",
        "Ce-Namespace": TEST_NAMESPACE,
        "Ce-Inferenceservicename": deployment_name,
        "Ce-Endpoint": "default",
    }
    expected_index = "inference-log-seldon-seldon-iris-test-deployment-default"

    return deployment_name, request_headers, expected_index


def seldon_iris_batch(_: RequestType):
    deployment_name, request_headers, expected_index = seldon_iris_deployment_info()

    request_data = '{"data":{"names":["SL","SW","PL","PW"],' + \
                   '"ndarray":[[6.8,2.8,4.8,1.4],[6.1,3.4,4.5,1.6]]}}'
    request_headers["Ce-Requestid"] = "m1a"

    expected_elastic_docs = {
        "m1a-item-0": {
            'request': {
                'elements': {"Sepal Length": 6.8, "Sepal Width": 2.8, "Petal Length": 4.8, "Petal Width": 1.4},
                'instance': [6.8, 2.8, 4.8, 1.4],
                'dataType': 'tabular',
                'names': ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
                'payload': {"data": {"names": ["SL", "SW", "PL", "PW"],
                                     "ndarray": [[6.8, 2.8, 4.8, 1.4], [6.1, 3.4, 4.5, 1.6]]}},
            },
            'ServingEngine': 'seldon',
            "Ce-Namespace": TEST_NAMESPACE,
            "Ce-Inferenceservicename": deployment_name,
            "Ce-Endpoint": "default",
            'RequestId': 'm1a'
        },
        "m1a-item-1": {
            'request': {
                'elements': {"Sepal Length": 6.1, "Sepal Width": 3.4, "Petal Length": 4.5, "Petal Width": 1.6},
                'instance': [6.1, 3.4, 4.5, 1.6],
                'dataType': 'tabular',
                'names': ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
                'payload': {"data": {"names": ["SL", "SW", "PL", "PW"],
                                     "ndarray": [[6.8, 2.8, 4.8, 1.4], [6.1, 3.4, 4.5, 1.6]]}},
            },
            'ServingEngine': 'seldon',
            "Ce-Namespace": TEST_NAMESPACE,
            "Ce-Inferenceservicename": deployment_name,
            "Ce-Endpoint": "default",
            'RequestId': 'm1a'
        }
    }
    return request_headers, request_data, expected_index, expected_elastic_docs


def seldon_iris_not_batch(_: RequestType):
    deployment_name, request_headers, expected_index = seldon_iris_deployment_info()

    request_data = '{"data":{"names":["SL","SW","PL","PW"],' + \
                   '"ndarray":[[6.3,2.8,4.8,1.4]]}}'
    request_headers["Ce-Requestid"] = "m1b"

    expected_elastic_docs = {
        "m1b": {
            'request': {
                'elements': {"Sepal Length": 6.3, "Sepal Width": 2.8, "Petal Length": 4.8, "Petal Width": 1.4},
                'instance': [6.3, 2.8, 4.8, 1.4],
                'dataType': 'tabular',
                'names': ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
                'payload': {"data": {"names": ["SL", "SW", "PL", "PW"],
                                     "ndarray": [[6.3, 2.8, 4.8, 1.4]]}},
            },
            'ServingEngine': 'seldon',
            "Ce-Namespace": TEST_NAMESPACE,
            "Ce-Inferenceservicename": deployment_name,
            "Ce-Endpoint": "default",
            'RequestId': 'm1b'
        }
    }
    return request_headers, request_data, expected_index, expected_elastic_docs


@pytest.fixture(scope="module")
def kfserving_iris_deployment(created_models, model_metadata, deploy_url_and_auth_header):
    test_model = TestModel.IRIS_KFSERVING
    deployment = deploy(test_model, model_metadata, deploy_url_and_auth_header)
    yield deployment
    delete_deployment(test_model, deploy_url_and_auth_header)


def kfserving_iris_deployment_info():
    test_model = TestModel.IRIS_KFSERVING
    deployment_name = f"{test_model.value}-test-deployment"
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Namespace": TEST_NAMESPACE,
        "Ce-Inferenceservicename": deployment_name,
        "Ce-Endpoint": "default",
    }
    expected_index = "inference-log-inferenceservice-seldon-iris-kfserving-test-deployment-default"
    return deployment_name, request_headers, expected_index


def kfserving_iris_batch(request_type: RequestType):
    deployment_name, request_headers, expected_index = kfserving_iris_deployment_info()
    request_headers["Ce-Requestid"] = "m2"
    expected_elastic_docs: Dict[str, Any] = {
        "m2-item-0": {
            'ServingEngine': 'inferenceservice',
            "Ce-Namespace": TEST_NAMESPACE,
            "Ce-Inferenceservicename": deployment_name,
            "Ce-Endpoint": "default",
            "RequestId": "m2",
        },
        "m2-item-1": {
            'ServingEngine': 'inferenceservice',
            "Ce-Namespace": TEST_NAMESPACE,
            "Ce-Inferenceservicename": deployment_name,
            "Ce-Endpoint": "default",
            "RequestId": "m2",
        }
    }

    if request_type == RequestType.REQUEST:
        request_data = '{"instances": [[6.8,  2.8,  4.8,  1.4],[6.0,  3.4,  4.5,  1.6]]}'
        request_headers["Ce-Type"] = "org.kubeflow.serving.inference.request"
        expected_elastic_docs["m2-item-0"]["request"] = {
            'elements': {"Sepal Length": 6.8, "Sepal Width": 2.8, "Petal Length": 4.8, "Petal Width": 1.4},
            'instance': [6.8, 2.8, 4.8, 1.4],
            'dataType': 'tabular',
            'payload': {'instances': [[6.8, 2.8, 4.8, 1.4], [6.0, 3.4, 4.5, 1.6]]},
        }
        expected_elastic_docs["m2-item-1"]["request"] = {
            'elements': {"Sepal Length": 6.0, "Sepal Width": 3.4, "Petal Length": 4.5, "Petal Width": 1.6},
            'instance': [6.0, 3.4, 4.5, 1.6],
            'dataType': 'tabular',
            'payload': {'instances': [[6.8, 2.8, 4.8, 1.4], [6.0, 3.4, 4.5, 1.6]]},
        }
    elif request_type == RequestType.RESPONSE:
        request_data = '{"predictions": [[0.008074020139120223,0.7781601484223128,0.21376583143856684],' + \
                       '[0.04569799579422263,0.5165292130301182,0.4377727911756591]]}'

        request_headers["Ce-Type"] = "org.kubeflow.serving.inference.response"
        expected_elastic_docs["m2-item-0"]["response"] = {
            'elements': {
                "Iris Species": {
                    "Setosa": 0.008074020139120223, "Versicolor": 0.7781601484223128, "Virginica": 0.21376583143856684
                }
            },
            'instance': [0.008074020139120223, 0.7781601484223128, 0.21376583143856684],
            'dataType': 'tabular',
            'payload': {"predictions": [[0.008074020139120223, 0.7781601484223128, 0.21376583143856684],
                                        [0.04569799579422263, 0.5165292130301182, 0.4377727911756591]]},
        }
        expected_elastic_docs["m2-item-1"]["response"] = {
            'elements': {
                "Iris Species": {
                    "Setosa": 0.04569799579422263, "Versicolor": 0.5165292130301182, "Virginica": 0.4377727911756591
                }
            },
            'instance': [0.04569799579422263, 0.5165292130301182, 0.4377727911756591],
            'dataType': 'tabular',
            'payload': {"predictions": [[0.008074020139120223, 0.7781601484223128, 0.21376583143856684],
                                        [0.04569799579422263, 0.5165292130301182, 0.4377727911756591]]},
        }

    elif request_type == RequestType.OUTLIER:
        request_data = '{"data": {"feature_score": null, "instance_score": null, "is_outlier": [1, 0]}, ' + \
                       '"meta": {"name": "OutlierVAE", "detector_type": "offline", "data_type": "image"}}'
        request_headers["Ce-Type"] = "org.kubeflow.serving.inference.outlier"
        expected_elastic_docs["m2-item-0"]["outlier"] = {
            'data': {'feature_score': None, 'instance_score': None, 'is_outlier': 1},
            'meta': {'name': 'OutlierVAE', 'detector_type': 'offline', 'data_type': 'image'},
        }
        expected_elastic_docs["m2-item-1"]["outlier"] = {
            'data': {'feature_score': None, 'instance_score': None, 'is_outlier': 0},
            'meta': {'name': 'OutlierVAE', 'detector_type': 'offline', 'data_type': 'image'},
        }
    else:
        raise Exception("invalid request_type provided")

    return request_headers, request_data, expected_index, expected_elastic_docs


@pytest.fixture(scope="module")
def seldon_moviesentiment_deployment(created_models, model_metadata, deploy_url_and_auth_header):
    test_model = TestModel.MOVIE_SENTIMENT
    deployment = deploy(test_model, model_metadata, deploy_url_and_auth_header)
    yield deployment
    delete_deployment(test_model, deploy_url_and_auth_header)


def seldon_moviesentiment_deployment_info():
    test_model = TestModel.MOVIE_SENTIMENT
    deployment_name = f"{test_model.value}-test-deployment"
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Namespace": TEST_NAMESPACE,
        "Ce-Inferenceservicename": deployment_name,
        "Ce-Endpoint": "default",
    }
    expected_index = "inference-log-seldon-seldon-movie-sentiment-test-deployment-default"
    return deployment_name, request_headers, expected_index


def seldon_moviesentiment_text_no_batch(request_type: RequestType):
    deployment_name, request_headers, expected_index = seldon_moviesentiment_deployment_info()
    request_headers["Ce-Requestid"] = "m3a"
    expected_elastic_docs: Dict[str, Any] = {
        "m3a": {
            'ServingEngine': 'seldon',
            "Ce-Namespace": TEST_NAMESPACE,
            "Ce-Inferenceservicename": deployment_name,
            "RequestId": "m3a",
        },
    }

    if request_type == RequestType.REQUEST:
        request_data = '{"data": {"names": ["Text review"],' + \
                       '"ndarray": ["this film was fantastic"]}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.request"
        expected_elastic_docs["m3a"]["request"] = {
            'elements': {'Review Comment': "this film was fantastic"},
            'instance': "this film was fantastic",
            'dataType': 'text',
            'names': ['Text review'],
            'payload': {"data": {"names": ["Text review"],
                                 "ndarray": ["this film was fantastic"]}}
        }
    elif request_type == RequestType.RESPONSE:
        request_data = '{"data":{"names":["t0","t1"],"ndarray":[[0.05,0.95]]}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.response"
        expected_elastic_docs["m3a"]["response"] = {
            'elements': {'Sentiment': {'Negative': 0.05, "Positive": 0.95}},
            'instance': [0.05, 0.95],
            'dataType': 'tabular',
            'names': ['t0', 't1'],
            'payload': {'data': {'names': ['t0', 't1'], 'ndarray': [[0.05, 0.95]]}}
        }
    else:
        raise Exception("invalid request_type provided")

    return request_headers, request_data, expected_index, expected_elastic_docs


def seldon_moviesentiment_text_batch(request_type: RequestType):
    deployment_name, request_headers, expected_index = seldon_moviesentiment_deployment_info()
    request_headers["Ce-Requestid"] = "m3b"
    expected_elastic_docs: Dict[str, Any] = {
        "m3b-item-0": {
            'ServingEngine': 'seldon',
            "Ce-Namespace": TEST_NAMESPACE,
            "Ce-Inferenceservicename": deployment_name,
            "RequestId": "m3b",
        },
        "m3b-item-1": {
            'ServingEngine': 'seldon',
            "Ce-Namespace": TEST_NAMESPACE,
            "Ce-Inferenceservicename": deployment_name,
            "RequestId": "m3b",
        }
    }

    if request_type == RequestType.REQUEST:
        request_data = '{"data": {"names": ["Text review"],' + \
                       '"ndarray": ["this film has great actors", "this film has poor actors"]}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.request"
        expected_elastic_docs["m3b-item-0"]["request"] = {
            'elements': {'Review Comment': "this film has great actors"},
            'instance': "this film has great actors",
            'dataType': 'text',
            'names': ['Text review'],
            'payload': {"data": {"names": ["Text review"],
                                 "ndarray": ["this film has great actors", "this film has poor actors"]}}
        }
        expected_elastic_docs["m3b-item-1"]["request"] = {
            'elements': {'Review Comment': "this film has poor actors"},
            'instance': "this film has poor actors",
            'dataType': 'text',
            'names': ['Text review'],
            'payload': {"data": {"names": ["Text review"],
                                 "ndarray": ["this film has great actors", "this film has poor actors"]}}
        }
    elif request_type == RequestType.RESPONSE:
        request_data = '{"data":{"names":["t0","t1"],"ndarray":[[0.2,0.8],[0.7,0.3]]}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.response"
        expected_elastic_docs["m3b-item-0"]["response"] = {
            'elements': {'Sentiment': {'Negative': 0.2, "Positive": 0.8}},
            'instance': [0.2, 0.8],
            'dataType': 'tabular',
            'names': ['t0', 't1'],
            'payload': {'data': {'names': ['t0', 't1'], 'ndarray': [[0.2, 0.8], [0.7, 0.3]]}}
        }
        expected_elastic_docs["m3b-item-1"]["response"] = {
            'elements': {'Sentiment': {'Negative': 0.7, "Positive": 0.3}},
            'instance': [0.7, 0.3],
            'dataType': 'tabular',
            'names': ['t0', 't1'],
            'payload': {'data': {'names': ['t0', 't1'], 'ndarray': [[0.2, 0.8], [0.7, 0.3]]}}
        }
    else:
        raise Exception("invalid request_type provided")

    return request_headers, request_data, expected_index, expected_elastic_docs


@pytest.fixture(scope="module")
def seldon_income_deployment(created_models, model_metadata, deploy_url_and_auth_header):
    test_model = TestModel.INCOME
    deployment = deploy(test_model, model_metadata, deploy_url_and_auth_header)
    yield deployment
    delete_deployment(test_model, deploy_url_and_auth_header)


def seldon_income_deployment_info():
    test_model = TestModel.INCOME
    deployment_name = f"{test_model.value}-test-deployment"
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Namespace": TEST_NAMESPACE,
        "Ce-Inferenceservicename": deployment_name,
        "Ce-Endpoint": "default",
    }
    expected_index = "inference-log-seldon-seldon-income-test-deployment-default"
    return deployment_name, request_headers, expected_index


def seldon_income_batch(request_type: RequestType):
    deployment_name, request_headers, expected_index = seldon_income_deployment_info()
    request_headers["Ce-Requestid"] = "m4"
    common_fields = {
        'ServingEngine': 'seldon',
        "Ce-Namespace": TEST_NAMESPACE,
        "Ce-Inferenceservicename": deployment_name,
        "RequestId": "m4",
    }
    expected_elastic_docs: Dict[str, Any] = {
        "m4-item-0": common_fields,
        "m4-item-1": copy.deepcopy(common_fields),
    }

    if request_type == RequestType.REQUEST:
        request_data = '{"data":{"names":["A","W","E","MS","O","Re","Ra","S","CG","CL","HPW","C"],' + \
                       '"ndarray":[[53,4,0,2,8,4,2,0,0,0,60,9],[22,3,1,1,5,4,3,1,0,0,40,2]]}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.request"
        expected_elastic_docs["m4-item-0"]["request"] = {
            'elements': {
                "Age": 53.0, "Workclass": "Private", "Education": "Associates", "Marital Status": "Separated",
                "Occupation": "White-Collar", "Relationship": "Unmarried", "Race": "Black", "Sex": "Female",
                "Capital Gain": 0.0, "Capital Loss": 0.0,
                "Hours per week": 60.0, "Country": "United-States",
            },
            'instance': [53.0, 4.0, 0.0, 2.0, 8.0, 4.0, 2.0, 0.0, 0.0, 0.0, 60.0, 9.0],
            'dataType': 'tabular',
            'names': ["A", "W", "E", "MS", "O", "Re", "Ra", "S", "CG", "CL", "HPW", "C"],
            'payload': {"data": {"names": ["A", "W", "E", "MS", "O", "Re", "Ra", "S", "CG", "CL", "HPW", "C"],
                                 "ndarray": [[53, 4, 0, 2, 8, 4, 2, 0, 0, 0, 60, 9],
                                             [22, 3, 1, 1, 5, 4, 3, 1, 0, 0, 40, 2]]}}
        }
        expected_elastic_docs["m4-item-1"]["request"] = {
            'elements': {
                "Age": 22.0, "Workclass": "Never-worked", "Education": "Bachelors", "Marital Status": "Never-Married",
                "Occupation": "Professional", "Relationship": "Unmarried", "Race": "Other", "Sex": "Male",
                "Capital Gain": 0.0, "Capital Loss": 0.0, "Hours per week": 40.0, "Country": "China",
            },
            'instance': [22.0, 3.0, 1.0, 1.0, 5.0, 4.0, 3.0, 1.0, 0.0, 0.0, 40.0, 2.0],
            'dataType': 'tabular',
            'names': ["A", "W", "E", "MS", "O", "Re", "Ra", "S", "CG", "CL", "HPW", "C"],
            'payload': {"data": {"names": ["A", "W", "E", "MS", "O", "Re", "Ra", "S", "CG", "CL", "HPW", "C"],
                                 "ndarray": [[53, 4, 0, 2, 8, 4, 2, 0, 0, 0, 60, 9],
                                             [22, 3, 1, 1, 5, 4, 3, 1, 0, 0, 40, 2]]}}
        }
    elif request_type == RequestType.RESPONSE:
        request_data = '{"data":{"names":["t:0","t:1"],"ndarray":[[0.85,0.15],[0.66,0.34]]},' + \
                       '"meta":{"requestPath":{"income-container":"seldonio/sklearnserver:1.7.0"}}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.response"
        expected_elastic_docs["m4-item-0"]["response"] = {
            'elements': {"Income": {"<=$50K": 0.85, ">$50K": 0.15}},
            'instance': [0.85, 0.15],
            'dataType': 'tabular',
            'meta': {'requestPath': {'income-container': 'seldonio/sklearnserver:1.7.0'}},
            'names': ['t:0', 't:1'],
            'payload': {"data": {"names": ["t:0", "t:1"], "ndarray": [[0.85, 0.15], [0.66, 0.34]]},
                        "meta": {"requestPath": {"income-container": "seldonio/sklearnserver:1.7.0"}}}
        }
        expected_elastic_docs["m4-item-1"]["response"] = {
            'elements': {"Income": {"<=$50K": 0.66, ">$50K": 0.34}},
            'instance': [0.66, 0.34],
            'dataType': 'tabular',
            'meta': {'requestPath': {'income-container': 'seldonio/sklearnserver:1.7.0'}},
            'names': ['t:0', 't:1'],
            'payload': {"data": {"names": ["t:0", "t:1"], "ndarray": [[0.85, 0.15], [0.66, 0.34]]},
                        "meta": {"requestPath": {"income-container": "seldonio/sklearnserver:1.7.0"}}}
        }
    elif request_type == RequestType.REFERENCE_REQUEST:
        expected_index = "reference-log-seldon-seldon-income-test-deployment"
        request_data = '{"instances":[[53,4,0,2,8,4,2,0,0,0,60,9],[22,3,1,1,5,4,3,1,0,0,40,2]]}'
        request_headers["Ce-Type"] = "io.seldon.serving.reference.request"
        expected_elastic_docs["m4-item-0"]["request"] = {
            'elements': {
                "Age": 53.0, "Workclass": "Private", "Education": "Associates", "Marital Status": "Separated",
                "Occupation": "White-Collar", "Relationship": "Unmarried", "Race": "Black", "Sex": "Female",
                "Capital Gain": 0.0, "Capital Loss": 0.0,
                "Hours per week": 60.0, "Country": "United-States",
            },
            'instance': [53, 4, 0, 2, 8, 4, 2, 0, 0, 0, 60, 9],
            'dataType': 'tabular',
        }
        expected_elastic_docs["m4-item-1"]["request"] = {
            'elements': {
                "Age": 22.0, "Workclass": "Never-worked", "Education": "Bachelors", "Marital Status": "Never-Married",
                "Occupation": "Professional", "Relationship": "Unmarried", "Race": "Other", "Sex": "Male",
                "Capital Gain": 0.0, "Capital Loss": 0.0, "Hours per week": 40.0, "Country": "China",
            },
            'instance': [22, 3, 1, 1, 5, 4, 3, 1, 0, 0, 40, 2],
            'dataType': 'tabular',
        }
    else:
        raise Exception("invalid request_type provided")

    return request_headers, request_data, expected_index, expected_elastic_docs


@pytest.fixture(scope="module")
def seldon_cifar10_deployment(created_models, model_metadata, deploy_url_and_auth_header):
    test_model = TestModel.CIFAR10
    deployment = deploy(test_model, model_metadata, deploy_url_and_auth_header)
    yield deployment
    delete_deployment(test_model, deploy_url_and_auth_header)


def seldon_cifar10_deployment_info():
    test_model = TestModel.CIFAR10
    deployment_name = f"{test_model.value}-test-deployment"
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Namespace": TEST_NAMESPACE,
        "Ce-Inferenceservicename": deployment_name,
        "Ce-Endpoint": "default",
    }
    expected_index = "inference-log-seldon-seldon-cifar10-test-deployment-default"
    return deployment_name, request_headers, expected_index


def seldon_cifar10_single(request_type: RequestType):
    deployment_name, request_headers, expected_index = seldon_cifar10_deployment_info()
    request_headers["Ce-Requestid"] = "m5a"
    common_fields = {
        'ServingEngine': 'seldon',
        "Ce-Namespace": TEST_NAMESPACE,
        "Ce-Inferenceservicename": deployment_name,
        "RequestId": "m5a",
    }
    expected_elastic_docs: Dict[str, Any] = {
        "m5a": common_fields,
    }

    if request_type == RequestType.REQUEST:
        with open("tests/cifardata.json", "rb") as f:
            raw_data = f.read()
        data = json.loads(raw_data)
        # batch_instances = data["instances"]
        # batch_instances.append(copy.deepcopy(batch_instances[0]))
        #
        # data["instances"] = batch_instances  # this might be redundant

        request_data = raw_data
        request_headers["Ce-Type"] = "io.seldon.serving.inference.request"
        expected_elastic_docs["m5a"]["request"] = {
            'elements': {'Input Image': data["instances"][0]},
            'instance': data["instances"][0],
            'dataType': 'image',
            'payload': data,
        }

    elif request_type == RequestType.RESPONSE:
        request_data = '{"predictions":[[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.55]]}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.response"
        expected_elastic_docs["m5a"]["response"] = {
            'elements': {"Image Class": {
                "Airplane": 0.01, "Automobile": 0.02, "Bird": 0.03, "Cat": 0.04, "Deer": 0.05, "Dog": 0.06,
                "Frog": 0.07, "Horse": 0.08, "Ship": 0.09, "Truck": 0.55}
            },
            'instance': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.55],
            'dataType': 'tabular',
            'payload': {"predictions": [[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.55]]},
        }
    else:
        raise Exception("invalid request_type provided")

    return request_headers, request_data, expected_index, expected_elastic_docs


def seldon_cifar10_batch(request_type: RequestType):
    deployment_name, request_headers, expected_index = seldon_cifar10_deployment_info()
    request_headers["Ce-Requestid"] = "m5b"
    common_fields = {
        'ServingEngine': 'seldon',
        "Ce-Namespace": TEST_NAMESPACE,
        "Ce-Inferenceservicename": deployment_name,
        "RequestId": "m5b",
    }
    expected_elastic_docs: Dict[str, Any] = {
        "m5b-item-0": common_fields,
        "m5b-item-1": copy.deepcopy(common_fields),
    }

    if request_type == RequestType.REQUEST:
        with open("tests/cifardata.json", "rb") as f:
            raw_data = f.read()
        data = json.loads(raw_data)
        batch_instances = data["instances"]
        batch_instances.append(copy.deepcopy(batch_instances[0]))
        data["instances"] = batch_instances  # this might be redundant

        request_data = raw_data
        request_headers["Ce-Type"] = "io.seldon.serving.inference.request"
        expected_elastic_docs["m5b-item-0"]["request"] = {
            'elements': None,
            'instance': data["instances"][0],
            'dataType': 'image',
            'payload': data,
        }
        expected_elastic_docs["m5b-item-1"]["request"] = {
            'elements': None,
            'instance': data["instances"][1],
            'dataType': 'image',
            'payload': data,
        }

    elif request_type == RequestType.RESPONSE:
        request_data = '{"predictions":[[0.55,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.01],' + \
                       '[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.55,0.09]]}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.response"
        expected_elastic_docs["m5b-item-0"]["response"] = {
            'elements': {"Image Class": {
                "Airplane": 0.55, "Automobile": 0.02, "Bird": 0.03, "Cat": 0.04, "Deer": 0.05, "Dog": 0.06,
                "Frog": 0.07, "Horse": 0.08, "Ship": 0.09, "Truck": 0.01}
            },
            'instance': [0.55, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01],
            'dataType': 'tabular',
            'payload': {"predictions": [[0.55, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01],
                                        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.55, 0.09]]},
        }
        expected_elastic_docs["m5b-item-1"]["response"] = {
            'elements': {"Image Class": {
                "Airplane": 0.01, "Automobile": 0.02, "Bird": 0.03, "Cat": 0.04, "Deer": 0.05, "Dog": 0.06,
                "Frog": 0.07, "Horse": 0.08, "Ship": 0.55, "Truck": 0.09}
            },
            'instance': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.55, 0.09],
            'dataType': 'tabular',
            'payload': {"predictions": [[0.55, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01],
                                        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.55, 0.09]]},
        }
    else:
        raise Exception("invalid request_type provided")

    return request_headers, request_data, expected_index, expected_elastic_docs
