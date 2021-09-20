from tests.utils import RequestType, retry_with_backoff
import pytest
from seldon_deploy_sdk import V1Model
import json
import requests
import os
from dotenv import dotenv_values
import urllib3
from typing import Dict, List
from enum import Enum

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TEST_NAMESPACE = "seldon"


class TestModel(Enum):
    IRIS = "iris"
    IRIS2 = "iris2"
    IRIS_BETA = "iris-beta"
    IRIS_KFSERVING = "iris-kfserving"
    INCOME = "income"
    INCOME_KFSERVING = "income-kfserving"
    CIFAR10 = "cifar10"


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
            "uri": "gs://seldon-models/sklearn/iris",
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
        TestModel.IRIS_BETA: {
            "uri": "gs://test-model-beta-v2.0.0",
            "name": "iris",
            "version": "v1.0.0",
            "artifact_type": "SKLEARN",
            "task_type": "classification",
            "tags": {"author": "Jon"},
            "prediction_schema": {"requests": [{"name": "Sepal Length", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Sepal Width", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Petal Length", "type": "REAL", "data_type": "FLOAT"},
                                               {"name": "Petal Width", "type": "REAL", "data_type": "FLOAT"}],
                                  "responses": [{"name": "Iris Species", "type": "PROBA", "data_type": "FLOAT",
                                                 "schema": [{"name": "Setosa"}, {"name": "Versicolor"},
                                                            {"name": "Virginica"}]}]}
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
        }
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
            print(f"--> Creating {model_name} model")
            model = model_metadata[model_name]
            body = V1Model(**model)
            r = requests.post(model_metadata_url, json=convert_to_json(body), headers=auth_header, verify=False)
            if r.status_code == 409:
                print(f"Model already exists: {body.uri} in project {body.project}")
                created_models.append(convert_to_json(body))
                continue
            assert r.status_code == 200, r.text
            print(f"Created model {body.uri} in project {body.project}")
            created_models.append(convert_to_json(body))
    except Exception as e:
        print(e)

    print(created_models)
    yield created_models
    for created_model in created_models:
        uri, project = created_model["URI"], created_model["project"]
        print(f"--> Deleting {uri} in project {project}")
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


def create_deployment(test_model: TestModel, model_uri: str, deploy_url, auth_header, artifact_type: str = "SKLEARN"):
    deployments_url = get_deployments_url(test_model, deploy_url)

    create_params = {"action": "create", "messages": "creating test deployment"}

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
        }
    }

    result = requests.post(deployments_url, json=deployment, params=create_params, headers=auth_header, verify=False)
    assert result.status_code == 200, result.text

    def created():
        r = requests.get(os.path.join(deployments_url, f"{test_model.value}-test-deployment"), headers=auth_header, verify=False)
        assert r.status_code == 200, result.text
        status = r.json().get("status", None)
        assert status is not None
        state = status.get("state", None)
        assert state is not None
        assert state == "Available"

    print("WAIT FOR BACKOFF")

    retry_with_backoff(created, backoff_in_seconds=3)
    return deployment


def delete_deployment(test_model: TestModel, deploy_url, auth_header):
    deployments_url = get_deployments_url(test_model, deploy_url)
    deployment_url = os.path.join(deployments_url, f"{test_model.value}-test-deployment")
    result = requests.delete(deployment_url, params={"action": "delete", "messages": "delete test deployment"},
                             headers=auth_header, verify=False)
    assert result.status_code == 200, result.text


@pytest.fixture(scope="function")
def deployment(request, created_models, model_metadata, deploy_url_and_auth_header):
    print("CREATING DEPLOYMENT")
    test_model = request.param
    deploy_url, auth_header = deploy_url_and_auth_header
    model_uri = model_metadata.get(test_model)["uri"]
    try:
        print("CREATING DEPLOYMENT...")
        deployment = create_deployment(test_model, model_uri, deploy_url, auth_header)

    except Exception as e:
        print(e)
        delete_deployment(test_model, deploy_url, auth_header)
        raise(e)

    yield deployment
    delete_deployment(test_model, deploy_url, auth_header)


def test_models(created_models, model_metadata, deploy_url_and_auth_header):
    deploy_url, auth_header = deploy_url_and_auth_header
    test_model = TestModel.IRIS2
    model_uri = model_metadata.get(test_model)["uri"]
    try:
        create_deployment(test_model, model_uri, deploy_url, auth_header)
        import time
        time.sleep(100)
    except Exception as e:
        print(e)
        raise(e)
    finally:
        delete_deployment(test_model, deploy_url, auth_header)


def seldon_iris_batch(_: RequestType):
    test_model = TestModel.IRIS
    deployment_name = f"{test_model.value}-test-deployment"

    request_data = '{"data":{"names":["SL","SW","PL","PW"],' + \
                   '"ndarray":[[6.8,2.8,4.8,1.4],[6.1,3.4,4.5,1.6]]}}'
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Type": "io.seldon.serving.inference.request",
        "Ce-Namespace": TEST_NAMESPACE,
        "Ce-Inferenceservicename": deployment_name,
        "Ce-Endpoint": "default",
        "Ce-Requestid": "m1",
    }
    expected_index = "inference-log-seldon-seldon-iris-test-deployment-default"
    expected_elastic_docs = {
        "m1-item-0": {
            'request': {
                'elements': {"Sepal Length": 6.8, "Sepal Width": 2.8, "Petal Length": 4.8, "Petal Width": 1.4},
                'instance': [6.8, 2.8, 4.8, 1.4],
                'dataType': 'tabular',
                'names': ["SL","SW","PL","PW"],
                'payload': {"data": {"names": ["SL","SW","PL","PW"],
                                     "ndarray": [[6.8, 2.8, 4.8, 1.4], [6.1, 3.4, 4.5, 1.6]]}},
            },
            'ServingEngine': 'seldon',
            "Ce-Namespace": TEST_NAMESPACE,
            "Ce-Inferenceservicename": deployment_name,
            "Ce-Endpoint": "default",
            'RequestId': 'm1'
        },
        "m1-item-1": {
            'request': {
                'elements': {"Sepal Length": 6.1, "Sepal Width": 3.4, "Petal Length": 4.5, "Petal Width": 1.6},
                'instance': [6.1, 3.4, 4.5, 1.6],
                'dataType': 'tabular',
                'names': ["SL","SW","PL","PW"],
                'payload': {"data": {"names": ["SL","SW","PL","PW"],
                                     "ndarray": [[6.8, 2.8, 4.8, 1.4], [6.1, 3.4, 4.5, 1.6]]}},
            },
            'ServingEngine': 'seldon',
            "Ce-Namespace": TEST_NAMESPACE,
            "Ce-Inferenceservicename": deployment_name,
            "Ce-Endpoint": "default",
            'RequestId': '2z7'
        }
    }
    return request_headers, request_data, expected_index, expected_elastic_docs


def seldon_iris_not_batch(_: RequestType):
    request_data = '{"data":{"names":["Sepal length","Sepal width","Petal length","Petal Width"],' + \
                   '"ndarray":[[6.3,2.8,4.8,1.4]]}}'
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Type": "io.seldon.serving.inference.request",
        "Ce-Namespace": "seldon",
        "Ce-Inferenceservicename": "iris",
        "Ce-Endpoint": "default",
        "Ce-Requestid": "2z8",
    }
    expected_index = "inference-log-seldon-seldon-iris-default"
    expected_elastic_docs = {
        "2z8": {
            'request': {
                'elements': {"Sepal length": 6.3, "Sepal width": 2.8, "Petal length": 4.8, "Petal Width": 1.4},
                'instance': [6.3, 2.8, 4.8, 1.4],
                'dataType': 'tabular',
                'names': ["Sepal length", "Sepal width", "Petal length", "Petal Width"],
                'payload': {"data": {"names": ["Sepal length", "Sepal width", "Petal length", "Petal Width"],
                                     "ndarray": [[6.3, 2.8, 4.8, 1.4]]}},
            },
            'ServingEngine': 'seldon',
            "Ce-Namespace": "seldon",
            "Ce-Inferenceservicename": "iris",
            "Ce-Endpoint": "default",
            'RequestId': '2z8'
        }
    }
    return request_headers, request_data, expected_index, expected_elastic_docs
