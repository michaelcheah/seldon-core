# This is basically an integration test
import copy
import os
import time

import pytest
import requests
from requests.exceptions import ConnectionError
from tests import no_metadata, metadata
from typing import Dict
import json
from tests.utils import RequestType


def is_responsive(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except ConnectionError as e:
        return False


@pytest.fixture(scope="session")
def elastic_service(docker_services, docker_ip):
    """Ensure that elastic service is up and responsive."""

    # `port_for` takes a container port and returns the corresponding host port
    elastic_port = docker_services.port_for("elastic_svc", 9200)
    elastic_url = f"http://{docker_ip}:{elastic_port}"
    check_url = f"{elastic_url}/_cat/indices"
    try:
        docker_services.wait_until_responsive(
            timeout=90.0, pause=0.1, check=lambda: is_responsive(check_url)
        )
        # Empty the database
        e = requests.delete(os.path.join(elastic_url, "_all"), params={"format": "json"})
        assert e.status_code == 200, e.text
        yield elastic_url
    except Exception as e:
        logs = docker_services._docker_compose.execute("logs elastic_svc").decode("utf-8")
        print(logs)
        raise e


@pytest.fixture(scope="session")
def logger_service(docker_services, docker_ip, elastic_service):
    logger_port = docker_services.port_for("request_logger", 8080)
    # logger_port = docker_services.port_for("request_logger", 8080)
    logger_url = f"http://{docker_ip}:{logger_port}/"
    check_url = f"{logger_url}status"
    try:
        docker_services.wait_until_responsive(
            timeout=270.0, pause=0.1, check=lambda: is_responsive(check_url)
        )
        yield logger_url
    except Exception as e:
        logs = docker_services._docker_compose.execute("logs request_logger").decode("utf-8")
        print(logs)
        raise e


def run_checks(logger_url, elastic_url, request_headers, request_data, expected_index, expected_elastic_docs):
    r = requests.post(logger_url, data=request_data, headers=request_headers)
    assert r.status_code == 200, f"{r.text}: {logger_url}, {request_data}, {request_headers}"

    e = requests.get(os.path.join(elastic_url, "_cat", "indices"), params={"format": "json"})
    assert e.status_code == 200, e.text
    indices = [i["index"] for i in e.json()]
    assert expected_index in indices, f"{expected_index} not found in {indices} ({expected_index})"

    d = requests.get(os.path.join(elastic_url, expected_index, "_search"), params={"format": "json"})
    assert d.status_code == 200, d.text
    doc_ids = [i["_id"] for i in d.json()["hits"]["hits"]]

    for expected_id in expected_elastic_docs:
        assert expected_id in doc_ids, f"{expected_index} not in {doc_ids}"
        i = requests.get(os.path.join(elastic_url, expected_index, "_source", expected_id))
        assert i.status_code == 200, i.text
        item = i.json()

        # assert expected_elastic_docs[expected_id].items() <= item.items()
        for key in expected_elastic_docs[expected_id]:
            assert key in item, f"no key {key} found in {expected_id} doc: {item.keys()}"

            val = item[key]
            expected_val = expected_elastic_docs[expected_id][key]

            if key in ["request"] and expected_val["dataType"] == "image":
                if type(expected_val) == dict:
                    new_expected_val = copy.deepcopy(expected_val)
                    if "ce-time" in val:
                        new_expected_val["ce-time"] = val["ce-time"]

                    if "names" in val and "names" not in new_expected_val:
                        new_expected_val["names"] = val["names"]
                    check_image_request_fields(val, new_expected_val)
                else:
                    check_image_request_fields(val, new_expected_val)

            else:
                if type(expected_val) == dict:
                    new_expected_val = copy.deepcopy(expected_val)
                    if "ce-time" in val:
                        new_expected_val["ce-time"] = val["ce-time"]

                    if "names" in val and "names" not in new_expected_val:
                        new_expected_val["names"] = val["names"]
                    assert val == new_expected_val, f"values at {key} in {expected_id} docs don't match"
                else:
                    assert val == expected_val, f"values at {key} in {expected_id} docs don't match"


def check_image_request_fields(instance: Dict, expected: Dict):
    assert set(instance.keys()) == set(expected.keys())
    for key in expected:
        if key in ["instance", "payload", "elements"] and expected[key] is not None:
            expected_json = json.dumps(expected[key], sort_keys=True, indent=None).replace("\n", "").replace(" ", "").replace("0.0,", "0,").replace("0.0]", "0]").replace("1.0", "1")
            instance_json = json.dumps(instance[key], sort_keys=True, indent=None).replace("\n", "").replace(" ", "").replace("0.0,", "0,").replace("0.0]", "0]").replace("1.0", "1")
            assert instance_json == expected_json, f"values at '{key}' in request don't match"
        else:
            assert instance[key] == expected[key], f"values at '{key}' in request not matching"


@pytest.mark.parametrize("scenario, request_type", [
    # (no_metadata.seldon_tensor, None),
    # (no_metadata.batch_seldon_ndarray, RequestType.REQUEST),
    # (no_metadata.batch_seldon_ndarray, RequestType.RESPONSE),
    # (no_metadata.batch_ndarray_string, RequestType.REQUEST),
    # (no_metadata.batch_ndarray_string, RequestType.RESPONSE),
    # (no_metadata.two_batches_tabular, None),
    # (no_metadata.one_per_batch_tensor, None),
    # (no_metadata.single_movie_sentiment_text, RequestType.REQUEST),
    # (no_metadata.single_movie_sentiment_text, RequestType.RESPONSE),
    # (no_metadata.batch_movie_sentiment_text, RequestType.REQUEST),
    # (no_metadata.batch_movie_sentiment_text, RequestType.RESPONSE),
    (no_metadata.kfserving_tensor_iris_batch_of_two, RequestType.REQUEST),
    (no_metadata.kfserving_tensor_iris_batch_of_two, RequestType.RESPONSE),
    # (no_metadata.kfserving_tensor_iris_batch_of_two, RequestType.OUTLIER),
    (no_metadata.kfserving_cifar10, RequestType.REQUEST),
    (no_metadata.kfserving_cifar10, RequestType.RESPONSE),
    # # (no_metadata.kfserving_cifar10, RequestType.OUTLIER),
    # (no_metadata.dummy_tabular, None),
    # # (no_metadata.json_data, RequestType.REQUEST),
    # # (no_metadata.json_data, RequestType.RESPONSE),
    # (no_metadata.tabular_input_multiple_output, RequestType.REQUEST),
    # (no_metadata.tabular_input_multiple_output, RequestType.RESPONSE),
    # (no_metadata.image_input, None),
    # (no_metadata.seldon_income_classifier, RequestType.REQUEST),
    # (no_metadata.seldon_income_classifier, RequestType.RESPONSE),
    # (no_metadata.mix_one_hot_categorical_float, RequestType.REQUEST),
    # (no_metadata.mix_one_hot_categorical_float, RequestType.RESPONSE),
    # (no_metadata.seldon_cifar10_image, RequestType.REQUEST),
    # (no_metadata.seldon_cifar10_image, RequestType.RESPONSE),
    # # (no_metadata.seldon_cifar10_image, RequestType.OUTLIER),
    # (no_metadata.seldon_iris_batch, None),
    # (no_metadata.seldon_iris_not_batch, None),
    (no_metadata.kfserving_income, None),
    # (no_metadata.seldon_drift, None),  # TODO: Finish this
])
def test_no_metadata(logger_service, elastic_service, scenario, request_type, docker_services):
    data, headers, expected_index, expected_docs = scenario(request_type)
    try:
        run_checks(
            logger_service, elastic_service, data, headers, expected_index, expected_docs
        )
    except AssertionError as e:
        # For debugging. Left for future reference
        logs = docker_services._docker_compose.execute("logs request_logger").decode("utf-8")
        print(logs)
        raise e


# @pytest.mark.parametrize("scenario, request_type", [
#     (metadata.seldon_iris_batch, None),
#     (metadata.seldon_iris_not_batch, None),
# ])
# def test_seldon_iris_metadata(logger_service, elastic_service, scenario, request_type, docker_services,
#                               seldon_iris_deployment):
#     data, headers, expected_index, expected_docs = scenario(request_type)
#     try:
#         run_checks(
#             logger_service, elastic_service, data, headers, expected_index, expected_docs
#         )
#     except AssertionError as e:
#         # For debugging. Left for future reference
#         logs = docker_services._docker_compose.execute("logs request_logger").decode("utf-8")
#         print(logs)
#         raise e


# @pytest.mark.parametrize("scenario, request_type", [
#     (metadata.kfserving_iris_batch, RequestType.REQUEST),
#     (metadata.kfserving_iris_batch, RequestType.RESPONSE),
#     (metadata.kfserving_iris_batch, RequestType.RESPONSE),
# ])
# def test_kfserving_iris_metadata(logger_service, elastic_service, scenario, request_type, docker_services,
#                                  kfserving_iris_deployment):
#     data, headers, expected_index, expected_docs = scenario(request_type)
#     try:
#         run_checks(
#             logger_service, elastic_service, data, headers, expected_index, expected_docs
#         )
#     except AssertionError as e:
#         # For debugging. Left for future reference
#         logs = docker_services._docker_compose.execute("logs request_logger").decode("utf-8")
#         print(logs)
#         raise e


# @pytest.mark.parametrize("scenario, request_type", [
#     (metadata.kfserving_iris_batch, RequestType.REQUEST),
#     (metadata.kfserving_iris_batch, RequestType.RESPONSE),
# ])
# def test_kfserving_iris_metadata(logger_service, elastic_service, scenario, request_type, docker_services,
#                                  kfserving_iris_deployment):
#     data, headers, expected_index, expected_docs = scenario(request_type)
#     try:
#         run_checks(
#             logger_service, elastic_service, data, headers, expected_index, expected_docs
#         )
#     except AssertionError as e:
#         # For debugging. Left for future reference
#         logs = docker_services._docker_compose.execute("logs request_logger").decode("utf-8")
#         print(logs)
#         raise e



# @pytest.mark.parametrize("scenario, request_type", [
#     (metadata.seldon_moviesentiment_text_no_batch, RequestType.REQUEST),
#     (metadata.seldon_moviesentiment_text_no_batch, RequestType.RESPONSE),
#     (metadata.seldon_moviesentiment_text_batch, RequestType.REQUEST),
#     (metadata.seldon_moviesentiment_text_batch, RequestType.RESPONSE),
# ])
# def test_moviesentiment_text_metadata(logger_service, elastic_service, scenario, request_type, docker_services,
#                                     seldon_moviesentiment_deployment):
#     data, headers, expected_index, expected_docs = scenario(request_type)
#     try:
#         run_checks(
#             logger_service, elastic_service, data, headers, expected_index, expected_docs
#         )
#     except AssertionError as e:
#         # For debugging. Left for future reference
#         logs = docker_services._docker_compose.execute("logs request_logger").decode("utf-8")
#         print(logs)
#         raise e

# @pytest.mark.parametrize("scenario, request_type", [
#     (metadata.seldon_income_batch, RequestType.REQUEST),
#     (metadata.seldon_income_batch, RequestType.RESPONSE),
#     (metadata.seldon_income_batch, RequestType.REFERENCE_REQUEST),
# ])
# def test_seldon_income_metadata(logger_service, elastic_service, scenario, request_type, docker_services,
#                                     seldon_income_deployment):
#     data, headers, expected_index, expected_docs = scenario(request_type)
#     try:
#         run_checks(
#             logger_service, elastic_service, data, headers, expected_index, expected_docs
#         )
#     except AssertionError as e:
#         # For debugging. Left for future reference
#         logs = docker_services._docker_compose.execute("logs request_logger").decode("utf-8")
#         print(logs)
#         raise e


# @pytest.mark.parametrize("scenario, request_type", [
#     # (metadata.seldon_cifar10_single, RequestType.REQUEST), # expected to fail
#     (metadata.seldon_cifar10_single, RequestType.RESPONSE),
#     # (metadata.seldon_cifar10_batch, RequestType.REQUEST),
#     (metadata.seldon_cifar10_batch, RequestType.RESPONSE),
# ])
# def test_seldon_cifar10_metadata(logger_service, elastic_service, scenario, request_type, docker_services,
#                                  seldon_cifar10_deployment):
#     data, headers, expected_index, expected_docs = scenario(request_type)
#     try:
#         run_checks(
#             logger_service, elastic_service, data, headers, expected_index, expected_docs
#         )
#     except AssertionError as e:
#         # For debugging. Left for future reference
#         logs = docker_services._docker_compose.execute("logs request_logger").decode("utf-8")
#         print(logs)
#         raise e
