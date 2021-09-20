# # This is basically an integration test
# import os
# import pytest
# import requests
# # from tests.services import logger_service, elastic_service
#
#
# def is_responsive(url):
#     try:
#         response = requests.get(url)
#         if response.status_code == 200:
#             return True
#         print(response.json())
#     except ConnectionError as e:
#         return False
#
#
# @pytest.fixture(scope="session")
# def elastic_service(docker_services, docker_ip):
#     """Ensure that elastic service is up and responsive."""
#
#     # `port_for` takes a container port and returns the corresponding host port
#     elastic_port = docker_services.port_for("elastic_svc", 9200)
#     elastic_url = f"http://{docker_ip}:{elastic_port}"
#     check_url = f"{elastic_url}/_cat/indices"
#     import time
#     time.sleep(5)
#     try:
#         docker_services.wait_until_responsive(
#             timeout=60.0, pause=1, check=lambda: is_responsive(check_url)
#         )
#         yield elastic_url
#     except Exception as e:
#         logs = docker_services._docker_compose.execute("logs elastic_svc").decode("utf-8")
#         print(logs)
#         raise e
#
#
# @pytest.fixture(scope="session")
# def logger_service(docker_services, docker_ip, elastic_service):
#     logger_port = docker_services.port_for("request_logger", 2222)
#     logger_url = f"http://{docker_ip}:{logger_port}/"
#     check_url = f"{logger_url}status"
#
#     try:
#         docker_services.wait_until_responsive(
#             timeout=30.0, pause=0.1, check=lambda: is_responsive(check_url)
#         )
#         yield logger_url
#     except Exception as e:
#         logs = docker_services._docker_compose.execute("logs request_logger").decode("utf-8")
#         print(logs)
#         raise e
#
#
# @pytest.mark.xfail(reason="tensor logging is broken", strict=True)
# def test_tensor(logger_service, elastic_service):
#     headers = {
#         "Content-Type": "application/json",
#         "Ce-Inferenceservicename": "tensor",
#         "Ce-Type": "io.seldon.serving.inference.request",
#         "Ce-Requestid": "1a",
#     }
#
#     r = requests.post(logger_service, '{"data":{"names":["a","b"],"tensor":{"shape":[2,2],"values":[1,2,3,4]}}}',
#                       headers=headers)
#     assert r.status_code == 200
#
#     expected_index = "inference-log-seldon-unknown-namespace-tensor-unknown-endpoint"
#
#     e = requests.get(os.path.join(elastic_service, "_cat", "indices"), params={"format": "json"})
#     assert e.status_code == 200
#     indices = [i["index"] for i in e.json()]
#     assert expected_index in indices
#
#     d = requests.get(os.path.join(elastic_service, "_search"), data={})
#     assert d.status_code == 200
#
#     hits = d.json()["hits"]
#     assert len(hits["hits"]) == 1
#
#
#
# def test_ndarray_no_metadata(logger_service, elastic_service):
#     headers = {
#         "Content-Type": "application/json",
#         "Ce-Inferenceservicename": "ndarray",
#         "Ce-Type": "io.seldon.serving.inference.request",
#         "Ce-Requestid": "2b",
#     }
#
#     r = requests.post(logger_service, '{"data":{"names":["a","b"],"ndarray":[[1,2],[3,4]]}}',
#                       headers=headers)
#     assert r.status_code == 200
#
#     expected_index = "inference-log-seldon-unknown-namespace-ndarray-unknown-endpoint"
#
#     e = requests.get(os.path.join(elastic_service, "_cat", "indices"), params={"format": "json"})
#     assert e.status_code == 200
#     indices = [i["index"] for i in e.json()]
#     assert expected_index in indices
#
#     d = requests.get(os.path.join(elastic_service, expected_index, "_search"), data={})
#     assert d.status_code == 200
#     hits = d.json()["hits"]
#     assert len(hits["hits"]) == 2
#
#     i0 = requests.get(os.path.join(elastic_service, expected_index, "_source", "2b-item-0"))
#     assert i0.status_code == 200
#     item0 = i0.json()
#
#     expected_item0 = {
#         'request': {
#             'elements': {'a': 1.0, 'b': 2.0},
#             'instance': [1.0, 2.0],
#             'dataType': 'tabular',
#             'names': ['a', 'b'],
#             'payload': {'data': {'names': ['a', 'b'], 'ndarray': [[1, 2], [3, 4]]}}
#         },
#         'ServingEngine': 'seldon',
#         'Ce-Inferenceservicename': 'ndarray',
#         'RequestId': '2b'
#     }
#
#     for key in expected_item0:
#         assert item0[key] == expected_item0[key]