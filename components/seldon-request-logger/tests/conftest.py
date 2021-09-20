import pytest
import requests

pytest_plugins = "tests.metadata"

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
#     try:
#         docker_services.wait_until_responsive(
#             timeout=60.0, pause=0.1, check=lambda: is_responsive(check_url)
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
