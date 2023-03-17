import requests
import json


def list_models(http_port: int = 8080):
    return requests.post(f"http://localhost:{http_port}/v2/repository/index", json={})


def send_request(request: dict, model: str, http_port: int = 8080):
    return requests.post(f"http://localhost:{http_port}/v2/models/{model}/infer", json=request)


def income_explainer(save=False):
    request = {"inputs": [
        {"data": [52, 4, 0, 2, 8, 4, 2, 0, 0, 0, 60, 9], "datatype": "INT64", "name": "income", "shape": [1, 12]}]}
    resp = send_request(request, "income_kernel_shap", 8083)
    print(resp.json())
    data = json.loads(resp.json()["outputs"][0]["data"][0])

    if save:
        with open("kernel-shap-data.json", "w") as f:
            json.dump(data, f, indent=2)


def income_ohe_explainer():
    OHE_REQUEST = {'inputs': [{'name': 'income',
                               'data': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                               'datatype': 'INT64', 'shape': [1, 53]}]}

    OHE_REQUEST_2 = {
        'inputs': [
            {"name": "Workclass", "data": [0, 0, 0, 0, 1, 0, 0, 0, 0], "datatype": "INT64", "shape": [1, 8]},
            {"name": "Education", "data": [0, 0, 0, 0, 1, 0, 0], "datatype": "INT64", "shape": [1, 6]},
            {"name": "Marital Status", "data": [0, 0, 1, 0], "datatype": "INT64", "shape": [1, 4]},
            {"name": "Occupation", "data": [0, 1, 0, 0, 0, 0, 0, 0, 0], "datatype": "INT64", "shape": [1, 9]},
            {"name": "Relationship", "data": [0, 0, 0, 0, 1, 0], "datatype": "INT64", "shape": [1, 6]},
            {"name": "Race", "data": [0, 0, 0, 0, 1], "datatype": "INT64", "shape": [1, 5]},
            {"name": "Sex", "data": [1, 0], "datatype": "INT64", "shape": [1, 2]},
            {"name": "Country", "data": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], "datatype": "INT64", "shape": [1, 11]},
        ],
        "parameters": {"content_type": "pd"},
    }

    OHE_REQUEST_V1 = {"data": {
        "ndarray": [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
    }}

    resp = send_request(OHE_REQUEST_2, "grouped_kernel_shap_explainer", 8083)
    print(resp.json())
    data = json.loads(resp.json()["outputs"][0]["data"][0])
    print(json.dumps(data, indent=2))

    with open("grouped-kernel-shap-data.json", "w") as f:
        json.dump(data, f, indent=2)


def diabetes_explainer(save: bool = False):
    request = {"inputs": [{"name": "income", "data": [0.01991321, 0.05068012, 0.10480869, 0.0700723, -0.03596778,
                                                      -0.0266789, -0.02499266, -0.00259226, 0.00370906, 0.04034337],
                           "datatype": "FP64", "shape": [1, 10]}]}

    resp = send_request(request, "diabetes_kernel_shap", 8083)
    print(resp.json())
    data = json.loads(resp.json()["outputs"][0]["data"][0])
    print(json.dumps(data, indent=2))
    if save:
        with open("diabetes-kernel-shap-data.json", "w") as f:
            json.dump(data, f, indent=2)


def multi_output_regression_explainer(save: bool = False):
    request = {"inputs": [{"name": "income", "data": [
        -1.175178985629452, 0.5590753767739638, -0.5553606728647008, -2.2114961433016136, -0.18048749685270615,
        0.8584582505947634, 0.5759513849213382, -0.1992742339964766, 0.9179546060506324, -0.5453976415950043,
        -0.1128789275859155, 1.1652381784847645, 2.2770773504411337, 1.8555424006857106, 1.6242084607123264,
        1.3373774213629348, 0.012160432383094865, -0.7659805105165982, 0.30396475467854656, 0.23524098720880424],
                           "datatype": "FP64", "shape": [1, 20]}]}

    resp = send_request(request, "multi_output_regression_kernel_shap", 8083)
    print(resp.json())
    data = json.loads(resp.json()["outputs"][0]["data"][0])
    print(json.dumps(data, indent=2))
    if save:
        with open("diabetes-kernel-shap-data.json", "w") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    print(list_models().json(), list_models(8083).json())

    income_explainer()
    # diabetes_explainer()
    # multi_output_regression_explainer()


    # request = {"inputs": [
    #     {"data": [52, 4, 0, 2, 8, 4, 2, 0, 0, 0, 60, 9], "datatype": "INT64", "name": "income", "shape": [1, 12]}]}
    # resp = send_request(request, "income_random_forest", 8080)
    # print(resp.json())
    # data = json.loads(resp.json()["outputs"][0]["data"][0])