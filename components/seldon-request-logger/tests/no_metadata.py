import requests
import os
import json
from enum import Enum
from typing import Dict, Any
from tests.utils import RequestType

def seldon_tensor(_: RequestType):
    request_data = '{"data":{"names":["a","b"],"tensor":{"shape":[2,2],"values":[1,2,3,4]}}}'
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Inferenceservicename": "tensor",
        "Ce-Type": "io.seldon.serving.inference.request",
        "Ce-Requestid": "1a",
    }
    expected_index = "inference-log-seldon-unknown-namespace-tensor-unknown-endpoint"

    expected_elastic_docs = {
        "1a-item-0": {
            "request": {
                'elements': {'a': 1.0, 'b': 2.0},
                'instance': [1.0, 2.0],
                "dataType": "tabular",
                'names': ['a', 'b'],
                'payload': {
                    'data': {'names': ['a', 'b'], 'tensor': {'shape': [2, 2], 'values': [1, 2, 3, 4]}},
                }
            }
        },
        "1a-item-1": {
            "request": {
                'elements': {'a': 3.0, 'b': 4.0},
                'instance': [3.0, 4.0],
                "dataType": "tabular",
                'names': ['a', 'b'],
                'payload': {
                    'data': {'names': ['a', 'b'], 'tensor': {'shape': [2, 2], 'values': [1, 2, 3, 4]}},
                }
            },
        },
    }
    return request_headers, request_data, expected_index, expected_elastic_docs


def batch_seldon_ndarray(request_type: RequestType):
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Inferenceservicename": "ndarray",
        "Ce-Requestid": "2b",
    }
    expected_index = "inference-log-seldon-unknown-namespace-ndarray-unknown-endpoint"
    expected_elastic_docs: Dict[str, Any] = {
        "2b-item-0": {

            'ServingEngine': 'seldon',
            'Ce-Inferenceservicename': 'ndarray',
            'RequestId': '2b'
        },
        "2b-item-1": {
            'ServingEngine': 'seldon',
            'Ce-Inferenceservicename': 'ndarray',
            'RequestId': '2b'
        },
    }

    if request_type == RequestType.REQUEST:
        request_data = '{"data":{"names":["a","b"],"ndarray":[[1,2],[3,4]]}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.request"
        expected_elastic_docs["2b-item-0"]["request"] = {
            'elements': {'a': 1.0, 'b': 2.0},
            'instance': [1.0, 2.0],
            'dataType': 'tabular',
            'names': ['a', 'b'],
            'payload': {'data': {'names': ['a', 'b'], 'ndarray': [[1, 2], [3, 4]]}}
        }
        expected_elastic_docs["2b-item-1"]["request"] = {
            'elements': {'a': 3.0, 'b': 4.0},
            'instance': [3.0, 4.0],
            'dataType': 'tabular',
            'names': ['a', 'b'],
            'payload': {'data': {'names': ['a', 'b'], 'ndarray': [[1, 2], [3, 4]]}}
        }
    else:
        request_data = '{"data":{"names":["c"],"ndarray":[[7],[8]]}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.response"
        expected_elastic_docs["2b-item-0"]["response"] = {
            'elements': {'c': 7.0},
            'instance': [7.0],
            'dataType': 'number',
            'names': ['c'],
            'payload': {'data': {'names': ['c'], 'ndarray': [[7], [8]]}}
        }
        expected_elastic_docs["2b-item-1"]["response"] = {
            'elements': {'c': 8.0},
            'instance': [8.0],
            'dataType': 'number',
            'names': ['c'],
            'payload': {'data': {'names': ['c'], 'ndarray': [[7], [8]]}}
        }

    return request_headers, request_data, expected_index, expected_elastic_docs


def batch_ndarray_string(request_type: RequestType):
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Namespace": "default",
        "Ce-Endpoint": "example",
        "Ce-RequestId": "3c",
    }
    expected_index = "inference-log-seldon-default-unknown-inferenceservice-example"
    expected_elastic_docs: Dict[str, Any] = {
        "3c-item-0": {
            'ServingEngine': 'seldon',
            "Ce-Namespace": "default",
            "Ce-Endpoint": "example",
            'RequestId': '3c'
        },
        "3c-item-1": {
            'ServingEngine': 'seldon',
            "Ce-Namespace": "default",
            "Ce-Endpoint": "example",
            'RequestId': '3c'
        }
    }

    if request_type == RequestType.REQUEST:
        request_data = '{"data":{"names":["a"],"ndarray":["test1","test2"]}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.request"
        expected_elastic_docs["3c-item-0"]["request"] = {
            'elements': {'a': "test1"},
            'instance': "test1",
            'dataType': 'text',
            'names': ['a'],
            'payload': {'data': {'names': ['a'], 'ndarray': ["test1", "test2"]}}
        }
        expected_elastic_docs["3c-item-1"]["request"] = {
            'elements': {'a': "test2"},
            'instance': "test2",
            'dataType': 'text',
            'names': ['a'],
            'payload': {'data': {'names': ['a'], 'ndarray': ["test1", "test2"]}}
        }
    else:
        request_data = '{"data":{"names":["c"],"ndarray":[[7],[8]]}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.response"
        expected_elastic_docs["3c-item-0"]["response"] = {
            'elements': {'c': 7.0},
            'instance': [7.0],
            'dataType': 'number',
            'names': ['c'],
            'payload': {'data': {'names': ['c'], 'ndarray': [[7], [8]]}}
        }
        expected_elastic_docs["3c-item-1"]["response"] = {
            'elements': {'c': 8.0},
            'instance': [8.0],
            'dataType': 'number',
            'names': ['c'],
            'payload': {'data': {'names': ['c'], 'ndarray': [[7], [8]]}}
        }

    return request_headers, request_data, expected_index, expected_elastic_docs


def two_batches_tabular(_: RequestType):
    request_data = '{"data":{"names":["a","b"],"tensor":{"shape":[2,2],"values":[1,2,3,4]}}}'
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Type": "io.seldon.serving.inference.request",
        "Ce-Requestid": "4d",
    }
    expected_index = "inference-log-seldon-unknown-namespace-unknown-inferenceservice-unknown-endpoint"
    expected_elastic_docs = {
        "4d-item-0": {
            'request': {
                'elements': {'a': 1.0, "b": 2.0},
                'instance': [1.0, 2.0],
                'dataType': 'tabular',
                'names': ["a", "b"],
                'payload': {'data': {'names': ["a", "b"], 'tensor': {'shape': [2, 2], 'values': [1, 2, 3, 4]}}},
            },
            'ServingEngine': 'seldon',
            'RequestId': '4d'
        },
        "4d-item-1": {
            'request': {
                'elements': {'a': 3.0, "b": 4.0},
                'instance': [3.0, 4.0],
                'dataType': 'tabular',
                'names': ["a", "b"],
                'payload': {'data': {'names': ["a", "b"], 'tensor': {'shape': [2, 2], 'values': [1, 2, 3, 4]}}},
            },
            'ServingEngine': 'seldon',
            'RequestId': '4d'
        }
    }
    return request_headers, request_data, expected_index, expected_elastic_docs


def one_per_batch_tensor(_: RequestType):
    request_data = '{"data":{"names":["c"],"tensor":{"shape":[2,1],"values":[5,6]}}}'
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Inferenceservicename": "tensor",
        "Ce-Type": "io.seldon.serving.inference.request",
        "Ce-Requestid": "5e",
    }
    expected_index = "inference-log-seldon-unknown-namespace-tensor-unknown-endpoint"
    expected_elastic_docs = {
        "5e-item-0": {
            'request': {
                'elements': {'c': 5.0},
                'instance': [5.0],
                'dataType': 'number',
                'names': ["c"],
                'payload': {'data': {'names': ["c"], 'tensor': {'shape': [2, 1], 'values': [5, 6]}}},
            },
            "Ce-Inferenceservicename": "tensor",
            'ServingEngine': 'seldon',
            'RequestId': '5e',
        },
        "5e-item-1": {
            'request': {
                'elements': {'c': 6.0},
                'instance': [6.0],
                'dataType': 'number',
                'names': ["c"],
                'payload': {'data': {'names': ["c"], 'tensor': {'shape': [2, 1], 'values': [5, 6]}}},
            },
            "Ce-Inferenceservicename": "tensor",
            'ServingEngine': 'seldon',
            'RequestId': '5e',
        }
    }
    return request_headers, request_data, expected_index, expected_elastic_docs


def batch_movie_sentiment_text(request_type: RequestType):
    request_headers = {
        "Ce-Inferenceservicename": "moviesentiment",
        "Content-Type": "application/json",
        "Ce-Requestid": "6f"
    }
    expected_index = "inference-log-seldon-unknown-namespace-moviesentiment-unknown-endpoint"
    expected_elastic_docs: Dict[str, Any] = {
        "6f": {
            'ServingEngine': 'seldon',
            "Ce-Inferenceservicename": "moviesentiment",
            'RequestId': '6f'
        },
    }

    if request_type == RequestType.REQUEST:
        request_data = '{"data": {"names": ["Text review"],"ndarray": ["this film has bad actors"]}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.request"
        expected_elastic_docs["6f"]["request"] = {
            'elements': {'Text review': "this film has bad actors"},
            'instance': "this film has bad actors",
            'dataType': 'text',
            'names': ['Text review'],
            'payload': {"data": {"names": ["Text review"], "ndarray": ["this film has bad actors"]}}
        }
    else:
        request_data = '{"data":{"names":["t0","t1"],"ndarray":[[0.5,0.5]]}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.response"
        expected_elastic_docs["6f"]["response"] = {
            'elements': {'t0': 0.5, "t1": 0.5},
            'instance': [0.5, 0.5],
            'dataType': 'tabular',
            'names': ['t0', 't1'],
            'payload': {'data': {'names': ['t0', 't1'], 'ndarray': [[0.5, 0.5]]}}
        }
    return request_headers, request_data, expected_index, expected_elastic_docs


def string_data(_: RequestType):
    payload_data = {
        "columns": ["DISPO_CD", "ENG_CD", "HUE_CD", "SALE_OFFER_CD", "SHADE_CD", "TRGTPRCE_MDLGRP_CD",
                    "TRGT_CUST_GROUP_CD", "TRG_CATG", "VIN", "calc_cd", "category", "color", "cond_cd", "country",
                    "cust_cd", "default_cond_cd", "dispo_date", "dispo_day", "drivetype", "floor_price", "mlge_arriv",
                    "mlge_dispo", "model", "modelyr", "region", "saleloc", "series_cd", "sys_enter_date", "tag",
                    "target_price", "v47", "v62", "v64", "vehvalue", "warranty_age", "wrstdt", "wsd"], "index": [0],
        "data": [[3, "L", "RD", "CAO", "DK", 41, 1, "RTR", "MAJ6P1CL3JC166908", None, "RPO", "RR", 5, "A", 7, 3,
                  "2018-07-11", 6766, None, 0.0, 2013, 2013, "ECO", 2018, 1, "C63", "P1C", "2018-06-16", None, 0.0, "5",
                  None, "5", "ecosport", 146.0, "2018-02-15", 26750.56]]}

    payload = {
        'strData': json.dumps(
            payload_data, indent=None
        ).replace("None", "null").replace("\n", "").replace(" ", "")}

    request_data = json.dumps(payload, indent=None).replace(" ", "")
    assert request_data.replace(" ", "") == json.dumps(payload, indent=None).replace(" ", "")

    request_headers = {
        "Content-Type": "application/json",
        "Ce-Inferenceservicename": "strdata",
        "Ce-Type": "io.seldon.serving.inference.request",
        "Ce-Requestid": "7g",
    }
    expected_index = "inference-log-seldon-unknown-namespace-strdata-unknown-endpoint"
    expected_elastic_docs = {
        "7g": {
            "request": {
                'dataType': 'text',
                'instance': payload['strData'],
                'payload': payload,
            },
        }
    }
    return request_headers, request_data, expected_index, expected_elastic_docs


def kfserving_tensor_iris_batch_of_two(request_type: RequestType):
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Namespace": "seldon",
        "Ce-Inferenceservicename": "iris-kf",
        "Ce-Endpoint": "default",
        "Ce-Requestid": "8h",
    }
    expected_index = "inference-log-inferenceservice-seldon-iris-kf-default"
    expected_elastic_docs: Dict[str, Any] = {
        "8h-item-0": {
            'ServingEngine': 'inferenceservice',
            "Ce-Namespace": "seldon",
            "Ce-Inferenceservicename": "iris-kf",
            "Ce-Endpoint": "default",
            "RequestId": "8h",
        },
        "8h-item-1": {
            'ServingEngine': 'inferenceservice',
            "Ce-Namespace": "seldon",
            "Ce-Inferenceservicename": "iris-kf",
            "Ce-Endpoint": "default",
            "RequestId": "8h",
        }
    }

    if request_type == RequestType.REQUEST:
        request_data = '{"instances": [[6.8,  2.8,  4.8,  1.4],[6.0,  3.4,  4.5,  1.6]]}'
        request_headers["Ce-Type"] = "org.kubeflow.serving.inference.request"
        expected_elastic_docs["8h-item-0"]["request"] = {
            'elements': None,
            'instance': [6.8, 2.8, 4.8, 1.4],
            'dataType': 'tabular',
            'payload': {'instances': [[6.8, 2.8, 4.8, 1.4], [6.0, 3.4, 4.5, 1.6]]},
        }
        expected_elastic_docs["8h-item-1"]["request"] = {
            'elements': None,
            'instance': [6.0, 3.4, 4.5, 1.6],
            'dataType': 'tabular',
            'payload': {'instances': [[6.8, 2.8, 4.8, 1.4], [6.0, 3.4, 4.5, 1.6]]},
        }
    elif request_type == RequestType.RESPONSE:
        request_data = '{"predictions": [1,2]}'
        request_headers["Ce-Type"] = "org.kubeflow.serving.inference.response"
        expected_elastic_docs["8h-item-0"]["response"] = {
            'elements': None,
            'instance': 1,
            'dataType': 'number',
            'payload': {'predictions': [1, 2]},
        }
        expected_elastic_docs["8h-item-1"]["response"] = {
            'elements': None,
            'instance': 2,
            'dataType': 'number',
            'payload': {'predictions': [1, 2]},
        }

    elif request_type == RequestType.OUTLIER:
        request_data = '{"data": {"feature_score": null, "instance_score": null, "is_outlier": [1, 0]}, ' + \
                       '"meta": {"name": "OutlierVAE", "detector_type": "offline", "data_type": "image"}}'
        request_headers["Ce-Type"] = "org.kubeflow.serving.inference.outlier"
        expected_elastic_docs["8h-item-0"]["outlier"] = {
            'data': {'feature_score': None, 'instance_score': None, 'is_outlier': 1},
            'meta': {'name': 'OutlierVAE', 'detector_type': 'offline', 'data_type': 'image'},
        }
        expected_elastic_docs["8h-item-1"]["outlier"] = {
            'data': {'feature_score': None, 'instance_score': None, 'is_outlier': 0},
            'meta': {'name': 'OutlierVAE', 'detector_type': 'offline', 'data_type': 'image'},
        }

    else:
        raise Exception("invalid request_type provided")

    return request_headers, request_data, expected_index, expected_elastic_docs


def kfserving_cifar10(request_type: RequestType):
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Namespace": "default",
        "Ce-Inferenceservicename": "cifar10",
        "Ce-Endpoint": "default",
        "CE-SpecVersion": "0.2",
        "Ce-Requestid": "9i",
    }
    expected_index = "inference-log-inferenceservice-default-cifar10-default"
    expected_elastic_docs: Dict[str, Any] = {
        "9i": {
            'ServingEngine': 'inferenceservice',
            "Ce-Namespace": "default",
            "Ce-Inferenceservicename": "cifar10",
            "Ce-Endpoint": "default",
            "RequestId": "9i",
        },
    }

    if request_type == RequestType.REQUEST:
        with open("tests/cifardata.json", "rb") as f:
            raw_data = f.read()
        data = json.loads(raw_data)
        request_data = raw_data
        request_headers["Ce-Type"] = "org.kubeflow.serving.inference.request"
        expected_elastic_docs["9i"]["request"] = {
            'elements': None,
            'instance': data["instances"][0],
            'dataType': 'image',
            'payload': data,
        }
    elif request_type == RequestType.RESPONSE:
        request_data = '{"predictions":[2]}'
        request_headers["Ce-Type"] = "org.kubeflow.serving.inference.response"
        expected_elastic_docs["9i"]["response"] = {
            'elements': None,
            'instance': 2,
            'dataType': 'number',
            'payload': {'predictions': [2]},
        }

    elif request_type == RequestType.OUTLIER:
        request_data = '{"data": {"feature_score": null, "instance_score": null, "is_outlier": [1]}, ' + \
                       '"meta": {"name": "OutlierVAE", "detector_type": "offline", "data_type": "image"}}'
        request_headers["Ce-Type"] = "org.kubeflow.serving.inference.outlier"
        expected_elastic_docs["9i"]["outlier"] = {
            'data': {'feature_score': None, 'instance_score': None, 'is_outlier': 1},
            'meta': {'name': 'OutlierVAE', 'detector_type': 'offline', 'data_type': 'image'},
        }

    else:
        raise Exception("invalid request_type provided")

    return request_headers, request_data, expected_index, expected_elastic_docs


def dummy_tabular(_: RequestType):
    request_data = '{"data": {"names": ["dummy feature"],"ndarray": [1.0]}}'
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Inferenceservicename": "dummytabular",
        "Ce-Type": "io.seldon.serving.inference.request",
        "Ce-Requestid": "10j",
    }
    expected_index = "inference-log-seldon-unknown-namespace-dummytabular-unknown-endpoint"
    expected_elastic_docs = {
        "10j": {
            'request': {
                'elements': {'dummy feature': 1.0},
                'instance': 1.0,
                'dataType': 'number',
                'names': ["dummy feature"],
                'payload': {"data": {"names": ["dummy feature"], "ndarray": [1.0]}},
            },
            "Ce-Inferenceservicename": "dummytabular",
            'ServingEngine': 'seldon',
            'RequestId': '10j',
        },
    }
    return request_headers, request_data, expected_index, expected_elastic_docs


def json_data(request_type: RequestType):
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Inferenceservicename": "jsonexample",
        "Ce-Requestid": "11k",
    }
    expected_index = "inference-log-seldon-unknown-namespace-jsonexample-unknown-endpoint"
    expected_elastic_docs: Dict[str, Any] = {
        "11k": {
            'ServingEngine': 'seldon',
            "Ce-Inferenceservicename": "jsonexample",
            "RequestId": "11k",
        },
    }

    if request_type == RequestType.REQUEST:
        request_data = "{\"jsonData\": {\"input\": \"{'input': '[[53  4  0  2  8  4  2  0  0  0 60  9]]'}\"},\"meta\": {}}"
        request_headers["Ce-Type"] = "io.seldon.serving.inference.request"
        expected_elastic_docs["11k"]["request"] = {
            'instance': {"input": "{'input': '[[53  4  0  2  8  4  2  0  0  0 60  9]]'}"},
            'dataType': 'json',
            'meta': {},
            'payload': {'jsonData': {'input': "{'input': '[[53  4  0  2  8  4  2  0  0  0 60  9]]'}"}, 'meta': {}},
        }
    elif request_type == RequestType.RESPONSE:
        request_data = "{\"jsonData\": {\"input\": \"{'input': '[[53  4  0  2  8  4  2  0  0  0 60  9]]'}\"},\"meta\": {}}"
        request_headers["Ce-Type"] = "io.seldon.serving.inference.response"
        expected_elastic_docs["11k"]["response"] = {
            'instance': {"input": "{'input': '[[53  4  0  2  8  4  2  0  0  0 60  9]]'}"},
            'dataType': 'json',
            'meta': {},
            'payload': {'jsonData': {'input': "{'input': '[[53  4  0  2  8  4  2  0  0  0 60  9]]'}"}, 'meta': {}},
        }
    else:
        raise Exception("invalid request_type provided")

    return request_headers, request_data, expected_index, expected_elastic_docs


def tabular_input_multiple_output(request_type: RequestType):
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Inferenceservicename": "tensor",
        "Ce-Requestid": "2z1",
    }
    expected_index = "inference-log-seldon-unknown-namespace-tensor-unknown-endpoint"
    expected_elastic_docs: Dict[str, Any] = {
        "2z1": {
            'ServingEngine': 'seldon',
            "Ce-Inferenceservicename": "tensor",
            "RequestId": "2z1",
        },
    }

    if request_type == RequestType.REQUEST:
        request_data = '{"inputs":[{"name":"INPUT0","data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],' + \
                       '"datatype":"INT32","shape":[1,16]},' + \
                       '{"name":"INPUT1","data":[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160],' + \
                       '"datatype":"INT32","shape":[1,16]}]}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.request"
        expected_elastic_docs["2z1"]["request"] = {
            'elements': None,
            'instance': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            'dataType': 'tabular',
            'payload': {"inputs": [
                {"name": "INPUT0", "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                 "datatype": "INT32", "shape": [1, 16]},
                {"name": "INPUT1", "data": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
                 "datatype": "INT32", "shape": [1, 16]}]},
        }
    elif request_type == RequestType.RESPONSE:
        request_data = '{"model_name": "simple", "model_version": "1", ' + \
                       '"outputs": [{"name": "OUTPUT0", "datatype": "INT32", ' + \
                       '"shape": [1, 16], "data": [2, 4, 6, 8, 10, 12, 14, 16, 8, 20, 22, 24, 26, 28, 30, 32]}, ' + \
                       '{"name": "OUTPUT1", "datatype": "INT32", "shape": [1, 16], ' + \
                       '"data": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}]}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.response"
        expected_elastic_docs["2z1"]["response"] = {
            'elements': None,
            # TODO: (only first output handled for now) note in the original implementation
            'instance': [2, 4, 6, 8, 10, 12, 14, 16, 8, 20, 22, 24, 26, 28, 30, 32],
            'dataType': 'tabular',
            'payload': {"model_name": "simple", "model_version": "1", "outputs": [
                {"name": "OUTPUT0", "datatype": "INT32", "shape": [1, 16],
                 "data": [2, 4, 6, 8, 10, 12, 14, 16, 8, 20, 22, 24, 26, 28, 30, 32]},
                {"name": "OUTPUT1", "datatype": "INT32", "shape": [1, 16],
                 "data": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}]},
        }
    else:
        raise Exception("invalid request_type provided")

    return request_headers, request_data, expected_index, expected_elastic_docs


def image_input(_: RequestType):
    request_data = '{"inputs":[{"name":"INPUT0","data":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],' + \
                   '"datatype":"INT32","shape":[1,1,2,8]},' + \
                   '{"name":"INPUT1","data":[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160],' + \
                   '"datatype":"INT32","shape":[1,1,2,8]}]}'
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Inferenceservicename": "tensor",
        "Ce-Type": "io.seldon.serving.inference.request",
        "Ce-Requestid": "2z2",
    }
    expected_index = 'inference-log-seldon-unknown-namespace-tensor-unknown-endpoint'
    expected_elastic_docs = {
        "2z2": {
            'request': {
                'instance': [[[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]],
                'dataType': 'image',
                'payload': {"inputs": [
                    {"name": "INPUT0", "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                     "datatype": "INT32", "shape": [1, 1, 2, 8]},
                    {"name": "INPUT1", "data": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
                     "datatype": "INT32", "shape": [1, 1, 2, 8]}]},
            },
            "Ce-Inferenceservicename": "tensor",
            'ServingEngine': 'seldon',
            'RequestId': '2z2',
        },
    }
    return request_headers, request_data, expected_index, expected_elastic_docs


def seldon_income_classifier(request_type: RequestType):
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Inferenceservicename": "income",
        "Ce-Endpoint": "default",
        "Ce-Requestid": "2z4",
    }
    expected_index = "inference-log-seldon-unknown-namespace-income-default"
    expected_elastic_docs: Dict[str, Any] = {
        "2z4": {
            'ServingEngine': 'seldon',
            "Ce-Inferenceservicename": "income",
            "Ce-Endpoint": "default",
            "RequestId": "2z4",
        },
    }

    if request_type == RequestType.REQUEST:
        request_data = '{"data":{"names":["Age","Workclass","Education","Marital Status",' + \
                       '"Occupation","Relationship","Race","Sex","Capital Gain","Capital Loss",' + \
                       '"Hours per week","Country"],"ndarray":[[53,4,0,2,8,4,2,0,0,0,60,9]]}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.request"
        expected_elastic_docs["2z4"]["request"] = {
            'elements': {
                "Age": 53.0, "Workclass": 4.0, "Education": 0.0, "Marital Status": 2.0, "Occupation": 8.0,
                "Relationship": 4.0, "Race": 2.0, "Sex": 0.0, "Capital Gain": 0.0, "Capital Loss": 0.0,
                "Hours per week": 60.0, "Country": 9.0,
            },
            'instance': [53.0, 4.0, 0.0, 2.0, 8.0, 4.0, 2.0, 0.0, 0.0, 0.0, 60.0, 9.0],
            'names': ["Age", "Workclass", "Education", "Marital Status", "Occupation", "Relationship", "Race",
                      "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country"],
            'dataType': 'tabular',
            'payload': {"data": {
                "names": ["Age", "Workclass", "Education", "Marital Status", "Occupation", "Relationship", "Race",
                          "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country"],
                "ndarray": [[53, 4, 0, 2, 8, 4, 2, 0, 0, 0, 60, 9]]}}
        }
    elif request_type == RequestType.RESPONSE:
        request_data = '{"data":{"names":["t:0","t:1"],"ndarray":[[0.8538818809164035,0.14611811908359656]]},' + \
                       '"meta":{"requestPath":{"income-container":"seldonio/sklearnserver:1.7.0"}}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.response"
        expected_elastic_docs["2z4"]["response"] = {
            'elements': {"t:0": 0.8538818809164035, "t:1": 0.14611811908359656},
            'instance': [0.8538818809164035, 0.14611811908359656],
            'dataType': 'tabular',
            'meta': {'requestPath': {'income-container': 'seldonio/sklearnserver:1.7.0'}},
            'names': ['t:0', 't:1'],
            'payload': {"data": {"names": ["t:0", "t:1"], "ndarray": [[0.8538818809164035, 0.14611811908359656]]},
                        "meta": {"requestPath": {"income-container": "seldonio/sklearnserver:1.7.0"}}}
        }
    else:
        raise Exception("invalid request_type provided")

    return request_headers, request_data, expected_index, expected_elastic_docs


def mix_one_hot_categorical_float(request_type: RequestType):
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Inferenceservicename": "dummy",
        "Ce-Endpoint": "default",
        "Ce-Namespace": "seldon",
        "Ce-Requestid": "2z5",
    }
    expected_index = "inference-log-seldon-seldon-dummy-default"
    expected_elastic_docs: Dict[str, Any] = {
        "2z5": {
            'ServingEngine': 'seldon',
            "Ce-Inferenceservicename": "dummy",
            "Ce-Endpoint": "default",
            "RequestId": "2z5",
        },
    }

    if request_type == RequestType.REQUEST:
        # TODO: This request originally had the ndarray as [[0,1,0,2.54]].
        request_data = '{"data":{"names":["dummy_one_hot_1","dummy_one_hot_2","dummy_categorical","dummy_float"],' + \
                       '"ndarray":[[0.0,1.0,0.0,2.54]]}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.request"
        expected_elastic_docs["2z5"]["request"] = {
            'elements': {"dummy_one_hot_1": 0.0, "dummy_one_hot_2": 1.0, "dummy_categorical": 0.0, "dummy_float": 2.54},
            'instance': [0.0, 1.0, 0.0, 2.54],
            'dataType': 'tabular',
            'names': ["dummy_one_hot_1", "dummy_one_hot_2", "dummy_categorical", "dummy_float"],
            'payload': {"data": {"names": ["dummy_one_hot_1", "dummy_one_hot_2", "dummy_categorical", "dummy_float"],
                                 "ndarray": [[0.0, 1.0, 0.0, 2.54]]}}
        }
    elif request_type == RequestType.RESPONSE:
        request_data = '{"data":{"names":["dummy_proba_0","dummy_proba_1","dummy_float"],' + \
                       '"ndarray":[[0.8538818809164035,0.14611811908359656,3.65]]}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.response"
        expected_elastic_docs["2z5"]["response"] = {
            'elements': {"dummy_proba_0": 0.8538818809164035, "dummy_proba_1": 0.14611811908359656,
                         "dummy_float": 3.65},
            'instance': [0.8538818809164035, 0.14611811908359656, 3.65],
            'dataType': 'tabular',
            'names': ["dummy_proba_0", "dummy_proba_1", "dummy_float"],
            'payload': {"data": {"names": ["dummy_proba_0", "dummy_proba_1", "dummy_float"],
                                 "ndarray": [[0.8538818809164035, 0.14611811908359656, 3.65]]}}
        }
    else:
        raise Exception("invalid request_type provided")

    return request_headers, request_data, expected_index, expected_elastic_docs


def seldon_cifar10_image(request_type: RequestType):
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Namespace": "seldon",
        "Ce-Inferenceservicename": "cifar10",
        "Ce-Endpoint": "default",
        "Ce-Requestid": "2z6",
    }
    expected_index = "inference-log-seldon-seldon-cifar10-default"
    expected_elastic_docs: Dict[str, Any] = {
        "2z6": {
            'ServingEngine': 'seldon',
            "Ce-Namespace": "seldon",
            "Ce-Inferenceservicename": "cifar10",
            "Ce-Endpoint": "default",
            "RequestId": "2z6",
        },
    }

    if request_type == RequestType.REQUEST:
        with open("tests/cifardata.json", "rb") as f:
            raw_data = f.read()
        data = json.loads(raw_data)
        request_data = raw_data
        request_headers["Ce-Type"] = "io.seldon.serving.inference.request"
        expected_elastic_docs["2z6"]["request"] = {
            'elements': None,
            'instance': data["instances"][0],
            'dataType': 'image',
            'payload': data,
        }
    elif request_type == RequestType.RESPONSE:
        request_data = '{"predictions":[[1.26448515e-6,4.88145879e-9,1.51533219e-9,8.49055848e-9,' + \
                       '5.51306611e-10,1.16171928e-9,5.77288495e-10,2.88396933e-7,0.000614895718,0.999383569]]}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.response"
        expected_elastic_docs["2z6"]["response"] = {
            'elements': None,
            'instance': [1.26448515e-6, 4.88145879e-9, 1.51533219e-9, 8.49055848e-9, 5.51306611e-10, 1.16171928e-9,
                         5.77288495e-10, 2.88396933e-7, 0.000614895718, 0.999383569],
            'dataType': 'tabular',
            'payload': {"predictions": [
                [1.26448515e-6, 4.88145879e-9, 1.51533219e-9, 8.49055848e-9, 5.51306611e-10, 1.16171928e-9,
                 5.77288495e-10, 2.88396933e-7, 0.000614895718, 0.999383569]]},
        }

    elif request_type == RequestType.OUTLIER:
        request_data = '{"data": {"feature_score": null, "instance_score": null, "is_outlier": [1]}, ' + \
                       '"meta": {"name": "OutlierVAE", "detector_type": "offline", "data_type": "image"}}'
        request_headers["Ce-Type"] = "io.seldon.serving.inference.outlier"
        expected_elastic_docs["2z6"]["outlier"] = {
            'data': {'feature_score': None, 'instance_score': None, 'is_outlier': 1},
            'meta': {'name': 'OutlierVAE', 'detector_type': 'offline', 'data_type': 'image'},
        }

    else:
        raise Exception("invalid request_type provided")

    return request_headers, request_data, expected_index, expected_elastic_docs


def seldon_iris_batch(_: RequestType):
    request_data = '{"data":{"names":["Sepal length","Sepal width","Petal length","Petal Width"],' + \
                   '"ndarray":[[6.8,2.8,4.8,1.4],[6.1,3.4,4.5,1.6]]}}'
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Type": "io.seldon.serving.inference.request",
        "Ce-Namespace": "seldon",
        "Ce-Inferenceservicename": "iris",
        "Ce-Endpoint": "default",
        "Ce-Requestid": "2z7",
    }
    expected_index = "inference-log-seldon-seldon-iris-default"
    expected_elastic_docs = {
        "2z7-item-0": {
            'request': {
                'elements': {"Sepal length": 6.8, "Sepal width": 2.8, "Petal length": 4.8, "Petal Width": 1.4},
                'instance': [6.8, 2.8, 4.8, 1.4],
                'dataType': 'tabular',
                'names': ["Sepal length", "Sepal width", "Petal length", "Petal Width"],
                'payload': {"data": {"names": ["Sepal length", "Sepal width", "Petal length", "Petal Width"],
                                     "ndarray": [[6.8, 2.8, 4.8, 1.4], [6.1, 3.4, 4.5, 1.6]]}},
            },
            'ServingEngine': 'seldon',
            "Ce-Namespace": "seldon",
            "Ce-Inferenceservicename": "iris",
            "Ce-Endpoint": "default",
            'RequestId': '2z7'
        },
        "2z7-item-1": {
            'request': {
                'elements': {"Sepal length": 6.1, "Sepal width": 3.4, "Petal length": 4.5, "Petal Width": 1.6},
                'instance': [6.1, 3.4, 4.5, 1.6],
                'dataType': 'tabular',
                'names': ["Sepal length", "Sepal width", "Petal length", "Petal Width"],
                'payload': {"data": {"names": ["Sepal length", "Sepal width", "Petal length", "Petal Width"],
                                     "ndarray": [[6.8, 2.8, 4.8, 1.4], [6.1, 3.4, 4.5, 1.6]]}},
            },
            'ServingEngine': 'seldon',
            "Ce-Namespace": "seldon",
            "Ce-Inferenceservicename": "iris",
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


def kfserving_income(_: RequestType):
    request_data = '{"instances":[[39, 7, 1, 1, 1, 1, 4, 1, 2174, 0, 40, 9]]}'
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Type": "org.kubeflow.serving.inference.request",
        "Ce-Namespace": "seldon",
        "Ce-Inferenceservicename": "income-kf",
        "Ce-Endpoint": "default",
        "Ce-Requestid": "2z9",
    }
    expected_index = "inference-log-inferenceservice-seldon-income-kf-default"
    expected_elastic_docs = {
        "2z9": {
            'request': {
                'elements': None,
                'instance': [39, 7, 1, 1, 1, 1, 4, 1, 2174, 0, 40, 9],
                'dataType': 'tabular',
                'payload': {"instances": [[39, 7, 1, 1, 1, 1, 4, 1, 2174, 0, 40, 9]]},
            },
            'ServingEngine': 'inferenceservice',
            "Ce-Namespace": "seldon",
            "Ce-Inferenceservicename": "income-kf",
            "Ce-Endpoint": "default",
            'RequestId': '2z9'
        }
    }
    return request_headers, request_data, expected_index, expected_elastic_docs


def seldon_drift(_: RequestType):
    data = {"drift": {"data": {"is_drift": True, "distance": {"Input Image": 0.533299999923706055},
                    "p_val": {"Input Image": 0.4361779987812042},"threshold": 0.0015625, "drift_type": "feature"},
            "meta": {"name": "KSDrift","detector_type": "offline","data_type": None}}}
    request_data = '{"data": {"feature_score": null, "instance_score": null, "is_outlier": [1]}, ' + \
                   '"meta": {"name": "KSDrift", "detector_type": "offline", "data_type": null}}'
    request_headers = {
        "Content-Type": "application/json",
        "Ce-Type": "io.seldon.serving.inference.drift",
        "Ce-Namespace": "development",
        "Ce-Inferenceservicename": "cifar10",
        "Ce-Endpoint": "default",
        "Ce-Requestid": "3z1",
    }
    expected_index = "drift-log-seldon-development-cifar10"
    expected_elastic_docs = {
        "3z1": {
            'drift': {
                'data': {
                    "is_drift": True,
                    "distance": {
                        "Input Image": 0.533299999923706055,
                    },
                    "p_val": {
                        "Input Image": 0.4361779987812042
                    },
                    "threshold": 0.0015625,
                    "drift_type": "feature"
                },
                "meta": {
                    "name": "KSDrift",
                    "detector_type": "offline",
                    "data_type": None,
                },
            },
            'ServingEngine': 'seldon',
            "Ce-Namespace": "development",
            "Ce-Inferenceservicename": "cifar10",
            "Ce-Endpoint": "default",
            'RequestId': '3z1',
        }
    }
    return request_headers, request_data, expected_index, expected_elastic_docs


