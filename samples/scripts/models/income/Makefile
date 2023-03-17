.PHONY: start_classifier
start_predictors:
	cd models/predictors && MLSERVER_HTTP_PORT=8080 mlserver start .

.PHONY: start_kernel_shap_explainer
start_explainers_v1:
	cd models/explainers/alibi-0.9.0 && MLSERVER_HTTP_PORT=8083 MLSERVER_GRPC_PORT=8084 MLSERVER_METRICS_PORT=8085 mlserver start .

.PHONY: start_kernel_shap_explainer
start_explainers_v2:
	cd models/explainers/alibi-0.9.1 && MLSERVER_HTTP_PORT=8083 MLSERVER_GRPC_PORT=8084 MLSERVER_METRICS_PORT=8085 mlserver start .

.PHONY: list_models
list_models:
	curl -k -X POST http://localhost:8080/v2/repository/index -d '{}' -H 'Content-Type: application/json'
	curl -k -X POST http://localhost:8083/v2/repository/index -d '{}' -H 'Content-Type: application/json'


.PHONY: send_classifier_request
send_classifier_request:
	curl -k -X POST http://localhost:8080/v2/models/classifier/infer \
        -H 'Content-Type: application/json' \
        -d '{"inputs": [{"data": [52, 4, 0, 2, 8, 4, 2, 0, 0, 0, 60, 9], "datatype": "INT64", "name": "income", "shape": [1, 12]}]}'

.PHONY: send_kernel_shap_request
send_kernel_shap_request:
	curl -k -X POST http://localhost:8083/v2/models/kernel_shap_explainer/infer \
	    -H 'Host: seldon.inference.seldon' \
        -H 'Content-Type: application/json' \
        -d '{"inputs": [{"data": [52, 4, 0, 2, 8, 4, 2, 0, 0, 0, 60, 9], "datatype": "INT64", "name": "income", "shape": [1, 12]}]}'

clean:
	find ./models -name "*.db" -type f -delete

VERSION=0.3
upload-all: clean
	gsutil cp -r models/* gs://dev-michael-seldon-models/demos/${VERSION}


V1_ENV==.v1_env
v1-env:
	python3 -m venv .v1_env
	.v1_env/bin/pip install -e /home/michael/repos/MLServer/
	.v1_env/bin/pip install alibi==0.7.0 alibi[shap]==0.7.0 shap numpy==1.23.5


v1-generate:
	.v1_env/bin/python train_v1.py

v1-upload-all: clean
	gsutil cp -r v1_models/* gs://dev-michael-seldon-models/v1-demos/${VERSION}


