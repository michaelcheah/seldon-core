from typing import Dict
from seldon_deploy_sdk.models.v1_model import V1Model
from seldon_deploy_sdk.models.v1_runtime_metadata import V1RuntimeMetadata
from seldon_deploy_sdk.models.v1_runtime_metadata_list_response import V1RuntimeMetadataListResponse
from seldon_deploy_sdk.models.v1_model_metadata_list_response import V1ModelMetadataListResponse


def init_api_patch():
    global metadata_api

    class PatchAPI:
        runtime_models: Dict[str, V1RuntimeMetadata] = None
        model_metadata: Dict[str, V1Model] = None

        def __init__(self):
            self.runtime_models = {}
            self.model_metadata = {}

        def get_runtime_metadata_name(
                self,
                deployment_name: str = "seldon",
                deployment_namespace: str = "seldon",
                predictor_name: str = "income",
                deployment_type: str = "SeldonDeployment") -> str:
            return f"{deployment_name}-{deployment_namespace}-{predictor_name}-{deployment_type}"

        def create_model(self, model_metadata: V1Model):
            self.model_metadata[model_metadata.uri] = model_metadata

        def create_deployment(self, runtime_metadata: V1RuntimeMetadata, model_uri: str):
            key = self.get_runtime_metadata_name(
                deployment_name=runtime_metadata.deployment_name,
                deployment_namespace=runtime_metadata.deployment_namespace,
                predictor_name=runtime_metadata.predictor_name,
                deployment_type=runtime_metadata.deployment_type,
            )
            runtime_metadata.model_uri = model_uri
            self.runtime_models[key] = runtime_metadata

        def model_metadata_service_list_runtime_metadata_for_model(
                self, deployment_name: str = "seldon",
                deployment_namespace: str = "seldon",
                predictor_name: str = "income",
                deployment_type: str = "SeldonDeployment",
                deployment_status: str = "Running") -> V1RuntimeMetadataListResponse:
            runtime_metadata = self.runtime_models[self.get_runtime_metadata_name(
                deployment_name=deployment_name,
                deployment_namespace=deployment_namespace,
                predictor_name=predictor_name,
                deployment_type=deployment_type,
            )]
            return V1RuntimeMetadataListResponse(
                runtime_metadata=[runtime_metadata]
            )

        def model_metadata_service_list_model_metadata(self, uri: str) -> V1ModelMetadataListResponse:
            model_metadata: V1Model = self.model_metadata[uri]
            return V1ModelMetadataListResponse(
                models=[model_metadata]
            )

    metadata_api = PatchAPI()

# if __name__ == "__main__":
#     init_api_patch()
#     global metadata_api
#
#     iris_model = V1Model(
#             uri="gs://test-model-beta-v2.0.0",
#             prediction_schema=V1PredictionSchema(
#                 requests=[
#                     V1FeatureSchema(
#                         name="Sepal Length",
#                         type=V1FeatureType.REAL,
#                         data_type=V1DataType.FLOAT,
#                     ),
#                     V1FeatureSchema(
#                         name="Sepal Width",
#                         type=V1FeatureType.REAL,
#                         data_type=V1DataType.FLOAT,
#                     ),
#                     V1FeatureSchema(
#                         name="Petal Length",
#                         type=V1FeatureType.REAL,
#                         data_type=V1DataType.FLOAT,
#                     ),
#                     V1FeatureSchema(
#                         name="Petal Width",
#                         type=V1FeatureType.REAL,
#                         data_type=V1DataType.FLOAT,
#                     ),
#                 ],
#                 responses=[
#                     V1FeatureSchema(
#                         name="Iris Species",
#                         type=V1FeatureType.PROBA,
#                         data_type=V1DataType.FLOAT,
#                         schema=[
#                             V1FeatureCategorySchema(
#                                 name="Setosa",
#                                 data_type=V1DataType.FLOAT,
#                             ),
#                             V1FeatureCategorySchema(
#                                 name="Versicolor",
#                                 data_type=V1DataType.FLOAT,
#                             ),
#                             V1FeatureCategorySchema(
#                                 name="Virginica",
#                                 data_type=V1DataType.FLOAT,
#                             ),
#                         ]
#                     ),
#                 ]
#             )
#         )
#
#     metadata_api.create_model(
#         iris_model
#     )
#
#     metadata_api.create_deployment(
#         V1RuntimeMetadata(
#             deployment_name="iris-classifier",
#             deployment_type="SeldonDeployment",
#             predictor_name="default",
#             deployment_namespace="seldon",
#         ),
#         iris_model.uri
#     )
#
#     runtime_metadata = metadata_api.model_metadata_service_list_runtime_metadata_for_model(
#         deployment_name="iris-classifier",
#         deployment_type="SeldonDeployment",
#         predictor_name="default",
#         deployment_namespace="seldon",
#         deployment_status="Running",
#     )
#
#     print(runtime_metadata)
#
#     print(runtime_metadata.runtime_metadata[0].model_uri)
#
#     model_uri = runtime_metadata.runtime_metadata[0].model_uri
#
#     metadata = metadata_api.model_metadata_service_list_model_metadata(model_uri)
#
#     print(metadata.models[0].prediction_schema.to_dict())