import pytest
from seldon_deploy_sdk.models.v1_model import V1Model
from seldon_deploy_sdk.models.v1_runtime_metadata import V1RuntimeMetadata

from seldon_deploy_sdk.models.v1_prediction_schema import V1PredictionSchema
from seldon_deploy_sdk.models.v1_feature_schema import V1FeatureSchema
from seldon_deploy_sdk.models.v1_data_type import V1DataType
from seldon_deploy_sdk.models.v1_feature_type import V1FeatureType
from seldon_deploy_sdk.models.v1_feature_category_schema import V1FeatureCategorySchema


def model():
    return V1Model(
        uri="gs://test-model-beta-v2.0.0",
        prediction_schema=V1PredictionSchema(
            requests=[
                V1FeatureSchema(
                    name="Sepal Length",
                    type=V1FeatureType.REAL,
                    data_type=V1DataType.FLOAT,
                ),
                V1FeatureSchema(
                    name="Sepal Width",
                    type=V1FeatureType.REAL,
                    data_type=V1DataType.FLOAT,
                ),
                V1FeatureSchema(
                    name="Petal Length",
                    type=V1FeatureType.REAL,
                    data_type=V1DataType.FLOAT,
                ),
                V1FeatureSchema(
                    name="Petal Width",
                    type=V1FeatureType.REAL,
                    data_type=V1DataType.FLOAT,
                ),
            ],
            responses=[
                V1FeatureSchema(
                    name="Iris Species",
                    type=V1FeatureType.PROBA,
                    data_type=V1DataType.FLOAT,
                    schema=[
                        V1FeatureCategorySchema(
                            name="Setosa",
                            data_type=V1DataType.FLOAT,
                        ),
                        V1FeatureCategorySchema(
                            name="Versicolor",
                            data_type=V1DataType.FLOAT,
                        ),
                        V1FeatureCategorySchema(
                            name="Virginica",
                            data_type=V1DataType.FLOAT,
                        ),
                    ]
                ),
            ]
        )
    )


def runtime_metadata():
    return V1RuntimeMetadata(
        deployment_name="iris-classifier",
        deployment_type="SeldonDeployment",
        predictor_name="default",
        deployment_namespace="seldon",
    )
