from typing import Any, Dict

import pytest
from seldon_deploy_sdk.models.v1_model import V1Model
from seldon_deploy_sdk.models.v1_runtime_metadata import V1RuntimeMetadata
from pydantic import BaseModel
from tests.fixtures import iris


class Scenario(BaseModel):
    model: V1Model
    runtime_metadata: V1RuntimeMetadata
    model_input: Any
    expected: dict

    class Config:
        arbitrary_types_allowed = True


@pytest.fixture
def iris_scenario():
    return Scenario(
        model=iris.model(),
        runtime_metadata=iris.runtime_metadata(),
        model_input=None,
        expected={},
    )