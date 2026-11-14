from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictFloat,
    StrictInt,
    StrictStr,
)


class ParameterDefinition(BaseModel):
    """Describe the expected type of a function parameter."""

    type: Literal["number", "string"]


class ReturnDefinition(BaseModel):
    """Describe the expected return type of a function."""

    type: Literal["number", "string"]


class FunctionDefinition(BaseModel):
    """Describe a function available for LLM function calling."""

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class TestCase(BaseModel):
    """Represent a single function-calling test prompt."""

    prompt: str


class FunctionCallOutput(BaseModel):
    """Represent one generated function call."""

    model_config = ConfigDict(extra="forbid")

    prompt: StrictStr
    name: StrictStr
    parameters: dict[
        StrictStr,
        StrictInt | StrictFloat | StrictStr
    ]
