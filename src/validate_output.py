from src.models import FunctionCallOutput, FunctionDefinition
from src.parse_output import parse_function_call


def find_function_definition(
    output: FunctionCallOutput,
    definitions: list[FunctionDefinition]
) -> FunctionDefinition | None:
    """Find the definition matching the generated function name."""
    for definition in definitions:
        if definition.name == output.name:
            return definition
    return None


def has_valid_parameter_names(
    output: FunctionCallOutput,
    definition: FunctionDefinition
) -> bool:
    """Check that generated parameter names match the definition."""
    expected = set(definition.parameters.keys())
    received = set(output.parameters.keys())

    return expected == received


def has_valid_parameter_types(
    output: FunctionCallOutput,
    definition: FunctionDefinition
) -> bool:
    """Check that generated parameter values match expected types."""
    for name, parameter in definition.parameters.items():
        value = output.parameters[name]

        if parameter.type == "string":
            if not isinstance(value, str):
                return False
        elif parameter.type == "number":
            if not isinstance(value, (int, float)):
                return False
            if isinstance(value, bool):
                return False

    return True


def validate_function_call(
    output: FunctionCallOutput,
    definitions: list[FunctionDefinition]
) -> bool:
    """Validate a generated function call against its definition."""
    definition = find_function_definition(output, definitions)

    if definition is None:
        return False

    if not has_valid_parameter_names(output, definition):
        return False

    if not has_valid_parameter_types(output, definition):
        return False

    return True


def build_function_call_output(
    parsed_data: dict[str, object],
    prompt: str,
) -> FunctionCallOutput:
    """Convert parsed model output into the required output model."""
    if set(parsed_data.keys()) != {"name", "arguments"}:
        raise ValueError("Invalid function-call structure")

    name = parsed_data["name"]
    arguments = parsed_data["arguments"]

    if not isinstance(name, str):
        raise ValueError("Function name must be a string")

    if not isinstance(arguments, dict):
        raise ValueError("Function arguments must be an object")

    return FunctionCallOutput.model_validate({
        "prompt": prompt,
        "name": name,
        "parameters": arguments,
    })


def process_generated_output(
    generated_text: str,
    prompt: str,
    definitions: list[FunctionDefinition],
) -> FunctionCallOutput:
    """Parse, convert and validate generated function-call output."""
    parsed_data = parse_function_call(generated_text)

    output = build_function_call_output(
        parsed_data,
        prompt,
    )

    if not validate_function_call(output, definitions):
        raise ValueError("Invalid generated function call")

    return output
