from src.models import FunctionCallOutput, FunctionDefinition


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
