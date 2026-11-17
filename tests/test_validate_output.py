
from src.models import FunctionCallOutput
from src.load_data import load_function_definitions
from src.validate_output import validate_function_call


def main() -> None:
    """Test function-call validation."""
    definitions = load_function_definitions(
        "data/input/functions_definition.json"
    )

    output = FunctionCallOutput(
        prompt="What is the sum of 2 and 3?",
        name="fn_add_numbers",
        parameters={"a": 2, "b": 3},
    )

    print(
        "Valid function call:",
        validate_function_call(output, definitions),
    )


if __name__ == "__main__":
    main()
