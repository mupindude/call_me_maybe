
from src.models import FunctionCallOutput
from src.load_data import load_function_definitions
from src.validate_output import validate_function_call
from src.validate_output import build_function_call_output
from src.validate_output import process_generated_output


def main() -> None:
    """Test function-call validation."""
    definitions = load_function_definitions(
        "data/input/functions_definition.json"
    )

    output = FunctionCallOutput(
        prompt="What is the sum of 2 and 3?",
        name="fn_add_numbers",
        parameters={"a": 2.0, "b": 3.0},
    )

    print(
        "Valid function call:",
        validate_function_call(output, definitions),
    )

    # Unknown function
    unknown_function = FunctionCallOutput(
        prompt="Test",
        name="fn_unknown",
        parameters={"a": 2, "b": 3},
    )

    print(
        "Unknown function:",
        validate_function_call(unknown_function, definitions),
    )

    # Incorrect parameter names
    wrong_names = FunctionCallOutput(
        prompt="Test",
        name="fn_add_numbers",
        parameters={"a": 2, "c": 3},
    )

    print(
        "Wrong parameter names:",
        validate_function_call(wrong_names, definitions),
    )

    # Incorrect parameter types
    wrong_types = FunctionCallOutput(
        prompt="Test",
        name="fn_add_numbers",
        parameters={"a": "hello", "b": 3},
    )

    print(
        "Wrong parameter types:",
        validate_function_call(wrong_types, definitions),
    )

    integer_types = FunctionCallOutput(
        prompt="Test",
        name="fn_add_numbers",
        parameters={"a": 2, "b": 3},
        )

    print(
        "Integer parameters rejected:",
        not validate_function_call(integer_types, definitions),
    )

    parsed_data = {
        "name": "fn_add_numbers",
        "arguments": {
            "a": 2.0,
            "b": 3.0,
        },
    }

    result = build_function_call_output(
        parsed_data,
        "What is the sum of 2 and 3?",
        definitions,
    )

    print("Converted output:", result.model_dump())

    print(
        "Final validation:",
        validate_function_call(result, definitions),
    )

    generated_text = (
        '<tool_call>\n'
        '{"name": "fn_add_numbers", '
        '"arguments": {"a": 2.0, "b": 3.0}}\n'
        '</tool_call>'
    )

    result = process_generated_output(
        generated_text,
        "What is the sum of 2 and 3?",
        definitions,
    )

    print("Pipeline result:", result.model_dump())

    integer_data = {
        "name": "fn_add_numbers",
        "arguments": {
            "a": 265,
            "b": 345,
        },
    }

    integer_result = build_function_call_output(
        integer_data,
        "What is the sum of 265 and 345?",
        definitions,
    )

    print(
        "Integer normalization:",
        integer_result.parameters,
    )

    try:
        bad_text = (
            '<tool_call>\n'
            '{"name": "fn_unknown", '
            '"arguments": {"a": 2, "b": 3}}\n'
            '</tool_call>'
        )

        res = process_generated_output(
            bad_text,
            "What is the sum of 2 and 3?",
            definitions,
        )

        print("Pipeline result:", res.model_dump())

    except ValueError as error:
        print("Invalid function correctly rejected", error)


if __name__ == "__main__":
    main()
