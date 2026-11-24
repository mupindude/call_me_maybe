
from src.json_prefix import PrefixStatus
from src.load_data import load_function_definitions
from src.schema_prefix import check_tool_call_prefix


def main() -> None:
    """Test schema-aware function-call prefix validation."""
    definitions = load_function_definitions(
        "data/input/functions_definition.json"
    )

    valid_cases = [
        '{"name": "fn_add_numbers", "arguments": {"a": 265, "b": 345}}',
        '{"name": "fn_add_numbers", "arguments": {"a": 265.0, "b": 345.0}}',
        '{"name": "fn_add_numbers", "arguments": {"a": 2.65e2, "b": 3.45e2}}',
    ]

    for text in valid_cases:
        status = check_tool_call_prefix(text, definitions)
        print(status == PrefixStatus.VALID, repr(text), status.name)

    invalid_cases = [
        '{"name": "fn_add_numbers", "arguments": {"a": true, "b": 3}}',
        '{"name": "fn_add_numbers", "arguments": {"a": 265., "b": 345}}',
        '{"name": "fn_add_numbers", "arguments": {"a": 265, "b": "345"}}',
    ]

    for text in invalid_cases:
        status = check_tool_call_prefix(text, definitions)
        print(status == PrefixStatus.INVALID, repr(text), status.name)

    incomplete_cases = [
        '{"name": "fn_add_numbers", "arguments": {"a": 265, "b":',
        '{"name": "fn_add_numbers", "arguments": {"a": 2.',
    ]

    for text in incomplete_cases:
        status = check_tool_call_prefix(text, definitions)
        print(status == PrefixStatus.INCOMPLETE, repr(text), status.name)


if __name__ == "__main__":
    main()
