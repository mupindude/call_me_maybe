
from src.parse_output import parse_function_call


def main() -> None:
    """Test parsing generated function calls."""
    generated_text = (
        '<tool_call>\n'
        '{"name": "fn_add_numbers", '
        '"arguments": {"a": 2, "b": 3}}\n'
        '</tool_call>'
    )

    result = parse_function_call(generated_text)

    print(result)
    print(type(result))

    try:
        parse_function_call("This is not a function call")
    except ValueError as error:
        print("Invalid tags handled:", error)

    try:
        parse_function_call(
            '<tool_call>{"name": }</tool_call>'
        )
    except ValueError as error:
        print("Invalid JSON handled:", error)


if __name__ == "__main__":
    main()
