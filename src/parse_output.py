
import json


def parse_function_call(generated_text: str) -> dict[str, object]:
    """Extract and parse the JSON function call from generated text."""
    opening_tag = "<tool_call>"
    closing_tag = "</tool_call>"

    start = generated_text.find(opening_tag)
    end = generated_text.find(closing_tag)

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Invalid function-call tags")

    json_text = generated_text[
        start + len(opening_tag):end
    ].strip()

    try:
        parsed_data = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise ValueError("Invalid function-call JSON") from error

    if not isinstance(parsed_data, dict):
        raise ValueError("Function call must be a JSON object")

    return parsed_data
