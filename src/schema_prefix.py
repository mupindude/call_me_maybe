from src.json_prefix import (
    JSONPrefixParser,
    PrefixStatus,
    check_json_string,
    decode_json_string,
)
from src.models import FunctionDefinition


def check_tool_call_structure(
    parsed_data: object,
    definitions: list[FunctionDefinition],
) -> bool:
    """Check the outer structure of a parsed tool call."""
    if not isinstance(parsed_data, dict):
        return False

    if set(parsed_data.keys()) != {"name", "arguments"}:
        return False

    name = parsed_data["name"]
    arguments = parsed_data["arguments"]

    if not isinstance(name, str):
        return False

    if not isinstance(arguments, dict):
        return False

    for definition in definitions:
        if definition.name == name:
            return True

    return False


def check_parameter_names(
    arguments: dict[str, object],
    definition: FunctionDefinition,
) -> bool:
    """Check whether all argument names match the function definition."""
    expected = set(definition.parameters.keys())
    received = set(arguments.keys())
    return expected == received


def check_parameter_types(
    arguments: dict[str, object],
    definition: FunctionDefinition,
) -> bool:
    """Check whether argument values match their expected types."""
    for name, parameter in definition.parameters.items():
        value = arguments[name]

        if parameter.type == "string":
            if not isinstance(value, str):
                return False

        elif parameter.type == "number":
            if isinstance(value, bool):
                return False
            if not isinstance(value, (int, float)):
                return False

    return True


def validate_tool_call(
    parsed_data: object,
    definitions: list[FunctionDefinition],
) -> bool:
    """Validate a completed tool call against function definitions."""
    if not check_tool_call_structure(parsed_data, definitions):
        return False

    if not isinstance(parsed_data, dict):
        return False

    name = parsed_data["name"]
    arguments = parsed_data["arguments"]

    for definition in definitions:
        if definition.name == name:
            if not check_parameter_names(arguments, definition):
                return False

            if not check_parameter_types(arguments, definition):
                return False

            return True

    return False


def decode_complete_key_prefix(text: str) -> str:
    """Decode all complete characters in an unfinished JSON string."""
    raw_key = text[1:]
    index = 0
    safe_end = len(raw_key)

    while index < len(raw_key):
        if raw_key[index] != "\\":
            index += 1
            continue

        if index + 1 >= len(raw_key):
            safe_end = index
            break

        if raw_key[index + 1] == "u":
            if index + 6 > len(raw_key):
                safe_end = index
                break
            index += 6
        else:
            index += 2

    complete_prefix = raw_key[:safe_end]
    return decode_json_string('"' + complete_prefix + '"')


def check_string_prefix(
    text: str,
    allowed_values: list[str],
) -> PrefixStatus:
    """Check whether a JSON string can become an allowed value."""
    status = check_json_string(text)

    if status == PrefixStatus.INVALID:
        return PrefixStatus.INVALID

    if status == PrefixStatus.INCOMPLETE:
        try:
            value = decode_complete_key_prefix(text)
        except (ValueError, TypeError):
            return PrefixStatus.INVALID

        if any(item.startswith(value) for item in allowed_values):
            return PrefixStatus.INCOMPLETE

        return PrefixStatus.INVALID

    try:
        value = decode_json_string(text)
    except (ValueError, TypeError):
        return PrefixStatus.INVALID

    if value in allowed_values:
        return PrefixStatus.VALID

    return PrefixStatus.INVALID


def check_tool_call_key_progress(text: str) -> PrefixStatus:
    """Check whether a JSON key prefix can become an allowed tool-call key."""
    return check_string_prefix(text, ["name", "arguments"])


def check_first_root_key_prefix(text: str) -> PrefixStatus:
    """Validate the first key in a tool-call JSON object."""
    parser = JSONPrefixParser(text)

    parser.skip_whitespace()

    if not parser.consume("{"):
        if parser.current_char() is None:
            return PrefixStatus.INCOMPLETE
        return PrefixStatus.INVALID

    parser.skip_whitespace()

    if parser.current_char() is None:
        return PrefixStatus.INCOMPLETE

    if parser.current_char() != '"':
        return PrefixStatus.INVALID

    start = parser.index
    status = parser.parse_string()

    if status == PrefixStatus.INVALID:
        return PrefixStatus.INVALID

    key_text = text[start:parser.index] if (
        status == PrefixStatus.VALID
    ) else text[start:]

    return check_tool_call_key_progress(key_text)


def check_root_key_colon(text: str) -> PrefixStatus:
    """Check whether a completed root key is followed by a colon."""
    parser = JSONPrefixParser(text)
    parser.skip_whitespace()

    if not parser.consume("{"):
        return PrefixStatus.INVALID

    remaining = text[parser.index:]

    if not remaining.strip():
        return PrefixStatus.INCOMPLETE

    return check_string_colon(remaining)


def check_function_name_value(
    text: str,
    definitions: list[FunctionDefinition],
) -> PrefixStatus:
    """Check whether a JSON string can become a defined function name."""
    names = [definition.name for definition in definitions]
    return check_string_prefix(text, names)


def check_string_colon(text: str) -> PrefixStatus:
    """Check whether a JSON string is followed by a colon."""
    parser = JSONPrefixParser(text)
    parser.skip_whitespace()

    status = parser.parse_string()
    if status != PrefixStatus.VALID:
        return status

    parser.skip_whitespace()

    if parser.consume(":"):
        return PrefixStatus.INCOMPLETE

    if parser.current_char() is None:
        return PrefixStatus.INCOMPLETE

    return PrefixStatus.INVALID


def check_string_comma(text: str) -> PrefixStatus:
    """Check whether a JSON string is followed by a comma."""
    parser = JSONPrefixParser(text)
    parser.skip_whitespace()

    status = parser.parse_string()
    if status != PrefixStatus.VALID:
        return status

    parser.skip_whitespace()

    if parser.consume(","):
        return PrefixStatus.INCOMPLETE

    if parser.current_char() is None:
        return PrefixStatus.INCOMPLETE

    return PrefixStatus.INVALID


def check_arguments_key_value(text: str) -> PrefixStatus:
    """Check whether a JSON string can become the arguments key."""
    return check_string_prefix(text, ["arguments"])


def check_argument_key_value(
    text: str,
    definition: FunctionDefinition,
) -> PrefixStatus:
    """Check whether a JSON string can become a defined argument name."""
    parameter_names = list(definition.parameters.keys())
    return check_string_prefix(text, parameter_names)


def get_unused_parameter_names(
    definition: FunctionDefinition,
    used_names: set[str],
) -> list[str]:
    """Return parameter names that have not been used yet."""
    return [
        name
        for name in definition.parameters
        if name not in used_names
    ]


def extract_used_argument_names(text: str) -> set[str]:
    """Extract completed keys from a JSON object prefix."""
    parser = JSONPrefixParser(text)
    used_names: set[str] = set()

    parser.skip_whitespace()

    if not parser.consume("{"):
        return used_names

    parser.skip_whitespace()

    while parser.current_char() == '"':
        start = parser.index

        status = parser.parse_string()
        if status != PrefixStatus.VALID:
            break

        key_text = text[start:parser.index]

        parser.skip_whitespace()

        if not parser.consume(":"):
            break

        status = parser.parse_value()
        if status != PrefixStatus.VALID:
            break

        key = decode_json_string(key_text)
        used_names.add(key)

        parser.skip_whitespace()

        if not parser.consume(","):
            break

        parser.skip_whitespace()

    return used_names


def check_unused_argument_key_value(
    text: str,
    definition: FunctionDefinition,
    used_names: set[str],
) -> PrefixStatus:
    """Check whether a JSON string can become an unused argument name."""
    allowed_names = get_unused_parameter_names(definition, used_names)
    return check_string_prefix(text, allowed_names)


def check_arguments_object_start(text: str) -> PrefixStatus:
    """Check whether the arguments value starts with a JSON object."""
    parser = JSONPrefixParser(text)
    parser.skip_whitespace()

    if parser.consume("{"):
        return PrefixStatus.INCOMPLETE

    if parser.current_char() is None:
        return PrefixStatus.INCOMPLETE

    return PrefixStatus.INVALID
