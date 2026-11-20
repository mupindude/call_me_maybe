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


def check_tool_call_key_prefix(text: str) -> PrefixStatus:
    """Check whether text can become a valid tool-call key."""
    allowed_keys = ("name", "arguments")

    for key in allowed_keys:
        if key.startswith(text):
            if text == key:
                return PrefixStatus.VALID
            return PrefixStatus.INCOMPLETE

    return PrefixStatus.INVALID


def check_tool_call_json_key(text: str) -> PrefixStatus:
    """Validate a complete JSON key against the allowed tool-call keys."""
    try:
        key = decode_json_string(text)
    except (ValueError, TypeError):
        return PrefixStatus.INVALID

    return check_tool_call_key_prefix(key)


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


def check_tool_call_key_progress(text: str) -> PrefixStatus:
    """Check whether a JSON key prefix can become an allowed tool-call key."""
    string_status = check_json_string(text)

    if string_status == PrefixStatus.INVALID:
        return PrefixStatus.INVALID

    if string_status == PrefixStatus.INCOMPLETE:
        try:
            decoded_prefix = decode_complete_key_prefix(text)
        except (ValueError, TypeError):
            return PrefixStatus.INVALID

        return check_tool_call_key_prefix(decoded_prefix)

    try:
        key = decode_json_string(text)
    except (ValueError, TypeError):
        return PrefixStatus.INVALID

    return check_tool_call_key_prefix(key)


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


def check_function_name_value(
    text: str,
    definitions: list[FunctionDefinition],
) -> PrefixStatus:
    """Check whether a JSON string can become a defined function name."""
    status = check_json_string(text)

    if status == PrefixStatus.INVALID:
        return PrefixStatus.INVALID

    if status == PrefixStatus.INCOMPLETE:
        try:
            value = decode_complete_key_prefix(text)
        except (ValueError, TypeError):
            return PrefixStatus.INVALID

        candidates = [definition.name for definition in definitions]

        if any(name.startswith(value) for name in candidates):
            return PrefixStatus.INCOMPLETE

        return PrefixStatus.INVALID

    try:
        value = decode_json_string(text)
    except (ValueError, TypeError):
        return PrefixStatus.INVALID

    if any(value == definition.name for definition in definitions):
        return PrefixStatus.VALID

    return PrefixStatus.INVALID


def check_function_name_colon(text: str) -> PrefixStatus:
    """Check whether a completed function name is followed by a colon."""
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


def check_function_name_comma(text: str) -> PrefixStatus:
    """Check whether a completed function name value is followed by a comma."""
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
    status = check_json_string(text)

    if status == PrefixStatus.INVALID:
        return PrefixStatus.INVALID

    if status == PrefixStatus.INCOMPLETE:
        try:
            value = decode_complete_key_prefix(text)
        except (ValueError, TypeError):
            return PrefixStatus.INVALID

        if "arguments".startswith(value):
            return PrefixStatus.INCOMPLETE

        return PrefixStatus.INVALID

    try:
        value = decode_json_string(text)
    except (ValueError, TypeError):
        return PrefixStatus.INVALID

    if value == "arguments":
        return PrefixStatus.VALID

    return PrefixStatus.INVALID


def check_arguments_key_colon(text: str) -> PrefixStatus:
    """Check whether the arguments key is followed by a colon."""
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


def check_arguments_object_start(text: str) -> PrefixStatus:
    """Check whether the arguments value starts with a JSON object."""
    parser = JSONPrefixParser(text)
    parser.skip_whitespace()

    if parser.consume("{"):
        return PrefixStatus.INCOMPLETE

    if parser.current_char() is None:
        return PrefixStatus.INCOMPLETE

    return PrefixStatus.INVALID
