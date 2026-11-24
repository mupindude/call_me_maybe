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


def find_function_definition(
    name: str,
    definitions: list[FunctionDefinition],
) -> FunctionDefinition | None:
    """Find a function definition by its name."""
    for definition in definitions:
        if definition.name == name:
            return definition

    return None


def extract_json_string(
    text: str,
    start: int,
) -> tuple[PrefixStatus, str | None, int]:
    """Extract a complete JSON string starting at a given index."""
    parser = JSONPrefixParser(text)
    parser.index = start

    status = parser.parse_string()

    if status != PrefixStatus.VALID:
        return status, None, parser.index

    value = decode_json_string(text[start:parser.index])

    return PrefixStatus.VALID, value, parser.index


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

    return check_string_prefix(key_text, ["name"])


def check_function_name_value(
    text: str,
    definitions: list[FunctionDefinition],
) -> PrefixStatus:
    """Check whether a JSON string can become a defined function name."""
    names = [definition.name for definition in definitions]
    return check_string_prefix(text, names)


def check_arguments_key_value(text: str) -> PrefixStatus:
    """Check whether a JSON string can become the arguments key."""
    return check_string_prefix(text, ["arguments"])


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


def check_unused_argument_key_value(
    text: str,
    definition: FunctionDefinition,
    used_names: set[str],
) -> PrefixStatus:
    """Check whether a JSON string can become an unused argument name."""
    allowed_names = get_unused_parameter_names(definition, used_names)
    return check_string_prefix(text, allowed_names)


def parse_parameter_value(
    parser: JSONPrefixParser,
    parameter_type: str,
) -> PrefixStatus:
    """Parse a parameter value according to its declared type."""
    parser.skip_whitespace()
    char = parser.current_char()

    if char is None:
        return PrefixStatus.INCOMPLETE

    if parameter_type == "string":
        if char != '"':
            return PrefixStatus.INVALID
        return parser.parse_string()

    if parameter_type == "number":
        if char != "-" and char not in "0123456789":
            return PrefixStatus.INVALID
        return parser.parse_number()

    return PrefixStatus.INVALID


def parse_argument_key(
    parser: JSONPrefixParser,
    text: str,
    definition: FunctionDefinition,
    used_names: set[str],
) -> tuple[PrefixStatus, str | None]:
    """Parse and validate one argument key. ie names & preventing dups"""
    start = parser.index

    status = parser.parse_string()

    if status == PrefixStatus.INCOMPLETE:
        key_prefix = text[start:]
        status = check_unused_argument_key_value(
            key_prefix,
            definition,
            used_names,
        )
        return status, None

    if status != PrefixStatus.VALID:
        return status, None

    key_text = text[start:parser.index]
    key = decode_json_string(key_text)

    allowed_names = get_unused_parameter_names(
        definition,
        used_names,
    )

    if key not in allowed_names:
        return PrefixStatus.INVALID, None

    return PrefixStatus.VALID, key


def parse_argument_pair(
    parser: JSONPrefixParser,
    text: str,
    definition: FunctionDefinition,
    used_names: set[str],
) -> tuple[PrefixStatus, str | None]:
    """Parse one argument key-value pair."""
    status, key = parse_argument_key(
        parser,
        text,
        definition,
        used_names,
    )

    if status != PrefixStatus.VALID:
        return status, None
    if key is None:
        return PrefixStatus.INVALID, None

    parser.skip_whitespace()

    if not parser.consume(":"):
        if parser.current_char() is None:
            return PrefixStatus.INCOMPLETE, None
        return PrefixStatus.INVALID, None

    parameter_type = definition.parameters[key].type

    status = parse_parameter_value(
        parser,
        parameter_type,
    )

    if status != PrefixStatus.VALID:
        return status, None

    return PrefixStatus.VALID, key


def check_argument_object(
    text: str,
    definition: FunctionDefinition,
) -> PrefixStatus:
    """Validate an argument object against a function definition."""
    parser = JSONPrefixParser(text)
    used_names: set[str] = set()

    parser.skip_whitespace()

    if not parser.consume("{"):
        if parser.current_char() is None:
            return PrefixStatus.INCOMPLETE
        return PrefixStatus.INVALID

    parser.skip_whitespace()

    if parser.current_char() is None:
        return PrefixStatus.INCOMPLETE

    if parser.consume("}"):
        if not definition.parameters:
            return PrefixStatus.VALID
        return PrefixStatus.INVALID

    while True:
        status, key = parse_argument_pair(
            parser,
            text,
            definition,
            used_names,
        )

        if status != PrefixStatus.VALID:
            return status

        if key is None:
            return PrefixStatus.INVALID

        used_names.add(key)

        parser.skip_whitespace()

        if parser.consume("}"):
            parser.skip_whitespace()

            if parser.current_char() is not None:
                return PrefixStatus.INVALID

            if used_names == set(definition.parameters.keys()):
                return PrefixStatus.VALID

            return PrefixStatus.INVALID

        if not parser.consume(","):
            if parser.current_char() is None:
                return PrefixStatus.INCOMPLETE
            return PrefixStatus.INVALID

        parser.skip_whitespace()

        if parser.current_char() is None:
            return PrefixStatus.INCOMPLETE


def check_tool_call_prefix(
    text: str,
    definitions: list[FunctionDefinition],
) -> PrefixStatus:
    """Validate a tool-call JSON prefix against function definitions."""
    parser = JSONPrefixParser(text)
    parser.skip_whitespace()

    if not parser.consume("{"):
        if parser.current_char() is None:
            return PrefixStatus.INCOMPLETE
        return PrefixStatus.INVALID

    parser.skip_whitespace()

    if parser.current_char() is None:
        return PrefixStatus.INCOMPLETE

    key_start = parser.index
    status = parser.parse_string()

    if status == PrefixStatus.INCOMPLETE:
        return check_first_root_key_prefix(text)

    if status != PrefixStatus.VALID:
        return PrefixStatus.INVALID

    key = decode_json_string(text[key_start:parser.index])

    if key != "name":
        return PrefixStatus.INVALID

    parser.skip_whitespace()

    if not parser.consume(":"):
        if parser.current_char() is None:
            return PrefixStatus.INCOMPLETE
        return PrefixStatus.INVALID

    parser.skip_whitespace()

    if parser.current_char() is None:
        return PrefixStatus.INCOMPLETE

    name_start = parser.index

    status, name, name_end = extract_json_string(
        text,
        name_start,
    )

    if status == PrefixStatus.INCOMPLETE:
        return check_function_name_value(text[name_start:], definitions)

    if status != PrefixStatus.VALID or name is None:
        return PrefixStatus.INVALID

    definition = find_function_definition(name, definitions)

    if definition is None:
        return PrefixStatus.INVALID

    parser.index = name_end
    parser.skip_whitespace()

    if not parser.consume(","):
        if parser.current_char() is None:
            return PrefixStatus.INCOMPLETE
        return PrefixStatus.INVALID

    parser.skip_whitespace()

    if parser.current_char() is None:
        return PrefixStatus.INCOMPLETE

    key_start = parser.index
    status = parser.parse_string()

    if status == PrefixStatus.INCOMPLETE:
        return check_arguments_key_value(text[key_start:])

    if status != PrefixStatus.VALID:
        return PrefixStatus.INVALID

    key = decode_json_string(text[key_start:parser.index])

    if key != "arguments":
        return PrefixStatus.INVALID

    parser.skip_whitespace()

    if not parser.consume(":"):
        if parser.current_char() is None:
            return PrefixStatus.INCOMPLETE
        return PrefixStatus.INVALID

    parser.skip_whitespace()

    if parser.current_char() is None:
        return PrefixStatus.INCOMPLETE

    arguments_start = parser.index

    status = parser.parse_object()

    if status != PrefixStatus.VALID:
        return check_argument_object(
            text[arguments_start:],
            definition,
        )

    arguments_end = parser.index

    status = check_argument_object(
        text[arguments_start:arguments_end],
        definition,
    )

    if status != PrefixStatus.VALID:
        return status

    parser.skip_whitespace()

    if not parser.consume("}"):
        if parser.current_char() is None:
            return PrefixStatus.INCOMPLETE
        return PrefixStatus.INVALID

    parser.skip_whitespace()

    if parser.current_char() is not None:
        return PrefixStatus.INVALID

    return PrefixStatus.VALID
