from enum import Enum


class PrefixStatus(Enum):
    """Represent the validation status of a JSON prefix."""

    VALID = "valid"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


def check_json_string(text: str) -> PrefixStatus:
    """Check whether text represents a complete or incomplete JSON string."""
    if not text.startswith('"'):
        return PrefixStatus.INVALID

    index = 1

    while index < len(text):
        char = text[index]

        if char == '"':
            if index == len(text) - 1:
                return PrefixStatus.VALID
            return PrefixStatus.INVALID

        if ord(char) < 0x20:
            return PrefixStatus.INVALID

        if char == "\\":
            index += 1

            if index >= len(text):
                return PrefixStatus.INCOMPLETE

            escape = text[index]

            if escape == "u":
                remaining = text[index + 1:index + 5]

                if any(c not in "0123456789abcdefABCDEF" for c in remaining):
                    return PrefixStatus.INVALID

                if len(remaining) < 4:
                    return PrefixStatus.INCOMPLETE

                index += 4

            elif escape not in '"\\/bfnrt':
                return PrefixStatus.INVALID

        index += 1

    return PrefixStatus.INCOMPLETE


def is_json_digit(char: str) -> bool:
    """Return whether a character is an ASCII digit."""
    return char in "0123456789"


def check_json_number(text: str) -> PrefixStatus:
    """Check whether text represents a valid or incomplete JSON number."""
    if not text:
        return PrefixStatus.INCOMPLETE

    index = 0
    length = len(text)

    if text[index] == "-":
        index += 1
        if index == length:
            return PrefixStatus.INCOMPLETE

    if text[index] == "0":
        index += 1
        if index < length and is_json_digit(text[index]):
            return PrefixStatus.INVALID
    elif text[index] in "123456789":
        while index < length and is_json_digit(text[index]):
            index += 1
    else:
        return PrefixStatus.INVALID

    if index < length and text[index] == ".":
        index += 1
        if index == length:
            return PrefixStatus.INCOMPLETE
        if not is_json_digit(text[index]):
            return PrefixStatus.INVALID
        while index < length and is_json_digit(text[index]):
            index += 1

    if index < length and text[index] in "eE":
        index += 1

        if index < length and text[index] in "+-":
            index += 1

        if index == length:
            return PrefixStatus.INCOMPLETE

        if not is_json_digit(text[index]):
            return PrefixStatus.INVALID

        while index < length and is_json_digit(text[index]):
            index += 1

    if index < length:
        return PrefixStatus.INVALID

    return PrefixStatus.VALID


def check_json_literal(text: str) -> PrefixStatus:
    """Check whether text is a valid or incomplete JSON literal."""
    literals = ("true", "false", "null")

    for literal in literals:
        if literal.startswith(text):
            if text == literal:
                return PrefixStatus.VALID
            return PrefixStatus.INCOMPLETE

    return PrefixStatus.INVALID


def skip_whitespace(text: str, index: int) -> int:
    """Return the index of the next non-whitespace character."""
    while index < len(text) and text[index] in " \t\n\r":
        index += 1
    return index


def check_json_value_start(text: str, index: int) -> PrefixStatus:
    """Check whether a JSON value can start at the given index."""
    if index >= len(text):
        return PrefixStatus.INCOMPLETE

    char = text[index]

    if char == '"':
        return check_json_string(text[index:])

    if char == "-" or is_json_digit(char):
        return check_json_number(text[index:])

    if char in "tfn":
        return check_json_literal(text[index:])

    if char in "{[":
        return PrefixStatus.INCOMPLETE

    return PrefixStatus.INVALID


class JSONPrefixParser:
    """Parse JSON text while tracking the current position."""

    def __init__(self, text: str) -> None:
        """Initialise the parser with JSON text."""
        self.text = text
        self.index = 0

    def skip_whitespace(self) -> None:
        """Advance past any JSON whitespace."""
        while (
            self.index < len(self.text)
            and self.text[self.index] in " \t\n\r"
        ):
            self.index += 1

    def current_char(self) -> str | None:
        """Return the current character, or None at the end."""
        if self.index >= len(self.text):
            return None
        return self.text[self.index]

    def consume(self, expected: str) -> bool:
        """Consume the expected character if it matches."""
        if self.current_char() == expected:
            self.index += 1
            return True
        return False
