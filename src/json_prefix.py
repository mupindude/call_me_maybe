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

    def parse_string(self) -> PrefixStatus:
        """Validate a JSON string and advance past its closing quote."""
        start = self.index

        if self.current_char() != '"':
            return PrefixStatus.INVALID

        self.index += 1

        while self.index < len(self.text):
            char = self.text[self.index]

            if char == '"':
                self.index += 1
                return PrefixStatus.VALID

            if ord(char) < 0x20:
                self.index = start
                return PrefixStatus.INVALID

            if char == "\\":
                self.index += 1

                if self.index >= len(self.text):
                    self.index = start
                    return PrefixStatus.INCOMPLETE

                escape = self.text[self.index]

                if escape == "u":
                    hex_start = self.index + 1
                    hex_end = hex_start + 4
                    digits = self.text[hex_start:hex_end]

                    if any(c not in "0123456789abcdefABCDEF"
                           for c in digits):
                        self.index = start
                        return PrefixStatus.INVALID

                    if len(digits) < 4:
                        self.index = start
                        return PrefixStatus.INCOMPLETE

                    self.index = hex_end

                elif escape in '"\\/bfnrt':
                    self.index += 1

                else:
                    self.index = start
                    return PrefixStatus.INVALID

            else:
                self.index += 1

        self.index = start
        return PrefixStatus.INCOMPLETE

    def parse_number(self) -> PrefixStatus:
        """Validate a JSON number and advance past it."""
        start = self.index

        while (
            self.index < len(self.text)
            and self.text[self.index] in "0123456789-+.eE"
        ):
            self.index += 1

        number_text = self.text[start:self.index]
        status = check_json_number(number_text)

        if status != PrefixStatus.VALID:
            self.index = start

        return status

    def parse_literal(self) -> PrefixStatus:
        """Validate a JSON literal and advance past it."""
        start = self.index
        remaining = self.text[start:]

        for literal in ("true", "false", "null"):
            if remaining.startswith(literal):
                self.index += len(literal)
                return PrefixStatus.VALID

        for literal in ("true", "false", "null"):
            if literal.startswith(remaining):
                return PrefixStatus.INCOMPLETE

        return PrefixStatus.INVALID

    def parse_value(self) -> PrefixStatus:
        """Parse a JSON value based on its starting character."""
        self.skip_whitespace()
        char = self.current_char()

        if char is None:
            return PrefixStatus.INCOMPLETE

        if char == '"':
            return self.parse_string()

        if char == "-" or is_json_digit(char):
            return self.parse_number()

        if char in "tfn":
            return self.parse_literal()

        if char in "{[":
            return PrefixStatus.INCOMPLETE

        return PrefixStatus.INVALID

    def parse_object(self) -> PrefixStatus:
        """Parse a JSON object containing zero or more key-value pairs."""
        if not self.consume("{"):
            return PrefixStatus.INVALID

        self.skip_whitespace()

        if self.consume("}"):
            return PrefixStatus.VALID

        while True:
            status = self.parse_object_pair()

            if status != PrefixStatus.VALID:
                return status

            self.skip_whitespace()

            if self.consume("}"):
                return PrefixStatus.VALID

            if not self.consume(","):
                if self.current_char() is None:
                    return PrefixStatus.INCOMPLETE
                return PrefixStatus.INVALID

            self.skip_whitespace()

    def parse_object_pair(self) -> PrefixStatus:
        """Parse one key-value pair inside an object."""
        if self.current_char() != '"':
            if self.current_char() is None:
                return PrefixStatus.INCOMPLETE
            return PrefixStatus.INVALID

        status = self.parse_string()
        if status != PrefixStatus.VALID:
            return status

        self.skip_whitespace()

        if not self.consume(":"):
            if self.current_char() is None:
                return PrefixStatus.INCOMPLETE
            return PrefixStatus.INVALID

        return self.parse_value()
