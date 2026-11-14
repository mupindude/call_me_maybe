import json

from pydantic import ValidationError

from src.models import FunctionDefinition, TestCase


def load_function_definitions(file_path: str) -> list[FunctionDefinition]:
    """Open a file, load its JSON and validate function definitions."""
    lst_defs: list[FunctionDefinition] = []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        for item in raw_data:
            valid_def = FunctionDefinition.model_validate(item)
            lst_defs.append(valid_def)

    except FileNotFoundError:
        print("File does not exist, check path.")
        return []
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return []
    except ValidationError as e:
        print(f"Invalid function definition: {e}")
        return []

    return lst_defs


def load_test_cases(file_path: str) -> list[TestCase]:
    """Read the test cases JSON and validate each prompt."""
    test_cases: list[TestCase] = []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        for item in raw_data:
            valid_case = TestCase.model_validate(item)
            test_cases.append(valid_case)

    except FileNotFoundError:
        print("File does not exist, check path.")
        return []
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return []
    except ValidationError as e:
        print(f"Invalid test case: {e}")
        return []

    return test_cases
