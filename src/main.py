import json
import sys
from pathlib import Path

from src.arguments import parse_arguments
from src.generation import generate_constrained_text
from src.load_data import (
    load_function_definitions,
    load_test_cases,
)
from src.model_resources import initialize_model_resources
from src.models import FunctionCallOutput
from src.prompt_builder import build_qwen_prompt
from src.validate_output import process_generated_output


def save_results(
    results: list[FunctionCallOutput],
    output_path: str,
) -> None:
    """Save validated function calls to a JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            [result.model_dump() for result in results],
            file,
            indent=4,
            ensure_ascii=False,
        )


def main() -> None:
    """Run the function-calling application."""
    args = parse_arguments()

    definitions = load_function_definitions(
        args.functions_definition
    )
    test_cases = load_test_cases(args.input)

    if not definitions:
        print("No valid function definitions loaded.")
        sys.exit(1)

    if not test_cases:
        print("No valid test cases loaded.")
        sys.exit(1)

    try:
        model, token_bytes = initialize_model_resources()
        results: list[FunctionCallOutput] = []

        for test_case in test_cases:
            prompt = build_qwen_prompt(
                test_case.prompt,
                definitions,
            )

            generated_text = generate_constrained_text(
                model,
                prompt,
                token_bytes,
                definitions,
                max_new_tokens=100,
            )

            result = process_generated_output(
                generated_text,
                test_case.prompt,
                definitions,
            )
            results.append(result)

            print("Validated output:", result.model_dump())

        save_results(results, args.output)
        print(
            f"\nSuccessfully saved {len(results)} function calls "
            f"to {args.output}"
        )

    except Exception as error:
        print(f"Application error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
