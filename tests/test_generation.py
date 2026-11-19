import json
from llm_sdk import Small_LLM_Model

from src.generation import generate_text
from src.load_data import (
    load_function_definitions,
    load_test_cases,
)
from src.prompt_builder import build_qwen_prompt
from src.tokenizer_utils import (
    get_end_token_id,
    get_valid_token_ids,
    load_id_to_token,
)
from src.validate_output import process_generated_output
from src.models import FunctionCallOutput


def main() -> None:
    """Test function-call generation with the Qwen model."""
    definitions = load_function_definitions("data/input/functions_definition.json")

    test_cases = load_test_cases("data/input/function_calling_tests.json")

    model = Small_LLM_Model()

    id_to_token = load_id_to_token(model)
    valid_token_ids = get_valid_token_ids(id_to_token)
    end_token_id = get_end_token_id(model)
    results: list[FunctionCallOutput] = []
    for test_case in test_cases:
        prompt = build_qwen_prompt(
            test_case.prompt,
            definitions,
        )

        generated_text = generate_text(
            model,
            prompt,
            valid_token_ids,
            end_token_id,
            max_new_tokens=60,
        )

        try:
            result = process_generated_output(
                generated_text,
                test_case.prompt,
                definitions,
            )
            results.append(result)
            print("Validated output:", result.model_dump())

        except (ValueError, TypeError) as error:
            print("Failed to process generated output:", error)

        print(f"\nUser prompt: {test_case.prompt}")
        print("Generated function call:")
        print(generated_text)

    output_path = "data/output/function_calls.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            [result.model_dump() for result in results],
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nSaved {len(results)} function calls to {output_path}")


if __name__ == "__main__":
    main()
