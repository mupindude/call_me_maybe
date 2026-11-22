
import json
from llm_sdk import Small_LLM_Model

from src.generation import generate_constrained_text
from src.load_data import (
    load_function_definitions,
    load_test_cases,
)
from src.prompt_builder import build_qwen_prompt
from src.tokenizer_utils import (
    build_byte_encoder,
    build_token_bytes,
    load_added_token_ids,
    load_id_to_token,
)
from src.validate_output import process_generated_output
from src.models import FunctionCallOutput


def main() -> None:
    """Test constrained function-call generation with the Qwen model."""
    definitions = load_function_definitions(
        "data/input/functions_definition.json"
        )
    test_cases = load_test_cases("data/input/function_calling_tests.json")

    model = Small_LLM_Model()

    id_to_token = load_id_to_token(model)
    added_token_ids = load_added_token_ids(model)
    byte_encoder = build_byte_encoder()
    token_bytes = build_token_bytes(
        id_to_token,
        byte_encoder,
        added_token_ids,
    )

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
