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


def main() -> None:
    """Test function-call generation with the Qwen model."""
    definitions = load_function_definitions("data/input/functions_definition.json")

    test_cases = load_test_cases("data/input/function_calling_tests.json")

    model = Small_LLM_Model()

    id_to_token = load_id_to_token(model)
    valid_token_ids = get_valid_token_ids(id_to_token)
    end_token_id = get_end_token_id(model)

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

        print(f"\nUser prompt: {test_case.prompt}")
        print("Generated function call:")
        print(generated_text)


if __name__ == "__main__":
    main()
