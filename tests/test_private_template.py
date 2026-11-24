from llm_sdk import Small_LLM_Model

from src.generation import generate_constrained_text
from src.load_data import load_function_definitions
from src.prompt_builder import build_qwen_prompt
from src.tokenizer_utils import (
    build_byte_encoder,
    build_token_bytes,
    load_added_token_ids,
    load_id_to_token,
)
from src.validate_output import process_generated_output


def main() -> None:
    """Test the private template prompt in isolation."""
    definitions = load_function_definitions(
        "/home/devine/evaluation/moulinette/data/input/"
        "functions_definition.json"
    )

    prompt = 'Format template: Say "hello" to {name}'

    model = Small_LLM_Model()

    id_to_token = load_id_to_token(model)
    added_token_ids = load_added_token_ids(model)
    byte_encoder = build_byte_encoder()
    token_bytes = build_token_bytes(
        id_to_token,
        byte_encoder,
        added_token_ids,
    )

    qwen_prompt = build_qwen_prompt(prompt, definitions)

    generated_text = generate_constrained_text(
        model,
        qwen_prompt,
        token_bytes,
        definitions,
        max_new_tokens=100,
    )

    print("Generated text:")
    print(generated_text)

    result = process_generated_output(
        generated_text,
        prompt,
        definitions,
    )

    print("Validated result:", result.model_dump())


if __name__ == "__main__":
    main()
