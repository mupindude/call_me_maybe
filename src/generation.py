from llm_sdk import Small_LLM_Model

from src.tokenizer_utils import (
    mask_invalid_logits,
    select_next_token,
)


def generate_text(
    model: Small_LLM_Model,
    prompt: str,
    valid_token_ids: set[int],
    end_token_id: int,
    max_new_tokens: int = 100,
) -> str:
    """Generate text using greedy decoding."""
    input_ids = model.encode(prompt).tolist()[0]
    generated_token_ids: list[int] = []

    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        masked_logits = mask_invalid_logits(logits, valid_token_ids)
        next_token_id = select_next_token(masked_logits)

        if next_token_id == end_token_id:
            break

        generated_token_ids.append(next_token_id)
        input_ids.append(next_token_id)

    decoded_text = model.decode(generated_token_ids)

    if isinstance(decoded_text, str):
        return decoded_text

    raise TypeError("Model decoder did not return a string")
