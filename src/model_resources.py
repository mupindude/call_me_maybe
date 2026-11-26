from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import (
    build_byte_encoder,
    build_token_bytes,
    load_tokenizer_data,
)


def initialize_model_resources() -> tuple[
    Small_LLM_Model,
    dict[int, bytes],
    dict[str, int],
]:
    """Initialize the model and tokenizer resources."""
    model = Small_LLM_Model()

    (
        id_to_token,
        added_token_ids,
        added_tokens,
    ) = load_tokenizer_data(model)

    byte_encoder = build_byte_encoder()

    token_bytes = build_token_bytes(
        id_to_token,
        byte_encoder,
        added_token_ids,
    )

    return model, token_bytes, added_tokens
