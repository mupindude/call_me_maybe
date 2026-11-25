from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import (
    build_byte_encoder,
    build_token_bytes,
    load_added_token_ids,
    load_id_to_token,
)


def initialize_model_resources(
) -> tuple[Small_LLM_Model, dict[int, bytes]]:
    """Initialize the model and tokenizer resources."""
    model = Small_LLM_Model()

    id_to_token = load_id_to_token(model)
    added_token_ids = load_added_token_ids(model)
    byte_encoder = build_byte_encoder()

    token_bytes = build_token_bytes(
        id_to_token,
        byte_encoder,
        added_token_ids,
    )

    return model, token_bytes
