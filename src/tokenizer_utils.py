
import json

from llm_sdk import Small_LLM_Model


def load_id_to_token(model: Small_LLM_Model) -> dict[int, str]:
    """Load the tokenizer vocabulary as a token-ID-to-string mapping."""
    path = model.get_path_to_tokenizer_file()

    with open(path, "r", encoding="utf-8") as file:
        tokenizer = json.load(file)

    id_to_token = {
        token_id: token
        for token, token_id in tokenizer["model"]["vocab"].items()
    }

    for added_token in tokenizer["added_tokens"]:
        id_to_token[added_token["id"]] = added_token["content"]

    return id_to_token


def get_valid_token_ids(id_to_token: dict[int, str]) -> set[int]:
    """Return all valid token IDs from the tokenizer vocabulary."""
    return set(id_to_token.keys())


def mask_invalid_logits(
    logits: list[float],
    valid_token_ids: set[int]
) -> list[float]:
    """Mask logits for token IDs absent from the tokenizer vocabulary."""
    masked_logits = logits.copy()

    for token_id in range(len(masked_logits)):
        if token_id not in valid_token_ids:
            masked_logits[token_id] = float("-inf")

    return masked_logits


def select_next_token(masked_logits: list[float]) -> int:
    """Select the token ID with the highest logit."""
    return max(
        range(len(masked_logits)),
        key=masked_logits.__getitem__
    )


def get_end_token_id(model: Small_LLM_Model) -> int:
    """Retrieve the end-of-message token ID from the tokenizer."""
    path = model.get_path_to_tokenizer_file()

    with open(path, "r", encoding="utf-8") as file:
        tokenizer = json.load(file)

    for token in tokenizer["added_tokens"]:
        if token["content"] == "<|im_end|>":
            token_id = token["id"]
            if isinstance(token_id, int):
                return token_id
    raise ValueError("Invalid end-of-message token ID")
