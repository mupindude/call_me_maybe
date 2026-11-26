
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


def load_added_token_ids(model: Small_LLM_Model) -> set[int]:
    """Load the IDs of all added tokenizer tokens."""
    path = model.get_path_to_tokenizer_file()

    with open(path, "r", encoding="utf-8") as file:
        tokenizer = json.load(file)

    return {
        token["id"]
        for token in tokenizer["added_tokens"]
        if isinstance(token.get("id"), int)
    }


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
    """Retrieve the end-of-message token ID."""
    return get_added_token_id(model, "<|im_end|>")


def build_byte_encoder() -> dict[str, int]:
    """Build a mapping from ByteLevel characters to their byte values."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = byte_values.copy()

    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1

    return {
        chr(unicode_value): byte_value
        for byte_value, unicode_value in zip(
            byte_values,
            unicode_values,
        )
    }


def token_to_bytes(
    token: str,
    byte_encoder: dict[str, int],
) -> bytes:
    """Convert a ByteLevel vocabulary token into its original bytes."""
    try:
        return bytes(byte_encoder[char] for char in token)
    except KeyError as error:
        raise ValueError("Token contains an unknown ByteLevel "
                         "character") from error


def build_token_bytes(
    id_to_token: dict[int, str],
    byte_encoder: dict[str, int],
    added_token_ids: set[int],
) -> dict[int, bytes]:
    """Build a byte mapping for ordinary vocabulary tokens."""
    token_bytes: dict[int, bytes] = {}

    for token_id, token in id_to_token.items():
        if token_id in added_token_ids:
            continue

        token_bytes[token_id] = token_to_bytes(
            token,
            byte_encoder,
        )

    return token_bytes


def decode_utf8_prefix(data: bytes) -> tuple[str, bool]:
    """Decode complete UTF-8 text and identify incomplete sequences."""
    try:
        return data.decode("utf-8"), True
    except UnicodeDecodeError as error:
        if error.reason == "unexpected end of data":
            try:
                prefix = data[:error.start].decode("utf-8")
                return prefix, False
            except UnicodeDecodeError:
                raise ValueError("Invalid UTF-8 prefix") from error
        raise ValueError("Invalid UTF-8 prefix") from error


def get_added_token_id(
    model: Small_LLM_Model,
    content: str,
) -> int:
    """Retrieve the ID of an added token by its content."""
    path = model.get_path_to_tokenizer_file()

    with open(path, "r", encoding="utf-8") as file:
        tokenizer = json.load(file)

    for token in tokenizer["added_tokens"]:
        if token["content"] == content:
            token_id = token["id"]
            if isinstance(token_id, int):
                return token_id

    raise ValueError(f"Added token not found: {content}")


def load_tokenizer_data(
    model: Small_LLM_Model,
) -> tuple[dict[int, str], set[int], dict[str, int]]:
    """Load vocabulary and added-token information in one pass."""
    path = model.get_path_to_tokenizer_file()

    with open(path, "r", encoding="utf-8") as file:
        tokenizer = json.load(file)

    id_to_token = {
        token_id: token
        for token, token_id in tokenizer["model"]["vocab"].items()
    }

    added_token_ids: set[int] = set()
    added_tokens: dict[str, int] = {}

    for token in tokenizer["added_tokens"]:
        token_id = token["id"]
        content = token["content"]

        id_to_token[token_id] = content
        added_token_ids.add(token_id)
        added_tokens[content] = token_id

    return id_to_token, added_token_ids, added_tokens
