import numpy as np
from src.json_prefix import PrefixStatus
from src.models import FunctionDefinition
from src.schema_prefix import check_tool_call_prefix
from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import (
    decode_utf8_prefix,
    get_added_token_id,
    mask_invalid_logits,
    select_next_token,
)


def select_valid_token(
    logits: list[float],
    generated_bytes: bytes,
    token_bytes: dict[int, bytes],
    definitions: list[FunctionDefinition],
) -> tuple[int, bytes]:
    """Select the highest-scoring token that preserves a valid JSON prefix."""
    scores = np.asarray(logits, dtype=np.float64)
    finite_ids = np.flatnonzero(np.isfinite(scores))

    if len(finite_ids) == 0:
        raise ValueError("No valid token candidates")

    candidate_count = min(64, len(finite_ids))

    while True:
        candidate_positions = np.argpartition(
            scores[finite_ids],
            -candidate_count,
        )[-candidate_count:]

        candidate_ids = finite_ids[candidate_positions]
        candidate_ids = sorted(
            candidate_ids,
            key=lambda token_id: scores[token_id],
            reverse=True,
        )

        for token_id in candidate_ids:
            token_piece = token_bytes.get(int(token_id))

            if token_piece is None:
                continue

            candidate_bytes = generated_bytes + token_piece

            try:
                candidate_text, complete = decode_utf8_prefix(
                    candidate_bytes
                )
            except ValueError:
                continue

            status = check_tool_call_prefix(
                candidate_text,
                definitions,
            )

            if status == PrefixStatus.INVALID:
                continue

            if not complete and status != PrefixStatus.INCOMPLETE:
                continue

            return int(token_id), token_piece

        if candidate_count == len(finite_ids):
            break

        candidate_count = min(candidate_count * 2, len(finite_ids))

    raise ValueError("No token preserves a valid JSON prefix")


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


def generate_constrained_text(
    model: Small_LLM_Model,
    prompt: str,
    token_bytes: dict[int, bytes],
    definitions: list[FunctionDefinition],
    max_new_tokens: int = 100,
) -> str:
    """Generate a function call while enforcing its JSON schema."""
    opening_id = get_added_token_id(model, "<tool_call>")
    closing_id = get_added_token_id(model, "</tool_call>")
    end_id = get_added_token_id(model, "<|im_end|>")

    input_ids = model.encode(prompt).tolist()[0]
    generated_token_ids: list[int] = []

    # Force the opening structural marker.
    input_ids.append(opening_id)
    generated_token_ids.append(opening_id)

    generated_bytes = b""
    json_complete = False

    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)

        token_id, token_piece = select_valid_token(
            logits,
            generated_bytes,
            token_bytes,
            definitions,
        )

        generated_bytes += token_piece
        input_ids.append(token_id)
        generated_token_ids.append(token_id)

        json_text, complete = decode_utf8_prefix(generated_bytes)

        if complete:
            status = check_tool_call_prefix(json_text, definitions)
            if status == PrefixStatus.VALID:
                json_complete = True
                break

    if not json_complete:
        raise ValueError("Maximum tokens reached before completing JSON")

    # Force the closing structural markers.
    input_ids.append(closing_id)
    generated_token_ids.append(closing_id)

    input_ids.append(end_id)
    generated_token_ids.append(end_id)

    decoded_text = model.decode(generated_token_ids)

    if isinstance(decoded_text, str):
        return decoded_text

    raise TypeError("Model decoder did not return a string")
