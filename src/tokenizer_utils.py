
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
