from src.generation import select_valid_token
from src.models import FunctionDefinition


def main() -> None:
    """Test schema-aware token selection without loading the model."""
    definition = FunctionDefinition(
        name="fn_add_numbers",
        description="Add two numbers.",
        parameters={
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        returns={"type": "number"},
    )

    definitions = [definition]

    generated = (
        b'{"name":"fn_add_numbers","arguments":{"a":'
    )

    token_bytes = {
        1: b"true",
        2: b"265",
        3: b"265.0",
    }

    logits = [-1.0, 10.0, 9.0, 8.0]

    token_id, token_piece = select_valid_token(
        logits,
        generated,
        token_bytes,
        definitions,
    )

    print("Selected token ID:", token_id)
    print("Selected token:", token_piece)
    print("Integer selected:", token_piece == b"265")


if __name__ == "__main__":
    main()
