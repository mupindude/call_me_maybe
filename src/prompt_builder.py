
import json
from src.models import FunctionDefinition


def build_function_call_prompt(
    user_prompt: str,
    definitions: list[FunctionDefinition],
) -> str:
    """Build a prompt that guides the LLM to select a function."""
    function_data = [
        definition.model_dump()
        for definition in definitions
    ]

    formatted_definitions = json.dumps(
        function_data,
        indent=2,
    )

    return f"""You are a function-calling assistant.

Your task is to select the most appropriate function for the user's request.

Available functions:
{formatted_definitions}

User request:
{user_prompt}

Return only a valid JSON object with exactly these keys:
- "name": the selected function's name
- "parameters": an object containing the required arguments

Rules:
- Select only from the available functions.
- Use the exact parameter names from the selected function.
- Use the correct data types for all parameter values.
- Include all required parameters.
- Do not include explanations or Markdown.
"""


def build_tool_definitions(
    definitions: list[FunctionDefinition],
) -> list[dict[str, object]]:
    """Convert function definitions into Qwen tool definitions."""
    tools: list[dict[str, object]] = []

    for definition in definitions:
        properties: dict[str, dict[str, str]] = {}

        for param_name, param_definition in (
            definition.parameters.items()
        ):
            properties[param_name] = {
                "type": param_definition.type
            }

        function_parameters: dict[str, object] = {
            "type": "object",
            "properties": properties,
            "required": list(definition.parameters.keys()),
        }

        function_details: dict[str, object] = {
            "name": definition.name,
            "description": definition.description,
            "parameters": function_parameters,
        }

        tool: dict[str, object] = {
            "type": "function",
            "function": function_details,
        }

        tools.append(tool)

    return tools


def build_qwen_prompt(
    user_prompt: str,
    definitions: list[FunctionDefinition],
) -> str:
    """Build a Qwen chat prompt for function calling."""
    tools = build_tool_definitions(definitions)

    formatted_tools = "\n".join(
        json.dumps(tool, ensure_ascii=False)
        for tool in tools
    )

    return (
        "<|im_start|>system\n"
        "You are a function-calling assistant. Call exactly "
        "one available function; never answer directly.\n\n"
        "Available functions:\n"
        "<tools>\n"
        f"{formatted_tools}\n"
        "</tools>\n\n"
        "Rules:\n"
        "- Use only an available function.\n"
        "- Use its exact parameter names and correct types.\n"
        "- Include every required parameter.\n"
        "- Use integers without decimals, numbers as numeric "
        "values, booleans as true or false, and strings as "
        "quoted text.\n"
        "- Never use a boolean where an integer is required.\n"
        "- For template parameters, copy the user's template "
        "character-for-character. Preserve spaces, braces, "
        "quotes, apostrophes, and punctuation. Never rewrite "
        "or normalize it.\n\n"
        "Example:\n"
        'User: Format template: Say "hello" to {name}\n'
        'Argument: {"template": "Say \\"hello\\" to {name}"}\n\n'
        "Return exactly one function call in this format:\n"
        "<tool_call>\n"
        '{"name": <function-name>, '
        '"arguments": <args-json-object>}\n'
        "</tool_call>"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
