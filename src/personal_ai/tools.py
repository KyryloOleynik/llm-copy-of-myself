from __future__ import annotations

import ast
import json
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from personal_ai.google_calendar import query_google_calendar
from personal_ai.retrieval import search_retrieval


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolRuntime:
    retrieval_database: Path
    google_credentials_file: Path
    google_token_file: Path
    google_calendar_ids: tuple[str, ...]
    google_calendar_time_zone: str


@dataclass(frozen=True)
class ToolDefinition:
    schema: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolRuntime], Any]


_TOOL_REGISTRY: dict[str, ToolDefinition] = {}
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def register_tool(
    schema: dict[str, Any],
    handler: Callable[[dict[str, Any], ToolRuntime], Any],
) -> None:
    """Register one schema and handler; all prompts receive the resulting registry."""
    name = schema["function"]["name"]
    if name in _TOOL_REGISTRY:
        raise ValueError(f"Tool is already registered: {name}")
    _TOOL_REGISTRY[name] = ToolDefinition(schema=schema, handler=handler)


def parse_tool_calls(output: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for payload in _TOOL_CALL_PATTERN.findall(output):
        value = json.loads(payload)
        name = value.get("name")
        arguments = value.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError("Tool call must contain a string name and object arguments")
        calls.append(ToolCall(name=name, arguments=arguments))
    return calls


def _calculate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _calculate_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _calculate_node(node.left)
        right = _calculate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("Exponent is too large")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_calculate_node(node.operand))
    raise ValueError("Unsupported calculator expression")


def calculate(expression: str) -> int | float:
    if not expression or len(expression) > 128:
        raise ValueError("Calculator expression is empty or too long")
    value = _calculate_node(ast.parse(expression, mode="eval"))
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return round(value, 10) if isinstance(value, float) else value


def _calculate_handler(arguments: dict[str, Any], _runtime: ToolRuntime) -> dict[str, Any]:
    return {"result": calculate(str(arguments.get("expression", "")))}


def _memory_handler(arguments: dict[str, Any], runtime: ToolRuntime) -> dict[str, Any]:
    return {
        "results": search_retrieval(
            runtime.retrieval_database,
            str(arguments.get("query", "")),
            int(arguments.get("limit", 5)),
        )
    }


def _calendar_handler(arguments: dict[str, Any], runtime: ToolRuntime) -> dict[str, Any]:
    return query_google_calendar(
        runtime.google_credentials_file,
        runtime.google_token_file,
        list(runtime.google_calendar_ids),
        str(arguments.get("action", "")),
        str(arguments.get("start", "")),
        str(arguments.get("end", "")),
        query=str(arguments.get("query", "")),
        minimum_free_minutes=int(arguments.get("minimum_free_minutes", 30)),
        limit=int(arguments.get("limit", 20)),
        time_zone=runtime.google_calendar_time_zone,
    )


register_tool(
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Safely calculate a numeric arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression using numbers and + - * / // % **.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    _calculate_handler,
)
register_tool(
    {
        "type": "function",
        "function": {
            "name": "search_personal_memory",
            "description": (
                "Search Rodion's private identity, story, notes, and Telegram memory. "
                "Use it instead of guessing personal facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific personal fact or memory to retrieve.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of passages to return, from 1 to 10.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    _memory_handler,
)
register_tool(
    {
        "type": "function",
        "function": {
            "name": "query_google_calendar",
            "description": (
                "Read Rodion's authorized Google Calendar. Use action 'events' to find "
                "what he is doing, or 'free_time' to find when he is available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["events", "free_time"],
                        "description": "Whether to return scheduled events or free intervals.",
                    },
                    "start": {
                        "type": "string",
                        "description": "Inclusive ISO-8601 start datetime with timezone.",
                    },
                    "end": {
                        "type": "string",
                        "description": "Inclusive ISO-8601 end datetime with timezone.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional event text search; only used for events.",
                        "default": "",
                    },
                    "minimum_free_minutes": {
                        "type": "integer",
                        "description": "Shortest acceptable free interval, from 1 to 1440 minutes.",
                        "default": 30,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of events, from 1 to 50.",
                        "default": 20,
                    },
                },
                "required": ["action", "start", "end"],
            },
        },
    },
    _calendar_handler,
)


TOOL_SCHEMAS = [definition.schema for definition in _TOOL_REGISTRY.values()]


def execute_tool_call(
    call: ToolCall,
    retrieval_database: Path,
    google_credentials_file: Path = Path("private_data/google_calendar_credentials.json"),
    google_token_file: Path = Path("private_data/google_calendar_token.json"),
    google_calendar_ids: tuple[str, ...] = ("primary",),
    google_calendar_time_zone: str = "Europe/Kyiv",
) -> str:
    definition = _TOOL_REGISTRY.get(call.name)
    if definition is None:
        raise ValueError(f"Unsupported tool: {call.name}")
    result = definition.handler(
        call.arguments,
        ToolRuntime(
            retrieval_database=retrieval_database,
            google_credentials_file=google_credentials_file,
            google_token_file=google_token_file,
            google_calendar_ids=google_calendar_ids,
            google_calendar_time_zone=google_calendar_time_zone,
        ),
    )
    return json.dumps(result, ensure_ascii=False)
