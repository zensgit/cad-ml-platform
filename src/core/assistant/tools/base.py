"""
Base class for Function Calling tools.

All tools inherit from BaseTool and implement the execute() method.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Abstract base class for all assistant tools.

    Each tool exposes:
    - ``name``: unique identifier used in LLM function-calling payloads
    - ``description``: human-readable description (Chinese, for the CAD domain)
    - ``input_schema``: JSON-Schema-style dict describing accepted parameters
    - ``execute(params)``: async method that runs the tool logic
    """

    name: str
    description: str
    input_schema: Dict[str, Any]

    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters.

        Args:
            params: Validated parameter dict matching ``input_schema``.

        Returns:
            Result dict.  On failure the dict should contain an ``"error"`` key.
        """
        ...

    def to_schema(self) -> Dict[str, Any]:
        """Return the tool definition in a provider-agnostic format.

        The structure is compatible with Anthropic's tool-use API and can be
        trivially converted to OpenAI function-calling format.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
