"""Base class for all graph nodes."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from tinyts.state import AgentState

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """Base class for all workflow nodes.

    Each node is responsible for:
    1. Reading specific fields from AgentState
    2. Performing its designated task
    3. Updating AgentState with results
    4. Logging its actions
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

    def __call__(self, state: AgentState) -> AgentState:
        """Execute the node and return updated state.

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        self.logger.info(f"Executing node: {self.name}")
        try:
            updated_state = self.execute(state)
            self.logger.info(f"Node {self.name} completed successfully")
            return updated_state
        except Exception as e:
            self.logger.error(f"Node {self.name} failed: {str(e)}", exc_info=True)
            raise

    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """Execute the node's logic.

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        pass
