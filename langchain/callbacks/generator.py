"""Callback Handler that yields output."""
from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish


class GeneratorCallbackHandler(BaseCallbackHandler):
    """Callback Handler that yields output."""

    buffer: List[Tuple[str, str]] = []
    """buffer containting tuples of text and source, where source is either "llm",
      "tool" or "chain".    
    """
    finished: bool = False
    pause: float = 0.2
    timeout: float = 5

    _LLM = "llm"
    _TOOL = "tool"
    _CHAIN = "chain"

    llm_prefix: Optional[str] = None

    # todo: fix attributiosn llm/tool/chain

    def __iter__(self) -> GeneratorCallbackHandler:
        return self

    def __next__(self) -> Tuple[str, str]:
        time_of_last_yield = time.time()
        while True:
            if len(self.buffer) == 0 and self.finished:
                raise StopIteration
            if len(self.buffer) == 0 and not self.finished:
                if time.time() - time_of_last_yield > self.timeout:
                    raise TimeoutError("Generator timed out")
                time.sleep(self.pause)
                continue

            time_of_last_yield = time.time()
            text = self.buffer.pop(0)
            return text

    def __init__(self, pause: float = 0.2, timout: float = 5) -> None:
        """Initialize callback handler."""
        self.finished = False
        self.pause = pause

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized["name"]
        self.buffer.append((f"Entering new {class_name} chain...", self._CHAIN))

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        self.buffer.append(("Finished chain.", self._CHAIN))

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        self.buffer.append((action.log, self._TOOL))

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        if observation_prefix is not None:
            output = observation_prefix + output
        self.buffer.append((output, self._TOOL))

        # save llm prefix for use in on_text
        self.llm_prefix = llm_prefix

    def on_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:
        """Run on arbitrary text."""
        if self.llm_prefix:
            text = self.llm_prefix + text
        self.buffer.append((text, self._LLM))

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        self.buffer.append((finish.log, self._TOOL))
        self.finished = True
