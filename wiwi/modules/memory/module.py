"""Short-term memory module for Wiwi4.0."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from wiwi.interfaces.base_module import BaseModule, ModuleInfo, ModuleState
from wiwi.interfaces.ports import PortType
from wiwi.interfaces.messages import Message


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryModule(BaseModule):
    """
    Short-term conversation memory module.

    Stores conversation history and provides context for LLM requests.

    Features:
    - Rolling history with configurable max length
    - Context building for LLM
    - History clearing and management
    - Message metadata storage

    Config options:
        max_history_length: Maximum number of messages to store (default: 20)
        max_tokens_per_turn: Maximum tokens per message (default: 500)
    """

    @property
    def module_info(self) -> ModuleInfo:
        return ModuleInfo(
            name="memory",
            version="1.0.0",
            description="Short-term conversation memory",
            author="Wiwi Team",
            category="memory",
            input_ports={PortType.TEXT_IN, PortType.TEXT_OUT, PortType.MEMORY_WRITE, PortType.COMMAND_IN},
            output_ports={PortType.TEXT_IN, PortType.MEMORY_READ},
            dependencies=set(),
            optional_dependencies=set(),
            config_schema={
                "max_history_length": {"type": "integer", "default": 20},
                "max_tokens_per_turn": {"type": "integer", "default": 500}
            }
        )

    def __init__(self, kernel, config: Dict[str, Any]):
        super().__init__(kernel, config)

        self._max_history = config.get("max_history_length", 20)
        self._max_tokens = config.get("max_tokens_per_turn", 500)
        self._history: deque[ConversationTurn] = deque(maxlen=self._max_history)

    async def initialize(self) -> None:
        """Initialize the memory module."""
        self._logger.info(
            f"Memory module initialized (max_history={self._max_history})"
        )

    async def start(self) -> None:
        """Start the memory module."""
        self._state = ModuleState.RUNNING

    async def stop(self) -> None:
        """Stop and clear the memory module."""
        self._history.clear()
        self._state = ModuleState.STOPPED

    async def handle_input(self, message: Message) -> Optional[Message]:
        """
        Handle incoming messages.

        TEXT_IN: User message - add to history and forward to LLM with context
        TEXT_OUT: LLM response - store in history
        MEMORY_WRITE: Store assistant response
        COMMAND_IN: Handle memory commands (clear, history, etc.)
        """
        if message.port == PortType.TEXT_IN:
            return await self._handle_user_message(message)

        elif message.port == PortType.TEXT_OUT:
            # Store LLM response in history
            if message.source == "llm_brain":
                return await self._handle_llm_response(message)

        elif message.port == PortType.MEMORY_WRITE:
            return await self._handle_memory_write(message)

        elif message.port == PortType.COMMAND_IN:
            return await self._handle_command(message)

        return None

    async def _handle_llm_response(self, message: Message) -> None:
        """Store LLM response in history."""
        content = str(message.payload)

        turn = ConversationTurn(
            role="assistant",
            content=content,
            metadata=message.metadata.copy()
        )
        self._history.append(turn)

        self._logger.debug(f"Stored LLM response ({len(content)} chars)")
        return None

    async def _handle_user_message(self, message: Message) -> Message:
        """Handle incoming user message."""
        content = str(message.payload)

        # Truncate if needed
        if len(content) > self._max_tokens * 4:  # Rough char estimate
            content = content[:self._max_tokens * 4]
            self._logger.warning("User message truncated due to length")

        # Add to history
        turn = ConversationTurn(
            role="user",
            content=content,
            metadata=message.metadata.copy()
        )
        self._history.append(turn)

        # Build context for LLM
        context = self._build_context()

        # Forward to LLM with context
        return Message(
            source=self.name,
            target="llm_brain",
            port=PortType.TEXT_IN,
            payload=content,
            metadata={"context": context, "history_length": len(self._history)},
            correlation_id=message.correlation_id
        )

    async def _handle_memory_write(self, message: Message) -> None:
        """Store assistant response in history."""
        content = str(message.payload)

        turn = ConversationTurn(
            role="assistant",
            content=content,
            metadata=message.metadata.copy()
        )
        self._history.append(turn)

        self._logger.debug(f"Stored assistant response ({len(content)} chars)")
        return None

    async def _handle_command(self, message: Message) -> Optional[Message]:
        """Handle memory commands."""
        command = str(message.payload).lower().strip()

        if command == "clear":
            self._history.clear()
            return message.reply(
                payload="Memory cleared",
                port=PortType.TEXT_OUT
            )

        elif command == "history":
            history_text = self._format_history()
            return message.reply(
                payload=history_text,
                port=PortType.MEMORY_READ
            )

        elif command == "length":
            return message.reply(
                payload=str(len(self._history)),
                port=PortType.MEMORY_READ
            )

        return None

    def add_turn(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Добавить новый ход в историю."""
        if metadata is None:
            metadata = {}
        turn = ConversationTurn(role=role, content=content, metadata=metadata)
        self._history.append(turn)
        self._logger.debug(f"Added turn: {role} - {content[:50]}...")

    def _build_context(self) -> List[Dict[str, str]]:
        """
        Build context for LLM from conversation history.

        Returns:
            List of dicts with 'role' and 'content' keys
        """
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self._history
        ]

    def _format_history(self) -> str:
        """Format history as readable text."""
        if not self._history:
            return "No conversation history."

        lines = []
        for i, turn in enumerate(self._history, 1):
            role = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{i}. [{role}]: {turn.content[:100]}...")

        return "\n".join(lines)

    # === Public API ===

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self._history.clear()
        self._logger.info("History cleared")

    def get_history(self) -> List[ConversationTurn]:
        """Get copy of conversation history."""
        return list(self._history)

    def get_context(self) -> List[Dict[str, str]]:
        """Get context for LLM."""
        return self._build_context()

    def add_system_message(self, content: str) -> None:
        """Add a system message to history."""
        turn = ConversationTurn(
            role="system",
            content=content,
            metadata={"type": "system"}
        )
        self._history.append(turn)

    @property
    def history_length(self) -> int:
        """Current number of messages in history."""
        return len(self._history)
