"""Base LLM backend interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generator


class BaseLLMBackend(ABC):
    """
    Base class for LLM backends.

    All LLM backends must implement this interface.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backend.

        Args:
            config: Backend configuration
        """
        self._config = config
        self._model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the LLM model into memory."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model and free resources."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: User prompt
            context: Conversation history as list of {role, content} dicts
            **kwargs: Additional generation parameters

        Returns:
            Generated response text
        """
        pass

    def generate_stream(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response.

        Args:
            prompt: User prompt
            context: Conversation history
            **kwargs: Additional generation parameters

        Yields:
            Response tokens
        """
        # Default implementation: just yield full response
        yield self.generate(prompt, context, **kwargs)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "loaded": self.is_loaded,
            "backend": self.__class__.__name__
        }
