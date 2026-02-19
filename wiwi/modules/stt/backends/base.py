"""Base STT backend interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseSTTBackend(ABC):
    """
    Base class for STT backends.

    All STT backends must implement this interface.

    Config options (vary by backend):
        model_size: Model size (tiny, base, small, medium, large)
        language: Language code (ru, en, etc.)
        device: Device for inference (cuda, cuda:0, cpu)
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the backend with configuration."""
        self._config = config
        self._model = None

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the STT model into memory.

        Raises:
            ImportError: If required dependencies are missing.
            RuntimeError: If model loading fails.
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model and free resources."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, mono, normalized to [-1, 1])
            sample_rate: Audio sample rate in Hz (default: 16000)
            language: Language code override (e.g., "ru", "en"). If None, uses config.
            **kwargs: Additional backend-specific parameters.

        Returns:
            Transcribed text string.

        Raises:
            RuntimeError: If model is not loaded.
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information (loaded, backend name, etc.)
        """
        return {
            "loaded": self.is_loaded,
            "backend": self.__class__.__name__
        }
