"""Base TTS backend interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional
import numpy as np


class BaseTTSBackend(ABC):
    """
    Base class for TTS backends.

    All TTS backends must implement this interface.

    Config options (vary by backend):
        model: Model name or path
        device: Device for inference (cuda, cuda:0, cuda:1, cpu)
        language: Language code (ru, en, etc.)
        sample_rate: Output sample rate in Hz
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the backend with configuration."""
        self._config = config
        self._model = None
        self._sample_rate = config.get("sample_rate", 24000)

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the TTS model into memory.

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
    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Synthesize speech from text (batch mode).

        Args:
            text: Text to synthesize
            language: Language code override (e.g., "ru", "en"). If None, uses config.
            **kwargs: Additional backend-specific parameters.

        Returns:
            Audio data as float32 numpy array, normalized to [-1, 1]

        Raises:
            RuntimeError: If model is not loaded.
        """
        pass

    @abstractmethod
    def synthesize_stream(
        self,
        text: str,
        language: Optional[str] = None,
        **kwargs
    ) -> Iterator[np.ndarray]:
        """
        Synthesize speech from text with streaming output.

        Yields audio chunks as they are generated for low-latency playback.

        Args:
            text: Text to synthesize
            language: Language code override (e.g., "ru", "en"). If None, uses config.
            **kwargs: Additional backend-specific parameters.

        Yields:
            Audio chunks as float32 numpy arrays, normalized to [-1, 1]

        Raises:
            RuntimeError: If model is not loaded.
        """
        pass

    def set_voice(self, audio_path: str) -> None:
        """
        Set the voice from a reference audio file (for voice cloning backends).

        Args:
            audio_path: Path to reference audio file (WAV format recommended)

        Raises:
            NotImplementedError: If backend doesn't support voice cloning.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support voice cloning"
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._model is not None

    @property
    def sample_rate(self) -> int:
        """Get the output sample rate in Hz."""
        return self._sample_rate

    @property
    def supports_streaming(self) -> bool:
        """Check if backend supports streaming synthesis."""
        return True

    @property
    def supports_voice_cloning(self) -> bool:
        """Check if backend supports voice cloning."""
        return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information (loaded, backend name, etc.)
        """
        return {
            "loaded": self.is_loaded,
            "backend": self.__class__.__name__,
            "sample_rate": self._sample_rate,
            "supports_streaming": self.supports_streaming,
            "supports_voice_cloning": self.supports_voice_cloning,
        }
