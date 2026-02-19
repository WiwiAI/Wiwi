"""faster-whisper backend for STT module."""

import logging
from typing import Any, Dict, Optional
import numpy as np

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

from wiwi.modules.stt.backends.base import BaseSTTBackend


class FasterWhisperBackend(BaseSTTBackend):
    """
    STT backend using faster-whisper (CTranslate2-based Whisper).

    This backend provides fast and efficient speech-to-text transcription
    using the CTranslate2 inference engine.

    Config options:
        model_size: Model size (tiny, base, small, medium, large-v2, large-v3)
                   Default: "base"
        device: Device to use (cuda, cuda:0, cuda:1, cpu, auto)
                Default: "cuda"
        compute_type: Compute type (int8, int8_float16, float16, float32)
                      Default: "int8" (recommended for GPU memory savings)
        language: Default language code (e.g., "ru", "en")
                  Default: "ru"
        beam_size: Beam size for decoding. Default: 5
        vad_filter: Enable built-in VAD filter. Default: True
        min_silence_duration_ms: Minimum silence duration for VAD. Default: 500
        speech_pad_ms: Padding around speech for VAD. Default: 400
    """

    MODEL_SIZES = {
        "tiny": "tiny",
        "base": "base",
        "small": "small",
        "medium": "medium",
        "large": "large-v2",
        "large-v2": "large-v2",
        "large-v3": "large-v3"
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._logger = logging.getLogger("wiwi.stt.faster_whisper")

        # Model configuration
        self._model_size = config.get("model_size", "base")
        self._compute_type = config.get("compute_type", "int8")
        self._language = config.get("language", "ru")

        # Parse device: "cuda", "cuda:0", "cuda:1", "cpu"
        device_str = config.get("device", "cuda")
        self._device_index = 0
        if device_str.startswith("cuda:"):
            try:
                self._device_index = int(device_str.split(":")[1])
                self._device = "cuda"
            except (ValueError, IndexError):
                self._device = "cuda"
        else:
            self._device = device_str

        # Decoding parameters
        self._beam_size = config.get("beam_size", 5)
        self._vad_filter = config.get("vad_filter", True)
        self._min_silence_ms = config.get("min_silence_duration_ms", 500)
        self._speech_pad_ms = config.get("speech_pad_ms", 400)

    def load_model(self) -> None:
        """Load the Whisper model."""
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper is required for this backend. "
                "Install it with: pip install faster-whisper"
            )

        model_name = self.MODEL_SIZES.get(self._model_size, self._model_size)

        self._logger.info(
            f"Loading faster-whisper model: {model_name} "
            f"on {self._device}:{self._device_index} with {self._compute_type}"
        )

        try:
            self._model = WhisperModel(
                model_name,
                device=self._device,
                device_index=self._device_index,
                compute_type=self._compute_type
            )
            self._logger.info("faster-whisper model loaded successfully")
        except Exception as e:
            self._logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load faster-whisper model: {e}")

    def unload_model(self) -> None:
        """Unload the model and free GPU memory."""
        if self._model:
            del self._model
            self._model = None
            self._logger.info("faster-whisper model unloaded")

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
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Audio sample rate (must be 16000 for Whisper)
            language: Language code override
            **kwargs: Additional parameters:
                - beam_size: Override beam size
                - vad_filter: Override VAD filter setting

        Returns:
            Transcribed text string.
        """
        if not self._model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure correct format: float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if audio is in int16 range
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0

        # Resample if needed (Whisper requires 16kHz)
        if sample_rate != 16000:
            self._logger.warning(
                f"Sample rate {sample_rate} != 16000. "
                "Whisper expects 16kHz audio. Transcription may be inaccurate."
            )

        lang = language or self._language
        beam_size = kwargs.get("beam_size", self._beam_size)
        vad_filter = kwargs.get("vad_filter", self._vad_filter)

        self._logger.debug(
            f"Transcribing {len(audio)/sample_rate:.2f}s of audio "
            f"(language={lang}, vad={vad_filter})"
        )

        try:
            segments, info = self._model.transcribe(
                audio,
                language=lang,
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters=dict(
                    min_silence_duration_ms=self._min_silence_ms,
                    speech_pad_ms=self._speech_pad_ms
                )
            )

            # Collect all segments
            text_parts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    text_parts.append(text)

            result = " ".join(text_parts).strip()
            self._logger.debug(f"Transcribed: '{result[:50]}...' ({len(result)} chars)")
            return result

        except Exception as e:
            self._logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            "backend": "faster_whisper",
            "model_size": self._model_size,
            "device": self._device,
            "compute_type": self._compute_type,
            "language": self._language,
            "vad_filter": self._vad_filter
        })
        return info
