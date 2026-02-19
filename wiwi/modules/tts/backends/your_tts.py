"""YourTTS backend with voice cloning support."""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
import numpy as np

try:
    import torch

    # PyTorch 2.6+ compatibility: patch torch.load to use weights_only=False
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

from wiwi.modules.tts.backends.base import BaseTTSBackend


class YourTTSBackend(BaseTTSBackend):
    """
    TTS backend using Coqui YourTTS with voice cloning support.

    YourTTS is a multilingual TTS model supporting voice cloning.
    It supports English, French, and Portuguese out of the box.

    Note: YourTTS does NOT natively support Russian. For Russian,
    use XTTS-v2 instead.

    Config options:
        model: Model identifier (default: "tts_models/multilingual/multi-dataset/your_tts")
        device: Device for inference (cuda:0, cuda:1, cpu)
        language: Default language code (default: "en")
        reference_audio: Path to reference audio for voice cloning
        sample_rate: Output sample rate (16000 is YourTTS native)
    """

    # Supported languages for YourTTS
    SUPPORTED_LANGUAGES = {"en", "fr-fr", "pt-br"}

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._logger = logging.getLogger("wiwi.tts.your_tts")

        self._model_name = config.get(
            "model", "tts_models/multilingual/multi-dataset/your_tts"
        )
        self._language = config.get("language", "en")
        self._reference_audio = config.get("reference_audio")

        # Parse device
        device_str = config.get("device", "cuda:1")
        self._device_str = device_str
        if device_str.startswith("cuda"):
            if ":" in device_str:
                self._device_idx = int(device_str.split(":")[1])
            else:
                self._device_idx = 0
            self._use_gpu = True
        else:
            self._device_idx = None
            self._use_gpu = False

        # YourTTS native sample rate
        self._sample_rate = 16000

        # TTS API instance
        self._tts: Optional[TTS] = None

        # Reference audio path (for voice cloning)
        self._speaker_wav: Optional[str] = None

    def load_model(self) -> None:
        """Load YourTTS model."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for YourTTS. Install with: pip install torch torchaudio"
            )

        if not TTS_AVAILABLE:
            raise ImportError(
                "Coqui TTS is required for YourTTS. Install with: pip install TTS"
            )

        self._logger.info(
            f"Loading YourTTS model on {self._device_str}"
        )

        try:
            # Set CUDA device before loading
            if self._use_gpu and self._device_idx is not None:
                torch.cuda.set_device(self._device_idx)
                self._logger.info(f"Set CUDA device to {self._device_idx}")

            # Initialize TTS with the model
            self._tts = TTS(self._model_name, progress_bar=False)

            if self._use_gpu:
                self._tts = self._tts.to(f"cuda:{self._device_idx}")

            # Store model reference for is_loaded check
            self._model = self._tts

            self._logger.info("YourTTS model loaded successfully")

            # Load voice from reference audio
            if self._reference_audio:
                self.set_voice(self._reference_audio)

        except Exception as e:
            self._logger.error(f"Failed to load YourTTS model: {e}")
            raise RuntimeError(f"Failed to load YourTTS model: {e}")

    def unload_model(self) -> None:
        """Unload model and free GPU memory."""
        if self._tts:
            del self._tts
            self._tts = None
            self._model = None
            self._speaker_wav = None

            if self._use_gpu:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            self._logger.info("YourTTS model unloaded")

    def set_voice(self, audio_path: str) -> None:
        """
        Set the voice from a reference audio file.

        Args:
            audio_path: Path to reference audio file (WAV format, 7+ seconds recommended)
        """
        audio_path = Path(audio_path).expanduser()
        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        self._speaker_wav = str(audio_path)
        self._logger.info(f"Voice set from: {audio_path}")

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
            language: Language code (default: config language)

        Returns:
            Audio as float32 numpy array at 16kHz
        """
        if not self._tts:
            raise RuntimeError("Model not loaded")

        lang = language or self._language

        # Validate language
        if lang not in self.SUPPORTED_LANGUAGES:
            self._logger.warning(
                f"Language '{lang}' not supported by YourTTS. "
                f"Supported: {self.SUPPORTED_LANGUAGES}. Using 'en'."
            )
            lang = "en"

        self._logger.debug(f"Synthesizing (batch): '{text[:50]}...'")

        try:
            # Synthesize to numpy array
            if self._speaker_wav:
                wav = self._tts.tts(
                    text=text,
                    speaker_wav=self._speaker_wav,
                    language=lang,
                )
            else:
                wav = self._tts.tts(
                    text=text,
                    language=lang,
                )

            # Convert to numpy array
            wav = np.array(wav, dtype=np.float32)

            # Normalize if needed
            max_val = np.abs(wav).max()
            if max_val > 1.0:
                wav = wav / max_val

            self._logger.debug(
                f"Synthesized {len(wav) / self._sample_rate:.2f}s of audio"
            )

            return wav

        except Exception as e:
            self._logger.error(f"Synthesis failed: {e}")
            raise RuntimeError(f"TTS synthesis failed: {e}")

    def synthesize_stream(
        self,
        text: str,
        language: Optional[str] = None,
        **kwargs
    ) -> Iterator[np.ndarray]:
        """
        Synthesize speech with streaming output.

        Note: YourTTS doesn't support native streaming, so we synthesize
        the full audio and yield it as a single chunk.

        Args:
            text: Text to synthesize
            language: Language code (default: config language)

        Yields:
            Audio as a single float32 numpy array at 16kHz
        """
        # YourTTS doesn't support streaming, so we just yield the full audio
        wav = self.synthesize(text, language, **kwargs)
        yield wav

    @property
    def supports_streaming(self) -> bool:
        """YourTTS doesn't support native streaming."""
        return False

    @property
    def supports_voice_cloning(self) -> bool:
        """YourTTS supports voice cloning."""
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            "backend": "your_tts",
            "model": self._model_name,
            "device": self._device_str,
            "language": self._language,
            "voice_loaded": self._speaker_wav is not None,
            "reference_audio": self._speaker_wav,
            "supported_languages": list(self.SUPPORTED_LANGUAGES),
        })

        # Add VRAM info if on GPU
        if self._use_gpu and self._model:
            try:
                allocated = torch.cuda.memory_allocated(self._device_idx) / 1e9
                reserved = torch.cuda.memory_reserved(self._device_idx) / 1e9
                info["vram_allocated_gb"] = round(allocated, 2)
                info["vram_reserved_gb"] = round(reserved, 2)
            except Exception:
                pass

        return info
