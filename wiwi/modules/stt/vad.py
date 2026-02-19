"""Voice Activity Detection using Silero VAD."""

import logging
from typing import Callable, List, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SileroVAD:
    """
    Voice Activity Detection using Silero VAD model.

    Features:
    - Real-time speech detection (~1ms per 30ms chunk)
    - Configurable speech/silence thresholds
    - Speech start/end callbacks
    - Audio buffering for complete speech segments

    Config options:
        threshold: Speech probability threshold (0.0-1.0). Default: 0.5
        min_speech_duration_ms: Minimum speech duration to trigger. Default: 250
        min_silence_duration_ms: Silence duration to end speech. Default: 500
        speech_pad_ms: Padding around speech segments. Default: 100
        sample_rate: Audio sample rate (8000 or 16000). Default: 16000
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
        sample_rate: int = 16000
    ):
        """
        Initialize Silero VAD.

        Args:
            threshold: Speech probability threshold (0.0-1.0)
            min_speech_duration_ms: Minimum speech duration to consider valid
            min_silence_duration_ms: Silence duration to end speech segment
            speech_pad_ms: Padding around speech (for pre-roll buffer)
            sample_rate: Audio sample rate (must be 8000 or 16000)
        """
        self._logger = logging.getLogger("wiwi.stt.vad")

        if sample_rate not in (8000, 16000):
            raise ValueError(f"Sample rate must be 8000 or 16000, got {sample_rate}")

        self._threshold = threshold
        self._min_speech_ms = min_speech_duration_ms
        self._min_silence_ms = min_silence_duration_ms
        self._speech_pad_ms = speech_pad_ms
        self._sample_rate = sample_rate

        self._model = None
        self._is_speaking = False
        self._speech_start_time: Optional[float] = None
        self._last_speech_time: Optional[float] = None

        # Silero VAD requires EXACTLY 512 samples at 16kHz or 256 at 8kHz
        self._vad_chunk_size = 512 if sample_rate == 16000 else 256
        self._chunk_accumulator: List[np.ndarray] = []
        self._accumulated_samples = 0

        # Callbacks
        self._on_speech_start: Optional[Callable[[], None]] = None
        self._on_speech_end: Optional[Callable[[np.ndarray], None]] = None

        # Audio buffer for current speech segment
        self._speech_buffer: List[np.ndarray] = []

        # Pre-roll buffer (to not cut off speech beginning)
        self._pre_roll_samples = int(sample_rate * speech_pad_ms / 1000)
        self._pre_roll_buffer: List[np.ndarray] = []

    def load_model(self) -> None:
        """Load Silero VAD model from torch hub."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for Silero VAD. "
                "Install it with: pip install torch"
            )

        self._logger.info("Loading Silero VAD model...")

        try:
            self._model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self._logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            self._logger.error(f"Failed to load Silero VAD: {e}")
            raise RuntimeError(f"Failed to load Silero VAD model: {e}")

    def unload_model(self) -> None:
        """Unload VAD model and free resources."""
        if self._model:
            del self._model
            self._model = None
            self._logger.info("Silero VAD model unloaded")

    def set_callbacks(
        self,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[np.ndarray], None]] = None
    ) -> None:
        """
        Set speech event callbacks.

        Args:
            on_speech_start: Called when speech starts (no arguments)
            on_speech_end: Called when speech ends (receives full audio as np.ndarray)
        """
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end

    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        current_time: float
    ) -> bool:
        """
        Process audio chunk and detect speech.

        This method should be called for each audio chunk from the microphone.
        It maintains internal state to track speech segments.

        Args:
            audio_chunk: Audio data as float32 numpy array (mono)
            current_time: Current timestamp in seconds

        Returns:
            True if speech is detected in this chunk, False otherwise

        Raises:
            RuntimeError: If model is not loaded
        """
        if self._model is None:
            raise RuntimeError("VAD model not loaded. Call load_model() first.")

        # Accumulate chunks
        self._chunk_accumulator.append(audio_chunk.copy())
        self._accumulated_samples += len(audio_chunk)

        if self._accumulated_samples < self._vad_chunk_size:
            # Not enough samples yet, return last known state
            return self._is_speaking

        # Combine accumulated chunks
        all_samples = np.concatenate(self._chunk_accumulator)
        self._chunk_accumulator = []
        self._accumulated_samples = 0

        # Process in exact chunks of _vad_chunk_size (512 for 16kHz)
        # Keep remainder for next call
        num_full_chunks = len(all_samples) // self._vad_chunk_size
        remainder = len(all_samples) % self._vad_chunk_size

        if remainder > 0:
            self._chunk_accumulator.append(all_samples[-remainder:])
            self._accumulated_samples = remainder
            all_samples = all_samples[:-remainder]

        if num_full_chunks == 0:
            return self._is_speaking

        # Process each VAD-sized chunk and average probability
        speech_probs = []
        for i in range(num_full_chunks):
            start = i * self._vad_chunk_size
            end = start + self._vad_chunk_size
            chunk = all_samples[start:end]

            audio_tensor = torch.from_numpy(chunk).float()

            # Normalize if needed
            if audio_tensor.abs().max() > 1.0:
                audio_tensor = audio_tensor / 32768.0

            try:
                prob = self._model(audio_tensor, self._sample_rate).item()
                speech_probs.append(prob)
            except Exception as e:
                self._logger.warning(f"VAD inference error: {e}")
                continue

        if not speech_probs:
            return self._is_speaking

        # Use max probability (if any chunk has speech, consider it speech)
        speech_prob = max(speech_probs)
        is_speech = speech_prob >= self._threshold

        # Use all processed samples for buffering
        audio_chunk = all_samples

        # Update pre-roll buffer (always keep last N samples)
        self._pre_roll_buffer.append(audio_chunk.copy())
        max_pre_roll_chunks = max(1, self._pre_roll_samples // len(audio_chunk))
        if len(self._pre_roll_buffer) > max_pre_roll_chunks:
            self._pre_roll_buffer.pop(0)

        if is_speech:
            self._last_speech_time = current_time

            if not self._is_speaking:
                # Speech just started
                self._is_speaking = True
                self._speech_start_time = current_time
                self._logger.debug(f"Speech started at {current_time:.2f}s")

                # Add pre-roll buffer to speech buffer
                self._speech_buffer = list(self._pre_roll_buffer)

                if self._on_speech_start:
                    try:
                        self._on_speech_start()
                    except Exception as e:
                        self._logger.error(f"on_speech_start callback error: {e}")
            else:
                # Continue collecting speech
                self._speech_buffer.append(audio_chunk.copy())
        else:
            if self._is_speaking and self._last_speech_time is not None:
                # Check if silence is long enough to end speech
                silence_duration_ms = (current_time - self._last_speech_time) * 1000

                if silence_duration_ms >= self._min_silence_ms:
                    # Speech ended
                    speech_duration_ms = (self._last_speech_time - self._speech_start_time) * 1000

                    if speech_duration_ms >= self._min_speech_ms:
                        self._logger.debug(
                            f"Speech ended: {speech_duration_ms:.0f}ms "
                            f"({len(self._speech_buffer)} chunks)"
                        )

                        # Combine buffered audio and call callback
                        if self._speech_buffer and self._on_speech_end:
                            try:
                                full_audio = np.concatenate(self._speech_buffer)
                                self._on_speech_end(full_audio)
                            except Exception as e:
                                self._logger.error(f"on_speech_end callback error: {e}")
                    else:
                        self._logger.debug(
                            f"Speech too short ({speech_duration_ms:.0f}ms), ignoring"
                        )

                    # Reset state
                    self._is_speaking = False
                    self._speech_buffer = []
                    self._speech_start_time = None
                else:
                    # Still in speech (silence too short), keep buffering
                    self._speech_buffer.append(audio_chunk.copy())

        return is_speech

    def reset(self) -> None:
        """Reset VAD state (clear buffers and flags)."""
        self._is_speaking = False
        self._speech_start_time = None
        self._last_speech_time = None
        self._speech_buffer = []
        self._pre_roll_buffer = []
        self._chunk_accumulator = []
        self._accumulated_samples = 0

        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass  # Some versions don't have reset_states

        self._logger.debug("VAD state reset")

    @property
    def is_speaking(self) -> bool:
        """Check if speech is currently being detected."""
        return self._is_speaking

    @property
    def is_loaded(self) -> bool:
        """Check if VAD model is loaded."""
        return self._model is not None

    @property
    def threshold(self) -> float:
        """Get current speech threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set speech threshold (0.0-1.0)."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self._threshold = value
