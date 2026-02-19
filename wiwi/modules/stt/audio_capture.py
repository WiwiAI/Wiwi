"""Audio capture using sounddevice with resampling support."""

import logging
import queue
from typing import Callable, List, Optional, Dict, Any
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class AudioCapture:
    """
    Real-time audio capture from microphone with automatic resampling.

    Features:
    - Asynchronous audio capture using sounddevice
    - Automatic resampling from native device rate (44100/48000) to target rate (16000)
    - Configurable sample rate and chunk size
    - Device selection
    - Callback-based or queue-based processing

    Config options:
        device: Audio device index or name (None = default)
        sample_rate: Target sample rate in Hz (default: 16000 for Whisper)
        channels: Number of channels (default: 1 for mono)
        chunk_duration_ms: Chunk duration in milliseconds (default: 64)
    """

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 64
    ):
        """
        Initialize audio capture with resampling support.

        Args:
            device: Audio device index (None = default input device)
            sample_rate: Target sample rate in Hz (16000 for Whisper)
            channels: Number of audio channels (1 = mono)
            chunk_duration_ms: Duration of each audio chunk in ms
        """
        self._logger = logging.getLogger("wiwi.stt.audio_capture")

        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError(
                "sounddevice is required for audio capture. "
                "Install it with: pip install sounddevice"
            )

        self._device = device
        self._target_sample_rate = sample_rate  # What we want (16000 for Whisper)
        self._channels = channels
        self._chunk_duration_ms = chunk_duration_ms

        # Will be set in start() based on device capabilities
        self._native_sample_rate: int = sample_rate
        self._needs_resampling = False
        self._resample_ratio = 1.0

        # Chunk sizes
        self._target_chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self._native_chunk_size = self._target_chunk_size

        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._audio_queue: queue.Queue = queue.Queue()

        # Callback for real-time processing
        self._on_audio: Optional[Callable[[np.ndarray], None]] = None

    def set_callback(
        self,
        on_audio: Callable[[np.ndarray], None]
    ) -> None:
        """
        Set audio callback for real-time processing.

        Args:
            on_audio: Callback function that receives audio chunks (float32 numpy array)
        """
        self._on_audio = on_audio

    def _resample_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Resample audio from native sample rate to target sample rate.

        Args:
            audio: Audio data at native sample rate

        Returns:
            Audio data resampled to target sample rate
        """
        if not self._needs_resampling:
            return audio

        if SCIPY_AVAILABLE:
            # High-quality resampling using scipy
            num_samples = int(len(audio) * self._resample_ratio)
            resampled = scipy.signal.resample(audio, num_samples)
            return resampled.astype(np.float32)
        else:
            # Simple linear interpolation fallback
            num_samples = int(len(audio) * self._resample_ratio)
            indices = np.linspace(0, len(audio) - 1, num_samples)
            resampled = np.interp(indices, np.arange(len(audio)), audio)
            return resampled.astype(np.float32)

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags
    ) -> None:
        """Internal callback for sounddevice stream."""
        if status:
            self._logger.warning(f"Audio callback status: {status}")

        # Convert to float32 mono
        if self._channels == 1:
            audio = indata[:, 0].astype(np.float32)
        else:
            # Take first channel for mono processing
            audio = indata[:, 0].astype(np.float32)

        # Resample if needed (44100/48000 -> 16000)
        if self._needs_resampling:
            audio = self._resample_audio(audio)

        # Call user callback or put in queue
        if self._on_audio:
            try:
                self._on_audio(audio)
            except Exception as e:
                self._logger.error(f"Audio callback error: {e}")
        else:
            self._audio_queue.put(audio.copy())

    def _determine_sample_rate(self) -> int:
        """
        Determine the best native sample rate for the device.

        Returns:
            Native sample rate to use (may differ from target)
        """
        # Get device info
        try:
            if self._device is not None:
                dev_info = sd.query_devices(self._device)
            else:
                dev_info = sd.query_devices(kind='input')
        except Exception as e:
            self._logger.warning(f"Could not query device info: {e}")
            return self._target_sample_rate

        default_rate = int(dev_info.get('default_samplerate', 44100))

        # Try target rate first (16000), then common rates
        rates_to_try = [self._target_sample_rate, 16000, 44100, 48000, 22050, 8000]

        for rate in rates_to_try:
            try:
                sd.check_input_settings(
                    device=self._device,
                    samplerate=rate,
                    channels=self._channels
                )
                return rate
            except sd.PortAudioError:
                continue

        # Fallback to device default
        return default_rate

    def start(self) -> None:
        """Start audio capture from microphone with automatic resampling."""
        if self._running:
            self._logger.warning("Audio capture already running")
            return

        # Determine native sample rate
        self._native_sample_rate = self._determine_sample_rate()

        # Check if resampling is needed
        if self._native_sample_rate != self._target_sample_rate:
            self._needs_resampling = True
            self._resample_ratio = self._target_sample_rate / self._native_sample_rate
            # Adjust chunk size for native rate
            self._native_chunk_size = int(
                self._native_sample_rate * self._chunk_duration_ms / 1000
            )
            self._logger.info(
                f"Resampling enabled: {self._native_sample_rate}Hz -> "
                f"{self._target_sample_rate}Hz (ratio: {self._resample_ratio:.4f})"
            )
        else:
            self._needs_resampling = False
            self._native_chunk_size = self._target_chunk_size

        self._logger.info(
            f"Starting audio capture: device={self._device}, "
            f"native_rate={self._native_sample_rate}Hz, target_rate={self._target_sample_rate}Hz, "
            f"chunk={self._native_chunk_size} samples ({self._chunk_duration_ms}ms)"
        )

        try:
            self._stream = sd.InputStream(
                device=self._device,
                samplerate=self._native_sample_rate,
                channels=self._channels,
                dtype=np.float32,
                blocksize=self._native_chunk_size,
                callback=self._audio_callback
            )
            self._stream.start()
            self._running = True
            self._logger.info("Audio capture started successfully")
        except Exception as e:
            self._logger.error(f"Failed to start audio capture: {e}")
            raise RuntimeError(f"Failed to start audio capture: {e}")

    def stop(self) -> None:
        """Stop audio capture."""
        if not self._running:
            return

        self._running = False

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                self._logger.warning(f"Error stopping stream: {e}")
            self._stream = None

        # Clear the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        self._logger.info("Audio capture stopped")

    def get_audio(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get audio chunk from queue (when not using callback).

        Args:
            timeout: Timeout in seconds

        Returns:
            Audio chunk as float32 numpy array, or None if timeout
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        """Check if audio capture is currently running."""
        return self._running

    @property
    def sample_rate(self) -> int:
        """Get target sample rate (output after resampling)."""
        return self._target_sample_rate

    @property
    def native_sample_rate(self) -> int:
        """Get native device sample rate (before resampling)."""
        return self._native_sample_rate

    @property
    def chunk_size(self) -> int:
        """Get target chunk size in samples."""
        return self._target_chunk_size

    @property
    def chunk_duration_ms(self) -> int:
        """Get chunk duration in milliseconds."""
        return self._chunk_duration_ms

    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        """
        List available audio input devices.

        Returns:
            List of device info dictionaries
        """
        if not SOUNDDEVICE_AVAILABLE:
            return []

        devices = sd.query_devices()
        input_devices = []

        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'default_samplerate': dev['default_samplerate'],
                    'is_default': i == sd.default.device[0]
                })

        return input_devices

    @staticmethod
    def get_default_device() -> Optional[Dict[str, Any]]:
        """
        Get default input device info.

        Returns:
            Device info dictionary or None
        """
        if not SOUNDDEVICE_AVAILABLE:
            return None

        try:
            dev = sd.query_devices(kind='input')
            return {
                'name': dev['name'],
                'channels': dev['max_input_channels'],
                'default_samplerate': dev['default_samplerate']
            }
        except Exception:
            return None
