"""Streaming audio output for TTS playback."""

import logging
import queue
import threading
from typing import Callable, Iterator, Optional
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class StreamingAudioPlayer:
    """
    Streaming audio player for low-latency TTS playback.

    Uses a persistent callback-based output stream for seamless audio
    across multiple sentences without gaps.

    Features:
    - Low-latency streaming playback
    - Persistent stream eliminates inter-sentence gaps
    - Queue-based buffering for smooth audio
    - Interrupt/stop capability
    - Volume control
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        device: Optional[int] = None,
        volume: float = 1.0,
        buffer_size: int = 1024,
    ):
        """
        Initialize streaming audio player.

        Args:
            sample_rate: Sample rate in Hz (default: 24000 for XTTS-v2)
            channels: Number of audio channels (default: 1 = mono)
            device: Output device index (None = default)
            volume: Playback volume 0.0-1.0 (default: 1.0)
            buffer_size: Audio buffer size in frames (default: 1024)
        """
        self._logger = logging.getLogger("wiwi.tts.audio_output")

        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError(
                "sounddevice is required. Install with: pip install sounddevice"
            )

        self._sample_rate = sample_rate
        self._channels = channels
        self._device = device
        self._volume = max(0.0, min(1.0, volume))
        self._buffer_size = buffer_size

        # Audio buffer queue for persistent stream
        self._audio_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()

        # Persistent stream state
        self._stream: Optional[sd.OutputStream] = None
        self._stream_lock = threading.Lock()
        self._is_playing = False
        self._stop_requested = False

        # Leftover samples between callbacks
        self._leftover = np.array([], dtype=np.float32)
        self._leftover_lock = threading.Lock()

        # Last sample value for smooth transitions when buffer is empty
        self._last_sample = 0.0

        # For tracking playback position
        self._samples_written = 0
        self._samples_played = 0

        # Auto-stop timer (stop stream after silence)
        self._last_audio_time = 0.0
        self._stream_timeout = 2.0  # Stop stream after 2s of no audio

    def _persistent_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status
    ) -> None:
        """
        Callback for persistent output stream.

        Uses a large internal buffer to prevent underruns.
        When buffer is empty, holds last sample value to avoid clicks.
        """
        if status:
            self._logger.warning(f"Audio status: {status}")

        samples_needed = frames
        samples_written = 0
        output = outdata[:, 0]

        # First, use any leftover samples
        with self._leftover_lock:
            if len(self._leftover) > 0:
                use_count = min(len(self._leftover), samples_needed)
                output[:use_count] = self._leftover[:use_count]
                self._leftover = self._leftover[use_count:]
                samples_written = use_count
                # Remember last sample for smooth transition
                self._last_sample = output[samples_written - 1]

        # Pull from queue
        while samples_written < samples_needed:
            try:
                chunk = self._audio_queue.get_nowait()

                if chunk is None:
                    # Stop marker - fade out smoothly
                    remaining = samples_needed - samples_written
                    if remaining > 0:
                        # Fade out from last sample to zero
                        fade = np.linspace(self._last_sample, 0, remaining, dtype=np.float32)
                        output[samples_written:] = fade
                    self._stop_requested = True
                    raise sd.CallbackStop()

                # Apply volume
                if self._volume < 1.0:
                    chunk = chunk * self._volume

                chunk = chunk.astype(np.float32)
                chunk_len = len(chunk)
                space_left = samples_needed - samples_written

                if chunk_len <= space_left:
                    output[samples_written:samples_written + chunk_len] = chunk
                    samples_written += chunk_len
                    self._last_sample = chunk[-1]
                else:
                    output[samples_written:] = chunk[:space_left]
                    with self._leftover_lock:
                        self._leftover = chunk[space_left:]
                    self._last_sample = chunk[space_left - 1]
                    samples_written = samples_needed

            except queue.Empty:
                # No audio available - hold last sample value (prevents clicks/noise)
                # This is much better than filling with zeros
                output[samples_written:] = self._last_sample
                break

        self._samples_played += samples_written

    def _ensure_stream_running(self) -> None:
        """Start persistent stream if not already running."""
        with self._stream_lock:
            if self._stream is None or not self._stream.active:
                self._stop_requested = False
                self._last_sample = 0.0  # Initialize last sample
                self._stream = sd.OutputStream(
                    samplerate=self._sample_rate,
                    channels=1,
                    dtype=np.float32,
                    device=self._device,
                    blocksize=2048,  # ~85ms blocks at 24kHz
                    latency=0.1,  # 100ms latency - balance between stability and responsiveness
                    callback=self._persistent_callback,
                )
                self._stream.start()
                self._logger.debug("Started persistent audio stream")

    def add_audio(self, audio: np.ndarray) -> None:
        """
        Add audio to the playback queue.

        Audio will play seamlessly with any previously queued audio.
        """
        if len(audio) == 0:
            return

        if audio.ndim > 1:
            audio = audio.squeeze()

        self._ensure_stream_running()
        self._audio_queue.put(audio.astype(np.float32))
        self._is_playing = True

    def add_audio_stream(self, chunk_iterator: Iterator[np.ndarray]) -> None:
        """
        Add audio chunks from iterator to playback queue.

        Chunks are added as they arrive, playing seamlessly.
        Prebuffers before starting to prevent underruns.
        """
        import collections
        import time

        # Collect initial buffer before starting playback
        initial_buffer = collections.deque()
        prebuffer_samples = 0
        # Prebuffer ~0.3s of audio before starting stream
        target_prebuffer = int(self._sample_rate * 0.3)

        chunk_iter = iter(chunk_iterator)

        # Phase 1: Prebuffer
        for chunk in chunk_iter:
            if self._stop_requested:
                return

            if chunk.ndim > 1:
                chunk = chunk.squeeze()

            chunk = chunk.astype(np.float32)
            initial_buffer.append(chunk)
            prebuffer_samples += len(chunk)

            if prebuffer_samples >= target_prebuffer:
                break

        if not initial_buffer:
            return

        # Start stream and add prebuffered chunks
        self._ensure_stream_running()

        for chunk in initial_buffer:
            self._audio_queue.put(chunk)

        self._is_playing = True

        # Phase 2: Continue adding remaining chunks
        for chunk in chunk_iter:
            if self._stop_requested:
                break

            if chunk.ndim > 1:
                chunk = chunk.squeeze()

            self._audio_queue.put(chunk.astype(np.float32))

    def play_stream(
        self,
        chunk_iterator: Iterator[np.ndarray],
        on_complete: Optional[Callable[[], None]] = None,
        prebuffer_chunks: int = 2,
        true_streaming: bool = True,
    ) -> None:
        """
        Play audio chunks from an iterator (streaming).

        Args:
            chunk_iterator: Iterator yielding audio chunks (float32 numpy arrays)
            on_complete: Optional callback when playback completes
            prebuffer_chunks: Number of chunks to buffer before starting playback
            true_streaming: If True, play chunks as they arrive (low latency).
                           If False, collect all chunks first (more reliable).
        """
        self._is_playing = True
        self._stop_requested = False
        self._samples_written = 0
        self._samples_played = 0

        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        try:
            if true_streaming:
                self._play_stream_realtime(chunk_iterator, prebuffer_chunks)
            else:
                self._play_stream_batch(chunk_iterator)

        except Exception as e:
            self._logger.error(f"Playback error: {e}")
            raise

        finally:
            self._is_playing = False

            if on_complete:
                on_complete()

    def _play_stream_batch(self, chunk_iterator: Iterator[np.ndarray]) -> None:
        """Collect all chunks then play (original behavior)."""
        all_audio = []

        self._logger.debug("Collecting audio chunks (batch mode)...")

        for chunk in chunk_iterator:
            if self._stop_requested:
                break

            if chunk.ndim > 1:
                chunk = chunk.squeeze()

            all_audio.append(chunk)

        if not all_audio or self._stop_requested:
            return

        full_audio = np.concatenate(all_audio)

        self._logger.debug(
            f"Playing {len(full_audio) / self._sample_rate:.2f}s of audio"
        )

        if self._volume < 1.0:
            full_audio = full_audio * self._volume

        sd.play(full_audio, self._sample_rate, device=self._device)
        sd.wait()

    def _play_stream_realtime(
        self,
        chunk_iterator: Iterator[np.ndarray],
        prebuffer_chunks: int = 2
    ) -> None:
        """
        True streaming playback using callback-based OutputStream.

        Uses a continuous audio stream with callback that pulls from buffer,
        eliminating gaps between chunks.
        """
        import collections

        # Ring buffer for seamless audio
        audio_buffer = collections.deque()
        buffer_lock = threading.Lock()
        producer_done = threading.Event()
        stream_finished = threading.Event()

        # Leftover samples from previous callback
        leftover = np.array([], dtype=np.float32)
        leftover_lock = threading.Lock()

        # Statistics
        played_samples = [0]  # Use list for mutability in callback

        def audio_callback(outdata: np.ndarray, frames: int, time_info, status):
            """Callback that continuously feeds audio to the stream."""
            nonlocal leftover

            if status:
                self._logger.warning(f"Audio status: {status}")

            samples_needed = frames
            samples_written = 0
            output = outdata[:, 0]  # Mono

            # First, use any leftover samples from previous call
            with leftover_lock:
                if len(leftover) > 0:
                    use_count = min(len(leftover), samples_needed)
                    output[:use_count] = leftover[:use_count]
                    leftover = leftover[use_count:]
                    samples_written = use_count

            # Then pull from buffer
            while samples_written < samples_needed:
                chunk = None
                with buffer_lock:
                    if audio_buffer:
                        chunk = audio_buffer.popleft()

                if chunk is None:
                    # No more data available
                    if producer_done.is_set():
                        # Producer finished - fill rest with silence and signal done
                        output[samples_written:] = 0
                        stream_finished.set()
                        raise sd.CallbackStop()
                    else:
                        # Waiting for more data - fill with silence (underrun)
                        output[samples_written:] = 0
                        break

                chunk_len = len(chunk)
                space_left = samples_needed - samples_written

                if chunk_len <= space_left:
                    # Entire chunk fits
                    output[samples_written:samples_written + chunk_len] = chunk
                    samples_written += chunk_len
                else:
                    # Partial chunk - save leftover
                    output[samples_written:] = chunk[:space_left]
                    with leftover_lock:
                        leftover = chunk[space_left:]
                    samples_written = samples_needed

            played_samples[0] += samples_written

        def producer_thread():
            """Thread that collects chunks from the iterator."""
            try:
                for chunk in chunk_iterator:
                    if self._stop_requested:
                        break

                    if chunk.ndim > 1:
                        chunk = chunk.squeeze()

                    # Apply volume
                    if self._volume < 1.0:
                        chunk = chunk * self._volume

                    with buffer_lock:
                        audio_buffer.append(chunk.astype(np.float32))

            finally:
                producer_done.set()

        # Start producer thread
        producer = threading.Thread(target=producer_thread, daemon=True)
        producer.start()

        # Wait for prebuffer (reduced for lower latency)
        prebuffer_samples = 0
        # ~0.15s prebuffer at 24kHz (was 0.5s)
        target_prebuffer = int(self._sample_rate * 0.15)

        self._logger.debug(f"Prebuffering ~{target_prebuffer / self._sample_rate:.2f}s...")

        while prebuffer_samples < target_prebuffer and not producer_done.is_set():
            with buffer_lock:
                prebuffer_samples = sum(len(c) for c in audio_buffer)

            if self._stop_requested:
                producer.join(timeout=1.0)
                return

            threading.Event().wait(0.005)

        self._logger.debug(
            f"Prebuffer ready ({prebuffer_samples / self._sample_rate:.2f}s), starting stream..."
        )

        # Create and start continuous output stream
        try:
            stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype=np.float32,
                device=self._device,
                blocksize=1024,  # ~42ms blocks at 24kHz
                callback=audio_callback,
            )

            with stream:
                # Wait for stream to finish or stop request
                while not stream_finished.is_set() and not self._stop_requested:
                    threading.Event().wait(0.01)

        except Exception as e:
            self._logger.error(f"Stream error: {e}")

        # Wait for producer to finish
        producer.join(timeout=2.0)

        self._logger.debug(
            f"Streaming complete: {played_samples[0] / self._sample_rate:.2f}s"
        )

    def play_stream_async(
        self,
        chunk_iterator: Iterator[np.ndarray],
        on_complete: Optional[Callable[[], None]] = None,
    ) -> threading.Thread:
        """
        Play audio chunks asynchronously in a background thread.

        Returns immediately while playback continues in background.

        Args:
            chunk_iterator: Iterator yielding audio chunks
            on_complete: Optional callback when playback completes

        Returns:
            Thread object (can be used to wait with .join())
        """
        thread = threading.Thread(
            target=self.play_stream,
            args=(chunk_iterator, on_complete),
            daemon=True,
        )
        thread.start()
        return thread

    def play(self, audio: np.ndarray, blocking: bool = True) -> None:
        """
        Play a complete audio array.

        Args:
            audio: Audio data as float32 numpy array
            blocking: If True, wait for playback to complete
        """
        if len(audio) == 0:
            return

        # Apply volume
        if self._volume < 1.0:
            audio = audio * self._volume

        self._logger.debug(
            f"Playing {len(audio) / self._sample_rate:.2f}s of audio "
            f"(blocking={blocking})"
        )

        try:
            self._is_playing = True
            sd.play(audio, self._sample_rate, device=self._device, blocking=blocking)
        except Exception as e:
            self._logger.error(f"Playback failed: {e}")
            raise
        finally:
            if blocking:
                self._is_playing = False

    def stop(self) -> None:
        """Stop current playback immediately."""
        self._stop_requested = True

        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        # Clear leftover
        with self._leftover_lock:
            self._leftover = np.array([], dtype=np.float32)

        # Stop persistent stream
        with self._stream_lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
                self._logger.debug("Stopped persistent stream")

        self._cleanup_stream()
        sd.stop()

        self._is_playing = False
        self._logger.debug("Playback stopped")

    def _cleanup_stream(self) -> None:
        """Clean up the output stream."""
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def wait(self) -> None:
        """Wait for current playback to finish."""
        sd.wait()
        self._is_playing = False

    def wait_until_done(self, timeout: float = 60.0) -> None:
        """
        Wait until all queued audio has been played.

        Blocks until the audio queue is empty and leftover buffer is consumed.

        Args:
            timeout: Maximum time to wait in seconds (default: 60s)
        """
        import time
        start = time.monotonic()

        while (time.monotonic() - start) < timeout:
            # Check if queue is empty and no leftover samples
            queue_empty = self._audio_queue.empty()
            with self._leftover_lock:
                leftover_empty = len(self._leftover) == 0

            if queue_empty and leftover_empty:
                # Give a small buffer for final samples to play out
                time.sleep(0.1)
                break

            time.sleep(0.05)

        self._is_playing = False

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing

    @property
    def volume(self) -> float:
        """Get current volume."""
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        """Set playback volume (0.0 to 1.0)."""
        self._volume = max(0.0, min(1.0, value))

    @property
    def sample_rate(self) -> int:
        """Get sample rate."""
        return self._sample_rate

    @staticmethod
    def list_output_devices():
        """
        List available audio output devices.

        Returns:
            List of dicts with device info
        """
        if not SOUNDDEVICE_AVAILABLE:
            return []

        devices = sd.query_devices()
        output_devices = []

        for i, dev in enumerate(devices):
            if dev["max_output_channels"] > 0:
                output_devices.append({
                    "index": i,
                    "name": dev["name"],
                    "channels": dev["max_output_channels"],
                    "default_samplerate": dev["default_samplerate"],
                    "is_default": i == sd.default.device[1],
                })

        return output_devices
