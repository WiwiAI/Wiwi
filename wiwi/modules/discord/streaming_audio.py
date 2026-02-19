"""
Streaming audio source for Discord voice with ring buffer.

Provides continuous audio streaming by accepting chunks asynchronously
while Discord's AudioPlayer thread reads 20ms frames synchronously.
"""

import collections
import logging
import threading
from typing import Optional

import discord
import numpy as np

logger = logging.getLogger(__name__)

# Discord audio constants
DISCORD_SAMPLE_RATE = 48000
DISCORD_CHANNELS = 2
DISCORD_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
DISCORD_FRAME_MS = 20
# 48000 * 0.020 * 2 * 2 = 3840 bytes per frame
DISCORD_FRAME_SIZE = int(
    DISCORD_SAMPLE_RATE * DISCORD_FRAME_MS / 1000 * DISCORD_CHANNELS * DISCORD_SAMPLE_WIDTH
)


class StreamingAudioSource(discord.AudioSource):
    """
    Custom AudioSource that reads from a thread-safe ring buffer.

    Allows continuous streaming by accepting audio chunks asynchronously
    while the Discord player thread reads 20ms frames synchronously.

    Buffer strategy:
    - Target latency: 200-300ms (10-15 frames worth)
    - Initial prebuffer: 150ms before starting playback
    - Underrun handling: Output silence, log warning
    """

    def __init__(self, target_latency_ms: int = 250, prebuffer_ms: int = 150):
        """
        Initialize streaming audio source.

        Args:
            target_latency_ms: Target buffer latency in milliseconds
            prebuffer_ms: Minimum buffer level before starting playback
        """
        self._buffer: collections.deque = collections.deque()
        self._buffer_lock = threading.Lock()
        self._buffer_bytes = 0

        # Leftover bytes from partial frame reads
        self._leftover = b""

        # Configuration
        self._target_latency_ms = target_latency_ms
        self._prebuffer_ms = prebuffer_ms
        self._prebuffer_bytes = int(
            DISCORD_SAMPLE_RATE
            * prebuffer_ms
            / 1000
            * DISCORD_CHANNELS
            * DISCORD_SAMPLE_WIDTH
        )

        # State
        self._finished = False
        self._started = False
        self._underrun_count = 0
        self._consecutive_underruns = 0  # For auto-finish detection
        # Auto-finish after ~2 seconds of silence (100 frames at 20ms each)
        # This gives TTS time to generate next sentence
        self._max_consecutive_underruns = 100

        # For smooth fade during underruns
        self._last_samples: Optional[np.ndarray] = None

    def add_audio(self, audio_bytes: bytes) -> None:
        """
        Add audio data to the buffer (thread-safe).

        Audio should already be in Discord format:
        48kHz, stereo, 16-bit signed PCM.

        Args:
            audio_bytes: PCM audio data in Discord format
        """
        if not audio_bytes:
            return

        with self._buffer_lock:
            self._buffer.append(audio_bytes)
            self._buffer_bytes += len(audio_bytes)
            # Reset consecutive underrun counter when new data arrives
            self._consecutive_underruns = 0

    def finish(self) -> None:
        """Signal that no more audio will be added."""
        self._finished = True
        logger.debug("StreamingAudioSource: finish() called")

    def read(self) -> bytes:
        """
        Read exactly one 20ms frame (3840 bytes).

        Called by Discord's AudioPlayer thread at ~50 FPS.

        Returns:
            3840 bytes of PCM audio, or b'' to signal end
        """
        frame = self._read_frame()

        if len(frame) == DISCORD_FRAME_SIZE:
            self._started = True
            return frame
        elif len(frame) > 0:
            # Partial frame at end - pad with silence
            silence = bytes(DISCORD_FRAME_SIZE - len(frame))
            return frame + silence
        else:
            # No data and finished
            return b""

    def _read_frame(self) -> bytes:
        """Internal method to read up to FRAME_SIZE bytes."""
        result = bytearray()
        needed = DISCORD_FRAME_SIZE

        # First, use leftover bytes
        if self._leftover:
            if len(self._leftover) >= needed:
                result = bytearray(self._leftover[:needed])
                self._leftover = self._leftover[needed:]
                return bytes(result)
            else:
                result.extend(self._leftover)
                needed -= len(self._leftover)
                self._leftover = b""

        # Pull from buffer
        while needed > 0:
            chunk = None
            with self._buffer_lock:
                if self._buffer:
                    chunk = self._buffer.popleft()
                    self._buffer_bytes -= len(chunk)

            if chunk is None:
                # Buffer empty
                if self._finished:
                    # End of stream - return what we have
                    break

                # Track consecutive underruns for auto-finish
                self._consecutive_underruns += 1
                self._underrun_count += 1

                if self._underrun_count == 1 or self._underrun_count % 100 == 0:
                    logger.warning(
                        f"Audio underrun #{self._underrun_count}, "
                        f"buffer={self.buffer_level_ms:.0f}ms"
                    )

                # Auto-finish after 5 seconds of silence (250 frames)
                # This allows TTS time to generate next sentence
                if self._consecutive_underruns >= 250:
                    logger.info(
                        f"Auto-finishing stream after {self._consecutive_underruns * 20}ms silence"
                    )
                    self._finished = True
                    return b""  # Signal end to Discord

                # Generate silence for remaining bytes
                silence = bytes(needed)
                result.extend(silence)
                break

            # Got data - reset consecutive underrun counter
            self._consecutive_underruns = 0

            if len(chunk) <= needed:
                result.extend(chunk)
                needed -= len(chunk)
            else:
                # Chunk larger than needed - save leftover
                result.extend(chunk[:needed])
                self._leftover = chunk[needed:]
                needed = 0

        return bytes(result)

    def is_opus(self) -> bool:
        """Audio is PCM, not Opus."""
        return False

    def cleanup(self) -> None:
        """Clean up resources."""
        with self._buffer_lock:
            self._buffer.clear()
            self._buffer_bytes = 0
        self._leftover = b""
        self._finished = True
        logger.debug(
            f"StreamingAudioSource cleanup: underruns={self._underrun_count}"
        )

    @property
    def buffer_level_ms(self) -> float:
        """Current buffer level in milliseconds."""
        with self._buffer_lock:
            bytes_buffered = self._buffer_bytes + len(self._leftover)
        return (
            bytes_buffered
            / (DISCORD_SAMPLE_RATE * DISCORD_CHANNELS * DISCORD_SAMPLE_WIDTH)
            * 1000
        )

    @property
    def is_ready(self) -> bool:
        """Check if enough data is buffered to start playback."""
        with self._buffer_lock:
            bytes_buffered = self._buffer_bytes + len(self._leftover)
        return bytes_buffered >= self._prebuffer_bytes

    @property
    def is_finished(self) -> bool:
        """Check if stream has ended."""
        return self._finished

    @property
    def underrun_count(self) -> int:
        """Number of underruns that occurred."""
        return self._underrun_count


def convert_to_discord_format(
    audio_bytes: bytes, source_rate: int, source_channels: int = 1
) -> Optional[bytes]:
    """
    Convert audio to Discord format (48kHz stereo int16).

    Args:
        audio_bytes: Input PCM audio (int16)
        source_rate: Source sample rate (e.g., 24000)
        source_channels: Source channel count (1=mono, 2=stereo)

    Returns:
        Converted audio bytes or None on error
    """
    try:
        samples = np.frombuffer(audio_bytes, dtype=np.int16)

        # Handle stereo input
        if source_channels == 2:
            # Convert to mono first (average channels)
            samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)

        # Resample if needed
        if source_rate != DISCORD_SAMPLE_RATE:
            ratio = DISCORD_SAMPLE_RATE / source_rate
            new_length = int(len(samples) * ratio)
            indices = np.linspace(0, len(samples) - 1, new_length)
            samples = np.interp(indices, np.arange(len(samples)), samples).astype(
                np.int16
            )

        # Convert mono to stereo (duplicate channel)
        stereo = np.column_stack((samples, samples)).flatten()
        return stereo.astype(np.int16).tobytes()

    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return None
