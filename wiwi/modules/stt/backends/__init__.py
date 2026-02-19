"""STT backends package."""

from wiwi.modules.stt.backends.base import BaseSTTBackend
from wiwi.modules.stt.backends.faster_whisper import FasterWhisperBackend

__all__ = ["BaseSTTBackend", "FasterWhisperBackend"]
