"""TTS backends."""

from wiwi.modules.tts.backends.base import BaseTTSBackend
from wiwi.modules.tts.backends.xtts import XTTSBackend

# TensorRT backend (optional - requires tensorrt)
try:
    from wiwi.modules.tts.backends.xtts_trt_backend import XttsTrtBackend
    _TRT_AVAILABLE = True
except ImportError:
    XttsTrtBackend = None
    _TRT_AVAILABLE = False

# Chatterbox Multilingual backend (optional - requires chatterbox-tts)
try:
    from wiwi.modules.tts.backends.chatterbox import ChatterboxBackend
    _CHATTERBOX_AVAILABLE = True
except ImportError:
    ChatterboxBackend = None
    _CHATTERBOX_AVAILABLE = False

__all__ = ["BaseTTSBackend", "XTTSBackend", "XttsTrtBackend", "ChatterboxBackend"]
