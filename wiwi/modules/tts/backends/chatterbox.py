"""Chatterbox Multilingual TTS backend with pseudo-streaming support."""

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False

from wiwi.modules.tts.backends.base import BaseTTSBackend


class ChatterboxBackend(BaseTTSBackend):
    """
    TTS backend using Chatterbox Multilingual with streaming support.

    Chatterbox Multilingual is a 500M parameter TTS model supporting 23 languages
    including Russian, with zero-shot voice cloning capabilities.

    Config options:
        device: Device for inference (cuda:0, cuda:1, cpu)
        language: Default language code (ru, en, de, fr, zh, etc.)
        reference_audio: Path to reference audio for voice cloning
        sample_rate: Output sample rate (24000 is native)

    Generation parameters:
        chunk_size: Streaming chunk size in tokens (default: 50)
        exaggeration: Voice expressiveness 0.0-1.0 (default: 0.7)
        cfg_weight: Classifier-free guidance weight (default: 0.3)
        temperature: Sampling temperature (default: 1.0)

    Performance options:
        use_compile: Enable torch.compile optimization (default: False)

    Supported languages:
        ar (Arabic), da (Danish), de (German), el (Greek), en (English),
        es (Spanish), fi (Finnish), fr (French), he (Hebrew), hi (Hindi),
        it (Italian), ja (Japanese), ko (Korean), ms (Malay), nl (Dutch),
        no (Norwegian), pl (Polish), pt (Portuguese), ru (Russian),
        sv (Swedish), sw (Swahili), tr (Turkish), zh (Chinese)
    """

    SUPPORTED_LANGUAGES = {
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
        "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru",
        "sv", "sw", "tr", "zh"
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._logger = logging.getLogger("wiwi.tts.chatterbox")

        # Device configuration
        self._device_str = config.get("device", "cuda:0")
        self._device = self._parse_device(self._device_str)

        # Native sample rate
        self._sample_rate = 24000

        # Language
        self._language = config.get("language", "ru")
        if self._language not in self.SUPPORTED_LANGUAGES:
            self._logger.warning(
                f"Language '{self._language}' not in supported list, using anyway"
            )

        # Voice cloning
        self._reference_audio: Optional[str] = config.get("reference_audio")
        self._voice_path: Optional[str] = None

        # Generation parameters
        self._chunk_size = config.get("chunk_size", 50)
        self._exaggeration = config.get("exaggeration", 0.7)
        self._cfg_weight = config.get("cfg_weight", 0.3)
        self._temperature = config.get("temperature", 1.0)

        # Performance options
        self._use_compile = config.get("use_compile", False)

        # Metrics tracking
        self._last_ttfa: Optional[float] = None
        self._last_rtf: Optional[float] = None

    def _parse_device(self, device_str: str) -> str:
        """Parse device string for torch."""
        if device_str.startswith("cuda"):
            return device_str
        return "cpu"

    def load_model(self) -> None:
        """Load Chatterbox Multilingual TTS model."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for Chatterbox. Install with: pip install torch torchaudio"
            )

        if not CHATTERBOX_AVAILABLE:
            raise ImportError(
                "chatterbox-tts is required. Install with: pip install chatterbox-tts"
            )

        self._logger.info(
            f"Loading Chatterbox Multilingual model on {self._device_str}"
        )

        try:
            # Set CUDA device before loading
            if self._device.startswith("cuda"):
                device_idx = int(self._device.split(":")[1]) if ":" in self._device else 0
                torch.cuda.set_device(device_idx)
                self._logger.info(f"Set CUDA device to {device_idx}")

            # Load model
            self._model = ChatterboxMultilingualTTS.from_pretrained(device=self._device)

            # Update sample rate from model
            if hasattr(self._model, 'sr'):
                self._sample_rate = self._model.sr
                self._logger.debug(f"Model sample rate: {self._sample_rate}")

            # Apply torch.compile if requested
            if self._use_compile and hasattr(torch, 'compile'):
                self._logger.info("Applying torch.compile optimization...")
                self._model = torch.compile(self._model, mode="reduce-overhead")
                self._logger.info("torch.compile optimization applied")

            self._logger.info(
                f"Chatterbox Multilingual model loaded successfully (language: {self._language})"
            )

            # Load reference voice if configured
            if self._reference_audio:
                self.set_voice(self._reference_audio)

        except Exception as e:
            self._logger.error(f"Failed to load Chatterbox model: {e}")
            raise RuntimeError(f"Failed to load Chatterbox model: {e}")

    def unload_model(self) -> None:
        """Unload model and free GPU memory."""
        if self._model:
            del self._model
            self._model = None
            self._voice_path = None

            if self._device.startswith("cuda"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            self._logger.info("Chatterbox model unloaded")

    def warmup(self) -> None:
        """
        Warmup model with short synthesis.

        Pre-compiles CUDA kernels and optimizes memory allocation.
        """
        if not self._model:
            return

        self._logger.info("Warming up Chatterbox model...")

        try:
            # Short warmup text in configured language
            warmup_texts = {
                "ru": "Тест.",
                "en": "Test.",
                "de": "Test.",
                "fr": "Test.",
                "zh": "测试。",
                "ja": "テスト。",
            }
            warmup_text = warmup_texts.get(self._language, "Test.")

            # Run one synthesis to warmup
            chunks = list(self.synthesize_stream(warmup_text))

            # Sync CUDA
            if self._device.startswith("cuda"):
                torch.cuda.synchronize()

            self._logger.info(f"Warmup complete ({len(chunks)} chunks)")

        except Exception as e:
            self._logger.warning(f"Warmup failed (non-critical): {e}")

    def set_voice(self, audio_path: str) -> None:
        """
        Set voice from reference audio file.

        Args:
            audio_path: Path to reference audio (WAV format recommended)
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        audio_path = Path(audio_path).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        self._logger.info(f"Setting voice from: {audio_path}")
        self._voice_path = str(audio_path)
        self._logger.info("Voice reference set successfully")

    def _process_chunk(self, chunk: Any) -> np.ndarray:
        """
        Process audio chunk to float32 numpy array.

        Args:
            chunk: Audio chunk (tensor or numpy array)

        Returns:
            float32 numpy array normalized to [-1, 1]
        """
        # Convert tensor to numpy
        if hasattr(chunk, 'cpu'):  # torch.Tensor
            chunk_np = chunk.squeeze().cpu().numpy()
        else:
            chunk_np = np.asarray(chunk).squeeze()

        # Ensure float32
        if chunk_np.dtype != np.float32:
            chunk_np = chunk_np.astype(np.float32)

        # Normalize if needed
        max_val = np.abs(chunk_np).max()
        if max_val > 1.0:
            chunk_np = chunk_np / max_val

        return chunk_np

    def synthesize_stream(
        self,
        text: str,
        language: Optional[str] = None,
        **kwargs
    ) -> Iterator[np.ndarray]:
        """
        Synthesize speech with streaming output.

        Uses a background thread to run async streaming and yields chunks
        as they become available for true streaming behavior.

        Args:
            text: Text to synthesize
            language: Language code override (e.g., "ru", "en")
            **kwargs: Override generation parameters

        Yields:
            Audio chunks as float32 numpy arrays at 24kHz
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        lang = language or self._language
        self._logger.debug(f"Synthesizing (stream): '{text[:50]}...' [lang={lang}]")

        # Build generation kwargs
        chunk_size = kwargs.get("chunk_size", self._chunk_size)

        # Queue for passing chunks from async to sync
        chunk_queue: queue.Queue = queue.Queue()
        finished = threading.Event()
        error_holder: list = []

        # Metrics
        start_time = time.perf_counter()
        first_chunk_received = threading.Event()

        async def _async_producer():
            """Async producer that reads from stream_generate."""
            try:
                gen_kwargs = {
                    "language_id": lang,
                    "chunk_size": chunk_size,
                }

                # Add voice reference if set
                if self._voice_path:
                    gen_kwargs["audio_prompt_path"] = self._voice_path

                async for audio_chunk in self._model.stream_generate(text, **gen_kwargs):
                    if not first_chunk_received.is_set():
                        first_chunk_received.set()
                    chunk_queue.put(audio_chunk)

            except Exception as e:
                error_holder.append(e)
            finally:
                finished.set()

        def _run_async():
            """Run async producer in new event loop."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_async_producer())
            finally:
                loop.close()

        # Start async producer in background thread
        producer_thread = threading.Thread(target=_run_async, daemon=True)
        producer_thread.start()

        # Yield chunks as they arrive
        chunk_count = 0
        total_samples = 0

        try:
            while not finished.is_set() or not chunk_queue.empty():
                try:
                    chunk = chunk_queue.get(timeout=0.1)
                    chunk_count += 1

                    # Record TTFA on first chunk
                    if chunk_count == 1:
                        self._last_ttfa = time.perf_counter() - start_time
                        self._logger.info(f"TTFA: {self._last_ttfa * 1000:.1f}ms")

                    # Process and yield chunk
                    chunk_np = self._process_chunk(chunk)
                    total_samples += len(chunk_np)
                    yield chunk_np

                except queue.Empty:
                    continue

            # Check for errors
            if error_holder:
                raise error_holder[0]

            # Final metrics
            total_time = time.perf_counter() - start_time
            audio_duration = total_samples / self._sample_rate if total_samples > 0 else 0
            self._last_rtf = total_time / audio_duration if audio_duration > 0 else 0

            self._logger.info(
                f"Streamed {chunk_count} chunks, {audio_duration:.2f}s audio, "
                f"RTF: {self._last_rtf:.3f}"
            )

        except Exception as e:
            self._logger.error(f"Streaming synthesis failed: {e}")
            raise RuntimeError(f"TTS synthesis failed: {e}")

        finally:
            # Ensure thread is cleaned up
            producer_thread.join(timeout=1.0)

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Synthesize speech (batch mode).

        Collects all streaming chunks into a single array.

        Args:
            text: Text to synthesize
            language: Language code override

        Returns:
            Audio as float32 numpy array at 24kHz
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        self._logger.debug(f"Synthesizing (batch): '{text[:50]}...'")

        chunks = list(self.synthesize_stream(text, language, **kwargs))

        if not chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(chunks)

    @property
    def supports_streaming(self) -> bool:
        """Chatterbox supports streaming."""
        return True

    @property
    def supports_voice_cloning(self) -> bool:
        """Chatterbox supports voice cloning."""
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information with performance metrics."""
        info = super().get_model_info()
        info.update({
            "backend": "chatterbox_multilingual",
            "device": self._device_str,
            "language": self._language,
            "voice_loaded": self._voice_path is not None,
            "reference_audio": self._voice_path,
            # Generation parameters
            "chunk_size": self._chunk_size,
            "exaggeration": self._exaggeration,
            "cfg_weight": self._cfg_weight,
            "temperature": self._temperature,
            # Performance options
            "compile_enabled": self._use_compile,
            # Last metrics
            "last_ttfa_ms": round(self._last_ttfa * 1000, 1) if self._last_ttfa else None,
            "last_rtf": round(self._last_rtf, 3) if self._last_rtf else None,
        })

        # Add VRAM info if on GPU
        if self._device.startswith("cuda") and self._model:
            try:
                device_idx = int(self._device.split(":")[1]) if ":" in self._device else 0
                allocated = torch.cuda.memory_allocated(device_idx) / 1e9
                reserved = torch.cuda.memory_reserved(device_idx) / 1e9
                info["vram_allocated_gb"] = round(allocated, 2)
                info["vram_reserved_gb"] = round(reserved, 2)
            except Exception:
                pass

        return info
