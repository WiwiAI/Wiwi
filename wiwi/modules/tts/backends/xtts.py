"""XTTS-v2 TTS backend with streaming support."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import numpy as np
try:
    import torch
    import torchaudio

    # PyTorch 2.6+ compatibility: patch torch.load to use weights_only=False
    # This is needed for TTS library which uses pickle-based model files
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    XTTS_AVAILABLE = True
except ImportError:
    XTTS_AVAILABLE = False

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

from wiwi.modules.tts.backends.base import BaseTTSBackend


class XTTSBackend(BaseTTSBackend):
    """
    TTS backend using Coqui XTTS-v2 with streaming support.

    XTTS-v2 is a multilingual TTS model supporting 17 languages including Russian.
    It supports voice cloning from a short reference audio (~6-15 seconds).

    Config options:
        model: Model identifier (default: "tts_models/multilingual/multi-dataset/xtts_v2")
        device: Device for inference (cuda:0, cuda:1, cpu)
        language: Default language code (default: "ru")
        reference_audio: Path to reference audio for voice cloning
        sample_rate: Output sample rate (24000 is XTTS-v2 native)

    Performance options:
        use_deepspeed: Enable DeepSpeed optimization (2-3x faster). Default: False
        use_compile: Enable torch.compile() optimization (PyTorch 2.0+). Default: False
        use_fp16: Enable FP16 inference (may affect quality). Default: False
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._logger = logging.getLogger("wiwi.tts.xtts")

        self._model_name = config.get(
            "model", "tts_models/multilingual/multi-dataset/xtts_v2"
        )
        self._language = config.get("language", "ru")
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

        # XTTS-v2 native sample rate
        self._sample_rate = 24000

        # Character limit for Russian (XTTS limitation)
        self._max_chars = 170  # Slightly under 182 for safety

        # Speaker embeddings (set after loading voice)
        self._gpt_cond_latent = None
        self._speaker_embedding = None

        # Note: CUDA stream async transfer removed - caused sync issues and white noise
        # Standard synchronous transfer is fast enough

        # Performance options
        # Note: fp16 doesn't work well with XTTS streaming (LayerNorm requires float32)
        self._use_fp16 = config.get("use_fp16", False)  # Half precision - disabled by default
        self._use_deepspeed = config.get("use_deepspeed", False)  # DeepSpeed acceleration
        self._use_compile = config.get("use_compile", False)  # torch.compile optimization
        self._speed = config.get("speed", 1.0)  # Speech speed multiplier
        # Use XTTS defaults for best quality/speed balance
        self._temperature = config.get("temperature", 0.75)
        self._length_penalty = config.get("length_penalty", 1.0)
        self._repetition_penalty = config.get("repetition_penalty", 10.0)  # XTTS default
        self._top_k = config.get("top_k", 50)
        self._top_p = config.get("top_p", 0.85)  # XTTS default
        # Streaming chunk size (lower = faster first chunk, but more overhead)
        self._stream_chunk_size = config.get("stream_chunk_size", 20)

    def load_model(self) -> None:
        """Load XTTS-v2 model."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for XTTS. Install with: pip install torch torchaudio"
            )

        if not XTTS_AVAILABLE:
            raise ImportError(
                "Coqui TTS is required for XTTS. Install with: pip install TTS"
            )

        # Check DeepSpeed availability
        if self._use_deepspeed and not DEEPSPEED_AVAILABLE:
            self._logger.warning(
                "DeepSpeed requested but not installed. Install with: pip install deepspeed"
            )
            self._use_deepspeed = False

        self._logger.info(
            f"Loading XTTS-v2 model on {self._device_str}"
            + (" with DeepSpeed" if self._use_deepspeed else "")
        )

        try:
            # Set CUDA device before loading
            if self._use_gpu and self._device_idx is not None:
                torch.cuda.set_device(self._device_idx)
                self._logger.info(f"Set CUDA device to {self._device_idx}")

            if self._use_deepspeed:
                # Load model directly with DeepSpeed support
                self._load_model_with_deepspeed()
            else:
                # Use TTS API for automatic download and loading
                self._load_model_standard()

            self._model.eval()
            self._logger.info("XTTS-v2 model loaded successfully")

            # Load voice from reference audio BEFORE converting to fp16
            # (get_conditioning_latents requires float32 internally)
            if self._reference_audio:
                self._load_voice_internal(self._reference_audio)

            # Apply fp16 for faster inference AFTER loading voice (only if not using DeepSpeed)
            if self._use_fp16 and self._use_gpu and not self._use_deepspeed:
                self._model = self._model.half()
                # Convert conditioning latents to fp16 too
                if self._gpt_cond_latent is not None:
                    self._gpt_cond_latent = self._gpt_cond_latent.half()
                if self._speaker_embedding is not None:
                    self._speaker_embedding = self._speaker_embedding.half()
                self._logger.info("Enabled fp16 mode for faster inference")

            # Apply torch.compile optimization (PyTorch 2.0+)
            if self._use_compile and hasattr(torch, 'compile'):
                self._logger.info("Applying torch.compile optimization...")
                self._model = torch.compile(self._model, mode="reduce-overhead")
                self._logger.info("torch.compile optimization applied")

        except Exception as e:
            self._logger.error(f"Failed to load XTTS-v2 model: {e}")
            raise RuntimeError(f"Failed to load XTTS-v2 model: {e}")

    def _load_model_standard(self) -> None:
        """Load model using standard TTS API."""
        from TTS.api import TTS

        # Initialize TTS with the model
        tts_api = TTS(self._model_name, gpu=self._use_gpu)

        # Get the underlying XTTS model for streaming
        self._model = tts_api.synthesizer.tts_model

        # Move to correct device
        if self._use_gpu:
            self._model = self._model.cuda()

    def _load_model_with_deepspeed(self) -> None:
        """Load model with DeepSpeed optimization for 2-3x faster inference."""
        import os
        from TTS.utils.manage import ModelManager
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        # Get model path using ModelManager
        manager = ModelManager()

        # Download model if needed - for XTTS it returns the directory path
        model_path, config_path, model_item = manager.download_model(self._model_name)

        # For XTTS, model_path is the checkpoint directory
        # config_path may be None, so we need to find it ourselves
        if model_path is None:
            # Fallback: get the model directory from ModelManager's output_path
            model_dir = os.path.join(
                manager.output_prefix,
                self._model_name.replace("/", "--")
            )
        else:
            model_dir = str(Path(model_path).parent) if os.path.isfile(model_path) else model_path

        # Find config.json in model directory
        if config_path is None or not os.path.exists(config_path):
            config_path = os.path.join(model_dir, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"XTTS config not found at {config_path}")

        self._logger.info(f"Loading XTTS config from: {config_path}")
        self._logger.info(f"Loading XTTS checkpoint from: {model_dir}")

        # Load config
        config = XttsConfig()
        config.load_json(config_path)

        # Initialize model from config
        self._model = Xtts.init_from_config(config)

        # Load checkpoint with DeepSpeed enabled
        self._model.load_checkpoint(
            config,
            checkpoint_dir=model_dir,
            use_deepspeed=True
        )

        # Move to GPU (required after DeepSpeed init)
        if self._use_gpu:
            self._model.cuda()

        self._logger.info("DeepSpeed optimization enabled (2-3x faster inference)")

    def warmup(self) -> None:
        """
        Warmup the model with a short synthesis to reduce first-call latency.

        This pre-compiles CUDA kernels and optimizes memory allocation.
        """
        if not self._model or self._gpt_cond_latent is None:
            return

        self._logger.info("Warming up TTS model...")

        try:
            # Short warmup text
            warmup_text = "Тест."

            with torch.inference_mode():
                # Run one streaming inference to warmup all kernels
                chunks = list(self._model.inference_stream(
                    warmup_text,
                    self._language,
                    self._gpt_cond_latent,
                    self._speaker_embedding,
                    temperature=self._temperature,
                    stream_chunk_size=self._stream_chunk_size,
                    enable_text_splitting=False,
                ))

            # Sync CUDA
            if self._use_gpu:
                torch.cuda.synchronize()

            self._logger.info(f"Warmup complete ({len(chunks)} chunks)")

        except Exception as e:
            self._logger.warning(f"Warmup failed (non-critical): {e}")

    def unload_model(self) -> None:
        """Unload model and free GPU memory."""
        if self._model:
            del self._model
            self._model = None
            self._gpt_cond_latent = None
            self._speaker_embedding = None

            if self._use_gpu:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            self._logger.info("XTTS-v2 model unloaded")

    def _load_voice_internal(self, audio_path: str) -> None:
        """
        Internal method to load voice (used during initialization, before fp16).

        Args:
            audio_path: Path to reference audio file
        """
        audio_path = Path(audio_path).expanduser()
        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        self._logger.info(f"Loading voice from: {audio_path}")

        # Get conditioning latents from reference audio
        self._gpt_cond_latent, self._speaker_embedding = (
            self._model.get_conditioning_latents(
                audio_path=[str(audio_path)]
            )
        )
        self._logger.info("Voice loaded successfully")

    def set_voice(self, audio_path: str) -> None:
        """
        Set the voice from a reference audio file.

        Args:
            audio_path: Path to reference audio file (WAV format, 6-15 seconds recommended)
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        # For runtime voice changes, we need to temporarily convert model to float32
        # if it's in fp16 mode, load voice, then convert back
        was_fp16 = False
        if self._use_fp16 and self._use_gpu:
            # Check if model is in fp16 mode
            try:
                sample_param = next(self._model.parameters())
                if sample_param.dtype == torch.float16:
                    was_fp16 = True
                    self._model = self._model.float()
            except StopIteration:
                pass

        try:
            self._load_voice_internal(audio_path)

            # Convert embeddings and model back to fp16 if needed
            if was_fp16:
                self._model = self._model.half()
                self._gpt_cond_latent = self._gpt_cond_latent.half()
                self._speaker_embedding = self._speaker_embedding.half()

        except Exception as e:
            # Restore fp16 mode on error
            if was_fp16:
                self._model = self._model.half()
            self._logger.error(f"Failed to load voice: {e}")
            raise RuntimeError(f"Failed to load voice from {audio_path}: {e}")

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
            Audio as float32 numpy array at 24kHz
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        if self._gpt_cond_latent is None or self._speaker_embedding is None:
            raise RuntimeError(
                "No voice loaded. Call set_voice() with a reference audio file."
            )

        lang = language or self._language

        self._logger.debug(f"Synthesizing (batch): '{text[:50]}...'")

        try:
            with torch.inference_mode():
                outputs = self._model.inference(
                    text,
                    lang,
                    self._gpt_cond_latent,
                    self._speaker_embedding,
                    temperature=self._temperature,
                    length_penalty=self._length_penalty,
                    repetition_penalty=self._repetition_penalty,
                    top_k=self._top_k,
                    top_p=self._top_p,
                    speed=self._speed,
                    **kwargs
                )

            # Get waveform
            wav = outputs["wav"]

            # Convert to numpy
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()

            # Ensure float32
            if wav.dtype != np.float32:
                wav = wav.astype(np.float32)

            # Trim leading noise (XTTS sometimes produces noise at start)
            abs_wav = np.abs(wav)
            threshold = 0.02
            above_threshold = np.where(abs_wav > threshold)[0]
            if len(above_threshold) > 0:
                start_idx = max(0, above_threshold[0] - 50)
                if start_idx > 0:
                    wav = wav[start_idx:]
                    self._logger.debug(f"Trimmed {start_idx} leading samples")

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

    def _split_text(self, text: str, max_chars: int) -> List[str]:
        """
        Split text into chunks that fit within character limit.

        Only splits on sentence boundaries (. ! ?). No splitting on commas.
        If a sentence is too long, splits by words only.
        """
        if len(text) <= max_chars:
            return [text]

        chunks = []

        # Split by sentences only (. ! ?)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                # Sentence too long - split by words (no comma splitting)
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                words = sentence.split()
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= max_chars:
                        current_chunk += " " + word if current_chunk else word
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = word
            elif len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def synthesize_stream(
        self,
        text: str,
        language: Optional[str] = None,
        **kwargs
    ) -> Iterator[np.ndarray]:
        """
        Synthesize speech with streaming output.

        Yields audio chunks as they are generated for low-latency playback.
        Automatically splits long text into parts.

        Args:
            text: Text to synthesize
            language: Language code (default: config language)

        Yields:
            Audio chunks as float32 numpy arrays at 24kHz
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        if self._gpt_cond_latent is None or self._speaker_embedding is None:
            raise RuntimeError(
                "No voice loaded. Call set_voice() with a reference audio file."
            )

        lang = language or self._language

        self._logger.debug(f"Synthesizing (stream): '{text[:50]}...'")

        try:
            chunk_count = 0
            total_samples = 0

            # Use inference_stream for streaming synthesis
            # Let XTTS handle text splitting internally if needed
            inference_kwargs = {
                "temperature": self._temperature,
                "length_penalty": self._length_penalty,
                "repetition_penalty": self._repetition_penalty,
                "top_k": self._top_k,
                "top_p": self._top_p,
                "speed": self._speed,
                "stream_chunk_size": self._stream_chunk_size,
                "enable_text_splitting": True,  # Let XTTS split if text too long
            }
            inference_kwargs.update(kwargs)

            chunks = self._model.inference_stream(
                text,
                lang,
                self._gpt_cond_latent,
                self._speaker_embedding,
                **inference_kwargs
            )

            for chunk in chunks:
                chunk_count += 1

                # Convert tensor to numpy (synchronous - fast enough)
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.cpu().numpy()

                # Squeeze and ensure float32
                if chunk.ndim > 1:
                    chunk = chunk.squeeze()

                if chunk.dtype != np.float32:
                    chunk = chunk.astype(np.float32, copy=False)

                # Trim leading noise from first chunk
                # XTTS sometimes produces noise at the start
                if chunk_count == 1:
                    abs_chunk = np.abs(chunk)
                    threshold = 0.02
                    above_threshold = np.where(abs_chunk > threshold)[0]
                    if len(above_threshold) > 0:
                        start_idx = max(0, above_threshold[0] - 50)
                        if start_idx > 0:
                            chunk = chunk[start_idx:]
                            self._logger.debug(f"Trimmed {start_idx} leading samples")

                    if len(chunk) == 0:
                        continue

                total_samples += len(chunk)
                yield chunk

            self._logger.debug(
                f"Streamed {chunk_count} chunks, "
                f"{total_samples / self._sample_rate:.2f}s total"
            )

        except Exception as e:
            self._logger.warning(f"Streaming synthesis failed: {e}, falling back to batch synthesis")
            # Fallback to batch synthesis
            try:
                audio = self.synthesize(text, language, **kwargs)
                yield audio
            except Exception as e2:
                self._logger.error(f"Batch synthesis also failed: {e2}")
                raise RuntimeError(f"TTS synthesis failed: {e2}")

    @property
    def supports_streaming(self) -> bool:
        """XTTS-v2 supports streaming."""
        return True

    @property
    def supports_voice_cloning(self) -> bool:
        """XTTS-v2 supports voice cloning."""
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            "backend": "xtts_v2",
            "model": self._model_name,
            "device": self._device_str,
            "language": self._language,
            "voice_loaded": self._gpt_cond_latent is not None,
            "reference_audio": self._reference_audio,
            # Performance options
            "deepspeed_enabled": self._use_deepspeed,
            "compile_enabled": self._use_compile,
            "fp16_enabled": self._use_fp16,
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
