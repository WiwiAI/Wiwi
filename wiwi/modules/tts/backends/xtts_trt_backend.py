"""
TensorRT backend for XTTS-v2.

HYBRID APPROACH:
- PyTorch for first GPT step (prefix) - includes conditioning
- TensorRT (gpt_decode.trt) for subsequent steps - KV cache only
- TensorRT (hifigan_decoder.trt) for vocoder

This hybrid approach:
1. Uses PyTorch for ONE prefix step (prepends conditioning to embeddings)
2. Uses TRT for ALL decode steps (95%+ of inference time)
3. After synthesis starts, PyTorch model can be unloaded

Memory footprint:
- During synthesis: gpt_decode.trt (~1.5GB) + hifigan_decoder.trt (~37MB) + KV cache
- PyTorch model is used only for conditioning and prefix step

Required files:
    1. gpt_decode.trt - GPT decode model (single token + KV cache)
    2. hifigan_decoder.trt - HiFi-GAN vocoder

Usage in config:
    tts:
      backend: xtts_trt
      trt_gpt_decode_path: models/trt/gpt_decode.trt
      trt_decoder_path: models/trt/hifigan_decoder.trt
      xtts_checkpoint_dir: ~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2
      speaker_ref_path: voices/speaker.wav
      device: cuda:0
      language: ru
"""

import logging
import os
from typing import Any, Dict, Iterator, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from wiwi.modules.tts.backends.base import BaseTTSBackend


# ============================================================================
# Sampling Utilities
# ============================================================================

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("inf"),
) -> torch.Tensor:
    """Filter logits using top-k and/or nucleus (top-p) filtering."""
    top_k = min(top_k, logits.size(-1))

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    """Sample next token from logits distribution."""
    if temperature != 1.0:
        logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ============================================================================
# TensorRT Backend
# ============================================================================

class XttsTrtBackend(BaseTTSBackend):
    """XTTS-v2 TensorRT Backend for high-performance TTS."""

    # XTTS model constants
    NUM_LAYERS = 30
    NUM_HEADS = 16
    HIDDEN_SIZE = 1024
    HEAD_DIM = 64
    COND_LEN = 32
    SAMPLE_RATE = 24000

    # XTTS GPT token configuration (from loaded model)
    # These are set in load_model() from the actual model config
    # Default values here match XTTS v2:
    NUM_AUDIO_TOKENS = 1026  # mel vocab: 0-1023 codes + 1024 start + 1025 stop
    MEL_START_TOKEN = 1024   # start_audio_token
    MEL_STOP_TOKEN = 1025    # stop_audio_token
    START_TEXT_TOKEN = 261   # from tokenizer
    STOP_TEXT_TOKEN = 0      # from tokenizer

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._logger = logging.getLogger("wiwi.tts.xtts_trt")
        self._logger.info("Initializing XTTS-v2 TensorRT Hybrid Backend")

        # Paths from config - HYBRID: only decode TRT engine needed
        self._trt_gpt_decode_path = config.get("trt_gpt_decode_path") or config.get("trt_gpt_path")
        self._trt_decoder_path = config.get("trt_decoder_path")
        self._xtts_checkpoint_dir = config.get("xtts_checkpoint_dir")
        self._speaker_ref_path = config.get("speaker_ref_path")

        # Device
        device_str = config.get("device", "cuda:0")
        self._device = torch.device(device_str)
        self._device_idx = int(device_str.split(":")[1]) if ":" in device_str else 0

        # Language
        self._language = config.get("language", "ru")

        # Generation parameters
        self._temperature = config.get("temperature", 0.75)
        self._top_k = config.get("top_k", 50)
        self._top_p = config.get("top_p", 0.85)
        self._repetition_penalty = config.get("repetition_penalty", 10.0)
        self._max_mel_tokens = config.get("max_mel_tokens", 500)
        self._stream_chunk_tokens = config.get("stream_chunk_tokens", 20)

        self._sample_rate = self.SAMPLE_RATE

        # DeepSpeed configuration
        self._use_deepspeed = config.get("use_deepspeed", False)
        self._deepspeed_dtype = config.get("deepspeed_dtype", "fp16")  # fp16, bf16, fp32

        # Runtime state - HYBRID: separate engines
        self._gpt_decode_engine = None
        self._gpt_decode_context = None
        self._decoder_engine = None
        self._decoder_context = None
        self._pytorch_model = None
        self._tokenizer = None
        self._ds_engine = None  # DeepSpeed inference engine

        # Cached speaker embeddings
        self._gpt_cond_latent: Optional[torch.Tensor] = None
        self._speaker_embedding: Optional[torch.Tensor] = None

    def load_model(self) -> None:
        """Load TensorRT engines and PyTorch model for hybrid inference.

        Supports multiple modes:
        - Full TRT: GPT decode + HiFi-GAN decoder as TRT engines
        - Hybrid: PyTorch GPT + TRT HiFi-GAN decoder
        - Pure PyTorch: All PyTorch (fallback if no TRT engines)
        """
        self._logger.info("Loading XTTS-v2 backend...")

        if not self._xtts_checkpoint_dir:
            raise ValueError("xtts_checkpoint_dir not specified")

        torch.cuda.set_device(self._device_idx)

        # Load TensorRT engines if paths provided
        if self._trt_gpt_decode_path and os.path.exists(self._trt_gpt_decode_path):
            import tensorrt as trt
            self._logger.info(f"Loading GPT decode engine: {self._trt_gpt_decode_path}")
            self._gpt_decode_engine, self._gpt_decode_context = self._load_trt_engine(self._trt_gpt_decode_path)
        else:
            self._logger.info("No TRT GPT decode engine, using PyTorch")

        if self._trt_decoder_path and os.path.exists(self._trt_decoder_path):
            import tensorrt as trt
            self._logger.info(f"Loading HiFi-GAN decoder engine: {self._trt_decoder_path}")
            self._decoder_engine, self._decoder_context = self._load_trt_engine(self._trt_decoder_path)
        else:
            self._logger.info("No TRT HiFi-GAN engine, using PyTorch")

        # Load PyTorch model for conditioning AND first step (prefix)
        self._logger.info(f"Loading XTTS checkpoint: {self._xtts_checkpoint_dir}")
        self._load_pytorch_model()

        # Load default speaker
        if self._speaker_ref_path:
            self.set_voice(self._speaker_ref_path)

        # Determine mode
        if self._gpt_decode_engine and self._decoder_engine:
            self._model = "XTTS-TRT-Full"
        elif self._decoder_engine:
            self._model = "XTTS-TRT-Hybrid"
        else:
            self._model = "XTTS-PyTorch"

        self._logger.info(f"Backend loaded successfully (mode: {self._model})")

    def _load_trt_engine(self, engine_path: str):
        """Load TensorRT engine from file."""
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        context = engine.create_execution_context()
        return engine, context

    def _load_pytorch_model(self) -> None:
        """Load PyTorch XTTS model for conditioning computation.

        If use_deepspeed is enabled, applies DeepSpeed inference optimization
        to the GPT transformer for faster autoregressive generation.
        """
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        try:
            from TTS.config.shared_configs import BaseDatasetConfig
            from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
            torch.serialization.add_safe_globals([
                XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs
            ])
        except Exception:
            pass

        import os
        config_path = os.path.join(self._xtts_checkpoint_dir, "config.json")

        config = XttsConfig()
        config.load_json(config_path)

        self._pytorch_model = Xtts.init_from_config(config)
        self._pytorch_model.load_checkpoint(
            config,
            checkpoint_dir=self._xtts_checkpoint_dir,
            use_deepspeed=False  # We'll apply DeepSpeed ourselves for more control
        )
        self._pytorch_model.to(self._device)
        self._pytorch_model.eval()

        self._tokenizer = self._pytorch_model.tokenizer

        # Apply DeepSpeed inference optimization if enabled
        if self._use_deepspeed:
            self._apply_deepspeed_optimization()

    def _apply_deepspeed_optimization(self) -> None:
        """Apply DeepSpeed inference optimization to GPT model.

        DeepSpeed-Inference replaces transformer layers with optimized kernels
        that provide significant speedup for autoregressive generation.
        """
        try:
            import deepspeed
        except ImportError:
            self._logger.warning("DeepSpeed not installed, skipping optimization")
            return

        self._logger.info(f"Applying DeepSpeed inference optimization (dtype={self._deepspeed_dtype})")

        # Determine dtype
        if self._deepspeed_dtype == "fp16":
            ds_dtype = torch.float16
        elif self._deepspeed_dtype == "bf16":
            ds_dtype = torch.bfloat16
        else:
            ds_dtype = torch.float32

        # Apply DeepSpeed to GPT transformer
        # The GPT model in XTTS is: self._pytorch_model.gpt.gpt (HuggingFace GPT2Model)
        gpt_model = self._pytorch_model.gpt.gpt

        try:
            # DeepSpeed inference config
            ds_config = {
                "tensor_parallel": {"tp_size": 1},  # Single GPU
                "dtype": ds_dtype,
                "replace_with_kernel_inject": True,  # Use optimized kernels
                "enable_cuda_graph": False,  # Disable for variable sequence lengths
            }

            # Initialize DeepSpeed inference
            self._ds_engine = deepspeed.init_inference(
                model=gpt_model,
                config=ds_config,
            )

            # Replace the GPT model with DeepSpeed-optimized version
            self._pytorch_model.gpt.gpt = self._ds_engine.module

            self._logger.info("DeepSpeed inference optimization applied successfully")
            self._logger.info(f"  - Kernel injection: enabled")
            self._logger.info(f"  - Dtype: {self._deepspeed_dtype}")

        except Exception as e:
            self._logger.warning(f"Failed to apply DeepSpeed optimization: {e}")
            self._logger.warning("Falling back to standard PyTorch inference")
            self._ds_engine = None

    def unload_model(self) -> None:
        """Unload models and free resources."""
        self._gpt_decode_context = None
        self._gpt_decode_engine = None
        self._decoder_context = None
        self._decoder_engine = None

        # Clean up DeepSpeed engine
        if self._ds_engine is not None:
            del self._ds_engine
            self._ds_engine = None

        if self._pytorch_model:
            del self._pytorch_model
            self._pytorch_model = None

        self._gpt_cond_latent = None
        self._speaker_embedding = None

        torch.cuda.empty_cache()
        self._model = None
        self._logger.info("XTTS-v2 TensorRT Hybrid backend unloaded")

    @property
    def supports_voice_cloning(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    def set_voice(self, audio_path: str) -> None:
        """Set speaker voice from reference audio."""
        if not self._pytorch_model:
            raise RuntimeError("PyTorch model not loaded")

        self._logger.info(f"Loading voice from: {audio_path}")

        gpt_cond_latent, speaker_embedding = (
            self._pytorch_model.get_conditioning_latents(audio_path=[audio_path])
        )

        self._gpt_cond_latent = gpt_cond_latent.to(self._device)
        self._speaker_embedding = speaker_embedding.to(self._device)

        self._logger.info("Voice loaded successfully")
        # NOTE: PyTorch model is kept loaded for prefix step in hybrid mode

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """Synthesize speech from text."""
        chunks = list(self.synthesize_stream(text, language, **kwargs))
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    def synthesize_stream(
        self,
        text: str,
        language: Optional[str] = None,
        **kwargs
    ) -> Iterator[np.ndarray]:
        """Synthesize speech with streaming output using hybrid PyTorch+TRT approach.

        XTTS uses a two-phase approach:
        1. Generate mel codes (tokens 0-1023) autoregressively
        2. Convert codes to latents via separate forward pass

        HYBRID IMPLEMENTATION:
        - Phase 1: PyTorch prefix + TRT decode loop → mel codes
        - Phase 2: PyTorch forward pass (codes → latents) + TRT HiFi-GAN
        """
        if not self.is_loaded:
            raise RuntimeError("XTTS-TRT Hybrid backend not loaded")

        if self._gpt_cond_latent is None or self._speaker_embedding is None:
            raise RuntimeError("Speaker voice not set. Call set_voice() first.")

        if self._pytorch_model is None:
            raise RuntimeError("PyTorch model not loaded")

        lang = language or self._language
        self._logger.debug(f"Synthesizing: '{text[:50]}...'")

        # Tokenize text
        text_tokens = self._tokenize_text(text, lang)

        # ===== PHASE 1: Generate mel codes =====
        mel_codes = self._generate_mel_codes_hybrid(text_tokens)
        self._logger.debug(f"Generated {len(mel_codes)} mel codes")

        if not mel_codes:
            return

        # ===== PHASE 2: Convert codes to latents =====
        mel_codes_tensor = torch.tensor([mel_codes], dtype=torch.long, device=self._device)
        gpt_latents = self._codes_to_latents(text_tokens, mel_codes_tensor)

        # ===== PHASE 3: Decode to audio =====
        # For streaming, we can chunk the latents
        chunk_size = self._stream_chunk_tokens
        total_frames = gpt_latents.shape[1]

        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            latent_chunk = gpt_latents[:, start:end, :]

            audio_chunk = self._decode_latents_tensor(latent_chunk)
            yield audio_chunk

    def _generate_mel_codes_hybrid(
        self,
        text_tokens: torch.Tensor,
    ) -> List[int]:
        """Generate mel codes using hybrid PyTorch prefix + TRT decode.

        Phase 1 of XTTS: Autoregressive generation of mel tokens (0-1023).

        Returns:
            List of mel code integers
        """
        # PyTorch prefix pass - get initial logits and KV cache
        logits, kv_cache = self._pytorch_prefix_for_codes(text_tokens, self._gpt_cond_latent)

        mel_codes = []

        # Sample first token
        next_token = sample_next_token(
            logits[:, -1, :],
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
        )
        token_id = next_token.item()

        if token_id < self.MEL_START_TOKEN:  # Valid mel token (0-1023)
            mel_codes.append(token_id)

        # Decode loop (TRT or PyTorch)
        use_trt_decode = self._gpt_decode_engine is not None

        for step in range(self._max_mel_tokens - 1):
            if token_id >= self.MEL_START_TOKEN:  # Start/stop token
                break

            if use_trt_decode:
                logits, kv_cache = self._trt_decode_for_codes(next_token, kv_cache)
            else:
                logits, kv_cache = self._pytorch_decode_for_codes(next_token, kv_cache)

            # Apply repetition penalty
            if self._repetition_penalty != 1.0 and mel_codes:
                for prev_token in set(mel_codes[-50:]):
                    if prev_token < logits.shape[-1]:
                        logits[0, 0, prev_token] /= self._repetition_penalty

            next_token = sample_next_token(
                logits[:, -1, :],
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p,
            )
            token_id = next_token.item()

            if token_id >= self.MEL_START_TOKEN:
                self._logger.debug(f"Stop/start token {token_id} at step {step + 1}")
                break

            mel_codes.append(token_id)

        return mel_codes

    def _pytorch_prefix_for_codes(
        self,
        text_tokens: torch.Tensor,
        gpt_cond_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """PyTorch prefix pass for mel code generation.

        IMPORTANT: Must match XTTS GPT.compute_embeddings() + GPT2InferenceModel.forward():
        1. Pad text with start_text_token and stop_text_token
        2. Add text_pos_embedding to text embeddings
        3. Prepend conditioning latents
        4. Add start_audio_token embedding with mel_pos_embedding at position 0

        The GPT2InferenceModel stores prefix_emb = [cond_latents, text_emb_with_pos]
        and then processes start_audio_token with mel_pos_embedding at position 0.

        Returns logits and KV cache.
        """
        gpt = self._pytorch_model.gpt
        cond_len = gpt_cond_latent.shape[1]

        # 1. Pad text tokens with start/stop (like GPT.compute_embeddings)
        text_tokens_padded = F.pad(text_tokens, (0, 1), value=gpt.stop_text_token)
        text_tokens_padded = F.pad(text_tokens_padded, (1, 0), value=gpt.start_text_token)

        # 2. Get text embeddings WITH position embeddings
        text_emb = gpt.text_embedding(text_tokens_padded)
        text_emb = text_emb + gpt.text_pos_embedding(text_tokens_padded)

        # 3. Build prefix: [cond_latents, text_emb]
        prefix_emb = torch.cat([gpt_cond_latent, text_emb], dim=1)
        prefix_len = prefix_emb.shape[1]

        # 4. Add start_audio_token embedding with mel_pos_embedding
        # In GPT2InferenceModel, on first call:
        #   - prefix_emb is stored
        #   - input_ids has start_audio_token at position prefix_len
        #   - emb = mel_embedding(start_audio_token) + mel_pos_embedding(position=0)
        # Because mel_pos_embedding is relative to mel sequence start
        start_audio_token = torch.tensor([[gpt.start_audio_token]], device=self._device)
        start_audio_emb = gpt.mel_embedding(start_audio_token)
        # mel_pos_embedding for first mel token (position 0)
        start_audio_emb = start_audio_emb + gpt.mel_pos_embedding(start_audio_token)

        # Full embedding: [prefix, start_audio]
        emb = torch.cat([prefix_emb, start_audio_emb], dim=1)

        # Attention mask for full sequence
        total_len = emb.shape[1]
        attention_mask = torch.ones(1, total_len, dtype=torch.long, device=self._device)

        with torch.no_grad():
            gpt_out = gpt.gpt(
                inputs_embeds=emb,
                past_key_values=None,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )

        # Get logits from last position (after start_audio_token)
        hidden = gpt_out.last_hidden_state[:, -1:, :]  # Just last position
        logits = gpt.mel_head(hidden)

        # Convert KV cache to dict
        kv_cache = {}
        for i, (key, value) in enumerate(gpt_out.past_key_values):
            kv_cache[f"past_key.{i}"] = key.float().contiguous()
            kv_cache[f"past_value.{i}"] = value.float().contiguous()

        # Store prefix_len for subsequent decode steps
        # prefix_len = cond_len + text_len_with_tokens
        kv_cache["prefix_len"] = prefix_len

        return logits, kv_cache

    def _trt_decode_for_codes(
        self,
        input_ids: torch.Tensor,
        kv_cache: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Decode step for mel code generation.

        TEMPORARILY using PyTorch until TRT engine is rebuilt with mel_embedding.
        """
        # Use PyTorch for now (TRT engine needs rebuild with mel_embedding)
        return self._pytorch_decode_for_codes(input_ids, kv_cache)

    def _pytorch_decode_for_codes(
        self,
        input_ids: torch.Tensor,
        kv_cache: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """PyTorch decode step for mel code generation.

        IMPORTANT: Must match GPT2InferenceModel.forward() decode path:
        - Uses mel_embedding for mel tokens
        - Adds mel_pos_embedding with position relative to mel sequence start

        In GPT2InferenceModel, for decode step:
          position = attention_mask.shape[1] - prefix_len - 1
        where prefix_len = cond_len + text_len (stored prefix)

        But since we're tracking mel position ourselves, we compute:
          mel_position = past_len - prefix_len
        where past_len includes prefix + previous mel tokens.
        """
        gpt = self._pytorch_model.gpt

        # Ensure input_ids is [batch, 1]
        if input_ids.dim() == 0:
            input_ids = input_ids.unsqueeze(0).unsqueeze(0)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)

        # Get MEL embedding
        mel_emb = gpt.mel_embedding(input_ids)  # [batch, 1, 1024]

        # Reconstruct past_key_values from dict
        past_key_values = []
        for i in range(self.NUM_LAYERS):
            key = kv_cache[f"past_key.{i}"]
            value = kv_cache[f"past_value.{i}"]
            past_key_values.append((key, value))
        past_key_values = tuple(past_key_values)

        # Attention mask and position calculation
        past_len = kv_cache["past_key.0"].shape[2]
        attention_mask = torch.ones(1, past_len + 1, dtype=torch.long, device=self._device)

        # Calculate mel position for position embedding
        # prefix_len = cond_len(32) + text_with_start_stop(text_len+2)
        # first mel token (start_audio) was at position 0 in mel_pos_embedding
        # so current mel position = past_len - prefix_len
        # But we stored prefix in KV cache, so:
        #   past_len after prefix step = prefix_len + 1 (start_audio)
        #   on subsequent steps, mel_position = past_len - prefix_len
        # Since we don't track prefix_len explicitly, we use the pattern from GPT2InferenceModel:
        #   mel_pos = attention_mask.shape[1] - prefix_len - 1
        # Simpler: track mel step count. For now, use get_fixed_embedding with relative position.

        # GPT2InferenceModel uses: pos = attention_mask.shape[1] - (prefix_len + 1)
        # We need to compute prefix_len. From KV cache shape after first step:
        # past_len = prefix_len + num_generated_mel_tokens
        # On step N, we're generating mel token N, so position in mel_pos_embedding = N

        # Simpler approach: mel_pos_embedding is a LearnedPositionEmbeddings
        # We need position index. For step N (0-indexed), it's N.
        # Currently past_len = prefix_len + N (after N mel tokens generated including start_audio)
        # So mel_position = N = past_len - prefix_len

        # We track prefix_len in kv_cache or compute from cond_latent + text
        # For now, approximate: cond_len=32, assume text_len~10 -> prefix_len~44
        # Better: store prefix_len in first call

        # Use mel_pos_embedding.get_fixed_embedding(position, device)
        # The position should be: current_mel_step
        # After prefix step, past_len = prefix_len + 1, so mel_step = 1
        # Generally: mel_step = past_len - prefix_len

        # Estimate prefix_len: typically cond(32) + text_with_tokens(5-20+2) ≈ 39-54
        # We'll use the attention_mask length pattern like GPT2InferenceModel

        # From GPT2InferenceModel: pos = attention_mask.shape[1] - (prefix_len + 1)
        # After first call (with start_audio), past_len = prefix_len + 1
        # For subsequent calls, we want position = past_len - prefix_len

        # Let's store prefix_len in the kv_cache dict
        if "prefix_len" in kv_cache:
            prefix_len = kv_cache["prefix_len"]
        else:
            # Fallback: assume standard layout
            # This shouldn't happen if _pytorch_prefix_for_codes stores it
            prefix_len = 39  # approximate

        mel_position = past_len - prefix_len
        mel_pos_emb = gpt.mel_pos_embedding.get_fixed_embedding(mel_position, self._device)
        mel_emb = mel_emb + mel_pos_emb

        with torch.no_grad():
            gpt_out = gpt.gpt(
                inputs_embeds=mel_emb,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )

        # Get logits
        hidden = gpt_out.last_hidden_state
        logits = gpt.mel_head(hidden)

        # Convert KV cache to dict
        new_kv_cache = {}
        for i, (key, value) in enumerate(gpt_out.past_key_values):
            new_kv_cache[f"past_key.{i}"] = key.float().contiguous()
            new_kv_cache[f"past_value.{i}"] = value.float().contiguous()

        # Preserve prefix_len for next decode step
        new_kv_cache["prefix_len"] = prefix_len

        return logits, new_kv_cache

    def _codes_to_latents(
        self,
        text_tokens: torch.Tensor,
        mel_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Convert mel codes to latents using PyTorch XTTS GPT.

        Phase 2 of XTTS: Forward pass with [cond, text_emb, mel_emb] to get latents.

        IMPORTANT: This must match GPT.forward() with return_latent=True exactly:
        1. Pad text with start/stop tokens
        2. Pad mel codes with start/stop tokens
        3. Add position embeddings to both text and mel
        4. Concat: [cond_latents, text_emb, mel_emb]
        5. Forward through transformer
        6. Skip conditioning in output
        7. Apply final_norm
        8. Return mel part with sub=-5 trim

        Args:
            text_tokens: [batch, text_len] - text token IDs (without start/stop)
            mel_codes: [batch, mel_len] - generated mel codes (0-1023, without start/stop)

        Returns:
            gpt_latents: [batch, mel_len, 1024] - latents for HiFi-GAN
        """
        gpt = self._pytorch_model.gpt

        # 1. Pad text tokens with start/stop
        text_inputs = F.pad(text_tokens, (0, 1), value=gpt.stop_text_token)
        text_inputs = F.pad(text_inputs, (1, 0), value=gpt.start_text_token)

        # 2. Pad mel codes with start/stop
        audio_codes = F.pad(mel_codes, (1, 0), value=gpt.start_audio_token)
        audio_codes = F.pad(audio_codes, (0, 1), value=gpt.stop_audio_token)

        # 3. Get embeddings WITH position embeddings (critical!)
        text_emb = gpt.text_embedding(text_inputs) + gpt.text_pos_embedding(text_inputs)
        mel_emb = gpt.mel_embedding(audio_codes) + gpt.mel_pos_embedding(audio_codes)

        # 4. Concat: [cond_latents, text_emb, mel_emb]
        # Note: gpt_cond_latent is already [B, 32, 1024] - no transpose needed
        emb = torch.cat([self._gpt_cond_latent, text_emb, mel_emb], dim=1)

        # 5. Forward through transformer
        with torch.no_grad():
            gpt_out = gpt.gpt(
                inputs_embeds=emb,
                return_dict=True,
            )

        # 6. Skip conditioning tokens in output
        cond_len = self._gpt_cond_latent.shape[1]
        hidden = gpt_out.last_hidden_state[:, cond_len:]

        # 7. Apply final_norm
        hidden = gpt.final_norm(hidden)

        # 8. Extract mel part (skip text) with sub=-5 trim
        # hidden now has shape [B, text_len + mel_len, 1024]
        text_len = text_emb.shape[1]
        mel_len = mel_emb.shape[1]

        # Get mel part (last mel_len positions)
        mel_latents = hidden[:, -mel_len:, :]

        # Apply sub=-5 trim like original GPT.forward() with return_latent=True
        sub = -5
        mel_latents = mel_latents[:, :sub, :]

        return mel_latents

    def _decode_latents_tensor(self, latents: torch.Tensor) -> np.ndarray:
        """Decode latent tensor to audio using HiFi-GAN TensorRT engine."""
        batch = latents.shape[0]
        time_steps = latents.shape[1]

        latents = latents.float().contiguous()
        speaker_emb = self._speaker_embedding.float().contiguous()

        # Set shapes
        self._decoder_context.set_input_shape("latent_codes", tuple(latents.shape))
        self._decoder_context.set_input_shape("speaker_embedding", tuple(speaker_emb.shape))

        # Get output shape
        inferred_shape = self._decoder_context.get_tensor_shape("waveform")
        output_samples = inferred_shape[1] if inferred_shape[1] > 0 else time_steps * 1100

        output_waveform = torch.empty(batch, output_samples, dtype=torch.float32, device=self._device)

        # Set addresses
        self._decoder_context.set_tensor_address("latent_codes", latents.data_ptr())
        self._decoder_context.set_tensor_address("speaker_embedding", speaker_emb.data_ptr())
        self._decoder_context.set_tensor_address("waveform", output_waveform.data_ptr())

        # Execute
        stream = torch.cuda.current_stream().cuda_stream
        success = self._decoder_context.execute_async_v3(stream)
        torch.cuda.synchronize()

        if not success:
            raise RuntimeError("TensorRT decoder execution failed")

        waveform = output_waveform.squeeze().cpu().numpy()
        max_val = np.abs(waveform).max()
        if max_val > 1.0:
            waveform = waveform / max_val

        return waveform.astype(np.float32)

    def _tokenize_text(self, text: str, language: str) -> torch.Tensor:
        """Tokenize text for XTTS."""
        tokens = self._tokenizer.encode(text, lang=language)
        return torch.tensor([tokens], dtype=torch.int64, device=self._device)

    def _pytorch_prefix_forward(
        self,
        input_ids: torch.Tensor,
        gpt_cond_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """First GPT forward pass using PyTorch (prefix step).

        This step:
        - Prepends conditioning latents to text embeddings
        - Runs full text sequence through GPT
        - Returns KV cache for subsequent TRT decode steps

        Args:
            input_ids: [batch, seq_len] - text token IDs
            gpt_cond_latent: [batch, 32, 1024] - speaker conditioning

        Returns:
            logits: [batch, seq_len, vocab_size]
            latent_codes: [batch, seq_len, hidden_size]
            kv_cache: dict of past_key.{i} and past_value.{i} tensors
        """
        batch = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        cond_len = gpt_cond_latent.shape[1]

        gpt = self._pytorch_model.gpt

        # Get text embeddings
        text_emb = gpt.text_embedding(input_ids)  # [batch, seq_len, 1024]

        # Prepend conditioning
        emb = torch.cat([gpt_cond_latent, text_emb], dim=1)  # [batch, cond+seq, 1024]

        # Attention mask for full sequence
        total_len = cond_len + seq_len
        attention_mask = torch.ones(batch, total_len, dtype=torch.long, device=self._device)

        # Run GPT transformer (no past KV cache)
        with torch.no_grad():
            gpt_out = gpt.gpt(
                inputs_embeds=emb,
                past_key_values=None,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )

        hidden_states = gpt_out.last_hidden_state
        present_key_values = gpt_out.past_key_values

        # Skip conditioning tokens in output (keep only text positions)
        hidden_states = hidden_states[:, cond_len:, :]

        # Compute logits and latent codes
        logits = gpt.mel_head(hidden_states)
        latent_codes = gpt.final_norm(hidden_states)

        # Convert present_key_values to dict format for TRT compatibility
        kv_cache = {}
        for i, (key, value) in enumerate(present_key_values):
            kv_cache[f"past_key.{i}"] = key.float().contiguous()
            kv_cache[f"past_value.{i}"] = value.float().contiguous()

        return logits, latent_codes, kv_cache

    def _trt_decode_forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Single step GPT forward pass using TensorRT (decode step).

        This step:
        - Processes single token
        - Uses existing KV cache (no conditioning prepended)
        - Returns updated KV cache

        The gpt_decode.trt model expects:
        - input_ids: [batch, 1] - single token
        - attention_mask: [batch, past_len + 1]
        - past_key.{i}, past_value.{i}: [batch, heads, past_len, head_dim]

        Args:
            input_ids: [batch, 1] - single token tensor or [batch] scalar
            kv_cache: dict of past_key.{i} and past_value.{i} tensors

        Returns:
            logits: [batch, 1, vocab_size]
            latent_codes: [batch, 1, hidden_size]
            kv_cache: updated dict with new_len = past_len + 1
        """
        # Ensure input_ids is [batch, 1]
        if input_ids.dim() == 0:
            input_ids = input_ids.unsqueeze(0).unsqueeze(0)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)

        batch = input_ids.shape[0]
        seq_len = 1
        past_len = kv_cache["past_key.0"].shape[2]

        # Attention mask for full sequence (past + new)
        total_len = past_len + seq_len
        attention_mask = torch.ones(batch, total_len, dtype=torch.int64, device=self._device)

        # Output KV cache length
        present_len = past_len + seq_len

        # Set input shapes
        self._gpt_decode_context.set_input_shape("input_ids", (batch, seq_len))
        self._gpt_decode_context.set_input_shape("attention_mask", (batch, total_len))
        for i in range(self.NUM_LAYERS):
            self._gpt_decode_context.set_input_shape(
                f"past_key.{i}", (batch, self.NUM_HEADS, past_len, self.HEAD_DIM)
            )
            self._gpt_decode_context.set_input_shape(
                f"past_value.{i}", (batch, self.NUM_HEADS, past_len, self.HEAD_DIM)
            )

        # Allocate outputs
        logits = torch.empty(batch, seq_len, self.VOCAB_SIZE, dtype=torch.float32, device=self._device)
        latent_codes = torch.empty(batch, seq_len, self.HIDDEN_SIZE, dtype=torch.float32, device=self._device)

        present_cache = {}
        for i in range(self.NUM_LAYERS):
            present_cache[f"present_key.{i}"] = torch.empty(
                batch, self.NUM_HEADS, present_len, self.HEAD_DIM,
                dtype=torch.float32, device=self._device
            )
            present_cache[f"present_value.{i}"] = torch.empty(
                batch, self.NUM_HEADS, present_len, self.HEAD_DIM,
                dtype=torch.float32, device=self._device
            )

        # Set tensor addresses
        self._gpt_decode_context.set_tensor_address("input_ids", input_ids.contiguous().data_ptr())
        self._gpt_decode_context.set_tensor_address("attention_mask", attention_mask.contiguous().data_ptr())

        for name, tensor in kv_cache.items():
            self._gpt_decode_context.set_tensor_address(name, tensor.contiguous().data_ptr())

        self._gpt_decode_context.set_tensor_address("logits", logits.data_ptr())
        self._gpt_decode_context.set_tensor_address("latent_codes", latent_codes.data_ptr())

        for name, tensor in present_cache.items():
            self._gpt_decode_context.set_tensor_address(name, tensor.data_ptr())

        # Execute TRT inference
        stream = torch.cuda.current_stream().cuda_stream
        success = self._gpt_decode_context.execute_async_v3(stream)
        torch.cuda.synchronize()

        if not success:
            raise RuntimeError("TensorRT GPT decode execution failed")

        # Build new KV cache from present
        new_kv_cache = {}
        for i in range(self.NUM_LAYERS):
            new_kv_cache[f"past_key.{i}"] = present_cache[f"present_key.{i}"]
            new_kv_cache[f"past_value.{i}"] = present_cache[f"present_value.{i}"]

        return logits, latent_codes, new_kv_cache

    def _decode_latents(self, latent_buffer: List[torch.Tensor]) -> np.ndarray:
        """Decode latent codes to audio using HiFi-GAN TensorRT engine."""
        if not latent_buffer:
            return np.array([], dtype=np.float32)

        # Concatenate: [batch, time, hidden]
        latents = torch.cat(latent_buffer, dim=1).float().contiguous()
        batch = latents.shape[0]
        time_steps = latents.shape[1]

        # Speaker embedding: [batch, 512, 1]
        speaker_emb = self._speaker_embedding.float().contiguous()

        # Set shapes
        self._decoder_context.set_input_shape("latent_codes", tuple(latents.shape))
        self._decoder_context.set_input_shape("speaker_embedding", tuple(speaker_emb.shape))

        # Get inferred output shape
        inferred_shape = self._decoder_context.get_tensor_shape("waveform")
        output_samples = inferred_shape[1] if inferred_shape[1] > 0 else time_steps * 1100

        output_waveform = torch.empty(batch, output_samples, dtype=torch.float32, device=self._device)

        # Set addresses
        self._decoder_context.set_tensor_address("latent_codes", latents.data_ptr())
        self._decoder_context.set_tensor_address("speaker_embedding", speaker_emb.data_ptr())
        self._decoder_context.set_tensor_address("waveform", output_waveform.data_ptr())

        # Execute
        stream = torch.cuda.current_stream().cuda_stream
        success = self._decoder_context.execute_async_v3(stream)
        torch.cuda.synchronize()

        if not success:
            raise RuntimeError("TensorRT decoder execution failed")

        waveform = output_waveform.squeeze().cpu().numpy()

        # Normalize
        max_val = np.abs(waveform).max()
        if max_val > 1.0:
            waveform = waveform / max_val

        return waveform.astype(np.float32)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            "backend": "xtts_trt_hybrid",
            "mode": "PyTorch prefix + TRT decode",
            "trt_gpt_decode_path": self._trt_gpt_decode_path,
            "trt_decoder_path": self._trt_decoder_path,
            "device": str(self._device),
            "language": self._language,
            "voice_loaded": self._gpt_cond_latent is not None,
            "pytorch_loaded": self._pytorch_model is not None,
        })
        return info
