#!/usr/bin/env python3
"""
Export XTTS-v2 components to ONNX format.

XTTS v2 Architecture:
    Text → GPT-2 Encoder (autoregressive, with KV cache) → HiFi-GAN Decoder → Audio
                ↑
        Conditioning Latents (gpt_cond_latent: [B, 32, 1024])
        Speaker Embedding (speaker_embedding: [B, 512])

This script exports:
    1. gpt.onnx - GPT autoregressive model with KV cache support
    2. hifigan_decoder.onnx - HiFi-GAN vocoder

Conditioning latents are pre-computed using PyTorch (get_conditioning_latents)
and passed as inputs to the ONNX models.

Usage:
    # Export GPT model
    python export_xtts_onnx.py --checkpoint-dir ~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2 --export-gpt

    # Export HiFi-GAN decoder
    python export_xtts_onnx.py --checkpoint-dir ~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2 --export-decoder
"""

import argparse
import os
import logging
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch import nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# GPT Wrapper for ONNX Export with KV Cache
# ============================================================================

class XTTSGPTUnifiedWrapper(nn.Module):
    """
    UNIFIED Wrapper for XTTS GPT - handles both first and subsequent steps.

    IMPORTANT: ONNX tracing "bakes in" control flow, so we CANNOT use if/else.
    Instead, we use a different approach:

    This wrapper ALWAYS:
    1. Takes past_key_values as input (can be zeros/empty for first step)
    2. Takes gpt_cond_latent as input (ignored on decode by masking)
    3. Uses cond_scale tensor to control conditioning application

    Key insight: On decode steps, past_key_values already contain conditioning,
    so we just need to NOT add conditioning again.

    cond_scale = 1.0 for first step (apply conditioning)
    cond_scale = 0.0 for decode steps (skip conditioning)

    Input tensor shapes:
        - input_ids: [batch, seq_len]
        - gpt_cond_latent: [batch, 32, 1024] - Always passed, scaled by cond_scale
        - attention_mask: [batch, total_len]
        - cond_scale: [1] - 1.0 for first step, 0.0 for decode
        - past_key_values: List of [batch, num_heads, past_len, head_dim]

    Output tensor shapes:
        - logits: [batch, seq_len, vocab_size]
        - latent_codes: [batch, seq_len, latent_dim]
        - present_key_values: List of [batch, num_heads, new_len, head_dim]

    Memory: Single TRT engine ~750MB (same as PyTorch)
    """

    def __init__(self, xtts_model, num_layers: int = 30):
        super().__init__()
        self.xtts = xtts_model
        self.gpt = xtts_model.gpt
        self.num_layers = num_layers

        self.num_heads = self.gpt.gpt.config.num_attention_heads
        self.hidden_size = self.gpt.gpt.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        gpt_cond_latent: torch.Tensor,
        attention_mask: torch.Tensor,
        cond_scale: torch.Tensor,
        *past_key_values_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, ...]:
        """
        Unified forward - uses cond_scale to control conditioning.

        cond_scale=1.0: First step - prepend conditioning
        cond_scale=0.0: Decode - use KV cache, no conditioning
        """
        # Get text token embeddings
        text_emb = self.gpt.text_embedding(input_ids)  # [batch, seq_len, 1024]

        # Scale conditioning (0.0 on decode = zero tensor = no effect when concat)
        # BUT - concat still changes shape! We need different approach.
        #
        # Better: ALWAYS prepend conditioning, but on decode steps,
        # the past_key_values is LARGER (includes conditioning from step 0)
        # and attention_mask accounts for it.
        #
        # Actually the cleanest solution:
        # - First step: emb = [cond, text], past_kv = None
        # - Decode: emb = [text], past_kv from previous (already has cond in it)
        #
        # The problem is we can't use if/else. But we CAN use torch.where!

        # Reconstruct past_key_values
        past_key_values = []
        for i in range(self.num_layers):
            key = past_key_values_flat[i * 2]
            value = past_key_values_flat[i * 2 + 1]
            past_key_values.append((key, value))
        past_key_values = tuple(past_key_values)

        # Check if we have actual past (past_len > 0)
        past_len = past_key_values_flat[0].shape[2] if len(past_key_values_flat) > 0 else 0

        # Scale conditioning: on first step (cond_scale=1), prepend. On decode (cond_scale=0), don't.
        # Using multiplication: cond * scale = cond (first) or zeros (decode)
        scaled_cond = gpt_cond_latent * cond_scale  # [B, 32, 1024]

        # Build embeddings: always cat, but scaled_cond is zeros on decode
        # This preserves shape consistency for ONNX
        emb = torch.cat([scaled_cond, text_emb], dim=1)  # [B, 32+seq, 1024]

        # Run GPT
        # past_key_values will be None-like (zeros) on first step,
        # actual cache on decode steps
        gpt_out = self.gpt.gpt(
            inputs_embeds=emb,
            past_key_values=past_key_values if past_len > 0 else None,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        hidden_states = gpt_out.last_hidden_state
        present_key_values = gpt_out.past_key_values

        # Skip conditioning tokens in output
        # First step: skip 32 cond tokens (keep text tokens)
        # Decode: also skip 32 (the zeros from scaled_cond)
        cond_len = gpt_cond_latent.shape[1]
        hidden_states = hidden_states[:, cond_len:, :]

        # Get logits and latent codes
        logits = self.gpt.mel_head(hidden_states)
        latent_codes = self.gpt.final_norm(hidden_states)

        # Flatten present_key_values
        present_flat = []
        for layer_kv in present_key_values:
            present_flat.append(layer_kv[0])
            present_flat.append(layer_kv[1])

        return (logits, latent_codes, *present_flat)


class XTTSGPTPrefixWrapper(nn.Module):
    """
    Wrapper for XTTS GPT FIRST STEP (prefix) - includes conditioning.

    This model is used for the first forward pass where:
    - Text tokens are padded with start/stop tokens
    - Position embeddings are added to text embeddings
    - Conditioning latents are prepended
    - start_audio_token embedding is appended
    - KV cache is empty (past_len = 0)
    - Output KV cache includes conditioning + text + start_audio

    Input tensor shapes:
        - input_ids: [batch, seq_len] - Text tokens (WITHOUT start/stop padding)
        - gpt_cond_latent: [batch, 32, 1024] - Speaker conditioning

    Output tensor shapes:
        - logits: [batch, 1, vocab_size] - Next token prediction (for start_audio position)
        - present_key_values: List of [batch, num_heads, prefix_len+1, head_dim]
    """

    def __init__(self, xtts_model, num_layers: int = 30):
        super().__init__()
        self.xtts = xtts_model
        self.gpt = xtts_model.gpt
        self.num_layers = num_layers

        self.num_heads = self.gpt.gpt.config.num_attention_heads
        self.hidden_size = self.gpt.gpt.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        # Store special tokens
        self.start_text_token = self.gpt.start_text_token
        self.stop_text_token = self.gpt.stop_text_token
        self.start_audio_token = self.gpt.start_audio_token

        self.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        gpt_cond_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """First step forward pass - prepend conditioning, add start_audio.

        Matches GPT.compute_embeddings() + GPT2InferenceModel.forward() first call.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 1. Pad text tokens with start/stop
        # [batch, seq_len] -> [batch, seq_len + 2]
        text_tokens_padded = torch.cat([
            torch.full((batch_size, 1), self.start_text_token, dtype=torch.long, device=device),
            input_ids,
            torch.full((batch_size, 1), self.stop_text_token, dtype=torch.long, device=device),
        ], dim=1)

        # 2. Get text embeddings WITH position embeddings
        text_emb = self.gpt.text_embedding(text_tokens_padded)
        text_emb = text_emb + self.gpt.text_pos_embedding(text_tokens_padded)

        # 3. Build prefix: [cond_latents, text_emb]
        prefix_emb = torch.cat([gpt_cond_latent, text_emb], dim=1)
        prefix_len = prefix_emb.shape[1]

        # 4. Add start_audio_token embedding with mel_pos_embedding at position 0
        start_audio_tokens = torch.full((batch_size, 1), self.start_audio_token, dtype=torch.long, device=device)
        start_audio_emb = self.gpt.mel_embedding(start_audio_tokens)
        start_audio_emb = start_audio_emb + self.gpt.mel_pos_embedding(start_audio_tokens)

        # 5. Full embedding: [prefix, start_audio]
        emb = torch.cat([prefix_emb, start_audio_emb], dim=1)

        # 6. Attention mask
        total_len = emb.shape[1]
        attention_mask = torch.ones(batch_size, total_len, dtype=torch.long, device=device)

        # 7. Run through GPT transformer
        gpt_out = self.gpt.gpt(
            inputs_embeds=emb,
            past_key_values=None,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        # 8. Get logits from last position (after start_audio_token)
        hidden = gpt_out.last_hidden_state[:, -1:, :]
        logits = self.gpt.mel_head(hidden)

        # Flatten present_key_values
        present_flat = []
        for layer_kv in gpt_out.past_key_values:
            present_flat.append(layer_kv[0])  # key
            present_flat.append(layer_kv[1])  # value

        return (logits, *present_flat)


class XTTSGPTDecodeWrapper(nn.Module):
    """
    Wrapper for XTTS GPT SUBSEQUENT STEPS (decode) - no conditioning.

    This model is used for autoregressive decoding where:
    - Single new MEL token embedding is processed with position embedding
    - Past KV cache contains prefix (conditioning + text) + previous MEL tokens
    - Output KV cache grows by 1

    CRITICAL: Position embedding is needed for correct output!
    mel_position = total_seq_len - prefix_len - 1
    where prefix_len = cond_len(32) + text_len_with_tokens

    Input tensor shapes:
        - input_ids: [batch, 1] - Single new MEL token (0-1023)
        - mel_position: [batch, 1] - Position index for mel_pos_embedding
        - attention_mask: [batch, past_len + 1] - Mask for all positions
        - past_key_values: List of [batch, num_heads, past_len, head_dim]

    Output tensor shapes:
        - logits: [batch, 1, vocab_size]
        - present_key_values: List of [batch, num_heads, past_len+1, head_dim]
    """

    def __init__(self, xtts_model, num_layers: int = 30):
        super().__init__()
        self.xtts = xtts_model
        self.gpt = xtts_model.gpt
        self.num_layers = num_layers

        self.num_heads = self.gpt.gpt.config.num_attention_heads
        self.hidden_size = self.gpt.gpt.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        mel_position: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_key_values_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Decode step forward pass - single MEL token with position embedding.

        IMPORTANT: Uses mel_embedding + mel_pos_embedding!
        Position is relative to start of MEL sequence (not absolute position).
        """
        # Reconstruct past_key_values
        past_key_values = []
        for i in range(self.num_layers):
            key = past_key_values_flat[i * 2]
            value = past_key_values_flat[i * 2 + 1]
            past_key_values.append((key, value))
        past_key_values = tuple(past_key_values)

        # Get single MEL token embedding with position embedding
        mel_emb = self.gpt.mel_embedding(input_ids)  # [batch, 1, 1024]

        # Add position embedding using the provided position index
        # mel_pos_embedding.get_fixed_embedding expects scalar position
        # For ONNX export, we need to handle this differently
        # Use direct embedding lookup instead
        mel_pos_emb = self.gpt.mel_pos_embedding.emb(mel_position)  # [batch, 1, 1024]
        mel_emb = mel_emb + mel_pos_emb

        # Run through GPT with KV cache
        gpt_out = self.gpt.gpt(
            inputs_embeds=mel_emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        hidden_states = gpt_out.last_hidden_state
        present_key_values = gpt_out.past_key_values

        # Get logits
        logits = self.gpt.mel_head(hidden_states)

        # Flatten present_key_values
        present_flat = []
        for layer_kv in present_key_values:
            present_flat.append(layer_kv[0])
            present_flat.append(layer_kv[1])

        return (logits, *present_flat)


# Legacy wrapper for backward compatibility (used if only one model needed)
class XTTSGPTWrapper(nn.Module):
    """Legacy wrapper - wraps both prefix and decode functionality."""

    def __init__(self, xtts_model, num_layers: int = 30):
        super().__init__()
        self.prefix_model = XTTSGPTPrefixWrapper(xtts_model, num_layers)
        self.decode_model = XTTSGPTDecodeWrapper(xtts_model, num_layers)
        self.num_layers = num_layers
        self.num_heads = self.prefix_model.num_heads
        self.head_dim = self.prefix_model.head_dim
        self.eval()

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "XTTSGPTWrapper should not be called directly. "
            "Use XTTSGPTPrefixWrapper for first step and XTTSGPTDecodeWrapper for subsequent steps."
        )


# ============================================================================
# HiFi-GAN Decoder Wrapper
# ============================================================================

class HiFiGANWrapper(nn.Module):
    """
    Wrapper for XTTS HiFi-GAN decoder for ONNX export.

    The HiFi-GAN decoder converts GPT latent codes to audio waveform.

    XTTS HifiDecoder.forward() expects:
        - latents: [batch, time_steps, 1024] - GPT latent output
        - g: [batch, 512, 1] - Speaker embedding (3D tensor!)

    Input tensor shapes (for ONNX):
        - latent_codes: [batch, time_steps, 1024] - From GPT
        - speaker_embedding: [batch, 512, 1] - Speaker characteristics (3D)

    Output tensor shapes:
        - waveform: [batch, audio_samples] - 24kHz audio
    """

    def __init__(self, xtts_model):
        super().__init__()
        self.hifigan = xtts_model.hifigan_decoder
        self.eval()

    def forward(
        self,
        latent_codes: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert latent codes to audio waveform.

        Args:
            latent_codes: [batch, time_steps, 1024] - GPT latent output
            speaker_embedding: [batch, 512, 1] - Speaker embedding (3D)

        Returns:
            waveform: [batch, audio_samples] at 24kHz
        """
        # The XTTS HiFi-GAN decoder takes:
        # - latents: [B, T, C] format (time_steps, 1024)
        # - g: speaker embedding [B, 512, 1]
        waveform = self.hifigan(latent_codes, g=speaker_embedding)

        return waveform.squeeze(1)  # [batch, audio_samples]


# ============================================================================
# Export Functions
# ============================================================================

def export_gpt_prefix(
    model,
    outdir: Path,
    opset: int = 17,
    num_layers: int = 30,
) -> Path:
    """
    Export GPT PREFIX model (first step with conditioning) to ONNX.

    This model:
    - Takes raw text tokens (without start/stop padding)
    - Adds start/stop text tokens internally
    - Adds text position embeddings
    - Prepends conditioning latents
    - Adds start_audio_token with mel position embedding
    - Outputs logits for next token prediction
    - Outputs KV cache with full prefix

    Args:
        model: XTTS model instance
        outdir: Output directory
        opset: ONNX opset version
        num_layers: Number of transformer layers

    Returns:
        Path to exported ONNX file
    """
    logger.info("Exporting GPT PREFIX model (first step)...")

    wrapper = XTTSGPTPrefixWrapper(model, num_layers=num_layers)
    wrapper.eval()

    num_heads = wrapper.num_heads
    head_dim = wrapper.head_dim

    logger.info(f"GPT config: {num_layers} layers, {num_heads} heads, {head_dim} head_dim")
    logger.info(f"start_text_token: {wrapper.start_text_token}")
    logger.info(f"stop_text_token: {wrapper.stop_text_token}")
    logger.info(f"start_audio_token: {wrapper.start_audio_token}")

    # Dummy inputs
    batch_size = 1
    seq_len = 5  # Variable text length (without start/stop)
    cond_len = 32

    # text_embedding vocab size is ~6681
    input_ids = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long)
    gpt_cond_latent = torch.randn(batch_size, cond_len, 1024)

    # Input/output names
    # Note: output is now just logits (single position) + KV cache
    input_names = ["input_ids", "gpt_cond_latent"]
    output_names = ["logits"]

    for i in range(num_layers):
        output_names.append(f"present_key.{i}")
        output_names.append(f"present_value.{i}")

    # Dynamic axes
    # After prefix: present_seq_len = cond_len + (seq_len + 2) + 1 = cond_len + seq_len + 3
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "gpt_cond_latent": {0: "batch"},
        "logits": {0: "batch"},  # Shape: [batch, 1, vocab_size]
    }

    for i in range(num_layers):
        dynamic_axes[f"present_key.{i}"] = {0: "batch", 2: "present_seq_len"}
        dynamic_axes[f"present_value.{i}"] = {0: "batch", 2: "present_seq_len"}

    # Export
    onnx_path = outdir / "gpt_prefix.onnx"

    logger.info(f"Tracing prefix model...")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, gpt_cond_latent),
            str(onnx_path),
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )

    logger.info(f"Exported GPT PREFIX model -> {onnx_path}")
    return onnx_path


def export_gpt_decode(
    model,
    outdir: Path,
    opset: int = 17,
    num_layers: int = 30,
) -> Path:
    """
    Export GPT DECODE model (subsequent steps with KV cache) to ONNX.

    This model:
    - Takes single MEL token (0-1023)
    - Takes mel_position for position embedding lookup
    - Takes KV cache from previous step
    - Adds mel_embedding + mel_pos_embedding
    - Outputs logits for next token
    - Outputs updated KV cache

    Args:
        model: XTTS model instance
        outdir: Output directory
        opset: ONNX opset version
        num_layers: Number of transformer layers

    Returns:
        Path to exported ONNX file
    """
    logger.info("Exporting GPT DECODE model (subsequent steps)...")

    wrapper = XTTSGPTDecodeWrapper(model, num_layers=num_layers)
    wrapper.eval()

    num_heads = wrapper.num_heads
    head_dim = wrapper.head_dim

    logger.info(f"GPT config: {num_layers} layers, {num_heads} heads, {head_dim} head_dim")

    # Dummy inputs - single MEL token with past KV cache
    batch_size = 1
    seq_len = 1  # Single token
    past_len = 45  # cond(32) + text(~11 with start/stop) + start_audio(1) + some mel codes

    # mel_embedding vocab size is 1026 (0-1023 mel codes + 1024 start + 1025 stop)
    # During decode we generate mel tokens 0-1023
    input_ids = torch.randint(0, 1024, (batch_size, seq_len), dtype=torch.long)

    # mel_position: position in MEL sequence (0, 1, 2, ...)
    # After prefix step, we're at position 1 (position 0 was start_audio_token)
    mel_position = torch.tensor([[1]], dtype=torch.long)

    attention_mask = torch.ones(batch_size, past_len + seq_len, dtype=torch.long)

    # Past KV cache
    past_kv_flat = []
    for _ in range(num_layers):
        key = torch.randn(batch_size, num_heads, past_len, head_dim)
        value = torch.randn(batch_size, num_heads, past_len, head_dim)
        past_kv_flat.extend([key, value])

    # Input/output names - note: mel_position is now an input
    input_names = ["input_ids", "mel_position", "attention_mask"]
    output_names = ["logits"]

    for i in range(num_layers):
        input_names.append(f"past_key.{i}")
        input_names.append(f"past_value.{i}")
        output_names.append(f"present_key.{i}")
        output_names.append(f"present_value.{i}")

    # Dynamic axes
    dynamic_axes = {
        "input_ids": {0: "batch"},  # seq_len fixed at 1
        "mel_position": {0: "batch"},  # [batch, 1]
        "attention_mask": {0: "batch", 1: "total_seq_len"},
        "logits": {0: "batch"},
    }

    for i in range(num_layers):
        dynamic_axes[f"past_key.{i}"] = {0: "batch", 2: "past_seq_len"}
        dynamic_axes[f"past_value.{i}"] = {0: "batch", 2: "past_seq_len"}
        dynamic_axes[f"present_key.{i}"] = {0: "batch", 2: "present_seq_len"}
        dynamic_axes[f"present_value.{i}"] = {0: "batch", 2: "present_seq_len"}

    # Export
    onnx_path = outdir / "gpt_decode.onnx"

    logger.info(f"Tracing decode model...")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, mel_position, attention_mask, *past_kv_flat),
            str(onnx_path),
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )

    logger.info(f"Exported GPT DECODE model -> {onnx_path}")
    return onnx_path


def export_gpt(
    model,
    outdir: Path,
    opset: int = 17,
    num_layers: int = 30,
) -> Tuple[Path, Path]:
    """
    Export both GPT PREFIX and DECODE models to ONNX.

    For TensorRT backend, two separate models are needed:
    - gpt_prefix.onnx: First step (with conditioning)
    - gpt_decode.onnx: Subsequent steps (KV cache only)

    Returns:
        Tuple of (prefix_path, decode_path)
    """
    prefix_path = export_gpt_prefix(model, outdir, opset, num_layers)
    decode_path = export_gpt_decode(model, outdir, opset, num_layers)
    return prefix_path, decode_path


def export_hifigan_decoder(
    model,
    outdir: Path,
    opset: int = 17,
) -> Path:
    """
    Export HiFi-GAN decoder to ONNX.

    Args:
        model: XTTS model instance
        outdir: Output directory
        opset: ONNX opset version

    Returns:
        Path to exported ONNX file
    """
    logger.info("Exporting HiFi-GAN decoder...")

    wrapper = HiFiGANWrapper(model)
    wrapper.eval()

    # Dummy inputs - CORRECT SHAPES based on XTTS inference output:
    # gpt_latents: [B, T, 1024] - time_steps x latent_dim
    # speaker_embedding: [B, 512, 1] - 3D tensor!
    batch_size = 1
    time_steps = 16  # Variable (matches typical GPT output)
    latent_dim = 1024
    speaker_dim = 512

    # latent_codes: [batch, time_steps, latent_dim] = [1, 16, 1024]
    latent_codes = torch.randn(batch_size, time_steps, latent_dim)
    # speaker_embedding: [batch, speaker_dim, 1] = [1, 512, 1]
    speaker_embedding = torch.randn(batch_size, speaker_dim, 1)

    logger.info(f"Dummy latent_codes shape: {latent_codes.shape}")
    logger.info(f"Dummy speaker_embedding shape: {speaker_embedding.shape}")

    # Export
    decoder_path = outdir / "hifigan_decoder.onnx"

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (latent_codes, speaker_embedding),
            str(decoder_path),
            opset_version=opset,
            input_names=["latent_codes", "speaker_embedding"],
            output_names=["waveform"],
            dynamic_axes={
                "latent_codes": {0: "batch", 1: "time_steps"},
                "speaker_embedding": {0: "batch"},
                "waveform": {0: "batch", 1: "audio_samples"},
            },
            do_constant_folding=True,
            verbose=False,
        )

    logger.info(f"Exported HiFi-GAN decoder -> {decoder_path}")
    return decoder_path


def load_xtts_model(checkpoint_dir: str, device: str = "cpu"):
    """
    Load XTTS model from checkpoint directory.

    Args:
        checkpoint_dir: Path to XTTS checkpoint directory
        device: Device to load model on (cpu recommended for export)

    Returns:
        Loaded XTTS model
    """
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    # PyTorch 2.6+ security: allow unpickling XTTS classes
    try:
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
        torch.serialization.add_safe_globals([
            XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs
        ])
    except Exception as e:
        logger.warning(f"Could not add safe globals: {e}")

    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    logger.info(f"Loading XTTS model from: {checkpoint_dir}")

    config = XttsConfig()
    config.load_json(config_path)

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, use_deepspeed=False)
    model.to(device)
    model.eval()

    logger.info("XTTS model loaded successfully")
    return model


def verify_onnx(onnx_path: Path) -> bool:
    """
    Verify ONNX model is valid.

    Args:
        onnx_path: Path to ONNX file

    Returns:
        True if valid
    """
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        logger.info(f"ONNX model verified: {onnx_path}")
        return True
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export XTTS-v2 components to ONNX format."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to XTTS checkpoint directory (contains config.json and model.pth)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="models/onnx",
        help="Output directory for ONNX files"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--export-gpt",
        action="store_true",
        help="Export GPT model with KV cache"
    )
    parser.add_argument(
        "--export-decoder",
        action="store_true",
        help="Export HiFi-GAN decoder"
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export all components"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported ONNX models"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for model loading (cpu recommended for export)"
    )

    args = parser.parse_args()

    if not args.export_gpt and not args.export_decoder and not args.export_all:
        parser.error("Specify --export-gpt, --export-decoder, or --export-all")

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_xtts_model(args.checkpoint_dir, device=args.device)

    exported_files = []

    # Export components
    if args.export_gpt or args.export_all:
        gpt_path = export_gpt(model, outdir, opset=args.opset)
        exported_files.append(gpt_path)

    if args.export_decoder or args.export_all:
        decoder_path = export_hifigan_decoder(model, outdir, opset=args.opset)
        exported_files.append(decoder_path)

    # Verify
    if args.verify:
        logger.info("Verifying exported models...")
        for path in exported_files:
            verify_onnx(path)

    logger.info(f"Export complete. Files saved to: {outdir}")
    for f in exported_files:
        # f can be Path or tuple of Paths
        if isinstance(f, tuple):
            for path in f:
                logger.info(f"  - {path.name}")
        else:
            logger.info(f"  - {f.name}")


if __name__ == "__main__":
    main()
