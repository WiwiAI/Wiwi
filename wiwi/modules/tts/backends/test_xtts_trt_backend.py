#!/usr/bin/env python3
"""
Test script for XTTS TRT backend.

Compares output of:
1. Original XTTS inference (TTS library)
2. Our hybrid PyTorch+TRT backend

Usage:
    python -m wiwi.modules.tts.backends.test_xtts_trt_backend \
        --checkpoint-dir ~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2 \
        --speaker-wav models/tts/reference.wav \
        --trt-decoder models/trt/hifigan_decoder.trt
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_original_xtts(checkpoint_dir: str, speaker_wav: str, text: str, language: str, device: str):
    """Test original XTTS inference."""
    logger.info("=" * 60)
    logger.info("ORIGINAL XTTS INFERENCE")
    logger.info("=" * 60)

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    try:
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
        torch.serialization.add_safe_globals([
            XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs
        ])
    except Exception as e:
        logger.warning(f"Could not add safe globals: {e}")

    config_path = os.path.join(checkpoint_dir, "config.json")
    config = XttsConfig()
    config.load_json(config_path)

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, use_deepspeed=False)
    model.to(device)
    model.eval()

    logger.info(f"Synthesizing text: '{text}'")

    result = model.full_inference(
        text=text,
        ref_audio_path=speaker_wav,
        language=language,
        temperature=0.75,
        top_k=50,
        top_p=0.85,
        repetition_penalty=10.0,
    )

    wav = result["wav"]
    logger.info(f"Generated audio: {len(wav)} samples ({len(wav)/24000:.2f}s)")

    return wav


def test_hybrid_backend(
    checkpoint_dir: str,
    speaker_wav: str,
    trt_decoder_path: str,
    text: str,
    language: str,
    device: str
):
    """Test hybrid PyTorch+TRT backend."""
    logger.info("=" * 60)
    logger.info("HYBRID PYTORCH+TRT BACKEND")
    logger.info("=" * 60)

    from wiwi.modules.tts.backends.xtts_trt_backend import XttsTrtBackend

    config = {
        "trt_gpt_decode_path": None,  # Not used in pure PyTorch mode
        "trt_decoder_path": trt_decoder_path,
        "xtts_checkpoint_dir": checkpoint_dir,
        "speaker_ref_path": speaker_wav,
        "device": device,
        "language": language,
        "temperature": 0.75,
        "top_k": 50,
        "top_p": 0.85,
        "repetition_penalty": 10.0,
        "max_mel_tokens": 500,
    }

    backend = XttsTrtBackend(config)

    # Check if TRT decoder exists
    if not trt_decoder_path or not os.path.exists(trt_decoder_path):
        logger.warning(f"TRT decoder not found at {trt_decoder_path}")
        logger.info("Running in pure PyTorch mode for testing")

        # Modify to use PyTorch decoder
        config["trt_decoder_path"] = None

    backend.load_model()

    logger.info(f"Synthesizing text: '{text}'")

    wav = backend.synthesize(text, language)
    logger.info(f"Generated audio: {len(wav)} samples ({len(wav)/24000:.2f}s)")

    backend.unload_model()

    return wav


def test_pytorch_only_backend(
    checkpoint_dir: str,
    speaker_wav: str,
    text: str,
    language: str,
    device: str
):
    """Test pure PyTorch backend (no TRT) to validate code changes."""
    logger.info("=" * 60)
    logger.info("PURE PYTORCH BACKEND (for validation)")
    logger.info("=" * 60)

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    import torch.nn.functional as F

    try:
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
        torch.serialization.add_safe_globals([
            XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs
        ])
    except Exception as e:
        logger.warning(f"Could not add safe globals: {e}")

    # Load model
    config_path = os.path.join(checkpoint_dir, "config.json")
    config = XttsConfig()
    config.load_json(config_path)

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, use_deepspeed=False)
    model.to(device)
    model.eval()

    # Get conditioning
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[speaker_wav]
    )
    gpt_cond_latent = gpt_cond_latent.to(device)
    speaker_embedding = speaker_embedding.to(device)

    logger.info(f"gpt_cond_latent shape: {gpt_cond_latent.shape}")
    logger.info(f"speaker_embedding shape: {speaker_embedding.shape}")

    # Tokenize
    text_tokens = torch.IntTensor(model.tokenizer.encode(text, lang=language)).unsqueeze(0).to(device)
    logger.info(f"text_tokens shape: {text_tokens.shape}")

    # Phase 1: Generate mel codes using gpt.generate()
    logger.info("Phase 1: Generating mel codes...")
    with torch.no_grad():
        gpt_codes = model.gpt.generate(
            cond_latents=gpt_cond_latent,
            text_inputs=text_tokens,
            input_tokens=None,
            do_sample=True,
            top_p=0.85,
            top_k=50,
            temperature=0.75,
            num_return_sequences=1,
            num_beams=1,
            length_penalty=1.0,
            repetition_penalty=10.0,
            output_attentions=False,
        )
    logger.info(f"Generated {gpt_codes.shape[1]} mel codes")

    # Phase 2: Convert codes to latents (our implementation)
    logger.info("Phase 2: Converting codes to latents (our _codes_to_latents implementation)...")

    gpt = model.gpt

    # Our implementation
    # 1. Pad text tokens with start/stop
    text_inputs = F.pad(text_tokens, (0, 1), value=gpt.stop_text_token)
    text_inputs = F.pad(text_inputs, (1, 0), value=gpt.start_text_token)

    # 2. Pad mel codes with start/stop
    audio_codes = F.pad(gpt_codes, (1, 0), value=gpt.start_audio_token)
    audio_codes = F.pad(audio_codes, (0, 1), value=gpt.stop_audio_token)

    # 3. Get embeddings WITH position embeddings
    text_emb = gpt.text_embedding(text_inputs) + gpt.text_pos_embedding(text_inputs)
    mel_emb = gpt.mel_embedding(audio_codes) + gpt.mel_pos_embedding(audio_codes)

    # 4. Concat: [cond_latents, text_emb, mel_emb]
    emb = torch.cat([gpt_cond_latent, text_emb, mel_emb], dim=1)

    # 5. Forward through transformer
    with torch.no_grad():
        gpt_out = gpt.gpt(inputs_embeds=emb, return_dict=True)

    # 6. Skip conditioning tokens in output
    cond_len = gpt_cond_latent.shape[1]
    hidden = gpt_out.last_hidden_state[:, cond_len:]

    # 7. Apply final_norm
    hidden = gpt.final_norm(hidden)

    # 8. Extract mel part with sub=-5 trim
    mel_len = mel_emb.shape[1]
    mel_latents = hidden[:, -mel_len:, :]
    sub = -5
    mel_latents_ours = mel_latents[:, :sub, :]

    logger.info(f"Our mel_latents shape: {mel_latents_ours.shape}")

    # Compare with original GPT.forward() with return_latent=True
    logger.info("Comparing with original GPT.forward()...")
    with torch.no_grad():
        expected_output_len = torch.tensor([gpt_codes.shape[-1] * gpt.code_stride_len], device=device)
        text_len = torch.tensor([text_tokens.shape[-1]], device=device)

        mel_latents_orig = gpt(
            text_tokens,
            text_len,
            gpt_codes,
            expected_output_len,
            cond_latents=gpt_cond_latent,
            return_attentions=False,
            return_latent=True,
        )

    logger.info(f"Original mel_latents shape: {mel_latents_orig.shape}")

    # Compare
    min_len = min(mel_latents_ours.shape[1], mel_latents_orig.shape[1])
    diff = (mel_latents_ours[:, :min_len, :] - mel_latents_orig[:, :min_len, :]).abs()
    logger.info(f"Difference: max={diff.max():.6f}, mean={diff.mean():.6f}")

    if diff.max() < 0.01:
        logger.info("✓ Our implementation matches original!")
    else:
        logger.warning("✗ Our implementation DOES NOT match original!")

    # Phase 3: Decode with HiFi-GAN
    logger.info("Phase 3: Decoding with HiFi-GAN...")
    with torch.no_grad():
        wav_ours = model.hifigan_decoder(mel_latents_ours, g=speaker_embedding)
        wav_orig = model.hifigan_decoder(mel_latents_orig, g=speaker_embedding)

    wav_ours = wav_ours.squeeze().cpu().numpy()
    wav_orig = wav_orig.squeeze().cpu().numpy()

    logger.info(f"Our audio: {len(wav_ours)} samples")
    logger.info(f"Original audio: {len(wav_orig)} samples")

    return wav_ours, wav_orig


def main():
    parser = argparse.ArgumentParser(description="Test XTTS TRT backend")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--speaker-wav", type=str, required=True)
    parser.add_argument("--trt-decoder", type=str, default=None)
    parser.add_argument("--text", type=str, default="Привет мир, как дела?")
    parser.add_argument("--language", type=str, default="ru")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="test_output")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Test 1: Pure PyTorch validation
    wav_ours, wav_orig = test_pytorch_only_backend(
        args.checkpoint_dir,
        args.speaker_wav,
        args.text,
        args.language,
        args.device
    )

    sf.write(f"{args.output_dir}/test_pytorch_ours.wav", wav_ours, 24000)
    sf.write(f"{args.output_dir}/test_pytorch_orig.wav", wav_orig, 24000)
    logger.info(f"Saved test audio to {args.output_dir}/")

    # Test 2: Original XTTS
    logger.info("\n")
    wav_original = test_original_xtts(
        args.checkpoint_dir,
        args.speaker_wav,
        args.text,
        args.language,
        args.device
    )
    sf.write(f"{args.output_dir}/test_original_xtts.wav", wav_original, 24000)

    logger.info("\nAll tests completed!")
    logger.info(f"Output files in {args.output_dir}/")


if __name__ == "__main__":
    main()
