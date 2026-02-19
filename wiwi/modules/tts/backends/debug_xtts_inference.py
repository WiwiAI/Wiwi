#!/usr/bin/env python3
"""
Debug script to compare original XTTS inference vs custom implementation.

This script helps identify discrepancies between:
1. Original TTS library XTTS inference
2. Custom PyTorch implementation (for TRT backend)

Run with:
    python -m wiwi.modules.tts.backends.debug_xtts_inference \
        --checkpoint-dir ~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2 \
        --speaker-wav voices/speaker.wav \
        --text "Привет мир"
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_xtts_model(checkpoint_dir: str, device: str = "cuda:0"):
    """Load XTTS model from checkpoint."""
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

    return model


def debug_original_inference(model, text: str, speaker_wav: str, language: str = "ru"):
    """
    Run original XTTS inference and capture intermediate values.
    """
    logger.info("=" * 60)
    logger.info("ORIGINAL XTTS INFERENCE")
    logger.info("=" * 60)

    device = model.device

    # Get conditioning latents
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[speaker_wav]
    )
    gpt_cond_latent = gpt_cond_latent.to(device)
    speaker_embedding = speaker_embedding.to(device)

    logger.info(f"gpt_cond_latent shape: {gpt_cond_latent.shape}")  # [1, 1024, 32] -> transpose -> [1, 32, 1024]
    logger.info(f"speaker_embedding shape: {speaker_embedding.shape}")  # [1, 512, 1]

    # Tokenize text
    text_tokens = torch.IntTensor(model.tokenizer.encode(text, lang=language)).unsqueeze(0).to(device)
    logger.info(f"text_tokens shape: {text_tokens.shape}")
    logger.info(f"text_tokens: {text_tokens[0].tolist()}")

    # Show model tokens
    gpt = model.gpt
    logger.info(f"start_text_token: {gpt.start_text_token}")
    logger.info(f"stop_text_token: {gpt.stop_text_token}")
    logger.info(f"start_audio_token: {gpt.start_audio_token}")
    logger.info(f"stop_audio_token: {gpt.stop_audio_token}")
    logger.info(f"num_audio_tokens: {gpt.num_audio_tokens}")

    # ========== Phase 1: Generate mel codes using original method ==========
    logger.info("-" * 40)
    logger.info("Phase 1: GPT generate (mel codes)")
    logger.info("-" * 40)

    # Examine compute_embeddings internals
    with torch.no_grad():
        # Original compute_embeddings (from GPT.compute_embeddings)
        text_inputs = F.pad(text_tokens, (0, 1), value=gpt.stop_text_token)
        text_inputs = F.pad(text_inputs, (1, 0), value=gpt.start_text_token)
        logger.info(f"text_inputs (with start/stop): {text_inputs.shape}")
        logger.info(f"text_inputs: {text_inputs[0].tolist()}")

        # Get embeddings WITH position embeddings
        text_emb = gpt.text_embedding(text_inputs)
        logger.info(f"text_emb shape (no pos): {text_emb.shape}")

        text_pos_emb = gpt.text_pos_embedding(text_inputs)
        logger.info(f"text_pos_emb shape: {text_pos_emb.shape}")

        text_emb_with_pos = text_emb + text_pos_emb
        logger.info(f"text_emb + pos shape: {text_emb_with_pos.shape}")

        # Concat with conditioning
        # Note: gpt_cond_latent is [1, 1024, 32] but compute_embeddings expects [1, 32, 1024]?
        # Let's check the actual flow
        logger.info(f"gpt_cond_latent for concat: {gpt_cond_latent.shape}")

        # In XTTS, gpt_cond_latent is returned as [1, 1024, T] from get_gpt_cond_latents
        # Then transposed in get_style_emb or during inference
        # Check if cond_latent needs transpose
        if gpt_cond_latent.shape[1] == 1024:
            cond_latent_for_concat = gpt_cond_latent.transpose(1, 2)
            logger.info(f"Transposed cond_latent: {cond_latent_for_concat.shape}")
        else:
            cond_latent_for_concat = gpt_cond_latent

        emb = torch.cat([cond_latent_for_concat, text_emb_with_pos], dim=1)
        logger.info(f"emb (cond + text) shape: {emb.shape}")

        # Generate mel codes
        logger.info("Running gpt.generate()...")
        gpt_codes = gpt.generate(
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
        logger.info(f"gpt_codes shape: {gpt_codes.shape}")
        logger.info(f"gpt_codes first 20: {gpt_codes[0, :20].tolist()}")
        logger.info(f"gpt_codes last 20: {gpt_codes[0, -20:].tolist()}")

    # ========== Phase 2: Convert codes to latents ==========
    logger.info("-" * 40)
    logger.info("Phase 2: Codes to Latents")
    logger.info("-" * 40)

    with torch.no_grad():
        # This is from XTTS.inference(), line 561-568
        expected_output_len = torch.tensor(
            [gpt_codes.shape[-1] * gpt.code_stride_len], device=device
        )
        text_len = torch.tensor([text_tokens.shape[-1]], device=device)

        logger.info(f"Calling gpt.forward() with return_latent=True")
        logger.info(f"  text_tokens: {text_tokens.shape}")
        logger.info(f"  text_len: {text_len}")
        logger.info(f"  gpt_codes: {gpt_codes.shape}")
        logger.info(f"  expected_output_len: {expected_output_len}")
        logger.info(f"  cond_latents: {gpt_cond_latent.shape}")

        # GPT.forward() with return_latent=True
        # This internally does:
        # 1. text_emb = text_embedding(text) + text_pos_embedding(text)
        # 2. mel_emb = mel_embedding(codes) + mel_pos_embedding(codes)
        # 3. Forward through transformer with prompt=cond_latents
        # 4. Return latents (not logits)

        gpt_latents = gpt(
            text_tokens,
            text_len,
            gpt_codes,
            expected_output_len,
            cond_latents=gpt_cond_latent,
            return_attentions=False,
            return_latent=True,
        )

        logger.info(f"gpt_latents shape: {gpt_latents.shape}")
        logger.info(f"gpt_latents mean: {gpt_latents.mean().item():.4f}")
        logger.info(f"gpt_latents std: {gpt_latents.std().item():.4f}")
        logger.info(f"gpt_latents min: {gpt_latents.min().item():.4f}")
        logger.info(f"gpt_latents max: {gpt_latents.max().item():.4f}")

    # ========== Phase 3: Decode with HiFi-GAN ==========
    logger.info("-" * 40)
    logger.info("Phase 3: HiFi-GAN Decode")
    logger.info("-" * 40)

    with torch.no_grad():
        logger.info(f"Calling hifigan_decoder()")
        logger.info(f"  gpt_latents: {gpt_latents.shape}")
        logger.info(f"  speaker_embedding (g): {speaker_embedding.shape}")

        wav = model.hifigan_decoder(gpt_latents, g=speaker_embedding)
        logger.info(f"wav shape: {wav.shape}")

    return {
        "gpt_cond_latent": gpt_cond_latent.cpu(),
        "speaker_embedding": speaker_embedding.cpu(),
        "text_tokens": text_tokens.cpu(),
        "gpt_codes": gpt_codes.cpu(),
        "gpt_latents": gpt_latents.cpu(),
        "wav": wav.cpu(),
    }


def debug_custom_codes_to_latents(model, text_tokens: torch.Tensor, gpt_codes: torch.Tensor, gpt_cond_latent: torch.Tensor):
    """
    Debug custom implementation of codes to latents (like in xtts_trt_backend).
    Compare with original GPT.forward() with return_latent=True.
    """
    logger.info("=" * 60)
    logger.info("CUSTOM CODES TO LATENTS (current xtts_trt_backend)")
    logger.info("=" * 60)

    device = model.device
    gpt = model.gpt

    # Move to device
    text_tokens = text_tokens.to(device)
    gpt_codes = gpt_codes.to(device)
    gpt_cond_latent = gpt_cond_latent.to(device)

    # Current implementation in xtts_trt_backend._codes_to_latents (BROKEN)
    logger.info("Current (broken) implementation:")
    with torch.no_grad():
        # This is what xtts_trt_backend currently does (WRONG):
        text_emb_broken = gpt.text_embedding(text_tokens)  # NO POSITION!
        mel_emb_broken = gpt.mel_embedding(gpt_codes)       # NO POSITION!

        if gpt_cond_latent.shape[1] == 1024:
            cond_for_concat = gpt_cond_latent.transpose(1, 2)
        else:
            cond_for_concat = gpt_cond_latent

        emb_broken = torch.cat([cond_for_concat, text_emb_broken, mel_emb_broken], dim=1)
        logger.info(f"emb_broken shape: {emb_broken.shape}")

        gpt_out_broken = gpt.gpt(inputs_embeds=emb_broken, return_dict=True)
        hidden_broken = gpt_out_broken.last_hidden_state
        hidden_broken = gpt.final_norm(hidden_broken)

        cond_len = cond_for_concat.shape[1]
        text_len = text_tokens.shape[1]
        offset = cond_len + text_len

        mel_latents_broken = hidden_broken[:, offset:, :]
        logger.info(f"mel_latents_broken shape: {mel_latents_broken.shape}")
        logger.info(f"mel_latents_broken mean: {mel_latents_broken.mean().item():.4f}")
        logger.info(f"mel_latents_broken std: {mel_latents_broken.std().item():.4f}")

    # Correct implementation (matching GPT.forward with return_latent=True)
    logger.info("\nCorrect implementation (with position embeddings):")
    with torch.no_grad():
        # Like GPT.forward() lines 496-500
        # Pad text with start/stop tokens first
        text_inputs = F.pad(text_tokens, (0, 1), value=gpt.stop_text_token)
        text_inputs = F.pad(text_inputs, (1, 0), value=gpt.start_text_token)

        # Pad mel codes with start/stop tokens
        audio_codes = F.pad(gpt_codes, (1, 0), value=gpt.start_audio_token)
        audio_codes = F.pad(audio_codes, (0, 1), value=gpt.stop_audio_token)

        logger.info(f"text_inputs (padded): {text_inputs.shape}")
        logger.info(f"audio_codes (padded): {audio_codes.shape}")

        # Get embeddings WITH position embeddings
        text_emb = gpt.text_embedding(text_inputs) + gpt.text_pos_embedding(text_inputs)
        mel_emb = gpt.mel_embedding(audio_codes) + gpt.mel_pos_embedding(audio_codes)

        logger.info(f"text_emb (with pos) shape: {text_emb.shape}")
        logger.info(f"mel_emb (with pos) shape: {mel_emb.shape}")

        # Concat with prompt
        if gpt_cond_latent.shape[1] == 1024:
            cond_for_concat = gpt_cond_latent.transpose(1, 2)
        else:
            cond_for_concat = gpt_cond_latent

        emb_correct = torch.cat([cond_for_concat, text_emb, mel_emb], dim=1)
        logger.info(f"emb_correct shape: {emb_correct.shape}")

        # Forward
        gpt_out_correct = gpt.gpt(inputs_embeds=emb_correct, return_dict=True)
        hidden_correct = gpt_out_correct.last_hidden_state

        # Skip prompt (conditioning)
        offset = cond_for_concat.shape[1]
        enc = hidden_correct[:, offset:]
        enc = gpt.final_norm(enc)

        # Get mel part (after text)
        # Like GPT.get_logits with return_latent=True
        text_part = enc[:, :text_emb.shape[1]]
        mel_part = enc[:, -mel_emb.shape[1]:]

        # Apply sub=-5 like original
        sub = -5
        mel_latents_correct = mel_part[:, :sub]

        logger.info(f"mel_latents_correct shape: {mel_latents_correct.shape}")
        logger.info(f"mel_latents_correct mean: {mel_latents_correct.mean().item():.4f}")
        logger.info(f"mel_latents_correct std: {mel_latents_correct.std().item():.4f}")

    return {
        "broken": mel_latents_broken.cpu(),
        "correct": mel_latents_correct.cpu(),
    }


def debug_gpt_forward_internals(model, text_tokens, gpt_codes, gpt_cond_latent):
    """
    Step through GPT.forward() to understand exact shapes and operations.
    """
    logger.info("=" * 60)
    logger.info("GPT.forward() INTERNALS")
    logger.info("=" * 60)

    device = model.device
    gpt = model.gpt

    text_tokens = text_tokens.to(device)
    gpt_codes = gpt_codes.to(device)
    gpt_cond_latent = gpt_cond_latent.to(device)

    with torch.no_grad():
        text_lengths = torch.tensor([text_tokens.shape[-1]], device=device)
        wav_lengths = torch.tensor([gpt_codes.shape[-1] * gpt.code_stride_len], device=device)

        logger.info(f"Input shapes:")
        logger.info(f"  text_tokens: {text_tokens.shape}")
        logger.info(f"  gpt_codes: {gpt_codes.shape}")
        logger.info(f"  gpt_cond_latent: {gpt_cond_latent.shape}")
        logger.info(f"  text_lengths: {text_lengths}")
        logger.info(f"  wav_lengths: {wav_lengths}")

        # Compute code_lengths like GPT.forward
        code_lengths = torch.ceil(wav_lengths / gpt.code_stride_len).long() + 3
        logger.info(f"  code_lengths: {code_lengths}")

        max_text_len = text_lengths.max().item()
        max_mel_len = code_lengths.max().item()
        logger.info(f"  max_text_len: {max_text_len}")
        logger.info(f"  max_mel_len: {max_mel_len}")

        # Pad if needed
        audio_codes = gpt_codes
        if max_mel_len > audio_codes.shape[-1]:
            audio_codes = F.pad(audio_codes, (0, max_mel_len - audio_codes.shape[-1]))
            logger.info(f"  audio_codes after pad: {audio_codes.shape}")

        # Add stop tokens
        text_inputs = F.pad(text_tokens[:, :max_text_len], (0, 1), value=gpt.stop_text_token)
        audio_codes = F.pad(audio_codes[:, :max_mel_len], (0, 1), value=gpt.stop_audio_token)

        logger.info(f"  text_inputs after stop: {text_inputs.shape}")
        logger.info(f"  audio_codes after stop: {audio_codes.shape}")

        # Build input and target tensors (add start token)
        text_inputs = F.pad(text_inputs, (1, 0), value=gpt.start_text_token)
        audio_codes = F.pad(audio_codes, (1, 0), value=gpt.start_audio_token)

        logger.info(f"  text_inputs after start: {text_inputs.shape}")
        logger.info(f"  audio_codes after start: {audio_codes.shape}")

        # Get embeddings with position
        text_emb = gpt.text_embedding(text_inputs) + gpt.text_pos_embedding(text_inputs)
        mel_emb = gpt.mel_embedding(audio_codes) + gpt.mel_pos_embedding(audio_codes)

        logger.info(f"  text_emb shape: {text_emb.shape}")
        logger.info(f"  mel_emb shape: {mel_emb.shape}")

        # Get style emb (cond_latents)
        # In return_latent mode, cond_latents is already computed
        if gpt_cond_latent.shape[1] == 1024:
            cond_latents = gpt_cond_latent.transpose(1, 2)
        else:
            cond_latents = gpt_cond_latent
        logger.info(f"  cond_latents shape: {cond_latents.shape}")

        # Build full embedding: [prompt, text, mel]
        prompt_offset = cond_latents.shape[1]
        emb = torch.cat([cond_latents, text_emb, mel_emb], dim=1)
        logger.info(f"  full emb shape: {emb.shape}")
        logger.info(f"  prompt_offset: {prompt_offset}")

        # Forward
        gpt_out = gpt.gpt(inputs_embeds=emb, return_dict=True)
        hidden_states = gpt_out.last_hidden_state

        logger.info(f"  hidden_states shape: {hidden_states.shape}")

        # Skip prompt in output
        enc = hidden_states[:, prompt_offset:]
        enc = gpt.final_norm(enc)

        logger.info(f"  enc (after skip prompt): {enc.shape}")

        # In return_latent mode, get_logits returns:
        # text_part = enc[:, :text_emb.shape[1]]
        # mel_part = enc[:, -mel_emb.shape[1]:]
        # return mel_part[:, :sub]

        text_part = enc[:, :text_emb.shape[1]]
        mel_part = enc[:, -mel_emb.shape[1]:]

        logger.info(f"  text_part shape: {text_part.shape}")
        logger.info(f"  mel_part shape: {mel_part.shape}")

        sub = -5
        mel_latents = mel_part[:, :sub]
        logger.info(f"  mel_latents (final, sub={sub}): {mel_latents.shape}")

        return mel_latents.cpu()


def main():
    parser = argparse.ArgumentParser(description="Debug XTTS inference")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--speaker-wav", type=str, required=True)
    parser.add_argument("--text", type=str, default="Привет мир")
    parser.add_argument("--language", type=str, default="ru")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-audio", action="store_true")

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading XTTS from {args.checkpoint_dir}")
    model = load_xtts_model(args.checkpoint_dir, args.device)

    # Run original inference with debugging
    results = debug_original_inference(model, args.text, args.speaker_wav, args.language)

    # Compare custom implementation
    custom_results = debug_custom_codes_to_latents(
        model,
        results["text_tokens"],
        results["gpt_codes"],
        results["gpt_cond_latent"]
    )

    # Step through GPT.forward internals
    manual_latents = debug_gpt_forward_internals(
        model,
        results["text_tokens"],
        results["gpt_codes"],
        results["gpt_cond_latent"]
    )

    # Compare results
    logger.info("=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)

    original = results["gpt_latents"]
    broken = custom_results["broken"]
    correct = custom_results["correct"]

    logger.info(f"Original latents: {original.shape}")
    logger.info(f"Broken latents: {broken.shape}")
    logger.info(f"Correct latents: {correct.shape}")
    logger.info(f"Manual latents: {manual_latents.shape}")

    # Check if shapes match for comparison
    min_len = min(original.shape[1], correct.shape[1], manual_latents.shape[1])

    if min_len > 0:
        orig_trim = original[:, :min_len, :]
        correct_trim = correct[:, :min_len, :]
        manual_trim = manual_latents[:, :min_len, :]

        diff_correct = (orig_trim - correct_trim).abs()
        diff_manual = (orig_trim - manual_trim).abs()

        logger.info(f"\nDifference (original vs correct): max={diff_correct.max():.6f}, mean={diff_correct.mean():.6f}")
        logger.info(f"Difference (original vs manual): max={diff_manual.max():.6f}, mean={diff_manual.mean():.6f}")

        if diff_correct.max() < 0.01:
            logger.info("✓ Correct implementation matches original!")
        else:
            logger.warning("✗ Correct implementation DOES NOT match original")

        if diff_manual.max() < 0.01:
            logger.info("✓ Manual implementation matches original!")
        else:
            logger.warning("✗ Manual implementation DOES NOT match original")

    if args.save_audio:
        import soundfile as sf
        wav = results["wav"].squeeze().numpy()
        sf.write("debug_original.wav", wav, 24000)
        logger.info("Saved debug_original.wav")


if __name__ == "__main__":
    main()
