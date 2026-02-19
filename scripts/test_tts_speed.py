#!/usr/bin/env python3
"""Test TTS speed to determine if real-time streaming is feasible."""

import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def test_tts_speed():
    """Test TTS synthesis speed and calculate RTF."""
    from wiwi.modules.tts.backends.xtts import XTTSBackend

    # Test config
    config = {
        "model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "device": "cuda:0",
        "language": "ru",
        "reference_audio": "models/tts/reference.wav",
        "use_deepspeed": True,
        "use_compile": True,  # Test without first
        "stream_chunk_size": 20,
        "temperature": 0.75,
        "top_k": 50,
        "top_p": 0.85,
    }

    print("=" * 60)
    print("TTS Speed Test")
    print("=" * 60)
    print(f"DeepSpeed: {config['use_deepspeed']}")
    print(f"torch.compile: {config['use_compile']}")
    print(f"Device: {config['device']}")
    print()

    # Load model
    print("Loading model...")
    start = time.time()
    backend = XTTSBackend(config)
    backend.load_model()
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    print()

    # Test sentences (Russian)
    test_texts = [
        "Привет, как дела?",
        "Это тест скорости синтеза речи.",
        "Сегодня хорошая погода, не правда ли?",
        "Искусственный интеллект развивается очень быстро в последние годы.",
    ]

    sample_rate = backend.sample_rate
    results = []

    print("Running synthesis tests...")
    print("-" * 60)

    # Warmup
    print("Warmup run...")
    _ = list(backend.synthesize_stream("Тест"))

    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: \"{text[:40]}...\"" if len(text) > 40 else f"\nTest {i+1}: \"{text}\"")

        # Measure streaming synthesis
        start = time.time()
        chunks = []
        first_chunk_time = None

        for chunk in backend.synthesize_stream(text):
            if first_chunk_time is None:
                first_chunk_time = time.time() - start
            chunks.append(chunk)

        total_time = time.time() - start

        # Calculate audio duration
        audio = np.concatenate(chunks)
        audio_duration = len(audio) / sample_rate

        # Real-time factor (RTF): time to generate / audio duration
        # RTF < 1.0 means faster than real-time
        rtf = total_time / audio_duration

        results.append({
            "text": text,
            "text_len": len(text),
            "audio_duration": audio_duration,
            "synthesis_time": total_time,
            "first_chunk_time": first_chunk_time,
            "rtf": rtf,
            "num_chunks": len(chunks),
        })

        print(f"  Text length: {len(text)} chars")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Synthesis time: {total_time:.2f}s")
        print(f"  First chunk: {first_chunk_time*1000:.0f}ms")
        print(f"  RTF: {rtf:.3f} {'✓ Real-time OK' if rtf < 1.0 else '✗ Too slow'}")
        print(f"  Chunks: {len(chunks)}")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    avg_rtf = np.mean([r["rtf"] for r in results])
    avg_first_chunk = np.mean([r["first_chunk_time"] for r in results]) * 1000

    print(f"Average RTF: {avg_rtf:.3f}")
    print(f"Average first chunk latency: {avg_first_chunk:.0f}ms")
    print()

    if avg_rtf < 0.5:
        print("✓ EXCELLENT: RTF < 0.5 - True streaming is feasible!")
        print("  Audio is generated 2x+ faster than real-time.")
        return True
    elif avg_rtf < 1.0:
        print("✓ GOOD: RTF < 1.0 - Real-time streaming possible with buffering.")
        return True
    else:
        print("✗ TOO SLOW: RTF >= 1.0 - Keep pseudo-streaming (batch by sentence).")
        return False


if __name__ == "__main__":
    result = test_tts_speed()
    sys.exit(0 if result else 1)
