#!/usr/bin/env python3
"""Test script for Chatterbox TTS RTF measurement."""

import time
import sys

def main():
    print("=" * 60)
    print("Chatterbox TTS RTF Test")
    print("=" * 60)

    # Test text (~100 characters)
    test_text = "Привет! Это тестовое предложение для проверки скорости синтеза речи на русском языке."
    print(f"\nТекст ({len(test_text)} символов):")
    print(f"  \"{test_text}\"")
    print()

    # Import and initialize
    print("Загрузка модели...")
    load_start = time.perf_counter()

    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        import torch
    except ImportError as e:
        print(f"ОШИБКА: {e}")
        print("Установите: pip install chatterbox-tts")
        sys.exit(1)

    device = "cuda:0"
    print(f"Device: {device}")

    if device.startswith("cuda"):
        device_idx = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(device_idx)
        print(f"CUDA device: {torch.cuda.get_device_name(device_idx)}")
        print(f"VRAM before load: {torch.cuda.memory_allocated(device_idx) / 1e9:.2f} GB")

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    load_time = time.perf_counter() - load_start

    print(f"Модель загружена за {load_time:.2f}s")

    if device.startswith("cuda"):
        print(f"VRAM after load: {torch.cuda.memory_allocated(device_idx) / 1e9:.2f} GB")

    sample_rate = model.sr
    print(f"Sample rate: {sample_rate} Hz")
    print()

    # Warmup
    print("Прогрев модели...")
    warmup_start = time.perf_counter()
    _ = model.generate("Тест.", language_id="ru")
    warmup_time = time.perf_counter() - warmup_start
    print(f"Прогрев завершён за {warmup_time:.2f}s")
    print()

    # Test synthesis
    print("=" * 60)
    print("ТЕСТ СИНТЕЗА")
    print("=" * 60)

    # Batch synthesis
    print("\n1. Batch синтез (model.generate):")
    batch_start = time.perf_counter()
    audio = model.generate(test_text, language_id="ru")
    batch_time = time.perf_counter() - batch_start

    audio_samples = audio.shape[-1]
    audio_duration = audio_samples / sample_rate
    batch_rtf = batch_time / audio_duration

    print(f"   Время синтеза: {batch_time:.3f}s")
    print(f"   Длина аудио: {audio_duration:.3f}s ({audio_samples} samples)")
    print(f"   RTF: {batch_rtf:.3f} {'✓' if batch_rtf < 1.0 else '✗'}")

    # Streaming synthesis (if available)
    print("\n2. Streaming синтез (model.stream_generate):")

    import asyncio

    async def test_streaming():
        chunks = []
        chunk_times = []

        stream_start = time.perf_counter()
        first_chunk_time = None

        async for audio_chunk in model.stream_generate(
            test_text,
            language_id="ru",
            chunk_size=50,
        ):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - stream_start

            chunk_times.append(time.perf_counter() - stream_start)
            chunks.append(audio_chunk)

        stream_time = time.perf_counter() - stream_start
        return chunks, chunk_times, stream_time, first_chunk_time

    try:
        chunks, chunk_times, stream_time, ttfa = asyncio.run(test_streaming())

        total_samples = sum(c.shape[-1] if hasattr(c, 'shape') else len(c) for c in chunks)
        stream_audio_duration = total_samples / sample_rate
        stream_rtf = stream_time / stream_audio_duration

        print(f"   Количество чанков: {len(chunks)}")
        print(f"   TTFA (первый чанк): {ttfa * 1000:.1f}ms")
        print(f"   Время синтеза: {stream_time:.3f}s")
        print(f"   Длина аудио: {stream_audio_duration:.3f}s ({total_samples} samples)")
        print(f"   RTF: {stream_rtf:.3f} {'✓' if stream_rtf < 1.0 else '✗'}")

    except Exception as e:
        print(f"   Ошибка: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("ИТОГИ")
    print("=" * 60)
    print(f"RTF (batch): {batch_rtf:.3f}")
    print(f"RTF < 1.0 означает синтез быстрее реального времени")

    if device.startswith("cuda"):
        print(f"\nVRAM использовано: {torch.cuda.memory_allocated(device_idx) / 1e9:.2f} GB")
        print(f"VRAM зарезервировано: {torch.cuda.memory_reserved(device_idx) / 1e9:.2f} GB")

    print()

if __name__ == "__main__":
    main()
