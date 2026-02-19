#!/usr/bin/env python3
"""
Benchmark script for XTTS TTS backends.

Compares inference speed of:
1. Original XTTS (TTS library baseline)
2. PyTorch backend (our implementation)
3. PyTorch + DeepSpeed
4. Hybrid PyTorch + TensorRT (if available)

Metrics measured:
- Time to first audio (TTFA) - latency until first audio chunk
- Real-time factor (RTF) - audio_duration / generation_time
- Tokens per second
- Memory usage

Usage:
    python -m wiwi.modules.tts.backends.benchmark_xtts \
        --checkpoint-dir ~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2 \
        --speaker-wav models/tts/reference.wav \
        --iterations 5
"""

import argparse
import gc
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    backend_name: str
    text: str
    text_len: int
    audio_samples: int
    audio_duration_s: float
    generation_time_s: float
    rtf: float  # Real-time factor
    tokens_generated: int
    tokens_per_second: float
    gpu_memory_mb: float


@dataclass
class BenchmarkSummary:
    """Summary statistics for a backend."""
    backend_name: str
    num_runs: int
    avg_rtf: float
    std_rtf: float
    min_rtf: float
    max_rtf: float
    avg_tokens_per_second: float
    avg_generation_time_s: float
    peak_gpu_memory_mb: float
    results: List[BenchmarkResult] = field(default_factory=list)


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def clear_gpu_cache():
    """Clear GPU cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class OriginalXTTSBenchmark:
    """Benchmark wrapper for original TTS library XTTS."""

    def __init__(self, checkpoint_dir: str, speaker_wav: str, device: str = "cuda:0"):
        self.checkpoint_dir = checkpoint_dir
        self.speaker_wav = speaker_wav
        self.device = device
        self.model = None
        self.sample_rate = 24000

    def load(self):
        """Load model."""
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

        config_path = os.path.join(self.checkpoint_dir, "config.json")
        config = XttsConfig()
        config.load_json(config_path)

        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=self.checkpoint_dir, use_deepspeed=False)
        self.model.to(self.device)
        self.model.eval()

        # Pre-compute conditioning
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[self.speaker_wav]
        )

        logger.info("Original XTTS loaded")

    def unload(self):
        """Unload model."""
        del self.model
        self.model = None
        clear_gpu_cache()

    def synthesize(self, text: str, language: str = "ru") -> Tuple[np.ndarray, int]:
        """Synthesize text and return (audio, num_tokens)."""
        with torch.no_grad():
            result = self.model.inference(
                text=text,
                language=language,
                gpt_cond_latent=self.gpt_cond_latent,
                speaker_embedding=self.speaker_embedding,
                temperature=0.75,
                top_k=50,
                top_p=0.85,
                repetition_penalty=10.0,
            )

        wav = result["wav"]
        # Estimate tokens from audio length (rough approximation)
        # Each mel token produces ~1024 samples at 24kHz
        num_tokens = len(wav) // 1024

        return wav, num_tokens


class PyTorchBackendBenchmark:
    """Benchmark wrapper for our PyTorch backend."""

    def __init__(
        self,
        checkpoint_dir: str,
        speaker_wav: str,
        device: str = "cuda:0",
        use_deepspeed: bool = False,
        deepspeed_dtype: str = "fp16",
        trt_decoder_path: str = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.speaker_wav = speaker_wav
        self.device = device
        self.use_deepspeed = use_deepspeed
        self.deepspeed_dtype = deepspeed_dtype
        self.trt_decoder_path = trt_decoder_path
        self.backend = None
        self.sample_rate = 24000

    def load(self):
        """Load backend."""
        from wiwi.modules.tts.backends.xtts_trt_backend import XttsTrtBackend

        config = {
            "trt_gpt_decode_path": None,
            "trt_decoder_path": self.trt_decoder_path,
            "xtts_checkpoint_dir": self.checkpoint_dir,
            "speaker_ref_path": self.speaker_wav,
            "device": self.device,
            "language": "ru",
            "temperature": 0.75,
            "top_k": 50,
            "top_p": 0.85,
            "repetition_penalty": 10.0,
            "max_mel_tokens": 500,
            "use_deepspeed": self.use_deepspeed,
            "deepspeed_dtype": self.deepspeed_dtype,
        }

        self.backend = XttsTrtBackend(config)
        self.backend.load_model()

        mode = f"Backend loaded (mode: {self.backend._model})"
        if self.use_deepspeed:
            mode += " + DeepSpeed"
        logger.info(mode)

    def unload(self):
        """Unload backend."""
        if self.backend:
            self.backend.unload_model()
        self.backend = None
        clear_gpu_cache()

    def synthesize(self, text: str, language: str = "ru") -> Tuple[np.ndarray, int]:
        """Synthesize text and return (audio, num_tokens)."""
        wav = self.backend.synthesize(text, language)
        num_tokens = len(wav) // 1024
        return wav, num_tokens


class HybridTRTBenchmark:
    """Benchmark wrapper for hybrid PyTorch + TRT backend."""

    def __init__(
        self,
        checkpoint_dir: str,
        speaker_wav: str,
        trt_decoder_path: str,
        device: str = "cuda:0",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.speaker_wav = speaker_wav
        self.trt_decoder_path = trt_decoder_path
        self.device = device
        self.backend = None
        self.sample_rate = 24000

    def load(self):
        """Load backend."""
        from wiwi.modules.tts.backends.xtts_trt_backend import XttsTrtBackend

        config = {
            "trt_gpt_decode_path": None,
            "trt_decoder_path": self.trt_decoder_path,
            "xtts_checkpoint_dir": self.checkpoint_dir,
            "speaker_ref_path": self.speaker_wav,
            "device": self.device,
            "language": "ru",
            "temperature": 0.75,
            "top_k": 50,
            "top_p": 0.85,
            "repetition_penalty": 10.0,
            "max_mel_tokens": 500,
            "use_deepspeed": False,
        }

        self.backend = XttsTrtBackend(config)
        self.backend.load_model()

        logger.info("Hybrid PyTorch + TRT backend loaded")

    def unload(self):
        """Unload backend."""
        if self.backend:
            self.backend.unload_model()
        self.backend = None
        clear_gpu_cache()

    def synthesize(self, text: str, language: str = "ru") -> Tuple[np.ndarray, int]:
        """Synthesize text and return (audio, num_tokens)."""
        wav = self.backend.synthesize(text, language)
        num_tokens = len(wav) // 1024
        return wav, num_tokens


def run_benchmark(
    backend,
    backend_name: str,
    texts: List[str],
    language: str,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
) -> BenchmarkSummary:
    """Run benchmark for a single backend."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {backend_name}")
    logger.info(f"{'='*60}")

    results = []

    # Warmup
    logger.info(f"Warmup runs: {warmup_runs}")
    for i in range(warmup_runs):
        _ = backend.synthesize(texts[0], language)

    clear_gpu_cache()
    torch.cuda.synchronize()

    # Benchmark runs
    logger.info(f"Benchmark runs: {benchmark_runs} x {len(texts)} texts")

    for run_idx in range(benchmark_runs):
        for text in texts:
            # Measure GPU memory before
            clear_gpu_cache()
            torch.cuda.synchronize()

            # Generate
            start_time = time.perf_counter()
            wav, num_tokens = backend.synthesize(text, language)
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            # Metrics
            generation_time = end_time - start_time
            audio_duration = len(wav) / backend.sample_rate
            rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
            tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0
            gpu_memory = get_gpu_memory_mb()

            result = BenchmarkResult(
                backend_name=backend_name,
                text=text[:50] + "..." if len(text) > 50 else text,
                text_len=len(text),
                audio_samples=len(wav),
                audio_duration_s=audio_duration,
                generation_time_s=generation_time,
                rtf=rtf,
                tokens_generated=num_tokens,
                tokens_per_second=tokens_per_second,
                gpu_memory_mb=gpu_memory,
            )
            results.append(result)

            logger.info(
                f"  Run {run_idx+1}: text_len={len(text):3d}, "
                f"audio={audio_duration:.2f}s, "
                f"gen={generation_time:.3f}s, "
                f"RTF={rtf:.3f}, "
                f"tok/s={tokens_per_second:.1f}"
            )

    # Summary statistics
    rtfs = [r.rtf for r in results]
    tps = [r.tokens_per_second for r in results]
    gen_times = [r.generation_time_s for r in results]
    memories = [r.gpu_memory_mb for r in results]

    summary = BenchmarkSummary(
        backend_name=backend_name,
        num_runs=len(results),
        avg_rtf=np.mean(rtfs),
        std_rtf=np.std(rtfs),
        min_rtf=np.min(rtfs),
        max_rtf=np.max(rtfs),
        avg_tokens_per_second=np.mean(tps),
        avg_generation_time_s=np.mean(gen_times),
        peak_gpu_memory_mb=np.max(memories),
        results=results,
    )

    logger.info(f"\n{backend_name} Summary:")
    logger.info(f"  Avg RTF: {summary.avg_rtf:.3f} ± {summary.std_rtf:.3f}")
    logger.info(f"  Min/Max RTF: {summary.min_rtf:.3f} / {summary.max_rtf:.3f}")
    logger.info(f"  Avg tokens/s: {summary.avg_tokens_per_second:.1f}")
    logger.info(f"  Peak GPU memory: {summary.peak_gpu_memory_mb:.1f} MB")

    return summary


def print_comparison(summaries: List[BenchmarkSummary]):
    """Print comparison table."""
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARK COMPARISON")
    logger.info(f"{'='*80}")

    # Header
    logger.info(f"{'Backend':<30} {'RTF':>8} {'±':>6} {'tok/s':>8} {'GPU MB':>8} {'Realtime':>10}")
    logger.info("-" * 80)

    # Data rows
    for s in summaries:
        realtime = "YES ✓" if s.avg_rtf < 1.0 else "NO ✗"
        logger.info(
            f"{s.backend_name:<30} "
            f"{s.avg_rtf:>8.3f} "
            f"{s.std_rtf:>6.3f} "
            f"{s.avg_tokens_per_second:>8.1f} "
            f"{s.peak_gpu_memory_mb:>8.0f} "
            f"{realtime:>10}"
        )

    logger.info("-" * 80)

    # Find fastest
    if summaries:
        fastest = min(summaries, key=lambda s: s.avg_rtf)
        logger.info(f"\nFastest: {fastest.backend_name} (RTF={fastest.avg_rtf:.3f})")

        # Speedup comparison
        if len(summaries) > 1:
            baseline = summaries[0]
            logger.info(f"\nSpeedup vs {baseline.backend_name}:")
            for s in summaries[1:]:
                speedup = baseline.avg_rtf / s.avg_rtf
                logger.info(f"  {s.backend_name}: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark XTTS backends")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Path to XTTS checkpoint directory")
    parser.add_argument("--speaker-wav", type=str, required=True,
                        help="Path to speaker reference audio")
    parser.add_argument("--trt-decoder", type=str, default=None,
                        help="Path to TensorRT HiFi-GAN decoder engine")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--language", type=str, default="ru",
                        help="Language code")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Number of warmup runs")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of benchmark iterations")
    parser.add_argument("--skip-original", action="store_true",
                        help="Skip original XTTS benchmark")
    parser.add_argument("--skip-deepspeed", action="store_true",
                        help="Skip DeepSpeed benchmark")
    parser.add_argument("--skip-trt", action="store_true",
                        help="Skip TensorRT benchmark")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (JSON)")

    args = parser.parse_args()

    # Test texts of varying lengths
    test_texts = [
        # Short (5-10 words)
        "Привет, как дела?",
        # Medium (15-25 words)
        "Сегодня отличная погода для прогулки в парке, не так ли? Давай встретимся у фонтана.",
        # Long (40+ words)
        "Искусственный интеллект продолжает развиваться с невероятной скоростью. "
        "Современные модели способны генерировать текст, изображения и даже музыку. "
        "Это открывает новые возможности для творчества и автоматизации рутинных задач.",
    ]

    summaries = []

    # 1. Original XTTS baseline
    if not args.skip_original:
        try:
            backend = OriginalXTTSBenchmark(
                checkpoint_dir=args.checkpoint_dir,
                speaker_wav=args.speaker_wav,
                device=args.device,
            )
            backend.load()

            summary = run_benchmark(
                backend=backend,
                backend_name="Original XTTS",
                texts=test_texts,
                language=args.language,
                warmup_runs=args.warmup,
                benchmark_runs=args.iterations,
            )
            summaries.append(summary)

            backend.unload()
        except Exception as e:
            logger.error(f"Original XTTS benchmark failed: {e}")

    # 2. PyTorch backend (our implementation) + TRT HiFi-GAN if available
    try:
        backend = PyTorchBackendBenchmark(
            checkpoint_dir=args.checkpoint_dir,
            speaker_wav=args.speaker_wav,
            device=args.device,
            use_deepspeed=False,
            trt_decoder_path=args.trt_decoder,
        )
        backend.load()

        summary = run_benchmark(
            backend=backend,
            backend_name=f"PyTorch GPT + {'TRT' if args.trt_decoder else 'PyTorch'} HiFi-GAN",
            texts=test_texts,
            language=args.language,
            warmup_runs=args.warmup,
            benchmark_runs=args.iterations,
        )
        summaries.append(summary)

        backend.unload()
    except Exception as e:
        logger.error(f"PyTorch backend benchmark failed: {e}")

    # 3. PyTorch + DeepSpeed + TRT HiFi-GAN if available
    if not args.skip_deepspeed:
        try:
            backend = PyTorchBackendBenchmark(
                checkpoint_dir=args.checkpoint_dir,
                speaker_wav=args.speaker_wav,
                device=args.device,
                use_deepspeed=True,
                deepspeed_dtype="fp16",
                trt_decoder_path=args.trt_decoder,
            )
            backend.load()

            summary = run_benchmark(
                backend=backend,
                backend_name=f"DeepSpeed GPT + {'TRT' if args.trt_decoder else 'PyTorch'} HiFi-GAN",
                texts=test_texts,
                language=args.language,
                warmup_runs=args.warmup,
                benchmark_runs=args.iterations,
            )
            summaries.append(summary)

            backend.unload()
        except Exception as e:
            logger.error(f"DeepSpeed benchmark failed: {e}")

    # Print comparison
    if summaries:
        print_comparison(summaries)

        # Save results
        if args.output:
            import json

            output_data = {
                "config": {
                    "checkpoint_dir": args.checkpoint_dir,
                    "speaker_wav": args.speaker_wav,
                    "device": args.device,
                    "language": args.language,
                    "warmup_runs": args.warmup,
                    "benchmark_runs": args.iterations,
                },
                "summaries": [
                    {
                        "backend_name": s.backend_name,
                        "num_runs": s.num_runs,
                        "avg_rtf": s.avg_rtf,
                        "std_rtf": s.std_rtf,
                        "min_rtf": s.min_rtf,
                        "max_rtf": s.max_rtf,
                        "avg_tokens_per_second": s.avg_tokens_per_second,
                        "avg_generation_time_s": s.avg_generation_time_s,
                        "peak_gpu_memory_mb": s.peak_gpu_memory_mb,
                    }
                    for s in summaries
                ],
            }

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"\nResults saved to: {args.output}")
    else:
        logger.warning("No benchmarks completed successfully")


if __name__ == "__main__":
    main()
