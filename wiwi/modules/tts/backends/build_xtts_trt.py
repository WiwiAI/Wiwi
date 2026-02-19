#!/usr/bin/env python3
"""
Build TensorRT engines from XTTS ONNX files.

This script converts ONNX models to optimized TensorRT engines with:
- Dynamic shape support for variable sequence lengths
- FP16 precision for faster inference
- Optimized memory workspace

Usage:
    # Build GPT engine
    python build_xtts_trt.py --onnx models/onnx/gpt.onnx --type gpt --fp16

    # Build HiFi-GAN decoder engine
    python build_xtts_trt.py --onnx models/onnx/hifigan_decoder.onnx --type decoder --fp16

    # Build all engines from directory
    python build_xtts_trt.py --onnx-dir models/onnx --out models/trt --fp16

Requirements:
    - TensorRT 8.x or 10.x
    - CUDA 12.x
    - trtexec (comes with TensorRT)
"""

import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# XTTS Model Specifications
# ============================================================================

# GPT model dimensions (XTTS v2)
GPT_NUM_LAYERS = 30
GPT_NUM_HEADS = 16
GPT_HIDDEN_SIZE = 1024
GPT_HEAD_DIM = GPT_HIDDEN_SIZE // GPT_NUM_HEADS  # 64
GPT_COND_LEN = 32  # Fixed conditioning length
GPT_NUM_AUDIO_TOKENS = 1026  # mel vocab: 0-1023 codes + 1024 start + 1025 stop

# Sequence length profiles for dynamic shapes
# Format: (min, optimal, max)
SEQ_LEN_PROFILES = {
    "min": 1,
    "opt": 256,
    "max": 1024,
}

PAST_SEQ_PROFILES = {
    "min": 0,
    "opt": 256,
    "max": 1024,
}

# HiFi-GAN decoder time steps
DECODER_TIME_PROFILES = {
    "min": 1,
    "opt": 200,
    "max": 1000,
}


def build_dynamic_shapes_gpt_prefix() -> Dict[str, str]:
    """
    Build dynamic shape specifications for GPT PREFIX model (first step).

    PREFIX model inputs:
        - input_ids: [batch, seq_len] - text tokens
        - gpt_cond_latent: [batch, 32, 1024] - conditioning

    No past KV cache inputs - outputs fresh cache.
    """
    shapes = {}
    batch = 1

    # input_ids: [batch, seq_len] - text tokens (1-256 typical)
    shapes["input_ids"] = {
        "min": f"{batch}x{SEQ_LEN_PROFILES['min']}",
        "opt": f"{batch}x32",  # Typical text length
        "max": f"{batch}x{SEQ_LEN_PROFILES['max']}",
    }

    # gpt_cond_latent: [batch, 32, 1024] - fixed shape
    shapes["gpt_cond_latent"] = {
        "min": f"{batch}x{GPT_COND_LEN}x{GPT_HIDDEN_SIZE}",
        "opt": f"{batch}x{GPT_COND_LEN}x{GPT_HIDDEN_SIZE}",
        "max": f"{batch}x{GPT_COND_LEN}x{GPT_HIDDEN_SIZE}",
    }

    return shapes


def build_dynamic_shapes_gpt_decode() -> Dict[str, str]:
    """
    Build dynamic shape specifications for GPT DECODE model (subsequent steps).

    DECODE model inputs:
        - input_ids: [batch, 1] - single MEL token (0-1023)
        - mel_position: [batch, 1] - position index for mel_pos_embedding
        - attention_mask: [batch, past_len + 1]
        - past_key/value: [batch, num_heads, past_len, head_dim]
    """
    shapes = {}
    batch = 1

    # input_ids: [batch, 1] - always single token
    shapes["input_ids"] = {
        "min": f"{batch}x1",
        "opt": f"{batch}x1",
        "max": f"{batch}x1",
    }

    # mel_position: [batch, 1] - position for mel_pos_embedding (0 to max_mel_tokens)
    # Position 0 = start_audio_token (in prefix), so decode starts at 1
    shapes["mel_position"] = {
        "min": f"{batch}x1",
        "opt": f"{batch}x1",
        "max": f"{batch}x1",
    }

    # attention_mask: [batch, past_len + 1]
    # past_len ranges from cond(32)+text(~8)+1 to cond(32)+text+500 tokens
    min_past = GPT_COND_LEN + 3 + 1  # Minimum: cond + text(1+start+stop) + start_audio
    opt_past = GPT_COND_LEN + 10 + 64  # Typical
    max_past = GPT_COND_LEN + PAST_SEQ_PROFILES["max"]

    shapes["attention_mask"] = {
        "min": f"{batch}x{min_past + 1}",
        "opt": f"{batch}x{opt_past + 1}",
        "max": f"{batch}x{max_past + 1}",
    }

    # Past KV cache for each layer
    for i in range(GPT_NUM_LAYERS):
        shapes[f"past_key.{i}"] = {
            "min": f"{batch}x{GPT_NUM_HEADS}x{min_past}x{GPT_HEAD_DIM}",
            "opt": f"{batch}x{GPT_NUM_HEADS}x{opt_past}x{GPT_HEAD_DIM}",
            "max": f"{batch}x{GPT_NUM_HEADS}x{max_past}x{GPT_HEAD_DIM}",
        }
        shapes[f"past_value.{i}"] = {
            "min": f"{batch}x{GPT_NUM_HEADS}x{min_past}x{GPT_HEAD_DIM}",
            "opt": f"{batch}x{GPT_NUM_HEADS}x{opt_past}x{GPT_HEAD_DIM}",
            "max": f"{batch}x{GPT_NUM_HEADS}x{max_past}x{GPT_HEAD_DIM}",
        }

    return shapes


def build_dynamic_shapes_gpt() -> Dict[str, str]:
    """
    Build dynamic shape specifications for legacy single GPT model.
    DEPRECATED: Use build_dynamic_shapes_gpt_prefix/decode instead.
    """
    shapes = {}
    batch = 1

    shapes["input_ids"] = {
        "min": f"{batch}x{SEQ_LEN_PROFILES['min']}",
        "opt": f"{batch}x{SEQ_LEN_PROFILES['opt']}",
        "max": f"{batch}x{SEQ_LEN_PROFILES['max']}",
    }

    shapes["gpt_cond_latent"] = {
        "min": f"{batch}x{GPT_COND_LEN}x{GPT_HIDDEN_SIZE}",
        "opt": f"{batch}x{GPT_COND_LEN}x{GPT_HIDDEN_SIZE}",
        "max": f"{batch}x{GPT_COND_LEN}x{GPT_HIDDEN_SIZE}",
    }

    min_total = GPT_COND_LEN + SEQ_LEN_PROFILES["min"] + PAST_SEQ_PROFILES["min"]
    opt_total = GPT_COND_LEN + SEQ_LEN_PROFILES["opt"] + PAST_SEQ_PROFILES["opt"]
    max_total = GPT_COND_LEN + SEQ_LEN_PROFILES["max"] + PAST_SEQ_PROFILES["max"]

    shapes["attention_mask"] = {
        "min": f"{batch}x{min_total}",
        "opt": f"{batch}x{opt_total}",
        "max": f"{batch}x{max_total}",
    }

    for i in range(GPT_NUM_LAYERS):
        shapes[f"past_key.{i}"] = {
            "min": f"{batch}x{GPT_NUM_HEADS}x{PAST_SEQ_PROFILES['min']}x{GPT_HEAD_DIM}",
            "opt": f"{batch}x{GPT_NUM_HEADS}x{PAST_SEQ_PROFILES['opt']}x{GPT_HEAD_DIM}",
            "max": f"{batch}x{GPT_NUM_HEADS}x{PAST_SEQ_PROFILES['max']}x{GPT_HEAD_DIM}",
        }
        shapes[f"past_value.{i}"] = {
            "min": f"{batch}x{GPT_NUM_HEADS}x{PAST_SEQ_PROFILES['min']}x{GPT_HEAD_DIM}",
            "opt": f"{batch}x{GPT_NUM_HEADS}x{PAST_SEQ_PROFILES['opt']}x{GPT_HEAD_DIM}",
            "max": f"{batch}x{GPT_NUM_HEADS}x{PAST_SEQ_PROFILES['max']}x{GPT_HEAD_DIM}",
        }

    return shapes


def build_dynamic_shapes_decoder() -> Dict[str, str]:
    """
    Build dynamic shape specifications for HiFi-GAN decoder.

    Returns:
        Dictionary mapping input names to shape strings for trtexec
    """
    batch = 1
    latent_dim = 1024
    speaker_dim = 512

    shapes = {
        # latent_codes: [batch, time_steps, latent_dim] - note: time is dim 1!
        "latent_codes": {
            "min": f"{batch}x{DECODER_TIME_PROFILES['min']}x{latent_dim}",
            "opt": f"{batch}x{DECODER_TIME_PROFILES['opt']}x{latent_dim}",
            "max": f"{batch}x{DECODER_TIME_PROFILES['max']}x{latent_dim}",
        },
        # speaker_embedding: [batch, 512, 1] - 3D tensor!
        "speaker_embedding": {
            "min": f"{batch}x{speaker_dim}x1",
            "opt": f"{batch}x{speaker_dim}x1",
            "max": f"{batch}x{speaker_dim}x1",
        },
    }

    return shapes


def format_shapes_for_trtexec(shapes: Dict[str, Dict[str, str]]) -> List[str]:
    """
    Format shape specifications for trtexec command line.

    Args:
        shapes: Dictionary of {input_name: {min/opt/max: shape_string}}

    Returns:
        List of trtexec arguments for shapes
    """
    args = []

    # Build min/opt/max shape strings
    min_shapes = []
    opt_shapes = []
    max_shapes = []

    for name, profile in shapes.items():
        min_shapes.append(f"{name}:{profile['min']}")
        opt_shapes.append(f"{name}:{profile['opt']}")
        max_shapes.append(f"{name}:{profile['max']}")

    args.append(f"--minShapes={','.join(min_shapes)}")
    args.append(f"--optShapes={','.join(opt_shapes)}")
    args.append(f"--maxShapes={','.join(max_shapes)}")

    return args


def build_engine(
    onnx_path: Path,
    outdir: Path,
    model_type: str = "gpt",
    fp16: bool = True,
    bf16: bool = False,
    int8: bool = False,
    workspace_mb: int = 8192,
    verbose: bool = False,
    builder_optimization_level: int = 3,
) -> Path:
    """
    Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX file
        outdir: Output directory for TRT engine
        model_type: "gpt" or "decoder"
        fp16: Enable FP16 precision
        bf16: Enable BF16 precision (requires Ampere+)
        int8: Enable INT8 precision (requires calibration)
        workspace_mb: GPU memory workspace in MB
        verbose: Enable verbose output
        builder_optimization_level: TRT optimization level (0-5)

    Returns:
        Path to built TRT engine
    """
    outdir.mkdir(parents=True, exist_ok=True)
    engine_path = outdir / (onnx_path.stem + ".trt")

    # Get dynamic shapes based on model type
    if model_type == "gpt":
        shapes = build_dynamic_shapes_gpt()
    elif model_type == "gpt_prefix":
        shapes = build_dynamic_shapes_gpt_prefix()
    elif model_type == "gpt_decode":
        shapes = build_dynamic_shapes_gpt_decode()
    elif model_type == "decoder":
        shapes = build_dynamic_shapes_decoder()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'gpt', 'gpt_prefix', 'gpt_decode', or 'decoder'")

    # Build trtexec command
    cmd = [
        "trtexec",
        f"--onnx={str(onnx_path)}",
        f"--saveEngine={str(engine_path)}",
        f"--workspace={workspace_mb}",
        f"--builderOptimizationLevel={builder_optimization_level}",
        "--buildOnly",
    ]

    # Add dynamic shape arguments
    cmd.extend(format_shapes_for_trtexec(shapes))

    # Precision flags
    if fp16:
        cmd.append("--fp16")
    if bf16:
        cmd.append("--bf16")
    if int8:
        cmd.append("--int8")
        logger.warning("INT8 requires calibration data for best results")

    if verbose:
        cmd.append("--verbose")

    # Additional optimizations
    cmd.append("--useCudaGraph")  # Enable CUDA graphs for faster execution
    cmd.append("--noTF32")  # Disable TF32 for deterministic results (optional)

    logger.info(f"Building TensorRT engine for {model_type}...")
    logger.info(f"ONNX: {onnx_path}")
    logger.info(f"Output: {engine_path}")
    logger.info(f"Precision: FP16={fp16}, BF16={bf16}, INT8={int8}")
    logger.info(f"Workspace: {workspace_mb}MB")

    if verbose:
        logger.info(f"Command: {' '.join(cmd)}")

    # Run trtexec
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

        if verbose or result.returncode != 0:
            print(result.stdout)

        if result.returncode != 0:
            raise RuntimeError(f"trtexec failed with code {result.returncode}")

        logger.info(f"Successfully built engine: {engine_path}")
        return engine_path

    except FileNotFoundError:
        raise RuntimeError(
            "trtexec not found. Make sure TensorRT is installed and trtexec is in PATH.\n"
            "Install TensorRT: https://developer.nvidia.com/tensorrt"
        )


def build_engine_python(
    onnx_path: Path,
    outdir: Path,
    model_type: str = "gpt",
    fp16: bool = True,
    workspace_mb: int = 8192,
) -> Path:
    """
    Build TensorRT engine using Python API (alternative to trtexec).

    This method provides more control but requires tensorrt Python package.

    Args:
        onnx_path: Path to ONNX file
        outdir: Output directory
        model_type: "gpt" or "decoder"
        fp16: Enable FP16
        workspace_mb: Workspace memory in MB

    Returns:
        Path to engine file
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "tensorrt Python package not found. Install with:\n"
            "pip install tensorrt"
        )

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    outdir.mkdir(parents=True, exist_ok=True)
    engine_path = outdir / (onnx_path.stem + ".trt")

    # Get shapes
    if model_type == "gpt":
        shapes = build_dynamic_shapes_gpt()
    elif model_type == "gpt_prefix":
        shapes = build_dynamic_shapes_gpt_prefix()
    elif model_type == "gpt_decode":
        shapes = build_dynamic_shapes_gpt_decode()
    elif model_type == "decoder":
        shapes = build_dynamic_shapes_decoder()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Building engine with TensorRT Python API...")

    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()

    for input_name, shape_spec in shapes.items():
        # Find input tensor
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            if tensor.name == input_name:
                # Parse shape strings to tuples
                min_shape = tuple(int(x) for x in shape_spec["min"].split("x"))
                opt_shape = tuple(int(x) for x in shape_spec["opt"].split("x"))
                max_shape = tuple(int(x) for x in shape_spec["max"].split("x"))

                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                break

    config.add_optimization_profile(profile)

    # Build engine
    logger.info("Building engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    logger.info(f"Saved engine: {engine_path}")
    return engine_path


def main():
    parser = argparse.ArgumentParser(
        description="Build TensorRT engines from XTTS ONNX models."
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--onnx",
        type=str,
        help="Path to single ONNX file"
    )
    input_group.add_argument(
        "--onnx-dir",
        type=str,
        help="Directory containing ONNX files (gpt.onnx, hifigan_decoder.onnx)"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="models/trt",
        help="Output directory for TRT engines"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["gpt", "decoder", "auto"],
        default="auto",
        help="Model type (auto-detected from filename if 'auto')"
    )

    # Precision options
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 precision")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 precision")

    # Build options
    parser.add_argument(
        "--workspace",
        type=int,
        default=8192,
        help="GPU workspace memory in MB (default: 8192)"
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=3,
        help="Builder optimization level 0-5 (default: 3)"
    )
    parser.add_argument(
        "--use-python-api",
        action="store_true",
        help="Use TensorRT Python API instead of trtexec"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    outdir = Path(args.out)

    def detect_model_type(path: Path) -> str:
        """Detect model type from filename."""
        name = path.stem.lower()
        if "gpt_prefix" in name:
            return "gpt_prefix"
        elif "gpt_decode" in name:
            return "gpt_decode"
        elif "gpt" in name:
            return "gpt"
        elif "decoder" in name or "hifigan" in name:
            return "decoder"
        else:
            raise ValueError(f"Cannot detect model type from filename: {path.name}")

    def build_single(onnx_path: Path, model_type: str):
        """Build single engine."""
        if args.use_python_api:
            return build_engine_python(
                onnx_path,
                outdir,
                model_type=model_type,
                fp16=args.fp16,
                workspace_mb=args.workspace,
            )
        else:
            return build_engine(
                onnx_path,
                outdir,
                model_type=model_type,
                fp16=args.fp16,
                bf16=args.bf16,
                int8=args.int8,
                workspace_mb=args.workspace,
                verbose=args.verbose,
                builder_optimization_level=args.opt_level,
            )

    if args.onnx:
        # Single file mode
        onnx_path = Path(args.onnx)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        model_type = args.type if args.type != "auto" else detect_model_type(onnx_path)
        build_single(onnx_path, model_type)

    else:
        # Directory mode
        onnx_dir = Path(args.onnx_dir)
        if not onnx_dir.exists():
            raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

        # Find and build all ONNX files
        onnx_files = list(onnx_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX files found in: {onnx_dir}")

        logger.info(f"Found {len(onnx_files)} ONNX files")

        for onnx_path in onnx_files:
            try:
                model_type = detect_model_type(onnx_path)
                build_single(onnx_path, model_type)
            except Exception as e:
                logger.error(f"Failed to build {onnx_path.name}: {e}")

    logger.info("Build complete!")


if __name__ == "__main__":
    main()
