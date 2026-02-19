# System Configuration

Hardware and software configuration for the development system.

## Hardware

### GPU (2x NVIDIA)

| GPU | Model | Purpose |
|-----|-------|---------|
| GPU 0 | **CMP 50HX** (TU102) | ~10GB VRAM, LLM inference (primary) |
| GPU 1 | **GeForce GTX 1650 SUPER** (TU116) | 4GB VRAM, display + STT (Whisper) |

**GPU Usage Guidelines:**
- **LLM inference:** Use GPU 0 (CMP 50HX) - larger VRAM
- **STT (Whisper):** Use GPU 1 (GTX 1650 SUPER) - separate from LLM
- For 12B Q4_K_M model (~7GB), offload all layers to CMP 50HX
- `--tensor-split 1,0` in llama-server routes to CMP 50HX only

### CPU

| Parameter | Value |
|-----------|-------|
| Model | Intel Xeon E5-2666 v3 |
| Frequency | 2.90 GHz |
| Cores/Threads | 10 / 20 |

**Recommendations:**
- `n_threads` for LLM: 8-16 (not all 20)
- Leave threads for system and other modules

### RAM

| Parameter | Value |
|-----------|-------|
| Total | 45 GB |
| Available | ~39 GB |
| Swap | 8 GB |

**Sufficient for:**
- Loading model to RAM before GPU offload
- Running multiple modules simultaneously
- STT/TTS models in parallel

---

## Software

### OS
- **Ubuntu 25.10** (Oracular)
- Kernel: 6.17.0-7-generic

### CUDA Stack

| Component | Version |
|-----------|---------|
| NVIDIA Driver | 580.105.08 |
| CUDA Toolkit | 12.4 (V12.4.131) |
| cuDNN | 9.1.0 |

### Python Environment

| Parameter | Value |
|-----------|-------|
| Conda | Miniconda3 (25.9.1) |
| Environment | `wiwi` |
| Python | 3.11.14 |
| Path | `/home/wiwi/miniconda3/envs/wiwi` |

### Key Packages

| Package | Version | Purpose |
|---------|---------|---------|
| aiohttp | 3.13.2 | HTTP client for llama-server |
| pyyaml | 6.0.3 | Configuration |
| rich | 14.2.0 | CLI interface |
| faster-whisper | 1.0.0+ | Speech-to-Text |
| torch | 2.6.0+cu124 | Deep learning framework |
| torchaudio | 2.6.0+cu124 | Audio processing |
| scipy | 1.16.0+ | Audio resampling |
| sounddevice | 0.4.6+ | Audio capture |

---

## LLM Model

| Parameter | Value |
|-----------|-------|
| File | `magnum-v4-12b-Q4_K_M.gguf` |
| Size | ~7.5 GB |
| Quantization | Q4_K_M |
| Parameters | 12B |

### Running llama-server

```bash
./llama.cpp/build/bin/llama-server \
    -m models/magnum-v4-12b-Q4_K_M.gguf \
    --port 8080 \
    --ctx-size 8192 \
    --batch-size 512 \
    --threads 8 \
    --gpu-layers 999 \
    --tensor-split 1,0 \
    --flash-attn
```

**Parameters:**
- `--gpu-layers 999` — offload all layers to GPU
- `--tensor-split 1,0` — use only CMP 50HX (GPU 0), ignore GTX 1650
- `--flash-attn` — enable Flash Attention for speedup
- `--ctx-size 8192` — context size

---

## STT Model (Whisper)

| Parameter | Value |
|-----------|-------|
| Backend | faster-whisper (CTranslate2) |
| Model | base (configurable) |
| Device | cuda:1 (GTX 1650 SUPER) |
| Compute Type | int8 |
| Sample Rate | 16000 Hz |

### cuDNN Requirement

faster-whisper requires cuDNN libraries. Set `LD_LIBRARY_PATH` before running:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

Or use the `run.sh` launcher script which sets this automatically.

### VAD (Voice Activity Detection)

| Parameter | Value |
|-----------|-------|
| Model | Silero VAD |
| Chunk Size | 512 samples (exactly) |
| Sample Rate | 16000 Hz |
| Threshold | 0.5 |

---

## Paths

| Path | Purpose |
|------|---------|
| `/home/wiwi/Wiwi4.0` | Project root |
| `/home/wiwi/miniconda3/envs/wiwi` | Conda environment |
| `./models/` | GGUF models |
| `./plugins/` | External plugins |
| `./config/` | Configuration files |

---

## Environment Activation

```bash
# Option 1: Using run.sh (recommended)
./run.sh

# Option 2: Manual activation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wiwi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
python -m wiwi

# Option 3: Direct call
~/miniconda3/envs/wiwi/bin/python -m wiwi
```

---

## Development Constraints

1. **GPU Memory:**
   - CMP 50HX (~10GB) - LLM only
   - GTX 1650 SUPER (4GB) - STT/TTS
   - Don't run both on same GPU

2. **LLM:** External llama-server process with HTTP API

3. **CPU Threads:** Don't use all 20 threads in a single module

4. **cuDNN:** Required for faster-whisper, set LD_LIBRARY_PATH

5. **Audio:** Resampling from 44100/48000 Hz to 16000 Hz handled automatically
