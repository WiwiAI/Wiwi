# Quick Start Guide

## Prerequisites

1. **llama-server** running in a separate terminal
2. **Conda environment** `wiwi` activated
3. **cuDNN libraries** in LD_LIBRARY_PATH (for STT)

---

## Step 1: Start llama-server

```bash
./llama.cpp/build/bin/llama-server   -m models/magnum-v4-12b-Q4_K_M.gguf   --port 8080 --ctx-size 128000 --gpu-layers -1 --parallel 4 -fa on -ctk q8_0 -ctv q8_0 -t 12 
```

Wait until you see: `llama server listening at http://localhost:8080`

---

## Step 2: Run Wiwi

### Option A: Using run.sh (Recommended)

```bash
./run.sh
```

This script automatically:
- Activates conda environment
- Sets LD_LIBRARY_PATH for cuDNN
- Starts Wiwi

### Option B: Manual

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wiwi

# Set cuDNN path (required for STT)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# Run
python -m wiwi
```

---

## Basic Usage

### Text Chat

Just type your message and press Enter:

```
>>> Hello!
Wiwi: Hello! How can I help you today?
```

### Voice Mode

Enable voice input:

```
>>> /voice on
ℹ Voice mode enabled. Speak now...
```

Speak into your microphone. When you stop speaking, your words will be transcribed and sent to the LLM.

Disable voice mode:
```
>>> /voice off
```

Or press `Ctrl+C` to disable voice mode.

### Available Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/voice` | Toggle voice mode |
| `/voice on` | Enable voice mode |
| `/voice off` | Disable voice mode |
| `/devices` | List audio devices |
| `/clear` | Clear conversation history |
| `/modules` | Show loaded modules |
| `/status` | Show system status |
| `/quit` | Exit |

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Disable voice mode (if active), or show exit hint |
| `Ctrl+C` (twice) | Graceful shutdown |
| `Ctrl+C` (third time) | Force exit |

---

## Troubleshooting

### "Unable to load libcudnn_ops.so.9"

cuDNN libraries not found. Make sure to set LD_LIBRARY_PATH:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

Or use `./run.sh` which sets this automatically.

### "LLM module not available"

llama-server is not running or not responding. Check:
1. Is llama-server running on port 8080?
2. Can you access http://localhost:8080/health ?

### "No audio input devices found"

Check microphone permissions and connection:
```
>>> /devices
```

### Voice mode not detecting speech

- Check if microphone is working
- Adjust VAD threshold in config (lower = more sensitive)
- Check if correct audio device is selected

---

## Configuration

Main config file: `config/default.yaml`

Key settings:
- `modules.llm_brain.api_url` - llama-server URL
- `modules.stt.device` - GPU for Whisper (cuda:1)
- `modules.stt.model_size` - Whisper model size
- `modules.stt.vad.threshold` - VAD sensitivity

See [MODULES.md](MODULES.md) for full configuration reference.
