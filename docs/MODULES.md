# Modules Reference

Reference guide for Wiwi modules.

---

## Core Modules (Built-in)

### memory

**Purpose:** Short-term memory, conversation context storage.

| Parameter | Value |
|-----------|-------|
| Category | memory |
| Input Ports | `TEXT_IN`, `MEMORY_WRITE`, `COMMAND_IN` |
| Output Ports | `TEXT_OUT`, `MEMORY_READ` |
| Dependencies | - |

**Configuration:**
```yaml
modules:
  memory:
    backend: "in_memory"
    max_history_length: 20      # Max messages in history
    max_tokens_per_turn: 500    # Max tokens per message
```

**API:**
- `clear_history()` - clear history
- `get_history()` - get history
- `get_context()` - get context for LLM
- `add_turn(role, content, metadata)` - add conversation turn
- `history_length` - current history length

---

### llm_brain

**Purpose:** LLM "brain" - response generation using local GGUF models.

| Parameter | Value |
|-----------|-------|
| Category | ai |
| Input Ports | `TEXT_IN` |
| Output Ports | `TEXT_OUT` |
| Dependencies | `memory` |

**Backend:** `llama_server` - llama.cpp server via HTTP API

---

#### Backend: llama_server (recommended)

Uses external llama-server process for inference. Best GPU performance.

**Starting the server:**
```bash
./llama.cpp/build/bin/llama-server \
    -m models/magnum-v4-12b-Q4_K_M.gguf \
    --port 8080 \
    --ctx-size 8192 \
    --batch-size 512 \
    --threads 8 \
    --gpu-layers 999 \
    --tensor-split 1,0
```

**Configuration:**
```yaml
modules:
  llm_brain:
    backend: "llama_server"
    api_url: "http://localhost:8080"
    timeout: 120              # Request timeout (sec)
    temperature: 0.7
    top_p: 0.9
    top_k: 40
    max_tokens: 512
    repeat_penalty: 1.1
    system_prompt: |
      You are Wiwi, a friendly AI assistant.
    stop_tokens:
      - "</s>"
      - "<|user|>"
```

**Features:**
- Async HTTP requests via aiohttp
- Streaming support (`generate_stream`)
- Health check on initialization
- Automatic reconnection

---

### cli_interface

**Purpose:** Command-line interface for user interaction.

| Parameter | Value |
|-----------|-------|
| Category | interface |
| Input Ports | `TEXT_OUT`, `AUDIO_OUT`, `SYSTEM_OUT` |
| Output Ports | `TEXT_IN`, `COMMAND_IN` |
| Dependencies | - |
| Optional Deps | `memory`, `stt` |

**Configuration:**
```yaml
modules:
  cli_interface:
    prompt: ">>> "
    assistant_prefix: "Wiwi: "
    colors: true
    history_file: "~/.wiwi_history"
    max_input_length: 4096
```

**Commands:**
| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/clear` | Clear history |
| `/quit`, `/exit` | Exit |
| `/modules` | List modules |
| `/reload <name>` | Reload module |
| `/status` | System status |
| `/voice` | Toggle voice mode |
| `/voice on` | Enable voice mode |
| `/voice off` | Disable voice mode |
| `/devices` | List audio devices |

---

### stt (Speech-to-Text)

**Purpose:** Real-time speech recognition with Voice Activity Detection.

| Parameter | Value |
|-----------|-------|
| Category | speech |
| Input Ports | `AUDIO_IN`, `COMMAND_IN` |
| Output Ports | `TEXT_OUT` |
| Dependencies | - |
| Optional Deps | `memory`, `llm_brain` |

**Backend:** `faster_whisper` - CTranslate2-based Whisper implementation

---

#### Backend: faster_whisper

High-performance Whisper implementation using CTranslate2.

**Requirements:**
- CUDA 12.x with cuDNN 9.x
- Set `LD_LIBRARY_PATH` for cuDNN (see SYSTEM.md)

**Configuration:**
```yaml
modules:
  stt:
    backend: "faster_whisper"
    model_size: "base"        # tiny, base, small, medium, large-v2, large-v3
    language: "ru"            # Language code
    device: "cuda:1"          # cuda:0, cuda:1, cpu
    compute_type: "int8"      # int8, float16, float32

    # Voice Activity Detection (Silero VAD)
    vad:
      threshold: 0.5          # Speech probability threshold (0.0-1.0)
      min_speech_duration_ms: 250   # Minimum speech duration
      min_silence_duration_ms: 500  # Silence to end speech
      speech_pad_ms: 100      # Padding around speech

    # Audio capture (sounddevice)
    audio:
      device: null            # null = default microphone
      sample_rate: 16000      # 16kHz for Whisper
      channels: 1             # Mono
      chunk_duration_ms: 64   # 64ms chunks (1024 samples at 16kHz)
```

**Features:**
- Real-time VAD using Silero VAD
- Automatic resampling from 44100/48000 Hz to 16000 Hz
- Voice mode toggle via `/voice` command
- Streaming transcription when speech ends
- Integration with Memory and LLM pipeline

**Model Sizes:**
| Model | Parameters | VRAM | Speed |
|-------|------------|------|-------|
| tiny | 39M | ~1GB | Fastest |
| base | 74M | ~1GB | Fast |
| small | 244M | ~2GB | Good |
| medium | 769M | ~5GB | Better |
| large-v2/v3 | 1550M | ~10GB | Best |

**Compute Types:**
| Type | Precision | Speed | Quality |
|------|-----------|-------|---------|
| int8 | 8-bit | Fastest | Good |
| float16 | 16-bit | Fast | Better |
| float32 | 32-bit | Slower | Best |

**API:**
- `enable_voice_mode()` - start listening
- `disable_voice_mode()` - stop listening
- `is_voice_mode_active` - check voice mode status
- `transcribe(audio, sample_rate)` - manual transcription
- `list_audio_devices()` - list microphones
- `get_backend_info()` - get backend information

---

### tts (Text-to-Speech)

**Purpose:** Speech synthesis using XTTS-v2.

| Parameter | Value |
|-----------|-------|
| Category | speech |
| Input Ports | `TEXT_IN` |
| Output Ports | `AUDIO_OUT` |
| Dependencies | - |

**Backend:** `xtts` - Coqui XTTS-v2 (multilingual, voice cloning)

**Configuration:**
```yaml
modules:
  tts:
    backend: "xtts"
    model: "tts_models/multilingual/multi-dataset/xtts_v2"
    device: "cuda:0"
    language: "ru"
    sample_rate: 24000
    auto_speak: true
    reference_audio: "models/tts/reference.wav"
    speed: 1.0
    temperature: 0.75
```

---

### discord

**Purpose:** Discord integration with text and voice channel support.

| Parameter | Value |
|-----------|-------|
| Category | interface |
| Input Ports | `TEXT_OUT`, `AUDIO_OUT`, `COMMAND_IN` |
| Output Ports | `TEXT_IN`, `AUDIO_IN`, `COMMAND_OUT` |
| Dependencies | - |
| Optional Deps | `memory`, `stt`, `tts` |

**Features:**
- Text channel responses (mentions, prefix "wiwi")
- Voice channel support with wake word detection
- OpenWakeWord for lightweight continuous listening (~1-3% CPU)
- Silero VAD for end-of-speech detection
- Programmatic sound effects for audio feedback

**Configuration:**
```yaml
modules:
  discord:
    token: "YOUR_BOT_TOKEN"
    respond_to_mentions: true
    respond_to_prefix: true
    voice_enabled: true

    wake_word:
      model_paths: []           # Custom models (.onnx)
      threshold: 0.5
      vad_threshold: 0.5
      min_silence_ms: 700
      max_recording_ms: 10000

    sounds:
      enabled: true
```

**Slash Commands:**
| Command | Description |
|---------|-------------|
| `/join` | Join user's voice channel |
| `/leave` | Leave voice channel |
| `/status` | Show bot status |

**Voice Pipeline:**
```
Discord Audio (48kHz stereo)
    │
    ▼ Resample to 16kHz mono
┌─────────────────┐
│ OpenWakeWord    │  ~1-3% CPU, always listening
└────────┬────────┘
         │ Wake word detected
         ▼
┌─────────────────┐
│  Silero VAD     │  Records until silence
└────────┬────────┘
         │ Complete command audio
         ▼
┌─────────────────┐
│  STT (Whisper)  │  Transcription
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Brain     │  Generate response
└─────────────────┘
```

**📚 Full Documentation:** [DISCORD.md](DISCORD.md)

---

## Planned Modules

### vision

**Purpose:** Image and video analysis.

| Parameter | Value |
|-----------|-------|
| Input Ports | `IMAGE_IN`, `VIDEO_IN` |
| Output Ports | `TEXT_OUT` |
| Dependencies | - |

---

## Module Development Guidelines

### System Resources

When developing modules, consider (see [SYSTEM.md](SYSTEM.md)):
- **GPU VRAM:** CMP 50HX ~10GB (LLM), GTX 1650 SUPER 4GB (STT)
- **CPU Threads:** 20 (don't use all)
- **RAM:** 45GB available

### Best Practices

1. **Async:** All I/O operations via `asyncio`
2. **Heavy computations:** Use `loop.run_in_executor()`
3. **GPU:** Specify device in config, don't hardcode
4. **Logging:** Use `self._logger`
5. **State:** Always set `ModuleState` in lifecycle methods
6. **Cleanup:** Release resources in `stop()` method

### Example with GPU

```python
async def initialize(self) -> None:
    device = self._config.get("device", "cuda")

    # Load model in executor (blocking)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: self._load_model(device)
    )

async def stop(self) -> None:
    # Cleanup
    if self._model:
        del self._model
        self._model = None
    self._state = ModuleState.STOPPED
```

### Audio Processing Example

```python
from wiwi.modules.stt.audio_capture import AudioCapture

# List devices
devices = AudioCapture.list_devices()

# Create capture with resampling
capture = AudioCapture(
    sample_rate=16000,      # Target rate
    chunk_duration_ms=64    # Chunk size
)

# Native rate detected automatically
print(f"Native rate: {capture.native_sample_rate}")
print(f"Needs resampling: {capture._needs_resampling}")

# Set callback
def on_audio(audio: np.ndarray):
    # audio is already resampled to 16kHz
    process(audio)

capture.set_callback(on_audio)
capture.start()
```

### VAD Integration Example

```python
from wiwi.modules.stt.vad import SileroVAD

vad = SileroVAD(
    sample_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500
)

vad.load_model()

# Set callbacks
vad.set_callbacks(
    on_speech_start=lambda: print("Speech started"),
    on_speech_end=lambda audio: transcribe(audio)
)

# Process chunks (512 samples at 16kHz)
for chunk in audio_stream:
    is_speech = vad.process_chunk(chunk, current_time)
```
