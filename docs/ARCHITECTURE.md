# Architecture

Modular AI assistant Wiwi4.0 architecture.

## Concept

**Constructor** - a central Kernel manages pluggable modules through unified input/output ports. Modules can be added/removed without modifying the core.

```
         ┌─────────────────────────────────────┐
         │            KERNEL                   │
         │  ┌─────────┐ ┌─────────┐ ┌────────┐│
         │  │EventBus │ │Registry │ │Config  ││
         │  └────┬────┘ └─────────┘ └────────┘│
         └───────┼────────────────────────────┘
                 │ Message routing
    ┌────────────┼────────────┬───────────┬─────────┐
    ▼            ▼            ▼           ▼         ▼
┌──────┐    ┌────────┐   ┌────────┐   ┌─────┐  ┌─────────┐
│ CLI  │◄──►│ Memory │◄──►│  LLM   │   │ STT │  │ Discord │
└──────┘    └────────┘   └────────┘   └─────┘  └─────────┘
```

---

## Core Components

### Kernel (`wiwi/core/kernel.py`)

Central dispatcher:
- Configuration loading from YAML
- Module discovery in `wiwi/modules/` and `plugins/`
- Lifecycle management: `load → initialize → start → stop → unload`
- Dependency resolution between modules
- Hot reload: `await kernel.reload_module("name")`

### EventBus (`wiwi/core/event_bus.py`)

Asynchronous message bus (Pub/Sub):
- Modules communicate via `Message` objects
- Routing by `PortType`
- Support for targeted (`target="module"`) and broadcast (`target=None`)
- Middleware for message transformation
- Priority queue

### Registry (`wiwi/core/registry.py`)

Module registry:
- Storage of module classes and instances
- Module metadata (`ModuleInfo`)
- Queries by categories and dependencies

---

## Module System

### PortType (`wiwi/interfaces/ports.py`)

Typed input/output channels:

| Port | Direction | Data |
|------|-----------|------|
| `TEXT_IN` / `TEXT_OUT` | Input/Output | Text |
| `AUDIO_IN` / `AUDIO_OUT` | Input/Output | Audio (bytes/numpy) |
| `MEMORY_READ` / `MEMORY_WRITE` | Read/Write | Memory data |
| `COMMAND_IN` / `COMMAND_OUT` | Input/Output | Commands |
| `SYSTEM_IN` / `SYSTEM_OUT` | Input/Output | System events |

### BaseModule (`wiwi/interfaces/base_module.py`)

Contract for all modules:

```python
class MyModule(BaseModule):
    @property
    def module_info(self) -> ModuleInfo:
        return ModuleInfo(
            name="my_module",
            version="1.0.0",
            description="...",
            input_ports={PortType.TEXT_IN},
            output_ports={PortType.TEXT_OUT},
            dependencies={"memory"},
        )

    async def initialize(self) -> None: ...  # Load resources
    async def start(self) -> None: ...       # Start
    async def stop(self) -> None: ...        # Stop
    async def handle_input(self, message: Message) -> Optional[Message]: ...
```

### ModuleState

Lifecycle states:
```
UNINITIALIZED → INITIALIZING → READY → RUNNING → STOPPING → STOPPED
                                 ↓
                              PAUSED
                                 ↓
                              ERROR
```

---

## Message Flow

### Standard Pipeline (Text)

```
User Input
    │
    ▼
┌─────────────────┐
│  CLI Interface  │  TEXT_IN
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Memory      │  Adds context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Brain     │  Generates response
└────────┬────────┘
         │
         ▼ TEXT_OUT
┌─────────────────┐
│  CLI Interface  │  Outputs response
└────────┬────────┘
         │
         ▼ MEMORY_WRITE
┌─────────────────┐
│     Memory      │  Saves to history
└─────────────────┘
```

### Voice Pipeline (STT)

```
Microphone
    │
    ▼ 44100/48000 Hz
┌─────────────────┐
│ Audio Capture   │  Resamples to 16kHz
└────────┬────────┘
         │
         ▼ 16000 Hz (512 samples)
┌─────────────────┐
│   Silero VAD    │  Detects speech start/end
└────────┬────────┘
         │
         ▼ Complete speech segment
┌─────────────────┐
│ faster-whisper  │  Transcribes audio → text
└────────┬────────┘
         │
         ▼ TEXT
┌─────────────────┐
│     Memory      │  Adds to conversation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Brain     │  Generates response
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CLI Interface  │  Outputs response
└─────────────────┘
```

### Discord Voice Pipeline

```
Discord Voice Channel
    │
    ▼ 48kHz stereo (Opus)
┌─────────────────┐
│ WiwiAudioSink   │  Receives audio per user
└────────┬────────┘
         │
         ▼ Resample to 16kHz mono
┌─────────────────┐
│ OpenWakeWord    │  Lightweight detection (~1-3% CPU)
│   (LISTENING)   │  Always active
└────────┬────────┘
         │ Wake word detected → Play sound
         ▼
┌─────────────────┐
│  Silero VAD     │  Records command
│   (RECORDING)   │  Until 700ms silence
└────────┬────────┘
         │ Complete audio
         ▼
┌─────────────────┐
│ faster-whisper  │  Transcribes command
└────────┬────────┘
         │
         ▼ TEXT_IN
┌─────────────────┐
│     Memory      │  Adds to conversation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Brain     │  Generates response
└────────┬────────┘
         │
         ▼ TEXT_OUT
┌─────────────────┐
│ Discord Module  │  Sends text + optional TTS
└─────────────────┘
```

### Message Structure

```python
@dataclass
class Message:
    source: str              # Sender module
    port: PortType           # Port type
    payload: Any             # Data
    target: Optional[str]    # Recipient (None = broadcast)
    metadata: Dict[str, Any] # Additional data (context, etc.)
    correlation_id: str      # ID for tracking chain
```

---

## Configuration

### Structure `config/default.yaml`

```yaml
general:
  name: "Wiwi"
  language: "ru"

enabled_modules:        # Auto-load on start
  - "memory"
  - "llm_brain"
  - "cli_interface"
  - "stt"

modules:                # Configuration for each module
  llm_brain:
    backend: "llama_server"
    api_url: "http://localhost:8080"
    ...

  stt:
    backend: "faster_whisper"
    model_size: "base"
    device: "cuda:1"
    ...

paths:
  models_dir: "./models"
  plugins_dir: "./plugins"
```

### Environment Variables

Support for `${VAR:-default}` syntax:
```yaml
model_path: "${WIWI_MODEL_PATH:-./models/model.gguf}"
```

---

## Adding New Modules

### Built-in Module

1. Create `wiwi/modules/<name>/module.py`
2. Implement `BaseModule` with `module_info`
3. Add to `enabled_modules` or load dynamically

### External Plugin

1. Create `plugins/<name>/module.py`
2. Same structure, auto-discovered on startup

### Minimal Module Example

```python
# wiwi/modules/my_module/module.py
from wiwi.interfaces.base_module import BaseModule, ModuleInfo, ModuleState
from wiwi.interfaces.ports import PortType
from wiwi.interfaces.messages import Message

class MyModule(BaseModule):
    @property
    def module_info(self) -> ModuleInfo:
        return ModuleInfo(
            name="my_module",
            version="1.0.0",
            description="My custom module",
            input_ports={PortType.TEXT_IN},
            output_ports={PortType.TEXT_OUT},
        )

    async def initialize(self) -> None:
        self._logger.info("Initializing...")

    async def start(self) -> None:
        self._state = ModuleState.RUNNING

    async def stop(self) -> None:
        self._state = ModuleState.STOPPED

    async def handle_input(self, message: Message) -> Optional[Message]:
        if message.port == PortType.TEXT_IN:
            # Process and respond
            return message.reply(payload="Processed!")
        return None
```

---

## Signal Handling

### Graceful Shutdown

The application handles `SIGINT` (Ctrl+C) and `SIGTERM` signals:

1. First `Ctrl+C`:
   - If voice mode active → disables voice mode
   - Otherwise → shows exit hint
2. Second `Ctrl+C`: Initiates graceful shutdown
3. Third `Ctrl+C`: Force exit

### Module Cleanup

On shutdown, modules are stopped in reverse order:
1. CLI Interface stops listening
2. STT stops audio capture
3. LLM Brain disconnects
4. Memory saves state
5. EventBus drains queue

---

## File Structure

```
wiwi/
├── core/
│   ├── kernel.py         # Central dispatcher
│   ├── event_bus.py      # Message bus
│   ├── registry.py       # Module registry
│   ├── config_loader.py  # Configuration loader
│   └── exceptions.py     # Exceptions
├── interfaces/
│   ├── base_module.py    # Base module class
│   ├── ports.py          # Port types
│   └── messages.py       # Message types
├── modules/
│   ├── memory/           # Short-term memory
│   ├── llm_brain/        # LLM brain
│   ├── cli_interface/    # CLI interface
│   ├── stt/              # Speech-to-Text
│   │   ├── module.py     # Main STT module
│   │   ├── audio_capture.py  # Microphone capture
│   │   ├── vad.py        # Voice Activity Detection
│   │   └── backends/
│   │       ├── base.py           # Backend interface
│   │       └── faster_whisper.py # Whisper backend
│   ├── tts/              # Text-to-Speech
│   │   └── module.py     # XTTS-v2 module
│   └── discord/          # Discord integration
│       ├── module.py     # Main Discord bot module
│       ├── wake_word.py  # OpenWakeWord + VAD pipeline
│       └── sounds.py     # Sound effect generator
└── utils/
    └── logging_setup.py
```
