"""STT (Speech-to-Text) module for Wiwi4.0."""

import asyncio
import time
from typing import Any, Dict, Optional
import numpy as np

from wiwi.interfaces.base_module import BaseModule, ModuleInfo, ModuleState
from wiwi.interfaces.ports import PortType
from wiwi.interfaces.messages import Message, AudioData
from wiwi.modules.stt.backends.base import BaseSTTBackend
from wiwi.modules.stt.vad import SileroVAD
from wiwi.modules.stt.audio_capture import AudioCapture


class STTModule(BaseModule):
    """
    Speech-to-Text module with Voice Activity Detection.

    This module provides real-time speech recognition capabilities:
    - Voice Activity Detection (VAD) using Silero VAD
    - Speech transcription using faster-whisper
    - Voice mode for CLI interaction

    Features:
    - Automatic speech detection and segmentation
    - Streaming transcription when speech ends
    - Integration with LLM pipeline via EventBus

    Config options:
        backend: STT backend ("faster_whisper"). Default: "faster_whisper"
        model_size: Model size (tiny, base, small, medium, large). Default: "base"
        language: Language code (ru, en, etc.). Default: "ru"
        device: Device for inference (cuda, cuda:0, cuda:1, cpu). Default: "cuda:1"
        compute_type: Compute type (int8, float16, float32). Default: "int8"

        vad:
            threshold: Speech probability threshold (0.0-1.0). Default: 0.5
            min_speech_duration_ms: Minimum speech duration. Default: 250
            min_silence_duration_ms: Silence to end speech. Default: 500
            speech_pad_ms: Padding around speech. Default: 100

        audio:
            device: Audio device index (null = default). Default: null
            sample_rate: Sample rate in Hz. Default: 16000
            channels: Number of channels. Default: 1
            chunk_duration_ms: Chunk duration in ms. Default: 30
    """

    @property
    def module_info(self) -> ModuleInfo:
        return ModuleInfo(
            name="stt",
            version="1.0.0",
            description="Speech-to-Text with VAD for voice input",
            author="Wiwi Team",
            category="speech",
            input_ports={PortType.AUDIO_IN, PortType.COMMAND_IN},
            output_ports={PortType.TEXT_OUT},
            dependencies=set(),
            optional_dependencies={"memory", "llm_brain"},
            config_schema={
                "backend": {"type": "string", "default": "faster_whisper"},
                "model_size": {"type": "string", "default": "base"},
                "language": {"type": "string", "default": "ru"},
                "device": {"type": "string", "default": "cuda:1"}
            }
        )

    def __init__(self, kernel, config: Dict[str, Any]):
        super().__init__(kernel, config)

        self._backend: Optional[BaseSTTBackend] = None
        self._backend_name = config.get("backend", "faster_whisper")

        # VAD configuration
        vad_config = config.get("vad", {})
        self._vad = SileroVAD(
            threshold=vad_config.get("threshold", 0.5),
            min_speech_duration_ms=vad_config.get("min_speech_duration_ms", 250),
            min_silence_duration_ms=vad_config.get("min_silence_duration_ms", 500),
            speech_pad_ms=vad_config.get("speech_pad_ms", 100),
            sample_rate=config.get("audio", {}).get("sample_rate", 16000)
        )

        # Audio capture configuration
        audio_config = config.get("audio", {})
        self._audio_capture = AudioCapture(
            device=audio_config.get("device"),
            sample_rate=audio_config.get("sample_rate", 16000),
            channels=audio_config.get("channels", 1),
            chunk_duration_ms=audio_config.get("chunk_duration_ms", 30)
        )

        # Voice mode state
        self._voice_mode_active = False
        self._processing_lock = asyncio.Lock()
        self._current_time = 0.0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def initialize(self) -> None:
        """Initialize STT module: load backend and VAD models."""
        self._state = ModuleState.INITIALIZING
        self._logger.info(f"Initializing STT module with backend: {self._backend_name}")

        # Store event loop reference for callbacks
        self._loop = asyncio.get_event_loop()

        # Create backend
        if self._backend_name == "faster_whisper":
            from wiwi.modules.stt.backends.faster_whisper import FasterWhisperBackend
            self._backend = FasterWhisperBackend(self._config)
        else:
            raise ValueError(f"Unknown STT backend: {self._backend_name}")

        # Load models in executor (blocking operations)
        self._logger.info("Loading STT backend model...")
        await self._loop.run_in_executor(None, self._backend.load_model)

        self._logger.info("Loading VAD model...")
        await self._loop.run_in_executor(None, self._vad.load_model)

        # Set up VAD callbacks
        self._vad.set_callbacks(
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end_sync
        )

        # Set up audio callback
        self._audio_capture.set_callback(self._on_audio_chunk)

        self._state = ModuleState.READY
        self._logger.info("STT module initialized successfully")

    async def start(self) -> None:
        """Start the STT module."""
        self._state = ModuleState.RUNNING
        self._logger.info("STT module started (voice mode inactive by default)")

    async def stop(self) -> None:
        """Stop and cleanup the STT module."""
        self._state = ModuleState.STOPPING

        # Stop voice mode if active
        if self._voice_mode_active:
            await self._stop_voice_mode()

        # Unload models
        if self._backend:
            self._backend.unload_model()

        self._vad.unload_model()

        self._state = ModuleState.STOPPED
        self._logger.info("STT module stopped")

    async def handle_input(self, message: Message) -> Optional[Message]:
        """Handle incoming messages."""
        if message.port == PortType.COMMAND_IN:
            return await self._handle_command(message)
        elif message.port == PortType.AUDIO_IN:
            return await self._handle_audio(message)
        return None

    async def _handle_command(self, message: Message) -> Optional[Message]:
        """Handle STT commands."""
        command = str(message.payload).lower().strip()

        if command == "voice_on":
            await self._start_voice_mode()
            return message.reply(
                payload="Voice mode enabled. Speak now...",
                port=PortType.TEXT_OUT
            )
        elif command == "voice_off":
            await self._stop_voice_mode()
            return message.reply(
                payload="Voice mode disabled.",
                port=PortType.TEXT_OUT
            )
        elif command == "voice_toggle":
            if self._voice_mode_active:
                await self._stop_voice_mode()
                return message.reply(
                    payload="Voice mode disabled.",
                    port=PortType.TEXT_OUT
                )
            else:
                await self._start_voice_mode()
                return message.reply(
                    payload="Voice mode enabled. Speak now...",
                    port=PortType.TEXT_OUT
                )
        elif command == "status":
            status = "active" if self._voice_mode_active else "inactive"
            return message.reply(
                payload=f"STT voice mode: {status}",
                port=PortType.TEXT_OUT
            )

        return None

    async def _handle_audio(self, message: Message) -> Optional[Message]:
        """Handle incoming audio data (for manual transcription)."""
        if not isinstance(message.payload, AudioData):
            return None

        audio_data = message.payload
        audio_array = np.frombuffer(audio_data.data, dtype=np.float32)

        # Transcribe
        text = await self.transcribe(audio_array, audio_data.sample_rate)

        if text:
            return Message(
                source=self.name,
                target="cli_interface",
                port=PortType.TEXT_OUT,
                payload=text,
                metadata={"source": "stt", "duration_ms": audio_data.duration_ms},
                correlation_id=message.correlation_id
            )

        return None

    async def _start_voice_mode(self) -> None:
        """Start voice mode (listening from microphone)."""
        if self._voice_mode_active:
            self._logger.warning("Voice mode already active")
            return

        self._voice_mode_active = True
        self._current_time = 0.0
        self._vad.reset()
        self._audio_capture.start()

        self._logger.info("Voice mode started - listening...")

    async def _stop_voice_mode(self) -> None:
        """Stop voice mode."""
        if not self._voice_mode_active:
            return

        self._voice_mode_active = False
        self._audio_capture.stop()
        self._vad.reset()

        self._logger.info("Voice mode stopped")

    def _on_audio_chunk(self, audio: np.ndarray) -> None:
        """
        Handle audio chunk from capture (sync callback from sounddevice thread).

        This is called from a separate thread, so we need to be careful
        with async operations.
        """
        chunk_duration = len(audio) / self._audio_capture.sample_rate
        self._current_time += chunk_duration

        # Process through VAD (sync)
        self._vad.process_chunk(audio, self._current_time)

    def _on_speech_start(self) -> None:
        """Called when speech starts (from VAD)."""
        self._logger.debug("Speech detected - listening...")

    def _on_speech_end_sync(self, audio: np.ndarray) -> None:
        """
        Called when speech ends (sync callback from VAD).

        Schedule async processing in the event loop.
        """
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(
                self._process_speech(audio),
                self._loop
            )

    async def _process_speech(self, audio: np.ndarray) -> None:
        """Process completed speech segment."""
        async with self._processing_lock:
            duration = len(audio) / self._audio_capture.sample_rate
            self._logger.info(f"Processing speech segment: {duration:.2f}s")

            # Transcribe
            text = await self.transcribe(audio)

            if text and self._event_bus:
                self._logger.info(f"Transcribed: {text}")

                # Print user speech
                print(f"\n[Voice] You: {text}")

                # Publish transcribed text to EventBus
                # Memory module will receive TEXT_IN, add context, forward to LLM
                # LLM will respond with TEXT_OUT (broadcast)
                # TTS module will receive TEXT_OUT and speak
                # CLI module will receive TEXT_OUT and display
                message = Message(
                    source=self.name,
                    target=None,  # Goes to memory first (subscribed to TEXT_IN)
                    port=PortType.TEXT_IN,
                    payload=text,
                    metadata={"source": "voice"}
                )
                await self._event_bus.publish(message)

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as float32 numpy array
            sample_rate: Sample rate in Hz

        Returns:
            Transcribed text string
        """
        if not self._backend or not self._backend.is_loaded:
            raise RuntimeError("STT backend not ready")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._backend.transcribe(
                audio,
                sample_rate=sample_rate,
                language=self._config.get("language", "ru")
            )
        )

    # === Public API ===

    @property
    def is_voice_mode_active(self) -> bool:
        """Check if voice mode is currently active."""
        return self._voice_mode_active

    async def enable_voice_mode(self) -> None:
        """Enable voice mode programmatically."""
        await self._start_voice_mode()

    async def disable_voice_mode(self) -> None:
        """Disable voice mode programmatically."""
        await self._stop_voice_mode()

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the STT backend."""
        if self._backend:
            return self._backend.get_model_info()
        return {"loaded": False}

    @staticmethod
    def list_audio_devices():
        """List available audio input devices."""
        return AudioCapture.list_devices()
