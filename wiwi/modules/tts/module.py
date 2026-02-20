"""TTS (Text-to-Speech) module for Wiwi4.0."""

import asyncio
import gc
import re
import threading
import uuid
from typing import Any, Dict, Optional

import numpy as np

from wiwi.interfaces.base_module import BaseModule, ModuleInfo, ModuleState
from wiwi.interfaces.ports import PortType
from wiwi.interfaces.messages import Message, AudioData
from wiwi.modules.tts.backends.base import BaseTTSBackend
from wiwi.modules.tts.audio_output import StreamingAudioPlayer


class TTSModule(BaseModule):
    """
    Text-to-Speech module with streaming audio output.

    This module provides real-time speech synthesis capabilities:
    - Streaming synthesis using XTTS-v2 for low latency
    - Voice cloning from reference audio
    - Auto-speak LLM responses when voice mode is active

    Features:
    - Real-time streaming playback (<200ms to first audio)
    - Russian language support
    - Integration with voice mode

    Config options:
        backend: TTS backend ("xtts" or "your_tts"). Default: "xtts"
        model: Model identifier. Default depends on backend:
            - xtts: "tts_models/multilingual/multi-dataset/xtts_v2"
            - your_tts: "tts_models/multilingual/multi-dataset/your_tts"
        device: Device for inference (cuda:0, cuda:1, cpu). Default: "cuda:1"
        language: Language code (ru, en, etc.). Default: "ru"
        sample_rate: Output sample rate. Default: 24000
        auto_speak: Auto-enable when voice mode active. Default: true
        reference_audio: Path to reference audio for voice cloning.

        audio:
            device: Audio output device index (null = default). Default: null
            volume: Playback volume (0.0-1.0). Default: 1.0
    """

    @property
    def module_info(self) -> ModuleInfo:
        return ModuleInfo(
            name="tts",
            version="1.0.0",
            description="Text-to-Speech with streaming output",
            author="Wiwi Team",
            category="speech",
            input_ports={PortType.TEXT_OUT, PortType.AUDIO_OUT, PortType.COMMAND_IN},
            output_ports={PortType.AUDIO_OUT},
            dependencies=set(),
            optional_dependencies={"stt", "llm_brain", "cli_interface"},
            config_schema={
                "backend": {"type": "string", "default": "xtts"},
                "model": {"type": "string", "default": "tts_models/multilingual/multi-dataset/xtts_v2"},
                "device": {"type": "string", "default": "cuda:1"},
                "language": {"type": "string", "default": "ru"},
                "sample_rate": {"type": "integer", "default": 24000},
                "auto_speak": {"type": "boolean", "default": True},
            }
        )

    def __init__(self, kernel, config: Dict[str, Any]):
        super().__init__(kernel, config)

        self._backend: Optional[BaseTTSBackend] = None
        self._backend_name = config.get("backend", "xtts")
        self._audio_player: Optional[StreamingAudioPlayer] = None

        # TTS state
        self._tts_enabled = False
        self._auto_speak = config.get("auto_speak", True)

        # Discord voice mode - when active, don't play locally
        self._discord_voice_active = False

        # Event loop reference
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Synchronous synthesis lock - only one sentence at a time
        self._synth_lock = threading.Lock()
        self._is_speaking = False

    async def initialize(self) -> None:
        """Initialize TTS module: load backend and audio output."""
    
        self._state = ModuleState.INITIALIZING
        self._logger.info(f"Initializing TTS module with backend: {self._backend_name}")

        # Store event loop reference
        self._loop = asyncio.get_event_loop()

        # Create backend
        if self._backend_name == "xtts":
            from wiwi.modules.tts.backends.xtts import XTTSBackend
            self._backend = XTTSBackend(self._config)
        elif self._backend_name == "your_tts":
            from wiwi.modules.tts.backends.your_tts import YourTTSBackend
            self._backend = YourTTSBackend(self._config)
        elif self._backend_name == "xtts_trt":
            from wiwi.modules.tts.backends.xtts_trt_backend import XttsTrtBackend
            self._backend = XttsTrtBackend(self._config)
        elif self._backend_name == "chatterbox":
            from wiwi.modules.tts.backends.chatterbox import ChatterboxBackend
            self._backend = ChatterboxBackend(self._config)
        else:
            raise ValueError(f"Unknown TTS backend: {self._backend_name}")

        # Load model in executor (blocking GPU operation)
        self._logger.info("Loading TTS backend model...")
        await self._loop.run_in_executor(None, self._backend.load_model)

        # Warmup model to reduce first-call latency
        if hasattr(self._backend, 'warmup'):
            self._logger.info("Warming up TTS model...")
            await self._loop.run_in_executor(None, self._backend.warmup)

        # Initialize audio output
        audio_config = self._config.get("audio", {})
        self._audio_player = StreamingAudioPlayer(
            sample_rate=self._backend.sample_rate,
            device=audio_config.get("device"),
            volume=audio_config.get("volume", 1.0),
        )

        self._state = ModuleState.READY
        self._logger.info("TTS module initialized successfully")

    async def start(self) -> None:
        """Start the TTS module."""
        self._state = ModuleState.RUNNING
        self._logger.info("TTS module started (TTS inactive by default)")

    async def stop(self) -> None:
        """Stop and cleanup the TTS module."""
        self._state = ModuleState.STOPPING

        # Stop any ongoing playback
        if self._audio_player:
            self._audio_player.stop()

        # Unload model
        if self._backend:
            self._backend.unload_model()

        self._state = ModuleState.STOPPED
        self._logger.info("TTS module stopped")

    async def handle_input(self, message: Message) -> Optional[Message]:
        """Handle incoming messages."""
        # Ignore our own messages to prevent feedback loops
        if message.source == self.name:
            return None

        if message.port == PortType.COMMAND_IN:
            return await self._handle_command(message)
        elif message.port == PortType.AUDIO_OUT:
            # Streaming text from LLM or CLI - synthesize immediately
            # (but not AudioData which is actual audio, not text)
            if message.metadata.get("streaming"):
                await self._handle_streaming_text(message)
            return None
        elif message.port == PortType.TEXT_OUT:
            # Full text from LLM - only synthesize if not already streamed
            if not message.metadata.get("streamed"):
                await self._handle_text(message)
            return None
        return None

    async def _handle_streaming_text(self, message: Message) -> None:
        """Handle streaming text input for immediate synthesis."""
        if not self._tts_enabled:
            return

        # Ignore our own audio output messages
        if message.source == self.name:
            return

        # Ignore AudioData payloads (these are audio, not text)
        if hasattr(message.payload, "data") and hasattr(message.payload, "sample_rate"):
            return

        # Ignore None payloads
        if message.payload is None:
            return

        text = str(message.payload)
        if not text.strip():
            return

        # Preprocess text for TTS
        text = self._preprocess_text(text)
        if not text:
            return

        # Synthesize and play immediately
        await self._speak(text)

    async def _handle_command(self, message: Message) -> Optional[Message]:
        """Handle TTS commands."""
        command = str(message.payload).lower().strip()

        if command == "tts_on":
            await self._enable_tts()
            return message.reply(
                payload="TTS enabled.",
                port=PortType.TEXT_OUT
            )
        elif command == "tts_off":
            await self._disable_tts()
            return message.reply(
                payload="TTS disabled.",
                port=PortType.TEXT_OUT
            )
        elif command == "tts_toggle":
            if self._tts_enabled:
                await self._disable_tts()
                return message.reply(
                    payload="TTS disabled.",
                    port=PortType.TEXT_OUT
                )
            else:
                await self._enable_tts()
                return message.reply(
                    payload="TTS enabled.",
                    port=PortType.TEXT_OUT
                )
        elif command == "voice_on":
            # STT voice mode activated - auto-enable TTS if configured
            if self._auto_speak:
                await self._enable_tts()
                self._logger.info("TTS auto-enabled (voice mode active)")
        elif command == "voice_off":
            # STT voice mode deactivated - disable TTS
            if self._auto_speak:
                await self._disable_tts()
                self._logger.info("TTS auto-disabled (voice mode inactive)")
            self._discord_voice_active = False
        elif command == "discord_voice_on":
            # Discord voice channel active - output to Discord, not local
            self._discord_voice_active = True
            if self._auto_speak:
                await self._enable_tts()
            self._logger.info("Discord voice mode active - local playback disabled")
        elif command == "discord_voice_off":
            # Discord voice channel inactive - restore local playback
            self._discord_voice_active = False
            if self._auto_speak:
                await self._disable_tts()
            self._logger.info("Discord voice mode inactive - local playback enabled")
        elif command == "tts_stop":
            # Stop current playback
            if self._audio_player:
                self._audio_player.stop()
            return message.reply(
                payload="TTS playback stopped.",
                port=PortType.TEXT_OUT
            )
        elif command == "status":
            status = "enabled" if self._tts_enabled else "disabled"
            return message.reply(
                payload=f"TTS: {status}",
                port=PortType.TEXT_OUT
            )

        return None

    async def _handle_text(self, message: Message) -> None:
        """Handle text input for synthesis."""
        # Only process if TTS is enabled and text is from LLM
        if not self._tts_enabled:
            return

        if message.source != "llm_brain":
            return

        text = str(message.payload)
        if not text.strip():
            return

        # Preprocess text for TTS
        text = self._preprocess_text(text)
        if not text:
            return

        # Synthesize and play asynchronously
        await self._speak(text)

    async def _enable_tts(self) -> None:
        """Enable TTS."""
        if self._tts_enabled:
            return

        self._tts_enabled = True
        self._logger.info("TTS enabled")

    async def _disable_tts(self) -> None:
        """Disable TTS."""
        if not self._tts_enabled:
            return

        self._tts_enabled = False

        # Stop any ongoing playback
        if self._audio_player:
            self._audio_player.stop()

        self._logger.info("TTS disabled")

    def _synthesize_and_play(self, text: str) -> None:
        """
        Synchronously synthesize and play one sentence.

        This runs in a thread and blocks until playback completes.
        No queues - direct streaming from synthesis to audio output.
        Prebuffers a few chunks before starting playback to prevent underruns.

        For Discord: emits audio chunks progressively as they are generated,
        allowing pseudo-streaming even in batch fallback mode.
        """
        with self._synth_lock:
            if not self._tts_enabled:
                return

            self._is_speaking = True
            self._logger.info(f"Synthesizing: {text[:50]}...")

            # Disable GC during synthesis+playback to prevent micro-freezes
            gc_was_enabled = gc.isenabled()
            gc.disable()

            try:
                # Check if we should play locally (not when Discord voice is active)
                play_locally = not self._discord_voice_active
                true_streaming = self._config.get("true_streaming", True)

                # For local playback, optional non-streaming mode (more stable, fewer underflows)
                if play_locally and not true_streaming:
                    audio = self._backend.synthesize(text)
                    self._audio_player.play(audio, blocking=True)
                    return

                # Prebuffer several chunks to reduce output underflow,
                # while still starting before full sentence is generated.
                prebuffer_chunks = max(1, int(self._config.get("prebuffer_chunks", 3)))
                startup_chunks = []
                stream_started = False
                chunk_count = 0

                # For Discord streaming: emit smaller chunks for lower latency
                sample_rate = self._config.get("sample_rate", 24000)
                # ~80ms chunks for ~200-300ms total latency with prebuffer
                emit_threshold_samples = int(sample_rate * 0.08)
                pending_chunks = []
                pending_samples = 0

                # Generate unique stream ID for this utterance
                stream_id = str(uuid.uuid4())[:8]

                for chunk in self._backend.synthesize_stream(text):
                    if not self._tts_enabled:
                        break

                    chunk_count += 1

                    # Only play locally if Discord voice is not active
                    if play_locally:
                        if not stream_started:
                            startup_chunks.append(chunk)
                            if len(startup_chunks) >= prebuffer_chunks:
                                # Flush prebuffer and start continuous playback
                                for c in startup_chunks:
                                    self._audio_player.add_audio(c)
                                startup_chunks = []
                                stream_started = True
                        else:
                            # Normal streaming after startup
                            self._audio_player.add_audio(chunk)

                    # Stream to Discord with smaller chunks
                    if self._discord_voice_active and self._loop:
                        pending_chunks.append(chunk)
                        pending_samples += len(chunk)

                        # Emit when we have enough audio accumulated
                        if pending_samples >= emit_threshold_samples:
                            self._emit_audio_chunk_streaming(
                                pending_chunks, sample_rate, stream_id, is_final=False
                            )
                            pending_chunks = []
                            pending_samples = 0

                # Handle short utterances that ended before prebuffer was reached
                if play_locally and startup_chunks:
                    for c in startup_chunks:
                        self._audio_player.add_audio(c)

                self._logger.debug(f"Synthesized {chunk_count} chunks, local_playback={play_locally}")

                # Emit remaining audio for Discord with is_final=True
                if pending_chunks and self._loop:
                    self._emit_audio_chunk_streaming(
                        pending_chunks, sample_rate, stream_id, is_final=True
                    )
                elif self._discord_voice_active and self._loop:
                    # No pending chunks but need to signal end of stream
                    self._emit_stream_end(stream_id)

                # Wait for playback to complete before returning (only if playing locally)
                if play_locally:
                    self._audio_player.wait_until_done()

            except Exception as e:
                self._logger.error(f"Synthesis error: {e}")
            finally:
                self._is_speaking = False
                # Re-enable GC after synthesis
                if gc_was_enabled:
                    gc.enable()
                    gc.collect()  # Run collection now, between sentences

    def _emit_audio_chunk(self, chunks: list, sample_rate: int) -> None:
        """Emit accumulated audio chunks to EventBus for Discord (legacy)."""
        if not chunks or not self._loop:
            return

        try:
            full_audio = np.concatenate(chunks)
            audio_bytes = (full_audio * 32767).astype(np.int16).tobytes()
            duration_ms = len(full_audio) / sample_rate * 1000

            self._logger.debug(f"Emitting AUDIO_OUT chunk: {len(audio_bytes)} bytes, {duration_ms:.0f}ms")

            audio_data = AudioData(
                data=audio_bytes,
                sample_rate=sample_rate,
                channels=1
            )
            asyncio.run_coroutine_threadsafe(
                self.emit(PortType.AUDIO_OUT, audio_data),
                self._loop
            )
        except Exception as e:
            self._logger.error(f"Error emitting audio chunk: {e}")

    def _emit_audio_chunk_streaming(
        self,
        chunks: list,
        sample_rate: int,
        stream_id: str,
        is_final: bool = False,
    ) -> None:
        """
        Emit audio chunk with streaming metadata for real-time Discord playback.

        Args:
            chunks: List of numpy arrays (float32)
            sample_rate: Audio sample rate
            stream_id: Unique identifier for this stream
            is_final: True if this is the last chunk of the stream
        """
        if not chunks or not self._loop:
            return

        try:
            full_audio = np.concatenate(chunks)
            audio_bytes = (full_audio * 32767).astype(np.int16).tobytes()
            duration_ms = len(full_audio) / sample_rate * 1000

            self._logger.debug(
                f"Streaming chunk [{stream_id}]: {len(audio_bytes)} bytes, "
                f"{duration_ms:.0f}ms, final={is_final}"
            )

            audio_data = AudioData(
                data=audio_bytes,
                sample_rate=sample_rate,
                channels=1,
            )

            # Use publish_direct for lower latency
            message = Message(
                source=self.name,
                port=PortType.AUDIO_OUT,
                payload=audio_data,
                metadata={
                    "streaming": True,
                    "stream_id": stream_id,
                    "is_final": is_final,
                },
            )

            asyncio.run_coroutine_threadsafe(
                self._kernel.event_bus.publish_direct(message),
                self._loop,
            )
        except Exception as e:
            self._logger.error(f"Error emitting streaming chunk: {e}")

    def _emit_stream_end(self, stream_id: str) -> None:
        """
        Emit end-of-stream signal without audio data.

        Used when the last actual audio chunk was already emitted
        but we need to signal stream completion.
        """
        if not self._loop:
            return

        try:
            self._logger.debug(f"Emitting stream end [{stream_id}]")

            message = Message(
                source=self.name,
                port=PortType.AUDIO_OUT,
                payload=None,
                metadata={
                    "streaming": True,
                    "stream_id": stream_id,
                    "is_final": True,
                    "empty": True,
                },
            )

            asyncio.run_coroutine_threadsafe(
                self._kernel.event_bus.publish_direct(message),
                self._loop,
            )
        except Exception as e:
            self._logger.error(f"Error emitting stream end: {e}")

    async def _speak(self, text: str) -> None:
        """
        Synthesize and play text synchronously.

        Blocks until synthesis and playback complete.
        Only one sentence is processed at a time - no queue buildup.
        """
        if not self._backend or not self._backend.is_loaded:
            self._logger.error("TTS backend not ready")
            return

        # Run synthesis+playback in executor, blocking until done
        await self._loop.run_in_executor(None, self._synthesize_and_play, text)

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for TTS synthesis.

        Removes markdown formatting, code blocks, etc.
        """
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)

        # Remove inline code
        text = re.sub(r'`[^`]+`', '', text)

        # Remove markdown bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # Remove markdown links, keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # Remove markdown headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    # === Public API ===

    @property
    def is_tts_enabled(self) -> bool:
        """Check if TTS is currently enabled."""
        return self._tts_enabled

    async def enable_tts(self) -> None:
        """Enable TTS programmatically."""
        await self._enable_tts()

    async def disable_tts(self) -> None:
        """Disable TTS programmatically."""
        await self._disable_tts()

    async def speak(self, text: str) -> None:
        """Speak text programmatically."""
        if not self._tts_enabled:
            await self._enable_tts()

        text = self._preprocess_text(text)
        if text:
            await self._speak(text)

    def stop_speaking(self) -> None:
        """Stop current speech playback."""
        if self._audio_player:
            self._audio_player.stop()

    def set_voice(self, audio_path: str) -> None:
        """
        Set the voice from a reference audio file.

        Args:
            audio_path: Path to reference audio file
        """
        if self._backend and hasattr(self._backend, 'set_voice'):
            self._backend.set_voice(audio_path)
        else:
            raise NotImplementedError("Current backend doesn't support voice cloning")

    def set_volume(self, volume: float) -> None:
        """Set playback volume (0.0 to 1.0)."""
        if self._audio_player:
            self._audio_player.volume = max(0.0, min(1.0, volume))

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the TTS backend."""
        if self._backend:
            return self._backend.get_model_info()
        return {"loaded": False}

    @staticmethod
    def list_audio_devices():
        """List available audio output devices."""
        return StreamingAudioPlayer.list_output_devices()
