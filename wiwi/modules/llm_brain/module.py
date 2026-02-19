"""LLM Brain module for Wiwi4.0."""

import asyncio
from typing import Any, Dict, Optional

from wiwi.interfaces.base_module import BaseModule, ModuleInfo, ModuleState
from wiwi.interfaces.ports import PortType
from wiwi.interfaces.messages import Message
from wiwi.modules.llm_brain.backends.base import BaseLLMBackend


class LLMBrainModule(BaseModule):
    """
    LLM Brain module - the "brain" of the assistant.

    Handles text generation using local LLM models.

    Backends:
    - llama_server: llama.cpp server via HTTP API (recommended for GPU)
    - llama_cpp: llama-cpp-python library (in-process)

    Config options:
        backend: Backend to use ("llama_server" or "llama_cpp")

        For llama_server:
            api_url: Server URL (default: http://localhost:8080)
            timeout: Request timeout (default: 120)

        For llama_cpp:
            model_path: Path to GGUF model
            n_gpu_layers: GPU layers for acceleration

        Common:
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 512)
            system_prompt: System prompt
    """

    @property
    def module_info(self) -> ModuleInfo:
        return ModuleInfo(
            name="llm_brain",
            version="1.1.0",
            description="LLM processing via llama.cpp server or library",
            author="Wiwi Team",
            category="ai",
            input_ports={PortType.TEXT_IN},
            output_ports={PortType.TEXT_OUT},
            dependencies=set(),
            optional_dependencies={"memory"},
            config_schema={
                "backend": {
                    "type": "string",
                    "enum": ["llama_server", "llama_cpp"],
                    "default": "llama_server"
                },
                "api_url": {"type": "string", "default": "http://localhost:8080"},
                "model_path": {"type": "string"},
                "temperature": {"type": "float", "default": 0.7},
                "max_tokens": {"type": "integer", "default": 512}
            }
        )

    def __init__(self, kernel, config: Dict[str, Any]):
        super().__init__(kernel, config)
        self._backend: Optional[BaseLLMBackend] = None
        self._backend_name = config.get("backend", "llama_server")

    async def initialize(self) -> None:
        """Initialize the LLM module."""
        self._logger.info(f"Initializing LLM module with backend: {self._backend_name}")

        # Create backend
        if self._backend_name == "llama_server":
            from wiwi.modules.llm_brain.backends.llama_server import LlamaServerBackend
            self._backend = LlamaServerBackend(self._config)
        elif self._backend_name == "llama_cpp":
            from wiwi.modules.llm_brain.backends.llama_cpp import LlamaCppBackend
            self._backend = LlamaCppBackend(self._config)
        else:
            raise ValueError(f"Unknown backend: {self._backend_name}")

        # Initialize backend
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._backend.load_model)

        # For llama_server, check health
        if self._backend_name == "llama_server":
            healthy = await self._backend.health_check()
            if healthy:
                self._logger.info(f"Connected to llama-server at {self._config.get('api_url')}")
            else:
                self._logger.warning(
                    f"llama-server not responding at {self._config.get('api_url')}. "
                    "Make sure it's running."
                )

        self._logger.info("LLM module initialized")

    async def start(self) -> None:
        """Start the LLM module."""
        self._state = ModuleState.RUNNING

    async def stop(self) -> None:
        """Stop and cleanup."""
        if self._backend:
            # Close HTTP session for llama_server
            if hasattr(self._backend, 'close_session'):
                await self._backend.close_session()
            self._backend.unload_model()
        self._state = ModuleState.STOPPED

    async def handle_input(self, message: Message) -> Optional[Message]:
        """
        Handle incoming text generation requests with streaming.

        Expects TEXT_IN messages with:
        - payload: The user prompt
        - metadata.context: Optional conversation context

        Streams sentences to TTS as they complete, then sends full response.
        """
        if message.port != PortType.TEXT_IN:
            return None

        # Only process messages that have been routed through memory module
        if message.source != "memory":
            return None

        if not self._backend or not self._backend.is_loaded:
            self._logger.error("Backend not ready")
            return message.reply(
                payload="Error: LLM backend not ready",
                port=PortType.TEXT_OUT
            )

        prompt = str(message.payload)
        context = message.metadata.get("context", [])

        try:
            # Use streaming generation
            full_response = ""
            sentence_buffer = ""
            sentence_endings = ".!?"  # Only split on sentence-ending punctuation

            # Print prefix for voice mode
            if message.metadata.get("source") == "voice":
                print("[Voice] Wiwi: ", end="", flush=True)

            async for token in self.generate_stream(prompt, context):
                # Print token for real-time feedback
                if message.metadata.get("source") == "voice":
                    print(token, end="", flush=True)

                full_response += token
                sentence_buffer += token

                # Check for complete sentence to send to TTS
                if self._event_bus:
                    for i, char in enumerate(sentence_buffer):
                        if char in sentence_endings:
                            if i == len(sentence_buffer) - 1 or sentence_buffer[i + 1] in ' \n':
                                sentence = sentence_buffer[:i + 1].strip()
                                sentence_buffer = sentence_buffer[i + 1:].lstrip()

                                if sentence and len(sentence) > 2:
                                    # Send sentence to TTS immediately (direct dispatch)
                                    tts_message = Message(
                                        source=self.name,
                                        port=PortType.AUDIO_OUT,  # Special port for TTS streaming
                                        payload=sentence,
                                        target="tts",
                                        metadata={"streaming": True},
                                        correlation_id=message.correlation_id
                                    )
                                    await self._event_bus.publish_direct(tts_message)
                                break

            # Newline after streaming
            if message.metadata.get("source") == "voice":
                print()

            # Send remaining buffer to TTS
            if self._event_bus and sentence_buffer.strip():
                tts_message = Message(
                    source=self.name,
                    port=PortType.AUDIO_OUT,
                    payload=sentence_buffer.strip(),
                    target="tts",
                    metadata={"streaming": True, "final": True},
                    correlation_id=message.correlation_id
                )
                await self._event_bus.publish_direct(tts_message)

        except Exception as e:
            self._logger.error(f"Generation error: {e}")
            return message.reply(
                payload=f"Error generating response: {e}",
                port=PortType.TEXT_OUT
            )

        # Create final response message (for memory storage, no TTS - already sent)
        response = Message(
            source=self.name,
            target=None,  # Broadcast
            port=PortType.TEXT_OUT,
            payload=full_response,
            metadata={
                "backend": self._backend_name,
                "prompt_length": len(prompt),
                "response_length": len(full_response),
                "streamed": True  # Mark as already streamed to TTS
            },
            correlation_id=message.correlation_id
        )

        return response

    # === Public API ===

    async def generate(
        self,
        prompt: str,
        context: Optional[list] = None,
        **kwargs
    ) -> str:
        """
        Generate a response directly (for programmatic use).

        Args:
            prompt: User prompt
            context: Conversation context
            **kwargs: Additional generation parameters

        Returns:
            Generated response text
        """
        if not self._backend or not self._backend.is_loaded:
            raise RuntimeError("Backend not ready")

        if self._backend_name == "llama_server":
            return await self._backend.generate_async(prompt, context, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._backend.generate(prompt, context, **kwargs)
            )

    async def generate_stream(self, prompt: str, context: Optional[list] = None, **kwargs):
        """
        Generate a streaming response.

        Yields tokens as they are generated (only for llama_server backend).
        """
        if not self._backend or not self._backend.is_loaded:
            raise RuntimeError("Backend not ready")

        if self._backend_name == "llama_server":
            async for token in self._backend.generate_stream_async(prompt, context, **kwargs):
                yield token
        else:
            # Fallback: yield full response
            response = await self.generate(prompt, context, **kwargs)
            yield response

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the backend."""
        if self._backend:
            return self._backend.get_model_info()
        return {"loaded": False}

    @property
    def is_model_loaded(self) -> bool:
        """Check if backend is ready."""
        return self._backend is not None and self._backend.is_loaded