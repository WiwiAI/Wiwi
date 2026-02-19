"""llama.cpp server HTTP backend for LLM module."""

import asyncio
import json
from typing import Any, Dict, Generator, List, Optional
import logging

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from wiwi.modules.llm_brain.backends.base import BaseLLMBackend


class LlamaServerBackend(BaseLLMBackend):
    """
    Backend for llama.cpp server via HTTP API.

    Uses OpenAI-compatible /v1/chat/completions endpoint.

    Config options:
        api_url: Server URL (default: http://localhost:8080)
        timeout: Request timeout in seconds (default: 120)
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 512)
        system_prompt: System prompt for the model
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._logger = logging.getLogger("wiwi.llm.llama_server")

        # Server config
        self._api_url = config.get("api_url", "http://localhost:8080")
        self._timeout = config.get("timeout", 120)

        # Generation params
        self._temperature = config.get("temperature", 0.7)
        self._top_p = config.get("top_p", 0.9)
        self._max_tokens = config.get("max_tokens", 512)
        self._system_prompt = config.get("system_prompt", "")

        # Session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None

    def load_model(self) -> None:
        """Check connection to llama-server."""
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for llama-server backend. "
                "Install it with: pip install aiohttp"
            )
        self._model = True
        self._logger.info(f"Configured to connect to llama-server at {self._api_url}")

    def unload_model(self) -> None:
        """Close HTTP session."""
        self._model = None
        self._logger.info("Disconnected from llama-server")

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close_session(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            session = await self._ensure_session()
            async with session.get(f"{self._api_url}/health") as resp:
                return resp.status == 200
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False

    def generate(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """Synchronous generation (runs async internally)."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.generate_async(prompt, context, **kwargs)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                self.generate_async(prompt, context, **kwargs)
            )

    async def generate_async(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate response via OpenAI-compatible /v1/chat/completions endpoint.
        """
        if not self._model:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        # Build messages
        messages = self._build_messages(prompt, context)

        # Request payload
        payload = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
            "top_p": kwargs.get("top_p", self._top_p),
            "stream": False
        }

        self._logger.debug(f"Sending request to {self._api_url}/v1/chat/completions")

        try:
            session = await self._ensure_session()
            async with session.post(
                f"{self._api_url}/v1/chat/completions",
                json=payload
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Server error {resp.status}: {error_text}")

                result = await resp.json()

                # Extract response from OpenAI format
                choices = result.get("choices", [])
                if choices:
                    response_text = choices[0].get("message", {}).get("content", "")
                else:
                    response_text = ""

                self._logger.debug(f"Generated {len(response_text)} chars")
                return response_text.strip()

        except asyncio.TimeoutError:
            raise RuntimeError(f"Request timeout after {self._timeout}s")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Connection error: {e}")

    async def generate_stream_async(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ):
        """
        Generate streaming response via /completion endpoint.
        Yields tokens as they are generated.
        """
        if not self._model:
            raise RuntimeError("Backend not initialized")

        # Build full prompt text for /completion endpoint
        full_prompt = self._build_prompt_text(prompt, context)

        payload = {
            "prompt": full_prompt,
            "n_predict": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
            "top_p": kwargs.get("top_p", self._top_p),
            "stream": True,
            # Stop tokens to prevent model from generating user messages
            "stop": ["[INST]", "</s>", "<s>", "<<SYS>>"]
        }

        try:
            session = await self._ensure_session()
            async with session.post(
                f"{self._api_url}/completion",
                json=payload
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Server error {resp.status}: {error_text}")

                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        try:
                            chunk = json.loads(data)
                            # llama-server uses "stop": true to indicate end
                            if chunk.get("stop", False):
                                break
                            token = chunk.get("content", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue

        except asyncio.TimeoutError:
            raise RuntimeError(f"Stream timeout after {self._timeout}s")

    def generate_stream(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Sync wrapper for streaming (not recommended, use async)."""
        yield self.generate(prompt, context, **kwargs)

    def _build_messages(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Build messages array for OpenAI-compatible API.
        Ensures proper alternation: system (optional) -> user -> assistant -> user -> ...
        """
        messages = []

        # System prompt (optional, must be first if present)
        if self._system_prompt:
            messages.append({
                "role": "system",
                "content": self._system_prompt.strip()
            })

        # Conversation context - ensure alternation
        if context:
            for turn in context:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    # Skip if same role as last message (avoid duplicates)
                    if messages and messages[-1]["role"] == role:
                        continue
                    messages.append({"role": role, "content": content})

        # Current user prompt - check we don't have two users in a row
        if messages and messages[-1]["role"] == "user":
            # Last message was user, combine or skip
            messages[-1]["content"] += "\n" + prompt
        else:
            messages.append({"role": "user", "content": prompt})

        return messages

    def _build_prompt_text(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build full prompt text for /completion endpoint.
        Uses Mistral/Llama2 instruct format: <s>[INST] message [/INST] response</s>
        """
        parts = []

        # System prompt as first instruction (if present)
        system_prefix = ""
        if self._system_prompt:
            system_prefix = f"<<SYS>>\n{self._system_prompt.strip()}\n<</SYS>>\n\n"

        # Build conversation in Mistral format
        # Format: <s>[INST] {system} {user1} [/INST] {assistant1}</s><s>[INST] {user2} [/INST] ...

        if context:
            first_turn = True
            for i, turn in enumerate(context):
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if not content:
                    continue

                if role == "user":
                    if first_turn and system_prefix:
                        parts.append(f"<s>[INST] {system_prefix}{content} [/INST]")
                        first_turn = False
                    else:
                        parts.append(f"<s>[INST] {content} [/INST]")
                elif role == "assistant":
                    # Assistant response closes the turn
                    if parts:
                        parts[-1] = parts[-1] + f" {content}</s>"

            # Add current user prompt
            parts.append(f"<s>[INST] {prompt} [/INST]")
        else:
            # No context - just the current prompt with optional system
            if system_prefix:
                parts.append(f"<s>[INST] {system_prefix}{prompt} [/INST]")
            else:
                parts.append(f"<s>[INST] {prompt} [/INST]")

        return "".join(parts)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            "backend": "llama_server",
            "api_url": self._api_url,
            "endpoint": "/v1/chat/completions"
        })
        return info
