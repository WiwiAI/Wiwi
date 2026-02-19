"""llama-cpp-python backend for LLM module."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Generator
import logging

from wiwi.modules.llm_brain.backends.base import BaseLLMBackend


class LlamaCppBackend(BaseLLMBackend):
    """
    Backend for GGUF models using llama-cpp-python.

    Config options:
        model_path: Path to GGUF model file
        context_length: Context window size (default: 4096)
        n_gpu_layers: Number of layers to offload to GPU (default: 0)
        n_threads: Number of CPU threads (default: 4)
        temperature: Sampling temperature (default: 0.7)
        top_p: Top-p sampling (default: 0.9)
        top_k: Top-k sampling (default: 40)
        max_tokens: Maximum tokens to generate (default: 512)
        repeat_penalty: Repetition penalty (default: 1.1)
        system_prompt: System prompt for the model
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._logger = logging.getLogger("wiwi.llm.llama_cpp")

        # Extract config
        self._model_path = Path(config.get("model_path", "./models/model.gguf")).expanduser()
        self._context_length = config.get("context_length", 4096)
        self._n_gpu_layers = config.get("n_gpu_layers", 0)
        self._n_threads = config.get("n_threads", 4)
        self._temperature = config.get("temperature", 0.7)
        self._top_p = config.get("top_p", 0.9)
        self._top_k = config.get("top_k", 40)
        self._max_tokens = config.get("max_tokens", 512)
        self._repeat_penalty = config.get("repeat_penalty", 1.1)
        self._system_prompt = config.get("system_prompt", "")

    def load_model(self) -> None:
        """Load the GGUF model."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python"
            )

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self._model_path}\n"
                f"Please download a GGUF model and place it in the models directory."
            )

        self._logger.info(f"Loading model: {self._model_path}")

        self._model = Llama(
            model_path=str(self._model_path),
            n_ctx=self._context_length,
            n_gpu_layers=self._n_gpu_layers,
            n_threads=self._n_threads,
            verbose=False
        )

        self._logger.info("Model loaded successfully")

    def unload_model(self) -> None:
        """Unload the model."""
        if self._model:
            del self._model
            self._model = None
            self._logger.info("Model unloaded")

    def generate(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """Generate a response."""
        if not self._model:
            raise RuntimeError("Model not loaded")

        # Build full prompt with context
        full_prompt = self._build_prompt(prompt, context)

        # Override parameters from kwargs
        temperature = kwargs.get("temperature", self._temperature)
        top_p = kwargs.get("top_p", self._top_p)
        top_k = kwargs.get("top_k", self._top_k)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        repeat_penalty = kwargs.get("repeat_penalty", self._repeat_penalty)

        # Generate
        self._logger.debug(f"Generating response (max_tokens={max_tokens})")

        output = self._model(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=["</s>", "User:", "Human:", "\n\n\n"],
            echo=False
        )

        response = output["choices"][0]["text"].strip()

        self._logger.debug(f"Generated {len(response)} chars")
        return response

    def generate_stream(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate a streaming response."""
        if not self._model:
            raise RuntimeError("Model not loaded")

        full_prompt = self._build_prompt(prompt, context)

        temperature = kwargs.get("temperature", self._temperature)
        top_p = kwargs.get("top_p", self._top_p)
        top_k = kwargs.get("top_k", self._top_k)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        repeat_penalty = kwargs.get("repeat_penalty", self._repeat_penalty)

        for output in self._model(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=["</s>", "User:", "Human:", "\n\n\n"],
            echo=False,
            stream=True
        ):
            token = output["choices"][0]["text"]
            if token:
                yield token

    def _build_prompt(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build the full prompt with system prompt and context.

        Uses a simple chat format compatible with most models.
        """
        parts = []

        # System prompt
        if self._system_prompt:
            parts.append(f"System: {self._system_prompt}\n")

        # Conversation context
        if context:
            for turn in context[:-1]:  # Exclude the last one (current prompt)
                role = turn.get("role", "user")
                content = turn.get("content", "")

                if role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
                elif role == "system":
                    parts.append(f"System: {content}")

        # Current prompt
        parts.append(f"User: {prompt}")
        parts.append("Assistant:")

        return "\n".join(parts)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            "model_path": str(self._model_path),
            "context_length": self._context_length,
            "n_gpu_layers": self._n_gpu_layers,
            "n_threads": self._n_threads
        })
        return info
