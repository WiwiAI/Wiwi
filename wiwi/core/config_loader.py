"""Configuration loader for Wiwi4.0."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging

import yaml

from wiwi.core.exceptions import ConfigNotFoundError, ConfigValidationError


class ConfigLoader:
    """
    Configuration loader with support for:
    - YAML configuration files
    - Environment variable substitution
    - Default values
    - Configuration merging

    Usage:
        loader = ConfigLoader(Path("config/default.yaml"))
        config = loader.load()
        value = loader.get("modules.llm_brain.model_path")
    """

    DEFAULT_CONFIG_PATHS = [
        Path("config/default.yaml"),
        Path("config/config.yaml"),
        Path.home() / ".config/wiwi/config.yaml",
    ]

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the config loader.

        Args:
            config_path: Path to configuration file. If None, searches default locations.
        """
        self._config_path = config_path
        self._config: Dict[str, Any] = {}
        self._logger = logging.getLogger("wiwi.config")

    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file.

        Returns:
            Loaded configuration dictionary

        Raises:
            ConfigNotFoundError: If no configuration file found
        """
        # Find config file
        config_path = self._find_config_file()

        if config_path is None:
            self._logger.warning("No configuration file found, using defaults")
            self._config = self._get_default_config()
            return self._config

        self._logger.info(f"Loading configuration from: {config_path}")

        # Load YAML
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigValidationError([f"Invalid YAML: {e}"])

        # Apply environment variable substitution
        self._config = self._substitute_env_vars(self._config)

        # Merge with defaults
        default_config = self._get_default_config()
        self._config = self._deep_merge(default_config, self._config)

        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Supports dot notation for nested keys: "modules.llm_brain.model_path"

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key (dot notation supported)
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def _find_config_file(self) -> Optional[Path]:
        """Find the configuration file."""
        if self._config_path and self._config_path.exists():
            return self._config_path

        for path in self.DEFAULT_CONFIG_PATHS:
            if path.exists():
                return path

        return None

    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in config.

        Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax.
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._expand_env_string(config)
        return config

    def _expand_env_string(self, value: str) -> str:
        """Expand environment variables in a string."""
        import re

        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(3) if match.group(3) else ""

            # Handle :- syntax for default value
            if ":-" in var_name:
                var_name, default_value = var_name.split(":-", 1)

            return os.environ.get(var_name, default_value)

        # Match ${VAR} or ${VAR:-default}
        pattern = r"\$\{([^}:]+)(:-([^}]*))?\}"
        return re.sub(pattern, replace_env_var, value)

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.

        Override values take precedence over base values.
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "general": {
                "name": "Wiwi",
                "version": "4.0.0",
                "language": "ru",
                "log_level": "INFO"
            },
            "enabled_modules": [
                "memory",
                "cli_interface"
            ],
            "modules": {
                "llm_brain": {
                    "backend": "llama_cpp",
                    "model_path": "./models/model.gguf",
                    "context_length": 4096,
                    "n_gpu_layers": 0,
                    "n_threads": 4,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 512
                },
                "memory": {
                    "backend": "in_memory",
                    "max_history_length": 20,
                    "max_tokens_per_turn": 500
                },
                "cli_interface": {
                    "prompt": ">>> ",
                    "colors": True,
                    "history_file": "~/.wiwi_history"
                },
                "stt": {
                    "backend": "whisper",
                    "model_size": "base",
                    "language": "ru",
                    "device": "cpu"
                },
                "tts": {
                    "backend": "silero",
                    "model": "v3_1_ru",
                    "speaker": "aidar",
                    "sample_rate": 48000
                }
            },
            "paths": {
                "models_dir": "./models",
                "plugins_dir": "./plugins",
                "cache_dir": "~/.cache/wiwi",
                "logs_dir": "./logs"
            }
        }

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save current configuration to file.

        Args:
            path: Path to save to. Uses original path if None.
        """
        save_path = path or self._config_path
        if save_path is None:
            save_path = Path("config/config.yaml")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

        self._logger.info(f"Configuration saved to: {save_path}")
