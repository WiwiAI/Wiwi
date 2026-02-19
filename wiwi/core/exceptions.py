"""Custom exceptions for Wiwi4.0."""


class WiwiError(Exception):
    """Base exception for all Wiwi4.0 errors."""
    pass


class ModuleError(WiwiError):
    """Base exception for module-related errors."""
    pass


class ModuleNotFoundError(ModuleError):
    """Raised when a requested module is not found."""

    def __init__(self, module_name: str):
        self.module_name = module_name
        super().__init__(f"Module '{module_name}' not found")


class ModuleLoadError(ModuleError):
    """Raised when a module fails to load."""

    def __init__(self, module_name: str, reason: str):
        self.module_name = module_name
        self.reason = reason
        super().__init__(f"Failed to load module '{module_name}': {reason}")


class ModuleDependencyError(ModuleError):
    """Raised when module dependencies cannot be resolved."""

    def __init__(self, module_name: str, missing_deps: list):
        self.module_name = module_name
        self.missing_deps = missing_deps
        super().__init__(
            f"Module '{module_name}' has unresolved dependencies: {missing_deps}"
        )


class ModuleStateError(ModuleError):
    """Raised when module is in invalid state for operation."""

    def __init__(self, module_name: str, current_state: str, expected_states: list):
        self.module_name = module_name
        self.current_state = current_state
        self.expected_states = expected_states
        super().__init__(
            f"Module '{module_name}' is in state '{current_state}', "
            f"expected one of: {expected_states}"
        )


class ConfigError(WiwiError):
    """Base exception for configuration errors."""
    pass


class ConfigNotFoundError(ConfigError):
    """Raised when configuration file is not found."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        super().__init__(f"Configuration file not found: {config_path}")


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""

    def __init__(self, errors: list):
        self.errors = errors
        super().__init__(f"Configuration validation failed: {errors}")


class EventBusError(WiwiError):
    """Base exception for event bus errors."""
    pass


class MessageDeliveryError(EventBusError):
    """Raised when message cannot be delivered."""

    def __init__(self, message_id: str, reason: str):
        self.message_id = message_id
        self.reason = reason
        super().__init__(f"Failed to deliver message '{message_id}': {reason}")


class PipelineError(WiwiError):
    """Base exception for pipeline errors."""
    pass


class PipelineNotFoundError(PipelineError):
    """Raised when requested pipeline is not found."""

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        super().__init__(f"Pipeline '{pipeline_name}' not found")
