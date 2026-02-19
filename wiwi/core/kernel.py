"""Central kernel (constructor) for Wiwi4.0."""

import asyncio
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type
import logging

from wiwi.interfaces.base_module import BaseModule, ModuleInfo, ModuleState
from wiwi.interfaces.ports import PortType
from wiwi.interfaces.messages import Message
from wiwi.core.event_bus import EventBus
from wiwi.core.registry import ModuleRegistry
from wiwi.core.config_loader import ConfigLoader
from wiwi.core.exceptions import (
    ModuleNotFoundError,
    ModuleLoadError,
    ModuleDependencyError,
    ModuleStateError
)


class Kernel:
    """
    Central kernel (Constructor) of Wiwi4.0.

    The kernel is responsible for:
    1. Loading and validating configuration
    2. Discovering and registering modules
    3. Managing module lifecycle
    4. Resolving module dependencies
    5. Coordinating message routing

    Usage:
        kernel = Kernel(config_path)
        await kernel.start()

        # Access modules
        llm = kernel.get_module("llm_brain")

        # Hot reload modules
        await kernel.reload_module("my_module")

        await kernel.stop()
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the kernel.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self._config_loader = ConfigLoader(config_path)
        self._config = self._config_loader.load()

        # Initialize components
        self._event_bus = EventBus()
        self._registry = ModuleRegistry()

        # Module storage
        self._modules: Dict[str, BaseModule] = {}
        self._module_classes: Dict[str, Type[BaseModule]] = {}

        # State
        self._running = False
        self._logger = logging.getLogger("wiwi.kernel")

        # Callbacks
        self._on_module_loaded: List[Callable[[str], None]] = []
        self._on_module_unloaded: List[Callable[[str], None]] = []

    # === Properties ===

    @property
    def event_bus(self) -> EventBus:
        """Get the event bus."""
        return self._event_bus

    @property
    def registry(self) -> ModuleRegistry:
        """Get the module registry."""
        return self._registry

    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Check if kernel is running."""
        return self._running

    # === Module Discovery ===

    async def discover_modules(self) -> None:
        """
        Discover all available modules.

        Searches in:
        1. wiwi.modules.* - built-in modules
        2. plugins/ - external plugins
        """
        # Built-in modules
        builtin_path = Path(__file__).parent.parent / "modules"
        await self._scan_directory(builtin_path, "wiwi.modules")

        # External plugins
        plugins_path = Path(self._config.get("paths", {}).get("plugins_dir", "./plugins"))
        if plugins_path.exists():
            await self._scan_directory(plugins_path, "plugins")

        self._logger.info(
            f"Discovered {len(self._module_classes)} modules: "
            f"{list(self._module_classes.keys())}"
        )

    async def _scan_directory(self, path: Path, package_prefix: str) -> None:
        """Scan a directory for modules."""
        if not path.exists():
            return

        for module_dir in path.iterdir():
            if not module_dir.is_dir() or module_dir.name.startswith("_"):
                continue

            module_file = module_dir / "module.py"
            if not module_file.exists():
                continue

            try:
                await self._load_module_class(module_dir, package_prefix)
            except Exception as e:
                self._logger.error(f"Error scanning {module_dir}: {e}")

    async def _load_module_class(self, module_dir: Path, package_prefix: str) -> None:
        """Load a module class from directory."""
        module_file = module_dir / "module.py"
        full_module_name = f"{package_prefix}.{module_dir.name}.module"

        # Load the module
        spec = importlib.util.spec_from_file_location(full_module_name, module_file)
        if spec is None or spec.loader is None:
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find BaseModule subclass
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseModule)
                and attr is not BaseModule
            ):
                # Get module name from module_info
                # Create temporary instance to get info
                try:
                    # Use object.__new__ to avoid __init__
                    temp = object.__new__(attr)
                    temp._config = {}
                    temp._kernel = None
                    temp._state = ModuleState.UNINITIALIZED
                    temp._event_bus = None
                    temp._logger = logging.getLogger("temp")

                    module_name = temp.module_info.name
                    module_info = temp.module_info

                    self._module_classes[module_name] = attr
                    self._registry.register_class(module_name, attr, module_info)

                    self._logger.debug(f"Discovered module: {module_name}")
                except Exception as e:
                    self._logger.error(f"Error getting module info from {attr_name}: {e}")
                break

    # === Module Loading ===

    async def load_module(self, module_name: str) -> Optional[BaseModule]:
        """
        Load and initialize a module.

        Args:
            module_name: Name of the module to load

        Returns:
            Loaded module instance or None on error

        Raises:
            ModuleNotFoundError: If module not found
            ModuleLoadError: If module fails to load
        """
        # Already loaded?
        if module_name in self._modules:
            return self._modules[module_name]

        # Find module class
        if module_name not in self._module_classes:
            raise ModuleNotFoundError(module_name)

        module_class = self._module_classes[module_name]
        module_config = self._config.get("modules", {}).get(module_name, {})

        try:
            # Create instance
            module = module_class(self, module_config)
            module._event_bus = self._event_bus
            module._logger = logging.getLogger(f"wiwi.module.{module_name}")

            # Resolve dependencies first
            await self._resolve_dependencies(module)

            # Initialize
            module._state = ModuleState.INITIALIZING
            await module.initialize()
            module._state = ModuleState.READY

            # Register
            self._modules[module_name] = module
            self._registry.register_instance(module_name, module)

            # Subscribe to events
            self._event_bus.subscribe(
                module_name,
                module.handle_input,
                ports=module.module_info.input_ports
            )

            self._logger.info(f"Module loaded: {module_name}")

            # Notify callbacks
            for callback in self._on_module_loaded:
                callback(module_name)

            return module

        except Exception as e:
            self._logger.error(f"Failed to load module {module_name}: {e}")
            raise ModuleLoadError(module_name, str(e))

    async def _resolve_dependencies(self, module: BaseModule) -> None:
        """Resolve module dependencies."""
        info = module.module_info

        # Required dependencies
        for dep_name in info.dependencies:
            if dep_name not in self._modules:
                if dep_name not in self._module_classes:
                    raise ModuleDependencyError(module.name, [dep_name])
                await self.load_module(dep_name)

        # Optional dependencies
        for dep_name in info.optional_dependencies:
            if dep_name not in self._modules and dep_name in self._module_classes:
                try:
                    await self.load_module(dep_name)
                except Exception as e:
                    self._logger.warning(
                        f"Optional dependency {dep_name} failed to load: {e}"
                    )

    async def unload_module(self, module_name: str) -> bool:
        """
        Unload a module.

        Args:
            module_name: Name of the module to unload

        Returns:
            True if successfully unloaded
        """
        if module_name not in self._modules:
            return False

        module = self._modules[module_name]

        # Check for dependents
        dependents = self._registry.get_dependents(module_name)
        loaded_dependents = [d for d in dependents if d in self._modules]
        if loaded_dependents:
            self._logger.error(
                f"Cannot unload {module_name}: depended on by {loaded_dependents}"
            )
            return False

        try:
            # Stop module
            module._state = ModuleState.STOPPING
            await module.stop()
            module._state = ModuleState.STOPPED

            # Unsubscribe
            self._event_bus.unsubscribe(module_name)

            # Unregister
            del self._modules[module_name]
            self._registry.unregister_instance(module_name)

            self._logger.info(f"Module unloaded: {module_name}")

            # Notify callbacks
            for callback in self._on_module_unloaded:
                callback(module_name)

            return True

        except Exception as e:
            self._logger.error(f"Error unloading module {module_name}: {e}")
            module._state = ModuleState.ERROR
            return False

    async def reload_module(self, module_name: str) -> Optional[BaseModule]:
        """
        Hot-reload a module.

        Args:
            module_name: Name of the module to reload

        Returns:
            Reloaded module instance
        """
        self._logger.info(f"Reloading module: {module_name}")

        # Unload if loaded
        if module_name in self._modules:
            await self.unload_module(module_name)

        # Re-discover module class (in case code changed)
        if module_name in self._module_classes:
            # Get the module directory
            for path in [
                Path(__file__).parent.parent / "modules",
                Path(self._config.get("paths", {}).get("plugins_dir", "./plugins"))
            ]:
                module_dir = path / module_name
                if module_dir.exists():
                    # Clear from module classes
                    del self._module_classes[module_name]
                    self._registry.unregister_class(module_name)

                    # Reload
                    prefix = "wiwi.modules" if "modules" in str(path) else "plugins"
                    await self._load_module_class(module_dir, prefix)
                    break

        # Load fresh
        return await self.load_module(module_name)

    # === Lifecycle ===

    async def start(self) -> None:
        """Start the kernel and all enabled modules."""
        self._logger.info("Starting Wiwi4.0 Kernel...")

        # Start event bus
        await self._event_bus.start()

        # Discover modules
        await self.discover_modules()

        # Load enabled modules
        enabled = self._config.get("enabled_modules", [])
        for module_name in enabled:
            try:
                await self.load_module(module_name)
            except Exception as e:
                self._logger.error(f"Failed to load enabled module {module_name}: {e}")

        # Start all loaded modules
        for module in self._modules.values():
            if module.state == ModuleState.READY:
                try:
                    await module.start()
                    module._state = ModuleState.RUNNING
                except Exception as e:
                    self._logger.error(f"Failed to start module {module.name}: {e}")
                    module._state = ModuleState.ERROR

        self._running = True
        self._logger.info(
            f"Wiwi4.0 Kernel started with {len(self._modules)} modules"
        )

    async def stop(self) -> None:
        """Stop the kernel and all modules."""
        self._logger.info("Stopping Wiwi4.0 Kernel...")
        self._running = False

        # Stop modules in reverse order
        for module_name in reversed(list(self._modules.keys())):
            try:
                await self.unload_module(module_name)
            except Exception as e:
                self._logger.error(f"Error stopping module {module_name}: {e}")

        # Stop event bus
        await self._event_bus.stop()

        self._logger.info("Wiwi4.0 Kernel stopped")

    # === Module Access ===

    def get_module(self, module_name: str) -> Optional[BaseModule]:
        """
        Get a loaded module by name.

        Args:
            module_name: Module name

        Returns:
            Module instance or None
        """
        return self._modules.get(module_name)

    def list_modules(self) -> Dict[str, ModuleInfo]:
        """
        Get info for all loaded modules.

        Returns:
            Dict of module name -> ModuleInfo
        """
        return {
            name: module.module_info
            for name, module in self._modules.items()
        }

    def list_available_modules(self) -> List[str]:
        """
        Get names of all discovered (available) modules.

        Returns:
            List of module names
        """
        return list(self._module_classes.keys())

    # === Message Routing ===

    async def send_message(
        self,
        source: str,
        target: str,
        port: PortType,
        payload: Any,
        wait_response: bool = False,
        timeout: float = 30.0
    ) -> Optional[Message]:
        """
        Send a message between modules.

        Args:
            source: Source module name
            target: Target module name
            port: Port type
            payload: Message payload
            wait_response: Whether to wait for response
            timeout: Response timeout in seconds

        Returns:
            Response message if wait_response=True, else None
        """
        message = Message(
            source=source,
            target=target,
            port=port,
            payload=payload
        )

        if wait_response:
            return await self._event_bus.publish_and_wait(message, timeout)
        else:
            await self._event_bus.publish(message)
            return None

    # === Callbacks ===

    def on_module_loaded(self, callback: Callable[[str], None]) -> None:
        """Register callback for module load events."""
        self._on_module_loaded.append(callback)

    def on_module_unloaded(self, callback: Callable[[str], None]) -> None:
        """Register callback for module unload events."""
        self._on_module_unloaded.append(callback)

    # === Configuration ===

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config_loader.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config_loader.set(key, value)
