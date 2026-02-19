"""Module registry for Wiwi4.0."""

from typing import Dict, List, Optional, Type, TYPE_CHECKING
import logging

from wiwi.interfaces.base_module import ModuleInfo

if TYPE_CHECKING:
    from wiwi.interfaces.base_module import BaseModule


class ModuleRegistry:
    """
    Registry for module classes and instances.

    Provides:
    - Module class registration (discovered modules)
    - Module instance registration (loaded modules)
    - Module lookup by name
    - Module metadata storage

    Usage:
        registry = ModuleRegistry()

        # Register a module class
        registry.register_class("my_module", MyModuleClass)

        # Register a running instance
        registry.register_instance("my_module", my_module_instance)

        # Lookup
        module = registry.get_instance("my_module")
    """

    def __init__(self):
        """Initialize the registry."""
        self._module_classes: Dict[str, Type["BaseModule"]] = {}
        self._module_instances: Dict[str, "BaseModule"] = {}
        self._module_info: Dict[str, ModuleInfo] = {}
        self._logger = logging.getLogger("wiwi.registry")

    # === Class Registration ===

    def register_class(
        self,
        name: str,
        module_class: Type["BaseModule"],
        info: Optional[ModuleInfo] = None
    ) -> None:
        """
        Register a module class.

        Args:
            name: Module name
            module_class: Module class
            info: Optional module info (will be extracted from class if not provided)
        """
        self._module_classes[name] = module_class

        if info:
            self._module_info[name] = info

        self._logger.debug(f"Registered module class: {name}")

    def unregister_class(self, name: str) -> None:
        """
        Unregister a module class.

        Args:
            name: Module name
        """
        if name in self._module_classes:
            del self._module_classes[name]
        if name in self._module_info:
            del self._module_info[name]
        self._logger.debug(f"Unregistered module class: {name}")

    def get_class(self, name: str) -> Optional[Type["BaseModule"]]:
        """
        Get a module class by name.

        Args:
            name: Module name

        Returns:
            Module class or None
        """
        return self._module_classes.get(name)

    def list_classes(self) -> List[str]:
        """
        Get list of registered module class names.

        Returns:
            List of module names
        """
        return list(self._module_classes.keys())

    # === Instance Registration ===

    def register_instance(self, name: str, instance: "BaseModule") -> None:
        """
        Register a module instance.

        Args:
            name: Module name
            instance: Module instance
        """
        self._module_instances[name] = instance
        self._module_info[name] = instance.module_info
        self._logger.debug(f"Registered module instance: {name}")

    def unregister_instance(self, name: str) -> None:
        """
        Unregister a module instance.

        Args:
            name: Module name
        """
        if name in self._module_instances:
            del self._module_instances[name]
        self._logger.debug(f"Unregistered module instance: {name}")

    def unregister(self, name: str) -> None:
        """
        Unregister both class and instance.

        Args:
            name: Module name
        """
        self.unregister_class(name)
        self.unregister_instance(name)

    def get_instance(self, name: str) -> Optional["BaseModule"]:
        """
        Get a module instance by name.

        Args:
            name: Module name

        Returns:
            Module instance or None
        """
        return self._module_instances.get(name)

    def list_instances(self) -> List[str]:
        """
        Get list of registered module instance names.

        Returns:
            List of module names
        """
        return list(self._module_instances.keys())

    # === Module Info ===

    def get_info(self, name: str) -> Optional[ModuleInfo]:
        """
        Get module info by name.

        Args:
            name: Module name

        Returns:
            ModuleInfo or None
        """
        return self._module_info.get(name)

    def list_all(self) -> Dict[str, ModuleInfo]:
        """
        Get all registered module info.

        Returns:
            Dict of module name -> ModuleInfo
        """
        return self._module_info.copy()

    # === Queries ===

    def is_registered(self, name: str) -> bool:
        """
        Check if a module is registered (class or instance).

        Args:
            name: Module name

        Returns:
            True if registered
        """
        return name in self._module_classes or name in self._module_instances

    def is_loaded(self, name: str) -> bool:
        """
        Check if a module instance is loaded.

        Args:
            name: Module name

        Returns:
            True if loaded
        """
        return name in self._module_instances

    def get_by_category(self, category: str) -> List[str]:
        """
        Get modules by category.

        Args:
            category: Category name (e.g., "ai", "memory", "interface")

        Returns:
            List of module names in the category
        """
        return [
            name for name, info in self._module_info.items()
            if info.category == category
        ]

    def get_dependencies(self, name: str) -> set:
        """
        Get dependencies for a module.

        Args:
            name: Module name

        Returns:
            Set of dependency module names
        """
        info = self._module_info.get(name)
        if info:
            return info.dependencies | info.optional_dependencies
        return set()

    def get_dependents(self, name: str) -> List[str]:
        """
        Get modules that depend on the given module.

        Args:
            name: Module name

        Returns:
            List of dependent module names
        """
        dependents = []
        for module_name, info in self._module_info.items():
            if name in info.dependencies or name in info.optional_dependencies:
                dependents.append(module_name)
        return dependents

    def clear(self) -> None:
        """Clear all registrations."""
        self._module_classes.clear()
        self._module_instances.clear()
        self._module_info.clear()
        self._logger.debug("Registry cleared")
