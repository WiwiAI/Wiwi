"""Base module interface for Wiwi4.0."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Set, TYPE_CHECKING
import logging

from wiwi.interfaces.ports import PortType
from wiwi.interfaces.messages import Message

if TYPE_CHECKING:
    from wiwi.core.kernel import Kernel
    from wiwi.core.event_bus import EventBus


class ModuleState(Enum):
    """Состояния жизненного цикла модуля."""

    UNINITIALIZED = auto()  # Модуль создан, но не инициализирован
    INITIALIZING = auto()   # Идет инициализация
    READY = auto()          # Готов к запуску
    RUNNING = auto()        # Работает
    PAUSED = auto()         # Приостановлен
    STOPPING = auto()       # Останавливается
    STOPPED = auto()        # Остановлен
    ERROR = auto()          # Ошибка


@dataclass
class ModuleInfo:
    """
    Метаданные модуля.

    Каждый модуль должен предоставить эту информацию для регистрации в системе.
    """

    name: str                                           # Уникальное имя модуля
    version: str                                        # Версия модуля
    description: str                                    # Описание
    author: str = "Unknown"                             # Автор
    category: str = "general"                           # Категория (ai, memory, interface, etc.)
    input_ports: Set[PortType] = field(default_factory=set)    # Входные порты
    output_ports: Set[PortType] = field(default_factory=set)   # Выходные порты
    dependencies: Set[str] = field(default_factory=set)        # Обязательные зависимости
    optional_dependencies: Set[str] = field(default_factory=set)  # Опциональные зависимости
    config_schema: Optional[Dict[str, Any]] = None      # Схема конфигурации


class BaseModule(ABC):
    """
    Базовый абстрактный класс для всех модулей Wiwi4.0.

    Каждый модуль должен:
    1. Наследоваться от этого класса
    2. Реализовать все абстрактные методы
    3. Объявить свои порты ввода/вывода в module_info

    Пример использования:

        class MyModule(BaseModule):
            @property
            def module_info(self) -> ModuleInfo:
                return ModuleInfo(
                    name="my_module",
                    version="1.0.0",
                    description="My awesome module",
                    input_ports={PortType.TEXT_IN},
                    output_ports={PortType.TEXT_OUT}
                )

            async def initialize(self) -> None:
                # Загрузка ресурсов
                pass

            async def start(self) -> None:
                self._state = ModuleState.RUNNING

            async def stop(self) -> None:
                self._state = ModuleState.STOPPED

            async def handle_input(self, message: Message) -> Optional[Message]:
                # Обработка входящего сообщения
                return message.reply(payload="processed")
    """

    def __init__(self, kernel: "Kernel", config: Dict[str, Any]):
        """
        Инициализация базового модуля.

        Args:
            kernel: Ссылка на ядро системы
            config: Конфигурация модуля из config файла
        """
        self._kernel = kernel
        self._config = config
        self._state = ModuleState.UNINITIALIZED
        self._event_bus: Optional["EventBus"] = None
        self._logger: logging.Logger = logging.getLogger(f"wiwi.module.{self.__class__.__name__}")

    # === Properties ===

    @property
    @abstractmethod
    def module_info(self) -> ModuleInfo:
        """
        Возвращает метаданные модуля.

        Должен быть реализован в каждом модуле.
        """
        pass

    @property
    def state(self) -> ModuleState:
        """Текущее состояние модуля."""
        return self._state

    @property
    def name(self) -> str:
        """Уникальное имя модуля."""
        return self.module_info.name

    @property
    def config(self) -> Dict[str, Any]:
        """Конфигурация модуля."""
        return self._config

    @property
    def kernel(self) -> "Kernel":
        """Ссылка на ядро системы."""
        return self._kernel

    @property
    def logger(self) -> logging.Logger:
        """Логгер модуля."""
        return self._logger

    # === Lifecycle Methods ===

    @abstractmethod
    async def initialize(self) -> None:
        """
        Инициализация модуля.

        Вызывается один раз при загрузке модуля.
        Здесь происходит загрузка моделей, подключение к ресурсам и т.д.

        Raises:
            Exception: При ошибке инициализации
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        Запуск модуля.

        Вызывается после инициализации всех зависимостей.
        Модуль должен установить state = RUNNING.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Остановка модуля.

        Освобождение ресурсов, сохранение состояния.
        Модуль должен установить state = STOPPED.
        """
        pass

    async def pause(self) -> None:
        """
        Приостановка модуля (опционально).

        По умолчанию просто меняет состояние.
        """
        self._state = ModuleState.PAUSED
        self._logger.info(f"Module {self.name} paused")

    async def resume(self) -> None:
        """
        Возобновление работы модуля (опционально).

        По умолчанию просто меняет состояние.
        """
        self._state = ModuleState.RUNNING
        self._logger.info(f"Module {self.name} resumed")

    # === Message Handling ===

    @abstractmethod
    async def handle_input(self, message: Message) -> Optional[Message]:
        """
        Обработка входящего сообщения.

        Args:
            message: Входящее сообщение

        Returns:
            Ответное сообщение или None если ответ не требуется
        """
        pass

    async def emit(
        self,
        port: PortType,
        payload: Any,
        target: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> None:
        """
        Отправка сообщения через указанный порт.

        Args:
            port: Порт вывода
            payload: Данные для отправки
            target: Целевой модуль (None = broadcast)
            metadata: Дополнительные метаданные
            priority: Приоритет сообщения

        Raises:
            ValueError: Если порт не объявлен как выходной
        """
        if port not in self.module_info.output_ports:
            raise ValueError(
                f"Port {port} is not declared as output port for module {self.name}"
            )

        if self._event_bus is None:
            raise RuntimeError(f"Module {self.name} is not connected to event bus")

        message = Message(
            source=self.name,
            target=target,
            port=port,
            payload=payload,
            metadata=metadata or {},
            priority=priority
        )

        await self._event_bus.publish(message)

    # === Configuration ===

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Получить значение из конфигурации модуля.

        Args:
            key: Ключ конфигурации (поддерживает точечную нотацию: "a.b.c")
            default: Значение по умолчанию

        Returns:
            Значение конфигурации или default
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def validate_config(self) -> bool:
        """
        Валидация конфигурации модуля.

        Returns:
            True если конфигурация валидна
        """
        schema = self.module_info.config_schema
        if schema is None:
            return True

        # Базовая валидация required полей
        for field_name, field_schema in schema.items():
            if isinstance(field_schema, dict) and field_schema.get("required", False):
                if field_name not in self._config:
                    self._logger.error(f"Missing required config field: {field_name}")
                    return False

        return True

    # === Health Check ===

    async def health_check(self) -> Dict[str, Any]:
        """
        Проверка здоровья модуля.

        Returns:
            Словарь с информацией о состоянии
        """
        return {
            "name": self.name,
            "state": self._state.name,
            "healthy": self._state in (ModuleState.READY, ModuleState.RUNNING),
            "version": self.module_info.version
        }

    # === Module Communication ===

    def get_module(self, module_name: str) -> Optional["BaseModule"]:
        """
        Получить другой модуль по имени.

        Args:
            module_name: Имя модуля

        Returns:
            Экземпляр модуля или None
        """
        return self._kernel.get_module(module_name)

    # === String Representation ===

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, state={self._state.name})>"
