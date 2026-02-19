"""Message types for inter-module communication."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time
import uuid

from wiwi.interfaces.ports import PortType


@dataclass
class Message:
    """
    Унифицированное сообщение для коммуникации между модулями.

    Attributes:
        source: Имя модуля-отправителя
        target: Имя модуля-получателя (None = broadcast всем подписчикам)
        port: Тип порта, определяющий тип данных
        payload: Полезная нагрузка (данные)
        metadata: Дополнительные метаданные
        correlation_id: ID для отслеживания цепочки сообщений
        timestamp: Время создания сообщения
        priority: Приоритет обработки (выше = важнее)
    """

    source: str
    port: PortType
    payload: Any
    target: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    priority: int = 0

    def reply(
        self,
        payload: Any,
        port: Optional[PortType] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "Message":
        """
        Создать ответное сообщение.

        Args:
            payload: Данные ответа
            port: Порт для ответа (по умолчанию комплементарный)
            metadata: Метаданные ответа

        Returns:
            Новое сообщение-ответ
        """
        reply_port = port or PortType.get_complementary(self.port) or self.port

        return Message(
            source=self.target or "unknown",
            target=self.source,
            port=reply_port,
            payload=payload,
            metadata=metadata or {},
            correlation_id=self.correlation_id,
            priority=self.priority
        )

    def forward(
        self,
        target: str,
        source: Optional[str] = None,
        port: Optional[PortType] = None
    ) -> "Message":
        """
        Переслать сообщение другому модулю.

        Args:
            target: Новый получатель
            source: Новый отправитель (по умолчанию текущий target)
            port: Новый порт (по умолчанию тот же)

        Returns:
            Новое пересланное сообщение
        """
        return Message(
            source=source or self.target or self.source,
            target=target,
            port=port or self.port,
            payload=self.payload,
            metadata={**self.metadata, "forwarded_from": self.source},
            correlation_id=self.correlation_id,
            priority=self.priority
        )

    def with_metadata(self, **kwargs) -> "Message":
        """
        Создать копию сообщения с дополнительными метаданными.

        Args:
            **kwargs: Метаданные для добавления

        Returns:
            Новое сообщение с обновленными метаданными
        """
        new_metadata = {**self.metadata, **kwargs}
        return Message(
            source=self.source,
            target=self.target,
            port=self.port,
            payload=self.payload,
            metadata=new_metadata,
            correlation_id=self.correlation_id,
            timestamp=self.timestamp,
            priority=self.priority
        )

    def __repr__(self) -> str:
        payload_repr = repr(self.payload)
        if len(payload_repr) > 50:
            payload_repr = payload_repr[:47] + "..."
        return (
            f"Message(source={self.source!r}, target={self.target!r}, "
            f"port={self.port.name}, payload={payload_repr})"
        )


@dataclass
class AudioData:
    """Контейнер для аудио данных."""

    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    format: str = "wav"  # wav, mp3, ogg, etc.
    duration_ms: Optional[int] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Длительность в секундах."""
        if self.duration_ms is not None:
            return self.duration_ms / 1000.0
        return None


@dataclass
class SystemEvent:
    """Системное событие."""

    event_type: str  # module_loaded, module_unloaded, error, etc.
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
