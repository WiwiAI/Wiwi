"""Port type definitions for module I/O."""

from enum import Enum, auto


class PortType(Enum):
    """
    Типы портов ввода/вывода для модулей.

    Порты определяют какой тип данных модуль может принимать и отправлять.
    Это позволяет ядру правильно маршрутизировать сообщения между модулями.
    """

    # Текстовые порты
    TEXT_IN = "text_input"
    TEXT_OUT = "text_output"

    # Аудио порты
    AUDIO_IN = "audio_input"
    AUDIO_OUT = "audio_output"

    # Порты памяти
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"

    # Командные порты
    COMMAND_IN = "command_input"
    COMMAND_OUT = "command_output"

    # Порты событий
    EVENT_IN = "event_input"
    EVENT_OUT = "event_output"

    # Системные порты
    SYSTEM_IN = "system_input"
    SYSTEM_OUT = "system_output"

    def is_input(self) -> bool:
        """Проверка, является ли порт входным."""
        return self.value.endswith("_input") or self.value.endswith("_read")

    def is_output(self) -> bool:
        """Проверка, является ли порт выходным."""
        return self.value.endswith("_output") or self.value.endswith("_write")

    @classmethod
    def get_complementary(cls, port: "PortType") -> "PortType | None":
        """
        Получить комплементарный порт (вход -> выход, выход -> вход).

        Например: TEXT_IN -> TEXT_OUT, AUDIO_OUT -> AUDIO_IN
        """
        mapping = {
            cls.TEXT_IN: cls.TEXT_OUT,
            cls.TEXT_OUT: cls.TEXT_IN,
            cls.AUDIO_IN: cls.AUDIO_OUT,
            cls.AUDIO_OUT: cls.AUDIO_IN,
            cls.MEMORY_READ: cls.MEMORY_WRITE,
            cls.MEMORY_WRITE: cls.MEMORY_READ,
            cls.COMMAND_IN: cls.COMMAND_OUT,
            cls.COMMAND_OUT: cls.COMMAND_IN,
            cls.EVENT_IN: cls.EVENT_OUT,
            cls.EVENT_OUT: cls.EVENT_IN,
            cls.SYSTEM_IN: cls.SYSTEM_OUT,
            cls.SYSTEM_OUT: cls.SYSTEM_IN,
        }
        return mapping.get(port)
