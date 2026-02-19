"""
Генератор звуковых сигналов для Discord модуля.

Создаёт простые звуковые эффекты программно (без внешних файлов).
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def generate_tone(
    frequency: float,
    duration_ms: int,
    sample_rate: int = 48000,
    volume: float = 0.3,
    fade_ms: int = 10,
) -> bytes:
    """
    Генерация простого тона.

    Args:
        frequency: Частота в Hz
        duration_ms: Длительность в мс
        sample_rate: Частота дискретизации
        volume: Громкость (0.0-1.0)
        fade_ms: Длительность fade in/out в мс

    Returns:
        PCM audio bytes (stereo, int16)
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, dtype=np.float32)

    # Синусоида
    wave = np.sin(2 * np.pi * frequency * t)

    # Fade in/out для мягкого звука
    fade_samples = int(sample_rate * fade_ms / 1000)
    if fade_samples > 0 and fade_samples < num_samples // 2:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        wave[:fade_samples] *= fade_in
        wave[-fade_samples:] *= fade_out

    # Применяем громкость и конвертируем в int16
    wave = (wave * volume * 32767).astype(np.int16)

    # Делаем стерео (Discord требует)
    stereo = np.column_stack((wave, wave)).flatten()

    return stereo.tobytes()


def generate_chime(
    frequencies: list,
    duration_ms: int = 150,
    sample_rate: int = 48000,
    volume: float = 0.25,
) -> bytes:
    """
    Генерация аккорда/чайма из нескольких частот.

    Args:
        frequencies: Список частот
        duration_ms: Длительность в мс
        sample_rate: Частота дискретизации
        volume: Громкость

    Returns:
        PCM audio bytes (stereo, int16)
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, dtype=np.float32)

    # Смешиваем частоты
    wave = np.zeros(num_samples, dtype=np.float32)
    for freq in frequencies:
        wave += np.sin(2 * np.pi * freq * t)

    # Нормализуем
    wave = wave / len(frequencies)

    # Экспоненциальное затухание
    decay = np.exp(-t * 8)
    wave *= decay

    # Fade in
    fade_samples = int(sample_rate * 5 / 1000)  # 5ms
    if fade_samples > 0:
        wave[:fade_samples] *= np.linspace(0, 1, fade_samples)

    # Конвертируем
    wave = (wave * volume * 32767).astype(np.int16)
    stereo = np.column_stack((wave, wave)).flatten()

    return stereo.tobytes()


def generate_wake_sound(sample_rate: int = 48000) -> bytes:
    """
    Звук активации wake word - приятный двойной тон.

    Два коротких восходящих тона: "дин-дон"
    """
    # Первый тон (ниже)
    tone1 = generate_chime([523.25, 659.25], duration_ms=100, sample_rate=sample_rate, volume=0.2)

    # Пауза
    pause_samples = int(sample_rate * 0.05)  # 50ms
    pause = np.zeros(pause_samples * 2, dtype=np.int16)  # stereo

    # Второй тон (выше)
    tone2 = generate_chime([659.25, 783.99], duration_ms=120, sample_rate=sample_rate, volume=0.25)

    # Собираем
    result = np.concatenate([
        np.frombuffer(tone1, dtype=np.int16),
        pause,
        np.frombuffer(tone2, dtype=np.int16),
    ])

    return result.tobytes()


def generate_listening_sound(sample_rate: int = 48000) -> bytes:
    """
    Звук начала прослушивания - один мягкий тон.
    """
    return generate_chime(
        [880, 1108.73],  # A5 + C#6 (мажорная терция)
        duration_ms=150,
        sample_rate=sample_rate,
        volume=0.2,
    )


def generate_success_sound(sample_rate: int = 48000) -> bytes:
    """
    Звук успешного распознавания команды.
    """
    # Восходящий аккорд
    tone1 = generate_chime([523.25], duration_ms=80, sample_rate=sample_rate, volume=0.15)
    tone2 = generate_chime([659.25], duration_ms=80, sample_rate=sample_rate, volume=0.18)
    tone3 = generate_chime([783.99], duration_ms=100, sample_rate=sample_rate, volume=0.2)

    pause = np.zeros(int(sample_rate * 0.03) * 2, dtype=np.int16)

    result = np.concatenate([
        np.frombuffer(tone1, dtype=np.int16),
        pause,
        np.frombuffer(tone2, dtype=np.int16),
        pause,
        np.frombuffer(tone3, dtype=np.int16),
    ])

    return result.tobytes()


def generate_error_sound(sample_rate: int = 48000) -> bytes:
    """
    Звук ошибки - нисходящий тон.
    """
    tone1 = generate_tone(400, duration_ms=150, sample_rate=sample_rate, volume=0.2)
    tone2 = generate_tone(300, duration_ms=200, sample_rate=sample_rate, volume=0.15)

    pause = np.zeros(int(sample_rate * 0.05) * 2, dtype=np.int16)

    result = np.concatenate([
        np.frombuffer(tone1, dtype=np.int16),
        pause,
        np.frombuffer(tone2, dtype=np.int16),
    ])

    return result.tobytes()


def generate_cancel_sound(sample_rate: int = 48000) -> bytes:
    """
    Звук отмены/таймаута.
    """
    return generate_tone(350, duration_ms=100, sample_rate=sample_rate, volume=0.15)


class SoundEffects:
    """Класс для управления звуковыми эффектами."""

    def __init__(self, sample_rate: int = 48000, enabled: bool = True):
        self._sample_rate = sample_rate
        self._enabled = enabled

        # Кэшируем сгенерированные звуки
        self._cache = {}

    def get_wake_sound(self) -> Optional[bytes]:
        """Звук при детекции wake word."""
        if not self._enabled:
            return None

        if "wake" not in self._cache:
            self._cache["wake"] = generate_wake_sound(self._sample_rate)
        return self._cache["wake"]

    def get_listening_sound(self) -> Optional[bytes]:
        """Звук начала записи команды."""
        if not self._enabled:
            return None

        if "listening" not in self._cache:
            self._cache["listening"] = generate_listening_sound(self._sample_rate)
        return self._cache["listening"]

    def get_success_sound(self) -> Optional[bytes]:
        """Звук успешного распознавания."""
        if not self._enabled:
            return None

        if "success" not in self._cache:
            self._cache["success"] = generate_success_sound(self._sample_rate)
        return self._cache["success"]

    def get_error_sound(self) -> Optional[bytes]:
        """Звук ошибки."""
        if not self._enabled:
            return None

        if "error" not in self._cache:
            self._cache["error"] = generate_error_sound(self._sample_rate)
        return self._cache["error"]

    def get_cancel_sound(self) -> Optional[bytes]:
        """Звук отмены."""
        if not self._enabled:
            return None

        if "cancel" not in self._cache:
            self._cache["cancel"] = generate_cancel_sound(self._sample_rate)
        return self._cache["cancel"]

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
