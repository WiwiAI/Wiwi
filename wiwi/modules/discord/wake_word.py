"""
Wake Word детектор на базе OpenWakeWord.

Лёгкий детектор, работает на CPU с минимальной нагрузкой (~1-3% CPU).
Поддерживает кастомные wake words.
"""

import logging
import numpy as np
from typing import Optional, List, Callable, Dict
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class WakeWordState(Enum):
    """Состояние детектора."""
    LISTENING = "listening"      # Слушаем wake word (OWW)
    RECORDING = "recording"      # Записываем команду (VAD)
    PROCESSING = "processing"    # Обрабатываем команду


@dataclass
class WakeWordEvent:
    """Событие wake word."""
    user_id: int
    wake_word: str
    confidence: float
    audio_buffer: bytes  # Аудио после wake word


class OpenWakeWordDetector:
    """
    Детектор wake word на базе OpenWakeWord.

    Работает постоянно на CPU с минимальной нагрузкой.
    После детекции wake word передаёт управление VAD для записи команды.
    """

    def __init__(
        self,
        model_paths: Optional[List[str]] = None,
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ):
        """
        Args:
            model_paths: Пути к .onnx или .tflite моделям wake word
            threshold: Порог срабатывания (0.0-1.0)
            sample_rate: Частота дискретизации (должна быть 16kHz)
        """
        self._model_paths = model_paths or []
        self._threshold = threshold
        self._sample_rate = sample_rate

        self._oww_model = None
        self._is_initialized = False

        # Маппинг моделей на wake words
        self._model_names: List[str] = []

    def initialize(self) -> bool:
        """Инициализация OpenWakeWord."""
        try:
            from openwakeword.model import Model

            # Если указаны кастомные модели - используем их
            if self._model_paths:
                valid_paths = [p for p in self._model_paths if Path(p).exists()]
                if valid_paths:
                    self._oww_model = Model(
                        wakeword_models=valid_paths,
                        inference_framework="onnx",
                    )
                    self._model_names = [Path(p).stem for p in valid_paths]
                    logger.info(f"OpenWakeWord загружен с моделями: {self._model_names}")
                else:
                    logger.warning("Кастомные модели не найдены, используем встроенные")
                    self._oww_model = Model(inference_framework="onnx")
                    self._model_names = list(self._oww_model.models.keys())
            else:
                # Используем встроенные модели
                self._oww_model = Model(inference_framework="onnx")
                self._model_names = list(self._oww_model.models.keys())

            logger.info(f"OpenWakeWord инициализирован. Модели: {self._model_names}")
            self._is_initialized = True
            return True

        except ImportError:
            logger.error("OpenWakeWord не установлен: pip install openwakeword")
            return False
        except Exception as e:
            logger.error(f"Ошибка инициализации OpenWakeWord: {e}")
            return False

    def process_audio(self, audio_16k: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Обработка аудио чанка.

        Args:
            audio_16k: numpy array float32 [-1, 1], 16kHz mono

        Returns:
            Dict с вероятностями для каждой модели или None
        """
        if not self._is_initialized or self._oww_model is None:
            return None

        try:
            # OpenWakeWord ожидает float32 [-1, 1]
            if audio_16k.dtype != np.float32:
                audio_16k = audio_16k.astype(np.float32)

            # Нормализация если int16
            if audio_16k.max() > 1.0 or audio_16k.min() < -1.0:
                audio_16k = audio_16k / 32768.0

            # Предсказание
            prediction = self._oww_model.predict(audio_16k)

            return prediction

        except Exception as e:
            logger.error(f"Ошибка OWW predict: {e}")
            return None

    def check_wake_word(self, prediction: Dict[str, float]) -> Optional[tuple]:
        """
        Проверка превышения порога.

        Returns:
            (wake_word_name, confidence) или None
        """
        if not prediction:
            return None

        for name, scores in prediction.items():
            # scores может быть списком или числом
            if isinstance(scores, (list, np.ndarray)):
                score = float(scores[-1]) if len(scores) > 0 else 0.0
            else:
                score = float(scores)

            if score >= self._threshold:
                return (name, score)

        return None

    def reset(self) -> None:
        """Сброс состояния модели."""
        if self._oww_model:
            self._oww_model.reset()

    def cleanup(self) -> None:
        """Освобождение ресурсов."""
        self._oww_model = None
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def model_names(self) -> List[str]:
        return self._model_names


class SileroVAD:
    """
    Silero VAD для определения конца речи.

    Используется после детекции wake word для записи команды.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        min_silence_ms: int = 700,
        speech_pad_ms: int = 100,
    ):
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._min_silence_ms = min_silence_ms
        self._speech_pad_ms = speech_pad_ms

        self._model = None
        self._is_initialized = False

        # Размер чанка для VAD (512 сэмплов при 16kHz = 32ms)
        self._chunk_samples = 512

        # Буфер для накопления аудио
        self._buffer = np.array([], dtype=np.int16)

    def initialize(self) -> bool:
        """Инициализация Silero VAD."""
        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True,
            )

            self._model = model
            self._is_initialized = True
            logger.info("Silero VAD инициализирован (ONNX)")
            return True

        except Exception as e:
            logger.error(f"Ошибка инициализации Silero VAD: {e}")
            return False

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Проверка наличия речи в чанке.

        Args:
            audio_chunk: numpy array int16 или float32, 16kHz mono
        """
        if not self._is_initialized or self._model is None:
            return True  # Без VAD считаем всё речью

        try:
            import torch

            # Накапливаем аудио в буфер
            if audio_chunk.dtype != np.int16:
                audio_chunk = (audio_chunk * 32768).astype(np.int16)
            self._buffer = np.concatenate([self._buffer, audio_chunk])

            # Недостаточно данных
            if len(self._buffer) < self._chunk_samples:
                return False

            # Берём последние 512 сэмплов
            chunk = self._buffer[-self._chunk_samples:]

            # Сохраняем только последние данные (не больше 2x chunk_samples)
            if len(self._buffer) > self._chunk_samples * 2:
                self._buffer = self._buffer[-self._chunk_samples:]

            # Конвертируем в float32 [-1, 1]
            chunk = chunk.astype(np.float32) / 32768.0

            chunk_tensor = torch.from_numpy(chunk)

            prob = self._model(chunk_tensor, self._sample_rate).item()
            return prob >= self._threshold

        except Exception as e:
            logger.error(f"VAD ошибка: {e}")
            return False

    def reset(self) -> None:
        """Сброс состояния."""
        if self._model:
            self._model.reset_states()
        self._buffer = np.array([], dtype=np.int16)

    def cleanup(self) -> None:
        """Освобождение ресурсов."""
        self._model = None
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def min_silence_ms(self) -> int:
        return self._min_silence_ms

    @property
    def chunk_samples(self) -> int:
        return self._chunk_samples


class WakeWordPipeline:
    """
    Пайплайн: OpenWakeWord (постоянно) → VAD (после wake word) → STT.

    Состояния:
    1. LISTENING: OWW слушает wake word (лёгкая нагрузка)
    2. RECORDING: VAD записывает команду до паузы
    3. PROCESSING: Отправка на STT
    """

    def __init__(
        self,
        oww_model_paths: Optional[List[str]] = None,
        oww_threshold: float = 0.5,
        vad_threshold: float = 0.5,
        min_silence_ms: int = 700,
        max_recording_ms: int = 10000,
        sample_rate: int = 16000,
        on_wake_word: Optional[Callable[[int, str, float], None]] = None,
        on_command_ready: Optional[Callable[[int, bytes], None]] = None,
    ):
        """
        Args:
            oww_model_paths: Пути к моделям OpenWakeWord
            oww_threshold: Порог срабатывания wake word
            vad_threshold: Порог VAD для речи
            min_silence_ms: Минимальная тишина для завершения записи
            max_recording_ms: Максимальная длина записи
            sample_rate: Частота дискретизации
            on_wake_word: Callback при детекции wake word
            on_command_ready: Callback когда команда записана
        """
        self._sample_rate = sample_rate
        self._min_silence_ms = min_silence_ms
        self._max_recording_ms = max_recording_ms
        self._on_wake_word = on_wake_word
        self._on_command_ready = on_command_ready

        # Компоненты
        self._oww = OpenWakeWordDetector(
            model_paths=oww_model_paths,
            threshold=oww_threshold,
            sample_rate=sample_rate,
        )
        self._vad = SileroVAD(
            threshold=vad_threshold,
            sample_rate=sample_rate,
            min_silence_ms=min_silence_ms,
        )

        # Состояние по пользователям
        self._user_states: Dict[int, WakeWordState] = {}
        self._user_buffers: Dict[int, bytearray] = {}
        self._user_silence_start: Dict[int, float] = {}
        self._user_recording_start: Dict[int, float] = {}

        # Буфер для OWW (накапливаем ~80ms чанки)
        self._oww_buffer: Dict[int, np.ndarray] = {}
        self._oww_chunk_size = int(sample_rate * 0.08)  # 80ms = 1280 сэмплов

        # Cooldown после обработки команды (чтобы бот не срабатывал повторно)
        self._user_cooldown_until: Dict[int, float] = {}
        self._cooldown_ms = 3000  # 3 секунды cooldown

        # Буфер для накопления речи перед началом записи
        self._user_speech_buffer: Dict[int, bytearray] = {}
        self._user_speech_start: Dict[int, float] = {}
        self._min_speech_before_record_ms = 300  # Минимум 300ms речи перед записью

        self._is_initialized = False

    def initialize(self) -> bool:
        """Инициализация пайплайна."""
        oww_ok = self._oww.initialize()
        vad_ok = self._vad.initialize()

        if not oww_ok:
            logger.warning("OpenWakeWord не инициализирован, используем fallback")

        if not vad_ok:
            logger.warning("VAD не инициализирован")

        self._is_initialized = oww_ok or vad_ok
        return self._is_initialized

    def process_audio(self, user_id: int, audio_16k: bytes, timestamp: float) -> Optional[bytes]:
        """
        Обработка аудио чанка от пользователя.

        Args:
            user_id: ID пользователя
            audio_16k: PCM int16, 16kHz mono
            timestamp: Временная метка

        Returns:
            bytes с записанной командой или None
        """
        # Конвертируем в numpy
        samples = np.frombuffer(audio_16k, dtype=np.int16)

        # Инициализируем состояние пользователя
        if user_id not in self._user_states:
            self._user_states[user_id] = WakeWordState.LISTENING
            self._user_buffers[user_id] = bytearray()
            self._oww_buffer[user_id] = np.array([], dtype=np.int16)

        state = self._user_states[user_id]

        if state == WakeWordState.LISTENING:
            return self._process_listening(user_id, samples, timestamp)

        elif state == WakeWordState.RECORDING:
            return self._process_recording(user_id, samples, timestamp)

        return None

    def _process_listening(self, user_id: int, samples: np.ndarray, timestamp: float) -> None:
        """Режим прослушивания wake word."""
        # Проверяем cooldown - если недавно обработали команду, игнорируем
        cooldown_until = self._user_cooldown_until.get(user_id, 0)
        if timestamp < cooldown_until:
            return None

        # Отладочное логирование (каждые 50 чанков)
        if not hasattr(self, '_listen_debug_counter'):
            self._listen_debug_counter = 0
        self._listen_debug_counter += 1

        # Проверяем VAD - есть ли речь в этом чанке
        is_speech = self._vad.is_speech(samples)

        # Логирование уровня сигнала
        if self._listen_debug_counter % 100 == 1:
            max_amp = np.abs(samples).max()
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            logger.debug(
                f"[VAD] chunk #{self._listen_debug_counter}, samples={len(samples)}, "
                f"max={max_amp}, rms={rms:.0f}, is_speech={is_speech}"
            )

        if self._listen_debug_counter % 50 == 1:
            logger.debug(f"[LISTENING] user={user_id}, samples={len(samples)}, range=[{samples.min()},{samples.max()}], is_speech={is_speech}")

        if not is_speech:
            # Тишина - сбрасываем буфер накопления речи
            if user_id in self._user_speech_buffer:
                self._user_speech_buffer[user_id] = bytearray()
                self._user_speech_start.pop(user_id, None)
            return None

        # Есть речь - накапливаем в буфер
        if user_id not in self._user_speech_buffer:
            self._user_speech_buffer[user_id] = bytearray()

        if user_id not in self._user_speech_start:
            self._user_speech_start[user_id] = timestamp

        self._user_speech_buffer[user_id].extend(samples.tobytes())

        # Проверяем, достаточно ли накопилось речи для начала записи
        speech_duration_ms = (timestamp - self._user_speech_start[user_id]) * 1000

        if speech_duration_ms >= self._min_speech_before_record_ms:
            # Достаточно речи - начинаем полноценную запись
            logger.info(f"Speech detected for user {user_id} ({speech_duration_ms:.0f}ms), starting command recording.")

            self._user_states[user_id] = WakeWordState.RECORDING
            # Копируем накопленный буфер речи
            self._user_buffers[user_id] = bytearray(self._user_speech_buffer[user_id])
            self._user_recording_start[user_id] = self._user_speech_start[user_id]
            self._user_silence_start[user_id] = 0

            # Очищаем буфер накопления
            self._user_speech_buffer[user_id] = bytearray()
            self._user_speech_start.pop(user_id, None)

            # Сбрасываем OWW для следующего детекта
            self._oww.reset()

            # Callback о начале записи
            if self._on_wake_word:
                self._on_wake_word(user_id, "speech_detected", 1.0)

        return None

    def _process_recording(self, user_id: int, samples: np.ndarray, timestamp: float) -> Optional[bytes]:
        """Режим записи команды после wake word."""
        # Добавляем в буфер
        self._user_buffers[user_id].extend(samples.tobytes())

        # Проверяем VAD
        is_speech = self._vad.is_speech(samples)

        if is_speech:
            # Речь продолжается
            self._user_silence_start[user_id] = 0
        else:
            # Тишина
            if self._user_silence_start[user_id] == 0:
                self._user_silence_start[user_id] = timestamp

            silence_duration = (timestamp - self._user_silence_start[user_id]) * 1000

            # Достаточно тишины - команда завершена
            if silence_duration >= self._min_silence_ms:
                return self._finish_recording(user_id)

        # Проверяем таймаут
        recording_duration = (timestamp - self._user_recording_start[user_id]) * 1000
        if recording_duration >= self._max_recording_ms:
            logger.warning(f"Recording timeout for user {user_id}")
            return self._finish_recording(user_id)

        return None

    def _finish_recording(self, user_id: int) -> Optional[bytes]:
        """Завершение записи и возврат аудио."""
        import time
        audio_data = bytes(self._user_buffers[user_id])

        # Устанавливаем cooldown перед сбросом состояния
        current_time = time.time()
        self._user_cooldown_until[user_id] = current_time + (self._cooldown_ms / 1000.0)
        logger.debug(f"Cooldown set for user {user_id} until {self._cooldown_ms}ms from now")

        # Сброс состояния
        self._user_states[user_id] = WakeWordState.LISTENING
        self._user_buffers[user_id] = bytearray()
        self._user_silence_start.pop(user_id, None)
        self._user_recording_start.pop(user_id, None)
        self._vad.reset()

        # Минимальная длина (200ms)
        min_samples = int(self._sample_rate * 0.2) * 2  # int16 = 2 bytes
        if len(audio_data) < min_samples:
            logger.debug(f"Recording too short for user {user_id}")
            return None

        logger.info(f"Command recorded for user {user_id}: {len(audio_data)} bytes")

        # Callback
        if self._on_command_ready:
            self._on_command_ready(user_id, audio_data)

        return audio_data

    def reset_user(self, user_id: int) -> None:
        """Сброс состояния пользователя."""
        self._user_states[user_id] = WakeWordState.LISTENING
        self._user_buffers[user_id] = bytearray()
        self._oww_buffer[user_id] = np.array([], dtype=np.int16)
        self._user_silence_start.pop(user_id, None)
        self._user_recording_start.pop(user_id, None)
        self._user_cooldown_until.pop(user_id, None)
        self._user_speech_buffer.pop(user_id, None)
        self._user_speech_start.pop(user_id, None)

    def get_user_state(self, user_id: int) -> WakeWordState:
        """Получить состояние пользователя."""
        return self._user_states.get(user_id, WakeWordState.LISTENING)

    def cleanup(self) -> None:
        """Освобождение ресурсов."""
        self._oww.cleanup()
        self._vad.cleanup()
        self._user_states.clear()
        self._user_buffers.clear()
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def oww_models(self) -> List[str]:
        return self._oww.model_names
