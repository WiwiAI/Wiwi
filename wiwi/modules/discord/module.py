"""
Discord модуль для Wiwi.

Обеспечивает интеграцию с Discord:
- Текстовые сообщения в каналах
- Голосовое общение через voice channels
- Wake word детекция через OpenWakeWord (постоянно, лёгкая нагрузка)
- VAD для определения конца команды
"""

import asyncio
import io
import logging
import time
from typing import Optional, Dict, Any, Set
from collections import deque

import numpy as np
import discord
from discord.ext import commands

from wiwi.interfaces.base_module import BaseModule, ModuleInfo, ModuleState
from wiwi.interfaces.ports import PortType
from wiwi.interfaces.messages import Message

from .wake_word import WakeWordPipeline, WakeWordState
from .sounds import SoundEffects
from .streaming_audio import StreamingAudioSource, convert_to_discord_format

logger = logging.getLogger(__name__)


class WiwiAudioSink(discord.sinks.Sink):
    """
    Кастомный sink для приёма аудио из Discord.
    Передаёт аудио напрямую в WakeWordPipeline.
    """

    def __init__(self, module: "DiscordModule"):
        super().__init__()
        self._module = module

    def write(self, data: bytes, user: int) -> None:
        """Получение аудио от пользователя."""
        if self._module._is_listening:
            # Запускаем обработку в asyncio, используя thread-safe метод
            if self._module._main_loop:
                asyncio.run_coroutine_threadsafe(
                    self._module._on_audio_chunk(user, data), self._module._main_loop
                )

    def cleanup(self) -> None:
        pass


class DiscordModule(BaseModule):
    """
    Discord бот модуль для Wiwi.

    Использует OpenWakeWord для постоянного прослушивания wake word
    с минимальной нагрузкой на CPU (~1-3%).
    После детекции wake word включается VAD для записи команды.
    """

    def __init__(self, kernel, config: Dict[str, Any]):
        super().__init__(kernel, config)

        # Discord bot
        self._bot: Optional[discord.Bot] = None
        self._bot_task: Optional[asyncio.Task] = None
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        # Конфигурация Discord
        self._token = config.get("token", "")
        self._prefix = config.get("command_prefix", "!")
        self._allowed_channels: Set[int] = set(config.get("allowed_channels", []))
        self._allowed_users: Set[int] = set(config.get("allowed_users", []))
        self._respond_to_mentions = config.get("respond_to_mentions", True)
        self._respond_to_prefix = config.get("respond_to_prefix", True)

        # Голосовые настройки
        self._voice_enabled = config.get("voice_enabled", True)
        self._auto_join_voice = config.get("auto_join_voice", False)
        self._voice_channel_id: Optional[int] = config.get("voice_channel_id")

        # Wake word настройки
        oww_config = config.get("wake_word", {})
        self._oww_model_paths = oww_config.get("model_paths", [])
        self._oww_threshold = oww_config.get("threshold", 0.5)
        self._vad_threshold = oww_config.get("vad_threshold", 0.5)
        self._min_silence_ms = oww_config.get("min_silence_ms", 700)
        self._max_recording_ms = oww_config.get("max_recording_ms", 10000)

        # Звуковые эффекты
        sounds_config = config.get("sounds", {})
        self._sounds_enabled = sounds_config.get("enabled", True)
        self._sounds = SoundEffects(sample_rate=48000, enabled=self._sounds_enabled)

        # Состояние
        self._voice_client: Optional[discord.VoiceClient] = None
        self._current_channel: Optional[discord.TextChannel] = None
        self._pending_responses: Dict[int, discord.Message] = {}
        self._voice_text_channel: Optional[discord.TextChannel] = None

        # Wake Word Pipeline
        self._pipeline: Optional[WakeWordPipeline] = None

        # Аудио
        self._audio_sink: Optional[WiwiAudioSink] = None
        self._audio_queue: deque = deque()
        self._is_playing = False
        self._is_listening = False

        # Streaming audio
        self._streaming_source: Optional[StreamingAudioSource] = None
        self._current_stream_id: Optional[str] = None
        self._playback_monitor_task: Optional[asyncio.Task] = None

        # Параметры аудио
        self._discord_sample_rate = 48000
        self._target_sample_rate = 16000

    @property
    def module_info(self) -> ModuleInfo:
        return ModuleInfo(
            name="discord",
            version="2.0.0",
            description="Discord интеграция с OpenWakeWord + VAD",
            author="Wiwi",
            category="interface",
            input_ports={PortType.TEXT_OUT, PortType.AUDIO_OUT, PortType.COMMAND_IN},
            output_ports={PortType.TEXT_IN, PortType.AUDIO_IN, PortType.COMMAND_OUT, PortType.COMMAND_IN},
            dependencies=set(),
            optional_dependencies={"memory", "stt", "tts"},
            config_schema={
                "type": "object",
                "properties": {
                    "token": {"type": "string"},
                    "wake_word": {
                        "type": "object",
                        "properties": {
                            "model_paths": {"type": "array"},
                            "threshold": {"type": "number"},
                            "vad_threshold": {"type": "number"},
                        }
                    },
                },
                "required": ["token"],
            },
        )

    async def initialize(self) -> None:
        """Инициализация модуля."""
        self._state = ModuleState.INITIALIZING
        self._main_loop = asyncio.get_running_loop()

        # Manually load opus unconditionally
        try:
            discord.opus.load_opus('/home/wiwi/miniconda3/envs/wiwi/lib/libopus.so')
            self.logger.info("Forcing manual load of opus library from conda environment.")
        except Exception as e:
            self.logger.error(f"Could not manually load opus library: {e}")
            self._state = ModuleState.ERROR
            raise RuntimeError("Opus library could not be loaded.") from e

        if not self._token:
            logger.error("Discord token не указан!")
            self._state = ModuleState.ERROR
            raise ValueError("Discord token is required")

        # Инициализируем Wake Word Pipeline
        await self._init_wake_word_pipeline()

        # Создаем Discord бота
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        intents.members = True

        self._bot = discord.Bot(
            command_prefix=self._prefix,
            intents=intents,
            help_command=None,
        )

        self._setup_event_handlers()
        self._setup_slash_commands()

        self._state = ModuleState.READY
        logger.info("Discord модуль инициализирован (OpenWakeWord + VAD)")

    async def _init_wake_word_pipeline(self) -> None:
        """Инициализация Wake Word Pipeline."""
        logger.info(f"Инициализация Wake Word Pipeline с моделями: {self._oww_model_paths}")
        self._pipeline = WakeWordPipeline(
            oww_model_paths=self._oww_model_paths if self._oww_model_paths else None,
            oww_threshold=self._oww_threshold,
            vad_threshold=self._vad_threshold,
            min_silence_ms=self._min_silence_ms,
            max_recording_ms=self._max_recording_ms,
            sample_rate=self._target_sample_rate,
            on_wake_word=self._on_wake_word_detected,
            on_command_ready=self._on_command_ready,
        )

        # Инициализируем в executor (может быть долго)
        if self._main_loop:
            await self._main_loop.run_in_executor(None, self._pipeline.initialize)

        if self._pipeline.is_initialized:
            logger.info(f"Wake Word Pipeline готов. Модели OWW: {self._pipeline.oww_models}")
        else:
            logger.warning("Wake Word Pipeline не полностью инициализирован")

    def _on_wake_word_detected(self, user_id: int, wake_word: str, confidence: float) -> None:
        """Callback при детекции wake word."""
        logger.info(f"Wake word '{wake_word}' от пользователя {user_id} (conf: {confidence:.2f})")

        # Воспроизводим звук активации
        if self._main_loop:
            asyncio.run_coroutine_threadsafe(self._play_sound("wake"), self._main_loop)

    def _on_command_ready(self, user_id: int, audio_data: bytes) -> None:
        """Callback когда команда записана."""
        logger.info(f"Команда от {user_id}: {len(audio_data)} bytes")

        # Отправляем на транскрипцию
        if self._main_loop:
            asyncio.run_coroutine_threadsafe(
                self._process_voice_command(user_id, audio_data), self._main_loop
            )

    async def _process_voice_command(self, user_id: int, audio_data: bytes) -> None:
        """Обработка голосовой команды."""
        # Транскрибируем
        text = await self._transcribe_audio(audio_data)

        if text and text.strip():
            logger.info(f"Команда от {user_id}: {text}")

            # Отправляем в систему Wiwi
            await self.emit(
                PortType.TEXT_IN,
                text.strip(),
                metadata={
                    "source_type": "discord_voice",
                    "user_id": user_id,
                    "channel_id": self._voice_text_channel.id if self._voice_text_channel else None,
                },
            )

    def _setup_event_handlers(self) -> None:
        """Настройка обработчиков событий Discord."""

        @self._bot.event
        async def on_ready():
            logger.info(f"Discord бот подключен как {self._bot.user}")

            if self._auto_join_voice and self._voice_channel_id:
                await self._join_voice_channel(self._voice_channel_id)

        @self._bot.event
        async def on_message(message: discord.Message):
            if message.author == self._bot.user:
                return

            if not self._is_allowed(message):
                return

            should_respond = False
            content = message.content

            if self._respond_to_mentions and self._bot.user in message.mentions:
                should_respond = True
                content = content.replace(f"<@{self._bot.user.id}>", "").strip()
                content = content.replace(f"<@!{self._bot.user.id}>", "").strip()

            if self._respond_to_prefix and content.lower().startswith("wiwi"):
                should_respond = True
                content = content[4:].strip()
                if content.startswith(",") or content.startswith(":"):
                    content = content[1:].strip()

            if not should_respond:
                return

            if not content:
                await message.reply("Привет! Чем могу помочь?")
                return

            self._current_channel = message.channel
            self._pending_responses[message.id] = message

            async with message.channel.typing():
                await self.emit(
                    PortType.TEXT_IN,
                    content,
                    metadata={
                        "source_type": "discord",
                        "channel_id": message.channel.id,
                        "user_id": message.author.id,
                        "user_name": str(message.author),
                        "message_id": message.id,
                    },
                )

    def _setup_slash_commands(self) -> None:
        """Настройка slash команд."""

        @self._bot.slash_command(name="join", description="Присоединиться к голосовому каналу")
        async def join_voice(interaction: discord.Interaction):
            if not interaction.user.voice:
                await interaction.response.send_message(
                    "Вы должны быть в голосовом канале!",
                    ephemeral=True
                )
                return

            await interaction.response.defer(ephemeral=True)

            channel = interaction.user.voice.channel
            self._voice_text_channel = interaction.channel

            try:
                # Retry логика для сетевых проблем
                max_retries = 3
                last_error = None

                for attempt in range(max_retries):
                    try:
                        if self._voice_client and self._voice_client.is_connected():
                            await self._voice_client.move_to(channel)
                        else:
                            self._voice_client = await channel.connect(timeout=30.0)
                        break
                    except Exception as conn_err:
                        last_error = conn_err
                        logger.warning(f"Попытка {attempt + 1}/{max_retries} не удалась: {conn_err}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                        else:
                            raise last_error

                # Wait until the client is truly connected
                for _ in range(10):
                    if self._voice_client and self._voice_client.is_connected():
                        break
                    await asyncio.sleep(0.5)
                else:
                    raise Exception("Не удалось подключиться к голосовому каналу (тайм-аут)")

                await self._start_listening()

                # Информация о состоянии
                oww_status = "активен" if self._pipeline and self._pipeline.is_initialized else "недоступен"
                oww_models_list = self._pipeline.oww_models if self._pipeline else []
                oww_models = ", ".join(oww_models_list[:3]) if oww_models_list else "нет"

                await interaction.followup.send(
                    f"Подключилась к **{channel.name}**!\n"
                    f"OpenWakeWord: {oww_status}\n"
                    f"Модели: {oww_models}\n"
                    f"Скажите wake word для активации",
                )
                logger.info(f"Подключен к голосовому каналу: {channel.name}")

            except Exception as e:
                import traceback
                logger.error(f"Ошибка подключения: {e}\n{traceback.format_exc()}")
                await interaction.followup.send(f"Ошибка: {e}")

        @self._bot.slash_command(name="leave", description="Покинуть голосовой канал")
        async def leave_voice(interaction: discord.Interaction):
            if not self._voice_client or not self._voice_client.is_connected():
                await interaction.response.send_message("Я не в голосовом канале!", ephemeral=True)
                return

            await self._stop_listening()
            await self._voice_client.disconnect()
            self._voice_client = None

            await interaction.response.send_message("Отключилась от голосового канала!")
            logger.info("Отключен от голосового канала")

        @self._bot.slash_command(name="status", description="Статус Wiwi бота")
        async def bot_status(interaction: discord.Interaction):
            voice_status = "Подключена" if self._voice_client and self._voice_client.is_connected() else "Не подключена"

            channel_name = "—"
            if self._voice_client and self._voice_client.channel:
                channel_name = self._voice_client.channel.name

            oww_status = "активен" if self._pipeline and self._pipeline.is_initialized else "недоступен"
            oww_models_list = self._pipeline.oww_models if self._pipeline else []
            oww_models = ", ".join(oww_models_list[:3]) if oww_models_list else "нет"

            await interaction.response.send_message(
                f"**Статус Wiwi**\n"
                f"Голос: {voice_status}\n"
                f"Канал: {channel_name}\n"
                f"OpenWakeWord: {oww_status}\n"
                f"Модели: {oww_models}",
                ephemeral=True
            )

    def _is_allowed(self, message: discord.Message) -> bool:
        """Проверка разрешений."""
        if not self._allowed_channels and not self._allowed_users:
            return True
        if self._allowed_channels and message.channel.id not in self._allowed_channels:
            return False
        if self._allowed_users and message.author.id not in self._allowed_users:
            return False
        return True

    async def start(self) -> None:
        """Запуск модуля."""
        if self._state != ModuleState.READY:
            raise RuntimeError("Модуль не инициализирован")

        self._state = ModuleState.RUNNING
        self._bot_task = asyncio.create_task(self._run_bot())
        logger.info("Discord модуль запущен")

    async def _run_bot(self) -> None:
        """Запуск бота."""
        try:
            await self._bot.start(self._token)
        except discord.LoginFailure:
            logger.error("Неверный Discord token!")
            self._state = ModuleState.ERROR
        except Exception as e:
            logger.error(f"Ошибка Discord бота: {e}")
            self._state = ModuleState.ERROR

    async def stop(self) -> None:
        """Остановка модуля."""
        self._state = ModuleState.STOPPING

        await self._stop_listening()

        if self._voice_client and self._voice_client.is_connected():
            await self._voice_client.disconnect()
            self._voice_client = None

        if self._bot:
            await self._bot.close()

        if self._bot_task:
            self._bot_task.cancel()
            try:
                await self._bot_task
            except asyncio.CancelledError:
                pass

        if self._pipeline:
            self._pipeline.cleanup()

        self._state = ModuleState.STOPPED
        logger.info("Discord модуль остановлен")

    async def handle_input(self, message: Message) -> Optional[Message]:
        """Обработка входящих сообщений."""
        logger.debug(f"Discord handle_input: port={message.port}, source={message.source}")

        if message.port == PortType.TEXT_OUT:
            await self._send_text_response(message)
        elif message.port == PortType.AUDIO_OUT:
            logger.debug(
                f"Received AUDIO_OUT from {message.source}, "
                f"streaming={message.metadata.get('streaming', False)}"
            )
            await self._play_audio(message.payload, metadata=message.metadata)
        elif message.port == PortType.COMMAND_IN:
            await self._handle_command(message)
        return None

    async def _send_text_response(self, message: Message) -> None:
        """Отправка ответа в Discord."""
        text = message.payload
        if not text:
            return

        metadata = message.metadata or {}
        original_message_id = metadata.get("message_id")
        source_type = metadata.get("source_type", "")

        # Голосовой ответ с TTS (CHANGED)
        if source_type == "discord_voice" and self._voice_text_channel:
            user_id = metadata.get("user_id")
            tts_module = self.get_module("tts")
            if tts_module and hasattr(tts_module, "generate_chunks"):
                tts_gen = tts_module.generate_chunks(text)  # NEW: асинхронный генератор чанков TTS
                # NEW: потоковое воспроизведение в голосовой канал
                asyncio.create_task(self._stream_tts_to_voice(tts_gen))
                return


        # Текстовый ответ
        if original_message_id and original_message_id in self._pending_responses:
            original = self._pending_responses.pop(original_message_id)
            await self._send_to_channel(original.channel, text, reply_to=original)
            return

        if self._current_channel:
            await self._send_to_channel(self._current_channel, text)

    async def _stream_tts_to_voice(self, tts_generator):  # NEW
        """Потоковое воспроизведение TTS в голосовом канале."""  # NEW
        if not self._voice_client or not self._voice_client.is_connected():  # NEW
            return  # NEW

        async for chunk in tts_generator:  # NEW
            self._audio_queue.append(chunk)  # NEW: добавляем в очередь аудио
            if hasattr(self, '_audio_event'):  # NEW
                self._audio_event.set()  # NEW: сигнал, что появились новые данные



    async def _send_to_channel(
        self,
        channel: discord.TextChannel,
        text: str,
        reply_to: Optional[discord.Message] = None
    ) -> None:
        """Отправка в канал."""
        try:
            chunks = [text[i:i+1990] for i in range(0, len(text), 1990)] if len(text) > 2000 else [text]

            for i, chunk in enumerate(chunks):
                if i == 0 and reply_to:
                    await reply_to.reply(chunk)
                else:
                    await channel.send(chunk)
        except discord.errors.HTTPException as e:
            logger.error(f"Ошибка отправки: {e}")

    async def _handle_command(self, message: Message) -> None:
        """Обработка команд."""
        command = message.payload
        metadata = message.metadata or {}

        if command == "voice_join":
            channel_id = metadata.get("channel_id")
            if channel_id:
                await self._join_voice_channel(channel_id)
        elif command == "voice_leave":
            await self._leave_voice_channel()

    async def _join_voice_channel(self, channel_id: int) -> bool:
        """Подключение к голосовому каналу."""
        if not self._bot:
            return False

        channel = self._bot.get_channel(channel_id)
        if not channel or not isinstance(channel, discord.VoiceChannel):
            return False

        try:
            if self._voice_client and self._voice_client.is_connected():
                await self._voice_client.move_to(channel)
            else:
                self._voice_client = await channel.connect()

            if self._voice_enabled:
                await self._start_listening()

            return True
        except Exception as e:
            logger.error(f"Ошибка подключения: {e}")
            return False

    async def _leave_voice_channel(self) -> None:
        """Отключение от канала."""
        await self._stop_listening()
        if self._voice_client and self._voice_client.is_connected():
            await self._voice_client.disconnect()
            self._voice_client = None

    async def _start_listening(self) -> None:
        """Начать прослушивание."""
        if self._is_listening or not self._voice_client:
            return

        self._is_listening = True
        self._audio_sink = WiwiAudioSink(self)

        self._voice_client.start_recording(
            self._audio_sink,
            self._on_recording_stopped,
            self._voice_text_channel
        )

        # Включаем TTS для голосового ответа (Discord mode - без локального вывода)
        await self.emit(PortType.COMMAND_IN, "discord_voice_on")

        logger.info("Прослушивание начато (OpenWakeWord)")

    async def _stop_listening(self) -> None:
        """Остановить прослушивание."""
        if not self._is_listening:
            return

        self._is_listening = False

        if self._voice_client and self._voice_client.recording:
            self._voice_client.stop_recording()

        # Выключаем TTS (Discord mode)
        await self.emit(PortType.COMMAND_IN, "discord_voice_off")

        logger.info("Прослушивание остановлено")

    async def _on_recording_stopped(self, sink, channel):
        """Callback остановки записи."""
        pass

    async def _on_audio_chunk(self, user_id: int, data: bytes) -> None:
        """Обработка аудио чанка от пользователя."""
        if not self._is_listening or not self._pipeline:
            return

        # Периодическое логирование входящего аудио
        if not hasattr(self, '_chunk_counter'):
            self._chunk_counter = 0
        self._chunk_counter += 1

        if self._chunk_counter % 100 == 1:
            raw_samples = np.frombuffer(data, dtype=np.int16)
            logger.debug(
                f"Discord audio chunk #{self._chunk_counter}: "
                f"bytes={len(data)}, samples={len(raw_samples)}, "
                f"range=[{raw_samples.min()}, {raw_samples.max()}]"
            )

        # Конвертируем Discord аудио (48kHz stereo) в 16kHz mono
        audio_16k = self._convert_to_16k_mono(data)
        if audio_16k is None:
            return

        # Отправляем в pipeline
        timestamp = time.time()

        # Pipeline работает синхронно, запускаем в executor
        if self._main_loop:
            command_audio = await self._main_loop.run_in_executor(
                None,
                self._pipeline.process_audio,
                user_id,
                audio_16k,
                timestamp
            )

        # Если команда готова - она будет обработана через callback
        # (on_command_ready уже вызывается внутри pipeline)

    def _convert_to_16k_mono(self, data: bytes) -> Optional[bytes]:
        """Конвертация Discord аудио (48kHz stereo) в 16kHz mono."""
        try:
            from scipy import signal

            samples = np.frombuffer(data, dtype=np.int16)

            if len(samples) < 2:
                return None

            # Stereo -> Mono (усреднение каналов для лучшего качества)
            if len(samples) % 2 == 0:
                left = samples[0::2].astype(np.float32)
                right = samples[1::2].astype(np.float32)
                mono = ((left + right) / 2.0).astype(np.int16)
            else:
                mono = samples

            # 48kHz -> 16kHz с правильным ресэмплингом (anti-aliasing)
            # scipy.signal.resample_poly - быстрый и качественный ресэмплинг
            # Ratio: 16000/48000 = 1/3
            resampled = signal.resample_poly(mono.astype(np.float32), up=1, down=3)
            resampled = np.clip(resampled, -32768, 32767).astype(np.int16)

            return resampled.tobytes()

        except Exception as e:
            logger.error(f"Ошибка конвертации: {e}")
            return None

    async def _transcribe_audio(self, audio_bytes: bytes) -> Optional[str]:
        """Транскрипция аудио через STT модуль."""
        try:
            stt_module = self.get_module("stt")
            if stt_module and hasattr(stt_module, "transcribe"):
                # Конвертируем bytes (int16) в numpy float32 [-1, 1]
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0

                # Логируем характеристики аудио
                duration_sec = len(audio_float32) / 16000
                max_amplitude = np.abs(audio_float32).max()
                rms = np.sqrt(np.mean(audio_float32 ** 2))
                logger.info(
                    f"Transcribing: {duration_sec:.2f}s, "
                    f"max_amp={max_amplitude:.4f}, rms={rms:.4f}, "
                    f"samples={len(audio_float32)}"
                )

                # Проверяем что аудио не слишком тихое
                if max_amplitude < 0.01:
                    logger.warning(f"Audio too quiet: max_amplitude={max_amplitude:.4f}")

                return await stt_module.transcribe(audio_float32, sample_rate=16000)
            else:
                logger.warning("STT модуль недоступен")
                return None
        except Exception as e:
            logger.error(f"Ошибка транскрипции: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def _play_sound(self, sound_type: str) -> None:
        """
        Воспроизведение звукового эффекта.

        Args:
            sound_type: Тип звука ('wake', 'listening', 'success', 'error', 'cancel')
        """
        if not self._voice_client or not self._voice_client.is_connected():
            return

        sound_data = None
        if sound_type == "wake":
            sound_data = self._sounds.get_wake_sound()
        elif sound_type == "listening":
            sound_data = self._sounds.get_listening_sound()
        elif sound_type == "success":
            sound_data = self._sounds.get_success_sound()
        elif sound_type == "error":
            sound_data = self._sounds.get_error_sound()
        elif sound_type == "cancel":
            sound_data = self._sounds.get_cancel_sound()

        if sound_data:
            await self._play_raw_audio(sound_data)

    async def _play_raw_audio(self, audio_bytes: bytes) -> None:
        """Воспроизведение сырых аудио данных (уже в формате Discord)."""
        if not self._voice_client or not self._voice_client.is_connected():
            return

        # Ждём если что-то воспроизводится
        while self._voice_client.is_playing():
            await asyncio.sleep(0.05)

        try:
            audio_source = discord.PCMAudio(io.BytesIO(audio_bytes))
            self._voice_client.play(audio_source)

            # Ждём окончания
            while self._voice_client.is_playing():
                await asyncio.sleep(0.05)
        except Exception as e:
            logger.error(f"Ошибка воспроизведения звука: {e}")

    async def _play_audio(
        self, audio_data: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Воспроизведение аудио в голосовом канале.

        Поддерживает два режима:
        - Streaming: real-time playback через StreamingAudioSource
        - Batch (legacy): последовательное воспроизведение чанков

        Args:
            audio_data: AudioData объект или bytes
            metadata: Метаданные сообщения (streaming, stream_id, is_final)
        """
        if not self._voice_client or not self._voice_client.is_connected():
            logger.warning("Cannot play audio: voice client not connected")
            return

        metadata = metadata or {}
        is_streaming = metadata.get("streaming", False)
        stream_id = metadata.get("stream_id")
        is_final = metadata.get("is_final", False)
        is_empty = metadata.get("empty", False)

        if is_streaming:
            await self._handle_streaming_audio(audio_data, stream_id, is_final, is_empty)
        else:
            # Legacy batch mode
            await self._handle_batch_audio(audio_data)

    async def _handle_streaming_audio(
        self,
        audio_data: Any,
        stream_id: Optional[str],
        is_final: bool,
        is_empty: bool,
    ) -> None:
        """
        Обработка streaming чанка аудио.

        Все чанки добавляются в единый буфер. Новый StreamingAudioSource
        создаётся только если буфер пуст и воспроизведение не идёт.

        Args:
            audio_data: AudioData объект или bytes
            stream_id: Уникальный идентификатор потока
            is_final: True если это последний чанк
            is_empty: True если это пустой сигнал окончания
        """
        # Handle empty end-of-stream signal - just ignore, buffer handles it
        if is_empty:
            logger.debug(f"Received empty end signal for stream {stream_id}")
            return

        # Extract audio bytes
        if hasattr(audio_data, "data"):
            audio_bytes = audio_data.data
            sample_rate = getattr(audio_data, "sample_rate", 24000)
        elif isinstance(audio_data, bytes):
            audio_bytes = audio_data
            sample_rate = 24000
        else:
            logger.warning(f"Unknown audio data type: {type(audio_data)}")
            return

        # Convert to Discord format (48kHz stereo)
        discord_audio = convert_to_discord_format(audio_bytes, sample_rate)
        if not discord_audio:
            logger.error("Failed to convert audio to Discord format")
            return

        # If we have an active source that's not finished, just add to it
        if self._streaming_source and not self._streaming_source.is_finished:
            self._streaming_source.add_audio(discord_audio)
            logger.debug(
                f"Added to buffer: {len(discord_audio)} bytes, "
                f"buffer={self._streaming_source.buffer_level_ms:.0f}ms"
            )
            return

        # Source is finished or doesn't exist - need to handle new data
        # Queue it for playback after current finishes
        if self._voice_client and self._voice_client.is_playing():
            # Current playback still going - add to pending queue
            if not hasattr(self, '_pending_audio'):
                self._pending_audio = []
            self._pending_audio.append(discord_audio)
            logger.debug(
                f"Queued audio while playing: {len(discord_audio)} bytes, "
                f"pending={len(self._pending_audio)} chunks"
            )

            # Start a task to create new stream after current finishes
            if not hasattr(self, '_pending_stream_task') or self._pending_stream_task is None or self._pending_stream_task.done():
                self._pending_stream_task = asyncio.create_task(
                    self._start_pending_stream(stream_id)
                )
            return

        # Start new stream immediately
        logger.info(f"Starting new audio stream: {stream_id}")
        await self._start_streaming_playback(stream_id)

        # Add to streaming buffer
        if self._streaming_source:
            self._streaming_source.add_audio(discord_audio)
            logger.debug(
                f"Stream {stream_id}: added {len(discord_audio)} bytes, "
                f"buffer={self._streaming_source.buffer_level_ms:.0f}ms"
            )

    async def _start_streaming_playback(self, stream_id: str) -> None:
        """
        Запуск воспроизведения нового потока.

        Args:
            stream_id: Уникальный идентификатор потока
        """
        # Stop any existing playback
        if self._voice_client and self._voice_client.is_playing():
            self._voice_client.stop()

        # Cancel existing monitor task
        if self._playback_monitor_task and not self._playback_monitor_task.done():
            self._playback_monitor_task.cancel()
            try:
                await self._playback_monitor_task
            except asyncio.CancelledError:
                pass

        # Clean up old source
        if self._streaming_source:
            self._streaming_source.cleanup()

        # Create new streaming source
        # Prebuffer 2 seconds to prevent stuttering on first phrases
        self._streaming_source = StreamingAudioSource(
            target_latency_ms=2000, prebuffer_ms=2000
        )
        self._current_stream_id = stream_id

        # Start playback monitor task
        self._playback_monitor_task = asyncio.create_task(
            self._streaming_playback_monitor(stream_id)
        )

    async def _start_pending_stream(self, stream_id: str) -> None:
        """
        Ждёт завершения текущего воспроизведения и запускает новый поток
        с накопленными данными.
        """
        # Wait for current playback to finish
        while self._voice_client and self._voice_client.is_playing():
            await asyncio.sleep(0.05)

        # Get pending audio
        pending = getattr(self, '_pending_audio', [])
        if not pending:
            return

        self._pending_audio = []

        logger.info(f"Starting pending stream: {stream_id}, {len(pending)} chunks queued")

        # Start new stream
        await self._start_streaming_playback(stream_id)

        # Add all pending audio
        if self._streaming_source:
            for chunk in pending:
                self._streaming_source.add_audio(chunk)
            logger.debug(
                f"Added {len(pending)} pending chunks, "
                f"buffer={self._streaming_source.buffer_level_ms:.0f}ms"
            )

    async def _streaming_playback_monitor(self, stream_id: str) -> None:
        """
        Монитор для запуска воспроизведения после prebuffer.

        Ждёт пока буфер заполнится до prebuffer уровня, затем
        запускает play() с StreamingAudioSource.

        Args:
            stream_id: Идентификатор потока для логирования
        """
        source = self._streaming_source
        if not source:
            return

        # Wait for prebuffer (with timeout)
        prebuffer_timeout = 5.0
        start_time = asyncio.get_event_loop().time()

        while not source.is_ready:
            if source.is_finished:
                # Stream ended before prebuffer filled - play what we have
                logger.debug(f"Stream {stream_id} ended before prebuffer filled")
                break

            if asyncio.get_event_loop().time() - start_time > prebuffer_timeout:
                logger.warning(f"Prebuffer timeout for stream {stream_id}")
                break

            await asyncio.sleep(0.01)

        # Start playback
        if not self._voice_client or not self._voice_client.is_connected():
            logger.warning("Voice client disconnected before playback start")
            return

        logger.info(
            f"Starting streaming playback: {stream_id}, "
            f"buffer={source.buffer_level_ms:.0f}ms"
        )

        # Play the streaming source
        def after_playback(error):
            if error:
                logger.error(f"Playback error: {error}")
            logger.debug(
                f"Streaming playback ended: {stream_id}, "
                f"underruns={source.underrun_count}"
            )

        try:
            self._voice_client.play(source, after=after_playback)
        except Exception as e:
            logger.error(f"Failed to start playback: {e}")

    async def _handle_batch_audio(self, audio_data: Any) -> None:
        """
        Legacy batch mode воспроизведения.

        Используется для звуковых эффектов и non-streaming аудио.
        """
        logger.debug(
            f"Batch audio: type={type(audio_data)}, queue_size={len(self._audio_queue)}"
        )
        self._audio_queue.append(audio_data)

        # Signal that new audio arrived
        if hasattr(self, "_audio_event"):
            self._audio_event.set()

        if self._is_playing:
            return

        await self._process_audio_queue()

    async def _process_audio_queue(self) -> None:
        """
        Обработка очереди аудио (legacy batch mode).

        Используется для звуковых эффектов и non-streaming аудио.
        """
        self._is_playing = True

        # Event для сигнализации о новых данных
        if not hasattr(self, "_audio_event"):
            self._audio_event = asyncio.Event()

        # Счётчик пустых итераций для определения конца
        empty_wait_count = 0
        max_empty_waits = 10  # 10 * 50ms = 500ms

        while True:
            if self._audio_queue:
                empty_wait_count = 0
                audio_data = self._audio_queue.popleft()

                try:
                    if hasattr(audio_data, "data"):
                        audio_bytes = audio_data.data
                        sample_rate = getattr(audio_data, "sample_rate", 24000)
                    elif isinstance(audio_data, bytes):
                        audio_bytes = audio_data
                        sample_rate = 24000
                    else:
                        continue

                    discord_audio = self._convert_to_discord_format(audio_bytes, sample_rate)

                    if discord_audio and self._voice_client and self._voice_client.is_connected():
                        audio_source = discord.PCMAudio(io.BytesIO(discord_audio))
                        self._voice_client.play(audio_source)

                        # Ждём окончания воспроизведения
                        while self._voice_client.is_playing():
                            await asyncio.sleep(0.05)

                except Exception as e:
                    logger.error(f"Batch playback error: {e}")
            else:
                empty_wait_count += 1
                if empty_wait_count >= max_empty_waits:
                    break

                self._audio_event.clear()
                try:
                    await asyncio.wait_for(self._audio_event.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass

        self._is_playing = False
        logger.debug("Batch audio queue processing finished")

    def _convert_to_discord_format(self, audio_bytes: bytes, source_rate: int) -> Optional[bytes]:
        """Конвертация в формат Discord (48kHz stereo)."""
        try:
            samples = np.frombuffer(audio_bytes, dtype=np.int16)

            if source_rate != 48000:
                ratio = 48000 / source_rate
                new_length = int(len(samples) * ratio)
                indices = np.linspace(0, len(samples) - 1, new_length)
                samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.int16)

            stereo = np.column_stack((samples, samples)).flatten()
            return stereo.tobytes()

        except Exception as e:
            logger.error(f"Ошибка конвертации: {e}")
            return None
