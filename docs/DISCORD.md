# Discord Module

Модуль интеграции Wiwi с Discord. Поддерживает текстовое и голосовое общение через Discord сервер.

---

## Возможности

- **Текстовые каналы:** ответы на упоминания (@mention) и сообщения с префиксом "wiwi"
- **Голосовые каналы:** подключение к voice chat, прослушивание и ответы
- **Wake Word Detection:** OpenWakeWord для постоянного лёгкого прослушивания (~1-3% CPU)
- **VAD:** Silero VAD для определения конца речи
- **Звуковые эффекты:** аудио-оповещения при детекции wake word

---

## Быстрый старт

### 1. Создание Discord бота

1. Перейдите на [Discord Developer Portal](https://discord.com/developers/applications)
2. Нажмите **New Application** → введите имя (например, "Wiwi")
3. Перейдите в **Bot** → **Add Bot**
4. Скопируйте **Token** (понадобится для конфигурации)
5. Включите **Privileged Gateway Intents**:
   - ✅ Message Content Intent
   - ✅ Server Members Intent
   - ✅ Voice States Intent

### 2. Приглашение бота на сервер

1. Перейдите в **OAuth2** → **URL Generator**
2. Выберите scopes: `bot`, `applications.commands`
3. Выберите permissions:
   - Send Messages
   - Read Message History
   - Connect (для голоса)
   - Speak (для голоса)
   - Use Voice Activity
4. Скопируйте URL и откройте в браузере
5. Выберите сервер и подтвердите

### 3. Установка зависимостей

```bash
pip install -e ".[discord]"
```

Это установит:
- `discord.py>=2.3.0` — библиотека Discord
- `PyNaCl>=1.5.0` — для голосовых каналов
- `openwakeword>=0.6.0` — wake word detection

### 4. Конфигурация

Отредактируйте `config/default.yaml`:

```yaml
enabled_modules:
  - "memory"
  - "llm_brain"
  - "cli_interface"
  - "stt"
  - "tts"
  - "discord"  # Раскомментируйте эту строку

modules:
  discord:
    token: "YOUR_BOT_TOKEN_HERE"  # Вставьте токен бота
```

### 5. Запуск

```bash
# Запустите llama-server (в отдельном терминале)
./llama.cpp/build/bin/llama-server \
    -m models/magnum-v4-12b-Q4_K_M.gguf \
    --port 8080 --ctx-size 8192 --gpu-layers 999 --tensor-split 1,0

# Запустите Wiwi
./run.sh
```

---

## Использование

### Текстовые команды

В текстовом канале Discord:

| Способ | Пример |
|--------|--------|
| @mention | `@Wiwi привет, как дела?` |
| Префикс "wiwi" | `wiwi, расскажи анекдот` |
| Префикс "wiwi:" | `wiwi: что такое Python?` |

### Slash-команды

| Команда | Описание |
|---------|----------|
| `/join` | Присоединиться к голосовому каналу (где находится пользователь) |
| `/leave` | Покинуть голосовой канал |
| `/status` | Показать статус бота |

### Голосовое управление

1. Зайдите в голосовой канал
2. Используйте `/join` — бот подключится к вашему каналу
3. Скажите wake word (по умолчанию: "hey jarvis", "alexa" и др.)
4. После звукового сигнала произнесите команду
5. Бот распознает речь и ответит

**Примечание:** Для кастомного wake word "виви" требуется обученная модель OpenWakeWord.

---

## Конфигурация

### Полная конфигурация модуля

```yaml
modules:
  discord:
    # === Основные настройки ===
    token: ""                     # ОБЯЗАТЕЛЬНО: токен бота
    command_prefix: "!"           # Префикс для команд (не используется)

    # === Текстовые каналы ===
    respond_to_mentions: true     # Отвечать на @mention
    respond_to_prefix: true       # Отвечать на "wiwi" в начале сообщения

    # === Ограничения доступа ===
    allowed_channels: []          # ID каналов (пусто = все каналы)
    allowed_users: []             # ID пользователей (пусто = все)

    # === Голосовые каналы ===
    voice_enabled: true           # Включить поддержку голоса
    auto_join_voice: false        # Автоподключение при старте
    voice_channel_id: null        # ID канала для автоподключения

    # === Wake Word Detection ===
    wake_word:
      model_paths: []             # Пути к кастомным моделям (.onnx)
      threshold: 0.5              # Порог срабатывания (0.0-1.0)
      vad_threshold: 0.5          # Порог VAD (0.0-1.0)
      min_silence_ms: 700         # Тишина для завершения записи (мс)
      max_recording_ms: 10000     # Максимальная длина записи (мс)

    # === Звуковые эффекты ===
    sounds:
      enabled: true               # Включить звуковые оповещения
```

### Ограничение доступа

Для ограничения бота определёнными каналами или пользователями:

```yaml
modules:
  discord:
    # Только в этих каналах (получить ID: ПКМ по каналу → Copy ID)
    allowed_channels:
      - 1234567890123456789
      - 9876543210987654321

    # Только эти пользователи (получить ID: ПКМ по пользователю → Copy ID)
    allowed_users:
      - 1111111111111111111
```

### Автоподключение к голосовому каналу

```yaml
modules:
  discord:
    auto_join_voice: true
    voice_channel_id: 1234567890123456789  # ID голосового канала
```

---

## Архитектура Wake Word

```
┌─────────────────────────────────────────────────────────────────┐
│                     Discord Voice Stream                         │
│                       (48kHz stereo)                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Конвертация → 16kHz mono                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │      WakeWordPipeline      │
              └─────────────┬─────────────┘
                            │
         ┌──────────────────┴──────────────────┐
         │                                     │
         ▼                                     ▼
┌─────────────────────┐              ┌─────────────────────┐
│   LISTENING State   │              │   RECORDING State   │
│                     │              │                     │
│  OpenWakeWord       │   wake word  │  Silero VAD         │
│  (~1-3% CPU)        │ ──────────►  │  записывает до      │
│  постоянно слушает  │   detected   │  паузы 700ms        │
└─────────────────────┘              └──────────┬──────────┘
                                                │
                                     ┌──────────┴──────────┐
                                     │                     │
                                     ▼                     ▼
                          ┌─────────────────┐   ┌─────────────────┐
                          │   STT Whisper   │   │  Звуковой сигнал │
                          │   транскрипция  │   │   "дин-дон"      │
                          └────────┬────────┘   └─────────────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │   LLM Brain     │
                          │   генерация     │
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │  Discord Text   │
                          │   + TTS (опц.)  │
                          └─────────────────┘
```

### Нагрузка на систему

| Компонент | CPU | GPU | Когда работает |
|-----------|-----|-----|----------------|
| OpenWakeWord | ~1-3% | — | Постоянно (LISTENING) |
| Silero VAD | ~1% | — | После wake word (RECORDING) |
| Whisper STT | ~5% | ~1GB | Только транскрипция команды |
| LLM | ~10% | ~7GB | Только генерация ответа |

---

## Звуковые эффекты

Модуль воспроизводит звуковые сигналы для обратной связи:

| Событие | Звук | Описание |
|---------|------|----------|
| Wake word detected | "дин-дон" | Двойной восходящий тон |
| Listening | Мягкий аккорд | Начало записи команды |
| Success | Восходящая гамма | Команда успешно обработана |
| Error | Нисходящий тон | Ошибка |
| Cancel | Короткий низкий тон | Таймаут/отмена |

Отключение звуков:
```yaml
modules:
  discord:
    sounds:
      enabled: false
```

---

## Кастомный Wake Word

По умолчанию используются встроенные модели OpenWakeWord:
- `hey_jarvis`
- `alexa`
- `hey_mycroft`
- и другие

### Обучение модели для "виви"

1. **Установите инструменты:**
```bash
pip install openwakeword
```

2. **Соберите аудио-примеры:**
   - ~50-100 записей произнесения "виви" разными голосами
   - Формат: WAV, 16kHz, mono
   - Длительность: 0.5-2 секунды каждая

3. **Обучите модель:**
```bash
# Следуйте инструкциям:
# https://github.com/dscripka/openWakeWord#training-new-models
```

4. **Подключите модель:**
```yaml
modules:
  discord:
    wake_word:
      model_paths:
        - "models/wake_word/wiwi.onnx"
      threshold: 0.5
```

---

## Структура файлов модуля

```
wiwi/modules/discord/
├── __init__.py          # Экспорт модуля
├── module.py            # Основной модуль DiscordModule
├── wake_word.py         # WakeWordPipeline (OWW + VAD)
└── sounds.py            # Генератор звуковых эффектов
```

### Компоненты

| Файл | Класс | Описание |
|------|-------|----------|
| `module.py` | `DiscordModule` | Основной модуль, Discord бот |
| `module.py` | `WiwiAudioSink` | Приём аудио из Discord |
| `wake_word.py` | `OpenWakeWordDetector` | Детектор wake word |
| `wake_word.py` | `SileroVAD` | Voice Activity Detection |
| `wake_word.py` | `WakeWordPipeline` | Объединённый пайплайн |
| `sounds.py` | `SoundEffects` | Генерация звуковых эффектов |

---

## API модуля

### ModuleInfo

```python
ModuleInfo(
    name="discord",
    version="2.0.0",
    description="Discord интеграция с OpenWakeWord + VAD",
    category="interface",
    input_ports={TEXT_OUT, AUDIO_OUT, COMMAND_IN},
    output_ports={TEXT_IN, AUDIO_IN, COMMAND_OUT},
    dependencies=set(),
    optional_dependencies={"memory", "stt", "tts"},
)
```

### Порты

| Порт | Направление | Описание |
|------|-------------|----------|
| `TEXT_OUT` | Input | Получение ответов от LLM для отправки в Discord |
| `AUDIO_OUT` | Input | Получение аудио от TTS для воспроизведения |
| `COMMAND_IN` | Input | Получение команд (voice_join, voice_leave) |
| `TEXT_IN` | Output | Отправка текста пользователя в систему |
| `AUDIO_IN` | Output | Отправка аудио на STT (опционально) |
| `COMMAND_OUT` | Output | Отправка команд в систему |

### Метаданные сообщений

При отправке `TEXT_IN` модуль добавляет метаданные:

```python
metadata = {
    "source_type": "discord" | "discord_voice",
    "channel_id": int,        # ID канала Discord
    "user_id": int,           # ID пользователя Discord
    "user_name": str,         # Имя пользователя
    "message_id": int,        # ID сообщения (для текстовых)
    "guild_id": int,          # ID сервера (опционально)
}
```

---

## Troubleshooting

### Бот не отвечает на сообщения

1. Проверьте, включен ли `Message Content Intent` в Discord Developer Portal
2. Убедитесь, что бот имеет права читать сообщения в канале
3. Проверьте `allowed_channels` / `allowed_users` в конфигурации

### Бот не подключается к голосовому каналу

1. Убедитесь, что установлен `PyNaCl`: `pip install PyNaCl`
2. Проверьте права бота: Connect, Speak
3. Проверьте логи на наличие ошибок

### Wake word не срабатывает

1. Убедитесь, что OpenWakeWord установлен: `pip install openwakeword`
2. Проверьте порог срабатывания `threshold` (понизьте до 0.3-0.4)
3. Используйте встроенные wake words ("hey jarvis", "alexa")
4. Проверьте качество микрофона

### Ошибка "Token is invalid"

1. Проверьте токен в конфигурации
2. Убедитесь, что токен скопирован полностью
3. Перегенерируйте токен в Discord Developer Portal

### Нет звуковых сигналов

1. Проверьте `sounds.enabled: true` в конфигурации
2. Убедитесь, что бот подключен к голосовому каналу
3. Проверьте громкость в Discord

---

## Примеры использования

### Минимальная конфигурация

```yaml
enabled_modules:
  - "memory"
  - "llm_brain"
  - "discord"

modules:
  discord:
    token: "YOUR_TOKEN"
```

### Полная конфигурация с голосом

```yaml
enabled_modules:
  - "memory"
  - "llm_brain"
  - "stt"
  - "tts"
  - "discord"

modules:
  discord:
    token: "YOUR_TOKEN"
    voice_enabled: true
    wake_word:
      threshold: 0.4
      min_silence_ms: 600
    sounds:
      enabled: true
```

### Ограниченный доступ

```yaml
modules:
  discord:
    token: "YOUR_TOKEN"
    allowed_channels:
      - 123456789012345678  # #wiwi-chat
    allowed_users:
      - 987654321098765432  # Только админ
```
