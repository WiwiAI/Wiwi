# Wiwi 4.0

Модульный AI-ассистент с поддержкой:
- **LLM** (через `llama-server` или `llama-cpp-python`)
- **STT** (speech-to-text, `faster-whisper`)
- **TTS** (XTTS / Chatterbox)
- **Discord-бота** (текст + голос)
- **CLI-интерфейса**

> Проект на стадии активной разработки (alpha), но уже рабочий для локального использования.

---

## Что умеет

- модульная архитектура (ядро + подключаемые модули);
- локальная память диалога;
- голосовой ввод (VAD + STT);
- голосовой вывод (TTS, streaming);
- запуск в терминале и через Discord;
- конфигурация через YAML (`config/default.yaml`).

Документация в репозитории:
- `docs/start.md` — быстрый старт
- `docs/MODULES.md` — модули и их параметры
- `docs/ARCHITECTURE.md` — архитектура
- `docs/DISCORD.md` — Discord настройка

---

## Требования

- Linux (рекомендовано)
- Python **3.10+**
- (опционально) CUDA + cuDNN для ускорения STT/TTS
- (опционально) `llama.cpp` + `llama-server`

---

## Установка

### 1) Клонирование

```bash
git clone https://github.com/WiwiAI/Wiwi.git
cd Wiwi
```

### 2) Виртуальное окружение

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 3) Зависимости

Базовые:

```bash
pip install -r requirements.txt
```

Или через extras из `pyproject.toml`:

```bash
pip install -e .
# опционально:
# pip install -e .[llm]
# pip install -e .[stt]
# pip install -e .[tts]
# pip install -e .[discord]
# pip install -e .[all]
```

---

## Быстрый запуск (CLI)

```bash
python -m wiwi
```

или

```bash
./run.sh
```

`run.sh` удобен, если у тебя setup через conda/cudnn.

### Полезные команды внутри CLI

- `/help`
- `/voice on` / `/voice off`
- `/devices`
- `/clear`
- `/status`
- `/quit`

---

## Настройка LLM (рекомендуемый путь: llama-server)

В конфиге (`config/default.yaml`) по умолчанию используется backend `llama_server` и URL `http://localhost:8080`.

Пример запуска `llama-server` (подстрой модель/параметры под свою машину):

```bash
./llama.cpp/build/bin/llama-server \
  -m models/your-model.gguf \
  --port 8080
```

После этого запускай Wiwi.

---

## Конфиг

Основной файл: `config/default.yaml`

Критичные поля:
- `enabled_modules` — какие модули включены
- `modules.llm_brain.*` — LLM backend и генерация
- `modules.stt.*` — speech-to-text и VAD
- `modules.tts.*` — voice output
- `modules.discord.*` — токен бота и настройки Discord

Запуск с кастомным конфигом:

```bash
python -m wiwi -c path/to/config.yaml
```

---

## Discord (безопасно)

⚠️ **Никогда не коммить токены в репозиторий.**

- В `config/default.yaml` поле `modules.discord.token` должно быть пустым в git.
- Перед локальным запуском подставляй токен только у себя.
- Если токен утёк — сразу перевыпусти его в Discord Developer Portal.

Подробнее: `docs/DISCORD.md`

---

## Что не включено в этот репозиторий

Чтобы репозиторий был лёгким и безопасным, обычно **не коммитятся**:
- тяжёлые модели (`models/`)
- локальные сборки/подмодули (`llama.cpp/`, `chatterbox-streaming/`)
- приватные конфиги и ключи (`.env`, `*.key`, `*.pem`)

Это ожидаемое поведение: скачал код → докинул модели и локальные настройки у себя.

---

## Разработка

```bash
pip install -e .[dev]
```

Базовые проверки:

```bash
pytest
ruff check .
black .
```

---

## Лицензия

MIT (см. `pyproject.toml`).

---

## Roadmap (кратко)

- улучшение setup/installer;
- стабилизация voice pipeline;
- упрощение конфигов для первого запуска;
- больше готовых пресетов под разное железо.
