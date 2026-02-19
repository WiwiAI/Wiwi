# Wiwi 4.0

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/WiwiAI/Wiwi)

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

### One-command quick start (минимальный сценарий)

Если у тебя уже поднят `llama-server` на `http://localhost:8080`, можно стартануть почти одной командой:

```bash
git clone https://github.com/WiwiAI/Wiwi.git && cd Wiwi && \
python -m venv .venv && source .venv/bin/activate && \
pip install -U pip && pip install -r requirements.txt && \
cp config/example.yaml config/local.yaml && \
python -m wiwi -c config/local.yaml
```

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

Также есть безопасный шаблон для старта: `config/example.yaml`.

```bash
cp config/example.yaml config/local.yaml
python -m wiwi -c config/local.yaml
```

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

## Discord

- В `config/default.yaml` поле `modules.discord.token` должно быть пустым в git.
- Перед локальным запуском подставляй токен только у себя.

Подробнее: `docs/DISCORD.md`

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

## Для контрибьюторов

- Правила участия: `CONTRIBUTING.md`
- Безопасность: `SECURITY.md`
- Шаблоны issue/PR: `.github/`

## Лицензия

MIT (см. `LICENSE`).

---

## Known issues

- Первый запуск TTS/STT на GPU может быть долгим (инициализация/компиляция).
- Для некоторых конфигураций CUDA нужен корректный `LD_LIBRARY_PATH` (cuDNN).
- Без запущенного `llama-server` backend `llama_server` не ответит.
- Репозиторий не содержит модели: их нужно скачать отдельно локально.

## Roadmap (кратко)

- улучшение setup/installer;
- стабилизация voice pipeline;
- упрощение конфигов для первого запуска;
- больше готовых пресетов под разное железо.
