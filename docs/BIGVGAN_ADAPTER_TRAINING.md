# BigVGAN Adapter Training Guide

Руководство по обучению адаптера для использования BigVGAN v2 с XTTS GPT.

## Проблема

XTTS GPT выдаёт латентные коды `[B, T, 1024]`, а BigVGAN ожидает mel-спектрограммы `[B, 100, T]`. Нужен адаптер который конвертирует между этими форматами.

## Архитектура адаптера

```
XTTS GPT Latents [B, T, 1024]
         ↓
    Transpose [B, 1024, T]
         ↓
    Conv1d (1024 → 512)
         ↓
    LayerNorm + GELU
         ↓
    Conv1d (512 → 256)
         ↓
    LayerNorm + GELU
         ↓
    Conv1d (256 → 100)
         ↓
BigVGAN Mel [B, 100, T]
```

## Реализация адаптера

```python
import torch
import torch.nn as nn

class GPTLatentToMelAdapter(nn.Module):
    """
    Адаптер для конвертации XTTS GPT латентов в mel-спектрограммы для BigVGAN.

    Input: [B, T, 1024] - GPT latent codes
    Output: [B, 100, T] - Mel spectrogram for BigVGAN
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        mel_dim: int = 100,
        hidden_dims: list = [512, 256],
        kernel_size: int = 3,
    ):
        super().__init__()

        layers = []
        in_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim

        # Final projection to mel dimension
        layers.append(nn.Conv1d(in_dim, mel_dim, kernel_size, padding=kernel_size // 2))

        self.net = nn.Sequential(*layers)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [B, T, 1024] - GPT latent codes

        Returns:
            mel: [B, 100, T] - Mel spectrogram
        """
        # Transpose: [B, T, C] -> [B, C, T]
        x = latents.transpose(1, 2)

        # Apply adapter network
        mel = self.net(x)

        return mel
```

## Данные для обучения

### Вариант 1: Парные данные из XTTS

Собрать пары (GPT latents, reference mel) из XTTS inference:

```python
import torch
import torchaudio
from TTS.tts.models.xtts import Xtts

def collect_training_pairs(model, audio_paths, output_dir):
    """Собрать пары латентов и mel-спектрограмм."""

    pairs = []

    for audio_path in audio_paths:
        # 1. Загрузить аудио и получить mel
        wav, sr = torchaudio.load(audio_path)
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)

        # Вычислить mel для BigVGAN (100 bands)
        mel = compute_mel_spectrogram(wav, n_mels=100, sr=24000)

        # 2. Получить GPT латенты через XTTS
        # (нужен текст транскрипции)
        text = get_transcription(audio_path)
        gpt_cond, speaker_emb = model.get_conditioning_latents(audio_path=[audio_path])

        result = model.inference(
            text, "en", gpt_cond, speaker_emb,
            return_latents=True  # Нужно модифицировать XTTS
        )
        gpt_latents = result['gpt_latents']

        # 3. Выровнять по длине (интерполяция)
        # GPT latents и mel могут иметь разную временную шкалу

        pairs.append({
            'gpt_latents': gpt_latents,
            'mel': mel,
            'audio_path': audio_path,
        })

    return pairs
```

### Вариант 2: Self-supervised из аудио

Использовать только аудио без транскрипций:

```python
def self_supervised_training_step(adapter, bigvgan, audio_batch):
    """
    Self-supervised: audio -> mel -> adapter -> mel' -> bigvgan -> audio'
    Loss: |audio - audio'| + |mel - mel'|
    """

    # Ground truth mel
    mel_gt = compute_mel(audio_batch)

    # Simulate GPT latents (можно использовать encoder или random projection)
    # В реальности нужен encoder который учится вместе с adapter
    latents = mel_to_latent_encoder(mel_gt)

    # Adapter: latents -> mel
    mel_pred = adapter(latents)

    # BigVGAN: mel -> audio
    audio_pred = bigvgan(mel_pred)

    # Losses
    mel_loss = F.l1_loss(mel_pred, mel_gt)
    audio_loss = F.l1_loss(audio_pred, audio_batch)

    return mel_loss + audio_loss
```

## Обучение

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import bigvgan

def train_adapter(
    adapter: GPTLatentToMelAdapter,
    train_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-4,
):
    """Обучение адаптера."""

    # Загрузить BigVGAN для валидации
    vocoder = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x')
    vocoder.eval()
    vocoder.cuda()

    adapter.cuda()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        adapter.train()
        total_loss = 0

        for batch in train_loader:
            gpt_latents = batch['gpt_latents'].cuda()  # [B, T, 1024]
            mel_target = batch['mel'].cuda()            # [B, 100, T]

            optimizer.zero_grad()

            # Forward
            mel_pred = adapter(gpt_latents)

            # L1 Loss on mel
            loss = F.l1_loss(mel_pred, mel_target)

            # Optional: Multi-resolution STFT loss
            # loss += multi_resolution_stft_loss(mel_pred, mel_target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Validation
        if epoch % 10 == 0:
            adapter.eval()
            with torch.no_grad():
                # Generate audio sample
                sample_latents = next(iter(train_loader))['gpt_latents'][:1].cuda()
                mel_pred = adapter(sample_latents)
                audio = vocoder(mel_pred)

                # Save audio
                torchaudio.save(f'val_epoch_{epoch}.wav', audio.cpu(), 24000)

            print(f'Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}')

    return adapter
```

## Экспорт в ONNX

После обучения адаптер можно экспортировать:

```python
def export_adapter_onnx(adapter, output_path):
    """Экспорт обученного адаптера в ONNX."""

    adapter.eval()
    adapter.cpu()

    dummy_latents = torch.randn(1, 100, 1024)  # [B, T, latent_dim]

    torch.onnx.export(
        adapter,
        dummy_latents,
        output_path,
        opset_version=17,
        input_names=['gpt_latents'],
        output_names=['mel'],
        dynamic_axes={
            'gpt_latents': {0: 'batch', 1: 'time'},
            'mel': {0: 'batch', 2: 'time'},
        },
    )

    print(f'Exported adapter to {output_path}')
```

## Интеграция с XTTS TRT Backend

После обучения адаптера, создать новый backend:

```python
class XttsBigVGANBackend(BaseTTSBackend):
    """XTTS с BigVGAN вокодером через обученный адаптер."""

    def __init__(self, config):
        super().__init__(config)

        # TensorRT engines
        self.gpt_engine = load_trt_engine(config['trt_gpt_path'])
        self.adapter_engine = load_trt_engine(config['trt_adapter_path'])
        self.bigvgan_engine = load_trt_engine(config['trt_bigvgan_path'])

    def synthesize(self, text, **kwargs):
        # 1. GPT: text -> latents
        latents = self.run_gpt(text)

        # 2. Adapter: latents -> mel
        mel = self.run_adapter(latents)

        # 3. BigVGAN: mel -> audio
        audio = self.run_bigvgan(mel)

        return audio
```

## Датасеты для обучения

Рекомендуемые датасеты:

1. **LibriTTS-R** - чистая речь, много часов
   - https://www.openslr.org/141/

2. **VCTK** - многоговорящий датасет
   - https://datashare.ed.ac.uk/handle/10283/3443

3. **LJSpeech** - один диктор, высокое качество
   - https://keithito.com/LJ-Speech-Dataset/

## Требования

- GPU с 8+ GB VRAM для обучения
- ~10-50 часов аудио для хороших результатов
- 50-100 эпох обучения
- Время обучения: ~2-4 часа на A100

## Альтернативный подход: Fine-tune BigVGAN

Вместо адаптера можно fine-tune сам BigVGAN:

1. Заменить первый conv слой BigVGAN: `Conv1d(100, 512)` → `Conv1d(1024, 512)`
2. Fine-tune на парах (GPT latents, audio)
3. Это может дать лучшее качество, но требует больше данных

---

## Ссылки

- [BigVGAN GitHub](https://github.com/NVIDIA/BigVGAN)
- [BigVGAN v2 HuggingFace](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x)
- [XTTS Documentation](https://docs.coqui.ai/en/latest/models/xtts.html)
